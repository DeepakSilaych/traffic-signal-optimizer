import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict, deque
import sumolib
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import DefaultObservationFunction
from torch_geometric.nn import GATv2Conv

D_MODEL = 64
N_HEADS = 4
T_HIST = 6
GRID_H = GRID_W = 18
PATCH_SIZE = 3
N_PATCHES = (GRID_H // PATCH_SIZE) ** 2


def _grid_edge_index(h=6, w=6):
    src, dst = [], []
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if j + 1 < w:
                src += [idx, idx + 1]
                dst += [idx + 1, idx]
            if i + 1 < h:
                src += [idx, idx + w]
                dst += [idx + w, idx]
    return torch.tensor([src, dst], dtype=torch.long)


def _region_indices():
    regions = []
    for ri in range(3):
        for rj in range(3):
            patches = []
            for di in range(2):
                for dj in range(2):
                    patches.append((ri * 2 + di) * 6 + (rj * 2 + dj))
            regions.append(patches)
    return regions


def discover_topology(net_file, sumo_env):
    ts_ids = sorted(sumo_env.traffic_signals.keys())
    n_nodes = len(ts_ids)
    ts_id_to_idx = {tid: i for i, tid in enumerate(ts_ids)}

    controllable_idx = []
    max_phases = 1
    for i, tid in enumerate(ts_ids):
        ts = sumo_env.traffic_signals[tid]
        n_gp = ts.num_green_phases
        if n_gp >= 2:
            controllable_idx.append(i)
        max_phases = max(max_phases, n_gp)

    approach_map = {}
    k_counts = []
    for tid in ts_ids:
        ts = sumo_env.traffic_signals[tid]
        edge_lanes = defaultdict(list)
        for lane in ts.lanes:
            edge = lane.rsplit('_', 1)[0]
            if lane not in edge_lanes[edge]:
                edge_lanes[edge].append(lane)
        approach_map[tid] = dict(edge_lanes)
        k_counts.append(len(edge_lanes))
    max_k = max(k_counts)

    net = sumolib.net.readNet(net_file)
    src, dst = [], []
    for i, tid_i in enumerate(ts_ids):
        node_i = net.getNode(tid_i)
        if node_i is None:
            continue
        neighbors = set()
        for edge in node_i.getIncoming():
            neighbors.add(edge.getFromNode().getID())
        for edge in node_i.getOutgoing():
            neighbors.add(edge.getToNode().getID())
        for nb_id in neighbors:
            if nb_id in ts_id_to_idx:
                j = ts_id_to_idx[nb_id]
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return {
        'ts_ids': ts_ids,
        'n_nodes': n_nodes,
        'controllable_idx': controllable_idx,
        'n_controllable': len(controllable_idx),
        'max_k': max_k,
        'max_phases': max_phases,
        'approach_map': approach_map,
        'k_counts': k_counts,
        'edge_index': edge_index,
    }


# ======================== Level 1: SubNodeEncoder ========================

class SubNodeEncoder(nn.Module):
    def __init__(self, in_channels=2, D=D_MODEL, T=T_HIST):
        super().__init__()
        self.D = D
        self.T = T

        self.patch_cnn = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=PATCH_SIZE, stride=PATCH_SIZE),
            nn.ReLU(),
        )

        self.temporal_score = nn.Linear(D, 1)
        self.spatial_score = nn.Linear(D, 1)

        grid_ei = _grid_edge_index(6, 6)
        self.register_buffer('grid_edge_index', grid_ei)
        adj = torch.zeros(N_PATCHES, N_PATCHES)
        adj[grid_ei[0], grid_ei[1]] = 1.0
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        self.register_buffer('norm_adj', adj / deg)

        self.gcn_w = nn.Linear(D, D)

        self.regions = _region_indices()
        self.inter_proj = nn.Linear(D, D)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        patches = self.patch_cnn(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.view(B, T, N_PATCHES, self.D)

        t_w = F.softmax(self.temporal_score(patches).squeeze(-1), dim=1)
        patches_t = (patches * t_w.unsqueeze(-1)).sum(dim=1)

        s_w = F.softmax(self.spatial_score(patches_t).squeeze(-1), dim=1)
        patches_s = patches_t * s_w.unsqueeze(-1)

        gcn_out = self.gcn_w(torch.matmul(self.norm_adj, patches_s))
        gcn_out = F.relu(gcn_out)

        spatial_tokens = []
        for region in self.regions:
            tok = gcn_out[:, region, :].mean(dim=1)
            spatial_tokens.append(tok)
        spatial_tokens = torch.stack(spatial_tokens, dim=1)

        inter_token = self.inter_proj(gcn_out.mean(dim=1, keepdim=True))

        return torch.cat([spatial_tokens, inter_token], dim=1)


class SiblingCrossAttention(nn.Module):
    def __init__(self, D=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(D)

    def forward(self, tokens, k_mask=None):
        B, K, N_TOK, D = tokens.shape
        inter = tokens[:, :, 9, :]
        attn_mask = None
        if k_mask is not None:
            attn_mask = ~k_mask.bool()
        refined, _ = self.attn(inter, inter, inter, key_padding_mask=attn_mask)
        refined = self.norm(inter + refined)
        out = tokens.clone()
        out[:, :, 9, :] = refined
        return out


# ======================== Level 2: NodeAggregator ========================

class MAB(nn.Module):
    def __init__(self, D=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))
        self.norm2 = nn.LayerNorm(D)

    def forward(self, q, kv):
        out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + out)
        return self.norm2(q + self.ffn(q))


class SAB(nn.Module):
    def __init__(self, D=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.mab = MAB(D, n_heads)

    def forward(self, x):
        return self.mab(x, x)


class PMA(nn.Module):
    def __init__(self, D=D_MODEL, n_heads=N_HEADS, n_seeds=1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, D) * 0.02)
        self.mab = MAB(D, n_heads)

    def forward(self, x):
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        return self.mab(seeds, x)


class NodeAggregator(nn.Module):
    def __init__(self, D=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.type_embed = nn.Embedding(2, D)
        self.encoder = SAB(D, n_heads)
        self.pma_internal = PMA(D, n_heads, n_seeds=3)
        self.pma_social = PMA(D, n_heads, n_seeds=1)

    def forward(self, tokens, k_mask=None):
        B, K, N_TOK, D = tokens.shape

        type_ids = torch.zeros(N_TOK, dtype=torch.long, device=tokens.device)
        type_ids[9] = 1
        tokens = tokens + self.type_embed(type_ids)

        flat = tokens.view(B, K * N_TOK, D)
        flat = self.encoder(flat)

        reshaped = flat.view(B, K, N_TOK, D)
        spatial = reshaped[:, :, :9, :].reshape(B, K * 9, D)
        interaction = reshaped[:, :, 9:, :].reshape(B, K, D)

        internal = self.pma_internal(spatial)
        social = self.pma_social(interaction)

        return torch.cat([internal, social], dim=1)


# ======================== Level 3: DynamicGraphNet ========================

class DynamicGraphNet(nn.Module):
    def __init__(self, D=D_MODEL, n_heads=N_HEADS, n_nodes=9, edge_index=None):
        super().__init__()
        assert D % n_heads == 0
        self.n_nodes = n_nodes
        self.gatv2 = GATv2Conv(D, D // n_heads, heads=n_heads, concat=True)
        self.gru = nn.GRUCell(D, D)
        self.fusion = nn.Sequential(
            nn.Linear(4 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, 4 * D)
        )
        self.norm = nn.LayerNorm(4 * D)
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            self.register_buffer('edge_index', torch.zeros(2, 0, dtype=torch.long))

    def forward(self, node_tokens, hidden=None):
        B, N, _, D = node_tokens.shape
        social = node_tokens[:, :, 3, :].contiguous()
        internal = node_tokens[:, :, :3, :].contiguous()

        all_social = social.reshape(B * N, D)
        ei = self.edge_index
        batched_ei = torch.cat([ei + i * N for i in range(B)], dim=1)
        msg = self.gatv2(all_social, batched_ei)
        msg = msg.view(B, N, D)

        if hidden is None:
            hidden = social
        updated = self.gru(
            msg.reshape(B * N, D), hidden.reshape(B * N, D)
        ).view(B, N, D)

        cat = torch.cat([internal.reshape(B, N, 3 * D), updated], dim=-1)
        fused = self.norm(cat + self.fusion(cat))
        return fused.view(B, N, 4, D), updated


# ======================== Level 4: Actor & Critic ========================

class Actor(nn.Module):
    def __init__(self, D=D_MODEL, max_phases=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * D + max_phases + 1, 2 * D),
            nn.GELU(),
            nn.Linear(2 * D, max_phases),
        )

    def forward(self, tokens, phase, min_green_flag):
        B, N, _, D = tokens.shape
        flat = tokens.reshape(B, N, 4 * D)
        extra = torch.cat([phase, min_green_flag.unsqueeze(-1)], dim=-1)
        x = torch.cat([flat, extra], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, D=D_MODEL):
        super().__init__()
        self.encoder = SAB(D, N_HEADS)
        self.head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1)
        )

    def forward(self, tokens):
        B, N, _, D = tokens.shape
        flat = tokens.reshape(B, N * 4, D)
        enc = self.encoder(flat)
        pooled = enc.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# ======================== Full Model ========================

class HierarchicalSignalNet(nn.Module):
    def __init__(self, n_nodes, max_k, max_phases, controllable_idx, edge_index,
                 D=D_MODEL, n_heads=N_HEADS, T=T_HIST, in_channels=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_k = max_k
        self.max_phases = max_phases
        self.controllable_idx = controllable_idx

        self.sub_node_encoder = SubNodeEncoder(in_channels, D, T)
        self.sibling_cross_attn = SiblingCrossAttention(D, n_heads)
        self.node_aggregator = NodeAggregator(D, n_heads)
        self.dynamic_graph_net = DynamicGraphNet(D, n_heads, n_nodes, edge_index)
        self.actor = Actor(D, max_phases)
        self.critic = Critic(D)

    def forward(self, grids, k_mask, phase, min_green_flag, gru_hidden=None):
        B, N, K, T, H, W, C = grids.shape

        flat_grids = grids.reshape(B * N * K, T, H, W, C)
        flat_tokens = self.sub_node_encoder(flat_grids)
        sub_tokens = flat_tokens.view(B * N, K, 10, -1)

        flat_k_mask = k_mask.view(B * N, K)
        sub_tokens = self.sibling_cross_attn(sub_tokens, flat_k_mask)

        node_tokens = self.node_aggregator(sub_tokens, flat_k_mask)
        node_tokens = node_tokens.view(B, N, 4, -1)

        node_tokens, new_hidden = self.dynamic_graph_net(node_tokens, gru_hidden)

        logits = self.actor(node_tokens, phase, min_green_flag)
        value = self.critic(node_tokens)

        return logits, value, new_hidden

    def get_action_and_value(self, grids, k_mask, phase, min_green_flag,
                             gru_hidden=None, action=None):
        logits, value, new_hidden = self.forward(
            grids, k_mask, phase, min_green_flag, gru_hidden
        )

        ctrl_logits = logits[:, self.controllable_idx, :]
        dist = torch.distributions.Categorical(logits=ctrl_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value, new_hidden


# ======================== Environment ========================

def _approach_to_grid(lane_densities, lane_queues):
    grid = np.zeros((GRID_H, GRID_W, 2), dtype=np.float32)
    n = len(lane_densities)
    rows_per_lane = GRID_H // max(n, 1)
    for i in range(n):
        r0 = i * rows_per_lane
        r1 = min(r0 + rows_per_lane, GRID_H)
        grid[r0:r1, :, 0] = lane_densities[i]
        grid[r0:r1, :, 1] = lane_queues[i]
    return grid


class NetworkEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, net_file, route_file, num_seconds=3600,
                 delta_time=10, min_green=45, max_green=120,
                 sumo_seed=42, use_gui=False):
        super().__init__()
        self.net_file = net_file

        self._sumo_env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            min_green=min_green,
            max_green=max_green,
            single_agent=False,
            observation_class=DefaultObservationFunction,
            reward_fn='diff-waiting-time',
            sumo_seed=sumo_seed,
        )

        topo = discover_topology(net_file, self._sumo_env)
        self.ts_ids = topo['ts_ids']
        self.n_nodes = topo['n_nodes']
        self.controllable_idx = topo['controllable_idx']
        self.n_controllable = topo['n_controllable']
        self.max_k = topo['max_k']
        self.max_phases = topo['max_phases']
        self.inter_edge_index = topo['edge_index']
        self._approach_map = topo['approach_map']
        self._k_counts = topo['k_counts']

        self._buffers = {
            ts_id: {
                edge: deque(maxlen=T_HIST)
                for edge in self._approach_map[ts_id]
            }
            for ts_id in self.ts_ids
        }

        self.action_space = spaces.MultiDiscrete([self.max_phases] * self.n_controllable)
        self.observation_space = spaces.Dict({
            'grids': spaces.Box(-1, 1, shape=(self.n_nodes, self.max_k, T_HIST, GRID_H, GRID_W, 2)),
            'k_mask': spaces.MultiBinary([self.n_nodes, self.max_k]),
            'phase': spaces.Box(0, 1, shape=(self.n_nodes, self.max_phases)),
            'min_green_flag': spaces.Box(0, 1, shape=(self.n_nodes,)),
        })

    def _collect_obs(self):
        sumo = self._sumo_env.sumo

        grids = np.zeros((self.n_nodes, self.max_k, T_HIST, GRID_H, GRID_W, 2), dtype=np.float32)
        k_mask = np.zeros((self.n_nodes, self.max_k), dtype=np.float32)
        phase = np.zeros((self.n_nodes, self.max_phases), dtype=np.float32)
        min_green_flag = np.zeros(self.n_nodes, dtype=np.float32)

        for i, ts_id in enumerate(self.ts_ids):
            ts = self._sumo_env.traffic_signals[ts_id]

            if ts.green_phase < self.max_phases:
                phase[i, ts.green_phase] = 1.0
            can_switch = ts.time_since_last_phase_change >= (ts.min_green + ts.yellow_time)
            min_green_flag[i] = 1.0 if can_switch else 0.0

            for k, (edge, lanes) in enumerate(self._approach_map[ts_id].items()):
                if k >= self.max_k:
                    break
                k_mask[i, k] = 1.0

                densities, queues = [], []
                for lane in lanes:
                    try:
                        occ = sumo.lane.getLastStepOccupancy(lane)
                        halt = sumo.lane.getLastStepHaltingNumber(lane)
                        length = sumo.lane.getLength(lane)
                        cap = length / 7.5
                        densities.append(min(occ / 100.0, 1.0))
                        queues.append(min(halt / max(cap, 1.0), 1.0))
                    except Exception:
                        densities.append(0.0)
                        queues.append(0.0)

                frame = _approach_to_grid(densities, queues)
                self._buffers[ts_id][edge].append(frame)

                buf = list(self._buffers[ts_id][edge])
                while len(buf) < T_HIST:
                    buf.insert(0, np.zeros((GRID_H, GRID_W, 2), dtype=np.float32))
                grids[i, k] = np.stack(buf[-T_HIST:], axis=0)

        return {
            'grids': grids,
            'k_mask': k_mask,
            'phase': phase,
            'min_green_flag': min_green_flag,
        }

    def reset(self, seed=None, options=None):
        self._sumo_env.reset()
        for ts_id in self.ts_ids:
            for edge in self._buffers[ts_id]:
                self._buffers[ts_id][edge].clear()
        return self._collect_obs(), {}

    def step(self, action):
        actions = {}
        ctrl_idx = 0
        for i, ts_id in enumerate(self.ts_ids):
            if i in self.controllable_idx:
                a = int(action[ctrl_idx])
                ts = self._sumo_env.traffic_signals[ts_id]
                actions[ts_id] = min(a, ts.num_green_phases - 1)
                ctrl_idx += 1
            else:
                actions[ts_id] = 0

        _, reward_dict, done_dict, info = self._sumo_env.step(actions)

        total_reward = sum(reward_dict.values())
        truncated = done_dict.get('__all__', False)

        return self._collect_obs(), total_reward, False, truncated, info

    def close(self):
        self._sumo_env.close()
