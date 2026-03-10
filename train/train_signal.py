import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sumo
os.environ['SUMO_HOME'] = sumo.SUMO_HOME

import numpy as np
import torch
import torch.nn as nn
from architecture.signal_optimizer import HierarchicalSignalNet, NetworkEnv, D_MODEL

ROOT = Path(__file__).resolve().parent.parent
NET_FILE = str(ROOT / 'sumo_config' / 'city_network.net.xml')
ROUTE_FILE = str(ROOT / 'sumo_config' / 'city_routes.rou.xml')
CKPT_DIR = ROOT / 'checkpoints' / 'signal'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SECONDS = 3600
DELTA_TIME = 10
MIN_GREEN = 45
MAX_GREEN = 120

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_STEPS = 128
N_EPOCHS = 4
MINIBATCH_SIZE = 32
TOTAL_UPDATES = 200

DEVICE = 'cpu'


def to_tensor(obs, device=DEVICE):
    return {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}


def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + np.array(values)
    return advantages, returns


def collect_rollout(env, model, n_steps, device=DEVICE):
    storage = {
        'grids': [], 'k_mask': [], 'phase': [], 'min_green_flag': [],
        'actions': [], 'log_probs': [], 'rewards': [], 'dones': [],
        'values': [], 'hiddens': [],
    }

    obs, _ = env.reset()
    gru_hidden = None

    for _ in range(n_steps):
        t = to_tensor(obs, device)
        with torch.no_grad():
            action, lp, _, val, new_hidden = model.get_action_and_value(
                t['grids'], t['k_mask'], t['phase'], t['min_green_flag'],
                gru_hidden=gru_hidden,
            )

        for k in ['grids', 'k_mask', 'phase', 'min_green_flag']:
            storage[k].append(obs[k])
        storage['actions'].append(action[0].cpu().numpy())
        storage['log_probs'].append(lp[0].cpu().numpy())
        storage['values'].append(val.item())
        storage['hiddens'].append(
            gru_hidden[0].cpu().numpy() if gru_hidden is not None
            else np.zeros((env.n_nodes, D_MODEL), dtype=np.float32)
        )

        obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
        done = terminated or truncated
        storage['rewards'].append(reward)
        storage['dones'].append(float(done))

        gru_hidden = new_hidden.detach()
        if done:
            obs, _ = env.reset()
            gru_hidden = None

    with torch.no_grad():
        t = to_tensor(obs, device)
        _, _, _, next_val, _ = model.get_action_and_value(
            t['grids'], t['k_mask'], t['phase'], t['min_green_flag'],
            gru_hidden=gru_hidden,
        )

    advantages, returns = compute_gae(
        storage['rewards'], storage['values'], storage['dones'], next_val.item()
    )

    batch = {}
    for k in ['grids', 'k_mask', 'phase', 'min_green_flag', 'actions', 'log_probs', 'hiddens']:
        batch[k] = torch.FloatTensor(np.array(storage[k])).to(device)
    batch['advantages'] = torch.FloatTensor(advantages).to(device)
    batch['returns'] = torch.FloatTensor(returns).to(device)

    return batch


def ppo_update(model, optimizer, batch, n_nodes, n_epochs=N_EPOCHS, minibatch_size=MINIBATCH_SIZE):
    T = batch['grids'].shape[0]
    indices = np.arange(T)

    total_loss_sum = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, minibatch_size):
            end_idx = min(start + minibatch_size, T)
            mb = indices[start:end_idx]

            grids = batch['grids'][mb]
            k_mask = batch['k_mask'][mb]
            phase = batch['phase'][mb]
            mg = batch['min_green_flag'][mb]
            old_actions = batch['actions'][mb].long()
            old_lp = batch['log_probs'][mb]
            hidden = batch['hiddens'][mb].view(len(mb), 1, n_nodes, D_MODEL)[:, 0]

            _, new_lp, entropy, values, _ = model.get_action_and_value(
                grids, k_mask, phase, mg,
                gru_hidden=hidden,
                action=old_actions,
            )

            adv = batch['advantages'][mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = (new_lp - old_lp).exp()
            adv_expanded = adv.unsqueeze(-1).expand_as(ratio)
            surr1 = ratio * adv_expanded
            surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_expanded
            policy_loss = -torch.min(surr1, surr2).mean()

            ret = batch['returns'][mb]
            value_loss = ((values - ret) ** 2).mean()

            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return total_loss_sum / max(n_updates, 1)


def train():
    env = NetworkEnv(
        net_file=NET_FILE, route_file=ROUTE_FILE,
        num_seconds=NUM_SECONDS, delta_time=DELTA_TIME,
        min_green=MIN_GREEN, max_green=MAX_GREEN, sumo_seed=42,
    )

    model = HierarchicalSignalNet(
        n_nodes=env.n_nodes,
        max_k=env.max_k,
        max_phases=env.max_phases,
        controllable_idx=env.controllable_idx,
        edge_index=env.inter_edge_index,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    print(f"MAPPO Training on {env.n_nodes}-node network")
    print(f"Controllable: {env.n_controllable} / {env.n_nodes}")
    print(f"Max approaches: {env.max_k}, Max phases: {env.max_phases}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Updates: {TOTAL_UPDATES} x {N_STEPS} steps")

    best_reward = -float('inf')

    for update in range(1, TOTAL_UPDATES + 1):
        model.eval()
        batch = collect_rollout(env, model, N_STEPS, DEVICE)

        model.train()
        avg_loss = ppo_update(model, optimizer, batch, env.n_nodes)

        ep_reward = batch['rewards'].sum().item()
        ep_dones = batch['dones'].sum().item()
        avg_ep_reward = ep_reward / max(ep_dones, 1)

        if update % 5 == 0:
            print(f"Update {update}/{TOTAL_UPDATES}: "
                  f"loss={avg_loss:.4f} "
                  f"avg_reward={avg_ep_reward:.2f} "
                  f"episodes={int(ep_dones)}")

        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            torch.save({
                'model_state_dict': model.state_dict(),
                'update': update,
                'best_reward': best_reward,
                'n_nodes': env.n_nodes,
                'max_k': env.max_k,
                'max_phases': env.max_phases,
                'controllable_idx': env.controllable_idx,
            }, str(CKPT_DIR / 'mappo_signal.pt'))

    torch.save({
        'model_state_dict': model.state_dict(),
        'update': TOTAL_UPDATES,
        'n_nodes': env.n_nodes,
        'max_k': env.max_k,
        'max_phases': env.max_phases,
        'controllable_idx': env.controllable_idx,
    }, str(CKPT_DIR / 'mappo_signal_final.pt'))
    print(f"Training done. Best avg reward: {best_reward:.2f}")

    env.close()


if __name__ == '__main__':
    train()
