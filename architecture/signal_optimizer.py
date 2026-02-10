import torch
import torch.nn as nn


class LaneGNN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_lanes=8, num_layers=2):
        super().__init__()
        self.num_lanes = num_lanes
        self.node_embed = nn.Linear(node_features, hidden_dim)

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.update_layers.append(nn.GRUCell(hidden_dim, hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, adj):
        h = self.node_embed(node_feats)
        B, N, D = h.shape

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            h_i = h.unsqueeze(2).expand(B, N, N, D)
            h_j = h.unsqueeze(1).expand(B, N, N, D)
            messages = torch.relu(msg_layer(torch.cat([h_i, h_j], dim=-1)))
            messages = (messages * adj.unsqueeze(-1)).sum(dim=2)
            h = upd_layer(messages.view(B * N, D), h.view(B * N, D)).view(B, N, D)

        return self.norm(h)


class SignalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phases=4, num_rnn_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, num_phases),
            nn.Softplus()
        )

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        last = out[:, -1]
        phase_order = self.phase_head(last)
        durations = self.duration_head(last) + 5.0
        return phase_order, durations, hidden


class SignalOptimizer(nn.Module):
    def __init__(self, density_dim, num_lanes=8, num_phases=4, gnn_hidden=64,
                 rnn_hidden=128, seq_len=10, num_gnn_layers=2, num_rnn_layers=2):
        super().__init__()
        self.num_lanes = num_lanes
        self.num_phases = num_phases
        self.seq_len = seq_len

        self.density_to_lane = nn.Linear(density_dim, num_lanes)
        self.lane_gnn = LaneGNN(1, gnn_hidden, num_lanes, num_gnn_layers)
        self.signal_rnn = SignalRNN(gnn_hidden * num_lanes, rnn_hidden, num_phases, num_rnn_layers)

        self.register_buffer('adj', self._default_adjacency(num_lanes))

    def _default_adjacency(self, n):
        adj = torch.ones(n, n) - torch.eye(n)
        return adj

    def forward(self, past_densities, future_densities):
        B, T_past = past_densities.shape[:2]
        T_future = future_densities.shape[1]

        all_densities = torch.cat([past_densities, future_densities], dim=1)
        T_total = T_past + T_future

        lane_feats_seq = []
        for t in range(T_total):
            d = all_densities[:, t].view(B, -1)
            lane_vals = self.density_to_lane(d).unsqueeze(-1)
            adj = self.adj.unsqueeze(0).expand(B, -1, -1)
            gnn_out = self.lane_gnn(lane_vals, adj)
            lane_feats_seq.append(gnn_out.view(B, -1))

        rnn_input = torch.stack(lane_feats_seq, dim=1)
        phase_order, durations, _ = self.signal_rnn(rnn_input)

        return {
            'phase_logits': phase_order,
            'durations': durations
        }
