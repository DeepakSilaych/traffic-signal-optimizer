import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sumo
os.environ['SUMO_HOME'] = sumo.SUMO_HOME

import numpy as np
import torch
from architecture.signal_optimizer import HierarchicalSignalNet, NetworkEnv

ROOT = Path(__file__).resolve().parent.parent
NET_FILE = str(ROOT / 'sumo_config' / 'city_network.net.xml')
ROUTE_FILE = str(ROOT / 'sumo_config' / 'city_routes.rou.xml')
CKPT_PATH = str(ROOT / 'checkpoints' / 'signal' / 'mappo_signal.pt')

NUM_SECONDS = 3600
DELTA_TIME = 10
MIN_GREEN = 45
MAX_GREEN = 120
NUM_EVAL_EPISODES = 3
DEVICE = 'cpu'


def to_tensor(obs, device=DEVICE):
    return {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}


def make_env():
    return NetworkEnv(
        net_file=NET_FILE, route_file=ROUTE_FILE,
        num_seconds=NUM_SECONDS, delta_time=DELTA_TIME,
        min_green=MIN_GREEN, max_green=MAX_GREEN, sumo_seed=42,
    )


def run_episode(env, model=None):
    obs, _ = env.reset()
    gru_hidden = None
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        if model is not None:
            t = to_tensor(obs)
            with torch.no_grad():
                action, _, _, _, new_hidden = model.get_action_and_value(
                    t['grids'], t['k_mask'], t['phase'], t['min_green_flag'],
                    gru_hidden=gru_hidden,
                )
            action_np = action[0].numpy()
            gru_hidden = new_hidden
        else:
            action_np = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    return total_reward, steps, info


def evaluate():
    env = make_env()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = HierarchicalSignalNet(
        n_nodes=ckpt['n_nodes'],
        max_k=ckpt['max_k'],
        max_phases=ckpt['max_phases'],
        controllable_idx=ckpt['controllable_idx'],
        edge_index=env.inter_edge_index,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint (update {ckpt.get('update', '?')})")
    print(f"Network: {env.n_nodes} nodes, {env.n_controllable} controllable")

    rewards = []
    for ep in range(NUM_EVAL_EPISODES):
        r, steps, info = run_episode(env, model)
        rewards.append(r)
        wait = info.get('system_total_waiting_time', 0)
        print(f"Episode {ep+1}: reward={r:.2f} steps={steps} sys_wait={wait:.1f}")

    print(f"\nAvg Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    env.close()


def run_baseline():
    env = make_env()
    rewards = []
    for ep in range(NUM_EVAL_EPISODES):
        r, steps, info = run_episode(env, model=None)
        rewards.append(r)
        wait = info.get('system_total_waiting_time', 0)
        print(f"Baseline {ep+1}: reward={r:.2f} steps={steps} sys_wait={wait:.1f}")

    print(f"\nBaseline Avg: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    env.close()


if __name__ == '__main__':
    print("=== Trained MAPPO Agent ===")
    evaluate()
    print("\n=== Random Baseline ===")
    run_baseline()
