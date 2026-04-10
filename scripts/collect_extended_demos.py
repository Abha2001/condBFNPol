"""
Collect demonstration data from trained PPO expert on Extended Lunar Lander.

Saves data in zarr format compatible with BFN/DDPM training.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import zarr
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.lunar_lander_extended import (
    ExtendedHybridLunarLander,
    NUM_DISCRETE,
    MAX_PARAM_DIM,
)
from scripts.train_rl_extended import HybridActorCritic


def load_expert(checkpoint_path: str, device: str = "cuda"):
    """Load trained PPO expert."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create policy
    obs_dim = 8  # Lunar Lander state dim
    policy = HybridActorCritic(obs_dim, NUM_DISCRETE, MAX_PARAM_DIM).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    print(f"Loaded expert from: {checkpoint_path}")
    print(f"  Best reward: {checkpoint.get('best_reward', 'N/A')}")
    print(f"  Timestep: {checkpoint.get('timestep', 'N/A')}")

    return policy


def collect_episode(env, policy, device: str = "cuda"):
    """Collect a single episode."""
    obs, _ = env.reset()

    observations = [obs.copy()]
    discrete_actions = []
    continuous_actions = []
    rewards = []

    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            discrete_action, continuous_action, _, _, _ = policy.get_action(obs_tensor, deterministic=True)

        d_act = discrete_action.item()
        c_act = continuous_action.squeeze(0).cpu().numpy()

        # Store action
        discrete_actions.append(d_act)
        continuous_actions.append(c_act.copy())

        # Step environment
        env_action = {"k": d_act, "x_k": c_act}
        obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        observations.append(obs.copy())
        rewards.append(reward)
        total_reward += reward

    success = terminated and total_reward > 0

    return {
        'observations': np.array(observations[:-1]),  # Exclude final obs
        'discrete_actions': np.array(discrete_actions),
        'continuous_actions': np.array(continuous_actions),
        'rewards': np.array(rewards),
        'total_reward': total_reward,
        'success': success,
        'length': len(rewards),
    }


def create_zarr_dataset(episodes, output_path: str):
    """Create zarr dataset from collected episodes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute episode boundaries
    episode_ends = []
    current_idx = 0
    for ep in episodes:
        current_idx += ep['length']
        episode_ends.append(current_idx)

    # Concatenate all data
    all_obs = np.concatenate([ep['observations'] for ep in episodes], axis=0)
    all_discrete = np.concatenate([ep['discrete_actions'] for ep in episodes], axis=0)
    all_continuous = np.concatenate([ep['continuous_actions'] for ep in episodes], axis=0)

    # Create hybrid action format: [discrete_class, continuous_params...]
    # For BFN: action_dim = 1 + MAX_PARAM_DIM = 2
    all_actions = np.concatenate([
        all_discrete[:, None].astype(np.float32),
        all_continuous.astype(np.float32)
    ], axis=1)

    print(f"\nDataset statistics:")
    print(f"  Total steps: {len(all_obs)}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Obs shape: {all_obs.shape}")
    print(f"  Action shape: {all_actions.shape}")
    print(f"  Discrete actions: {NUM_DISCRETE}")

    # Action distribution
    action_counts = np.bincount(all_discrete, minlength=NUM_DISCRETE)
    print(f"\n  Action distribution:")
    for k in range(NUM_DISCRETE):
        pct = 100 * action_counts[k] / len(all_discrete)
        print(f"    k={k:2d}: {action_counts[k]:5d} ({pct:5.1f}%)")

    # Create zarr store
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)

    # Create data group
    data = root.create_group('data')

    # Store arrays
    data.create_dataset('state', data=all_obs.astype(np.float32), chunks=(1000, all_obs.shape[1]))
    data.create_dataset('action', data=all_actions.astype(np.float32), chunks=(1000, all_actions.shape[1]))

    # Also store separate discrete/continuous for flexibility
    data.create_dataset('discrete_action', data=all_discrete.astype(np.int64), chunks=(1000,))
    data.create_dataset('continuous_action', data=all_continuous.astype(np.float32), chunks=(1000, all_continuous.shape[1]))

    # Store metadata
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    # Store episode stats
    rewards = [ep['total_reward'] for ep in episodes]
    successes = [float(ep['success']) for ep in episodes]

    root.attrs['num_episodes'] = len(episodes)
    root.attrs['num_steps'] = len(all_obs)
    root.attrs['num_discrete'] = NUM_DISCRETE
    root.attrs['max_param_dim'] = MAX_PARAM_DIM
    root.attrs['action_dim'] = 1 + MAX_PARAM_DIM
    root.attrs['obs_dim'] = all_obs.shape[1]
    root.attrs['mean_reward'] = float(np.mean(rewards))
    root.attrs['std_reward'] = float(np.std(rewards))
    root.attrs['success_rate'] = float(np.mean(successes))

    print(f"\nSaved to: {output_path}")
    print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Success rate: {100*np.mean(successes):.1f}%")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PPO checkpoint')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--output', type=str, default='data/lunar_lander_extended/replay.zarr')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_reward', type=float, default=0, help='Filter episodes below this reward')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load expert
    policy = load_expert(args.checkpoint, device)

    # Create environment
    env = ExtendedHybridLunarLander(use_image_obs=False)

    # Collect episodes
    print(f"\nCollecting {args.num_episodes} episodes...")
    episodes = []
    rewards = []
    successes = []

    pbar = tqdm(total=args.num_episodes, desc="Collecting")

    while len(episodes) < args.num_episodes:
        episode = collect_episode(env, policy, device)

        # Filter by minimum reward if specified
        if episode['total_reward'] >= args.min_reward:
            episodes.append(episode)
            rewards.append(episode['total_reward'])
            successes.append(episode['success'])
            pbar.update(1)

            if len(episodes) % 100 == 0:
                pbar.set_postfix({
                    'reward': f"{np.mean(rewards[-100:]):.1f}",
                    'success': f"{100*np.mean(successes[-100:]):.0f}%"
                })

    pbar.close()

    print(f"\nCollection complete!")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Success rate: {100*np.mean(successes):.1f}%")

    # Create zarr dataset
    create_zarr_dataset(episodes, args.output)

    env.close()


if __name__ == '__main__':
    main()
