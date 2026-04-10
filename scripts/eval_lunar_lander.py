"""
Evaluation script for Lunar Lander policies.

Evaluates trained BFN, Diffusion, and Consistency policies on the
HybridLunarLander environment.

Usage:
    python scripts/eval_lunar_lander.py \
        --checkpoint outputs/train_bfn_lunar_lander/checkpoints/best.ckpt \
        --policy_type bfn \
        --num_episodes 100 \
        --render

    # Compare multiple policies
    python scripts/eval_lunar_lander.py \
        --compare \
        --bfn_ckpt outputs/train_bfn_lunar_lander/checkpoints/best.ckpt \
        --ddpm_ckpt outputs/train_ddpm_lunar_lander/checkpoints/best.ckpt \
        --ddim_ckpt outputs/train_ddim_lunar_lander/checkpoints/best.ckpt \
        --consistency_1step_ckpt outputs/train_consistency_1step_lunar_lander/checkpoints/best.ckpt \
        --num_episodes 50
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, List
import json

import numpy as np
import torch
from tqdm import tqdm
import imageio

# Import environment (direct import to avoid MuJoCo dependency)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the proper HybridLunarLander environment
from environments.lunar_lander import HybridLunarLander, NUM_DISCRETE, MAX_PARAM_DIM

# Action names for logging
ACTION_NAMES = ["COAST", "MAIN_ENGINE", "LEFT_BOOST", "RIGHT_BOOST"]


def save_video(frames: List[np.ndarray], path: str, fps: int = 30):
    """Save frames as video file."""
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception as e:
        print(f"Warning: Failed to save video {path}: {e}")


def load_policy(checkpoint_path: str, policy_type: str, device: str = "cuda",
                 use_ddim_from_ddpm: bool = False, ddim_steps: int = 20):
    """Load a trained policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        policy_type: Type of policy (bfn, ddpm, ddim, etc.)
        device: Device to load on
        use_ddim_from_ddpm: If True and loading ddpm, swap scheduler to DDIM
        ddim_steps: Number of DDIM inference steps when using use_ddim_from_ddpm
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if policy_type == "bfn":
        from policies.bfn_hybrid_lowdim_policy import BFNHybridLowdimPolicy
        policy = BFNHybridLowdimPolicy(
            obs_dim=8,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_discrete_actions=4,
            continuous_param_dim=2,
            device=device,
        )
    elif policy_type in ["ddpm", "ddim"]:
        from policies.diffusion_lowdim_policy import DiffusionLowdimPolicy
        # For ddim, we load as ddpm and swap scheduler later if use_ddim_from_ddpm
        scheduler_type = "ddpm"  # Always load as DDPM
        num_inference_steps = 100
        policy = DiffusionLowdimPolicy(
            obs_dim=8,
            action_dim=6,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            scheduler_type=scheduler_type,
            num_inference_steps=num_inference_steps,
            device=device,
        )
    elif policy_type == "edm":
        from policies.edm_lowdim_policy import EDMLowdimPolicy
        policy = EDMLowdimPolicy(
            obs_dim=8,
            action_dim=6,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=40,  # Default EDM steps
            device=device,
        )
    elif policy_type.startswith("consistency"):
        from policies.consistency_lowdim_policy import ConsistencyLowdimPolicy
        num_inference_steps = 3 if "3step" in policy_type else 1
        policy = ConsistencyLowdimPolicy(
            obs_dim=8,
            action_dim=6,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=num_inference_steps,
            device=device,
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Load weights
    if 'state_dicts' in checkpoint and 'model' in checkpoint['state_dicts']:
        policy.load_state_dict(checkpoint['state_dicts']['model'])
    else:
        policy.load_state_dict(checkpoint)

    # Swap DDPM scheduler to DDIM for faster inference
    if (use_ddim_from_ddpm or policy_type == "ddim") and hasattr(policy, "noise_scheduler"):
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        old_scheduler = policy.noise_scheduler
        ddim = DDIMScheduler(
            num_train_timesteps=old_scheduler.config.num_train_timesteps,
            beta_start=old_scheduler.config.beta_start,
            beta_end=old_scheduler.config.beta_end,
            beta_schedule=old_scheduler.config.beta_schedule,
            clip_sample=True,
            set_alpha_to_one=True,
            prediction_type=getattr(old_scheduler.config, "prediction_type", "epsilon"),
        )
        ddim.set_timesteps(ddim_steps)
        policy.noise_scheduler = ddim
        policy.num_inference_steps = ddim_steps
        print(f"[INFO] Replaced {type(old_scheduler).__name__} with DDIMScheduler ({ddim_steps} steps)")

    policy = policy.to(device)
    policy.eval()

    return policy


def decode_action(action: np.ndarray, policy_type: str) -> Dict:
    """Decode action array to environment action format.

    Args:
        action: Predicted action from policy
        policy_type: Type of policy (determines action format)

    Returns:
        Dict with 'k' (discrete) and 'x_k' (continuous) keys
    """
    if policy_type == "bfn":
        # BFN outputs: [discrete_class, continuous_0, continuous_1]
        discrete_k = int(np.clip(np.round(action[0]), 0, NUM_DISCRETE - 1))
        continuous_x = action[1:1 + MAX_PARAM_DIM]
    else:
        # Diffusion/Consistency outputs: [4 one-hot, 2 continuous]
        one_hot = action[:NUM_DISCRETE]
        discrete_k = int(np.argmax(one_hot))
        continuous_x = action[NUM_DISCRETE:NUM_DISCRETE + MAX_PARAM_DIM]

    # Clip continuous to valid range
    continuous_x = np.clip(continuous_x, -1.0, 1.0)

    return {
        "k": discrete_k,
        "x_k": continuous_x.astype(np.float32)
    }


def run_episode(
    env: HybridLunarLander,
    policy,
    policy_type: str,
    device: str = "cuda",
    render: bool = False,
    max_steps: int = 1000,
    record_video: bool = False,
) -> Dict:
    """Run a single episode.

    Returns:
        Dict with episode statistics and optionally recorded frames
    """
    obs, info = env.reset()
    if isinstance(obs, dict):
        obs = obs['state']

    total_reward = 0.0
    steps = 0
    action_counts = {k: 0 for k in range(NUM_DISCRETE)}
    inference_times = []
    frames = [] if record_video else None

    # Observation buffer for temporal context
    n_obs_steps = 2
    obs_buffer = [obs.copy() for _ in range(n_obs_steps)]

    done = False
    while not done and steps < max_steps:
        # Record frame if requested
        if record_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        # Build observation input
        obs_tensor = torch.from_numpy(
            np.stack(obs_buffer, axis=0)
        ).float().unsqueeze(0).to(device)

        # Get action from policy
        start_time = time.time()
        with torch.no_grad():
            # Normalizer was trained with 'state' key, so pass directly
            obs_dict = {'state': obs_tensor}
            action_pred = policy.predict_action(obs_dict)
            action = action_pred['action'][0, 0].cpu().numpy()  # First step

        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Decode and execute action
        env_action = decode_action(action, policy_type)
        action_counts[env_action['k']] += 1

        obs, reward, terminated, truncated, info = env.step(env_action)
        if isinstance(obs, dict):
            obs = obs['state']

        # Update observation buffer
        obs_buffer.pop(0)
        obs_buffer.append(obs.copy())

        total_reward += reward
        steps += 1
        done = terminated or truncated

        if render:
            env.render()

    # Final frame
    if record_video:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    return {
        'total_reward': total_reward,
        'steps': steps,
        'success': info.get('is_success', total_reward > 200),
        'action_counts': action_counts,
        'mean_inference_time': np.mean(inference_times),
        'terminated': terminated,
        'truncated': truncated,
        'frames': frames,
    }


def evaluate_policy(
    policy,
    policy_type: str,
    num_episodes: int = 100,
    device: str = "cuda",
    render: bool = False,
    seed: int = 42,
    record_video: bool = False,
    video_every: int = 10,
    video_dir: Optional[str] = None,
) -> Dict:
    """Evaluate a policy over multiple episodes.

    Args:
        record_video: Whether to record episode videos
        video_every: Record video every N episodes
        video_dir: Directory to save videos (required if record_video=True)
    """
    # HybridLunarLander already uses render_mode="rgb_array" by default
    env = HybridLunarLander(use_image_obs=False)

    results = {
        'rewards': [],
        'steps': [],
        'successes': [],
        'inference_times': [],
        'action_distribution': {k: 0 for k in range(NUM_DISCRETE)},
        'video_paths': [],
    }

    if record_video and video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    seeds = np.random.randint(0, 2**31 - 1, size=num_episodes)

    for i, ep_seed in enumerate(tqdm(seeds, desc=f"Evaluating {policy_type}")):
        env.reset(seed=int(ep_seed))
        should_record = record_video and (i % video_every == 0)

        ep_result = run_episode(
            env, policy, policy_type, device, render,
            record_video=should_record
        )

        results['rewards'].append(ep_result['total_reward'])
        results['steps'].append(ep_result['steps'])
        results['successes'].append(ep_result['success'])
        results['inference_times'].append(ep_result['mean_inference_time'])

        for k, count in ep_result['action_counts'].items():
            results['action_distribution'][k] += count

        # Save video if recorded
        if should_record and ep_result.get('frames') and video_dir:
            video_path = Path(video_dir) / f"{policy_type}_ep{i:03d}_r{ep_result['total_reward']:.0f}.mp4"
            save_video(ep_result['frames'], str(video_path))
            results['video_paths'].append(str(video_path))
            print(f"  Saved video: {video_path}")

    env.close()

    # Compute summary statistics
    results['mean_reward'] = np.mean(results['rewards'])
    results['std_reward'] = np.std(results['rewards'])
    results['success_rate'] = np.mean(results['successes'])
    results['mean_steps'] = np.mean(results['steps'])
    results['mean_inference_time_ms'] = np.mean(results['inference_times']) * 1000

    # Normalize action distribution
    total_actions = sum(results['action_distribution'].values())
    if total_actions > 0:
        results['action_distribution_pct'] = {
            k: v / total_actions * 100
            for k, v in results['action_distribution'].items()
        }

    return results


def compare_policies(
    checkpoints: Dict[str, str],
    num_episodes: int = 50,
    device: str = "cuda",
    output_path: Optional[str] = None,
    ddim_steps: int = 20,
):
    """Compare multiple policies.

    For 'ddim' policy type, will load DDPM checkpoint and swap scheduler to DDIM.
    """
    results = {}

    for name, ckpt_path in checkpoints.items():
        # For ddim, if no dedicated checkpoint, use ddpm checkpoint with scheduler swap
        use_ddim = (name == "ddim")
        if use_ddim and (ckpt_path is None or not Path(ckpt_path).exists()):
            # Use DDPM checkpoint for DDIM evaluation
            if 'ddpm' in checkpoints and checkpoints['ddpm'] and Path(checkpoints['ddpm']).exists():
                ckpt_path = checkpoints['ddpm']
                print(f"[INFO] Using DDPM checkpoint for DDIM evaluation: {ckpt_path}")
            else:
                print(f"Skipping {name}: no DDPM checkpoint available for DDIM evaluation")
                continue

        if ckpt_path is None or not Path(ckpt_path).exists():
            print(f"Skipping {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print('='*60)

        policy = load_policy(ckpt_path, name, device,
                            use_ddim_from_ddpm=use_ddim, ddim_steps=ddim_steps)
        results[name] = evaluate_policy(
            policy, name, num_episodes, device, render=False
        )

        print(f"\nResults for {name}:")
        print(f"  Mean Reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
        print(f"  Success Rate: {results[name]['success_rate']*100:.1f}%")
        print(f"  Mean Steps: {results[name]['mean_steps']:.1f}")
        print(f"  Inference Time: {results[name]['mean_inference_time_ms']:.2f} ms")

    # Summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Policy':<20} {'Reward':<15} {'Success':<10} {'Steps':<10} {'Inf Time (ms)':<15}")
    print("-"*80)

    for name, res in results.items():
        print(f"{name:<20} {res['mean_reward']:>6.1f} ± {res['std_reward']:<6.1f} "
              f"{res['success_rate']*100:>6.1f}%    "
              f"{res['mean_steps']:>6.1f}    "
              f"{res['mean_inference_time_ms']:>10.2f}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lunar Lander policies")

    # Single policy evaluation
    parser.add_argument('--checkpoint', type=str, help="Path to checkpoint")
    parser.add_argument('--policy_type', type=str,
                       choices=['bfn', 'ddpm', 'ddim', 'edm', 'consistency_1step', 'consistency_3step'],
                       help="Type of policy")

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help="Compare multiple policies")
    parser.add_argument('--bfn_ckpt', type=str, help="BFN checkpoint path")
    parser.add_argument('--ddpm_ckpt', type=str, help="DDPM checkpoint path")
    parser.add_argument('--ddim_ckpt', type=str, help="DDIM checkpoint path")
    parser.add_argument('--consistency_1step_ckpt', type=str,
                       help="Consistency 1-step checkpoint path")
    parser.add_argument('--consistency_3step_ckpt', type=str,
                       help="Consistency 3-step checkpoint path")
    parser.add_argument('--edm_ckpt', type=str, help="EDM checkpoint path")

    # Common options
    parser.add_argument('--num_episodes', type=int, default=100,
                       help="Number of episodes to evaluate")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device to use")
    parser.add_argument('--render', action='store_true',
                       help="Render episodes")
    parser.add_argument('--output', type=str, default=None,
                       help="Output path for results JSON")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    parser.add_argument('--record_video', action='store_true',
                       help="Record episode videos")
    parser.add_argument('--video_every', type=int, default=10,
                       help="Record video every N episodes")
    parser.add_argument('--video_dir', type=str, default='outputs/eval_videos',
                       help="Directory to save videos")

    args = parser.parse_args()

    if args.compare:
        checkpoints = {
            'bfn': args.bfn_ckpt,
            'ddpm': args.ddpm_ckpt,
            'ddim': args.ddim_ckpt,
            'edm': args.edm_ckpt,
            'consistency_1step': args.consistency_1step_ckpt,
            'consistency_3step': args.consistency_3step_ckpt,
        }
        compare_policies(
            checkpoints,
            num_episodes=args.num_episodes,
            device=args.device,
            output_path=args.output,
        )
    else:
        if not args.checkpoint or not args.policy_type:
            parser.error("--checkpoint and --policy_type are required for single policy evaluation")

        policy = load_policy(args.checkpoint, args.policy_type, args.device)
        results = evaluate_policy(
            policy,
            args.policy_type,
            num_episodes=args.num_episodes,
            device=args.device,
            render=args.render,
            seed=args.seed,
            record_video=args.record_video,
            video_every=args.video_every,
            video_dir=args.video_dir,
        )

        print(f"\nResults for {args.policy_type}:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Mean Steps: {results['mean_steps']:.1f}")
        print(f"  Inference Time: {results['mean_inference_time_ms']:.2f} ms")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
