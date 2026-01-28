#!/usr/bin/env python3
"""Detailed analysis of gripper predictions to address professor's questions."""

import sys
sys.path.insert(0, '/dss/dsshome1/0D/ge87gob2/condBFNPol')
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import torch
import numpy as np
import h5py
import json
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    if 'ema_model' in ckpt['state_dicts']:
        policy.load_state_dict(ckpt['state_dicts']['ema_model'])
    else:
        policy.load_state_dict(ckpt['state_dicts']['model'])
    policy.eval()
    return policy

def main():
    print("="*70)
    print("DETAILED GRIPPER ANALYSIS")
    print("="*70)
    
    # Load both policies
    bfn_ckpt = '/dss/dsshome1/0D/ge87gob2/condBFNPol/outputs/thesis_bfn_seed42/checkpoints/epoch=0500-train_loss=0.0005.ckpt'
    diff_ckpt = '/dss/dsshome1/0D/ge87gob2/condBFNPol/outputs/thesis_diffusion_seed42/checkpoints/epoch=0500-train_loss=0.0038.ckpt'

    print("\nLoading BFN policy...")
    bfn_policy = load_policy(bfn_ckpt)
    print("Loading Diffusion policy...")
    diff_policy = load_policy(diff_ckpt)

    dataset_path = '/dss/dsshome1/0D/ge87gob2/condBFNPol/data/robomimic/datasets/lift/ph/image.hdf5'

    bfn_preds = []
    diff_preds = []
    gt_actions = []

    print("\nCollecting predictions from 5 demos, 5 timesteps each...")
    
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())[:5]
        
        for demo_idx, demo_key in enumerate(demos):
            demo = f['data'][demo_key]
            T = demo['actions'].shape[0]
            
            for t in [5, 15, 25, 35, 45]:
                if t >= T - 2:
                    continue
                
                obs_dict = {}
                for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
                    if f'obs/{key}' in demo:
                        data = demo[f'obs/{key}'][t-1:t+1]
                        obs_dict[key] = torch.from_numpy(data).float().unsqueeze(0)
                
                for key in ['agentview_image', 'robot0_eye_in_hand_image']:
                    if f'obs/{key}' in demo:
                        data = demo[f'obs/{key}'][t-1:t+1].astype(np.float32) / 255.0
                        data = np.transpose(data, (0, 3, 1, 2))
                        obs_dict[key] = torch.from_numpy(data).float().unsqueeze(0)
                
                gt = demo['actions'][t]
                gt_actions.append(gt)
                
                with torch.no_grad():
                    bfn_pred = bfn_policy.predict_action(obs_dict)['action'].cpu().numpy()[0, 0]
                    diff_pred = diff_policy.predict_action(obs_dict)['action'].cpu().numpy()[0, 0]
                
                bfn_preds.append(bfn_pred)
                diff_preds.append(diff_pred)
            
            print(f"  Demo {demo_idx+1}/5 done")

    bfn_preds = np.array(bfn_preds)
    diff_preds = np.array(diff_preds)
    gt_actions = np.array(gt_actions)

    print("\n" + "="*70)
    print("RAW GRIPPER VALUES")
    print("="*70)
    
    print(f"\nGround Truth gripper values (dim 6):")
    print(f"  Values: {gt_actions[:, 6]}")
    print(f"  Unique: {np.unique(gt_actions[:, 6])}")
    
    print(f"\nBFN gripper predictions:")
    print(f"  Values: {bfn_preds[:, 6]}")
    print(f"  Min: {bfn_preds[:, 6].min():.4f}, Max: {bfn_preds[:, 6].max():.4f}")
    print(f"  Mean: {bfn_preds[:, 6].mean():.4f}, Std: {bfn_preds[:, 6].std():.4f}")
    
    print(f"\nDiffusion gripper predictions:")
    print(f"  Values: {diff_preds[:, 6]}")
    print(f"  Min: {diff_preds[:, 6].min():.4f}, Max: {diff_preds[:, 6].max():.4f}")
    print(f"  Mean: {diff_preds[:, 6].mean():.4f}, Std: {diff_preds[:, 6].std():.4f}")

    print("\n" + "="*70)
    print("PER-DIMENSION MSE BREAKDOWN")
    print("="*70)

    dim_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    
    results = {
        'bfn': {'per_dim_mse': {}, 'per_dim_mae': {}},
        'diffusion': {'per_dim_mse': {}, 'per_dim_mae': {}}
    }

    for i in range(7):
        gt = gt_actions[:, i]
        bfn = bfn_preds[:, i]
        diff = diff_preds[:, i]
        
        bfn_mse = np.mean((bfn - gt) ** 2)
        diff_mse = np.mean((diff - gt) ** 2)
        bfn_mae = np.mean(np.abs(bfn - gt))
        diff_mae = np.mean(np.abs(diff - gt))
        
        results['bfn']['per_dim_mse'][dim_names[i]] = float(bfn_mse)
        results['diffusion']['per_dim_mse'][dim_names[i]] = float(diff_mse)
        results['bfn']['per_dim_mae'][dim_names[i]] = float(bfn_mae)
        results['diffusion']['per_dim_mae'][dim_names[i]] = float(diff_mae)
        
        print(f"\n{dim_names[i]} (dim {i}):")
        print(f"  BFN MSE: {bfn_mse:.6f}, MAE: {bfn_mae:.6f}")
        print(f"  Diff MSE: {diff_mse:.6f}, MAE: {diff_mae:.6f}")

    # Continuous vs Discrete breakdown
    print("\n" + "="*70)
    print("CONTINUOUS vs DISCRETE BREAKDOWN")
    print("="*70)

    cont_dims = [0, 1, 2, 3, 4, 5]
    disc_dim = 6

    bfn_cont_mse = np.mean((bfn_preds[:, cont_dims] - gt_actions[:, cont_dims]) ** 2)
    diff_cont_mse = np.mean((diff_preds[:, cont_dims] - gt_actions[:, cont_dims]) ** 2)
    
    bfn_disc_mse = np.mean((bfn_preds[:, disc_dim] - gt_actions[:, disc_dim]) ** 2)
    diff_disc_mse = np.mean((diff_preds[:, disc_dim] - gt_actions[:, disc_dim]) ** 2)

    bfn_total_mse = np.mean((bfn_preds - gt_actions) ** 2)
    diff_total_mse = np.mean((diff_preds - gt_actions) ** 2)

    print(f"\nContinuous dimensions (0-5) - position & orientation:")
    print(f"  BFN MSE: {bfn_cont_mse:.6f}")
    print(f"  Diffusion MSE: {diff_cont_mse:.6f}")
    print(f"  Ratio (BFN/Diff): {bfn_cont_mse/diff_cont_mse:.2f}x")

    print(f"\nDiscrete dimension (6) - gripper:")
    print(f"  BFN MSE: {bfn_disc_mse:.6f}")
    print(f"  Diffusion MSE: {diff_disc_mse:.6f}")
    print(f"  Ratio (BFN/Diff): {bfn_disc_mse/diff_disc_mse:.2f}x" if diff_disc_mse > 0 else "  Ratio: inf")

    print(f"\nTotal MSE (all dimensions):")
    print(f"  BFN: {bfn_total_mse:.6f}")
    print(f"  Diffusion: {diff_total_mse:.6f}")
    print(f"  Ratio (BFN/Diff): {bfn_total_mse/diff_total_mse:.2f}x")

    # Contribution analysis
    print("\n" + "="*70)
    print("MSE CONTRIBUTION ANALYSIS")
    print("="*70)
    
    # How much each dimension contributes to total MSE
    print("\nContribution of each dimension to total MSE:")
    for i in range(7):
        bfn_dim_mse = np.mean((bfn_preds[:, i] - gt_actions[:, i]) ** 2)
        diff_dim_mse = np.mean((diff_preds[:, i] - gt_actions[:, i]) ** 2)
        
        bfn_contrib = bfn_dim_mse / (7 * bfn_total_mse) * 100
        diff_contrib = diff_dim_mse / (7 * diff_total_mse) * 100
        
        print(f"  {dim_names[i]}: BFN {bfn_contrib:.1f}%, Diff {diff_contrib:.1f}%")

    # Gripper classification accuracy
    print("\n" + "="*70)
    print("GRIPPER CLASSIFICATION ACCURACY")
    print("="*70)
    
    bfn_gripper_class = np.sign(bfn_preds[:, 6])
    diff_gripper_class = np.sign(diff_preds[:, 6])
    gt_gripper = gt_actions[:, 6]
    
    # Handle zeros (map to -1)
    bfn_gripper_class[bfn_gripper_class == 0] = -1
    diff_gripper_class[diff_gripper_class == 0] = -1
    
    bfn_acc = np.mean(bfn_gripper_class == gt_gripper) * 100
    diff_acc = np.mean(diff_gripper_class == gt_gripper) * 100
    
    print(f"\nBFN gripper accuracy: {bfn_acc:.1f}%")
    print(f"Diffusion gripper accuracy: {diff_acc:.1f}%")
    
    # Save results
    results['bfn']['continuous_mse'] = float(bfn_cont_mse)
    results['diffusion']['continuous_mse'] = float(diff_cont_mse)
    results['bfn']['discrete_mse'] = float(bfn_disc_mse)
    results['diffusion']['discrete_mse'] = float(diff_disc_mse)
    results['bfn']['total_mse'] = float(bfn_total_mse)
    results['diffusion']['total_mse'] = float(diff_total_mse)
    results['bfn']['gripper_accuracy'] = float(bfn_acc)
    results['diffusion']['gripper_accuracy'] = float(diff_acc)
    
    output_file = '/dss/dsshome1/0D/ge87gob2/condBFNPol/thesis_figures/gripper_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary for professor
    print("\n" + "="*70)
    print("SUMMARY FOR PROFESSOR")
    print("="*70)
    print("""
FINDINGS:
1. Gripper is binary (-1 closed, +1 open) in the dataset
2. BFN predicts gripper values that may not be properly discretized
3. Diffusion treats gripper as continuous and predicts intermediate values

KEY INSIGHT:
- If BFN gripper MSE >> Diffusion gripper MSE, the discrete handling needs tuning
- If continuous MSEs are similar, the main difference is in discrete handling
""")

if __name__ == '__main__':
    main()
