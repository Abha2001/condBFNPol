#!/usr/bin/env python3
"""Analyze gripper predictions vs continuous dimensions."""

import sys
sys.path.insert(0, '/dss/dsshome1/0D/ge87gob2/condBFNPol')
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import torch
import numpy as np
import h5py
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
    # Load both policies
    bfn_ckpt = '/dss/dsshome1/0D/ge87gob2/condBFNPol/outputs/thesis_bfn_seed42/checkpoints/epoch=0500-train_loss=0.0005.ckpt'
    diff_ckpt = '/dss/dsshome1/0D/ge87gob2/condBFNPol/outputs/thesis_diffusion_seed42/checkpoints/epoch=0500-train_loss=0.0038.ckpt'

    print("Loading BFN policy...")
    bfn_policy = load_policy(bfn_ckpt)
    print("Loading Diffusion policy...")
    diff_policy = load_policy(diff_ckpt)

    dataset_path = '/dss/dsshome1/0D/ge87gob2/condBFNPol/data/robomimic/datasets/lift/ph/image.hdf5'

    bfn_preds = []
    diff_preds = []
    gt_actions = []

    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())[:5]  # 5 demos
        
        for demo_key in demos:
            demo = f['data'][demo_key]
            T = demo['actions'].shape[0]
            
            for t in [10, 20, 30]:
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

    bfn_preds = np.array(bfn_preds)
    diff_preds = np.array(diff_preds)
    gt_actions = np.array(gt_actions)

    print("\n" + "="*70)
    print("PER-DIMENSION ANALYSIS")
    print("="*70)

    dim_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

    for i in range(7):
        gt = gt_actions[:, i]
        bfn = bfn_preds[:, i]
        diff = diff_preds[:, i]
        
        bfn_mse = np.mean((bfn - gt) ** 2)
        diff_mse = np.mean((diff - gt) ** 2)
        
        print(f"\n{dim_names[i]} (dim {i}):")
        print(f"  GT range: [{gt.min():.4f}, {gt.max():.4f}], mean={gt.mean():.4f}")
        print(f"  BFN range: [{bfn.min():.4f}, {bfn.max():.4f}], mean={bfn.mean():.4f}")
        print(f"  Diff range: [{diff.min():.4f}, {diff.max():.4f}], mean={diff.mean():.4f}")
        print(f"  BFN MSE: {bfn_mse:.6f}")
        print(f"  Diff MSE: {diff_mse:.6f}")

    # Overall MSE breakdown
    print("\n" + "="*70)
    print("MSE BREAKDOWN")
    print("="*70)

    continuous_dims = [0, 1, 2, 3, 4, 5]
    discrete_dim = [6]

    bfn_cont_mse = np.mean((bfn_preds[:, continuous_dims] - gt_actions[:, continuous_dims]) ** 2)
    diff_cont_mse = np.mean((diff_preds[:, continuous_dims] - gt_actions[:, continuous_dims]) ** 2)

    bfn_disc_mse = np.mean((bfn_preds[:, discrete_dim] - gt_actions[:, discrete_dim]) ** 2)
    diff_disc_mse = np.mean((diff_preds[:, discrete_dim] - gt_actions[:, discrete_dim]) ** 2)

    bfn_total_mse = np.mean((bfn_preds - gt_actions) ** 2)
    diff_total_mse = np.mean((diff_preds - gt_actions) ** 2)

    print(f"\nContinuous dims (0-5) MSE:")
    print(f"  BFN: {bfn_cont_mse:.6f}")
    print(f"  Diffusion: {diff_cont_mse:.6f}")

    print(f"\nDiscrete dim (6 - gripper) MSE:")
    print(f"  BFN: {bfn_disc_mse:.6f}")
    print(f"  Diffusion: {diff_disc_mse:.6f}")

    print(f"\nTotal MSE:")
    print(f"  BFN: {bfn_total_mse:.6f}")
    print(f"  Diffusion: {diff_total_mse:.6f}")

    # Calculate contribution
    bfn_weighted = (6 * bfn_cont_mse + bfn_disc_mse) / 7
    diff_weighted = (6 * diff_cont_mse + diff_disc_mse) / 7
    
    print(f"\n% of BFN total MSE from gripper: {100 * bfn_disc_mse / 7 / bfn_total_mse:.1f}%")
    print(f"% of Diff total MSE from gripper: {100 * diff_disc_mse / 7 / diff_total_mse:.1f}%")

    # Gripper accuracy
    print("\n" + "="*70)
    print("GRIPPER CLASSIFICATION ACCURACY")
    print("="*70)
    
    # Convert predictions to discrete (-1 or 1)
    bfn_gripper_discrete = np.sign(bfn_preds[:, 6])
    diff_gripper_discrete = np.sign(diff_preds[:, 6])
    gt_gripper = gt_actions[:, 6]
    
    bfn_gripper_acc = np.mean(bfn_gripper_discrete == gt_gripper) * 100
    diff_gripper_acc = np.mean(diff_gripper_discrete == gt_gripper) * 100
    
    print(f"BFN gripper accuracy: {bfn_gripper_acc:.1f}%")
    print(f"Diffusion gripper accuracy: {diff_gripper_acc:.1f}%")
    
    print(f"\nBFN gripper predictions distribution:")
    print(f"  Values: {bfn_preds[:, 6]}")
    
    print(f"\nDiffusion gripper predictions distribution:")
    print(f"  Values: {diff_preds[:, 6]}")

if __name__ == '__main__':
    main()
