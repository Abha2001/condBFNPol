#!/usr/bin/env python3
"""
Publication-quality prediction visualization with clear interpretation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, '/dss/dsshome1/0D/ge87gob2/condBFNPol')
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Anthropic colors
TEAL = '#2D8B8B'
CORAL = '#D97757'
SLATE = '#4A4A4A'
TEAL_LT = '#B4D7D7'
CORAL_LT = '#F3C8B4'


def setup_style():
    """Publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def load_checkpoint_and_dataset(ckpt_path):
    """Load checkpoint and dataset."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    
    if 'ema_model' in ckpt['state_dicts']:
        policy.load_state_dict(ckpt['state_dicts']['ema_model'])
    else:
        policy.load_state_dict(ckpt['state_dicts']['model'])
    
    return policy, dataset


def run_inference(policy, dataset, indices, device='cuda'):
    """Run inference on specific indices."""
    policy.to(device)
    policy.eval()
    
    results = []
    
    with torch.no_grad():
        for idx in indices:
            batch = dataset[idx]
            
            obs_dict = {}
            for key, value in batch['obs'].items():
                if isinstance(value, torch.Tensor):
                    obs_dict[key] = value.unsqueeze(0).to(device)
                elif isinstance(value, np.ndarray):
                    obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            
            gt_action = batch['action']
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.numpy()
            
            pred = policy.predict_action(obs_dict)
            pred_action = pred['action'].cpu().numpy()[0]
            
            min_len = min(gt_action.shape[0], pred_action.shape[0])
            
            results.append({
                'idx': idx,
                'gt': gt_action[:min_len],
                'pred': pred_action[:min_len],
            })
    
    return results


def compute_metrics(results):
    """Compute error metrics."""
    pos_mse_list = []
    grip_acc_list = []
    total_mae_list = []
    
    for r in results:
        gt, pred = r['gt'], r['pred']
        
        # Position MSE (dims 0-2)
        pos_mse = np.mean((gt[:, :3] - pred[:, :3])**2)
        pos_mse_list.append(pos_mse)
        
        # Gripper accuracy (dim 6)
        if gt.shape[1] > 6:
            gt_grip = (gt[:, 6] > 0.5).astype(float)
            pred_grip = (pred[:, 6] > 0.5).astype(float)
            grip_acc = np.mean(gt_grip == pred_grip)
            grip_acc_list.append(grip_acc)
        
        # Total MAE
        total_mae = np.mean(np.abs(gt - pred))
        total_mae_list.append(total_mae)
    
    return {
        'pos_mse': np.mean(pos_mse_list),
        'pos_mse_std': np.std(pos_mse_list),
        'grip_acc': np.mean(grip_acc_list) if grip_acc_list else 0,
        'total_mae': np.mean(total_mae_list),
        'total_mae_std': np.std(total_mae_list),
    }


def create_main_comparison_figure(bfn_results, diff_results, bfn_metrics, diff_metrics, output_dir):
    """Create the main comparison figure."""
    setup_style()
    
    fig = plt.figure(figsize=(7.0, 4.5))
    
    # Layout: 2 rows, 3 columns
    # Row 1: Sample predictions (3 panels)
    # Row 2: Metrics comparison (3 panels)
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    
    # ===== ROW 1: Sample Predictions =====
    sample_idx = 0
    gt = bfn_results[sample_idx]['gt']
    bfn_pred = bfn_results[sample_idx]['pred']
    diff_pred = diff_results[sample_idx]['pred']
    timesteps = np.arange(len(gt))
    
    # Panel (a): X Position
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(timesteps, gt[:, 0], color=SLATE, linestyle='-', linewidth=2, label='Ground Truth')
    ax.plot(timesteps, bfn_pred[:, 0], color=TEAL, linestyle='--', linewidth=1.5, label='BFN (Ours)')
    ax.plot(timesteps, diff_pred[:, 0], color=CORAL, linestyle=':', linewidth=1.5, label='Diffusion')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('X Position')
    ax.set_title('(a) X Trajectory', fontweight='bold')
    ax.legend(loc='best', fontsize=6)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Panel (b): Y Position
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(timesteps, gt[:, 1], color=SLATE, linestyle='-', linewidth=2)
    ax.plot(timesteps, bfn_pred[:, 1], color=TEAL, linestyle='--', linewidth=1.5)
    ax.plot(timesteps, diff_pred[:, 1], color=CORAL, linestyle=':', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Y Position')
    ax.set_title('(b) Y Trajectory', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Panel (c): Gripper
    ax = fig.add_subplot(gs[0, 2])
    grip_dim = 6 if gt.shape[1] > 6 else gt.shape[1] - 1
    ax.plot(timesteps, gt[:, grip_dim], color=SLATE, linestyle='-', linewidth=2)
    ax.plot(timesteps, bfn_pred[:, grip_dim], color=TEAL, linestyle='--', linewidth=1.5)
    ax.plot(timesteps, diff_pred[:, grip_dim], color=CORAL, linestyle=':', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Gripper State')
    ax.set_title('(c) Gripper Action', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # ===== ROW 2: Metrics =====
    
    # Panel (d): Position MSE Bar Chart
    ax = fig.add_subplot(gs[1, 0])
    x = [0, 1]
    heights = [bfn_metrics['pos_mse'], diff_metrics['pos_mse']]
    errors = [bfn_metrics['pos_mse_std'], diff_metrics['pos_mse_std']]
    bars = ax.bar(x, heights, 0.5, yerr=errors, 
                 color=[TEAL, CORAL], edgecolor='white', linewidth=1,
                 capsize=4, error_kw={'linewidth': 1})
    ax.set_xticks(x)
    ax.set_xticklabels(['BFN\n(Ours)', 'Diffusion'])
    ax.set_ylabel('Position MSE')
    ax.set_title('(d) Position Error ↓', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    
    # Add value labels
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f'{h:.4f}',
               ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Winner annotation
    if heights[0] < heights[1]:
        winner = "BFN"
        improvement = (heights[1] - heights[0]) / heights[1] * 100
        ax.annotate(f'BFN {improvement:.0f}% better', xy=(0.5, max(heights)*0.5),
                   ha='center', fontsize=7, color=TEAL, fontweight='bold')
    
    # Panel (e): Total MAE Bar Chart
    ax = fig.add_subplot(gs[1, 1])
    heights = [bfn_metrics['total_mae'], diff_metrics['total_mae']]
    errors = [bfn_metrics['total_mae_std'], diff_metrics['total_mae_std']]
    bars = ax.bar(x, heights, 0.5, yerr=errors,
                 color=[TEAL, CORAL], edgecolor='white', linewidth=1,
                 capsize=4, error_kw={'linewidth': 1})
    ax.set_xticks(x)
    ax.set_xticklabels(['BFN\n(Ours)', 'Diffusion'])
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('(e) Total MAE ↓', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.4f}',
               ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Panel (f): Summary Table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    # Determine winner for each metric
    def winner_str(bfn_val, diff_val, lower_better=True):
        if lower_better:
            if bfn_val < diff_val:
                return "✓ BFN"
            elif diff_val < bfn_val:
                return "✓ Diff"
            else:
                return "Tie"
        else:
            if bfn_val > diff_val:
                return "✓ BFN"
            elif diff_val > bfn_val:
                return "✓ Diff"
            else:
                return "Tie"
    
    table_data = [
        ['Metric', 'BFN', 'Diff', 'Winner'],
        ['Pos MSE ↓', f'{bfn_metrics["pos_mse"]:.4f}', f'{diff_metrics["pos_mse"]:.4f}', 
         winner_str(bfn_metrics["pos_mse"], diff_metrics["pos_mse"])],
        ['Total MAE ↓', f'{bfn_metrics["total_mae"]:.4f}', f'{diff_metrics["total_mae"]:.4f}',
         winner_str(bfn_metrics["total_mae"], diff_metrics["total_mae"])],
        ['Grip Acc ↑', f'{bfn_metrics["grip_acc"]:.1%}', f'{diff_metrics["grip_acc"]:.1%}',
         winner_str(bfn_metrics["grip_acc"], diff_metrics["grip_acc"], lower_better=False)],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                    colWidths=[0.3, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.8)
    
    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#E8E8E8')
        table[(0, j)].set_text_props(fontweight='bold')
    
    # Style data rows
    for i in range(1, 4):
        table[(i, 1)].set_facecolor(TEAL_LT)
        table[(i, 2)].set_facecolor(CORAL_LT)
        # Highlight winner
        winner_text = table_data[i][3]
        if "BFN" in winner_text:
            table[(i, 3)].set_text_props(color=TEAL, fontweight='bold')
        elif "Diff" in winner_text:
            table[(i, 3)].set_text_props(color=CORAL, fontweight='bold')
    
    ax.set_title('(f) Summary', fontweight='bold', pad=10)
    
    plt.suptitle('Prediction Quality Comparison: BFN-Policy vs Diffusion Policy',
                fontsize=10, fontweight='bold', y=0.98)
    
    # Save
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig_prediction_comparison.{fmt}'), dpi=300)
    plt.close()
    print(f"✓ Saved fig_prediction_comparison.pdf/png")


def create_trajectory_figure(bfn_results, diff_results, output_dir):
    """Create multi-sample trajectory comparison."""
    setup_style()
    
    n_samples = min(3, len(bfn_results))
    fig, axes = plt.subplots(n_samples, 3, figsize=(7.0, 2.0 * n_samples))
    
    for i in range(n_samples):
        gt = bfn_results[i]['gt']
        bfn_pred = bfn_results[i]['pred']
        diff_pred = diff_results[i]['pred']
        t = np.arange(len(gt))
        
        # X position
        ax = axes[i, 0]
        ax.plot(t, gt[:, 0], 'k-', linewidth=1.5, label='GT' if i == 0 else None)
        ax.plot(t, bfn_pred[:, 0], color=TEAL, linestyle='--', linewidth=1.2, 
               label='BFN' if i == 0 else None)
        ax.plot(t, diff_pred[:, 0], color=CORAL, linestyle=':', linewidth=1.2,
               label='Diff' if i == 0 else None)
        if i == 0:
            ax.set_title('X Position', fontweight='bold')
            ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel(f'Sample {i+1}')
        if i == n_samples - 1:
            ax.set_xlabel('Timestep')
        ax.grid(True, alpha=0.3)
        
        # Y position
        ax = axes[i, 1]
        ax.plot(t, gt[:, 1], 'k-', linewidth=1.5)
        ax.plot(t, bfn_pred[:, 1], color=TEAL, linestyle='--', linewidth=1.2)
        ax.plot(t, diff_pred[:, 1], color=CORAL, linestyle=':', linewidth=1.2)
        if i == 0:
            ax.set_title('Y Position', fontweight='bold')
        if i == n_samples - 1:
            ax.set_xlabel('Timestep')
        ax.grid(True, alpha=0.3)
        
        # Gripper
        ax = axes[i, 2]
        grip_dim = 6 if gt.shape[1] > 6 else gt.shape[1] - 1
        ax.plot(t, gt[:, grip_dim], 'k-', linewidth=1.5)
        ax.plot(t, bfn_pred[:, grip_dim], color=TEAL, linestyle='--', linewidth=1.2)
        ax.plot(t, diff_pred[:, grip_dim], color=CORAL, linestyle=':', linewidth=1.2)
        if i == 0:
            ax.set_title('Gripper', fontweight='bold')
        if i == n_samples - 1:
            ax.set_xlabel('Timestep')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Action Trajectory Predictions (Black=GT, Teal=BFN, Coral=Diffusion)',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig_trajectories.{fmt}'), dpi=300)
    plt.close()
    print(f"✓ Saved fig_trajectories.pdf/png")


def create_error_analysis_figure(bfn_results, diff_results, output_dir):
    """Create error analysis figure."""
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    
    # Compute per-timestep errors
    bfn_errors = [np.abs(r['gt'] - r['pred']) for r in bfn_results]
    diff_errors = [np.abs(r['gt'] - r['pred']) for r in diff_results]
    
    # Panel (a): Error over time
    ax = axes[0]
    bfn_time_err = np.mean([e.mean(axis=1) for e in bfn_errors], axis=0)
    diff_time_err = np.mean([e.mean(axis=1) for e in diff_errors], axis=0)
    
    ax.plot(bfn_time_err, color=TEAL, linewidth=2, label='BFN (Ours)')
    ax.plot(diff_time_err, color=CORAL, linewidth=2, label='Diffusion')
    ax.set_xlabel('Timestep in Chunk')
    ax.set_ylabel('Mean Abs Error')
    ax.set_title('(a) Error Over Horizon', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Per-dimension error
    ax = axes[1]
    bfn_dim_err = np.mean([e.mean(axis=0) for e in bfn_errors], axis=0)
    diff_dim_err = np.mean([e.mean(axis=0) for e in diff_errors], axis=0)
    
    n_dims = min(7, len(bfn_dim_err))
    x = np.arange(n_dims)
    width = 0.35
    labels = ['X', 'Y', 'Z', 'R1', 'R2', 'R3', 'Grip'][:n_dims]
    
    ax.bar(x - width/2, bfn_dim_err[:n_dims], width, color=TEAL, label='BFN')
    ax.bar(x + width/2, diff_dim_err[:n_dims], width, color=CORAL, label='Diffusion')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Abs Error')
    ax.set_title('(b) Per-Dimension Error', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel (c): Error distribution
    ax = axes[2]
    bfn_flat = np.concatenate([e.flatten() for e in bfn_errors])
    diff_flat = np.concatenate([e.flatten() for e in diff_errors])
    
    bins = np.linspace(0, np.percentile(np.concatenate([bfn_flat, diff_flat]), 95), 40)
    ax.hist(bfn_flat, bins=bins, alpha=0.6, color=TEAL, label=f'BFN (μ={np.mean(bfn_flat):.3f})', density=True)
    ax.hist(diff_flat, bins=bins, alpha=0.6, color=CORAL, label=f'Diff (μ={np.mean(diff_flat):.3f})', density=True)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Density')
    ax.set_title('(c) Error Distribution', fontweight='bold')
    ax.legend(fontsize=6)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig_error_analysis.{fmt}'), dpi=300)
    plt.close()
    print(f"✓ Saved fig_error_analysis.pdf/png")


def main():
    output_dir = '/dss/dsshome1/0D/ge87gob2/condBFNPol/thesis_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("PUBLICATION PREDICTION VISUALIZATION")
    print("=" * 50)
    
    # Find checkpoints
    outputs_dir = '/dss/dsshome1/0D/ge87gob2/condBFNPol/outputs'
    
    bfn_ckpt = None
    diff_ckpt = None
    
    for exp_name in sorted(os.listdir(outputs_dir)):
        if not exp_name.startswith('thesis_'):
            continue
        ckpt_dir = os.path.join(outputs_dir, exp_name, 'checkpoints')
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith('.ckpt'):
                    ckpt_path = os.path.join(ckpt_dir, f)
                    if 'bfn' in exp_name and bfn_ckpt is None:
                        bfn_ckpt = ckpt_path
                    elif 'diffusion' in exp_name and diff_ckpt is None:
                        diff_ckpt = ckpt_path
    
    print(f"BFN: {bfn_ckpt}")
    print(f"Diffusion: {diff_ckpt}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Use same indices for fair comparison
    np.random.seed(42)
    
    # Load BFN
    print("\nLoading BFN...")
    bfn_policy, bfn_dataset = load_checkpoint_and_dataset(bfn_ckpt)
    indices = np.random.choice(len(bfn_dataset), 10, replace=False)
    bfn_results = run_inference(bfn_policy, bfn_dataset, indices, device)
    bfn_metrics = compute_metrics(bfn_results)
    print(f"  BFN Metrics: Pos MSE={bfn_metrics['pos_mse']:.4f}, MAE={bfn_metrics['total_mae']:.4f}")
    del bfn_policy
    torch.cuda.empty_cache()
    
    # Load Diffusion
    print("\nLoading Diffusion...")
    diff_policy, diff_dataset = load_checkpoint_and_dataset(diff_ckpt)
    diff_results = run_inference(diff_policy, diff_dataset, indices, device)
    diff_metrics = compute_metrics(diff_results)
    print(f"  Diff Metrics: Pos MSE={diff_metrics['pos_mse']:.4f}, MAE={diff_metrics['total_mae']:.4f}")
    del diff_policy
    torch.cuda.empty_cache()
    
    # Generate figures
    print("\nGenerating figures...")
    create_main_comparison_figure(bfn_results, diff_results, bfn_metrics, diff_metrics, output_dir)
    create_trajectory_figure(bfn_results, diff_results, output_dir)
    create_error_analysis_figure(bfn_results, diff_results, output_dir)
    
    # Copy to figures_for_professor
    import shutil
    dest_dir = '/dss/dsshome1/0D/ge87gob2/condBFNPol/figures_for_professor'
    for f in os.listdir(output_dir):
        if f.startswith('fig_') and (f.endswith('.pdf') or f.endswith('.png')):
            shutil.copy(os.path.join(output_dir, f), dest_dir)
    
    # Print conclusion
    print("\n" + "=" * 50)
    print("CONCLUSION")
    print("=" * 50)
    
    print(f"\nPosition MSE:  BFN={bfn_metrics['pos_mse']:.4f}  vs  Diff={diff_metrics['pos_mse']:.4f}")
    print(f"Total MAE:     BFN={bfn_metrics['total_mae']:.4f}  vs  Diff={diff_metrics['total_mae']:.4f}")
    print(f"Gripper Acc:   BFN={bfn_metrics['grip_acc']:.1%}  vs  Diff={diff_metrics['grip_acc']:.1%}")
    
    # Determine overall winner
    bfn_wins = 0
    diff_wins = 0
    
    if bfn_metrics['pos_mse'] < diff_metrics['pos_mse']:
        bfn_wins += 1
        print("\n→ Position MSE: BFN WINS ✓")
    else:
        diff_wins += 1
        print("\n→ Position MSE: Diffusion WINS ✓")
    
    if bfn_metrics['total_mae'] < diff_metrics['total_mae']:
        bfn_wins += 1
        print("→ Total MAE: BFN WINS ✓")
    else:
        diff_wins += 1
        print("→ Total MAE: Diffusion WINS ✓")
    
    if bfn_metrics['grip_acc'] > diff_metrics['grip_acc']:
        bfn_wins += 1
        print("→ Gripper Accuracy: BFN WINS ✓")
    else:
        diff_wins += 1
        print("→ Gripper Accuracy: Diffusion WINS ✓")
    
    print(f"\n{'='*50}")
    if bfn_wins > diff_wins:
        print(f"OVERALL WINNER: BFN-Policy ({bfn_wins}/3 metrics)")
    elif diff_wins > bfn_wins:
        print(f"OVERALL WINNER: Diffusion Policy ({diff_wins}/3 metrics)")
    else:
        print("OVERALL: TIE")
    print("=" * 50)


if __name__ == '__main__':
    main()
