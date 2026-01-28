#!/usr/bin/env python3
"""
Generate publication-quality figures for thesis experiments.
Style: NeurIPS/ICML/ICLR Top-Tier Conference Quality
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from pathlib import Path
from collections import defaultdict

# ============================================================================
# ANTHROPIC RESEARCH COLOR PALETTE
# ============================================================================

ANTHROPIC_CORAL = '#D97757'
ANTHROPIC_TAN = '#C4A77D'
ANTHROPIC_SLATE = '#4A4A4A'
ANTHROPIC_SAND = '#F5F0E8'
ANTHROPIC_TEAL = '#2D8B8B'
ANTHROPIC_CORAL_LT = '#F3C8B4'
ANTHROPIC_TEAL_LT = '#B4D7D7'
ANTHROPIC_TAN_LT = '#EBDFC8'
ANTHROPIC_CREAM = '#FCFAF5'
ANTHROPIC_CORAL_DK = '#B85A3D'
ANTHROPIC_TEAL_DK = '#1E6B6B'

COLORS = {
    'bfn': ANTHROPIC_TEAL,
    'diffusion': ANTHROPIC_CORAL,
    'bfn_light': ANTHROPIC_TEAL_LT,
    'diffusion_light': ANTHROPIC_CORAL_LT,
    'bfn_dark': ANTHROPIC_TEAL_DK,
    'diffusion_dark': ANTHROPIC_CORAL_DK,
    'text': ANTHROPIC_SLATE,
    'grid': '#E5E5E5',
    'background': ANTHROPIC_SAND,
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        
        # Sizes (optimized for column width)
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        
        # Axes
        'axes.linewidth': 0.6,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Ticks
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Grid
        'axes.grid': False,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.5,
        
        # Figure
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': False,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.3,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
    })


def parse_train_log(log_path):
    """Parse training log file to extract metrics per epoch."""
    if not os.path.exists(log_path):
        return None
    
    epoch_losses = {}
    
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            match = re.search(r'Training epoch (\d+).*loss=([0-9.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epoch_losses[epoch] = loss
    
    if epoch_losses:
        max_epoch = max(epoch_losses.keys())
        losses = [epoch_losses.get(i, np.nan) for i in range(max_epoch + 1)]
        return {'train_loss': losses}
    
    return None


def parse_slurm_log(log_path):
    """Parse SLURM log file for final metrics."""
    if not os.path.exists(log_path):
        return None
    
    metrics = {}
    
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
        
        train_loss_match = re.search(r'wandb:\s+train_loss\s+([0-9.]+)', content)
        if train_loss_match:
            metrics['final_train_loss'] = float(train_loss_match.group(1))
        
        mse_match = re.search(r'wandb:\s+train_action_mse_error\s+([0-9.]+)', content)
        if mse_match:
            metrics['final_action_mse'] = float(mse_match.group(1))
        
        val_loss_match = re.search(r'wandb:\s+val_loss\s+([0-9.]+)', content)
        if val_loss_match:
            metrics['final_val_loss'] = float(val_loss_match.group(1))
    
    return metrics if metrics else None


def load_all_experiments(outputs_dir, logs_dir):
    """Load training data from all experiments."""
    experiments = {'bfn': [], 'diffusion': []}
    
    task_map = {
        'thesis_bfn_seed42': 0, 'thesis_bfn_seed43': 1, 'thesis_bfn_seed44': 2,
        'thesis_diffusion_seed42': 3, 'thesis_diffusion_seed43': 4, 'thesis_diffusion_seed44': 5,
    }
    
    for exp_name in sorted(os.listdir(outputs_dir)):
        exp_path = os.path.join(outputs_dir, exp_name)
        if not os.path.isdir(exp_path) or not exp_name.startswith('thesis_'):
            continue
        
        train_log = os.path.join(exp_path, 'train.log')
        history = parse_train_log(train_log)
        
        if history is None:
            continue
        
        model_type = 'bfn' if 'bfn' in exp_name else 'diffusion'
        seed = exp_name.split('seed')[-1]
        
        task_id = task_map.get(exp_name)
        if task_id is not None:
            slurm_log = os.path.join(logs_dir, f'thesis_5441080_{task_id}.out')
            final_metrics = parse_slurm_log(slurm_log)
            if final_metrics:
                history.update(final_metrics)
        
        experiments[model_type].append({
            'name': exp_name,
            'seed': seed,
            'history': history
        })
    
    return experiments


def smooth_curve(y, window=15):
    """Exponential moving average smoothing."""
    y = np.array(y, dtype=float)
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0] if not np.isnan(y[0]) else 0
    for i in range(1, len(y)):
        if np.isnan(y[i]):
            smoothed[i] = smoothed[i-1]
        else:
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


# ============================================================================
# FIGURE 1: Training Curves (Single Column Width)
# ============================================================================

def create_training_curves(experiments, output_dir):
    """Publication-quality training curves."""
    setup_style()
    
    # NeurIPS single column: 3.25 inches
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    
    for model_type, color, label in [
        ('bfn', COLORS['bfn'], 'BFN-Policy (Ours)'),
        ('diffusion', COLORS['diffusion'], 'Diffusion Policy')
    ]:
        all_losses = []
        min_len = float('inf')
        
        for exp in experiments[model_type]:
            if 'train_loss' in exp['history']:
                losses = exp['history']['train_loss']
                all_losses.append(losses)
                min_len = min(min_len, len(losses))
        
        if all_losses and min_len > 0:
            all_losses = np.array([l[:min_len] for l in all_losses])
            mean_loss = np.nanmean(all_losses, axis=0)
            std_loss = np.nanstd(all_losses, axis=0)
            
            # Smooth
            smoothed_mean = smooth_curve(mean_loss, window=25)
            smoothed_std = smooth_curve(std_loss, window=25)
            epochs = np.arange(len(smoothed_mean))
            
            # Plot with confidence band
            ax.plot(epochs, smoothed_mean, color=color, linewidth=1.5, 
                   label=label, zorder=3)
            ax.fill_between(epochs, 
                          smoothed_mean - smoothed_std,
                          smoothed_mean + smoothed_std,
                          color=color, alpha=0.15, linewidth=0, zorder=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_yscale('log')
    ax.set_xlim(0, 600)
    
    # Clean grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC', zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    leg = ax.legend(loc='upper right', borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig1_training_curves.{fmt}'), dpi=300)
    plt.close()
    print("✓ fig1_training_curves")


# ============================================================================
# FIGURE 2: Final Metrics Comparison (Bar Chart)
# ============================================================================

def create_final_metrics(experiments, output_dir):
    """Publication-quality bar chart."""
    setup_style()
    
    # Extract metrics
    bfn_losses = [e['history'].get('final_train_loss', np.nan) for e in experiments['bfn']]
    diff_losses = [e['history'].get('final_train_loss', np.nan) for e in experiments['diffusion']]
    bfn_mse = [e['history'].get('final_action_mse', np.nan) for e in experiments['bfn']]
    diff_mse = [e['history'].get('final_action_mse', np.nan) for e in experiments['diffusion']]
    
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    
    x = np.array([0, 1])
    width = 0.5
    
    # Panel A: Training Loss
    ax = axes[0]
    means = [np.nanmean(bfn_losses), np.nanmean(diff_losses)]
    stds = [np.nanstd(bfn_losses), np.nanstd(diff_losses)]
    
    bars = ax.bar(x, means, width, yerr=stds,
                 color=[COLORS['bfn'], COLORS['diffusion']],
                 edgecolor='white', linewidth=1,
                 capsize=3, error_kw={'linewidth': 1, 'capthick': 1})
    
    ax.set_xticks(x)
    ax.set_xticklabels(['BFN\n(Ours)', 'Diffusion'], fontsize=7)
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) Final Loss', fontsize=9, pad=6)
    ax.set_yscale('log')
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Add ratio annotation
    ratio = means[1] / means[0]
    ax.annotate(f'{ratio:.1f}×', xy=(0.5, means[0] * 1.8),
               ha='center', fontsize=8, fontweight='bold', color=COLORS['bfn'])
    
    # Panel B: Action MSE
    ax = axes[1]
    means = [np.nanmean(bfn_mse), np.nanmean(diff_mse)]
    stds = [np.nanstd(bfn_mse), np.nanstd(diff_mse)]
    
    bars = ax.bar(x, means, width, yerr=stds,
                 color=[COLORS['bfn'], COLORS['diffusion']],
                 edgecolor='white', linewidth=1,
                 capsize=3, error_kw={'linewidth': 1, 'capthick': 1})
    
    ax.set_xticks(x)
    ax.set_xticklabels(['BFN\n(Ours)', 'Diffusion'], fontsize=7)
    ax.set_ylabel('Action MSE')
    ax.set_title('(b) Prediction Error', fontsize=9, pad=6)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig2_final_metrics.{fmt}'), dpi=300)
    plt.close()
    print("✓ fig2_final_metrics")


# ============================================================================
# FIGURE 3: Convergence Speed Analysis
# ============================================================================

def create_convergence_analysis(experiments, output_dir):
    """Publication-quality convergence analysis."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    
    thresholds = [0.1, 0.05, 0.01, 0.005, 0.001]
    convergence_data = {'bfn': [], 'diffusion': []}
    
    for model_type in ['bfn', 'diffusion']:
        for exp in experiments[model_type]:
            if 'train_loss' not in exp['history']:
                continue
            
            losses = exp['history']['train_loss']
            epochs_to_threshold = []
            
            for thresh in thresholds:
                reached = len(losses)
                for epoch, loss in enumerate(losses):
                    if not np.isnan(loss) and loss <= thresh:
                        reached = epoch
                        break
                epochs_to_threshold.append(reached)
            
            convergence_data[model_type].append(epochs_to_threshold)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bfn_means = np.mean(convergence_data['bfn'], axis=0)
    bfn_stds = np.std(convergence_data['bfn'], axis=0)
    diff_means = np.mean(convergence_data['diffusion'], axis=0)
    diff_stds = np.std(convergence_data['diffusion'], axis=0)
    
    ax.bar(x - width/2, bfn_means, width, yerr=bfn_stds,
          label='BFN (Ours)', color=COLORS['bfn'],
          edgecolor='white', linewidth=0.8,
          capsize=2, error_kw={'linewidth': 0.8})
    ax.bar(x + width/2, diff_means, width, yerr=diff_stds,
          label='Diffusion', color=COLORS['diffusion'],
          edgecolor='white', linewidth=0.8,
          capsize=2, error_kw={'linewidth': 0.8})
    
    ax.set_xlabel('Loss Threshold')
    ax.set_ylabel('Epochs to Converge')
    ax.set_xticks(x)
    ax.set_xticklabels([f'≤{t}' for t in thresholds], fontsize=6)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    leg = ax.legend(loc='upper left', borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.5)
    
    # Speedup annotations
    for i in range(len(thresholds)):
        if diff_means[i] > bfn_means[i] and bfn_means[i] > 0:
            speedup = diff_means[i] / bfn_means[i]
            if speedup > 1.2:
                y_pos = max(bfn_means[i], diff_means[i]) + max(bfn_stds[i], diff_stds[i]) + 15
                ax.text(x[i], y_pos, f'{speedup:.1f}×', ha='center', 
                       fontsize=6, color=COLORS['bfn'], fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig3_convergence.{fmt}'), dpi=300)
    plt.close()
    print("✓ fig3_convergence")


# ============================================================================
# FIGURE 4: Combined Hero Figure (Full Width)
# ============================================================================

def create_hero_figure(experiments, output_dir):
    """Publication-quality combined figure."""
    setup_style()
    
    # Full width: 7 inches (two-column)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    
    # --- Panel A: Training Curves ---
    ax = axes[0]
    
    for model_type, color, label in [
        ('bfn', COLORS['bfn'], 'BFN (Ours)'),
        ('diffusion', COLORS['diffusion'], 'Diffusion')
    ]:
        all_losses = []
        min_len = float('inf')
        
        for exp in experiments[model_type]:
            if 'train_loss' in exp['history']:
                losses = exp['history']['train_loss']
                all_losses.append(losses)
                min_len = min(min_len, len(losses))
        
        if all_losses and min_len > 0:
            all_losses = np.array([l[:min_len] for l in all_losses])
            mean_loss = np.nanmean(all_losses, axis=0)
            std_loss = np.nanstd(all_losses, axis=0)
            
            smoothed_mean = smooth_curve(mean_loss, window=25)
            smoothed_std = smooth_curve(std_loss, window=25)
            epochs = np.arange(len(smoothed_mean))
            
            ax.plot(epochs, smoothed_mean, color=color, linewidth=1.2, label=label)
            ax.fill_between(epochs, smoothed_mean - smoothed_std,
                          smoothed_mean + smoothed_std,
                          color=color, alpha=0.15, linewidth=0)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) Training Convergence', fontsize=9, fontweight='bold', pad=6)
    ax.set_yscale('log')
    ax.set_xlim(0, 600)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC')
    ax.set_axisbelow(True)
    leg = ax.legend(loc='upper right', fontsize=6, borderaxespad=0.2)
    leg.get_frame().set_linewidth(0.4)
    
    # --- Panel B: Final Performance ---
    ax = axes[1]
    
    bfn_losses = [e['history'].get('final_train_loss', np.nan) for e in experiments['bfn']]
    diff_losses = [e['history'].get('final_train_loss', np.nan) for e in experiments['diffusion']]
    
    x = np.array([0, 1])
    means = [np.nanmean(bfn_losses), np.nanmean(diff_losses)]
    stds = [np.nanstd(bfn_losses), np.nanstd(diff_losses)]
    
    bars = ax.bar(x, means, 0.5, yerr=stds,
                 color=[COLORS['bfn'], COLORS['diffusion']],
                 edgecolor='white', linewidth=0.8,
                 capsize=2, error_kw={'linewidth': 0.8})
    
    ax.set_xticks(x)
    ax.set_xticklabels(['BFN\n(Ours)', 'Diffusion'], fontsize=7)
    ax.set_ylabel('Final Loss')
    ax.set_title('(b) Final Performance', fontsize=9, fontweight='bold', pad=6)
    ax.set_yscale('log')
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Improvement annotation
    ratio = means[1] / means[0]
    mid_y = np.sqrt(means[0] * means[1])
    ax.annotate('', xy=(0, means[0]), xytext=(1, means[1]),
               arrowprops=dict(arrowstyle='<->', color='#666666', lw=0.8))
    ax.text(0.5, mid_y * 0.5, f'{ratio:.1f}×', ha='center', fontsize=7,
           fontweight='bold', color=COLORS['bfn'])
    
    # --- Panel C: Key Results Table ---
    ax = axes[2]
    ax.axis('off')
    
    # Create mini table
    bfn_loss = np.nanmean(bfn_losses)
    diff_loss = np.nanmean(diff_losses)
    bfn_mse = np.nanmean([e['history'].get('final_action_mse', np.nan) for e in experiments['bfn']])
    diff_mse = np.nanmean([e['history'].get('final_action_mse', np.nan) for e in experiments['diffusion']])
    
    table_data = [
        ['Metric', 'BFN', 'Diff.', 'Δ'],
        ['Loss', f'{bfn_loss:.4f}', f'{diff_loss:.4f}', f'{diff_loss/bfn_loss:.1f}×'],
        ['MSE', f'{bfn_mse:.4f}', f'{diff_mse:.4f}', '—'],
        ['Steps', '20', '100', '5×'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                    colWidths=[0.28, 0.24, 0.24, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.6)
    
    # Style table
    for i in range(4):
        table[(0, i)].set_facecolor('#F0F0F0')
        table[(0, i)].set_text_props(fontweight='bold')
    
    for i in range(1, 4):
        table[(i, 1)].set_facecolor(COLORS['bfn_light'])
        table[(i, 2)].set_facecolor(COLORS['diffusion_light'])
        table[(i, 3)].set_text_props(fontweight='bold', color=COLORS['bfn_dark'])
    
    for i in range(4):
        for j in range(4):
            table[(i, j)].set_edgecolor('#CCCCCC')
            table[(i, j)].set_linewidth(0.5)
    
    ax.set_title('(c) Summary', fontsize=9, fontweight='bold', pad=6)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig_hero.{fmt}'), dpi=300)
    plt.close()
    print("✓ fig_hero")


# ============================================================================
# FIGURE 5: Results Table (Clean Academic Style)
# ============================================================================

def create_results_table(experiments, output_dir):
    """Publication-quality results table."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.axis('off')
    
    # Collect data
    rows = []
    for model_type, label in [('bfn', 'BFN-Policy'), ('diffusion', 'Diffusion')]:
        losses = [e['history'].get('final_train_loss', np.nan) for e in experiments[model_type]]
        mses = [e['history'].get('final_action_mse', np.nan) for e in experiments[model_type]]
        vals = [e['history'].get('final_val_loss', np.nan) for e in experiments[model_type]]
        
        rows.append([
            label,
            f'{np.nanmean(losses):.5f} ± {np.nanstd(losses):.5f}',
            f'{np.nanmean(mses):.4f} ± {np.nanstd(mses):.4f}',
            f'{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}',
        ])
    
    columns = ['Method', 'Train Loss ↓', 'Action MSE', 'Val Loss']
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 2.0)
    
    # Style
    for j in range(4):
        table[(0, j)].set_facecolor('#E8E8E8')
        table[(0, j)].set_text_props(fontweight='bold')
    
    # BFN row
    for j in range(4):
        table[(1, j)].set_facecolor(COLORS['bfn_light'])
    table[(1, 0)].set_text_props(fontweight='bold')
    
    # Diffusion row
    for j in range(4):
        table[(2, j)].set_facecolor(COLORS['diffusion_light'])
    
    for i in range(3):
        for j in range(4):
            table[(i, j)].set_edgecolor('#AAAAAA')
            table[(i, j)].set_linewidth(0.6)
    
    ax.set_title('RoboMimic Lift — Quantitative Results (600 epochs, 3 seeds)',
                fontsize=10, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig4_results_table.{fmt}'), dpi=300)
    plt.close()
    print("✓ fig4_results_table")


# ============================================================================
# MAIN
# ============================================================================

def main():
    outputs_dir = '/dss/dsshome1/0D/ge87gob2/condBFNPol/outputs'
    logs_dir = '/dss/dsshome1/0D/ge87gob2/condBFNPol/logs'
    output_dir = '/dss/dsshome1/0D/ge87gob2/condBFNPol/thesis_figures'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("GENERATING CONFERENCE-QUALITY FIGURES")
    print("Style: NeurIPS / ICML / ICLR")
    print("=" * 50)
    print()
    
    print("Loading experiments...")
    experiments = load_all_experiments(outputs_dir, logs_dir)
    print(f"  BFN: {len(experiments['bfn'])} runs")
    print(f"  Diffusion: {len(experiments['diffusion'])} runs")
    print()
    
    if not experiments['bfn'] or not experiments['diffusion']:
        print("ERROR: Missing experiment data!")
        return
    
    # Summary
    for m in ['bfn', 'diffusion']:
        for e in experiments[m]:
            loss = e['history'].get('final_train_loss', 'N/A')
            print(f"  {e['name']}: loss={loss}")
    print()
    
    print("Generating figures...")
    create_training_curves(experiments, output_dir)
    create_final_metrics(experiments, output_dir)
    create_convergence_analysis(experiments, output_dir)
    create_hero_figure(experiments, output_dir)
    create_results_table(experiments, output_dir)
    
    print()
    print("=" * 50)
    print(f"Saved to: {output_dir}/")
    print("=" * 50)
    
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  {f} ({size:.0f} KB)")


if __name__ == '__main__':
    main()
