#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatterySOC Multi-Agent Development Plan
生成思维链分工图 + 迭代循环架构图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_agent_architecture(save_path: str):
    """Draw multi-agent collaboration architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-8, 8)
    ax.axis('off')

    colors = {
        'title': '#1a1a2e',
        'eval': '#e94560',     # Agent-E: Evaluator (red)
        'algo': '#0f3460',     # Agent-A: Algorithm (dark blue)
        'viz': '#16213e',      # Agent-V: Visualization (navy)
        'devops': '#533483',   # Agent-D: DevOps (purple)
        'done': '#2ecc71',
        'target': '#f39c12',
        'flow': '#3498db',
    }

    def box(x, y, w, h, text, color, fs=10, fc='white', bold=False, alpha=0.95):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor='white', lw=2, alpha=alpha)
        ax.add_patch(b)
        wt = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=fc, weight=wt)

    def arrow(x1, y1, x2, y2, color='#3498db', style='->', lw=2.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    def line(x1, y1, x2, y2, color='#bdc3c7', lw=1.5, ls='--'):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=0)

    # ===== Title =====
    box(0, 7, 16, 1.3,
        'BatterySOC Multi-Agent Iterative Development\nTarget: ALL methods MAX Error < 5%',
        colors['title'], fs=16, bold=True)

    # ===== 4 Agent Boxes =====
    box(-8, 4.2, 5.5, 1.8,
        '[Agent-E] Evaluation Agent\n---\nRun tests on 3-4 files\nCompute MAE / RMSE / MAX Error\nDecide PASS/FAIL per method\nIdentify worst-performing method',
        colors['eval'], fs=9, bold=False)

    box(-1.5, 4.2, 5.5, 1.8,
        '[Agent-A] Algorithm Agent\n---\nFix worst method each iteration\nRLS param ID / Adaptive gain\nOCV rest calibration\nFlat-zone noise strategy',
        colors['algo'], fs=9, bold=False)

    box(5, 4.2, 5.5, 1.8,
        '[Agent-V] Visualization Agent\n---\nPlot SOC / Error / Params over time\nGenerate iteration comparison charts\nTrack convergence across iterations\nCreate diagnostic dashboards',
        colors['viz'], fs=9, bold=False)

    box(0, -0.5, 5.5, 1.8,
        '[Agent-D] DevOps Agent\n---\nGit add + commit with metrics\nTag each iteration (iter-N)\nPush to GitHub automatically\nMaintain iteration history log',
        colors['devops'], fs=9, bold=False)

    # ===== Iteration Loop Arrows =====
    # E -> A
    arrow(-5.2, 4.2, -4.3, 4.2, color=colors['eval'], lw=3)
    ax.text(-4.75, 4.8, 'Worst\nmethod', ha='center', fontsize=7, color=colors['eval'])

    # A -> V
    arrow(1.3, 4.2, 2.2, 4.2, color=colors['algo'], lw=3)
    ax.text(1.75, 4.8, 'Fixed\ncode', ha='center', fontsize=7, color=colors['algo'])

    # V -> D (down)
    arrow(5, 3.1, 2.8, 0.5, color=colors['viz'], lw=3)
    ax.text(4.5, 1.8, 'Charts +\nMetrics', ha='center', fontsize=7, color=colors['viz'])

    # D -> E (loop back)
    arrow(-2.8, -0.5, -8, 3.1, color=colors['devops'], lw=3)
    ax.text(-6.5, 1.0, 'Next\niteration', ha='center', fontsize=8, color=colors['devops'], weight='bold')

    # Loop label
    box(0, 1.8, 3, 0.7, 'ITERATE UNTIL\nALL MAX Error < 5%', colors['target'], fs=10, bold=True)

    # ===== Iteration Plan Timeline =====
    y_base = -3
    box(0, y_base, 22, 0.8, 'Iteration Plan Timeline', '#2c3e50', fs=13, bold=True)

    iters = [
        (-9, y_base-1.5, 'Iter 0: Baseline',
         'Run current code as-is\nMeasure MAX Error per method\nIdentify failure points',
         colors['eval'], 'DONE'),
        (-4.5, y_base-1.5, 'Iter 1: Auto-R0 + Rate Limit',
         'Auto-estimate initial R0\nRemove warmup delay\nAdd param rate limiting\nExpand param bounds',
         colors['algo'], 'TODO'),
        (0, y_base-1.5, 'Iter 2: OCV Calibration',
         'Detect rest periods\nVoltage stability check\nOCV lookup correction\nOnce-trigger per rest',
         colors['algo'], 'TODO'),
        (4.5, y_base-1.5, 'Iter 3: Flat-Zone Strategy',
         'EKF: disable V-correction in flat\nPF: increase meas noise in flat\nAH-integration dominant mode\nAdaptive gain by OCV slope',
         colors['algo'], 'TODO'),
        (9, y_base-1.5, 'Iter 4: Final Validation',
         'Full regression 4+ files\nAll methods MAX Error < 5%\nGenerate final comparison\nTag v1.0 release',
         '#2ecc71', 'TODO'),
    ]

    for x, y, title, desc, color, status in iters:
        box(x, y, 4.0, 2.0, f'{title}\n---\n{desc}', color, fs=7.5, alpha=0.9)
        if status == 'DONE':
            ax.text(x, y+1.3, '[DONE]', ha='center', fontsize=8, color='#2ecc71', weight='bold')
        else:
            ax.text(x, y+1.3, '[TODO]', ha='center', fontsize=8, color='#e74c3c', weight='bold')

    # Arrows between iterations
    for i in range(len(iters)-1):
        x1 = iters[i][0] + 2.0
        x2 = iters[i+1][0] - 2.0
        y = iters[i][1]
        arrow(x1, y, x2, y, color='#95a5a6', lw=2)

    # ===== Key Variables to Visualize =====
    y_kv = -6.5
    box(0, y_kv, 22, 1.5,
        'Key Variables Visualized Per Commit:  '
        'SOC(t) vs True  |  Error(t)  |  R0(t) R1(t) tau(t)  |  '
        'Kalman Gain(t)  |  Innovation(t)  |  V1(t)  |  '
        'MAE/RMSE/MaxErr per method per file  |  Iteration convergence trend',
        '#34495e', fs=9, alpha=0.85)

    # ===== Legend =====
    legend_items = [
        mpatches.Patch(color=colors['eval'], label='Agent-E: Evaluation'),
        mpatches.Patch(color=colors['algo'], label='Agent-A: Algorithm'),
        mpatches.Patch(color=colors['viz'], label='Agent-V: Visualization'),
        mpatches.Patch(color=colors['devops'], label='Agent-D: DevOps/Commit'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=10,
              framealpha=0.9, edgecolor='#bdc3c7', ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Architecture saved: {save_path}")


if __name__ == '__main__':
    out = Path(__file__).parent / 'docs'
    out.mkdir(exist_ok=True)
    draw_agent_architecture(str(out / 'multi_agent_architecture.png'))
    print(f"\nAll diagrams saved to: {out}")
