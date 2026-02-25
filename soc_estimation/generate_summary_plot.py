#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive summary visualization for SOC estimation results.
Shows key variables (SOC, R0, R1, Tau) across all methods and temperatures.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def main():
    results_dir = Path(__file__).resolve().parent.parent / "soc_results" / "detailed_results"
    output_dir = results_dir
    
    # Find all result CSV files
    csv_files = sorted(results_dir.glob("results_*.csv"))
    if not csv_files:
        print("No result CSV files found!")
        return
    
    print(f"Found {len(csv_files)} result files")
    
    # Extract temperature labels
    temp_labels = []
    for f in csv_files:
        name = f.stem.replace("results_", "")
        parts = name.split("-")
        # Find the temperature part (number before the date)
        for i, p in enumerate(parts):
            if p.endswith("C") or (p.isdigit() and i > 4):
                pass
        # Extract temperature from directory naming pattern
        temp_str = name.split("FUDS-")[1].split("-")[0] if "FUDS-" in name else "?"
        temp_labels.append(f"{temp_str}°C")
    
    # ===== Figure 1: SOC Comparison Across Temperatures =====
    fig, axes = plt.subplots(len(csv_files), 1, figsize=(16, 4*len(csv_files)))
    if len(csv_files) == 1:
        axes = [axes]
    
    method_colors = {
        'AH+OCV': '#2196F3',
        'EKF-PI': '#FF5722',
        'PF-PI': '#4CAF50'
    }
    
    all_maes = {m: [] for m in method_colors}
    all_max_errors = {m: [] for m in method_colors}
    
    for idx, (csv_file, temp_label) in enumerate(zip(csv_files, temp_labels)):
        df = pd.read_csv(csv_file)
        ax = axes[idx]
        time_hrs = df['time_s'] / 3600
        
        ax.plot(time_hrs, df['soc_true_pct'], 'k-', linewidth=2, alpha=0.7, label='True SOC')
        
        for method, color in method_colors.items():
            safe = method.replace(' ', '_').replace('+', '').replace('-', '').replace('(', '').replace(')', '')
            col = f'soc_{safe}_pct'
            err_col = f'error_{safe}_pct'
            if col in df.columns:
                mae = np.mean(np.abs(df[err_col]))
                max_err = np.max(np.abs(df[err_col]))
                all_maes[method].append(mae)
                all_max_errors[method].append(max_err)
                ax.plot(time_hrs, df[col], '--', color=color, linewidth=1.2, alpha=0.8,
                       label=f'{method} (MAE={mae:.2f}%)')
        
        ax.set_ylabel('SOC (%)', fontsize=11)
        ax.set_title(f'Temperature: {temp_label}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 110])
        
        if idx == len(csv_files) - 1:
            ax.set_xlabel('Time (hours)', fontsize=11)
    
    plt.suptitle('SOC Estimation Results - All Temperatures', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_soc_all_temps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_soc_all_temps.png")
    
    # ===== Figure 2: Error Analysis =====
    fig, axes = plt.subplots(len(csv_files), 1, figsize=(16, 3*len(csv_files)))
    if len(csv_files) == 1:
        axes = [axes]
    
    for idx, (csv_file, temp_label) in enumerate(zip(csv_files, temp_labels)):
        df = pd.read_csv(csv_file)
        ax = axes[idx]
        time_hrs = df['time_s'] / 3600
        
        for method, color in method_colors.items():
            safe = method.replace(' ', '_').replace('+', '').replace('-', '').replace('(', '').replace(')', '')
            err_col = f'error_{safe}_pct'
            if err_col in df.columns:
                ax.plot(time_hrs, df[err_col], color=color, linewidth=0.8, alpha=0.7, label=method)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=5, color='r', linestyle=':', alpha=0.3, label='±5% threshold')
        ax.axhline(y=-5, color='r', linestyle=':', alpha=0.3)
        ax.fill_between(time_hrs, -5, 5, alpha=0.05, color='green')
        ax.set_ylabel('Error (%)', fontsize=11)
        ax.set_title(f'SOC Error - {temp_label}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if idx == len(csv_files) - 1:
            ax.set_xlabel('Time (hours)', fontsize=11)
    
    plt.suptitle('SOC Estimation Error Analysis', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_error_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_error_analysis.png")
    
    # ===== Figure 3: MAE Bar Chart Summary =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Per-temperature MAE
    ax = axes[0]
    x = np.arange(len(temp_labels))
    width = 0.25
    for i, (method, color) in enumerate(method_colors.items()):
        maes = all_maes.get(method, [])
        if maes:
            bars = ax.bar(x + i*width - width, maes, width, label=method, color=color, alpha=0.8)
            for bar, mae in zip(bars, maes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{mae:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=5, color='r', linestyle='--', linewidth=2, alpha=0.5, label='5% Target')
    ax.set_xticks(x)
    ax.set_xticklabels(temp_labels)
    ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title('MAE by Temperature', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Overall summary
    ax = axes[1]
    methods = list(method_colors.keys())
    avg_maes = [np.mean(all_maes[m]) if all_maes[m] else 0 for m in methods]
    max_maes = [np.max(all_maes[m]) if all_maes[m] else 0 for m in methods]
    colors_list = [method_colors[m] for m in methods]
    
    x = np.arange(len(methods))
    bars1 = ax.bar(x - 0.2, avg_maes, 0.35, label='Avg MAE', color=colors_list, alpha=0.8)
    bars2 = ax.bar(x + 0.2, max_maes, 0.35, label='Max MAE', color=colors_list, alpha=0.4, 
                   edgecolor=colors_list, linewidth=2)
    
    for bar, val in zip(bars1, avg_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, max_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=5, color='r', linestyle='--', linewidth=2, alpha=0.5, label='5% Target')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title('Overall Performance Summary', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add PASS/FAIL verdict
    for i, (avg, mx) in enumerate(zip(avg_maes, max_maes)):
        verdict = "PASS" if avg < 5 else "FAIL"
        color = 'green' if avg < 5 else 'red'
        ax.text(i, -0.8, verdict, ha='center', va='top', fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    fig.savefig(output_dir / "summary_mae_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_mae_comparison.png")
    
    # ===== Print Summary Table =====
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'Avg MAE':>8} {'Max MAE':>8} {'Avg MaxErr':>10} {'Status':>8}")
    print(f"  {'-'*50}")
    for method in methods:
        if all_maes[method]:
            avg = np.mean(all_maes[method])
            mx = np.max(all_maes[method])
            avg_max = np.mean(all_max_errors[method]) if all_max_errors[method] else 0
            status = "PASS" if avg < 5 else "FAIL"
            print(f"  {method:<12} {avg:>7.2f}% {mx:>7.2f}% {avg_max:>9.2f}% [{status}]")
    print(f"{'='*70}")
    print("Done! All summary visualizations saved.")


if __name__ == "__main__":
    main()
