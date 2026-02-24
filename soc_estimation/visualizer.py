#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOC Estimation Visualization Module
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置字体以避免中文乱码（统一配置，使用英文标签避免乱码）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class SOCVisualizer:
    """SOC Estimation Visualizer"""
    
    def __init__(self, output_dir="soc_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_soc_comparison(self, time, soc_true, soc_estimated_dict, voltage=None, current=None, 
                           filename="soc_comparison.png"):
        """Plot SOC comparison"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # SOC Comparison
        ax1 = axes[0]
        if soc_true is not None:
            ax1.plot(time, soc_true, 'k-', label='True SOC', linewidth=2, alpha=0.7)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(soc_estimated_dict)))
        for i, (name, soc_est) in enumerate(soc_estimated_dict.items()):
            ax1.plot(time, soc_est, '--', label=name, linewidth=1.5, alpha=0.8, color=colors[i])
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('SOC (%)', fontsize=12)
        ax1.set_title('SOC Estimation Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Voltage and Current
        ax2 = axes[1]
        if voltage is not None:
            ax2_twin = ax2.twinx()
            ax2.plot(time, voltage, 'b-', label='Voltage', linewidth=1, alpha=0.7)
            ax2.set_ylabel('Voltage (V)', fontsize=12, color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            if current is not None:
                ax2_twin.plot(time, current, 'r-', label='Current', linewidth=1, alpha=0.7)
                ax2_twin.set_ylabel('Current (A)', fontsize=12, color='r')
                ax2_twin.tick_params(axis='y', labelcolor='r')
                ax2_twin.axhline(0, color='k', linestyle='--', linewidth=0.5)
        
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_title('Voltage and Current', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Error Analysis
        ax3 = axes[2]
        if soc_true is not None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(soc_estimated_dict)))
            for i, (name, soc_est) in enumerate(soc_estimated_dict.items()):
                error = soc_est - soc_true
                ax3.plot(time, error, '--', label=f'{name} Error', linewidth=1, alpha=0.7, color=colors[i])
        
        ax3.axhline(0, color='k', linestyle='-', linewidth=1)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Error (%)', fontsize=12)
        ax3.set_title('Estimation Error', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {output_path}")
    
    def plot_error_distribution(self, evaluator_results, filename="error_distribution.png"):
        """Plot error distribution"""
        n_estimators = len(evaluator_results)
        if n_estimators == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 误差直方图
        ax1 = axes[0, 0]
        for name, result in evaluator_results.items():
            if result and 'error' in result:
                ax1.hist(result['error'], bins=50, alpha=0.6, label=name, density=True)
        ax1.set_xlabel('Error (%)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color='k', linestyle='--', linewidth=1)
        
        # 散点图：估计值 vs 真实值
        ax2 = axes[0, 1]
        for name, result in evaluator_results.items():
            if result and 'soc_true' in result and 'soc_estimated' in result:
                ax2.scatter(result['soc_true'], result['soc_estimated'], 
                           alpha=0.5, s=1, label=name)
        
        if evaluator_results:
            first_result = list(evaluator_results.values())[0]
            if first_result and 'soc_true' in first_result:
                min_soc = first_result['soc_true'].min()
                max_soc = first_result['soc_true'].max()
                ax2.plot([min_soc, max_soc], [min_soc, max_soc], 'r--', linewidth=2, label='Ideal')
        
        ax2.set_xlabel('True SOC (%)', fontsize=12)
        ax2.set_ylabel('Estimated SOC (%)', fontsize=12)
        ax2.set_title('Estimated vs True SOC', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 误差统计对比
        ax3 = axes[1, 0]
        estimators = []
        mae_values = []
        rmse_values = []
        
        for name, result in evaluator_results.items():
            if result:
                estimators.append(name)
                mae_values.append(result['mae'])
                rmse_values.append(result['rmse'])
        
        if estimators:
            x = np.arange(len(estimators))
            width = 0.35
            ax3.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
            ax3.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
            ax3.set_xlabel('Estimator', fontsize=12)
            ax3.set_ylabel('Error (%)', fontsize=12)
            ax3.set_title('Error Metrics Comparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(estimators, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 累积误差分布
        ax4 = axes[1, 1]
        for name, result in evaluator_results.items():
            if result and 'error' in result:
                abs_error = np.abs(result['error'])
                sorted_error = np.sort(abs_error)
                cumulative = np.arange(1, len(sorted_error) + 1) / len(sorted_error)
                ax4.plot(sorted_error, cumulative * 100, label=name, linewidth=2)
        
        ax4.set_xlabel('Absolute Error (%)', fontsize=12)
        ax4.set_ylabel('Cumulative Probability (%)', fontsize=12)
        ax4.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存图表: {output_path}")
