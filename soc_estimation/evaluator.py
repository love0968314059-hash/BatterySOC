#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOC估计评估模块
"""

import numpy as np

class SOCEvaluator:
    """SOC估计评估器"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, soc_true, soc_estimated, estimator_name="Unknown"):
        """评估SOC估计结果"""
        if soc_true is None or soc_estimated is None:
            return None
        
        # 移除NaN
        valid_mask = ~(np.isnan(soc_true) | np.isnan(soc_estimated))
        soc_true_clean = soc_true[valid_mask]
        soc_estimated_clean = soc_estimated[valid_mask]
        
        if len(soc_true_clean) == 0:
            return None
        
        # 计算误差
        error = soc_estimated_clean - soc_true_clean
        
        # 评估指标
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))
        mean_error = np.mean(error)
        std_error = np.std(error)
        
        # 误差百分比
        mape = np.mean(np.abs(error / (soc_true_clean + 1e-6))) * 100
        
        # 误差分布
        error_95 = np.percentile(np.abs(error), 95)
        error_99 = np.percentile(np.abs(error), 99)
        
        result = {
            'estimator_name': estimator_name,
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'mean_error': mean_error,
            'std_error': std_error,
            'mape': mape,
            'error_95': error_95,
            'error_99': error_99,
            'n_samples': len(soc_true_clean),
            'error': error,
            'soc_true': soc_true_clean,
            'soc_estimated': soc_estimated_clean
        }
        
        self.results[estimator_name] = result
        return result
    
    def print_summary(self, result):
        """打印评估结果摘要"""
        if result is None:
            print("  评估失败: 缺少数据")
            return
        
        print(f"\n  {result['estimator_name']} 评估结果:")
        print(f"    MAE:  {result['mae']:.3f}%")
        print(f"    RMSE: {result['rmse']:.3f}%")
        print(f"    最大误差: {result['max_error']:.3f}%")
        print(f"    平均误差: {result['mean_error']:.3f}%")
        print(f"    误差标准差: {result['std_error']:.3f}%")
        print(f"    MAPE: {result['mape']:.3f}%")
        print(f"    95%误差: {result['error_95']:.3f}%")
        print(f"    99%误差: {result['error_99']:.3f}%")
        print(f"    样本数: {result['n_samples']:,}")
