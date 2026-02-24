#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOC估计主程序 - 完整版
- 传统算法（AH+OCV, EKF, PF）：所有文件独立运行，带初始SOC偏差
- EKF和PF集成RC参数在线辨识
- AI算法：跨文件训练/测试分离
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Matplotlib配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 添加路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from data_processor import BatteryDataProcessor
from data_resampler import DataResampler
from ocv_curve_builder import OCVCurveBuilder
from realtime_soc_estimator import RealtimeSOCEstimator
from advanced_soc_estimators import ParticleFilterSOC
from improved_ekf_estimator import ImprovedEKFSOCEstimator
from parameter_identifier import BatteryParameterIdentifier
from evaluator import SOCEvaluator

# 尝试导入AI方法
try:
    from improved_ai_estimator import ImprovedAISOCEstimator, TORCH_AVAILABLE
    AI_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    AI_AVAILABLE = False
    print("Warning: PyTorch not available, AI methods disabled")


def calculate_soc_labels(time, current, voltage, capacity, ocv_soc_table=None):
    """计算SOC标签"""
    time = np.asarray(time)
    current = np.asarray(current)
    voltage = np.asarray(voltage)
    
    dt = np.diff(time)
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0], dt])
    dt_median = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    dt[dt <= 0] = dt_median
    dt[dt > dt_median * 10] = dt_median
    
    cumulative_ah = np.cumsum(current * dt / 3600)
    soc_change = cumulative_ah / capacity * 100
    
    initial_soc = None
    
    rest_mask = np.abs(current) < 0.05
    high_voltage_mask = voltage >= 3.5
    high_rest_indices = np.where(rest_mask & high_voltage_mask)[0]
    
    if len(high_rest_indices) > 10:
        best_idx = high_rest_indices[np.argmax(voltage[high_rest_indices])]
        initial_soc = 100.0 - soc_change[best_idx]
    
    if initial_soc is None:
        max_v_idx = np.argmax(voltage)
        if voltage[max_v_idx] >= 3.55:
            initial_soc = 100.0 - soc_change[max_v_idx]
    
    if initial_soc is None:
        min_v_idx = np.argmin(voltage)
        if voltage[min_v_idx] <= 2.5:
            initial_soc = 0.0 - soc_change[min_v_idx]
    
    if initial_soc is None:
        initial_soc = 50.0
    
    soc = initial_soc + soc_change
    
    if soc.min() < 0:
        initial_soc = initial_soc - soc.min()
        soc = initial_soc + soc_change
    if soc.max() > 100:
        initial_soc = initial_soc - (soc.max() - 100)
        soc = initial_soc + soc_change
    
    return np.clip(soc, 0, 100)


def clear_results_directory(output_dir):
    """清除历史结果目录"""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_file(data_file, processor):
    """加载并预处理单个文件"""
    try:
        df = processor.load_data_file(data_file)
        if df is None:
            return None
        
        col_map = processor.identify_columns(df)
        clean_result = processor.clean_data(df, col_map)
        
        voltage_raw = clean_result['voltage']
        current_raw = clean_result['current']
        time_raw = clean_result['time']
        temperature_raw = clean_result['temperature']
        
        if voltage_raw is None or len(voltage_raw) < 100:
            return None
        
        temp_value = processor.extract_temperature_from_path(data_file)
        if temp_value is None:
            temp_value = 30
        
        if temperature_raw is None:
            temperature_raw = np.full(len(voltage_raw), float(temp_value))
        
        # 重采样
        resampler = DataResampler(target_dt=1.0, max_dt_ratio=10.0, verbose=False)
        resampled = resampler.resample(time_raw, voltage_raw, current_raw, temperature_raw)
        
        return {
            'time': resampled['time'],
            'voltage': resampled['voltage'],
            'current': resampled['current'],
            'temperature': resampled['temperature'] if resampled['temperature'] is not None else np.full(len(resampled['time']), float(temp_value)),
            'temp_value': temp_value,
            'filename': data_file.name,
            'filepath': data_file
        }
    except Exception as e:
        print(f"    Error loading {data_file.name}: {e}")
        return None


class EKFWithParameterIdentification:
    """带RC参数在线辨识的EKF"""
    
    def __init__(self, initial_soc, nominal_capacity, ocv_soc_table=None,
                 initial_r0=0.05, initial_r1=0.03, initial_tau=30.0):
        self.nominal_capacity = nominal_capacity
        self.ocv_soc_table = ocv_soc_table
        
        # 状态: [SOC, V1]
        self.soc = initial_soc / 100.0
        self.v1 = 0.0
        
        # 协方差矩阵
        self.P = np.diag([0.01, 0.001])
        
        # 参数辨识器
        self.param_identifier = BatteryParameterIdentifier(
            initial_r0=initial_r0, 
            initial_r1=initial_r1, 
            initial_tau=initial_tau,
            forgetting_factor=0.998
        )
        
        # 噪声参数
        self.Q = np.diag([1e-6, 1e-5])
        self.R = np.array([[0.001]])
        
        self.last_time = None
        self.soc_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
        self.soc_correction_history = []
        self.r0_history = []
        self.r1_history = []
        self.tau_history = []
    
    def _get_ocv(self, soc):
        """获取OCV"""
        if self.ocv_soc_table is None:
            return 3.2 + 0.4 * soc
        soc_percent = soc * 100.0
        soc_values = self.ocv_soc_table[:, 0]
        ocv_values = self.ocv_soc_table[:, 1]
        return np.interp(soc_percent, soc_values, ocv_values)
    
    def _get_docv_dsoc(self, soc):
        """获取dOCV/dSOC"""
        delta = 0.001
        soc_high = min(soc + delta, 1.0)
        soc_low = max(soc - delta, 0.0)
        return (self._get_ocv(soc_high) - self._get_ocv(soc_low)) / (soc_high - soc_low + 1e-10)
    
    def update(self, voltage, current, time=None):
        """单步更新"""
        # 计算时间间隔
        if time is not None and self.last_time is not None:
            dt = time - self.last_time
            if dt <= 0 or dt > 100:
                dt = 1.0
        else:
            dt = 1.0
        if time is not None:
            self.last_time = time
        
        # 获取当前参数
        params = self.param_identifier.get_params()
        r0 = params['r0']
        r1 = params['r1']
        tau = params['tau']
        
        # === 预测步骤 ===
        # SOC预测
        delta_soc = current * dt / 3600 / self.nominal_capacity
        soc_pred = np.clip(self.soc + delta_soc, 0.0, 1.0)
        
        # V1预测
        exp_factor = np.exp(-dt / (tau + 1e-6))
        v1_pred = self.v1 * exp_factor + current * r1 * (1 - exp_factor)
        
        # 状态转移矩阵
        F = np.array([[1.0, 0.0], [0.0, exp_factor]])
        
        # 预测协方差
        P_pred = F @ self.P @ F.T + self.Q
        
        # === 更新步骤 ===
        # 预测电压
        ocv_pred = self._get_ocv(soc_pred)
        v_pred = ocv_pred + current * r0 + v1_pred
        
        # 新息
        innovation = voltage - v_pred
        self.innovation_history.append(innovation)
        
        # 观测雅可比矩阵
        docv_dsoc = self._get_docv_dsoc(soc_pred)
        H = np.array([[docv_dsoc, 1.0]])
        
        # 卡尔曼增益
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T / (S[0, 0] + 1e-10)
        
        # 自适应增益调整
        innovation_mv = abs(innovation) * 1000
        if innovation_mv < 50:
            gain_factor = 0.3
        elif innovation_mv < 100:
            gain_factor = 0.1
        else:
            gain_factor = 0.02
        
        # 更新状态
        soc_correction = K[0, 0] * gain_factor * innovation
        self.soc = np.clip(soc_pred + soc_correction, 0.0, 1.0)
        self.v1 = v1_pred + K[1, 0] * innovation
        
        # 更新协方差
        I_KH = np.eye(2) - gain_factor * np.outer(K, H)
        self.P = I_KH @ P_pred
        
        # 更新参数辨识
        self.param_identifier.update(voltage, current, ocv_pred, time, dt)
        
        # 记录
        self.soc_history.append(self.soc * 100.0)
        self.kalman_gain_history.append(K[0, 0] * gain_factor)
        self.soc_correction_history.append(soc_correction * 100.0)
        self.r0_history.append(r0)
        self.r1_history.append(r1)
        self.tau_history.append(tau)
        
        return self.soc * 100.0
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """批量估计"""
        self.soc_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
        self.soc_correction_history = []
        self.r0_history = []
        self.r1_history = []
        self.tau_history = []
        self.last_time = None
        
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        if time is None:
            time = np.arange(len(voltage))
        time = np.asarray(time)
        
        soc_estimated = []
        for i in range(len(voltage)):
            soc = self.update(voltage[i], current[i], time[i])
            soc_estimated.append(soc)
        
        return np.array(soc_estimated)
    
    def get_diagnostics(self):
        """获取诊断信息"""
        return {
            'innovation_history': self.innovation_history,
            'kalman_gain_history': self.kalman_gain_history,
            'soc_correction_history': self.soc_correction_history,
            'r0_history': self.r0_history,
            'r1_history': self.r1_history,
            'tau_history': self.tau_history,
            'final_r0': self.r0_history[-1] if self.r0_history else 0.05,
            'final_r1': self.r1_history[-1] if self.r1_history else 0.03,
            'final_tau': self.tau_history[-1] if self.tau_history else 30.0
        }


class PFWithParameterIdentification:
    """带RC参数在线辨识的粒子滤波"""
    
    def __init__(self, initial_soc, nominal_capacity, ocv_soc_table=None,
                 n_particles=200, initial_r0=0.05, initial_r1=0.03, initial_tau=30.0):
        self.nominal_capacity = nominal_capacity
        self.ocv_soc_table = ocv_soc_table
        self.n_particles = n_particles
        
        # 粒子初始化: [SOC, V1]
        self.particles = np.zeros((n_particles, 2))
        self.particles[:, 0] = np.random.normal(initial_soc / 100.0, 0.02, n_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 1.0)
        self.particles[:, 1] = np.random.normal(0.0, 0.001, n_particles)
        
        self.weights = np.ones(n_particles) / n_particles
        
        # 参数辨识器
        self.param_identifier = BatteryParameterIdentifier(
            initial_r0=initial_r0, 
            initial_r1=initial_r1, 
            initial_tau=initial_tau,
            forgetting_factor=0.998
        )
        
        self.process_noise = 0.001
        self.measurement_noise = 0.001
        
        self.last_time = None
        self.soc_history = []
        self.r0_history = []
        self.r1_history = []
        self.tau_history = []
    
    def _get_ocv(self, soc):
        """获取OCV"""
        if self.ocv_soc_table is None:
            return 3.2 + 0.4 * soc
        soc_percent = soc * 100.0
        soc_values = self.ocv_soc_table[:, 0]
        ocv_values = self.ocv_soc_table[:, 1]
        return np.interp(soc_percent, soc_values, ocv_values)
    
    def update(self, voltage, current, time=None):
        """单步更新"""
        # 计算时间间隔
        if time is not None and self.last_time is not None:
            dt = time - self.last_time
            if dt <= 0 or dt > 100:
                dt = 1.0
        else:
            dt = 1.0
        if time is not None:
            self.last_time = time
        
        # 获取当前参数
        params = self.param_identifier.get_params()
        r0 = params['r0']
        r1 = params['r1']
        tau = params['tau']
        
        # === 预测步骤 ===
        exp_factor = np.exp(-dt / (tau + 1e-6))
        
        for i in range(self.n_particles):
            # SOC更新
            delta_soc = current * dt / 3600 / self.nominal_capacity
            self.particles[i, 0] += delta_soc
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0.0, 1.0)
            self.particles[i, 0] += np.random.normal(0, self.process_noise * 0.1)
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0.0, 1.0)
            
            # V1更新
            self.particles[i, 1] = (self.particles[i, 1] * exp_factor + 
                                    current * r1 * (1 - exp_factor))
            self.particles[i, 1] += np.random.normal(0, self.process_noise * 0.01)
        
        # === 更新步骤 ===
        for i in range(self.n_particles):
            ocv = self._get_ocv(self.particles[i, 0])
            v_pred = ocv + current * r0 + self.particles[i, 1]
            
            error = voltage - v_pred
            log_weight = -0.5 * (error ** 2) / (self.measurement_noise ** 2)
            log_weight = np.clip(log_weight, -50, 50)
            self.weights[i] = np.exp(log_weight)
        
        # 归一化权重
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # 重采样
        weight_sum_sq = np.sum(self.weights ** 2)
        neff = 1.0 / (weight_sum_sq + 1e-10)
        
        if neff < self.n_particles / 2:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            u = (np.random.rand() + np.arange(self.n_particles)) / self.n_particles
            new_particles = np.zeros_like(self.particles)
            j = 0
            for i in range(self.n_particles):
                while j < len(cumsum) - 1 and u[i] > cumsum[j]:
                    j += 1
                new_particles[i] = self.particles[j]
            self.particles = new_particles
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # 估计SOC
        soc_est = np.sum(self.particles[:, 0] * self.weights)
        soc_est = np.clip(soc_est, 0.0, 1.0)
        
        # 更新参数辨识
        ocv_est = self._get_ocv(soc_est)
        self.param_identifier.update(voltage, current, ocv_est, time, dt)
        
        # 记录
        self.soc_history.append(soc_est * 100.0)
        self.r0_history.append(r0)
        self.r1_history.append(r1)
        self.tau_history.append(tau)
        
        return soc_est * 100.0
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """批量估计"""
        self.soc_history = []
        self.r0_history = []
        self.r1_history = []
        self.tau_history = []
        self.last_time = None
        
        # 重新初始化粒子
        initial_soc = self.particles[0, 0]  # 保存初始SOC
        self.particles[:, 0] = np.random.normal(initial_soc, 0.02, self.n_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 1.0)
        self.particles[:, 1] = np.random.normal(0.0, 0.001, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        if time is None:
            time = np.arange(len(voltage))
        time = np.asarray(time)
        
        soc_estimated = []
        for i in range(len(voltage)):
            soc = self.update(voltage[i], current[i], time[i])
            soc_estimated.append(soc)
        
        return np.array(soc_estimated)
    
    def get_diagnostics(self):
        """获取诊断信息"""
        return {
            'r0_history': self.r0_history,
            'r1_history': self.r1_history,
            'tau_history': self.tau_history,
            'final_r0': self.r0_history[-1] if self.r0_history else 0.05,
            'final_r1': self.r1_history[-1] if self.r1_history else 0.03,
            'final_tau': self.tau_history[-1] if self.tau_history else 30.0
        }


def plot_results(output_dir, filename_prefix, time, soc_true, results_dict, 
                initial_soc_biased, true_initial_soc):
    """绘制结果图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # 1. SOC对比
    ax = axes[0]
    ax.plot(time, soc_true, 'k-', label='True SOC', linewidth=2, alpha=0.7)
    for i, (name, data) in enumerate(results_dict.items()):
        mae = data['metrics']['mae']
        ax.plot(time, data['soc_est'], '--', label=f"{name} (MAE={mae:.2f}%)", 
                linewidth=1.5, alpha=0.8, color=colors[i])
    ax.axhline(y=initial_soc_biased, color='r', linestyle=':', alpha=0.3, 
               label=f'Biased init: {initial_soc_biased:.1f}%')
    ax.set_ylabel('SOC (%)')
    ax.set_title(f'SOC Estimation (Initial bias: {initial_soc_biased-true_initial_soc:+.1f}%)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. 误差
    ax = axes[1]
    for i, (name, data) in enumerate(results_dict.items()):
        error = data['soc_est'] - soc_true
        ax.plot(time, error, label=name, linewidth=0.8, alpha=0.7, color=colors[i])
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=5, color='r', linestyle=':', alpha=0.3)
    ax.axhline(y=-5, color='r', linestyle=':', alpha=0.3)
    ax.set_ylabel('SOC Error (%)')
    ax.set_title('Estimation Error')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. MAE柱状图
    ax = axes[2]
    methods = list(results_dict.keys())
    maes = [results_dict[m]['metrics']['mae'] for m in methods]
    bar_colors = ['green' if mae < 5 else 'red' for mae in maes]
    bars = ax.bar(range(len(methods)), maes, color=bar_colors, alpha=0.7)
    ax.axhline(y=5, color='r', linestyle='--', label='5% threshold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('MAE (%)')
    ax.set_title('MAE Comparison')
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mae:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = output_dir / f"{filename_prefix}_results.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_param_identification(output_dir, filename_prefix, time, ekf_diag, pf_diag):
    """绘制参数辨识结果"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    n_ekf = len(ekf_diag.get('r0_history', []))
    n_pf = len(pf_diag.get('r0_history', []))
    time_ekf = time[:n_ekf] if len(time) >= n_ekf else np.arange(n_ekf)
    time_pf = time[:n_pf] if len(time) >= n_pf else np.arange(n_pf)
    
    # R0
    ax = axes[0]
    if ekf_diag.get('r0_history'):
        ax.plot(time_ekf, np.array(ekf_diag['r0_history'])*1000, 'b-', label='EKF', alpha=0.7)
    if pf_diag.get('r0_history'):
        ax.plot(time_pf, np.array(pf_diag['r0_history'])*1000, 'r--', label='PF', alpha=0.7)
    ax.set_ylabel('R0 (mOhm)')
    ax.set_title('Ohmic Resistance Identification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R1
    ax = axes[1]
    if ekf_diag.get('r1_history'):
        ax.plot(time_ekf, np.array(ekf_diag['r1_history'])*1000, 'b-', label='EKF', alpha=0.7)
    if pf_diag.get('r1_history'):
        ax.plot(time_pf, np.array(pf_diag['r1_history'])*1000, 'r--', label='PF', alpha=0.7)
    ax.set_ylabel('R1 (mOhm)')
    ax.set_title('Polarization Resistance Identification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # tau
    ax = axes[2]
    if ekf_diag.get('tau_history'):
        ax.plot(time_ekf, ekf_diag['tau_history'], 'b-', label='EKF', alpha=0.7)
    if pf_diag.get('tau_history'):
        ax.plot(time_pf, pf_diag['tau_history'], 'r--', label='PF', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    ax.set_title('Time Constant Identification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / f"{filename_prefix}_param_id.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def save_results_csv(output_dir, filename, time, voltage, current, temperature, 
                     soc_true, results_dict):
    """保存结果到CSV"""
    data = {
        'time_s': time,
        'voltage_V': voltage,
        'current_A': current,
        'temperature_C': temperature,
        'soc_true_pct': soc_true
    }
    
    for method_name, method_data in results_dict.items():
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('-', '')
        data[f'soc_{safe_name}_pct'] = method_data['soc_est']
        data[f'error_{safe_name}_pct'] = method_data['soc_est'] - soc_true
    
    df = pd.DataFrame(data)
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False, float_format='%.4f')
    return csv_path


def run_single_file(data, actual_capacity, capacity_estimated, ocv_soc_table, 
                   initial_soc_biased, soc_true, evaluator):
    """对单个文件运行所有传统方法"""
    time = data['time']
    voltage = data['voltage']
    current = data['current']
    temperature = data['temperature']
    
    results = {}
    
    # 1. AH+OCV
    estimator = RealtimeSOCEstimator(
        initial_soc=initial_soc_biased,
        nominal_capacity=capacity_estimated,
        ocv_soc_table=ocv_soc_table,
        rest_current_threshold=0.05,
        rest_duration_threshold=30.0
    )
    soc_est = estimator.estimate_batch(voltage, current, time, temperature)
    results['AH+OCV'] = {
        'soc_est': soc_est,
        'metrics': evaluator.evaluate(soc_true, soc_est)
    }
    
    # 2. EKF with Parameter Identification
    ekf = EKFWithParameterIdentification(
        initial_soc=initial_soc_biased,
        nominal_capacity=capacity_estimated,
        ocv_soc_table=ocv_soc_table
    )
    soc_est = ekf.estimate_batch(voltage, current, time, temperature)
    results['EKF-PI'] = {
        'soc_est': soc_est,
        'metrics': evaluator.evaluate(soc_true, soc_est),
        'diagnostics': ekf.get_diagnostics()
    }
    
    # 3. PF with Parameter Identification
    pf = PFWithParameterIdentification(
        initial_soc=initial_soc_biased,
        nominal_capacity=capacity_estimated,
        ocv_soc_table=ocv_soc_table,
        n_particles=200
    )
    soc_est = pf.estimate_batch(voltage, current, time, temperature)
    results['PF-PI'] = {
        'soc_est': soc_est,
        'metrics': evaluator.evaluate(soc_true, soc_est),
        'diagnostics': pf.get_diagnostics()
    }
    
    return results


def main():
    """主函数"""
    print("="*80)
    print("SOC Estimation System - Complete Version")
    print("="*80)
    print("Features:")
    print("  - Traditional methods: Run on ALL files with initial SOC bias")
    print("  - EKF/PF: Online RC parameter identification integrated")
    print("  - AI: Cross-file training/test separation")
    print("="*80)
    
    output_dir = PROJECT_ROOT / "soc_results" / "detailed_results"
    
    print(f"\n[1] Initialization")
    clear_results_directory(output_dir)
    print(f"    Output directory: {output_dir}")
    
    # 查找数据文件
    raw_data_dir = PROJECT_ROOT / "raw_data"
    data_files = []
    
    for temp_dir in raw_data_dir.glob("DST-US06-FUDS-*"):
        if not temp_dir.is_dir():
            continue
        temp_files = [f for f in temp_dir.glob("*.xlsx") 
                     if 'newprofile' not in f.name and '20120809' not in f.name]
        data_files.extend(temp_files)
    
    if len(data_files) == 0:
        print("    Error: No data files found")
        return
    
    print(f"    Found {len(data_files)} data files")
    
    # 配置
    INITIAL_SOC_BIAS = 10.0
    CAPACITY_ERROR = 0.05
    AI_TRAIN_RATIO = 0.8
    
    # 初始化
    processor = BatteryDataProcessor(data_dir=str(raw_data_dir))
    evaluator = SOCEvaluator()
    
    # 获取OCV和容量
    ocv_builder = OCVCurveBuilder(ocv_data_dir=str(raw_data_dir))
    ocv_builder.load_ocv_data(target_temperature=30, use_test_file=True)
    ocv_soc_table = ocv_builder.get_ocv_soc_table()
    actual_capacity = ocv_builder.actual_discharge_capacity or 1.1
    
    np.random.seed(42)
    capacity_estimated = actual_capacity * (1 + np.random.uniform(-CAPACITY_ERROR, CAPACITY_ERROR))
    
    print(f"    True capacity: {actual_capacity:.4f} Ah")
    print(f"    Estimated capacity: {capacity_estimated:.4f} Ah (error: {(capacity_estimated/actual_capacity-1)*100:+.1f}%)")
    print(f"    Initial SOC bias: ±{INITIAL_SOC_BIAS:.0f}%")
    
    # ====== 阶段2: 加载所有数据 ======
    print(f"\n[2] Loading and preprocessing ALL data files...")
    all_data = []
    for i, f in enumerate(data_files):
        data = load_and_preprocess_file(f, processor)
        if data is not None:
            soc_true = calculate_soc_labels(
                data['time'], data['current'], data['voltage'], 
                actual_capacity, ocv_soc_table
            )
            data['soc_true'] = soc_true
            all_data.append(data)
            print(f"    [{i+1}/{len(data_files)}] {f.name}: {len(data['time'])} pts, SOC: {soc_true.min():.1f}-{soc_true.max():.1f}%")
    
    print(f"    Loaded {len(all_data)} files successfully")
    
    # AI训练/测试分割
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    n_train = int(len(all_data) * AI_TRAIN_RATIO)
    train_indices = set(indices[:n_train])
    
    # ====== 阶段3: 对所有文件运行传统方法 ======
    print(f"\n[3] Running traditional methods on ALL files...")
    
    all_results = []
    
    for i, data in enumerate(all_data):
        true_initial_soc = data['soc_true'][0]
        bias_sign = 1 if np.random.rand() > 0.5 else -1
        initial_soc_biased = np.clip(true_initial_soc + bias_sign * INITIAL_SOC_BIAS, 0, 100)
        
        print(f"\n    [{i+1}/{len(all_data)}] {data['filename']}")
        print(f"        True init SOC: {true_initial_soc:.1f}%, Biased: {initial_soc_biased:.1f}%")
        
        results = run_single_file(
            data, actual_capacity, capacity_estimated, ocv_soc_table,
            initial_soc_biased, data['soc_true'], evaluator
        )
        
        # 打印结果
        for method, res in results.items():
            m = res['metrics']
            status = "PASS" if m['mae'] < 5 else "FAIL"
            print(f"        {method:<10}: MAE={m['mae']:.2f}%, RMSE={m['rmse']:.2f}% [{status}]")
        
        # 保存结果
        filename_prefix = Path(data['filename']).stem
        
        plot_results(output_dir, filename_prefix, data['time'], data['soc_true'], 
                    results, initial_soc_biased, true_initial_soc)
        
        # 参数辨识图
        ekf_diag = results.get('EKF-PI', {}).get('diagnostics', {})
        pf_diag = results.get('PF-PI', {}).get('diagnostics', {})
        if ekf_diag or pf_diag:
            plot_param_identification(output_dir, filename_prefix, data['time'], ekf_diag, pf_diag)
        
        save_results_csv(
            output_dir, f"results_{filename_prefix}.csv",
            data['time'], data['voltage'], data['current'], data['temperature'],
            data['soc_true'], results
        )
        
        all_results.append({
            'filename': data['filename'],
            'initial_bias': initial_soc_biased - true_initial_soc,
            'results': results,
            'is_train': i in train_indices
        })
    
    # ====== 阶段4: AI跨文件训练 ======
    if AI_AVAILABLE:
        print(f"\n[4] AI Cross-file Training...")
        
        train_data = [all_data[i] for i in train_indices]
        test_indices = [i for i in range(len(all_data)) if i not in train_indices]
        test_data = [all_data[i] for i in test_indices]
        
        print(f"    Training files: {len(train_data)}")
        print(f"    Test files: {len(test_data)}")
        
        # 训练AI模型
        from improved_ai_estimator import ImprovedAISOCEstimator
        
        ai_trainer = ImprovedAISOCEstimator(
            initial_soc=50.0,
            nominal_capacity=capacity_estimated,
            sequence_length=15,
            hidden_size=64
        )
        
        # 合并训练数据
        all_voltage = np.concatenate([d['voltage'] for d in train_data])
        all_current = np.concatenate([d['current'] for d in train_data])
        all_time = np.concatenate([d['time'] for d in train_data])
        all_temp = np.concatenate([d['temperature'] for d in train_data])
        all_soc = np.concatenate([d['soc_true'] for d in train_data])
        
        print(f"    Total training samples: {len(all_voltage)}")
        
        ai_trainer.estimate_batch(all_voltage, all_current, all_time, all_temp, all_soc)
        
        # 在测试集上评估
        print(f"\n[5] AI Testing on held-out files...")
        ai_test_results = {}
        
        for i, idx in enumerate(test_indices):
            data = all_data[idx]
            true_initial_soc = data['soc_true'][0]
            bias_sign = 1 if np.random.rand() > 0.5 else -1
            initial_soc_biased = np.clip(true_initial_soc + bias_sign * INITIAL_SOC_BIAS, 0, 100)
            
            ai_trainer.initial_soc = initial_soc_biased
            soc_est = ai_trainer.estimate_batch(
                data['voltage'], data['current'], data['time'], 
                data['temperature'], data['soc_true']
            )
            
            metrics = evaluator.evaluate(data['soc_true'], soc_est)
            ai_test_results[data['filename']] = {
                'soc_est': soc_est,
                'metrics': metrics
            }
            
            status = "PASS" if metrics['mae'] < 5 else "FAIL"
            print(f"    [{i+1}/{len(test_indices)}] {data['filename']}: MAE={metrics['mae']:.2f}% [{status}]")
            
            # 更新结果
            all_results[idx]['results']['AI-GRU'] = {
                'soc_est': soc_est,
                'metrics': metrics
            }
    
    # ====== 总结 ======
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # 统计各方法平均MAE
    method_maes = {}
    for res in all_results:
        for method, data in res['results'].items():
            if method not in method_maes:
                method_maes[method] = []
            method_maes[method].append(data['metrics']['mae'])
    
    print(f"\n  Average MAE across all files:")
    for method in sorted(method_maes.keys()):
        maes = method_maes[method]
        avg_mae = np.mean(maes)
        std_mae = np.std(maes)
        pass_rate = sum(1 for m in maes if m < 5) / len(maes) * 100
        status = "PASS" if avg_mae < 5 else "FAIL"
        print(f"    {method:<12}: MAE={avg_mae:.2f}±{std_mae:.2f}%, Pass rate={pass_rate:.0f}% [{status}]")
    
    print(f"\n  Output directory: {output_dir}")
    print(f"  Total files processed: {len(all_results)}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
