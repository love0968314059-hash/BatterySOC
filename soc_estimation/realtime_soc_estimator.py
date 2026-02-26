#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时SOC估计器
模拟实时数据输入，支持流式处理
"""

import numpy as np
from scipy.interpolate import interp1d


class RealtimeSOCEstimator:
    """
    实时SOC估计器（流式处理）
    支持逐点输入，实时更新SOC估计值
    """
    
    def __init__(self, 
                 initial_soc=50.0,
                 nominal_capacity=1.1,
                 ocv_soc_table=None,
                 rest_current_threshold=0.05,  # Rest current threshold (A)
                 rest_duration_threshold=30.0,  # Rest duration threshold (seconds)
                 enable_interpolation=False,
                 coulombic_efficiency_charge=1.0,  # Charging coulombic efficiency (LFP ~1.0)
                 coulombic_efficiency_discharge=1.0,  # Discharging coulombic efficiency
                 current_filter_window=1):  # Current filter window (1=no filter to avoid delay)
        """
        Initialize real-time SOC estimator
        
        Args:
            initial_soc: Initial SOC (%)
            nominal_capacity: Nominal capacity (Ah)
            ocv_soc_table: OCV-SOC lookup table, format: [[soc1, ocv1], [soc2, ocv2], ...]
            rest_current_threshold: Rest current threshold (A)
            rest_duration_threshold: Rest duration threshold (seconds)
            enable_interpolation: Enable interpolation correction (default False)
            coulombic_efficiency_charge: Coulombic efficiency for charging (default 0.99)
            coulombic_efficiency_discharge: Coulombic efficiency for discharging (default 1.0)
            current_filter_window: Current filter window size (default 5)
        """
        self.initial_soc = initial_soc
        self.current_soc = initial_soc  # Current SOC estimate
        self.nominal_capacity = nominal_capacity
        self.rest_current_threshold = rest_current_threshold
        self.rest_duration_threshold = rest_duration_threshold
        self.enable_interpolation = enable_interpolation
        self.coulombic_efficiency_charge = coulombic_efficiency_charge
        self.coulombic_efficiency_discharge = coulombic_efficiency_discharge
        self.current_filter_window = current_filter_window
        
        # OCV-SOC映射
        if ocv_soc_table is not None and len(ocv_soc_table) > 0:
            self.ocv_soc_table = np.array(ocv_soc_table)
            ocv_soc_soc = self.ocv_soc_table[:, 0]
            ocv_soc_ocv = self.ocv_soc_table[:, 1]
            self.ocv_to_soc_interp = interp1d(
                ocv_soc_ocv, ocv_soc_soc,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
        else:
            self.ocv_soc_table = None
            self.ocv_to_soc_interp = None
        
        # History buffer (for rest detection and interpolation)
        self.voltage_history = []
        self.current_history = []
        self.time_history = []
        self.soc_history = []
        self.rest_duration_history = []  # Rest duration for each point
        
        # Flag: track if already in rest state (for OCV calibration)
        self.was_resting = False
        
        # Previous time point (for dt calculation)
        self.last_time = None
        self.last_dt = 1.0  # Default time interval (seconds)
        
        # Current filter buffer
        self.current_filter_buffer = []
        
        # 统计信息
        self.n_ocv_calibrations = 0
        self.n_ah_updates = 0
    
    def update(self, voltage, current, time=None, temperature=None):
        """
        实时更新SOC估计值（单点输入）
        
        Args:
            voltage: 电压 (V)
            current: 电流 (A)
            time: 时间 (s)，如果为None则使用历史数据推断
            temperature: 温度 (°C)，可选
        
        Returns:
            当前SOC估计值 (%)
        """
        # 处理时间
        if time is None:
            if self.last_time is not None:
                time = self.last_time + self.last_dt
            else:
                time = 0.0
        
        # 计算时间间隔（关键修复：必须使用实际时间差，不能随意限制）
        if self.last_time is not None:
            dt = time - self.last_time
            if dt <= 0:
                # 时间倒退或不变，使用上一个dt（可能是数据问题）
                dt = self.last_dt if self.last_dt > 0 else 1.0
            elif dt > 3600:  # 超过1小时，可能是数据跳跃，使用上一个dt
                # 但记录警告
                dt = self.last_dt if self.last_dt > 0 else 1.0
            else:
                # 正常情况：使用实际时间差
                self.last_dt = dt
        else:
            # 第一个点：如果没有提供时间，假设1秒间隔
            dt = 1.0
            self.last_dt = dt
        
        self.last_time = time
        
        # Current filtering (moving average)
        self.current_filter_buffer.append(current)
        if len(self.current_filter_buffer) > self.current_filter_window:
            self.current_filter_buffer.pop(0)
        filtered_current = np.mean(self.current_filter_buffer)
        
        # Determine if resting
        is_rest = abs(filtered_current) < self.rest_current_threshold
        
        # Calculate rest duration
        if is_rest:
            if len(self.rest_duration_history) > 0:
                rest_duration = self.rest_duration_history[-1] + dt
            else:
                rest_duration = dt
        else:
            rest_duration = 0.0
        
        # Determine if OCV calibration condition is met (only calibrate once when threshold is first reached)
        # Fix: use was_resting flag to track if already in rest state
        # Improved: only calibrate if voltage is stable and difference is reasonable
        can_calibrate = False
        if (is_rest and 
            rest_duration >= self.rest_duration_threshold and
            not self.was_resting and
            self.ocv_soc_table is not None):
            # Additional check: voltage stability (voltage should be stable during rest)
            if len(self.voltage_history) >= 10:
                recent_voltages = np.array(self.voltage_history[-10:])
                voltage_std = np.std(recent_voltages)
                if voltage_std < 0.005:  # Voltage is stable (<5mV variation)
                    can_calibrate = True
            else:
                can_calibrate = True  # If not enough history, allow calibration
        
        # Update was_resting flag
        if is_rest and rest_duration >= self.rest_duration_threshold:
            self.was_resting = True
        elif not is_rest:
            self.was_resting = False  # Reset flag once leaving rest state
        
        # Update SOC
        if can_calibrate:
            # OCV calibration: use OCV-SOC curve mapping (only calibrate once when entering rest state)
            ocv_soc_ocv = self.ocv_soc_table[:, 1]
            
            if voltage < ocv_soc_ocv.min():
                soc_from_ocv = 0.0
            elif voltage > ocv_soc_ocv.max():
                soc_from_ocv = 100.0
            else:
                soc_from_ocv = float(self.ocv_to_soc_interp(voltage))
            
            # OCV calibration: very conservative weighted correction
            # Only calibrate if difference is reasonable (<10%) and voltage is very stable
            # Use minimal weight (10%) to minimize impact of OCV calibration errors
            soc_diff = soc_from_ocv - self.current_soc
            if abs(soc_diff) < 10.0:  # Only calibrate if difference < 10%
                calibration_weight = 0.1
                self.current_soc = self.current_soc + soc_diff * calibration_weight
                self.n_ocv_calibrations += 1
            # If difference is too large, skip calibration to avoid large errors
        else:
            # AH integration update (including rest periods, as OCV calibration only happens once when threshold is reached)
            # Key fix: continue integration even at boundaries, allow temporary exceedance before clipping
            # Apply coulombic efficiency based on current direction
            if filtered_current > 0:
                efficiency = self.coulombic_efficiency_charge  # Charging efficiency
            else:
                efficiency = self.coulombic_efficiency_discharge  # Discharging efficiency
            
            delta_soc = filtered_current * dt / 3600 / self.nominal_capacity * 100 * efficiency
            
            # Update SOC: always allow changes, clip only at the end
            # This ensures accurate integration even near boundaries
            self.current_soc = self.current_soc + delta_soc
            self.n_ah_updates += 1
        
        # Clip to 0-100% range only at the end (prevent numerical errors)
        # This allows accurate integration while maintaining physical bounds
        self.current_soc = np.clip(self.current_soc, 0, 100)
        
        # Save history data
        self.voltage_history.append(voltage)
        self.current_history.append(filtered_current)  # Save filtered current
        self.time_history.append(time)
        self.soc_history.append(self.current_soc)
        self.rest_duration_history.append(rest_duration)
        
        # 可选：插值修正（默认不启用）
        if self.enable_interpolation and len(self.soc_history) > 1:
            self._apply_interpolation()
        
        return self.current_soc
    
    def _apply_interpolation(self):
        """
        插值修正函数（可选）
        对非静止点，如果距离静止点较近，进行轻微修正
        """
        if len(self.rest_duration_history) < 2:
            return
        
        # 找到最近的静止点（满足校准条件的点）
        rest_indices = []
        for i in range(len(self.rest_duration_history)):
            if (self.rest_duration_history[i] >= self.rest_duration_threshold and
                abs(self.current_history[i]) < self.rest_current_threshold):
                rest_indices.append(i)
        
        if len(rest_indices) < 1:
            return
        
        # 对当前点进行修正（如果是非静止点）
        current_idx = len(self.soc_history) - 1
        if current_idx not in rest_indices:
            # 找到最近的静止点
            nearest_rest_idx = rest_indices[np.argmin(np.abs(np.array(rest_indices) - current_idx))]
            distance = abs(current_idx - nearest_rest_idx)
            
            # 如果距离不远（<100个点），进行轻微修正
            if distance < 100:
                # 计算修正量（基于静止点的OCV校准误差）
                # 这里简化处理：如果最近的静止点有OCV校准，进行轻微修正
                correction = (self.soc_history[nearest_rest_idx] - self.soc_history[nearest_rest_idx]) * np.exp(-distance/50)
                self.current_soc = self.soc_history[-1] + correction * 0.1  # 轻微修正，权重0.1
                self.current_soc = np.clip(self.current_soc, 0, 100)
                self.soc_history[-1] = self.current_soc
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """
        批量估计SOC（用于验证和对比）
        内部使用实时更新方法
        
        Args:
            voltage: 电压数组 (V)
            current: 电流数组 (A)
            time: 时间数组 (s)
            temperature: 温度数组 (°C)，可选
        
        Returns:
            SOC估计值数组 (%)
        """
        # 重置状态
        self.reset()
        
        # 确保输入是数组
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        if time is None:
            time = np.arange(len(current))
        time = np.asarray(time)
        
        soc_estimated = []
        
        # 逐点实时更新
        for i in range(len(voltage)):
            temp = temperature[i] if temperature is not None else None
            soc = self.update(voltage[i], current[i], time[i], temp)
            soc_estimated.append(soc)
        
        return np.array(soc_estimated)
    
    def reset(self):
        """重置估计器状态"""
        self.current_soc = self.initial_soc
        self.voltage_history = []
        self.current_history = []
        self.time_history = []
        self.soc_history = []
        self.rest_duration_history = []
        self.last_time = None
        self.last_dt = 1.0
        self.n_ocv_calibrations = 0
        self.n_ah_updates = 0
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'n_ocv_calibrations': self.n_ocv_calibrations,
            'n_ah_updates': self.n_ah_updates,
            'current_soc': self.current_soc,
            'total_points': len(self.soc_history)
        }


def apply_interpolation_correction(soc_est, soc_ah, valid_rest_mask, rest_indices, 
                                   enable=False, max_distance=100, correction_weight=0.1):
    """
    插值修正函数（可选，默认不执行）
    
    Args:
        soc_est: SOC估计值数组（已进行OCV校准）
        soc_ah: AH积分SOC数组
        valid_rest_mask: 满足静止条件的掩码
        rest_indices: 静止点索引数组
        enable: 是否启用插值修正（默认False）
        max_distance: 最大修正距离（点数）
        correction_weight: 修正权重
    
    Returns:
        修正后的SOC估计值数组
    """
    if not enable or len(rest_indices) < 1:
        return soc_est
    
    soc_est_corrected = soc_est.copy()
    
    for i in range(len(soc_est)):
        if not valid_rest_mask[i]:
            # 找到最近的静止点
            nearest_rest_idx = rest_indices[np.argmin(np.abs(rest_indices - i))]
            distance = abs(i - nearest_rest_idx)
            
            # 如果距离不远，进行轻微修正
            if distance < max_distance:
                correction = (soc_est[nearest_rest_idx] - soc_ah[nearest_rest_idx]) * np.exp(-distance/50)
                soc_est_corrected[i] = soc_ah[i] + correction * correction_weight
    
    return soc_est_corrected
