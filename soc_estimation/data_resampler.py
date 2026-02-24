#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据重采样模块
将不同时间间隔的数据处理为等间隔数据
- 时间间隔异常大时：电流填充为0，电压进行插值
- 时间间隔正常时：保持原数据或插值
"""

import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class DataResampler:
    """数据重采样器"""
    
    def __init__(self, target_dt=1.0, max_dt_ratio=10.0, verbose=True):
        """
        初始化重采样器
        
        Args:
            target_dt: 目标采样间隔（秒），默认1秒
            max_dt_ratio: 最大允许的时间间隔比例，超过此比例视为异常
            verbose: 是否打印详细信息
        """
        self.target_dt = target_dt
        self.max_dt_ratio = max_dt_ratio
        self.verbose = verbose
        
        # 统计信息
        self.stats = {
            'original_length': 0,
            'resampled_length': 0,
            'abnormal_gaps': 0,
            'interpolated_points': 0
        }
    
    def analyze_time_intervals(self, time):
        """
        分析时间间隔
        
        Args:
            time: 时间序列 (秒)
            
        Returns:
            dict: 时间间隔统计信息
        """
        time = np.asarray(time)
        dt = np.diff(time)
        
        # 基本统计
        dt_median = np.median(dt)
        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
        dt_min = np.min(dt)
        dt_max = np.max(dt)
        
        # 异常间隔检测（超过中位数的max_dt_ratio倍）
        abnormal_threshold = dt_median * self.max_dt_ratio
        abnormal_mask = dt > abnormal_threshold
        n_abnormal = np.sum(abnormal_mask)
        
        # 负间隔或零间隔检测
        invalid_mask = dt <= 0
        n_invalid = np.sum(invalid_mask)
        
        stats = {
            'dt_median': dt_median,
            'dt_mean': dt_mean,
            'dt_std': dt_std,
            'dt_min': dt_min,
            'dt_max': dt_max,
            'n_points': len(time),
            'n_intervals': len(dt),
            'abnormal_threshold': abnormal_threshold,
            'n_abnormal': n_abnormal,
            'n_invalid': n_invalid,
            'abnormal_indices': np.where(abnormal_mask)[0],
            'invalid_indices': np.where(invalid_mask)[0]
        }
        
        if self.verbose:
            print(f"  时间间隔分析:")
            print(f"    - 数据点数: {len(time)}")
            print(f"    - 间隔中位数: {dt_median:.3f}秒")
            print(f"    - 间隔均值: {dt_mean:.3f}秒")
            print(f"    - 间隔范围: {dt_min:.3f} - {dt_max:.3f}秒")
            print(f"    - 异常阈值: {abnormal_threshold:.3f}秒")
            print(f"    - 异常间隔数: {n_abnormal}")
            print(f"    - 无效间隔数: {n_invalid}")
        
        return stats
    
    def resample(self, time, voltage, current, temperature=None, 
                 discharge_capacity=None, charge_capacity=None):
        """
        重采样数据为等间隔
        
        对于异常大的时间间隔：
        - 电流填充为0（假设静置）
        - 电压线性插值
        - 温度保持不变（或插值）
        - 容量保持不变
        
        Args:
            time: 时间序列 (秒)
            voltage: 电压序列 (V)
            current: 电流序列 (A)
            temperature: 温度序列 (°C)，可选
            discharge_capacity: 放电容量序列 (Ah)，可选
            charge_capacity: 充电容量序列 (Ah)，可选
            
        Returns:
            dict: 重采样后的数据
        """
        time = np.asarray(time, dtype=float)
        voltage = np.asarray(voltage, dtype=float)
        current = np.asarray(current, dtype=float)
        
        # 分析时间间隔
        interval_stats = self.analyze_time_intervals(time)
        
        # 确定采样间隔
        # 如果数据本身的采样间隔接近目标，使用数据的中位数间隔
        actual_dt = interval_stats['dt_median']
        if abs(actual_dt - self.target_dt) < 0.5:
            # 使用实际采样间隔
            sample_dt = actual_dt
        else:
            # 使用目标采样间隔
            sample_dt = self.target_dt
        
        if self.verbose:
            print(f"  采样间隔: {sample_dt:.3f}秒")
        
        # 创建等间隔时间序列
        t_start = time[0]
        t_end = time[-1]
        new_time = np.arange(t_start, t_end + sample_dt, sample_dt)
        
        # 初始化输出数组
        n_new = len(new_time)
        new_voltage = np.zeros(n_new)
        new_current = np.zeros(n_new)
        new_temperature = np.zeros(n_new) if temperature is not None else None
        new_discharge_capacity = np.zeros(n_new) if discharge_capacity is not None else None
        new_charge_capacity = np.zeros(n_new) if charge_capacity is not None else None
        
        # 创建插值函数
        # 电压使用线性插值
        voltage_interp = interp1d(time, voltage, kind='linear', 
                                   fill_value='extrapolate', bounds_error=False)
        
        # 电流使用最近邻插值（保持阶跃特性）
        current_interp = interp1d(time, current, kind='nearest', 
                                   fill_value='extrapolate', bounds_error=False)
        
        if temperature is not None:
            temperature = np.asarray(temperature, dtype=float)
            temp_interp = interp1d(time, temperature, kind='linear', 
                                    fill_value='extrapolate', bounds_error=False)
        
        if discharge_capacity is not None:
            discharge_capacity = np.asarray(discharge_capacity, dtype=float)
            discharge_interp = interp1d(time, discharge_capacity, kind='linear', 
                                         fill_value='extrapolate', bounds_error=False)
        
        if charge_capacity is not None:
            charge_capacity = np.asarray(charge_capacity, dtype=float)
            charge_interp = interp1d(time, charge_capacity, kind='linear', 
                                      fill_value='extrapolate', bounds_error=False)
        
        # 逐点处理
        abnormal_count = 0
        interpolated_count = 0
        
        for i, t in enumerate(new_time):
            # 找到原始数据中最近的两个点
            idx = np.searchsorted(time, t)
            
            if idx == 0:
                # 在第一个点之前
                new_voltage[i] = voltage[0]
                new_current[i] = current[0]
                if new_temperature is not None:
                    new_temperature[i] = temperature[0]
                if new_discharge_capacity is not None:
                    new_discharge_capacity[i] = discharge_capacity[0]
                if new_charge_capacity is not None:
                    new_charge_capacity[i] = charge_capacity[0]
            elif idx >= len(time):
                # 在最后一个点之后
                new_voltage[i] = voltage[-1]
                new_current[i] = current[-1]
                if new_temperature is not None:
                    new_temperature[i] = temperature[-1]
                if new_discharge_capacity is not None:
                    new_discharge_capacity[i] = discharge_capacity[-1]
                if new_charge_capacity is not None:
                    new_charge_capacity[i] = charge_capacity[-1]
            else:
                # 在两个点之间
                t_prev = time[idx - 1]
                t_next = time[idx]
                dt = t_next - t_prev
                
                # 判断是否是异常间隔
                if dt > interval_stats['abnormal_threshold']:
                    # 异常间隔：电流填0，电压插值
                    abnormal_count += 1
                    new_voltage[i] = float(voltage_interp(t))
                    new_current[i] = 0.0  # 电流填充为0（假设静置）
                    if new_temperature is not None:
                        # 温度保持前一个点的值（静置时温度不变）
                        new_temperature[i] = temperature[idx - 1]
                    if new_discharge_capacity is not None:
                        # 容量保持不变
                        new_discharge_capacity[i] = discharge_capacity[idx - 1]
                    if new_charge_capacity is not None:
                        new_charge_capacity[i] = charge_capacity[idx - 1]
                else:
                    # 正常间隔：插值
                    interpolated_count += 1
                    new_voltage[i] = float(voltage_interp(t))
                    new_current[i] = float(current_interp(t))
                    if new_temperature is not None:
                        new_temperature[i] = float(temp_interp(t))
                    if new_discharge_capacity is not None:
                        new_discharge_capacity[i] = float(discharge_interp(t))
                    if new_charge_capacity is not None:
                        new_charge_capacity[i] = float(charge_interp(t))
        
        # 更新统计信息
        self.stats = {
            'original_length': len(time),
            'resampled_length': n_new,
            'sample_dt': sample_dt,
            'abnormal_gaps_filled': abnormal_count,
            'interpolated_points': interpolated_count,
            'interval_stats': interval_stats
        }
        
        if self.verbose:
            print(f"  重采样结果:")
            print(f"    - 原始数据点: {len(time)}")
            print(f"    - 重采样后: {n_new}")
            print(f"    - 异常间隔填充: {abnormal_count}")
            print(f"    - 插值点数: {interpolated_count}")
        
        result = {
            'time': new_time,
            'voltage': new_voltage,
            'current': new_current,
            'temperature': new_temperature,
            'discharge_capacity': new_discharge_capacity,
            'charge_capacity': new_charge_capacity,
            'stats': self.stats
        }
        
        return result
    
    def detect_and_fill_gaps(self, time, voltage, current, temperature=None):
        """
        检测并填充时间间隔中的空隙
        只填充异常大的间隔，不改变正常间隔
        
        Args:
            time: 时间序列 (秒)
            voltage: 电压序列 (V)
            current: 电流序列 (A)
            temperature: 温度序列 (°C)，可选
            
        Returns:
            dict: 填充空隙后的数据
        """
        time = np.asarray(time, dtype=float)
        voltage = np.asarray(voltage, dtype=float)
        current = np.asarray(current, dtype=float)
        
        # 分析时间间隔
        interval_stats = self.analyze_time_intervals(time)
        
        # 计算时间差
        dt = np.diff(time)
        abnormal_threshold = interval_stats['abnormal_threshold']
        
        # 找到异常间隔的位置
        abnormal_indices = np.where(dt > abnormal_threshold)[0]
        
        if len(abnormal_indices) == 0:
            if self.verbose:
                print(f"  未检测到异常时间间隔，返回原始数据")
            return {
                'time': time,
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'filled_gaps': 0
            }
        
        # 填充每个异常间隔
        new_time_list = []
        new_voltage_list = []
        new_current_list = []
        new_temperature_list = [] if temperature is not None else None
        
        filled_gaps = 0
        last_idx = 0
        
        for gap_idx in abnormal_indices:
            # 添加间隔前的数据
            new_time_list.append(time[last_idx:gap_idx + 1])
            new_voltage_list.append(voltage[last_idx:gap_idx + 1])
            new_current_list.append(current[last_idx:gap_idx + 1])
            if temperature is not None:
                new_temperature_list.append(temperature[last_idx:gap_idx + 1])
            
            # 填充异常间隔
            t_start = time[gap_idx]
            t_end = time[gap_idx + 1]
            gap_duration = t_end - t_start
            
            # 使用中位数间隔填充
            fill_dt = interval_stats['dt_median']
            n_fill = int(gap_duration / fill_dt) - 1
            
            if n_fill > 0:
                # 生成填充时间点
                fill_time = t_start + fill_dt * np.arange(1, n_fill + 1)
                
                # 电压线性插值
                v_start = voltage[gap_idx]
                v_end = voltage[gap_idx + 1]
                fill_voltage = np.linspace(v_start, v_end, n_fill + 2)[1:-1]
                
                # 电流填充为0
                fill_current = np.zeros(n_fill)
                
                # 温度保持不变
                if temperature is not None:
                    fill_temperature = np.full(n_fill, temperature[gap_idx])
                
                new_time_list.append(fill_time)
                new_voltage_list.append(fill_voltage)
                new_current_list.append(fill_current)
                if temperature is not None:
                    new_temperature_list.append(fill_temperature)
                
                filled_gaps += n_fill
            
            last_idx = gap_idx + 1
        
        # 添加最后一段数据
        new_time_list.append(time[last_idx:])
        new_voltage_list.append(voltage[last_idx:])
        new_current_list.append(current[last_idx:])
        if temperature is not None:
            new_temperature_list.append(temperature[last_idx:])
        
        # 合并数据
        new_time = np.concatenate(new_time_list)
        new_voltage = np.concatenate(new_voltage_list)
        new_current = np.concatenate(new_current_list)
        new_temperature = np.concatenate(new_temperature_list) if temperature is not None else None
        
        if self.verbose:
            print(f"  间隔填充结果:")
            print(f"    - 原始数据点: {len(time)}")
            print(f"    - 填充后: {len(new_time)}")
            print(f"    - 填充点数: {filled_gaps}")
        
        return {
            'time': new_time,
            'voltage': new_voltage,
            'current': new_current,
            'temperature': new_temperature,
            'filled_gaps': filled_gaps
        }


def calculate_soc_from_resampled_data(time, current, voltage, nominal_capacity, 
                                       initial_soc=None, ocv_soc_table=None):
    """
    从重采样后的数据计算SOC标签
    
    Args:
        time: 时间序列 (秒)
        current: 电流序列 (A)，正值为充电
        voltage: 电压序列 (V)
        nominal_capacity: 标称容量 (Ah)
        initial_soc: 初始SOC (%)，如果为None则从OCV估算
        ocv_soc_table: OCV-SOC查找表
        
    Returns:
        np.array: SOC序列 (%)
    """
    time = np.asarray(time)
    current = np.asarray(current)
    voltage = np.asarray(voltage)
    
    # 计算时间差
    dt = np.diff(time)
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0], dt])
    
    # 计算累积AH
    cumulative_ah = np.cumsum(current * dt / 3600)
    
    # 确定初始SOC
    if initial_soc is None:
        if ocv_soc_table is not None:
            # 从初始电压估算初始SOC
            from scipy.interpolate import interp1d
            ocv_soc_table = np.array(ocv_soc_table)
            ocv_to_soc = interp1d(ocv_soc_table[:, 1], ocv_soc_table[:, 0],
                                   kind='linear', fill_value='extrapolate')
            initial_soc = float(ocv_to_soc(voltage[0]))
            initial_soc = np.clip(initial_soc, 0, 100)
        else:
            # 默认从50%开始
            initial_soc = 50.0
    
    # 计算SOC变化
    soc_change = cumulative_ah / nominal_capacity * 100
    
    # 计算SOC
    soc = initial_soc + soc_change
    
    # 调整使SOC在0-100%范围内
    soc_min = soc.min()
    soc_max = soc.max()
    
    if soc_min < 0:
        initial_soc_adjusted = initial_soc - soc_min
        soc = initial_soc_adjusted + soc_change
    
    if soc.max() > 100:
        initial_soc_adjusted = initial_soc - (soc.max() - 100)
        soc = initial_soc_adjusted + soc_change
    
    soc = np.clip(soc, 0, 100)
    
    return soc


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    print("测试数据重采样模块")
    print("=" * 60)
    
    # 生成带有异常间隔的测试数据
    np.random.seed(42)
    
    # 正常采样间隔1秒
    t1 = np.arange(0, 100, 1)
    # 异常间隔（50秒间隔）
    t2 = np.arange(150, 250, 1)
    # 再次正常采样
    t3 = np.arange(260, 360, 1)
    
    time = np.concatenate([t1, t2, t3])
    
    # 生成电压（随时间下降）
    voltage = 3.6 - 0.003 * time + 0.01 * np.random.randn(len(time))
    
    # 生成电流（正常时有电流，间隔时无电流）
    current = -0.5 + 0.05 * np.random.randn(len(time))
    
    print(f"原始数据点数: {len(time)}")
    
    # 创建重采样器并重采样
    resampler = DataResampler(target_dt=1.0, max_dt_ratio=10.0)
    result = resampler.resample(time, voltage, current)
    
    print(f"\n重采样后数据点数: {len(result['time'])}")
    print(f"统计信息: {result['stats']}")
