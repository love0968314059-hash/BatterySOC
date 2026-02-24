#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池数据处理模块
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from read_excel_direct import read_xlsx_direct

class BatteryDataProcessor:
    """电池数据处理器"""
    
    def __init__(self, data_dir="raw_data"):
        self.data_dir = Path(data_dir)
    
    def load_data_file(self, filepath):
        """加载数据文件"""
        if filepath.suffix.lower() in ['.xlsx', '.xls']:
            return read_xlsx_direct(filepath)
        return None
    
    def identify_columns(self, df):
        """智能识别列名"""
        col_map = {}
        
        # 优先级匹配：先匹配精确的列名（包含单位）
        for col in df.columns:
            col_str = str(col)
            col_lower = col_str.lower().strip()
            
            # 电压 - 优先匹配包含(V)或Voltage的
            if 'voltage(v)' in col_lower or (col_str == 'Voltage(V)'):
                col_map['voltage'] = col
            elif 'voltage' in col_lower and 'voltage' not in col_map:
                col_map['voltage'] = col
            
            # 电流 - 优先匹配包含(A)或Current的
            if 'current(a)' in col_lower or (col_str == 'Current(A)'):
                col_map['current'] = col
            elif 'current' in col_lower and 'current' not in col_map:
                # 排除impedance
                if 'impedance' not in col_lower:
                    col_map['current'] = col
            
            # 温度 - 优先匹配Temperature
            if 'temperature' in col_lower:
                if 'temperature' not in col_map or 'temperature' in col_str:
                    col_map['temperature'] = col
            elif 'temp' in col_lower and 'temperature' not in col_map:
                col_map['temperature'] = col
            
            # 时间 - 优先匹配Test_Time
            if 'test_time' in col_lower or 'test time' in col_lower:
                col_map['time'] = col
            elif 'time' in col_lower and 'step' not in col_lower and 'time' not in col_map:
                col_map['time'] = col
            
            # 容量
            if 'discharge_capacity' in col_lower or ('discharge' in col_lower and 'capacity' in col_lower):
                if 'discharge_capacity' not in col_map:
                    col_map['discharge_capacity'] = col
            elif 'charge_capacity' in col_lower or ('charge' in col_lower and 'capacity' in col_lower):
                if 'charge_capacity' not in col_map:
                    col_map['charge_capacity'] = col
        
        return col_map
    
    def clean_data(self, df, col_map, save_filtered_data=False, output_dir=None):
        """
        清洗数据（仅过滤NaN值，不进行异常值检测）
        
        Args:
            df: 数据框
            col_map: 列映射
            save_filtered_data: 是否保存被过滤的数据（已废弃，保留兼容性）
            output_dir: 输出目录（已废弃，保留兼容性）
        """
        voltage = df[col_map['voltage']].values.astype(float)
        current = df[col_map['current']].values.astype(float)
        temperature = df[col_map['temperature']].values.astype(float) if 'temperature' in col_map else None
        time = df[col_map['time']].values.astype(float) if 'time' in col_map else None
        
        original_length = len(voltage)
        
        # NaN过滤（仅过滤缺失值）
        valid_mask = ~(np.isnan(voltage) | np.isnan(current))
        if temperature is not None:
            valid_mask = valid_mask & ~np.isnan(temperature)
        if time is not None:
            valid_mask = valid_mask & ~np.isnan(time)
        
        # 记录被NaN过滤的数据
        nan_filtered_count = np.sum(~valid_mask)
        
        result = {
            'voltage': voltage[valid_mask],
            'current': current[valid_mask],
            'temperature': temperature[valid_mask] if temperature is not None else None,
            'time': time[valid_mask] if time is not None else None,
            'outlier_mask': np.ones(len(voltage[valid_mask]), dtype=bool),  # 所有数据都保留，用于兼容性
            'original_length': original_length,
            'cleaned_length': np.sum(valid_mask),
            'nan_filtered_count': nan_filtered_count,
            'outlier_filtered_count': 0,  # 不再进行异常值过滤
            'nan_filtered_data': {
                'indices': np.where(~valid_mask)[0].tolist() if nan_filtered_count > 0 else [],
                'reason': 'NaN'
            },
            'outlier_filtered_data': {
                'indices': [],
                'reason': 'None (outlier detection disabled)'
            }
        }
        
        # 如果要求保存被过滤的数据
        if save_filtered_data and output_dir is not None:
            import json
            from pathlib import Path
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filtered_data_summary = {
                'original_length': original_length,
                'cleaned_length': int(np.sum(outlier_mask)),
                'nan_filtered_count': len(nan_filtered_indices),
                'outlier_filtered_count': len(outlier_filtered_indices),
                'nan_filtered_data': nan_filtered_data,
                'outlier_filtered_data': outlier_filtered_data,
                'statistics': {
                    'voltage_mean': float(voltage_mean),
                    'voltage_std': float(voltage_std),
                    'voltage_min': float(voltage_clean.min()),
                    'voltage_max': float(voltage_clean.max()),
                    'current_mean': float(current_mean),
                    'current_std': float(current_std),
                    'current_min': float(current_clean.min()),
                    'current_max': float(current_clean.max())
                }
            }
            
            filtered_data_file = output_path / 'filtered_data_summary.json'
            with open(filtered_data_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data_summary, f, indent=2, ensure_ascii=False)
            
            # 生成可视化图表
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # 配置字体以避免中文乱码
                plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'SimHei', 'STHeiti']
                plt.rcParams['axes.unicode_minus'] = False
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # 子图1: 电压分布（标注异常值）
                ax1 = axes[0, 0]
                ax1.plot(time_clean if time_clean is not None else np.arange(len(voltage_clean)), voltage_clean, 'b-', alpha=0.3, label='Normal Data')
                if len(outlier_filtered_data['indices']) > 0:
                    outlier_times = [time_clean[np.where(indices_clean == idx)[0][0]] for idx in outlier_filtered_indices if idx in indices_clean]
                    outlier_voltages = outlier_filtered_data['voltage']
                    if len(outlier_times) == len(outlier_voltages):
                        ax1.scatter(outlier_times, outlier_voltages, c='r', s=50, marker='x', label=f'Outliers Filtered ({len(outlier_voltages)} points)')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Voltage (V)')
                ax1.set_title('Voltage Data (Outliers Marked)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 子图2: 电流分布（标注异常值）
                ax2 = axes[0, 1]
                ax2.plot(time_clean if time_clean is not None else np.arange(len(current_clean)), current_clean, 'b-', alpha=0.3, label='Normal Data')
                if len(outlier_filtered_data['indices']) > 0:
                    outlier_times = [time_clean[np.where(indices_clean == idx)[0][0]] for idx in outlier_filtered_indices if idx in indices_clean]
                    outlier_currents = outlier_filtered_data['current']
                    if len(outlier_times) == len(outlier_currents):
                        ax2.scatter(outlier_times, outlier_currents, c='r', s=50, marker='x', label=f'Outliers Filtered ({len(outlier_currents)} points)')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Current (A)')
                ax2.set_title('Current Data (Outliers Marked)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 子图3: 电压Z-score分布
                ax3 = axes[1, 0]
                ax3.hist(voltage_zscore, bins=50, alpha=0.7, label='Normal Data')
                if len(outlier_filtered_data['voltage_zscore']) > 0:
                    ax3.axvline(x=3, color='r', linestyle='--', label='3-sigma Threshold')
                    outlier_voltage_zscore = outlier_filtered_data['voltage_zscore']
                    ax3.hist(outlier_voltage_zscore, bins=20, alpha=0.7, color='r', label=f'Outliers ({len(outlier_voltage_zscore)} points)')
                ax3.set_xlabel('Voltage Z-score (sigma)')
                ax3.set_ylabel('Count')
                ax3.set_title('Voltage Z-score Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # 子图4: 电流Z-score分布
                ax4 = axes[1, 1]
                ax4.hist(current_zscore, bins=50, alpha=0.7, label='Normal Data')
                if len(outlier_filtered_data['current_zscore']) > 0:
                    ax4.axvline(x=3, color='r', linestyle='--', label='3-sigma Threshold')
                    outlier_current_zscore = outlier_filtered_data['current_zscore']
                    ax4.hist(outlier_current_zscore, bins=20, alpha=0.7, color='r', label=f'Outliers ({len(outlier_current_zscore)} points)')
                ax4.set_xlabel('Current Z-score (sigma)')
                ax4.set_ylabel('Count')
                ax4.set_title('Current Z-score Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = output_path / 'data_cleaning_analysis.png'
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"    数据清洗分析图已保存: {plot_file}")
            except Exception as e:
                print(f"    ⚠ 生成数据清洗分析图失败: {e}")
        
        return result
    
    def calculate_soc_from_capacity(self, df, col_map, voltage=None, current=None, time=None, 
                                    nominal_capacity=1.1, actual_capacity=None, ocv_soc_table=None):
        """
        从容量和AH积分计算SOC标签
        使用截止电压倒推起始SOC
        
        Args:
            nominal_capacity: 额定容量（Ah）
            actual_capacity: 实际容量（Ah）
            ocv_soc_table: OCV-SOC曲线表
        """
        if voltage is None or current is None or time is None:
            return None
        
        # 使用实际容量
        capacity = actual_capacity if actual_capacity is not None else nominal_capacity
        
        # 计算时间间隔
        dt = np.diff(time)
        dt = np.concatenate([[dt[0] if len(dt) > 0 else 1], dt])
        dt[dt <= 0] = np.median(dt[dt > 0])
        dt[dt > np.median(dt) * 10] = np.median(dt[dt <= np.median(dt) * 10])
        
        # 计算累积电流积分（Ah）
        cumulative_ah = np.cumsum(current * dt / 3600)
        
        # 找到截止电压点
        # LFP电池：上限截止电压约3.6V（100% SOC），下限截止电压约2.0V（0% SOC）
        upper_cutoff_voltage = 3.6  # 充电截止电压
        lower_cutoff_voltage = 2.0  # 放电截止电压
        
        # 找到达到上限截止电压的点（SOC应该接近100%）
        upper_cutoff_mask = voltage >= upper_cutoff_voltage
        upper_cutoff_indices = np.where(upper_cutoff_mask)[0]
        
        # 找到达到下限截止电压的点（SOC应该接近0%）
        lower_cutoff_mask = voltage <= lower_cutoff_voltage
        lower_cutoff_indices = np.where(lower_cutoff_mask)[0]
        
        # 找到实际的最大和最小电压点（可能更准确）
        max_voltage_idx = np.argmax(voltage)
        min_voltage_idx = np.argmin(voltage)
        max_voltage = voltage[max_voltage_idx]
        min_voltage = voltage[min_voltage_idx]
        
        # 使用AH积分计算SOC变化
        # SOC变化 = 累积AH积分 / 容量 * 100%
        # 注意：cumulative_ah从0开始，所以soc_change也从0开始
        soc_change = cumulative_ah / capacity * 100
        
        # 倒推起始SOC
        # 方法1：优先找到充电结束后的静置点（真正的100% SOC点）
        # 在电压>=3.5V的区域，找到静置点（电流<0.05A）
        high_voltage_mask = voltage >= 3.5
        rest_mask = np.abs(current) < 0.05
        high_voltage_rest_mask = high_voltage_mask & rest_mask
        high_voltage_rest_indices = np.where(high_voltage_rest_mask)[0]
        
        if len(high_voltage_rest_indices) > 0:
            # 找到最后一个高电压静置点（最后一次充电结束后的100% SOC点）
            # 或者找到电压最高的静置点
            last_high_voltage_rest_idx = high_voltage_rest_indices[-1]
            # 找到电压最高的静置点
            rest_voltages = voltage[high_voltage_rest_indices]
            max_rest_voltage_idx_in_rest = np.argmax(rest_voltages)
            best_rest_idx = high_voltage_rest_indices[max_rest_voltage_idx_in_rest]
            
            # 使用电压最高的静置点（更可能是100% SOC）
            # 此时SOC应该是100%
            soc_at_rest = 100.0
            # 倒推起始SOC
            initial_soc = soc_at_rest - soc_change[best_rest_idx]
            print(f"  使用充电结束静置点({voltage[best_rest_idx]:.3f}V, 索引{best_rest_idx})倒推起始SOC: {initial_soc:.2f}%")
        # 方法2：如果最大电压点达到上限截止电压，使用它来倒推
        elif max_voltage >= upper_cutoff_voltage:
            # 最大电压点应该对应100% SOC
            soc_at_max = 100.0
            # 倒推起始SOC
            initial_soc = soc_at_max - soc_change[max_voltage_idx]
            print(f"  使用最大电压点({max_voltage:.3f}V, 索引{max_voltage_idx})倒推起始SOC: {initial_soc:.2f}%")
        # 方法2：如果有上限截止电压点，使用第一个来倒推
        elif len(upper_cutoff_indices) > 0:
            # 找到第一个达到上限截止电压的点
            first_upper_idx = upper_cutoff_indices[0]
            # 此时SOC应该是100%
            soc_at_upper = 100.0
            # 倒推起始SOC
            initial_soc = soc_at_upper - soc_change[first_upper_idx]
            print(f"  使用上限截止电压({upper_cutoff_voltage}V, 索引{first_upper_idx})倒推起始SOC: {initial_soc:.2f}%")
        # 方法3：如果有下限截止电压点，使用最小电压点来倒推
        elif min_voltage <= lower_cutoff_voltage:
            # 最小电压点应该对应0% SOC
            soc_at_min = 0.0
            # 倒推起始SOC
            initial_soc = soc_at_min - soc_change[min_voltage_idx]
            print(f"  使用最小电压点({min_voltage:.3f}V, 索引{min_voltage_idx})倒推起始SOC: {initial_soc:.2f}%")
        # 方法4：如果有下限截止电压点，使用第一个来倒推
        elif len(lower_cutoff_indices) > 0:
            # 找到第一个达到下限截止电压的点
            first_lower_idx = lower_cutoff_indices[0]
            # 此时SOC应该是0%
            soc_at_lower = 0.0
            # 倒推起始SOC
            initial_soc = soc_at_lower - soc_change[first_lower_idx]
            print(f"  使用下限截止电压({lower_cutoff_voltage}V, 索引{first_lower_idx})倒推起始SOC: {initial_soc:.2f}%")
        # 方法3：使用OCV曲线估算起始SOC
        elif ocv_soc_table is not None and len(ocv_soc_table) > 0:
            # 找到初始静置点
            rest_mask = np.abs(current) < 0.05
            initial_window_size = min(2000, len(rest_mask))
            initial_rest_mask = rest_mask[:initial_window_size]
            
            initial_rest_indices = []
            if np.any(initial_rest_mask):
                diff = np.diff(np.concatenate([[False], initial_rest_mask, [False]]).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                if len(starts) > 0:
                    lengths = ends - starts
                    longest_idx = np.argmax(lengths)
                    longest_start = starts[longest_idx]
                    longest_end = ends[longest_idx]
                    initial_rest_indices = np.arange(longest_start, longest_end)
            
            if len(initial_rest_indices) > 10:
                initial_ocv = np.median(voltage[initial_rest_indices])
                socs = ocv_soc_table[:, 0]
                ocvs = ocv_soc_table[:, 1]
                
                if initial_ocv >= ocvs.min() and initial_ocv <= ocvs.max():
                    initial_soc = np.interp(initial_ocv, ocvs, socs)
                elif initial_ocv < ocvs.min():
                    # 低于OCV曲线下限，使用0% SOC
                    initial_soc = 0.0
                else:
                    # 高于OCV曲线上限，使用100% SOC
                    initial_soc = 100.0
                print(f"  使用OCV曲线估算起始SOC: {initial_soc:.2f}% (OCV={initial_ocv:.3f}V)")
            else:
                # 如果没有静置点，使用第一个点的电压
                initial_ocv = voltage[0]
                socs = ocv_soc_table[:, 0]
                ocvs = ocv_soc_table[:, 1]
                
                if initial_ocv >= ocvs.min() and initial_ocv <= ocvs.max():
                    initial_soc = np.interp(initial_ocv, ocvs, socs)
                elif initial_ocv < ocvs.min():
                    initial_soc = 0.0
                else:
                    initial_soc = 100.0
                print(f"  使用OCV曲线估算起始SOC: {initial_soc:.2f}% (OCV={initial_ocv:.3f}V)")
        else:
            # 默认起始SOC为50%
            initial_soc = 50.0
            print(f"  使用默认起始SOC: {initial_soc:.2f}%")
        
        # 计算SOC = 起始SOC + SOC变化
        soc = initial_soc + soc_change
        
        # 检查SOC范围，确保符合AH积分与容量的关系
        soc_min = soc.min()
        soc_max = soc.max()
        
        # 如果SOC超出合理范围，调整起始SOC
        # 优先保证SOC在0-100%范围内
        if soc_min < 0:
            # 如果最小SOC小于0，调整起始SOC使得最小SOC为0
            initial_soc_adjusted = initial_soc - soc_min
            soc = initial_soc_adjusted + soc_change
            print(f"  调整起始SOC: {initial_soc:.2f}% -> {initial_soc_adjusted:.2f}% (使最小SOC>=0%)")
            initial_soc = initial_soc_adjusted
        
        if soc_max > 100:
            # 如果最大SOC大于100，调整起始SOC使得最大SOC为100
            initial_soc_adjusted = initial_soc - (soc_max - 100)
            soc = initial_soc_adjusted + soc_change
            print(f"  调整起始SOC: {initial_soc:.2f}% -> {initial_soc_adjusted:.2f}% (使最大SOC<=100%)")
            initial_soc = initial_soc_adjusted
        
        # 验证SOC标签的合理性
        # 检查：当SOC达到100%时，不应该继续充电（电流应该为0或负）
        # 但需要考虑：如果在充电结束静置点之前达到100%，这些充电点是合理的（充电过程中）
        soc_at_100_mask = soc >= 99.9
        soc_at_0_mask = soc <= 0.1
        
        # 找到充电结束静置点的索引（如果使用了）
        rest_point_idx = None
        if len(high_voltage_rest_indices) > 0:
            rest_voltages = voltage[high_voltage_rest_indices]
            max_rest_voltage_idx_in_rest = np.argmax(rest_voltages)
            rest_point_idx = high_voltage_rest_indices[max_rest_voltage_idx_in_rest]
        
        if np.any(soc_at_100_mask):
            # 只检查充电结束静置点之后的点（真正的100% SOC之后不应该再充电）
            if rest_point_idx is not None:
                soc_at_100_after_rest = soc_at_100_mask & (np.arange(len(soc)) > rest_point_idx)
                charging_at_100_after_rest = np.sum(current[soc_at_100_after_rest] > 0.01)
                if charging_at_100_after_rest > 0:
                    print(f"  警告: 充电结束后SOC达到100%时仍有{charging_at_100_after_rest}个点在充电，可能起始SOC计算有误")
            else:
                charging_at_100 = np.sum(current[soc_at_100_mask] > 0.01)
                if charging_at_100 > 0:
                    print(f"  警告: SOC达到100%时仍有{charging_at_100}个点在充电，可能起始SOC计算有误")
        
        if np.any(soc_at_0_mask):
            discharging_at_0 = np.sum(current[soc_at_0_mask] < -0.01)
            if discharging_at_0 > 0:
                print(f"  警告: SOC达到0%时仍有{discharging_at_0}个点在放电，可能起始SOC计算有误")
        
        # 验证SOC标签是否符合AH积分与容量的关系
        # SOC变化应该等于累积AH积分变化 / 容量 * 100%
        soc_change_range = soc.max() - soc.min()
        ah_change_range = cumulative_ah.max() - cumulative_ah.min()
        expected_soc_change = ah_change_range / capacity * 100
        
        if abs(soc_change_range - expected_soc_change) > 1.0:  # 允许1%的误差
            print(f"  警告: SOC变化({soc_change_range:.2f}%)与AH积分变化({expected_soc_change:.2f}%)不一致，差异{abs(soc_change_range - expected_soc_change):.2f}%")
        
        # 最终截断到0-100%
        soc = np.clip(soc, 0, 100)
        
        print(f"  最终SOC范围: {soc.min():.2f}% - {soc.max():.2f}%, 起始SOC: {initial_soc:.2f}%")
        
        return soc
    
    def process_file(self, filepath):
        """处理单个文件"""
        print(f"\n处理文件: {filepath.name}")
        
        # 加载数据
        df = self.load_data_file(filepath)
        if df is None:
            return None
        
        print(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")
        
        # 识别列
        col_map = self.identify_columns(df)
        print(f"  识别列: {col_map}")
        
        if 'voltage' not in col_map or 'current' not in col_map:
            print(f"  跳过: 缺少必需字段")
            return None
        
        # 先提取原始数据用于SOC计算
        voltage_raw = df[col_map['voltage']].values.astype(float)
        current_raw = df[col_map['current']].values.astype(float)
        time_raw = df[col_map['time']].values.astype(float) if 'time' in col_map else None
        
        # 数据清洗（已移除异常值检测，不再需要保存分析结果）
        cleaned = self.clean_data(df, col_map, save_filtered_data=False, output_dir=None)
        print(f"  清洗后: {cleaned['cleaned_length']} 行 (保留 {cleaned['cleaned_length']/cleaned['original_length']*100:.1f}%)")
        
        # 计算SOC（从容量和AH积分）- 使用原始数据计算，然后应用过滤
        soc_true = self.calculate_soc_from_capacity(df, col_map, voltage_raw, current_raw, time_raw, nominal_capacity=1.1)
        if soc_true is not None:
            # 先应用NaN过滤
            valid_mask_nan = ~(np.isnan(df[col_map['voltage']].values) | np.isnan(df[col_map['current']].values))
            if 'temperature' in col_map:
                valid_mask_nan = valid_mask_nan & ~np.isnan(df[col_map['temperature']].values)
            soc_true_filtered = soc_true[valid_mask_nan]
            
            # 应用异常值过滤（使用cleaned的outlier_mask）
            if len(soc_true_filtered) == len(cleaned['outlier_mask']):
                soc_true = soc_true_filtered[cleaned['outlier_mask']]
            else:
                # 长度不匹配，尝试重新对齐
                print(f"  警告: SOC长度不匹配 ({len(soc_true_filtered)} vs {len(cleaned['outlier_mask'])}), 跳过SOC")
                soc_true = None
        
        # 计算最大容量
        discharge_capacity_max = None
        if 'discharge_capacity' in col_map:
            discharge_capacity_max = df[col_map['discharge_capacity']].max()
        
        # 获取标签计算使用的容量和初始SOC
        initial_soc_used = None
        capacity_used = None
        if soc_true is not None and len(soc_true) > 0:
            initial_soc_used = soc_true[0]
            # 从calculate_soc_from_capacity的返回值中获取使用的容量
            # 需要重新调用一次来获取容量信息（或者从soc_true反推）
            # 暂时使用nominal_capacity（因为process_file调用时只传了nominal_capacity=1.1）
            capacity_used = 1.1  # process_file中使用的容量
        
        result = {
            'filename': filepath.name,
            'voltage': cleaned['voltage'],
            'current': cleaned['current'],
            'temperature': cleaned['temperature'],
            'time': cleaned['time'],
            'soc_true': soc_true,
            'col_map': col_map,
            'discharge_capacity_max': discharge_capacity_max,
            'initial_soc': initial_soc_used,  # 添加初始SOC
            'nominal_capacity': capacity_used  # 添加使用的容量
        }
        
        return result
    
    def process_all_files(self, max_files=None):
        """处理所有数据文件"""
        # 查找所有Excel文件
        data_files = []
        for ext in ['*.xlsx', '*.xls']:
            data_files.extend(self.data_dir.rglob(ext))
        
        if max_files:
            data_files = data_files[:max_files]
        
        processed_data = []
        for filepath in data_files:
            result = self.process_file(filepath)
            if result:
                result['filepath'] = filepath
                processed_data.append(result)
        
        return processed_data
    
    def extract_temperature_from_path(self, filepath):
        """从文件路径提取温度信息"""
        import re
        path_str = str(filepath)
        
        # 1. 优先处理N前缀表示负数温度的情况（如N10表示-10°C）
        # 匹配模式：N后跟数字（如N10, N0等）
        n_pattern = r'[_-]N(\d+)'
        n_match = re.search(n_pattern, path_str)
        if n_match:
            temp = -int(n_match.group(1))  # N10 -> -10
            if -50 <= temp <= 50:
                return temp
        
        # 2. 处理OCV格式：OCV0, OCV-10, OCV10等
        # OCV0特殊处理（0度）
        if 'OCV0' in path_str and ('OCV-0' not in path_str and 'OCV0-' not in path_str):
            return 0
        
        # OCV后跟数字的模式：OCV-10, OCV10等
        ocv_pattern = r'OCV(-?\d+)'
        ocv_match = re.search(ocv_pattern, path_str)
        if ocv_match:
            temp_str = ocv_match.group(1)
            if temp_str.startswith('-'):
                temp = int(temp_str)
            else:
                temp = int(temp_str)
            if -50 <= temp <= 50:
                return temp
        
        # 3. 处理标准温度格式：-XX 或 XX（如-10, 0, 25等）
        # 匹配模式：-XX 或 _XX（在DST-US06-FUDS等上下文中）
        temp_match = re.search(r'(?:DST-US06-FUDS|FUDS)[_-](-?\d+)', path_str)
        if temp_match:
            temp_str = temp_match.group(1)
            if temp_str.startswith('-'):
                temp = int(temp_str)
            else:
                temp = int(temp_str)
                # 如果数字太大（可能是日期），跳过
                if temp > 50:
                    return None
            if -50 <= temp <= 50:
                return temp
        
        # 4. 通用模式：查找所有-XX或_XX模式，选择合理的温度值
        all_matches = re.findall(r'[_-](-?\d+)', path_str)
        for temp_str in all_matches:
            if temp_str.startswith('-'):
                temp = int(temp_str)
            else:
                temp = int(temp_str)
                # 跳过过大的数字（可能是日期或ID）
                if temp > 50:
                    continue
            if -50 <= temp <= 50:
                return temp
        
        return None
