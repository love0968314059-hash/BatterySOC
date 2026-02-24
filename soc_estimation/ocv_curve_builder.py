#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCV-SOC曲线构建模块
从OCV测试数据中构建准确的SOC-OCV曲线
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

class OCVCurveBuilder:
    """OCV-SOC曲线构建器"""
    
    def __init__(self, ocv_data_dir="raw_data"):
        self.ocv_data_dir = Path(ocv_data_dir)
        self.ocv_soc_table = None
    
    def extract_temperature_from_name(self, name):
        """从名称中提取温度"""
        import re
        # 提取数字部分（可能是温度或SOC）
        numbers = re.findall(r'-?\d+', name)
        if numbers:
            temp = int(numbers[0])
            # 如果是负数或小于50，可能是温度
            if temp < 0 or (temp >= -50 and temp <= 50):
                return temp
        return None
    
    def load_ocv_data(self, target_temperature=None, use_test_file=True):
        """
        加载OCV测试数据
        Args:
            target_temperature: 目标温度，如果指定则只加载该温度的数据
            use_test_file: 是否使用OCV测试文件（更准确）
        """
        # 优先使用处理好的OCV曲线（两支电池的平均值）
        if target_temperature is not None:
            # 使用绝对路径，基于脚本位置
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            ocv_curve_file = project_root / f"soc_results/ocv_by_temperature/{target_temperature}C/ocv_soc_curve.csv"
            capacity_info_file = project_root / f"soc_results/ocv_by_temperature/{target_temperature}C/capacity_info.json"
            
            if ocv_curve_file.exists() and capacity_info_file.exists():
                try:
                    # 加载处理好的OCV曲线
                    ocv_df = pd.read_csv(ocv_curve_file)
                    self.ocv_soc_table = ocv_df[['soc', 'ocv']].values
                    
                    # 加载容量信息
                    import json
                    with open(capacity_info_file, 'r') as f:
                        capacity_info = json.load(f)
                    
                    self.nominal_capacity = capacity_info['nominal_capacity_ah']
                    self.actual_discharge_capacity = capacity_info['actual_discharge_capacity_ah']
                    self.actual_charge_capacity = capacity_info['actual_charge_capacity_ah']
                    
                    print(f"\n✓ 加载处理好的{target_temperature}°C OCV曲线（两支电池平均值）")
                    print(f"  数据点数: {len(self.ocv_soc_table)}")
                    print(f"  额定容量: {self.nominal_capacity:.4f} Ah")
                    print(f"  平均实际放电容量: {self.actual_discharge_capacity:.4f} Ah")
                    print(f"  平均实际充电容量: {self.actual_charge_capacity:.4f} Ah")
                    
                    return True
                except Exception as e:
                    print(f"加载处理好的OCV曲线失败: {e}")
                    print("回退到从OCV测试文件提取...")
        
        # 回退到从OCV测试文件提取（如果处理好的文件不存在）
        if use_test_file and target_temperature is not None:
            from soc_estimation.extract_ocv_proper import extract_ocv_from_test_file
            
            # 查找对应温度的OCV测试文件
            ocv_folder_name = f"OCV{target_temperature}" if target_temperature >= 0 else f"OCV{target_temperature}"
            ocv_folder = self.ocv_data_dir / ocv_folder_name
            if not ocv_folder.exists():
                # 尝试其他格式
                ocv_folders = [d for d in self.ocv_data_dir.iterdir() 
                              if d.is_dir() and (f"OCV{target_temperature}" in d.name or f"OCV-{abs(target_temperature)}" in d.name)]
                if ocv_folders:
                    ocv_folder = ocv_folders[0]
            
            if ocv_folder.exists():
                # 查找OCV测试文件
                ocv_files = list(ocv_folder.glob("A1-007-*.xlsx")) + list(ocv_folder.glob("A1-008-*.xlsx"))
                if ocv_files:
                    print(f"\n从OCV测试文件提取OCV曲线: {ocv_files[0].name}")
                    try:
                        ocv_curve, nominal_capacity, actual_discharge_capacity, actual_charge_capacity, discharge_points, charge_points = extract_ocv_from_test_file(ocv_files[0])
                        
                        # 转换为numpy数组格式
                        if ocv_curve:
                            soc_values = [p['soc'] for p in ocv_curve]
                            ocv_values = [p['ocv'] for p in ocv_curve]
                            self.ocv_soc_table = np.array([[s, v] for s, v in zip(soc_values, ocv_values)])
                            
                            print(f"成功从OCV测试文件提取: {len(self.ocv_soc_table)} 个数据点")
                            print(f"额定容量: {nominal_capacity:.4f} Ah")
                            print(f"实际放电容量: {actual_discharge_capacity:.4f} Ah")
                            print(f"实际充电容量: {actual_charge_capacity:.4f} Ah")
                            
                            # 保存容量信息
                            self.nominal_capacity = nominal_capacity
                            self.actual_discharge_capacity = actual_discharge_capacity
                            self.actual_charge_capacity = actual_charge_capacity
                            
                            return True
                    except Exception as e:
                        print(f"从OCV测试文件提取失败: {e}")
                        print("回退到从文件夹提取...")
        
        # 回退到原来的方法：从文件夹提取
        ocv_folders = sorted([d for d in self.ocv_data_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('OCV')])
        
        print(f"\n找到 {len(ocv_folders)} 个OCV测试数据文件夹")
        
        if target_temperature is not None:
            print(f"目标温度: {target_temperature}°C")
        
        ocv_data_points = []
        
        for ocv_folder in ocv_folders:
            # 从文件夹名提取温度（OCV后面是温度）
            folder_name = ocv_folder.name
            ocv_temperature = self.extract_temperature_from_name(folder_name)
            
            # 如果指定了目标温度，只处理匹配的
            if target_temperature is not None and ocv_temperature != target_temperature:
                continue
            
            # 如果指定了目标温度，只处理匹配的
            if target_temperature is not None and ocv_temperature != target_temperature:
                continue
            
            if ocv_temperature is not None:
                print(f"  处理OCV数据 (温度={ocv_temperature}°C): {ocv_folder.name}")
            else:
                print(f"  处理OCV数据: {ocv_folder.name} (无法确定温度)")
            
            # 读取该文件夹下的Excel文件
            excel_files = list(ocv_folder.glob("*.xlsx")) + list(ocv_folder.glob("*.xls"))
            excel_files = [f for f in excel_files if not f.name.startswith('~$')]
            
            for excel_file in excel_files:
                if target_temperature is None or ocv_temperature == target_temperature:
                    print(f"  读取: {ocv_folder.name}/{excel_file.name}")
                
                try:
                    if excel_file.suffix == '.xlsx':
                        df = read_xlsx_direct(excel_file)
                    else:
                        # xls文件需要xlrd
                        try:
                            import xlrd
                            df = pd.read_excel(excel_file, engine='xlrd')
                        except:
                            print(f"    跳过（需要xlrd）")
                            continue
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # 识别列
                    voltage_col = None
                    current_col = None
                    capacity_col = None
                    
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'voltage' in col_lower or col == 'Voltage(V)':
                            voltage_col = col
                        if 'current' in col_lower or col == 'Current(A)':
                            current_col = col
                        if 'discharge_capacity' in col_lower or 'Discharge_Capacity' in str(col):
                            capacity_col = col
                    
                    if voltage_col is None:
                        continue
                    
                    # 提取电压数据
                    voltage = df[voltage_col].values.astype(float)
                    voltage = voltage[~np.isnan(voltage)]
                    
                    if len(voltage) == 0:
                        continue
                    
                    # 正确提取OCV-SOC数据：从静置点按SOC分组提取OCV
                    if current_col and capacity_col:
                        current = df[current_col].values.astype(float)
                        discharge_capacity = df[capacity_col].values.astype(float)
                        
                        # 对齐长度
                        min_len = min(len(voltage), len(current), len(discharge_capacity))
                        voltage = voltage[:min_len]
                        current = current[:min_len]
                        discharge_capacity = discharge_capacity[:min_len]
                        
                        # 找到静置点
                        rest_mask = np.abs(current) < 0.1
                        
                        if np.sum(rest_mask) > 0:
                            # 从容量计算每个静置点的SOC
                            rest_discharge_cap = discharge_capacity[rest_mask]
                            rest_voltage = voltage[rest_mask]
                            
                            # 计算容量范围
                            max_discharge_cap = discharge_capacity.max()
                            min_discharge_cap = discharge_capacity.min()
                            capacity_range = max_discharge_cap - min_discharge_cap
                            
                            if capacity_range > 0:
                                # 计算每个静置点的SOC
                                rest_soc = (1 - (rest_discharge_cap - min_discharge_cap) / capacity_range) * 100
                                rest_soc = np.clip(rest_soc, 0, 100)
                                
                                # 按SOC分组（每5%一组）
                                for soc_bin in range(0, 101, 5):
                                    mask = (rest_soc >= soc_bin) & (rest_soc < soc_bin + 5)
                                    if np.sum(mask) > 0:
                                        # 提取该SOC范围的OCV
                                        soc_ocv_values = rest_voltage[mask]
                                        
                                        # 分别提取充电和放电后的OCV（如果有）
                                        # 这里先取所有静置点的中位数
                                        ocv_mean = np.median(soc_ocv_values)
                                        
                                        ocv_data_points.append({
                                            'soc': float(soc_bin + 2.5),  # 使用组中心
                                            'ocv': ocv_mean,
                                            'temperature': ocv_temperature,
                                            'source': f"{ocv_folder.name}/{excel_file.name}",
                                            'n_samples': np.sum(mask)
                                        })
                                
                                print(f"    提取了 {len([p for p in ocv_data_points if p.get('source', '').endswith(excel_file.name)])} 个OCV-SOC数据点")
                            else:
                                print(f"    警告: 容量范围为0，无法计算SOC")
                        else:
                            print(f"    警告: 没有找到静置点")
                    else:
                        print(f"    警告: 缺少必要的列（Current或Discharge_Capacity）")
                
                except Exception as e:
                    print(f"    读取失败: {e}")
                    continue
        
        # 按SOC排序并去重（相同SOC取平均值）
        if ocv_data_points:
            ocv_data_points.sort(key=lambda x: x['soc'])
            
            # 按SOC分组并计算平均值
            soc_groups = {}
            for p in ocv_data_points:
                soc = p['soc']
                if soc not in soc_groups:
                    soc_groups[soc] = []
                soc_groups[soc].append(p['ocv'])
            
            # 计算每个SOC的平均OCV
            unique_ocv_points = []
            for soc in sorted(soc_groups.keys()):
                ocv_mean = np.mean(soc_groups[soc])
                ocv_std = np.std(soc_groups[soc])
                unique_ocv_points.append({
                    'soc': soc,
                    'ocv': ocv_mean,
                    'ocv_std': ocv_std,
                    'n_samples': len(soc_groups[soc])
                })
            
            # 构建OCV-SOC查找表
            soc_values = [p['soc'] for p in unique_ocv_points]
            ocv_values = [p['ocv'] for p in unique_ocv_points]
            
            self.ocv_soc_table = np.array([[s, v] for s, v in zip(soc_values, ocv_values)])
            
            print(f"\n成功构建OCV-SOC曲线: {len(self.ocv_soc_table)} 个唯一数据点")
            print("OCV-SOC数据点（去重后）:")
            for p in unique_ocv_points:
                print(f"  SOC={p['soc']:5.1f}% -> OCV={p['ocv']:.3f}V (std={p['ocv_std']:.3f}V, n={p['n_samples']})")
            
            # 检查数据完整性
            if soc_values[0] > 0:
                print(f"  警告: 缺少SOC=0%的数据，使用最低SOC={soc_values[0]}%的数据")
            if soc_values[-1] < 100:
                print(f"  警告: 缺少SOC=100%的数据，最高SOC={soc_values[-1]}%")
            
            return True
        else:
            print("\n未找到有效的OCV数据")
            return False
    
    def get_ocv_soc_table(self):
        """获取OCV-SOC查找表"""
        return self.ocv_soc_table
    
    def save_ocv_curve(self, output_file="ocv_soc_curve.csv"):
        """保存OCV-SOC曲线到文件"""
        if self.ocv_soc_table is not None:
            df = pd.DataFrame(self.ocv_soc_table, columns=['SOC(%)', 'OCV(V)'])
            df.to_csv(output_file, index=False)
            print(f"\nOCV-SOC曲线已保存到: {output_file}")
            return output_file
        return None

if __name__ == "__main__":
    builder = OCVCurveBuilder()
    builder.load_ocv_data()
    if builder.ocv_soc_table is not None:
        builder.save_ocv_curve()
