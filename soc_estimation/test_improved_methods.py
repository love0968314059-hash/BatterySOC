#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的SOC估计方法
关键改进：
1. 先对原始数据进行重采样，再计算SOC标签（保证数据一致性）
2. 测试EKF从大初始误差收敛的能力
3. 对比多种SOC估计方法
"""

import sys
from pathlib import Path
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from data_processor import BatteryDataProcessor
from evaluator import SOCEvaluator
from ocv_curve_builder import OCVCurveBuilder
from realtime_soc_estimator import RealtimeSOCEstimator
from advanced_soc_estimators import ExtendedKalmanFilterSOC, ParticleFilterSOC
from data_resampler import DataResampler
from improved_ekf_estimator import ImprovedEKFSOCEstimator

# 尝试导入AI方法
try:
    from improved_ai_estimator import ImprovedAISOCEstimator, TORCH_AVAILABLE
    AI_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    AI_AVAILABLE = False


def log_to_file(log_file, message):
    """记录到日志文件"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def calculate_soc_labels(time, current, voltage, capacity, ocv_soc_table=None):
    """
    计算SOC标签（基于AH积分，从已知参考点倒推）
    
    关键：需要找到一个可靠的参考点来确定绝对SOC
    - 充电结束后静置点 → SOC≈100%
    - 放电截止点 → SOC≈0%
    """
    time = np.asarray(time)
    current = np.asarray(current)
    voltage = np.asarray(voltage)
    
    # 计算时间差
    dt = np.diff(time)
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0], dt])
    # 处理异常时间差
    dt_median = np.median(dt[dt > 0])
    dt[dt <= 0] = dt_median
    dt[dt > dt_median * 10] = dt_median
    
    # 计算累积AH（从0开始）
    cumulative_ah = np.cumsum(current * dt / 3600)
    
    # SOC变化 = 累积AH / 容量 * 100%
    soc_change = cumulative_ah / capacity * 100
    
    # 寻找参考点来确定起始SOC
    # 优先级：充电结束静置点 > 最大电压点 > 最小电压点 > OCV估算
    
    initial_soc = None
    
    # 方法1：寻找充电结束后的静置点（高电压+小电流）
    rest_mask = np.abs(current) < 0.05
    high_voltage_mask = voltage >= 3.5
    high_rest_mask = rest_mask & high_voltage_mask
    high_rest_indices = np.where(high_rest_mask)[0]
    
    if len(high_rest_indices) > 10:  # 至少10个点的静置
        # 找到电压最高的静置点
        best_idx = high_rest_indices[np.argmax(voltage[high_rest_indices])]
        # 此时SOC应该接近100%
        soc_at_rest = 100.0
        initial_soc = soc_at_rest - soc_change[best_idx]
        print(f"    使用充电静置点(V={voltage[best_idx]:.3f}V, idx={best_idx})确定初始SOC: {initial_soc:.2f}%")
    
    # 方法2：最大电压点
    if initial_soc is None:
        max_v_idx = np.argmax(voltage)
        max_v = voltage[max_v_idx]
        if max_v >= 3.55:  # 接近满充
            soc_at_max = 100.0
            initial_soc = soc_at_max - soc_change[max_v_idx]
            print(f"    使用最大电压点(V={max_v:.3f}V)确定初始SOC: {initial_soc:.2f}%")
    
    # 方法3：最小电压点
    if initial_soc is None:
        min_v_idx = np.argmin(voltage)
        min_v = voltage[min_v_idx]
        if min_v <= 2.5:  # 接近放空
            soc_at_min = 0.0
            initial_soc = soc_at_min - soc_change[min_v_idx]
            print(f"    使用最小电压点(V={min_v:.3f}V)确定初始SOC: {initial_soc:.2f}%")
    
    # 方法4：使用OCV表估算初始SOC
    if initial_soc is None and ocv_soc_table is not None:
        from scipy.interpolate import interp1d
        ocv_soc_table = np.array(ocv_soc_table)
        ocv_to_soc = interp1d(ocv_soc_table[:, 1], ocv_soc_table[:, 0],
                              kind='linear', fill_value='extrapolate')
        # 使用初始电压估算（假设起始点接近静置）
        initial_soc = float(ocv_to_soc(voltage[0]))
        initial_soc = np.clip(initial_soc, 0, 100)
        print(f"    使用OCV表估算初始SOC: {initial_soc:.2f}%")
    
    # 默认
    if initial_soc is None:
        initial_soc = 50.0
        print(f"    使用默认初始SOC: {initial_soc:.2f}%")
    
    # 计算SOC序列
    soc = initial_soc + soc_change
    
    # 调整确保SOC在0-100%范围内
    if soc.min() < 0:
        initial_soc = initial_soc - soc.min()
        soc = initial_soc + soc_change
    if soc.max() > 100:
        initial_soc = initial_soc - (soc.max() - 100)
        soc = initial_soc + soc_change
    
    soc = np.clip(soc, 0, 100)
    
    return soc


def test_improved_methods(debug_mode=True):
    """
    测试改进的SOC估计方法
    
    关键改进流程：
    1. 加载原始数据
    2. 对原始数据进行重采样（等间隔，异常间隔填0电流）
    3. 使用重采样后的数据重新计算SOC标签
    4. 测试各种SOC估计方法
    """
    # 设置路径
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    LOG_FILE = PROJECT_ROOT / "CONVERSATION_LOG.md"
    
    print("=" * 80)
    print("测试改进的SOC估计方法")
    print("=" * 80)
    
    # 记录到日志
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(LOG_FILE, f"\n---\n## {timestamp} - 测试改进的SOC估计方法（修正流程）\n")
    log_to_file(LOG_FILE, """### 关键改进
1. 先对原始数据进行重采样，再计算SOC标签（保证一致性）
2. 测试EKF从大初始误差收敛的能力
3. 根据OCV曲线斜率自适应调整EKF增益
""")
    
    # 加载数据
    raw_data_path = PROJECT_ROOT / "raw_data"
    processor = BatteryDataProcessor(data_dir=str(raw_data_path))
    
    # 获取数据文件
    data_files = []
    raw_data_dir = PROJECT_ROOT / "raw_data"
    for temp_dir in raw_data_dir.glob("DST-US06-FUDS-*"):
        if not temp_dir.is_dir():
            continue
        temp_files = [f for f in temp_dir.glob("*.xlsx") 
                     if 'newprofile' not in f.name and '20120809' not in f.name]
        data_files.extend(temp_files)
    
    if len(data_files) == 0:
        print("错误: 未找到数据文件")
        return
    
    print(f"找到 {len(data_files)} 个数据文件")
    
    if debug_mode:
        data_files = data_files[:1]
        print(f"调试模式: 只处理第一个文件")
    
    # 加载容量信息
    capacity_info_path = PROJECT_ROOT / "soc_results" / "ocv_by_temperature" / "all_temperatures_capacity_info.json"
    all_temperatures_capacity = {}
    if capacity_info_path.exists():
        with open(capacity_info_path, 'r') as f:
            all_temperatures_capacity = json.load(f)
    
    # 处理每个文件
    all_results = []
    
    for filepath in data_files:
        print(f"\n{'='*60}")
        print(f"处理文件: {filepath.name}")
        print(f"{'='*60}")
        
        # ================================================================
        # 第一步：加载原始数据（不使用processor.process_file的SOC计算）
        # ================================================================
        print(f"\n[步骤1] 加载原始数据...")
        
        df = processor.load_data_file(filepath)
        if df is None:
            print(f"  跳过: 无法加载文件")
            continue
        
        col_map = processor.identify_columns(df)
        if 'voltage' not in col_map or 'current' not in col_map or 'time' not in col_map:
            print(f"  跳过: 缺少必要列")
            continue
        
        clean_result = processor.clean_data(df, col_map)
        voltage_raw = clean_result['voltage']
        current_raw = clean_result['current']
        time_raw = clean_result['time']
        temperature_raw = clean_result['temperature']
        
        if voltage_raw is None or len(voltage_raw) < 100:
            print(f"  跳过: 数据太少")
            continue
        
        print(f"  原始数据点数: {len(voltage_raw)}")
        
        # 提取温度
        temp_value = processor.extract_temperature_from_path(filepath)
        if temp_value is None:
            temp_value = 30  # 默认30°C
        print(f"  温度: {temp_value}°C")
        
        # 加载OCV曲线和容量
        ocv_builder = OCVCurveBuilder(ocv_data_dir=str(raw_data_path))
        ocv_builder.load_ocv_data(target_temperature=temp_value, use_test_file=True)
        ocv_soc_table = ocv_builder.get_ocv_soc_table()
        nominal_capacity = ocv_builder.nominal_capacity or 1.1
        actual_capacity = ocv_builder.actual_discharge_capacity or nominal_capacity
        
        temp_key = str(temp_value)
        if temp_key in all_temperatures_capacity:
            temp_capacity_info = all_temperatures_capacity[temp_key]
            actual_capacity = temp_capacity_info.get('actual_discharge_capacity_ah', actual_capacity)
        
        print(f"  容量: {actual_capacity:.4f} Ah")
        
        # ================================================================
        # 第二步：数据重采样
        # ================================================================
        print(f"\n[步骤2] 数据重采样...")
        
        resampler = DataResampler(target_dt=1.0, max_dt_ratio=10.0, verbose=True)
        
        if temperature_raw is None:
            temperature_raw = np.full(len(voltage_raw), float(temp_value))
        
        resampled = resampler.resample(
            time_raw, voltage_raw, current_raw, temperature_raw
        )
        
        time_proc = resampled['time']
        voltage_proc = resampled['voltage']
        current_proc = resampled['current']
        temperature_proc = resampled['temperature'] if resampled['temperature'] is not None else np.full(len(time_proc), float(temp_value))
        
        print(f"  重采样后数据点数: {len(time_proc)}")
        
        # ================================================================
        # 第三步：重新计算SOC标签
        # ================================================================
        print(f"\n[步骤3] 重新计算SOC标签...")
        
        soc_true = calculate_soc_labels(
            time_proc, current_proc, voltage_proc, 
            actual_capacity, ocv_soc_table
        )
        
        initial_soc = soc_true[0]
        print(f"  初始SOC: {initial_soc:.3f}%")
        print(f"  SOC范围: {soc_true.min():.3f}% - {soc_true.max():.3f}%")
        
        # ================================================================
        # 第四步：测试各种SOC估计方法
        # ================================================================
        
        # ----------------------------------------------------------------
        # 方法1: 实时AH积分+OCV校准（基准方法，无初始误差）
        # ----------------------------------------------------------------
        print(f"\n[方法1] 实时AH积分+OCV校准（无初始误差）...")
        realtime_estimator = RealtimeSOCEstimator(
            initial_soc=initial_soc,
            nominal_capacity=actual_capacity,
            ocv_soc_table=ocv_soc_table,
            rest_current_threshold=0.05,
            rest_duration_threshold=30.0,
            enable_interpolation=False
        )
        soc_est_realtime = realtime_estimator.estimate_batch(
            voltage_proc, current_proc, time_proc, temperature_proc
        )
        
        # ----------------------------------------------------------------
        # 方法2: 原始EKF（有10%初始误差）
        # ----------------------------------------------------------------
        print(f"\n[方法2] 原始EKF（+10%初始误差）...")
        initial_error = 10.0
        initial_soc_with_error = min(100, initial_soc + initial_error)
        if initial_soc_with_error > 100:
            initial_soc_with_error = initial_soc - initial_error
        
        ekf_original = ExtendedKalmanFilterSOC(
            initial_soc=initial_soc_with_error,
            nominal_capacity=actual_capacity,
            ocv_soc_table=ocv_soc_table,
            process_noise=0.001,
            measurement_noise=0.001,
            r0=0.01,
            r1=0.005,
            c1=1500.0
        )
        soc_est_ekf_original = ekf_original.estimate_batch(
            voltage_proc, current_proc, time_proc, temperature_proc
        )
        print(f"  EKF初始SOC: {initial_soc_with_error:.2f}% (真实: {initial_soc:.2f}%)")
        
        # ----------------------------------------------------------------
        # 方法3: 改进的EKF（无初始误差，测试基本性能）
        # ----------------------------------------------------------------
        print(f"\n[方法3] 改进EKF（无初始误差，测试基本性能）...")
        ekf_improved_no_error = ImprovedEKFSOCEstimator(
            initial_soc=initial_soc,
            nominal_capacity=actual_capacity,
            ocv_soc_table=ocv_soc_table,
            process_noise_soc=0.0001,
            measurement_noise=0.0005,
            enable_voltage_correction=True
        )
        soc_est_ekf_improved_no_err = ekf_improved_no_error.estimate_batch(
            voltage_proc, current_proc, time_proc, temperature_proc
        )
        
        # ----------------------------------------------------------------
        # 方法4: 改进的EKF（+10%初始误差，测试收敛能力）
        # ----------------------------------------------------------------
        print(f"\n[方法4] 改进EKF（+10%初始误差，测试收敛能力）...")
        ekf_improved_with_error = ImprovedEKFSOCEstimator(
            initial_soc=initial_soc_with_error,
            nominal_capacity=actual_capacity,
            ocv_soc_table=ocv_soc_table,
            process_noise_soc=0.0001,
            measurement_noise=0.0005,
            initial_soc_uncertainty=0.15,  # 较大不确定性，允许收敛
            enable_voltage_correction=True
        )
        soc_est_ekf_improved_with_err = ekf_improved_with_error.estimate_batch(
            voltage_proc, current_proc, time_proc, temperature_proc
        )
        print(f"  改进EKF初始SOC: {initial_soc_with_error:.2f}% (真实: {initial_soc:.2f}%)")
        
        # 获取EKF诊断信息
        diag = ekf_improved_with_error.get_diagnostics()
        print(f"  总SOC修正量: {diag.get('total_soc_correction', 0):.3f}%")
        print(f"  收敛状态: {'是' if diag.get('converged', False) else '否'}")
        
        # ----------------------------------------------------------------
        # 方法5: 改进的EKF（-10%初始误差，测试负向收敛）
        # ----------------------------------------------------------------
        print(f"\n[方法5] 改进EKF（-10%初始误差，测试负向收敛）...")
        initial_soc_neg_error = max(0, initial_soc - initial_error)
        
        ekf_improved_neg_error = ImprovedEKFSOCEstimator(
            initial_soc=initial_soc_neg_error,
            nominal_capacity=actual_capacity,
            ocv_soc_table=ocv_soc_table,
            process_noise_soc=0.0001,
            measurement_noise=0.0005,
            initial_soc_uncertainty=0.15,
            enable_voltage_correction=True
        )
        soc_est_ekf_improved_neg_err = ekf_improved_neg_error.estimate_batch(
            voltage_proc, current_proc, time_proc, temperature_proc
        )
        print(f"  改进EKF初始SOC: {initial_soc_neg_error:.2f}% (真实: {initial_soc:.2f}%)")
        
        # ----------------------------------------------------------------
        # 方法6: 粒子滤波
        # ----------------------------------------------------------------
        print(f"\n[方法6] 粒子滤波...")
        pf_estimator = ParticleFilterSOC(
            initial_soc=initial_soc,
            nominal_capacity=actual_capacity,
            ocv_soc_table=ocv_soc_table,
            n_particles=200,
            process_noise=0.001,
            measurement_noise=0.001
        )
        soc_est_pf = pf_estimator.estimate_batch(
            voltage_proc, current_proc, time_proc, temperature_proc
        )
        
        # ----------------------------------------------------------------
        # 方法7: AI方法 (GRU)
        # ----------------------------------------------------------------
        soc_est_ai = None
        if AI_AVAILABLE:
            print(f"\n[方法7] AI方法 (GRU)...")
            try:
                ai_estimator = ImprovedAISOCEstimator(
                    initial_soc=initial_soc,
                    nominal_capacity=actual_capacity,
                    model_type='gru',
                    sequence_length=15,
                    hidden_size=64,
                    num_layers=2,
                    device='cpu'
                )
                soc_est_ai = ai_estimator.estimate_batch(
                    voltage_proc, current_proc, time_proc, temperature_proc,
                    soc_true=soc_true
                )
            except Exception as e:
                print(f"    AI方法失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[方法7] AI方法: 跳过（PyTorch未安装）")
        
        # ================================================================
        # 评估所有方法
        # ================================================================
        print(f"\n{'='*60}")
        print("评估结果")
        print(f"{'='*60}")
        
        evaluator = SOCEvaluator()
        
        # 确保长度匹配
        lengths = [len(soc_true), len(soc_est_realtime), 
                   len(soc_est_ekf_original), len(soc_est_ekf_improved_no_err),
                   len(soc_est_ekf_improved_with_err), len(soc_est_ekf_improved_neg_err),
                   len(soc_est_pf)]
        if soc_est_ai is not None:
            lengths.append(len(soc_est_ai))
        min_len = min(lengths)
        
        methods_to_eval = [
            ("实时AH+OCV (无误差)", soc_est_realtime[:min_len]),
            ("原始EKF (+10%误差)", soc_est_ekf_original[:min_len]),
            ("改进EKF (无误差)", soc_est_ekf_improved_no_err[:min_len]),
            ("改进EKF (+10%误差)", soc_est_ekf_improved_with_err[:min_len]),
            ("改进EKF (-10%误差)", soc_est_ekf_improved_neg_err[:min_len]),
            ("粒子滤波 (无误差)", soc_est_pf[:min_len])
        ]
        
        if soc_est_ai is not None:
            methods_to_eval.append(("AI方法 (GRU)", soc_est_ai[:min_len]))
        
        results = {}
        target_mae = 5.0
        
        print(f"\n{'方法':<30} {'MAE':<10} {'RMSE':<10} {'Max Err':<12} {'达标':<8}")
        print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
        
        for name, soc_est in methods_to_eval:
            eval_result = evaluator.evaluate(soc_true[:min_len], soc_est)
            results[name] = eval_result
            
            mae = eval_result['mae']
            rmse = eval_result['rmse']
            max_err = eval_result['max_error']
            status = "✓" if mae < target_mae else "✗"
            
            print(f"{name:<30} {mae:<10.3f} {rmse:<10.3f} {max_err:<12.3f} {status:<8}")
        
        # 记录到日志
        log_message = f"""
### 文件: {filepath.name}
- 温度: {temp_value}°C
- 原始数据点: {len(voltage_raw)}
- 重采样后数据点: {len(time_proc)}
- 初始SOC: {initial_soc:.3f}%
- SOC范围: {soc_true.min():.3f}% - {soc_true.max():.3f}%

| 方法 | MAE | RMSE | Max Error | 达标(<5%) |
|------|-----|------|-----------|---------|"""
        
        for name, eval_result in results.items():
            status = '✓' if eval_result['mae'] < 5 else '✗'
            log_message += f"\n| {name} | {eval_result['mae']:.3f}% | {eval_result['rmse']:.3f}% | {eval_result['max_error']:.3f}% | {status} |"
        
        log_to_file(LOG_FILE, log_message)
        
        # 保存结果
        all_results.append({
            'filename': filepath.name,
            'temperature': temp_value,
            'n_points_raw': len(voltage_raw),
            'n_points_resampled': len(time_proc),
            'initial_soc': initial_soc,
            'results': results
        })
    
    # 总结
    print(f"\n{'='*80}")
    print("测试完成总结")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    test_improved_methods(debug_mode=True)
