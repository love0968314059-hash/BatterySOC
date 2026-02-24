#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的扩展卡尔曼滤波(EKF) SOC估计器
核心改进：
1. 能够从大的初始误差收敛到正确的SOC
2. 根据OCV曲线斜率自适应调整增益（平坦区域减小增益避免震荡）
3. 电池参数(R0/R1/C1)随SOC变化
4. 自适应噪声参数调整
5. 数值稳定性保证
"""

import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class ImprovedEKFSOCEstimator:
    """
    改进的扩展卡尔曼滤波SOC估计器
    
    EKF原理：通过电压测量来校正基于AH积分的SOC预测
    - 预测步骤：使用AH积分预测SOC变化
    - 更新步骤：使用电压测量校正SOC估计
    
    对于LFP电池的平坦OCV曲线问题：
    - 在OCV斜率大的区域（低SOC<15%，高SOC>85%），更信任电压测量
    - 在OCV斜率小的区域（15%-85%），更依赖AH积分，但仍允许缓慢收敛
    """
    
    def __init__(self, 
                 initial_soc=50.0,
                 nominal_capacity=1.1,
                 ocv_soc_table=None,
                 process_noise_soc=0.0001,     # SOC过程噪声（较小，信任AH积分）
                 process_noise_v1=0.001,       # V1过程噪声
                 measurement_noise=0.0005,     # 测量噪声（较小，信任电压测量）
                 initial_soc_uncertainty=0.1,  # 初始SOC不确定性（对应10%误差）
                 enable_voltage_correction=True):  # 是否启用电压修正
        """
        初始化改进的EKF估计器
        
        Args:
            initial_soc: 初始SOC (%)，可以有较大误差
            nominal_capacity: 标称容量 (Ah)
            ocv_soc_table: OCV-SOC查找表，格式 [[soc1, ocv1], [soc2, ocv2], ...]
            process_noise_soc: SOC过程噪声方差
            process_noise_v1: RC电压过程噪声方差
            measurement_noise: 电压测量噪声方差
            initial_soc_uncertainty: 初始SOC不确定性（0-1范围，如0.1表示10%）
            enable_voltage_correction: 是否启用电压校正SOC
        """
        self.nominal_capacity = nominal_capacity
        self.enable_voltage_correction = enable_voltage_correction
        
        # 初始SOC（可能有误差）
        self.initial_soc = np.clip(initial_soc, 0, 100)
        
        # 噪声参数
        self.Q_soc_base = process_noise_soc
        self.Q_v1_base = process_noise_v1
        self.R_base = measurement_noise
        
        # SOC依赖的电池参数
        self._init_soc_dependent_params()
        
        # OCV-SOC映射
        self._init_ocv_soc_mapping(ocv_soc_table)
        
        # EKF状态：x = [SOC, V1]^T
        # SOC: 0-1范围
        # V1: RC回路极化电压
        self.x = np.array([self.initial_soc / 100.0, 0.0])
        
        # 协方差矩阵（初始不确定性）
        # 初始SOC可能有较大误差，协方差设置较大以允许收敛
        self.P = np.array([
            [initial_soc_uncertainty**2, 0.0],  # SOC初始不确定性
            [0.0, 0.001]                         # V1初始不确定性（较小）
        ])
        
        # 历史记录
        self.soc_history = []
        self.voltage_pred_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
        self.soc_correction_history = []
        
        self.last_time = None
        self.last_current = 0.0
        self.update_count = 0
        
        # 收敛监控
        self.convergence_window = []
        self.converged = False
    
    def _init_soc_dependent_params(self):
        """
        初始化SOC依赖的电池参数
        
        LFP电池典型参数变化：
        - R0（欧姆内阻）：低SOC时较大，中等SOC时最小
        - R1（极化电阻）：类似R0
        - C1（极化电容）：低SOC时较大
        """
        soc_points = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) / 100.0
        
        # R0随SOC变化（mΩ → Ω）
        r0_points = np.array([25, 18, 12, 10, 9, 8, 8.5, 9, 10, 12, 15]) / 1000.0
        # R1随SOC变化
        r1_points = np.array([15, 10, 7, 5, 4, 3.5, 4, 4.5, 5, 6, 8]) / 1000.0
        # C1随SOC变化
        c1_points = np.array([3000, 2500, 2000, 1800, 1600, 1500, 1400, 1300, 1200, 1100, 1000])
        
        self.r0_interp = interp1d(soc_points, r0_points, kind='cubic',
                                   fill_value='extrapolate', bounds_error=False)
        self.r1_interp = interp1d(soc_points, r1_points, kind='cubic',
                                   fill_value='extrapolate', bounds_error=False)
        self.c1_interp = interp1d(soc_points, c1_points, kind='cubic',
                                   fill_value='extrapolate', bounds_error=False)
    
    def _init_ocv_soc_mapping(self, ocv_soc_table):
        """初始化OCV-SOC映射"""
        if ocv_soc_table is not None and len(ocv_soc_table) > 0:
            self.ocv_soc_table = np.array(ocv_soc_table)
            socs = self.ocv_soc_table[:, 0] / 100.0
            ocvs = self.ocv_soc_table[:, 1]
            
            self.ocv_to_soc_interp = interp1d(
                ocvs, socs, kind='linear',
                fill_value='extrapolate', bounds_error=False
            )
            self.soc_to_ocv_interp = interp1d(
                socs, ocvs, kind='linear',
                fill_value='extrapolate', bounds_error=False
            )
            
            # 预计算OCV-SOC斜率用于自适应增益
            self._compute_ocv_slope_map(socs, ocvs)
        else:
            self.ocv_soc_table = None
            self.ocv_to_soc_interp = None
            self.soc_to_ocv_interp = None
            self.ocv_slope_map = None
    
    def _compute_ocv_slope_map(self, socs, ocvs):
        """预计算OCV-SOC曲线斜率映射"""
        # 计算数值斜率 dOCV/dSOC
        slopes = np.gradient(ocvs, socs)
        self.ocv_slope_interp = interp1d(
            socs, slopes, kind='linear',
            fill_value='extrapolate', bounds_error=False
        )
        
        # 记录斜率范围，用于自适应增益
        self.ocv_slope_min = np.min(np.abs(slopes))
        self.ocv_slope_max = np.max(np.abs(slopes))
        self.ocv_slope_median = np.median(np.abs(slopes))
    
    def get_r0(self, soc):
        """获取欧姆内阻"""
        return float(self.r0_interp(np.clip(soc, 0.0, 1.0)))
    
    def get_r1(self, soc):
        """获取极化电阻"""
        return float(self.r1_interp(np.clip(soc, 0.0, 1.0)))
    
    def get_c1(self, soc):
        """获取极化电容"""
        return float(self.c1_interp(np.clip(soc, 0.0, 1.0)))
    
    def _get_ocv(self, soc):
        """从SOC获取OCV"""
        if self.soc_to_ocv_interp is not None:
            return float(self.soc_to_ocv_interp(np.clip(soc, 0.0, 1.0)))
        else:
            # 默认LFP电池OCV-SOC近似
            return 2.5 + soc * 1.1
    
    def _get_docv_dsoc(self, soc):
        """计算dOCV/dSOC"""
        if self.ocv_slope_interp is not None:
            return float(self.ocv_slope_interp(np.clip(soc, 0.0, 1.0)))
        else:
            # 默认斜率
            return 1.1
    
    def _get_adaptive_gain_factor(self, soc, is_rest, innovation, current):
        """
        获取自适应增益因子（改进版：适用全SOC区间和动态工况）
        
        基于以下因素自适应调整增益：
        1. OCV曲线斜率 - 斜率大时电压信息更有价值
        2. 新息大小 - 新息过大说明模型失配，需谨慎
        3. 电流状态 - 大电流动态工况下模型不确定性大
        4. 新息一致性 - 连续新息同号说明存在系统误差需要修正
        
        Args:
            soc: 当前SOC (0-1)
            is_rest: 是否静置状态
            innovation: 新息（电压预测误差）
            current: 电流 (A)
        
        Returns:
            float: 增益调整因子 (0-1)
        """
        # 1. 基于OCV斜率的基础增益
        docv_dsoc = abs(self._get_docv_dsoc(soc))
        
        # LFP电池平坦区斜率约0.1-0.3 V/100%SOC，陡峭区可达2-3 V/100%SOC
        if docv_dsoc > 1.0:  # 陡峭区域
            base_gain = 0.5
        elif docv_dsoc > 0.5:  # 中等区域
            base_gain = 0.3
        elif docv_dsoc > 0.2:  # 轻微平坦
            base_gain = 0.1
        else:  # 平坦区域
            base_gain = 0.02  # 仍保留少量修正能力
        
        # 2. 静置状态修正
        if is_rest:
            rest_factor = 1.5  # 静置时电压更接近OCV
        else:
            rest_factor = 1.0
        
        # 3. 基于新息大小的调整
        innovation_mv = abs(innovation) * 1000  # 转为mV
        
        if innovation_mv < 20:
            innovation_factor = 1.2
        elif innovation_mv < 50:
            innovation_factor = 1.0
        elif innovation_mv < 100:
            innovation_factor = 0.5
        elif innovation_mv < 200:
            innovation_factor = 0.2
        else:
            innovation_factor = 0.05
        
        # 4. 基于电流的调整
        current_abs = abs(current)
        if current_abs < 0.1:
            current_factor = 1.2
        elif current_abs < 0.5:
            current_factor = 1.0
        elif current_abs < 1.0:
            current_factor = 0.7
        elif current_abs < 2.0:
            current_factor = 0.4
        else:
            current_factor = 0.2
        
        # 5. 基于新息一致性的调整（检测系统误差）
        n_recent = min(50, len(self.innovation_history))
        if n_recent >= 10:
            recent_innovations = self.innovation_history[-n_recent:]
            positive_count = sum(1 for x in recent_innovations if x > 0)
            same_sign_ratio = max(positive_count, n_recent - positive_count) / n_recent
            
            if same_sign_ratio > 0.8:
                consistency_factor = 1.5
            elif same_sign_ratio > 0.6:
                consistency_factor = 1.2
            else:
                consistency_factor = 1.0
        else:
            consistency_factor = 1.0
        
        # 综合计算增益因子
        gain_factor = base_gain * rest_factor * innovation_factor * current_factor * consistency_factor
        
        # 限制最大增益
        gain_factor = np.clip(gain_factor, 0.0, 0.8)
        
        return gain_factor
    
    def _get_adaptive_noise(self, soc, current, is_rest, dt):
        """
        获取自适应噪声参数
        
        Args:
            soc: 当前SOC (0-1)
            current: 电流 (A)
            is_rest: 是否静置
            dt: 时间步长 (s)
        
        Returns:
            Q: 过程噪声协方差矩阵
            R: 测量噪声协方差
        """
        # 基础过程噪声
        q_soc = self.Q_soc_base
        q_v1 = self.Q_v1_base
        r = self.R_base
        
        # 电流相关调整
        # 大电流时，模型不确定性增加
        current_factor = 1.0 + 0.3 * abs(current)
        q_v1 *= current_factor
        
        # 时间步长调整
        # 长时间步长时，过程噪声累积更多
        dt_factor = dt / 1.0  # 基准1秒
        q_soc *= dt_factor
        q_v1 *= dt_factor
        
        # 静置状态调整
        if is_rest:
            # 静置时V1应该衰减到0，减小V1的过程噪声
            q_v1 *= 0.5
            # 静置时电压测量更接近OCV，减小测量噪声
            r *= 0.3
        
        Q = np.array([
            [q_soc, 0.0],
            [0.0, q_v1]
        ])
        R = np.array([[r]])
        
        return Q, R
    
    def update(self, voltage, current, time=None, temperature=None):
        """
        EKF单步更新
        
        Args:
            voltage: 端电压 (V)
            current: 电流 (A)，正值充电，负值放电
            time: 时间戳 (s)
            temperature: 温度 (°C)，可选
        
        Returns:
            SOC估计值 (%)
        """
        self.update_count += 1
        
        # 处理时间步长
        if time is None:
            dt = 1.0
            if self.last_time is None:
                self.last_time = 0.0
        else:
            if self.last_time is not None:
                dt = time - self.last_time
                # 处理异常时间步长
                if dt <= 0:
                    dt = 1.0
                elif dt > 3600:
                    dt = 1.0  # 超过1小时视为数据跳跃
            else:
                dt = 1.0
            self.last_time = time
        
        # 当前状态
        soc = self.x[0]
        v1 = self.x[1]
        
        # 获取SOC依赖的参数
        r0 = self.get_r0(soc)
        r1 = self.get_r1(soc)
        c1 = self.get_c1(soc)
        tau1 = r1 * c1
        
        # 判断静置状态
        is_rest = abs(current) < 0.05
        
        # ==================== 预测步骤 ====================
        
        # SOC预测（AH积分）
        # delta_SOC = I * dt / (Capacity * 3600) * eta
        eta = 1.0  # 库伦效率
        delta_soc = current * dt / 3600 / self.nominal_capacity * eta
        soc_pred = soc + delta_soc
        soc_pred = np.clip(soc_pred, 0.0, 1.0)
        
        # V1预测（一阶RC模型）
        # 注意：电流正为充电，负为放电
        # 充电时V1>0（极化使端电压升高），放电时V1<0
        # V1(k+1) = V1(k) * exp(-dt/tau) + I * R1 * (1 - exp(-dt/tau))
        exp_factor = np.exp(-dt / (tau1 + 1e-6))
        v1_pred = v1 * exp_factor + current * r1 * (1 - exp_factor)
        
        x_pred = np.array([soc_pred, v1_pred])
        
        # 状态转移雅可比矩阵 F
        F = np.array([
            [1.0, 0.0],       # dSOC'/dSOC = 1
            [0.0, exp_factor]  # dV1'/dV1 = exp(-dt/tau)
        ])
        
        # 获取自适应噪声
        Q, R = self._get_adaptive_noise(soc, current, is_rest, dt)
        
        # 预测协方差
        P_pred = F @ self.P @ F.T + Q
        
        # ==================== 更新步骤 ====================
        
        # 使用预测SOC获取参数
        r0_pred = self.get_r0(soc_pred)
        
        # 预测电压
        # 电流符号约定：正为充电，负为放电
        # 端电压 = OCV + I*R0 + V1
        # 充电时(I>0)：端电压高于OCV；放电时(I<0)：端电压低于OCV
        ocv_pred = self._get_ocv(soc_pred)
        v_pred = ocv_pred + current * r0_pred + v1_pred
        
        # 观测雅可比矩阵 H = dh/dx
        # h(x) = OCV(SOC) + I*R0 + V1
        # dh/dSOC = dOCV/dSOC, dh/dV1 = 1
        docv_dsoc = self._get_docv_dsoc(soc_pred)
        H = np.array([[docv_dsoc, 1.0]])
        
        # 新息（测量残差）
        innovation = voltage - v_pred
        self.innovation_history.append(innovation)
        
        # 如果不启用电压校正，直接使用预测值
        if not self.enable_voltage_correction:
            self.x = x_pred
            self.P = P_pred
            soc_percent = self.x[0] * 100.0
            self.soc_history.append(soc_percent)
            self.voltage_pred_history.append(v_pred)
            self.kalman_gain_history.append(0.0)
            self.soc_correction_history.append(0.0)
            return soc_percent
        
        # 计算卡尔曼增益
        S = H @ P_pred @ H.T + R
        S_inv = 1.0 / (S[0, 0] + 1e-10)
        K = P_pred @ H.T * S_inv
        
        # 自适应增益调整（传入电流参数）
        gain_factor = self._get_adaptive_gain_factor(soc_pred, is_rest, innovation, current)
        
        # 限制SOC的卡尔曼增益
        # 这是防止在平坦OCV区域过度修正的关键
        k_soc = K[0, 0] * gain_factor
        k_v1 = K[1, 0]
        
        # 额外的增益限制
        # 每步SOC修正不超过一定幅度（防止震荡）
        max_soc_correction_rate = 0.02  # 每步最多修正2%
        if abs(k_soc * innovation) > max_soc_correction_rate:
            # 限制修正幅度
            k_soc = np.sign(k_soc) * max_soc_correction_rate / (abs(innovation) + 1e-6)
        
        K_adjusted = np.array([[k_soc], [k_v1]])
        
        self.kalman_gain_history.append(k_soc)
        
        # 状态更新
        x_update = x_pred + K_adjusted.flatten() * innovation
        x_update[0] = np.clip(x_update[0], 0.0, 1.0)
        self.x = x_update
        
        # 记录SOC修正量
        soc_correction = k_soc * innovation * 100  # 转换为百分比
        self.soc_correction_history.append(soc_correction)
        
        # 协方差更新（Joseph形式，保证正定性）
        I_KH = np.eye(2) - K_adjusted @ H
        self.P = I_KH @ P_pred @ I_KH.T + K_adjusted @ R @ K_adjusted.T
        
        # 确保协方差矩阵正定
        self.P = (self.P + self.P.T) / 2
        min_variance = 1e-10
        self.P[0, 0] = max(self.P[0, 0], min_variance)
        self.P[1, 1] = max(self.P[1, 1], min_variance)
        
        self.last_current = current
        self.voltage_pred_history.append(v_pred)
        
        soc_percent = self.x[0] * 100.0
        self.soc_history.append(soc_percent)
        
        # 收敛监控
        self.convergence_window.append(abs(innovation))
        if len(self.convergence_window) > 100:
            self.convergence_window.pop(0)
            # 检查是否收敛（新息稳定在小范围内）
            if np.mean(self.convergence_window[-50:]) < 0.02:
                self.converged = True
        
        return soc_percent
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """
        批量估计SOC
        
        Args:
            voltage: 电压序列 (V)
            current: 电流序列 (A)
            time: 时间序列 (s)
            temperature: 温度序列 (°C)
        
        Returns:
            SOC估计序列 (%)
        """
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        n = len(voltage)
        
        if time is None:
            time = np.arange(n, dtype=float)
        else:
            time = np.asarray(time, dtype=float)
        
        if temperature is None:
            temperature = np.full(n, 25.0)
        else:
            temperature = np.asarray(temperature)
        
        # 重置状态
        self.x = np.array([self.initial_soc / 100.0, 0.0])
        self.P = np.array([
            [0.1, 0.0],  # 初始SOC不确定性大，允许收敛
            [0.0, 0.001]
        ])
        
        self.soc_history = []
        self.voltage_pred_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
        self.soc_correction_history = []
        self.convergence_window = []
        self.converged = False
        self.last_time = None
        self.update_count = 0
        
        # 逐点估计
        soc_estimated = []
        for i in range(n):
            soc = self.update(voltage[i], current[i], time[i], temperature[i])
            soc_estimated.append(soc)
        
        return np.array(soc_estimated)
    
    def get_diagnostics(self):
        """获取诊断信息"""
        if len(self.innovation_history) == 0:
            return {}
        
        innovations = np.array(self.innovation_history)
        kalman_gains = np.array(self.kalman_gain_history)
        soc_corrections = np.array(self.soc_correction_history) if self.soc_correction_history else np.array([0])
        
        return {
            'n_updates': self.update_count,
            'innovation_mean': np.mean(innovations),
            'innovation_std': np.std(innovations),
            'innovation_max': np.max(np.abs(innovations)),
            'kalman_gain_mean': np.mean(kalman_gains),
            'kalman_gain_std': np.std(kalman_gains),
            'total_soc_correction': np.sum(soc_corrections),
            'max_soc_correction': np.max(np.abs(soc_corrections)),
            'converged': self.converged,
            'final_soc': self.x[0] * 100.0,
            'final_v1': self.x[1],
            'final_P_soc': self.P[0, 0],
            'final_P_v1': self.P[1, 1]
        }


# 测试代码
if __name__ == "__main__":
    print("测试改进的EKF估计器 - 初始误差收敛能力")
    print("=" * 60)
    
    # 创建OCV-SOC表（典型LFP电池）
    ocv_soc_table = np.array([
        [0, 2.50],
        [5, 2.80],
        [10, 3.00],
        [15, 3.10],
        [20, 3.20],
        [30, 3.25],
        [40, 3.28],
        [50, 3.30],
        [60, 3.32],
        [70, 3.34],
        [80, 3.36],
        [85, 3.40],
        [90, 3.45],
        [95, 3.55],
        [100, 3.60]
    ])
    
    # 模拟数据
    np.random.seed(42)
    n = 3000
    
    # 时间
    time = np.arange(n, dtype=float)
    
    # 电流：模拟动态工况（充放电循环）
    current = np.zeros(n)
    for i in range(n):
        t = i / n
        if t < 0.3:
            current[i] = -0.5  # 放电
        elif t < 0.35:
            current[i] = 0.0  # 静置
        elif t < 0.7:
            current[i] = 0.3  # 充电
        elif t < 0.75:
            current[i] = 0.0  # 静置
        else:
            current[i] = -0.3  # 放电
    
    # 添加一些噪声
    current += 0.02 * np.random.randn(n)
    
    # 计算真实SOC（AH积分）
    capacity = 1.1
    dt = np.diff(time)
    dt = np.concatenate([[1.0], dt])
    cumulative_ah = np.cumsum(current * dt / 3600)
    soc_true = 50.0 + cumulative_ah / capacity * 100
    soc_true = np.clip(soc_true, 0, 100)
    
    # 计算对应的电压
    soc_to_ocv = interp1d(
        ocv_soc_table[:, 0] / 100.0,
        ocv_soc_table[:, 1],
        kind='linear', fill_value='extrapolate'
    )
    voltage = soc_to_ocv(soc_true / 100.0) - current * 0.01 + 0.002 * np.random.randn(n)
    
    print(f"真实初始SOC: {soc_true[0]:.2f}%")
    print(f"真实最终SOC: {soc_true[-1]:.2f}%")
    
    # 测试不同初始误差
    for initial_error in [0, 10, 20, -10, -20]:
        print(f"\n--- 初始SOC误差: {initial_error:+d}% ---")
        
        # 创建EKF估计器
        ekf = ImprovedEKFSOCEstimator(
            initial_soc=soc_true[0] + initial_error,
            nominal_capacity=capacity,
            ocv_soc_table=ocv_soc_table,
            process_noise_soc=0.0001,
            measurement_noise=0.0005,
            initial_soc_uncertainty=0.15,
            enable_voltage_correction=True
        )
        
        # 批量估计
        soc_estimated = ekf.estimate_batch(voltage, current, time)
        
        # 计算误差
        error = soc_estimated - soc_true
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        
        print(f"  MAE: {mae:.3f}%")
        print(f"  RMSE: {rmse:.3f}%")
        print(f"  初始误差: {soc_estimated[0] - soc_true[0]:.3f}%")
        print(f"  最终误差: {soc_estimated[-1] - soc_true[-1]:.3f}%")
        
        # 诊断信息
        diag = ekf.get_diagnostics()
        print(f"  总SOC修正: {diag['total_soc_correction']:.3f}%")
        print(f"  收敛状态: {'是' if diag['converged'] else '否'}")
