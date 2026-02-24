#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级SOC估计算法
实现扩展卡尔曼滤波(EKF)和粒子滤波(PF)
"""

import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class ExtendedKalmanFilterSOC:
    """
    扩展卡尔曼滤波SOC估计器
    使用等效电路模型(ECM)进行SOC估计
    """
    
    def __init__(self, 
                 initial_soc=50.0,
                 nominal_capacity=1.1,
                 ocv_soc_table=None,
                 process_noise=0.01,
                 measurement_noise=0.01,
                 r0=0.01,
                 r1=0.005,
                 c1=1000.0,
                 tau1=None):
        """
        初始化EKF估计器
        
        Args:
            initial_soc: 初始SOC (%)
            nominal_capacity: 标称容量 (Ah)
            ocv_soc_table: OCV-SOC查找表
            process_noise: 过程噪声协方差
            measurement_noise: 测量噪声协方差
            r0: 欧姆内阻 (Ohm)
            r1: RC回路电阻 (Ohm)
            c1: RC回路电容 (F)
            tau1: RC时间常数 (s)，如果为None则从r1和c1计算
        """
        self.nominal_capacity = nominal_capacity
        self.initial_soc = initial_soc / 100.0  # 转换为0-1范围
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # 等效电路模型参数
        self.r0 = r0  # 欧姆内阻
        self.r1 = r1  # RC回路电阻
        self.c1 = c1  # RC回路电容
        self.tau1 = tau1 if tau1 is not None else r1 * c1  # RC时间常数
        
        # OCV-SOC映射
        if ocv_soc_table is not None and len(ocv_soc_table) > 0:
            self.ocv_soc_table = np.array(ocv_soc_table)
            ocv_soc_soc = self.ocv_soc_table[:, 0] / 100.0  # 转换为0-1范围
            ocv_soc_ocv = self.ocv_soc_table[:, 1]
            self.ocv_to_soc_interp = interp1d(
                ocv_soc_ocv, ocv_soc_soc,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            self.soc_to_ocv_interp = interp1d(
                ocv_soc_soc, ocv_soc_ocv,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
        else:
            self.ocv_soc_table = None
            self.ocv_to_soc_interp = None
            self.soc_to_ocv_interp = None
        
        # EKF状态：x = [SOC, V1]^T
        # SOC: 0-1范围
        # V1: RC回路电压
        self.x = np.array([self.initial_soc, 0.0])  # 状态向量（确保初始SOC正确）
        self.P = np.eye(2) * 0.01  # 协方差矩阵（降低初始不确定性）
        
        # 历史记录
        self.soc_history = []
        self.last_time = None
        self.last_current = 0.0
    
    def _get_ocv(self, soc):
        """从SOC获取OCV"""
        if self.soc_to_ocv_interp is not None:
            soc_clipped = np.clip(soc, 0.0, 1.0)
            return float(self.soc_to_ocv_interp(soc_clipped))
        else:
            # 默认线性映射
            return 2.0 + soc * 1.6
    
    def _get_docv_dsoc(self, soc):
        """计算dOCV/dSOC（OCV-SOC曲线的斜率）"""
        if self.ocv_soc_table is not None:
            soc_clipped = np.clip(soc, 0.0, 1.0)
            soc_percent = soc_clipped * 100.0
            
            # 使用数值微分
            eps = 0.01
            ocv1 = self._get_ocv(soc_clipped - eps)
            ocv2 = self._get_ocv(soc_clipped + eps)
            docv_dsoc = (ocv2 - ocv1) / (2 * eps)
            return docv_dsoc
        else:
            return 1.6  # 默认斜率
    
    def update(self, voltage, current, time=None, temperature=None):
        """
        更新EKF状态（单点输入）
        
        Args:
            voltage: 电压 (V)
            current: 电流 (A)，正值为充电
            time: 时间 (s)
            temperature: 温度 (°C)，可选
        
        Returns:
            当前SOC估计值 (%)
        """
        # 处理时间（关键修复：使用实际时间差）
        if time is None:
            if self.last_time is not None:
                dt = 1.0  # 默认1秒
            else:
                dt = 1.0
                self.last_time = 0.0
        else:
            if self.last_time is not None:
                dt = time - self.last_time
                if dt <= 0:
                    dt = 1.0  # 时间倒退或不变，使用默认值
                elif dt > 3600:  # 超过1小时，可能是数据跳跃
                    dt = 1.0
            else:
                dt = 1.0
            self.last_time = time
        
        # 预测步骤（状态预测）
        # SOC更新：主要依赖AH积分（更准确），EKF主要用于估计V1
        # 注意：电流正值为充电（SOC增加），负值为放电（SOC减少）
        eta = 1.0  # 库伦效率（LFP电池接近1.0）
        delta_soc = current * dt / 3600 / self.nominal_capacity * eta  # AH积分
        soc_pred = self.x[0] + delta_soc
        soc_pred = np.clip(soc_pred, 0.0, 1.0)
        
        # 关键改进：SOC主要依赖AH积分，EKF主要用于估计V1和OCV校准
        # 这样可以避免SOC估计被电压测量过度修正
        
        # V1更新：V1(k+1) = V1(k) * exp(-dt/tau1) + I(k) * R1 * (1 - exp(-dt/tau1))
        v1_pred = self.x[1] * np.exp(-dt / self.tau1) + current * self.r1 * (1 - np.exp(-dt / self.tau1))
        
        x_pred = np.array([soc_pred, v1_pred])
        
        # 状态转移雅可比矩阵 F = df/dx
        F = np.array([
            [1.0, 0.0],  # dSOC/dSOC, dSOC/dV1
            [0.0, np.exp(-dt / self.tau1)]  # dV1/dSOC, dV1/dV1
        ])
        
        # 过程噪声协方差矩阵 Q
        # SOC的过程噪声：允许一定的不确定性，以便EKF能够修正初始误差和累积误差
        # V1的过程噪声：RC回路相对稳定，噪声较小
        Q = np.array([
            [self.process_noise, 0.0],  # SOC噪声（允许修正初始误差）
            [0.0, self.process_noise * 0.1]  # V1噪声较小
        ])
        
        # 预测协方差
        P_pred = F @ self.P @ F.T + Q
        
        # 更新步骤（测量更新）
        # 测量方程：V = OCV(SOC) + I*R0 + V1
        # 电流符号约定：正为充电，负为放电
        # 充电时(I>0)：端电压高于OCV；放电时(I<0)：端电压低于OCV
        ocv_pred = self._get_ocv(soc_pred)
        v_pred = ocv_pred + current * self.r0 + v1_pred
        
        # 测量残差
        y = voltage - v_pred
        
        # 测量雅可比矩阵 H = dh/dx
        # h(x) = OCV(SOC) + I*R0 + V1，故 dh/dV1 = 1
        docv_dsoc = self._get_docv_dsoc(soc_pred)
        H = np.array([
            [docv_dsoc, 1.0]  # dV/dSOC, dV/dV1
        ])
        
        # 测量噪声协方差（调整以匹配实际电压测量精度）
        # 电压测量噪声通常较小（mV级别），但需要平衡SOC修正能力
        # 如果测量噪声太小，EKF会过度信任电压测量，可能导致过度修正
        R = np.array([[self.measurement_noise]])  # 使用设定的测量噪声
        
        # 卡尔曼增益（添加数值稳定性检查）
        S = H @ P_pred @ H.T + R
        if np.abs(np.linalg.det(S)) < 1e-10:
            K = P_pred @ H.T / (S[0, 0] + 1e-6)
        else:
            K = P_pred @ H.T @ np.linalg.inv(S)
        
        # 状态更新（EKF的核心功能：通过电压测量修正SOC和V1）
        # EKF应该能够修正初始SOC误差和AH积分的累积误差
        # 对于初始SOC误差修正，需要允许适中的增益，避免过度修正
        # 限制增益大小以避免数值不稳定，但允许SOC修正
        # SOC增益：允许适中范围（修正初始误差，但不至于过度修正）
        # V1增益：限制较小范围（V1相对稳定）
        is_first_point = len(self.soc_history) == 0
        is_rest = abs(current) < 0.05
        
        # 初始化K_used，确保所有代码路径都定义了它
        K_used = None
        
        if is_first_point:
            # 第一个点：使用非常保守的修正策略，避免过度修正
            # 如果第一个点静止，使用OCV映射来初始化，而不是大幅修正
            if is_rest:
                # 静止时：使用OCV映射来初始化SOC，而不是大幅修正
                # 计算从OCV得到的SOC
                if self.ocv_to_soc_interp is not None:
                    try:
                        soc_from_ocv = float(self.ocv_to_soc_interp(voltage))
                        # 使用OCV映射的SOC和当前SOC的加权平均
                        # 第一个点静止时，更信任OCV映射（因为电压稳定）
                        # 权重：OCV 50%，当前SOC 50%（更激进的修正）
                        self.x[0] = 0.5 * soc_pred + 0.5 * soc_from_ocv
                        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
                        # V1正常更新
                        K_v1_clipped = np.clip(K[1, 0], -5, 5)
                        self.x[1] = v1_pred + K_v1_clipped * y
                        K_used = np.array([[0.0], [K_v1_clipped]])  # SOC不通过增益修正
                        # 设置K_clipped用于后续代码
                        K_clipped = K_used
                    except:
                        # 如果OCV映射失败，使用保守的增益
                        K_soc_clipped = np.clip(K[0, 0], -1, 1)  # 非常小的增益
                        K_v1_clipped = np.clip(K[1, 0], -5, 5)
                        K_clipped = np.array([[K_soc_clipped], [K_v1_clipped]])
                        self.x = x_pred + K_clipped @ np.array([y])
                        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
                        K_used = K_clipped
                else:
                    # 没有OCV映射，使用非常小的增益
                    K_soc_clipped = np.clip(K[0, 0], -1, 1)  # 非常小的增益
                    K_v1_clipped = np.clip(K[1, 0], -5, 5)
                    K_clipped = np.array([[K_soc_clipped], [K_v1_clipped]])
                    self.x = x_pred + K_clipped @ np.array([y])
                    self.x[0] = np.clip(self.x[0], 0.0, 1.0)
                    K_used = K_clipped
            else:
                # 第一个点非静止：使用保守的增益
                K_soc_clipped = np.clip(K[0, 0], -2, 2)  # 较小的增益
                K_v1_clipped = np.clip(K[1, 0], -5, 5)
                K_clipped = np.array([[K_soc_clipped], [K_v1_clipped]])
                self.x = x_pred + K_clipped @ np.array([y])
                self.x[0] = np.clip(self.x[0], 0.0, 1.0)
                K_used = K_clipped
        else:
            # 后续点：允许较大的修正
            # 根据误差大小动态调整增益限制
            # 如果误差大，允许更大的修正
            current_error = abs(y)  # 当前电压预测误差
            if current_error > 0.1:  # 误差大于0.1V，允许更大的修正
                K_soc_max = 30
            elif current_error > 0.05:  # 误差大于0.05V，允许中等修正
                K_soc_max = 20
            else:  # 误差较小，使用较小的修正
                K_soc_max = 10
            
            K_soc_clipped = np.clip(K[0, 0], -K_soc_max, K_soc_max)  # SOC增益动态调整
            K_v1_clipped = np.clip(K[1, 0], -10, 10)  # V1增益限制较小范围
            K_clipped = np.array([[K_soc_clipped], [K_v1_clipped]])
            self.x = x_pred + K_clipped @ np.array([y])
            self.x[0] = np.clip(self.x[0], 0.0, 1.0)  # 限制SOC范围
            K_used = K_clipped
        
        # 确保K_used已定义
        if K_used is None:
            # 默认情况：使用保守的增益
            K_soc_clipped = np.clip(K[0, 0], -5, 5)
            K_v1_clipped = np.clip(K[1, 0], -5, 5)
            K_clipped = np.array([[K_soc_clipped], [K_v1_clipped]])
            self.x = x_pred + K_clipped @ np.array([y])
            self.x[0] = np.clip(self.x[0], 0.0, 1.0)
            K_used = K_clipped
        
        K_used = K_clipped
        
        # 协方差更新（确保协方差矩阵正定）
        I_KH = np.eye(2) - K_used @ H
        self.P = I_KH @ P_pred
        # 确保协方差矩阵对称且正定
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(2) * 1e-6  # 添加小的正定项以确保数值稳定性
        
        self.last_current = current
        self.soc_history.append(self.x[0] * 100.0)  # 转换为百分比
        
        return self.x[0] * 100.0
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """批量估计SOC"""
        # 重置状态（使用设定的初始SOC，可能包含误差）
        self.x = np.array([self.initial_soc, 0.0])
        # 初始协方差：如果初始SOC有误差，应该设置较大的不确定性
        # 这样EKF能够更快地修正初始误差
        # 对于SOC：设置较大的初始不确定性（允许修正初始误差）
        # 对于V1：设置较小的初始不确定性（V1初始为0是合理的）
        initial_covariance_soc = 0.5  # 较大的初始SOC协方差，允许EKF修正初始误差
        initial_covariance_v1 = 0.01  # 较小的初始V1协方差
        self.P = np.array([
            [initial_covariance_soc, 0.0],
            [0.0, initial_covariance_v1]
        ])
        self.soc_history = []
        self.last_time = None
        self.last_current = 0.0
        
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        if time is None:
            time = np.arange(len(current))
        time = np.asarray(time)
        
        soc_estimated = []
        for i in range(len(voltage)):
            temp = temperature[i] if temperature is not None else None
            soc = self.update(voltage[i], current[i], time[i], temp)
            soc_estimated.append(soc)
        
        return np.array(soc_estimated)


class ParticleFilterSOC:
    """
    粒子滤波SOC估计器
    使用粒子滤波进行SOC估计，适合非线性系统
    """
    
    def __init__(self,
                 initial_soc=50.0,
                 nominal_capacity=1.1,
                 ocv_soc_table=None,
                 n_particles=100,
                 process_noise=0.01,
                 measurement_noise=0.01,
                 r0=0.01,
                 r1=0.005,
                 c1=1000.0):
        """
        初始化粒子滤波估计器
        
        Args:
            initial_soc: 初始SOC (%)
            nominal_capacity: 标称容量 (Ah)
            ocv_soc_table: OCV-SOC查找表
            n_particles: 粒子数量
            process_noise: 过程噪声标准差
            measurement_noise: 测量噪声标准差
            r0: 欧姆内阻 (Ohm)
            r1: RC回路电阻 (Ohm)
            c1: RC回路电容 (F)
        """
        self.nominal_capacity = nominal_capacity
        self.initial_soc = initial_soc / 100.0  # 转换为0-1范围
        self.n_particles = n_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # 等效电路模型参数
        self.r0 = r0
        self.r1 = r1
        self.c1 = c1
        self.tau1 = r1 * c1
        
        # OCV-SOC映射
        if ocv_soc_table is not None and len(ocv_soc_table) > 0:
            self.ocv_soc_table = np.array(ocv_soc_table)
            ocv_soc_soc = self.ocv_soc_table[:, 0] / 100.0
            ocv_soc_ocv = self.ocv_soc_table[:, 1]
            self.soc_to_ocv_interp = interp1d(
                ocv_soc_soc, ocv_soc_ocv,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
        else:
            self.ocv_soc_table = None
            self.soc_to_ocv_interp = None
        
        # 初始化粒子
        # 每个粒子：[SOC, V1]
        self.particles = np.zeros((n_particles, 2))
        self.weights = np.ones(n_particles) / n_particles
        
        # 初始化粒子SOC在初始值附近
        self.particles[:, 0] = np.random.normal(self.initial_soc, 0.05, n_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 1.0)
        self.particles[:, 1] = np.random.normal(0.0, 0.01, n_particles)
        
        self.last_time = None
        self.soc_history = []
    
    def _get_ocv(self, soc):
        """从SOC获取OCV"""
        if self.soc_to_ocv_interp is not None:
            soc_clipped = np.clip(soc, 0.0, 1.0)
            return self.soc_to_ocv_interp(soc_clipped)
        else:
            return 2.0 + soc * 1.6
    
    def update(self, voltage, current, time=None, temperature=None):
        """
        更新粒子滤波状态（单点输入）
        
        Args:
            voltage: 电压 (V)
            current: 电流 (A)
            time: 时间 (s)
            temperature: 温度 (°C)，可选
        
        Returns:
            当前SOC估计值 (%)
        """
        # 处理时间（关键修复：使用实际时间差）
        if time is None:
            if self.last_time is not None:
                dt = 1.0
            else:
                dt = 1.0
                self.last_time = 0.0
        else:
            if self.last_time is not None:
                dt = time - self.last_time
                if dt <= 0:
                    dt = 1.0  # 时间倒退或不变，使用默认值
                elif dt > 3600:  # 超过1小时，可能是数据跳跃
                    dt = 1.0
            else:
                dt = 1.0
            self.last_time = time
        
        # 预测步骤：更新每个粒子
        for i in range(self.n_particles):
            # SOC更新（修复：电流正值为充电，SOC增加）
            eta = 1.0  # 库伦效率（LFP电池接近1.0）
            delta_soc = current * dt / 3600 / self.nominal_capacity * eta  # 修复符号
            self.particles[i, 0] += delta_soc
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0.0, 1.0)
            
            # 添加过程噪声（降低噪声以匹配实际数据）
            self.particles[i, 0] += np.random.normal(0, self.process_noise * 0.1)  # 降低SOC噪声
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0.0, 1.0)
            
            # V1更新
            self.particles[i, 1] = (self.particles[i, 1] * np.exp(-dt / self.tau1) + 
                                    current * self.r1 * (1 - np.exp(-dt / self.tau1)))
            self.particles[i, 1] += np.random.normal(0, self.process_noise * 0.01)  # 降低V1噪声
        
        # 更新步骤：计算权重
        for i in range(self.n_particles):
            # 预测电压
            # 电流符号约定：正为充电，负为放电
            # 端电压 = OCV + I*R0 + V1
            # 充电时(I>0)：端电压高于OCV；放电时(I<0)：端电压低于OCV
            ocv = self._get_ocv(self.particles[i, 0])
            v_pred = ocv + current * self.r0 + self.particles[i, 1]
            
            # 计算似然（权重）- 修复：使用更稳定的计算方式避免数值问题
            error = voltage - v_pred
            # 使用更稳定的权重计算（避免exp(-inf)）
            log_weight = -0.5 * (error ** 2) / (self.measurement_noise ** 2)
            # 限制log_weight范围，避免数值溢出
            log_weight = np.clip(log_weight, -50, 50)
            self.weights[i] = np.exp(log_weight)
        
        # 归一化权重（添加数值稳定性检查）
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:  # 避免除零
            self.weights /= weight_sum
        else:
            # 如果权重和太小，重置为均匀分布
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # 重采样（如果有效粒子数太少）
        weight_sum_sq = np.sum(self.weights ** 2)
        if weight_sum_sq > 1e-10:  # 避免除零
            neff = 1.0 / weight_sum_sq
        else:
            neff = 0.0
        
        if neff < self.n_particles / 2:
            # 系统重采样（修复：确保cumsum正确）
            cumsum = np.cumsum(self.weights)
            # 确保cumsum最后一个元素为1（数值稳定性）
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
        
        # 估计SOC（加权平均）- 添加数值稳定性检查
        soc_est = np.sum(self.particles[:, 0] * self.weights)
        # 检查NaN和Inf
        if np.isnan(soc_est) or np.isinf(soc_est):
            # 如果出现NaN，使用粒子均值
            soc_est = np.mean(self.particles[:, 0])
        soc_est = np.clip(soc_est, 0.0, 1.0)
        self.soc_history.append(soc_est * 100.0)
        
        return soc_est * 100.0
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """批量估计SOC"""
        # 重置状态（确保初始SOC正确）
        self.particles = np.zeros((self.n_particles, 2))
        self.weights = np.ones(self.n_particles) / self.n_particles
        # 初始化粒子SOC在初始值附近（减小初始不确定性）
        self.particles[:, 0] = np.random.normal(self.initial_soc, 0.01, self.n_particles)  # 降低初始不确定性
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 1.0)
        self.particles[:, 1] = np.random.normal(0.0, 0.001, self.n_particles)  # 降低V1初始不确定性
        self.last_time = None
        self.soc_history = []
        
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        if time is None:
            time = np.arange(len(current))
        time = np.asarray(time)
        
        soc_estimated = []
        for i in range(len(voltage)):
            temp = temperature[i] if temperature is not None else None
            soc = self.update(voltage[i], current[i], time[i], temp)
            soc_estimated.append(soc)
        
        return np.array(soc_estimated)
