#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池参数在线辨识模块
使用递归最小二乘法(RLS)实时辨识RC等效电路参数
"""

import numpy as np


class BatteryParameterIdentifier:
    """
    电池RC参数在线辨识器
    
    使用递归最小二乘法(RLS)辨识一阶RC等效电路参数：
    - R0: 欧姆内阻
    - R1: 极化电阻
    - C1: 极化电容
    
    模型方程：
    V_terminal = OCV(SOC) + I*R0 + V1
    V1' = -V1/(R1*C1) + I/C1
    
    离散化后：
    V1[k] = exp(-dt/tau)*V1[k-1] + R1*(1-exp(-dt/tau))*I[k]
    其中 tau = R1*C1
    """
    
    def __init__(self, initial_r0=0.05, initial_r1=0.03, initial_tau=30.0,
                 forgetting_factor=0.995, initial_p=1000.0):
        """
        初始化参数辨识器
        
        Args:
            initial_r0: R0初始值 (Ohm)
            initial_r1: R1初始值 (Ohm)
            initial_tau: tau初始值 (s), tau = R1*C1
            forgetting_factor: 遗忘因子 (0.9-0.999), 越大记忆越长
            initial_p: 协方差矩阵初始值
        """
        # 参数估计值: [R0, R1, tau]
        self.theta = np.array([initial_r0, initial_r1, initial_tau])
        
        # 协方差矩阵（3x3）
        self.P = np.eye(3) * initial_p
        
        # 遗忘因子
        self.lambda_ = forgetting_factor
        
        # 上一时刻的V1估计
        self.v1_est = 0.0
        
        # 历史记录
        self.r0_history = [initial_r0]
        self.r1_history = [initial_r1]
        self.tau_history = [initial_tau]
        self.v1_history = [0.0]
        
        # 参数约束
        self.r0_min, self.r0_max = 0.01, 0.2
        self.r1_min, self.r1_max = 0.005, 0.1
        self.tau_min, self.tau_max = 5.0, 200.0
        
        # 辨识控制
        self.last_time = None
        self.n_updates = 0
        self.excitation_threshold = 0.1  # 电流激励阈值
    
    @property
    def r0(self):
        return self.theta[0]
    
    @property
    def r1(self):
        return self.theta[1]
    
    @property
    def tau(self):
        return self.theta[2]
    
    @property
    def c1(self):
        """C1 = tau / R1"""
        return self.tau / (self.r1 + 1e-6)
    
    def get_params(self):
        """获取当前辨识的参数"""
        return {
            'r0': self.r0,
            'r1': self.r1,
            'c1': self.c1,
            'tau': self.tau
        }
    
    def update(self, voltage, current, ocv, time=None, dt=None):
        """
        更新参数估计
        
        Args:
            voltage: 端电压测量值 (V)
            current: 电流测量值 (A), 正为充电
            ocv: OCV估计值 (V), 基于当前SOC估计
            time: 时间戳 (s)
            dt: 时间间隔 (s), 如果提供则优先使用
            
        Returns:
            dict: 更新后的参数
        """
        # 计算时间间隔
        if dt is None:
            if time is not None and self.last_time is not None:
                dt = time - self.last_time
                if dt <= 0 or dt > 100:
                    dt = 1.0
            else:
                dt = 1.0
        
        if time is not None:
            self.last_time = time
        
        # 检查电流激励是否足够
        # 如果电流太小，参数辨识可能不准确
        current_abs = abs(current)
        
        # 预测V1
        exp_factor = np.exp(-dt / (self.tau + 1e-6))
        v1_pred = self.v1_est * exp_factor + current * self.r1 * (1 - exp_factor)
        
        # 预测端电压
        v_pred = ocv + current * self.r0 + v1_pred
        
        # 电压误差（新息）
        error = voltage - v_pred
        
        # 只在有足够电流激励时更新参数
        if current_abs > self.excitation_threshold:
            # 构建回归向量 phi
            # V = OCV + I*R0 + V1
            # V1[k] = exp(-dt/tau)*V1[k-1] + R1*(1-exp(-dt/tau))*I
            # 
            # 对参数求偏导：
            # dV/dR0 = I
            # dV/dR1 = (1 - exp(-dt/tau)) * I
            # dV/dtau = V1[k-1] * (dt/tau^2) * exp(-dt/tau) - R1 * I * (dt/tau^2) * exp(-dt/tau)
            
            phi = np.zeros(3)
            phi[0] = current  # dV/dR0
            phi[1] = (1 - exp_factor) * current  # dV/dR1
            
            # dV/dtau 较复杂
            dtau_factor = (dt / (self.tau**2 + 1e-6)) * exp_factor
            phi[2] = self.v1_est * dtau_factor - self.r1 * current * dtau_factor
            
            # RLS更新
            # K = P * phi / (lambda + phi' * P * phi)
            # theta = theta + K * error
            # P = (P - K * phi' * P) / lambda
            
            Pphi = self.P @ phi
            denom = self.lambda_ + phi @ Pphi
            
            if abs(denom) > 1e-10:
                K = Pphi / denom
                
                # 更新参数估计
                self.theta = self.theta + K * error
                
                # 更新协方差矩阵
                self.P = (self.P - np.outer(K, phi @ self.P)) / self.lambda_
                
                # 确保协方差矩阵正定
                self.P = 0.5 * (self.P + self.P.T)
                eigvals = np.linalg.eigvalsh(self.P)
                if np.min(eigvals) < 1e-6:
                    self.P += np.eye(3) * (1e-6 - np.min(eigvals) + 1e-6)
                
                self.n_updates += 1
        
        # 参数约束
        self.theta[0] = np.clip(self.theta[0], self.r0_min, self.r0_max)
        self.theta[1] = np.clip(self.theta[1], self.r1_min, self.r1_max)
        self.theta[2] = np.clip(self.theta[2], self.tau_min, self.tau_max)
        
        # 更新V1估计（使用新参数）
        exp_factor_new = np.exp(-dt / (self.tau + 1e-6))
        self.v1_est = self.v1_est * exp_factor_new + current * self.r1 * (1 - exp_factor_new)
        
        # 记录历史
        self.r0_history.append(self.r0)
        self.r1_history.append(self.r1)
        self.tau_history.append(self.tau)
        self.v1_history.append(self.v1_est)
        
        return self.get_params()
    
    def reset(self, initial_r0=None, initial_r1=None, initial_tau=None):
        """重置辨识器"""
        if initial_r0 is not None:
            self.theta[0] = initial_r0
        if initial_r1 is not None:
            self.theta[1] = initial_r1
        if initial_tau is not None:
            self.theta[2] = initial_tau
        
        self.P = np.eye(3) * 1000.0
        self.v1_est = 0.0
        self.last_time = None
        self.n_updates = 0
        
        self.r0_history = [self.r0]
        self.r1_history = [self.r1]
        self.tau_history = [self.tau]
        self.v1_history = [0.0]
    
    def get_statistics(self):
        """获取辨识统计信息"""
        return {
            'n_updates': self.n_updates,
            'r0_mean': np.mean(self.r0_history[-1000:]) if len(self.r0_history) > 0 else self.r0,
            'r1_mean': np.mean(self.r1_history[-1000:]) if len(self.r1_history) > 0 else self.r1,
            'tau_mean': np.mean(self.tau_history[-1000:]) if len(self.tau_history) > 0 else self.tau,
            'r0_std': np.std(self.r0_history[-1000:]) if len(self.r0_history) > 100 else 0,
            'r1_std': np.std(self.r1_history[-1000:]) if len(self.r1_history) > 100 else 0,
            'tau_std': np.std(self.tau_history[-1000:]) if len(self.tau_history) > 100 else 0,
        }


class JointStateParameterEKF:
    """
    联合状态-参数EKF估计器
    
    同时估计SOC状态和电池参数(R0, R1, tau)
    状态向量: [SOC, V1, R0, R1, tau]
    """
    
    def __init__(self, initial_soc, nominal_capacity, ocv_soc_table=None,
                 initial_r0=0.05, initial_r1=0.03, initial_tau=30.0):
        """
        初始化联合估计器
        
        Args:
            initial_soc: 初始SOC (0-100)
            nominal_capacity: 额定容量 (Ah)
            ocv_soc_table: OCV-SOC查找表
            initial_r0: R0初始值
            initial_r1: R1初始值
            initial_tau: tau初始值
        """
        self.nominal_capacity = nominal_capacity
        self.ocv_soc_table = ocv_soc_table
        
        # 状态向量: [SOC, V1, R0, R1, tau]
        self.x = np.array([
            initial_soc / 100.0,  # SOC (0-1)
            0.0,                   # V1
            initial_r0,            # R0
            initial_r1,            # R1
            initial_tau            # tau
        ])
        
        # 协方差矩阵
        self.P = np.diag([0.01, 0.001, 0.001, 0.001, 10.0])
        
        # 过程噪声
        self.Q = np.diag([1e-6, 1e-6, 1e-8, 1e-8, 1e-4])
        
        # 测量噪声
        self.R = np.array([[0.001]])
        
        # 参数约束
        self.param_bounds = {
            'soc': (0.0, 1.0),
            'v1': (-0.5, 0.5),
            'r0': (0.01, 0.2),
            'r1': (0.005, 0.1),
            'tau': (5.0, 200.0)
        }
        
        self.last_time = None
        self.soc_history = []
        self.r0_history = []
        self.r1_history = []
        self.tau_history = []
        self.innovation_history = []
    
    def _get_ocv(self, soc):
        """获取OCV"""
        if self.ocv_soc_table is None:
            # 默认LFP OCV曲线
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
        ocv_high = self._get_ocv(soc_high)
        ocv_low = self._get_ocv(soc_low)
        return (ocv_high - ocv_low) / (soc_high - soc_low + 1e-10)
    
    def update(self, voltage, current, time=None, dt=None):
        """
        EKF更新步骤
        
        Returns:
            float: SOC估计值 (%)
        """
        # 计算时间间隔
        if dt is None:
            if time is not None and self.last_time is not None:
                dt = time - self.last_time
                if dt <= 0 or dt > 100:
                    dt = 1.0
            else:
                dt = 1.0
        
        if time is not None:
            self.last_time = time
        
        # 当前状态
        soc, v1, r0, r1, tau = self.x
        
        # === 预测步骤 ===
        
        # SOC预测
        delta_soc = current * dt / 3600 / self.nominal_capacity
        soc_pred = np.clip(soc + delta_soc, 0.0, 1.0)
        
        # V1预测
        exp_factor = np.exp(-dt / (tau + 1e-6))
        v1_pred = v1 * exp_factor + current * r1 * (1 - exp_factor)
        
        # 参数预测（随机游走模型，参数不变）
        r0_pred = r0
        r1_pred = r1
        tau_pred = tau
        
        x_pred = np.array([soc_pred, v1_pred, r0_pred, r1_pred, tau_pred])
        
        # 状态转移雅可比矩阵 F (5x5)
        F = np.eye(5)
        F[1, 1] = exp_factor
        F[1, 3] = (1 - exp_factor) * current  # dV1/dR1
        F[1, 4] = (dt / tau**2) * exp_factor * (v1 - current * r1)  # dV1/dtau
        
        # 预测协方差
        P_pred = F @ self.P @ F.T + self.Q
        
        # === 更新步骤 ===
        
        # 预测电压
        ocv_pred = self._get_ocv(soc_pred)
        v_pred = ocv_pred + current * r0_pred + v1_pred
        
        # 观测雅可比矩阵 H (1x5)
        docv_dsoc = self._get_docv_dsoc(soc_pred)
        H = np.array([[docv_dsoc, 1.0, current, 0.0, 0.0]])
        
        # 新息
        innovation = voltage - v_pred
        self.innovation_history.append(innovation)
        
        # 卡尔曼增益
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T / (S[0, 0] + 1e-10)
        
        # 状态更新
        self.x = x_pred + K.flatten() * innovation
        
        # 协方差更新
        I_KH = np.eye(5) - np.outer(K, H)
        self.P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.R[0, 0]
        
        # 参数约束
        self.x[0] = np.clip(self.x[0], *self.param_bounds['soc'])
        self.x[1] = np.clip(self.x[1], *self.param_bounds['v1'])
        self.x[2] = np.clip(self.x[2], *self.param_bounds['r0'])
        self.x[3] = np.clip(self.x[3], *self.param_bounds['r1'])
        self.x[4] = np.clip(self.x[4], *self.param_bounds['tau'])
        
        # 记录历史
        soc_percent = self.x[0] * 100.0
        self.soc_history.append(soc_percent)
        self.r0_history.append(self.x[2])
        self.r1_history.append(self.x[3])
        self.tau_history.append(self.x[4])
        
        return soc_percent
    
    def estimate_batch(self, voltage, current, time=None, temperature=None):
        """批量估计"""
        self.soc_history = []
        self.r0_history = []
        self.r1_history = []
        self.tau_history = []
        self.innovation_history = []
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
    
    def get_params(self):
        """获取当前参数估计"""
        return {
            'r0': self.x[2],
            'r1': self.x[3],
            'tau': self.x[4],
            'c1': self.x[4] / (self.x[3] + 1e-6)
        }
    
    def get_diagnostics(self):
        """获取诊断信息"""
        return {
            'final_soc': self.x[0] * 100.0,
            'final_r0': self.x[2],
            'final_r1': self.x[3],
            'final_tau': self.x[4],
            'innovation_mean': np.mean(self.innovation_history) if self.innovation_history else 0,
            'innovation_std': np.std(self.innovation_history) if self.innovation_history else 0,
            'r0_history': self.r0_history,
            'r1_history': self.r1_history,
            'tau_history': self.tau_history
        }
