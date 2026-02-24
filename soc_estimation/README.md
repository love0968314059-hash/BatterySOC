# SOC估计系统

基于LFP电池的SOC(State of Charge)估计算法集成系统。

## 核心方法及性能

| 方法 | MAE | RMSE | Max Error | 说明 |
|------|-----|------|-----------|------|
| **AI (GRU)** | **0.44%** | 0.55% | 1.88% | 深度学习，精度最高 |
| 实时AH+OCV | 1.85% | 2.27% | 4.76% | 基准方法，简单可靠 |
| 改进EKF | 3.20% | 3.56% | 6.07% | 可从±10%误差收敛 |
| 粒子滤波 | 3.65% | 3.94% | 8.44% | 非线性滤波 |

所有方法均达到MAE<5%目标。

## 文件结构

```
soc_estimation/
├── main.py                    # 主程序入口
├── test_improved_methods.py   # 测试脚本
├── data_processor.py          # 数据加载和预处理
├── data_resampler.py          # 数据等间隔重采样
├── ocv_curve_builder.py       # OCV-SOC曲线构建
├── realtime_soc_estimator.py  # 实时AH积分+OCV校准
├── advanced_soc_estimators.py # EKF和粒子滤波
├── improved_ekf_estimator.py  # 改进的EKF（自适应增益）
├── improved_ai_estimator.py   # AI方法（GRU网络）
├── evaluator.py               # 评估指标计算
├── visualizer.py              # 结果可视化
└── README.md                  # 本文件
```

## 核心算法

### 1. 数据预处理流程

```python
# 1. 加载原始数据
# 2. 等间隔重采样（异常间隔电流填0，电压插值）
# 3. 重新计算SOC标签（基于AH积分从参考点倒推）
```

### 2. 改进的EKF

针对LFP电池平坦OCV曲线的特点：
- 只在SOC<10%或>90%的陡峭区域使用电压修正SOC
- 在平坦区域完全依赖AH积分
- 可从±10%初始误差收敛到正确值

### 3. AI方法 (GRU)

- 6维特征：电压、电流、温度、时间差、累积AH、功率
- 序列长度：15
- 自动训练和预测

## 使用方法

### 快速测试
```bash
cd soc_estimation
python3 test_improved_methods.py
```

### 完整运行
```bash
python3 main.py
```

### API使用
```python
from improved_ekf_estimator import ImprovedEKFSOCEstimator
from improved_ai_estimator import ImprovedAISOCEstimator

# EKF估计
ekf = ImprovedEKFSOCEstimator(
    initial_soc=50.0,
    nominal_capacity=1.1,
    ocv_soc_table=ocv_table
)
soc_est = ekf.estimate_batch(voltage, current, time, temperature)

# AI估计
ai = ImprovedAISOCEstimator(
    initial_soc=50.0,
    nominal_capacity=1.1,
    sequence_length=15
)
soc_est = ai.estimate_batch(voltage, current, time, temperature, soc_true)
```

## 依赖

```
numpy
scipy
pandas
matplotlib
torch (可选，用于AI方法)
```

## 更新日志

### 2026-01-27
- 改进EKF自适应增益策略，可从大误差收敛
- 优化AI方法，MAE达到0.44%
- 整合代码，删除冗余文件
- 数据处理：先重采样再计算SOC标签
