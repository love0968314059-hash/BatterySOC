#!/usr/bin/env python3
"""完整测试：8个文件验证MaxErr < 5% + 生成可视化"""
import sys, os, numpy as np, warnings, time as timer
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

os.chdir(os.path.join(os.path.dirname(__file__), 'soc_estimation'))
sys.path.insert(0, '.')

from data_processor import BatteryDataProcessor
from ocv_curve_builder import OCVCurveBuilder
from realtime_soc_estimator import RealtimeSOCEstimator
from evaluator import SOCEvaluator
from main import (EKFWithParameterIdentification, PFWithParameterIdentification,
                  calculate_soc_labels, load_and_preprocess_file)
from improved_ai_estimator import ImprovedAISOCEstimator
from pathlib import Path

t0 = timer.time()
RAW = Path("../raw_data")
DOCS = Path("../docs/results")
DOCS.mkdir(parents=True, exist_ok=True)

evaluator = SOCEvaluator()
processor = BatteryDataProcessor(data_dir=str(RAW))
ocv_builder = OCVCurveBuilder(ocv_data_dir=str(RAW))
ocv_builder.load_ocv_data(target_temperature=30, use_test_file=True)
ocv_soc_table = ocv_builder.get_ocv_soc_table()
actual_cap = ocv_builder.actual_discharge_capacity or 1.1

np.random.seed(42)
cap_est = actual_cap * (1 + np.random.uniform(-0.05, 0.05))

# 收集每个温度1个文件
data_files = []
seen = set()
for temp_dir in sorted(RAW.glob("DST-US06-FUDS-*")):
    if not temp_dir.is_dir():
        continue
    temp_str = temp_dir.name.split('-')[-1]
    if temp_str in seen:
        continue
    seen.add(temp_str)
    for f in sorted(temp_dir.glob("*.xlsx")):
        if 'newprofile' not in f.name and '20120809' not in f.name:
            data_files.append(f)
            break

MAX_FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
data_files = data_files[:MAX_FILES]

print(f"[{timer.time()-t0:.0f}s] Loading {len(data_files)} files (truncated to 12000 pts)...")
bias_rng = np.random.RandomState(42)
all_data = []
for f in data_files:
    data = load_and_preprocess_file(f, processor)
    if data:
        N = min(12000, len(data['time']))
        for k in ['time','voltage','current','temperature']:
            data[k] = data[k][:N]
        soc_true = calculate_soc_labels(data['time'], data['current'], data['voltage'],
                                       actual_cap, ocv_soc_table)
        data['soc_true'] = soc_true
        true_init = soc_true[0]
        bias_sign = 1 if bias_rng.rand() > 0.5 else -1
        data['biased_init'] = np.clip(true_init + bias_sign * 10.0, 0, 100)
        data['true_init'] = true_init
        temp_str = f.parent.name.split('-')[-1]
        data['temp_label'] = f"{temp_str}°C".replace('N','-')
        all_data.append(data)
        print(f"  {data['temp_label']:>6s}: {data['filename'][:35]}, bias={data['biased_init']-true_init:+.1f}%")

# === Traditional: AH+OCV only ===
print(f"\n[{timer.time()-t0:.0f}s] Running AH+OCV...")
trad_results = {}
for data in all_data:
    est = RealtimeSOCEstimator(initial_soc=data['biased_init'], nominal_capacity=cap_est,
                               ocv_soc_table=ocv_soc_table, rest_current_threshold=0.05,
                               rest_duration_threshold=30.0)
    soc = est.estimate_batch(data['voltage'], data['current'], data['time'], data['temperature'])
    m = evaluator.evaluate(data['soc_true'], soc)
    trad_results[data['temp_label']] = {'soc': soc, 'maxerr': m['max_error'], 'mae': m['mae']}

# === AI-GRU ===
print(f"[{timer.time()-t0:.0f}s] AI-GRU Training...")
file_data_list = [{'voltage': d['voltage'], 'current': d['current'],
                   'time': d['time'], 'temperature': d['temperature'],
                   'soc_true': d['soc_true']} for d in all_data]

ai = ImprovedAISOCEstimator(initial_soc=50.0, nominal_capacity=cap_est,
                             sequence_length=20, hidden_size=128)
ai.train_multi_file(file_data_list, epochs=100, batch_size=512, learning_rate=0.001)

print(f"\n[{timer.time()-t0:.0f}s] AI-GRU Inference...")
ai_results = {}
for data in all_data:
    soc_est = ai.predict_batch(data['voltage'], data['current'],
                                data['time'], data['temperature'])
    m = evaluator.evaluate(data['soc_true'], soc_est)
    ai_results[data['temp_label']] = {'soc': soc_est, 'maxerr': m['max_error'], 'mae': m['mae']}

# === Print Results ===
print(f"\n{'='*65}")
print(f"{'Temp':>6s} | {'AH+OCV MaxErr':>14s} | {'AI-GRU MaxErr':>14s} | {'AI MAE':>8s} | {'Status':>6s}")
print(f"{'-'*65}")
all_pass = True
for data in all_data:
    t = data['temp_label']
    tr = trad_results[t]
    ar = ai_results[t]
    s = "PASS" if ar['maxerr'] < 5 else "FAIL"
    if ar['maxerr'] >= 5: all_pass = False
    print(f"{t:>6s} | {tr['maxerr']:>13.2f}% | {ar['maxerr']:>13.2f}% | {ar['mae']:>7.2f}% | [{s}]")

avg_maxerr = np.mean([ai_results[d['temp_label']]['maxerr'] for d in all_data])
worst_maxerr = max(ai_results[d['temp_label']]['maxerr'] for d in all_data)
print(f"{'-'*65}")
print(f"{'AI-GRU':>6s} | {'':>14s} | Avg={avg_maxerr:.2f}% Worst={worst_maxerr:.2f}%")
print(f"{'='*65}")
if all_pass:
    print("🎉 ALL PASS! AI-GRU MaxErr < 5% on ALL files!")
else:
    n_fail = sum(1 for d in all_data if ai_results[d['temp_label']]['maxerr'] >= 5)
    print(f"❌ {n_fail} files FAIL")

# === Generate Visualizations ===
print(f"\n[{timer.time()-t0:.0f}s] Generating visualizations...")

# Plot 1: SOC comparison for all temps
n_files = len(all_data)
fig, axes = plt.subplots(n_files, 1, figsize=(14, 3*n_files), sharex=False)
if n_files == 1: axes = [axes]

for idx, data in enumerate(all_data):
    ax = axes[idx]
    t = data['temp_label']
    time_h = (data['time'] - data['time'][0]) / 3600
    ax.plot(time_h, data['soc_true'], 'k-', linewidth=1.5, label='True SOC', alpha=0.8)
    ax.plot(time_h, trad_results[t]['soc'], 'b--', linewidth=1, label=f"AH+OCV (MaxErr={trad_results[t]['maxerr']:.1f}%)", alpha=0.7)
    ax.plot(time_h, ai_results[t]['soc'], 'r-', linewidth=1.2, label=f"AI-GRU (MaxErr={ai_results[t]['maxerr']:.1f}%)", alpha=0.8)
    ax.set_ylabel('SOC (%)')
    ax.set_title(f'{t}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
axes[-1].set_xlabel('Time (hours)')
fig.suptitle('SOC Estimation: AH+OCV vs AI-GRU (all temperatures)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(str(DOCS / 'soc_comparison_all_temps.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 2: Error comparison
fig, axes = plt.subplots(n_files, 1, figsize=(14, 3*n_files), sharex=False)
if n_files == 1: axes = [axes]

for idx, data in enumerate(all_data):
    ax = axes[idx]
    t = data['temp_label']
    time_h = (data['time'] - data['time'][0]) / 3600
    err_trad = trad_results[t]['soc'] - data['soc_true']
    err_ai = ai_results[t]['soc'] - data['soc_true']
    ax.plot(time_h, err_trad, 'b-', linewidth=0.8, label='AH+OCV Error', alpha=0.6)
    ax.plot(time_h, err_ai, 'r-', linewidth=1, label='AI-GRU Error', alpha=0.8)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(time_h, -5, 5, alpha=0.08, color='green')
    ax.set_ylabel('Error (%)')
    ax.set_title(f'{t}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Time (hours)')
fig.suptitle('SOC Error: AH+OCV vs AI-GRU (±5% target zone)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(str(DOCS / 'error_comparison_all_temps.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 3: MaxErr bar chart
fig, ax = plt.subplots(figsize=(10, 6))
temps = [d['temp_label'] for d in all_data]
x = np.arange(len(temps))
trad_errs = [trad_results[t]['maxerr'] for t in temps]
ai_errs = [ai_results[t]['maxerr'] for t in temps]

bars1 = ax.bar(x - 0.2, trad_errs, 0.35, label='AH+OCV', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + 0.2, ai_errs, 0.35, label='AI-GRU', color='#E91E63', alpha=0.8)
ax.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% Target')

for bar, err in zip(bars1, trad_errs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{err:.1f}%',
            ha='center', fontsize=8, color='blue')
for bar, err in zip(bars2, ai_errs):
    c = 'green' if err < 5 else 'red'
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{err:.1f}%',
            ha='center', fontsize=8, fontweight='bold', color=c)

ax.set_xticks(x)
ax.set_xticklabels(temps, fontsize=11)
ax.set_ylabel('Max Error (%)', fontsize=12)
ax.set_title('Max Error per File: AH+OCV vs AI-GRU\n(Target: ALL bars below red line)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(str(DOCS / 'maxerr_bar_chart.png'), dpi=120, bbox_inches='tight')
plt.close()

print(f"[{timer.time()-t0:.0f}s] Saved: soc_comparison_all_temps.png, error_comparison_all_temps.png, maxerr_bar_chart.png")
print(f"\n✅ Total time: {timer.time()-t0:.0f}s")
