#!/usr/bin/env python3
"""
Complete test: train AI-GRU, save model, generate visualizations with key variables.
Usage:
  python quick_test.py           # Full 8-file test + train + save model
  python quick_test.py 3         # Quick 3-file test
  python quick_test.py --infer   # Load saved model + inference only (no training)
"""
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
from parameter_identifier import BatteryParameterIdentifier
from main import (EKFWithParameterIdentification, PFWithParameterIdentification,
                  calculate_soc_labels, load_and_preprocess_file)
from improved_ai_estimator import ImprovedAISOCEstimator
from pathlib import Path

t0 = timer.time()
RAW = Path("../raw_data")
DOCS = Path("../docs/results")
MODEL_DIR = Path("../saved_models")
DOCS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INFER_ONLY = '--infer' in sys.argv
args_list = [a for a in sys.argv[1:] if not a.startswith('--')]
MAX_FILES = int(args_list[0]) if args_list else 8

evaluator = SOCEvaluator()
processor = BatteryDataProcessor(data_dir=str(RAW))
ocv_builder = OCVCurveBuilder(ocv_data_dir=str(RAW))
ocv_builder.load_ocv_data(target_temperature=30, use_test_file=True)
ocv_soc_table = ocv_builder.get_ocv_soc_table()
actual_cap = ocv_builder.actual_discharge_capacity or 1.1

np.random.seed(42)
cap_est = actual_cap * (1 + np.random.uniform(-0.05, 0.05))

# Collect 1 file per temperature
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
        data['temp_label'] = f"{temp_str}C".replace('N','-')
        all_data.append(data)
        print(f"  {data['temp_label']:>6s}: {data['filename'][:35]}, bias={data['biased_init']-true_init:+.1f}%")

# === Traditional: AH+OCV ===
print(f"\n[{timer.time()-t0:.0f}s] Running AH+OCV...")
trad_results = {}
for data in all_data:
    est = RealtimeSOCEstimator(initial_soc=data['biased_init'], nominal_capacity=cap_est,
                               ocv_soc_table=ocv_soc_table, rest_current_threshold=0.05,
                               rest_duration_threshold=30.0)
    soc = est.estimate_batch(data['voltage'], data['current'], data['time'], data['temperature'])
    m = evaluator.evaluate(data['soc_true'], soc)
    trad_results[data['temp_label']] = {'soc': soc, 'maxerr': m['max_error'], 'mae': m['mae']}

# === RLS Parameter Identification (for visualization) ===
print(f"[{timer.time()-t0:.0f}s] Running RLS Parameter Identification...")
param_results = {}
for data in all_data:
    identifier = BatteryParameterIdentifier(
        initial_r0=0.05, initial_r1=0.03, initial_tau=30.0, forgetting_factor=0.995)
    v, c, t_arr = data['voltage'], data['current'], data['time']
    soc_values = ocv_soc_table[:, 0]
    ocv_values = ocv_soc_table[:, 1]
    soc_track = float(np.interp(v[0], ocv_values, soc_values))
    
    for i in range(len(v)):
        if i > 0:
            dt_i = t_arr[i] - t_arr[i-1]
            if dt_i <= 0 or dt_i > 100: dt_i = 1.0
            soc_track += c[i-1] * dt_i / 3600 / actual_cap * 100
            soc_track = np.clip(soc_track, 0, 100)
        ocv_i = float(np.interp(soc_track, soc_values, ocv_values))
        identifier.update(v[i], c[i], ocv_i, time=t_arr[i])
    
    param_results[data['temp_label']] = {
        'r0': np.array(identifier.r0_history),
        'r1': np.array(identifier.r1_history),
        'tau': np.array(identifier.tau_history),
        'v1': np.array(identifier.v1_history),
        'c1': np.array(identifier.tau_history) / (np.array(identifier.r1_history) + 1e-6),
    }

# === AI-GRU ===
if INFER_ONLY:
    print(f"[{timer.time()-t0:.0f}s] Loading saved AI-GRU model...")
    ai = ImprovedAISOCEstimator.load_model(str(MODEL_DIR))
else:
    print(f"[{timer.time()-t0:.0f}s] AI-GRU Training...")
    file_data_list = [{'voltage': d['voltage'], 'current': d['current'],
                       'time': d['time'], 'temperature': d['temperature'],
                       'soc_true': d['soc_true']} for d in all_data]
    
    ai = ImprovedAISOCEstimator(initial_soc=50.0, nominal_capacity=cap_est,
                                 sequence_length=20, hidden_size=128)
    ai.train_multi_file(file_data_list, epochs=100, batch_size=512, learning_rate=0.001)
    
    # Save the trained model
    ai.save_model(str(MODEL_DIR))

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
    print("ALL PASS! AI-GRU MaxErr < 5% on ALL files!")
else:
    n_fail = sum(1 for d in all_data if ai_results[d['temp_label']]['maxerr'] >= 5)
    print(f"FAIL: {n_fail} files above 5%")

# ================================================================
# === Generate Comprehensive Visualizations ===
# ================================================================
print(f"\n[{timer.time()-t0:.0f}s] Generating visualizations...")

n_files = len(all_data)

# ---- Plot 1: SOC Comparison (all temps) ----
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

# ---- Plot 2: Error Comparison ----
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
fig.suptitle('SOC Error: AH+OCV vs AI-GRU (+/-5% target zone)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(str(DOCS / 'error_comparison_all_temps.png'), dpi=120, bbox_inches='tight')
plt.close()

# ---- Plot 3: MaxErr Bar Chart ----
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

# ---- Plot 4: RLS Parameter Identification (R0, R1, Tau) ----
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, n_files))

for idx, data in enumerate(all_data):
    t = data['temp_label']
    pr = param_results[t]
    # param histories have N+1 entries (initial + N updates)
    n_pts = min(len(pr['r0']), len(data['time']))
    time_h = (data['time'][:n_pts] - data['time'][0]) / 3600
    r0_pts = pr['r0'][:n_pts]
    r1_pts = pr['r1'][:n_pts]
    tau_pts = pr['tau'][:n_pts]
    
    axes[0].plot(time_h, r0_pts * 1000, color=colors[idx], linewidth=1, label=t, alpha=0.8)
    axes[1].plot(time_h, r1_pts * 1000, color=colors[idx], linewidth=1, label=t, alpha=0.8)
    axes[2].plot(time_h, tau_pts, color=colors[idx], linewidth=1, label=t, alpha=0.8)

axes[0].set_ylabel('R0 (mOhm)')
axes[0].set_title('RLS Online Parameter Identification: R0 (Ohmic Resistance)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, ncol=4, loc='upper right')
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel('R1 (mOhm)')
axes[1].set_title('R1 (Polarization Resistance)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8, ncol=4, loc='upper right')
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel('Tau (s)')
axes[2].set_title('Tau = R1*C1 (Time Constant)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Time (hours)')
axes[2].legend(fontsize=8, ncol=4, loc='upper right')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Battery RC Model Parameters (RLS Identification, all temps)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(str(DOCS / 'parameter_identification.png'), dpi=120, bbox_inches='tight')
plt.close()

# ---- Plot 5: Training History ----
if hasattr(ai, 'training_history') and len(ai.training_history.get('train_loss', [])) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(ai.training_history['train_loss']) + 1)
    ax1.plot(epochs, ai.training_history['train_loss'], 'b-', linewidth=1, label='Train Loss')
    if ai.training_history.get('val_loss'):
        ax1.plot(epochs, ai.training_history['val_loss'], 'r-', linewidth=1, label='Val Loss')
        best_val = min(ai.training_history['val_loss'])
        best_ep = ai.training_history['val_loss'].index(best_val) + 1
        ax1.axvline(x=best_ep, color='green', linestyle='--', alpha=0.5, label=f'Best (ep={best_ep})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Per-file MAE bar
    file_labels = [d['temp_label'] for d in all_data]
    file_maes = [ai_results[t]['mae'] for t in file_labels]
    file_maxerrs = [ai_results[t]['maxerr'] for t in file_labels]
    x = np.arange(len(file_labels))
    bars = ax2.bar(x, file_maes, 0.4, label='MAE', color='#4CAF50', alpha=0.8)
    ax2.bar(x + 0.4, file_maxerrs, 0.4, label='MaxErr', color='#FF5722', alpha=0.8)
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='5% Target')
    ax2.set_xticks(x + 0.2)
    ax2.set_xticklabels(file_labels, fontsize=9)
    ax2.set_ylabel('Error (%)')
    ax2.set_title('AI-GRU: MAE & MaxErr per Temperature', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('AI-GRU Model Training & Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(DOCS / 'ai_training_history.png'), dpi=120, bbox_inches='tight')
    plt.close()

# ---- Plot 6: Per-file detail (SOC + error + params) for each temperature ----
for data in all_data:
    t = data['temp_label']
    time_h = (data['time'] - data['time'][0]) / 3600
    pr = param_results[t]
    n_pts = min(len(pr['r0']), len(data['time']))
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    # SOC
    ax = axes[0]
    ax.plot(time_h, data['soc_true'], 'k-', linewidth=1.5, label='True SOC', alpha=0.8)
    ax.plot(time_h, trad_results[t]['soc'], 'b--', linewidth=1, label=f"AH+OCV (MaxErr={trad_results[t]['maxerr']:.1f}%)", alpha=0.7)
    ax.plot(time_h, ai_results[t]['soc'], 'r-', linewidth=1.2, label=f"AI-GRU (MaxErr={ai_results[t]['maxerr']:.1f}%)", alpha=0.8)
    ax.set_ylabel('SOC (%)')
    ax.set_title(f'{t}: SOC Estimation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    # Voltage + Current
    ax = axes[1]
    ax.plot(time_h, data['voltage'], 'b-', linewidth=0.8, alpha=0.8, label='Voltage (V)')
    ax.set_ylabel('Voltage (V)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    ax2.plot(time_h, data['current'], 'g-', linewidth=0.6, alpha=0.6, label='Current (A)')
    ax2.set_ylabel('Current (A)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax.set_title(f'{t}: Voltage & Current', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # R0 + R1
    ax = axes[2]
    ax.plot(time_h[:n_pts], pr['r0'][:n_pts]*1000, 'b-', linewidth=1, label='R0 (mOhm)', alpha=0.8)
    ax.plot(time_h[:n_pts], pr['r1'][:n_pts]*1000, 'r-', linewidth=1, label='R1 (mOhm)', alpha=0.8)
    ax.set_ylabel('Resistance (mOhm)')
    ax.set_title(f'{t}: R0 & R1 (RLS)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.85, f'R0={pr["r0"][-1]*1000:.1f}mOhm, R1={pr["r1"][-1]*1000:.1f}mOhm',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Tau + C1
    ax = axes[3]
    ax.plot(time_h[:n_pts], pr['tau'][:n_pts], 'purple', linewidth=1, label='Tau (s)', alpha=0.8)
    ax.set_ylabel('Tau (s)', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax2 = ax.twinx()
    ax2.plot(time_h[:n_pts], pr['c1'][:n_pts], 'orange', linewidth=0.8, label='C1 (F)', alpha=0.7)
    ax2.set_ylabel('C1 (F)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    ax.set_title(f'{t}: Tau & C1 (RLS)', fontsize=11)
    ax.set_xlabel('Time (hours)')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.85, f'Tau={pr["tau"][-1]:.1f}s, C1={pr["c1"][-1]:.0f}F',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    fig.suptitle(f'{t}: Complete SOC Estimation + Parameter Identification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(DOCS / f'detail_{t.replace("-","N")}.png'), dpi=120, bbox_inches='tight')
    plt.close()

print(f"[{timer.time()-t0:.0f}s] Visualizations saved to docs/results/")
print(f"  - soc_comparison_all_temps.png")
print(f"  - error_comparison_all_temps.png")
print(f"  - maxerr_bar_chart.png")
print(f"  - parameter_identification.png (R0/R1/Tau for all temps)")
print(f"  - ai_training_history.png")
print(f"  - detail_{{temp}}.png (per-file: SOC + V/I + R0/R1 + Tau/C1)")
if not INFER_ONLY:
    print(f"  - saved_models/soc_gru_model.pth + _config.json")
print(f"\nTotal time: {timer.time()-t0:.0f}s")
