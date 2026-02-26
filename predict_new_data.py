#!/usr/bin/env python3
"""
SOC Prediction on New Data (using pre-trained AI-GRU model)

Usage:
  # Predict on a single Excel file:
  python predict_new_data.py --data path/to/data.xlsx --model saved_models/soc_gru_model

  # Predict on a directory of files:
  python predict_new_data.py --data path/to/dir/ --model saved_models/soc_gru_model

  # With known true SOC for evaluation:
  python predict_new_data.py --data path/to/data.xlsx --model saved_models/soc_gru_model --eval

  # Train a new model from data:
  python predict_new_data.py --data path/to/dir/ --train --save-model saved_models/my_model
"""

import sys, os, argparse, numpy as np, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOC_DIR = os.path.join(SCRIPT_DIR, 'soc_estimation')
sys.path.insert(0, SOC_DIR)

from data_processor import BatteryDataProcessor
from ocv_curve_builder import OCVCurveBuilder
from realtime_soc_estimator import RealtimeSOCEstimator
from evaluator import SOCEvaluator
from improved_ai_estimator import ImprovedAISOCEstimator
from parameter_identifier import BatteryParameterIdentifier
from pathlib import Path


def load_data_file(filepath, processor=None):
    """Load a single data file and return dict with voltage/current/time/temperature"""
    filepath = Path(filepath)
    
    if processor is None:
        processor = BatteryDataProcessor(data_dir=str(filepath.parent))
    
    print(f"  Loading: {filepath.name}")
    data = processor.load_data_file(str(filepath))
    if data is None:
        raise ValueError(f"Failed to load: {filepath}")
    
    result = {
        'filename': filepath.name,
        'filepath': str(filepath),
        'time': np.array(data['time'].values, dtype=float),
        'voltage': np.array(data['voltage'].values, dtype=float),
        'current': np.array(data['current'].values, dtype=float),
        'temperature': np.array(data['temperature'].values, dtype=float),
    }
    
    # Check for SOC column
    if 'soc' in data.columns:
        result['soc_true'] = np.array(data['soc'].values, dtype=float)
    
    return result


def calculate_soc_labels(time, current, voltage, capacity, ocv_soc_table):
    """Calculate true SOC from data using AH integration calibrated by OCV"""
    n = len(time)
    dt = np.diff(time)
    dt = np.concatenate([[1.0], dt])
    dt = np.clip(dt, 0.1, 100.0)
    
    # Get initial SOC from OCV lookup
    soc_values = ocv_soc_table[:, 0]
    ocv_values = ocv_soc_table[:, 1]
    initial_soc = float(np.interp(voltage[0], ocv_values, soc_values))
    
    soc = np.zeros(n)
    soc[0] = initial_soc
    for i in range(1, n):
        delta = current[i-1] * dt[i] / 3600 / capacity * 100
        soc[i] = np.clip(soc[i-1] + delta, 0, 100)
    return soc


def run_parameter_identification(data, ocv_soc_table):
    """Run RLS parameter identification and return R0/R1/Tau histories"""
    identifier = BatteryParameterIdentifier(
        initial_r0=0.05, initial_r1=0.03, initial_tau=30.0,
        forgetting_factor=0.995
    )
    
    voltage = data['voltage']
    current = data['current']
    time = data['time']
    
    soc_values = ocv_soc_table[:, 0]
    ocv_values = ocv_soc_table[:, 1]
    
    # Simple SOC tracking for OCV lookup
    n = len(voltage)
    soc = np.interp(voltage[0], ocv_values, soc_values)
    
    r0_hist, r1_hist, tau_hist, v1_hist = [], [], [], []
    
    for i in range(n):
        if i > 0:
            dt_i = time[i] - time[i-1]
            if dt_i <= 0 or dt_i > 100:
                dt_i = 1.0
            soc += current[i-1] * dt_i / 3600 / 1.1 * 100
            soc = np.clip(soc, 0, 100)
        
        ocv_i = float(np.interp(soc, soc_values, ocv_values))
        identifier.update(voltage[i], current[i], ocv_i, time=time[i])
        
        r0_hist.append(identifier.r0)
        r1_hist.append(identifier.r1)
        tau_hist.append(identifier.tau)
        v1_hist.append(identifier.v1_est)
    
    return {
        'r0': np.array(r0_hist),
        'r1': np.array(r1_hist),
        'tau': np.array(tau_hist),
        'v1': np.array(v1_hist),
        'c1': np.array(tau_hist) / (np.array(r1_hist) + 1e-6),
    }


def plot_results(data, soc_est, params, output_path, title="SOC Estimation Results",
                 soc_true=None, training_history=None):
    """Generate comprehensive visualization of results"""
    time_h = (data['time'] - data['time'][0]) / 3600
    
    n_rows = 5 if training_history else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows))
    
    # --- Row 1: SOC ---
    ax = axes[0]
    if soc_true is not None:
        ax.plot(time_h, soc_true, 'k-', linewidth=1.5, label='True SOC', alpha=0.8)
    ax.plot(time_h, soc_est, 'r-', linewidth=1.2, label='AI-GRU Predicted', alpha=0.9)
    ax.set_ylabel('SOC (%)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    if soc_true is not None:
        error = soc_est - soc_true
        mae = np.mean(np.abs(error))
        maxerr = np.max(np.abs(error))
        ax.text(0.02, 0.05, f'MAE={mae:.2f}%, MaxErr={maxerr:.2f}%',
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- Row 2: Voltage & Current ---
    ax = axes[1]
    ax.plot(time_h, data['voltage'], 'b-', linewidth=0.8, alpha=0.8, label='Voltage')
    ax.set_ylabel('Voltage (V)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(time_h, data['current'], 'g-', linewidth=0.6, alpha=0.6, label='Current')
    ax2.set_ylabel('Current (A)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax.set_title('Input: Voltage & Current', fontsize=11)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # --- Row 3: R0 and R1 (Parameter Identification) ---
    ax = axes[2]
    ax.plot(time_h, params['r0'] * 1000, 'b-', linewidth=1, label='R0 (mOhm)', alpha=0.8)
    ax.plot(time_h, params['r1'] * 1000, 'r-', linewidth=1, label='R1 (mOhm)', alpha=0.8)
    ax.set_ylabel('Resistance (mOhm)')
    ax.set_title('RLS Parameter Identification: R0, R1', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add final values annotation
    ax.text(0.02, 0.85, f'R0={params["r0"][-1]*1000:.1f}mOhm, R1={params["r1"][-1]*1000:.1f}mOhm',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # --- Row 4: Tau and C1 ---
    ax = axes[3]
    ax.plot(time_h, params['tau'], 'purple', linewidth=1, label='Tau (s)', alpha=0.8)
    ax.set_ylabel('Tau (s)', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax.set_title('RLS Parameter Identification: Tau (=R1*C1) & C1', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(time_h, params['c1'], 'orange', linewidth=0.8, label='C1 (F)', alpha=0.7)
    ax2.set_ylabel('C1 (F)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    ax.text(0.02, 0.85, f'Tau={params["tau"][-1]:.1f}s, C1={params["c1"][-1]:.0f}F',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # --- Row 5: Training History (if available) ---
    if training_history and len(training_history.get('train_loss', [])) > 0:
        ax = axes[4]
        epochs = range(1, len(training_history['train_loss']) + 1)
        ax.plot(epochs, training_history['train_loss'], 'b-', linewidth=1, label='Train Loss', alpha=0.8)
        if training_history.get('val_loss'):
            ax.plot(epochs, training_history['val_loss'], 'r-', linewidth=1, label='Val Loss', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('AI-GRU Training History', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Annotate best loss
        best_val = min(training_history.get('val_loss', [1.0]))
        best_epoch = training_history['val_loss'].index(best_val) + 1
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.05, f'Best Val Loss={best_val:.6f} (Epoch {best_epoch})',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    axes[-1].set_xlabel('Time (hours)')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SOC Prediction using pre-trained AI-GRU model')
    parser.add_argument('--data', required=True, help='Path to data file (.xlsx) or directory')
    parser.add_argument('--model', default='saved_models/soc_gru_model',
                        help='Path to saved model (prefix, without .pth)')
    parser.add_argument('--train', action='store_true', help='Train a new model from data')
    parser.add_argument('--save-model', default=None, help='Path to save trained model')
    parser.add_argument('--eval', action='store_true', help='Evaluate against true SOC')
    parser.add_argument('--output', default='prediction_results', help='Output directory for results')
    parser.add_argument('--ocv-data', default=None, help='Path to OCV data directory (for param ID)')
    parser.add_argument('--capacity', type=float, default=1.1, help='Nominal capacity (Ah)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (if --train)')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    data_path = Path(args.data)
    
    # Collect data files
    if data_path.is_dir():
        files = sorted(data_path.glob('**/*.xlsx'))
        files = [f for f in files if 'newprofile' not in f.name]
    else:
        files = [data_path]
    
    if not files:
        print(f"No data files found in {data_path}")
        return
    
    print(f"Found {len(files)} data file(s)")
    
    # Load OCV table if available
    ocv_soc_table = None
    ocv_dir = args.ocv_data or os.path.join(SCRIPT_DIR, 'raw_data')
    if os.path.exists(ocv_dir):
        try:
            ocv_builder = OCVCurveBuilder(ocv_data_dir=ocv_dir)
            ocv_builder.load_ocv_data(target_temperature=30, use_test_file=True)
            ocv_soc_table = ocv_builder.get_ocv_soc_table()
            print(f"  OCV-SOC table loaded ({len(ocv_soc_table)} points)")
        except Exception as e:
            print(f"  Warning: Could not load OCV data: {e}")
    
    # Load data
    processor = BatteryDataProcessor(data_dir=str(data_path if data_path.is_dir() else data_path.parent))
    all_data = []
    for f in files:
        try:
            d = load_data_file(f, processor)
            all_data.append(d)
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}")
    
    if not all_data:
        print("No valid data loaded.")
        return
    
    # === Mode 1: Train new model ===
    if args.train:
        print(f"\n=== Training AI-GRU on {len(all_data)} files ===")
        
        # Need true SOC for training
        for d in all_data:
            if 'soc_true' not in d and ocv_soc_table is not None:
                d['soc_true'] = calculate_soc_labels(
                    d['time'], d['current'], d['voltage'], args.capacity, ocv_soc_table)
        
        file_data_list = []
        for d in all_data:
            if 'soc_true' in d:
                file_data_list.append({
                    'voltage': d['voltage'], 'current': d['current'],
                    'time': d['time'], 'temperature': d['temperature'],
                    'soc_true': d['soc_true']
                })
        
        if not file_data_list:
            print("Error: No true SOC data available for training. Need OCV table or SOC column.")
            return
        
        ai = ImprovedAISOCEstimator(
            initial_soc=50.0, nominal_capacity=args.capacity,
            sequence_length=20, hidden_size=128)
        ai.train_multi_file(file_data_list, epochs=args.epochs, batch_size=512)
        
        # Save model
        save_dir = args.save_model or 'saved_models'
        model_dir = os.path.join(SCRIPT_DIR, save_dir) if not os.path.isabs(save_dir) else save_dir
        ai.save_model(model_dir)
        print(f"\n  Model saved to: {model_dir}/")
        
        # Also run inference on training data for visualization
        model = ai
    
    # === Mode 2: Load pre-trained model ===
    else:
        model_prefix = args.model
        model_dir = os.path.dirname(model_prefix)
        model_name = os.path.basename(model_prefix)
        
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(SCRIPT_DIR, model_dir)
        
        print(f"\n=== Loading pre-trained model from {model_dir}/{model_name} ===")
        model = ImprovedAISOCEstimator.load_model(model_dir, model_name)
    
    # === Run predictions ===
    print(f"\n=== Running predictions on {len(all_data)} files ===")
    evaluator = SOCEvaluator()
    
    results_summary = []
    
    for data in all_data:
        fname = data['filename']
        print(f"\n  Predicting: {fname}")
        
        # AI prediction
        soc_est = model.predict_batch(
            data['voltage'], data['current'], data['time'], data['temperature'])
        
        # Parameter identification
        params = None
        if ocv_soc_table is not None:
            params = run_parameter_identification(data, ocv_soc_table)
        else:
            # Generate empty params
            n = len(data['time'])
            params = {'r0': np.full(n, 0.05), 'r1': np.full(n, 0.03),
                      'tau': np.full(n, 30.0), 'c1': np.full(n, 1000.0), 'v1': np.zeros(n)}
        
        # Evaluate if true SOC available
        soc_true = data.get('soc_true')
        if soc_true is None and ocv_soc_table is not None and args.eval:
            soc_true = calculate_soc_labels(
                data['time'], data['current'], data['voltage'], args.capacity, ocv_soc_table)
        
        metrics = None
        if soc_true is not None:
            metrics = evaluator.evaluate(soc_true, soc_est)
            status = 'PASS' if metrics['max_error'] < 5 else 'FAIL'
            results_summary.append({
                'file': fname, 'mae': metrics['mae'], 
                'maxerr': metrics['max_error'], 'status': status
            })
            print(f"    MAE={metrics['mae']:.2f}%, MaxErr={metrics['max_error']:.2f}% [{status}]")
        
        # Generate visualization
        plot_name = Path(fname).stem + '_prediction.png'
        plot_path = os.path.join(args.output, plot_name)
        plot_results(
            data, soc_est, params, plot_path,
            title=f"SOC Prediction: {fname}",
            soc_true=soc_true,
            training_history=model.training_history if hasattr(model, 'training_history') else None
        )
        
        # Save CSV
        csv_name = Path(fname).stem + '_prediction.csv'
        csv_path = os.path.join(args.output, csv_name)
        import pandas as pd
        df = pd.DataFrame({
            'time_s': data['time'],
            'voltage_V': data['voltage'],
            'current_A': data['current'],
            'temperature_C': data['temperature'],
            'soc_predicted_%': soc_est,
            'R0_Ohm': params['r0'],
            'R1_Ohm': params['r1'],
            'Tau_s': params['tau'],
            'C1_F': params['c1'],
        })
        if soc_true is not None:
            df['soc_true_%'] = soc_true
            df['error_%'] = soc_est - soc_true
        df.to_csv(csv_path, index=False)
        print(f"    CSV saved: {csv_path}")
    
    # Print summary
    if results_summary:
        print(f"\n{'='*60}")
        print(f"{'File':>30s} | {'MAE':>7s} | {'MaxErr':>8s} | {'Status':>6s}")
        print(f"{'-'*60}")
        for r in results_summary:
            print(f"{r['file'][:30]:>30s} | {r['mae']:>6.2f}% | {r['maxerr']:>7.2f}% | [{r['status']}]")
        print(f"{'='*60}")
        
        avg_mae = np.mean([r['mae'] for r in results_summary])
        avg_maxerr = np.mean([r['maxerr'] for r in results_summary])
        n_pass = sum(1 for r in results_summary if r['status'] == 'PASS')
        print(f"Average MAE={avg_mae:.2f}%, Average MaxErr={avg_maxerr:.2f}%, Pass={n_pass}/{len(results_summary)}")
    
    print(f"\nAll results saved to: {args.output}/")


if __name__ == '__main__':
    main()
