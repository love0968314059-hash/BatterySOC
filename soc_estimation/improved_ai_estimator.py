#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的AI SOC估计器
简化设计，专注于实际效果
"""

import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except (ImportError, PermissionError, OSError) as e:
    TORCH_AVAILABLE = False
    torch = None
    # 定义空基类
    class Dataset:
        pass
    class DataLoader:
        pass
    class nn:
        class Module:
            pass
        class GRU:
            pass
        class Linear:
            pass
        class Dropout:
            pass
        class ReLU:
            pass
        @staticmethod
        def Sequential(*args):
            pass


class SimpleGRUModel(nn.Module):
    """GRU SOC估计模型 - 增强版"""
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.1):
        super(SimpleGRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 输出0-1范围
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.fc(last_output)
        return out


class ImprovedAISOCEstimator:
    """
    改进的AI SOC估计器
    简化设计，使用GRU网络
    """
    
    def __init__(self,
                 initial_soc=50.0,
                 nominal_capacity=1.1,
                 model_type='gru',
                 sequence_length=20,
                 hidden_size=128,
                 num_layers=2,
                 device='cpu'):
        """
        初始化AI SOC估计器
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install: pip install torch")
        
        self.initial_soc = initial_soc
        self.nominal_capacity = nominal_capacity
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.device = torch.device('cpu')  # 强制使用CPU
        
        # 6个核心特征
        self.input_size = 6
        
        # 创建模型
        self.model = SimpleGRUModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2
        ).to(self.device)
        
        self.is_trained = False
        self.soc_history = []
        
        # 特征归一化参数
        self.feature_mean = None
        self.feature_std = None
        
        # 训练历史（用于可视化）
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def _prepare_features(self, voltage, current, time, temperature):
        """
        准备简化的特征集
        特征：电压、电流、温度、时间差、累积AH、功率
        """
        n = len(voltage)
        voltage = np.asarray(voltage, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        time = np.asarray(time, dtype=np.float32)
        temperature = np.asarray(temperature, dtype=np.float32)
        
        # 时间差
        dt = np.diff(time)
        dt = np.concatenate([[1.0], dt])
        dt = np.clip(dt, 0.1, 100.0)
        
        # 累积AH（归一化）
        cumulative_ah = np.cumsum(current * dt / 3600)
        cumulative_ah_norm = cumulative_ah / self.nominal_capacity
        
        # 功率
        power = voltage * current
        
        # 6个特征
        features = np.column_stack([
            voltage,           # 电压
            current,           # 电流  
            temperature,       # 温度
            dt,                # 时间差
            cumulative_ah_norm,# 累积AH（归一化）
            power              # 功率
        ])
        
        return features
    
    def _create_sequences(self, features, targets):
        """创建序列数据"""
        n = len(features)
        X, y = [], []
        
        for i in range(n - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length - 1])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train_multi_file(self, file_data_list, epochs=150, batch_size=256, learning_rate=0.001):
        """
        多文件训练 - 避免跨文件边界创建序列
        file_data_list: list of dicts with keys: voltage, current, time, temperature, soc_true
        """
        print(f"  训练AI模型 ({len(file_data_list)} files)...")
        
        # 先计算全局特征统计
        all_features = []
        for fd in file_data_list:
            feat = self._prepare_features(fd['voltage'], fd['current'], fd['time'], fd['temperature'])
            all_features.append(feat)
        
        all_features_concat = np.concatenate(all_features)
        self.feature_mean = np.mean(all_features_concat, axis=0)
        self.feature_std = np.std(all_features_concat, axis=0) + 1e-8
        
        # 每个文件单独创建序列（避免跨文件边界）
        X_all, y_all = [], []
        for fd in file_data_list:
            features = self._prepare_features(fd['voltage'], fd['current'], fd['time'], fd['temperature'])
            features_norm = (features - self.feature_mean) / self.feature_std
            soc_norm = np.asarray(fd['soc_true'], dtype=np.float32) / 100.0
            
            # 正常序列
            X, y = self._create_sequences(features_norm, soc_norm)
            X_all.append(X)
            y_all.append(y)
            
            # Padded warmup sequences for this file
            for i in range(min(self.sequence_length, len(features_norm))):
                seq = np.zeros((self.sequence_length, features_norm.shape[1]), dtype=np.float32)
                available = features_norm[:i+1]
                pad_len = self.sequence_length - len(available)
                if pad_len > 0:
                    seq[:pad_len] = features_norm[0]
                seq[pad_len:] = available
                X_all.append(seq.reshape(1, *seq.shape))
                y_all.append(np.array([soc_norm[i]], dtype=np.float32))
        
        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        
        print(f"    训练样本: {len(X)}, 序列长度: {self.sequence_length}")
        
        # Shuffle and split
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X))
        n_train = int(len(X) * 0.9)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        self._do_training(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)
    
    def train(self, voltage, current, time, temperature, soc_true,
              epochs=150, batch_size=256, learning_rate=0.001):
        """训练模型 (单文件拼接版本 - 向后兼容)"""
        print(f"  训练AI模型...")
        
        features = self._prepare_features(voltage, current, time, temperature)
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8
        features_norm = (features - self.feature_mean) / self.feature_std
        soc_norm = np.asarray(soc_true, dtype=np.float32) / 100.0
        
        X, y = self._create_sequences(features_norm, soc_norm)
        
        # Padded warmup sequences
        X_padded, y_padded = [], []
        for i in range(min(self.sequence_length, len(features_norm))):
            seq = np.zeros((self.sequence_length, features_norm.shape[1]), dtype=np.float32)
            available = features_norm[:i+1]
            pad_len = self.sequence_length - len(available)
            if pad_len > 0:
                seq[:pad_len] = features_norm[0]
            seq[pad_len:] = available
            X_padded.append(seq)
            y_padded.append(soc_norm[i])
        
        if X_padded:
            X = np.concatenate([np.array(X_padded, dtype=np.float32), X])
            y = np.concatenate([np.array(y_padded, dtype=np.float32), y])
        
        print(f"    训练样本: {len(X)}, 序列长度: {self.sequence_length}")
        
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X))
        n_train = int(len(X) * 0.9)
        X_train, X_val = X[indices[:n_train]], X[indices[n_train:]]
        y_train, y_val = y[indices[:n_train]], y[indices[n_train:]]
        
        self._do_training(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)
    
    def _do_training(self, X_train, y_train, X_val, y_val,
                     epochs=150, batch_size=256, learning_rate=0.001):
        """核心训练逻辑"""
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val.reshape(-1, 1))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0
        
        # 清空训练历史
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                # 计算实际MAE (%)
                val_mae = np.sqrt(val_loss) * 100
                print(f"    Epoch {epoch+1}: Train MSE={train_loss:.6f}, Val MSE={val_loss:.6f}, Val MAE~{val_mae:.2f}%")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stop at epoch {epoch+1}")
                    break
        
        # 恢复最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        self.model.eval()
        self.is_trained = True
        print(f"    Training complete. Best val MSE: {best_val_loss:.6f}, Best val MAE: {np.sqrt(best_val_loss)*100:.2f}%")
    
    def predict_batch(self, voltage, current, time, temperature):
        """批量预测（不训练）- 使用padding从第0步开始预测，避免初始偏差"""
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        # 准备特征
        features = self._prepare_features(voltage, current, time, temperature)
        features_norm = (features - self.feature_mean) / self.feature_std
        
        n = len(features)
        soc_estimated = np.zeros(n, dtype=np.float32)
        
        # === Key fix: predict from step 0 using padding ===
        # Instead of using biased initial_soc for warmup, pad the sequence
        # with repeated first features and use the model for ALL steps.
        self.model.eval()
        
        # Batch prediction for efficiency
        batch_size = 512
        all_sequences = []
        
        for i in range(n):
            if i < self.sequence_length:
                # Pad with repeated features: use available features + repeat first
                seq = np.zeros((self.sequence_length, features_norm.shape[1]), dtype=np.float32)
                available = features_norm[:i+1]  # Features up to step i
                pad_len = self.sequence_length - len(available)
                # Pad by repeating the first feature
                if pad_len > 0:
                    seq[:pad_len] = features_norm[0]
                seq[pad_len:] = available
            else:
                seq = features_norm[i - self.sequence_length + 1:i + 1]
            all_sequences.append(seq)
        
        # Predict in batches
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = np.array(all_sequences[start:end], dtype=np.float32)
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                predictions = self.model(batch_tensor).cpu().numpy()[:, 0]
                soc_estimated[start:end] = predictions * 100.0
        
        return np.clip(soc_estimated, 0, 100)
    
    def estimate_batch(self, voltage, current, time=None, temperature=None, soc_true=None):
        """
        批量估计SOC
        如果提供soc_true，会先训练模型
        """
        n = len(voltage)
        
        if time is None:
            time = np.arange(n, dtype=float)
        if temperature is None:
            temperature = np.full(n, 25.0)
        
        # 如果有真实SOC，先训练
        if soc_true is not None and len(soc_true) > self.sequence_length * 3:
            self.train(voltage, current, time, temperature, soc_true)
        
        # 预测
        if self.is_trained:
            return self.predict_batch(voltage, current, time, temperature)
        else:
            # 未训练时使用AH积分
            return self._ah_integration(voltage, current, time, temperature)
    
    def save_model(self, save_dir, model_name="soc_gru_model"):
        """
        保存训练好的模型和配置参数
        
        Args:
            save_dir: 保存目录
            model_name: 模型名称前缀
        
        保存文件:
            - {model_name}.pth: PyTorch模型权重
            - {model_name}_config.json: 模型配置和归一化参数
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法保存")
        
        # 保存模型权重
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # 保存配置和归一化参数
        config = {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'hidden_size': self.model.gru.hidden_size,
            'num_layers': self.model.gru.num_layers,
            'nominal_capacity': self.nominal_capacity,
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'feature_names': ['voltage', 'current', 'temperature', 'dt', 'cumulative_ah_norm', 'power'],
            'training_history': {
                'train_loss': [float(x) for x in self.training_history.get('train_loss', [])],
                'val_loss': [float(x) for x in self.training_history.get('val_loss', [])],
            }
        }
        
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"  模型已保存:")
        print(f"    权重: {model_path}")
        print(f"    配置: {config_path}")
        return model_path, config_path
    
    @classmethod
    def load_model(cls, save_dir, model_name="soc_gru_model", device='cpu'):
        """
        从文件加载预训练模型
        
        Args:
            save_dir: 模型保存目录
            model_name: 模型名称前缀
            device: 设备 ('cpu' 或 'cuda')
            
        Returns:
            ImprovedAISOCEstimator: 加载了预训练权重的估计器
        """
        import os
        
        # 加载配置
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建估计器
        estimator = cls(
            initial_soc=50.0,
            nominal_capacity=config['nominal_capacity'],
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            device=device
        )
        
        # 加载归一化参数
        estimator.feature_mean = np.array(config['feature_mean'], dtype=np.float32)
        estimator.feature_std = np.array(config['feature_std'], dtype=np.float32)
        
        # 加载训练历史
        estimator.training_history = config.get('training_history', {'train_loss': [], 'val_loss': []})
        
        # 加载模型权重
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        estimator.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        estimator.model.eval()
        estimator.is_trained = True
        
        print(f"  模型已加载: {model_path}")
        print(f"    序列长度: {config['sequence_length']}, 隐藏层: {config['hidden_size']}")
        print(f"    特征: {config['feature_names']}")
        
        return estimator
    
    def _ah_integration(self, voltage, current, time, temperature):
        """AH积分备选方案"""
        n = len(current)
        dt = np.diff(time)
        dt = np.concatenate([[1.0], dt])
        dt = np.clip(dt, 0.1, 100.0)
        
        soc = np.zeros(n)
        soc[0] = self.initial_soc
        
        for i in range(1, n):
            delta_soc = current[i-1] * dt[i] / 3600 / self.nominal_capacity * 100
            soc[i] = np.clip(soc[i-1] + delta_soc, 0, 100)
        
        return soc


# 测试
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch未安装")
    else:
        print("测试AI估计器")
        
        # 简单测试数据
        np.random.seed(42)
        n = 2000
        time = np.arange(n, dtype=float)
        current = -0.5 + 0.1 * np.sin(time / 100)
        
        soc_true = [100.0]
        for i in range(1, n):
            dt = 1.0
            delta = current[i] * dt / 3600 / 1.1 * 100
            soc_true.append(np.clip(soc_true[-1] + delta, 0, 100))
        soc_true = np.array(soc_true)
        
        voltage = 2.5 + soc_true / 100 * 1.1
        temperature = np.full(n, 25.0)
        
        estimator = ImprovedAISOCEstimator(
            initial_soc=soc_true[0],
            nominal_capacity=1.1,
            sequence_length=15
        )
        
        soc_est = estimator.estimate_batch(voltage, current, time, temperature, soc_true)
        
        mae = np.mean(np.abs(soc_est - soc_true))
        print(f"MAE: {mae:.3f}%")
