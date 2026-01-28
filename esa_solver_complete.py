"""
Complete LFTSAD Solver with ESA Anomaly Detection Benchmark integration
Handles multi-channel satellite telemetry data with proper ESA metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from model.LFTSAD import LFTSAD
from data_factory.esa_data_loader import get_esa_loader, ESALabelsParser
from ESA_metrics import ESAScores, ADTQC, ChannelAwareFScore

import warnings
warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class ESASolver:
    """
    LFTSAD Solver for ESA Anomaly Detection Benchmark
    """
    
    def __init__(self, config):
        """
        Initialize solver with ESA data support
        
        Config should include:
            - train_csv_path: Path to training CSV
            - test_csv_path: Path to test CSV
            - labels_csv_path: Path to labels.csv (ESA format)
            - target_channels: List of channel names to use
            - batch_size, win_size, lr, num_epochs, etc.
            - use_esa_metrics: Whether to compute ESA metrics
            - beta: Beta for F-beta score
        """
        self.__dict__.update(config)
        
        # Validate required paths
        assert hasattr(self, 'test_csv_path'), "test_csv_path required"
        assert hasattr(self, 'labels_csv_path'), "labels_csv_path required"
        
        # Set defaults
        self.use_esa_metrics = getattr(self, 'use_esa_metrics', True)
        self.beta = getattr(self, 'beta', 0.5)
        self.target_channels = getattr(self, 'target_channels', None)
        self.step = getattr(self, 'step', 1)  # Step for sliding window
        
        # Load data
        print("="*60)
        print("Initializing ESA-LFTSAD Solver")
        print("="*60)
        
        # Create data loaders
        if hasattr(self, 'train_csv_path'):
            self.train_loader, self.train_dataset = get_esa_loader(
                csv_path=self.train_csv_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                step=self.win_size // 2,  # 50% overlap for training
                target_channels=self.target_channels,
                mode='train'
            )
        
        self.test_loader, self.test_dataset = get_esa_loader(
            csv_path=self.test_csv_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,  # Usually 1 for test to get all predictions
            target_channels=self.target_channels,
            mode='test'
        )
        
        # Load labels
        self.labels_parser = ESALabelsParser(self.labels_csv_path)
        
        # Get channel info
        self.channel_names = self.test_dataset.get_channel_names()
        self.n_channels = len(self.channel_names)
        
        print(f"Number of channels: {self.n_channels}")
        print(f"Channels: {self.channel_names[:5]}..." if len(self.channel_names) > 5 else f"Channels: {self.channel_names}")
        
        # Build model
        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.criterion_keep = nn.MSELoss(reduction='none')
        
    def build_model(self):
        """Build LFTSAD model"""
        self.model = LFTSAD(
            win_size=self.win_size,
            enc_in=self.n_channels,
            c_out=self.n_channels,
            d_model=getattr(self, 'd_model', 512),
            patch_size=getattr(self, 'patch_size', 5),
            channel=self.n_channels,
            patch_seq=getattr(self, 'patch_seq', 20),
            seq_size=getattr(self, 'seq_size', 20)
        )
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(self, 'lr', 1e-4)
        )
        
    def train(self):
        """Train the model"""
        if not hasattr(self, 'train_loader'):
            print("No training data provided, skipping training")
            return
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        time_now = time.time()
        train_steps = len(self.train_loader)
        
        for epoch in range(self.num_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            
            epoch_loss = 0.0
            
            for i, (input_data, labels, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                
                # input_data: (batch, win_size, n_channels)
                input = input_data.float().to(self.device)
                
                # Forward pass
                series, prior, series_seq, prior_seq = self.model(input)
                
                # Compute loss
                loss = 0.0
                for u in range(len(prior)):
                    loss += (
                        getattr(self, 'p_seq', 0.5) * self.criterion(series_seq[u], prior_seq[u]) +
                        (1 - getattr(self, 'p_seq', 0.5)) * self.criterion(series[u], prior[u])
                    )
                
                loss = loss / len(prior)
                epoch_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(f'\tBatch [{i+1}/{train_steps}] Loss: {loss.item():.6f} '
                          f'Speed: {speed:.4f}s/iter Left: {left_time:.1f}s')
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                self.optimizer.step()
            
            avg_loss = epoch_loss / train_steps
            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"Avg Loss: {avg_loss:.6f} "
                  f"Time: {time.time() - epoch_time:.2f}s")
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        
        print("Training completed!\n")
    
    def test(self):
        """Memory-optimized testing"""
        t_test_start = time.time()
        
        print("\n" + "="*60)
        print("Starting Testing and Evaluation (Memory Optimized)")
        print("="*60)
        
        self.model.eval()
        
        n_samples = len(self.test_dataset.data)
        n_channels = self.n_channels
        
        print(f"Total samples: {n_samples:,}, Channels: {n_channels}")
        print(f"Estimated memory for scores: {n_samples * n_channels * 4 / 1e9:.2f} GB")
        
        # =========================================================
        # OPTIMIZATION 1: Pre-allocate arrays instead of appending
        # =========================================================
        score_sum = np.zeros((n_samples, n_channels), dtype=np.float32)
        score_count = np.zeros(n_samples, dtype=np.float32)
        ground_truth = np.zeros((n_samples, n_channels), dtype=np.float32)
        
        # =========================================================
        # OPTIMIZATION 2: Process directly into pre-allocated arrays
        # =========================================================
        with torch.no_grad():
            for i, (input_data, labels, start_indices) in enumerate(tqdm(
                self.test_loader, desc="Testing", leave=True
            )):
                input = input_data.float().to(self.device)
                
                # Forward pass
                series, prior, series_seq, prior_seq = self.model(input)
                
                # Compute anomaly scores
                loss = torch.zeros_like(series[0])
                for u in range(len(prior)):
                    loss += (
                        getattr(self, 'p_seq', 0.5) * self.criterion_keep(series_seq[u], prior_seq[u]) +
                        (1 - getattr(self, 'p_seq', 0.5)) * self.criterion_keep(series[u], prior[u])
                    )
                
                # Convert to numpy ONCE and immediately process
                scores = loss.detach().cpu().numpy()  # (B, W, C)
                labels_np = labels.numpy()  # (B, W, C)
                start_idx_np = start_indices.numpy()
                
                # =========================================================
                # OPTIMIZATION 3: Accumulate directly, no intermediate storage
                # =========================================================
                batch_size = scores.shape[0]
                for b in range(batch_size):
                    start_idx = start_idx_np[b]
                    end_idx = min(start_idx + self.win_size, n_samples)
                    window_len = end_idx - start_idx
                    
                    score_sum[start_idx:end_idx] += scores[b, :window_len, :]
                    score_count[start_idx:end_idx] += 1
                    np.maximum(
                        ground_truth[start_idx:end_idx],
                        labels_np[b, :window_len, :],
                        out=ground_truth[start_idx:end_idx]
                    )
                
                # =========================================================
                # OPTIMIZATION 4: Explicitly delete tensors and clear cache
                # =========================================================
                del input, series, prior, series_seq, prior_seq, loss, scores, labels_np
                
                # Clear GPU cache periodically
                if (i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # =========================================================
        # OPTIMIZATION 5: In-place division
        # =========================================================
        score_count[score_count == 0] = 1
        anomaly_scores = score_sum  # Reuse the array
        anomaly_scores /= score_count[:, None]  # In-place division
        del score_sum, score_count  # Free memory
        gc.collect()
        
        print(f"\nReconstructed ground truth shape: {ground_truth.shape}")
        print(f"Anomaly scores shape: {anomaly_scores.shape}")
        
        # Compute threshold and predictions
        thresh = np.percentile(anomaly_scores, 100 - self.anormly_ratio, axis=0)
        pred_binary = (anomaly_scores > thresh[None, :]).astype(np.int8)  # Use int8 instead of int64!
        
        print(f"Thresholds: {np.array2string(thresh, formatter={'float_kind': lambda x: f'{x:.6f}'})}")
        
        t_test_end = time.time()
        t_test = t_test_end - t_test_start
        
        # Standard metrics
        print("\n" + "="*60)
        print("STANDARD METRICS (Channel-Aggregated)")
        print("="*60)
        
        gt_any = ground_truth.any(axis=1).astype(np.int8)
        pred_any = pred_binary.any(axis=1).astype(np.int8)
        
        accuracy = accuracy_score(gt_any, pred_any)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt_any, pred_any, average='binary', zero_division=0
        )
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f_score:.4f}")
        
        # ESA Metrics with memory optimization
        if self.use_esa_metrics:
            esa_results, channel_results, adtqc = self._compute_esa_metrics_optimized(
                predictions=pred_binary,
                ground_truth=ground_truth,
                anomaly_scores=anomaly_scores
            )
            return accuracy, precision, recall, f_score, esa_results, channel_results, adtqc, t_test
        else:
            return accuracy, precision, recall, f_score, t_test
        
    def _reconstruct_from_windows(self, scores, labels, start_indices):
        """
        scores: (n_windows, win_size, n_channels)
        labels: (n_windows, win_size, n_channels)
        returns:
        ground_truth: (n_samples, n_channels)
        anomaly_scores: (n_samples, n_channels)
        """
        n_samples = len(self.test_dataset.data)
        n_channels = labels.shape[2]

        score_sum = np.zeros((n_samples, n_channels), dtype=np.float32)
        score_count = np.zeros(n_samples, dtype=np.float32)

        label_array = np.zeros((n_samples, n_channels), dtype=np.float32)

        for i, start_idx in enumerate(start_indices):
            end_idx = min(start_idx + self.win_size, n_samples)
            window_len = end_idx - start_idx

            score_sum[start_idx:end_idx] += scores[i, :window_len, :]  # (window_len, C)
            score_count[start_idx:end_idx] += 1

            label_array[start_idx:end_idx] = np.maximum(
                label_array[start_idx:end_idx],
                labels[i, :window_len, :]
            )

        score_count[score_count == 0] = 1
        anomaly_scores = score_sum / score_count[:, None]  # (n_samples, C)

        ground_truth = label_array
        return ground_truth, anomaly_scores
    
    def _compute_esa_metrics(self, predictions, ground_truth, anomaly_scores):
        """
        Compute ESA metrics
        
        Args:
            predictions: (n_samples, n_channels) - binary predictions
            ground_truth: (n_samples, n_channels) - ground truth
            anomaly_scores: (n_samples, n_channels) - anomaly scores
        """
        print("\n" + "="*60)
        print("ESA ANOMALY DETECTION BENCHMARK METRICS")
        print("="*60)
        
        # Get timestamps
        timestamps = self.test_dataset.get_full_timestamps()
        full_range = (timestamps.iloc[0], timestamps.iloc[-1])
        
        
        
        # Get ground truth labels in ESA format
        y_true_df = self.labels_parser.get_labels_dataframe(
            channel_filter=self.channel_names
        )
        
        start, end = full_range

        # keep only events that overlap the telemetry time range
        y_true_df = y_true_df[
            (y_true_df["EndTime"] >= start) &
            (y_true_df["StartTime"] <= end)
        ].copy()

        # clip event bounds so all events lie inside full_range (satisfies metric assertions)
        y_true_df["StartTime"] = y_true_df["StartTime"].clip(lower=start)
        y_true_df["EndTime"]   = y_true_df["EndTime"].clip(upper=end)
        print("y_true_df columns:", y_true_df.columns.tolist())
        print("Filtered labels:", len(y_true_df),
            "range:", y_true_df["StartTime"].min(), "to", y_true_df["EndTime"].max())
        

        print(f"\nGround truth events: {len(y_true_df)}")
        print(f"Unique anomaly IDs: {y_true_df['ID'].nunique()}")
        print(f"Time range: {full_range[0]} to {full_range[1]}")
        
        """# Create predictions in ESA format (multi-channel)
        y_pred_dict = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            y_pred_channel = []
            for i in range(len(predictions)):
                y_pred_channel.append([timestamps.iloc[i], int(predictions[i, ch_idx])])
            y_pred_dict[ch_name] = y_pred_channel
        

        # single channel format
        y_any_pred_dict = {}
        global_pred = predictions.any(axis=1).astype(int)
        y_any_pred = []
        for i in range(len(global_pred)):
            y_any_pred.append([timestamps.iloc[i], int(global_pred[i])])

        #y_any_pred_dict["is_anomaly"] = y_any_pred"""

        ts = timestamps.tolist()  # avoid repeated .iloc

        y_pred_dict = {
            ch_name: [[t, int(p)] for t, p in zip(ts, predictions[:, ch_idx])]
            for ch_idx, ch_name in enumerate(self.channel_names)
        }

        global_pred = predictions.any(axis=1).astype(int)
        y_any_pred = [[t, int(p)] for t, p in zip(ts, global_pred)]
        
        # 1. ESA Scores (Event-wise and Affiliation-based)
        print("\n--- Event-wise and Affiliation-based Scores ---")
        try:
            # Use first channel for basic ESA scores
            esa_metric = ESAScores(
                betas=self.beta,
                full_range=full_range
            )
            print("Telemetry range:", full_range[0], "to", full_range[1])
            print("Labels range:", y_true_df["StartTime"].min(), "to", y_true_df["EndTime"].max())
            # Convert single channel for ESAScores
            #y_pred_first = y_pred_dict[self.channel_names[0]]
            esa_results = esa_metric.score(y_true_df, y_any_pred)
            
            for metric_name, value in esa_results.items():
                print(f"{metric_name:30s}: {value:8.4f}")
                
        except Exception as e:
            print(f"Error computing ESA scores: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Channel-Aware F-Score
        print("\n--- Channel-Aware F-Score ---")
        try:
            channel_metric = ChannelAwareFScore(
                beta=self.beta if isinstance(self.beta, float) else self.beta,
                full_range=full_range
                
            )
            
            channel_results = channel_metric.score(y_true_df, y_pred_dict)
            
            for metric_name, value in channel_results.items():
                print(f"{metric_name:30s}: {value:8.4f}")
                
        except Exception as e:
            print(f"Error computing channel-aware metrics: {e}")
        
        # 3. ADTQC (Latency metrics)
        print("\n--- ADTQC Latency Metrics ---")
        try:
            adtqc_metric = ADTQC(
                full_range=full_range
            )
            
            adtqc_results = adtqc_metric.score(y_true_df, y_pred_dict)
            
            for metric_name, value in adtqc_results.items():
                if isinstance(value, float):
                    print(f"{metric_name:30s}: {value:8.4f}")
                else:
                    print(f"{metric_name:30s}: {value}")
                    
        except Exception as e:
            print(f"Error computing ADTQC metrics: {e}")
        
        print("="*60 + "\n")

        return esa_results, channel_results, adtqc_results

# Example usage
if __name__ == "__main__":
    config = {
        # Data paths
        'train_csv_path': 'path/to/train.csv',  # Optional
        'test_csv_path': 'path/to/84_months.test.csv',
        'labels_csv_path': 'path/to/labels.csv',
        
        # Channel selection
        'target_channels': [
            "channel_41", "channel_42", "channel_43",
            "channel_44", "channel_45", "channel_46"
        ],
        
        # Model parameters
        'batch_size': 32,
        'win_size': 100,
        'step': 1,  # Step for sliding window in test
        'd_model': 512,
        'patch_size': 5,
        'patch_seq': 20,
        'seq_size': 20,
        
        # Training parameters
        'num_epochs': 10,
        'lr': 1e-4,
        'p_seq': 0.5,
        'sw_max_mean': 0,
        
        # Evaluation parameters
        'anormly_ratio': 1.0,  # Top 1% as anomalies
        'use_esa_metrics': True,
        'beta': 0.5,
    }
    
    # Create solver
    solver = ESASolver(config)
    
    # Train (if training data provided)
    if hasattr(solver, 'train_loader'):
        solver.train()
    
    # Test and evaluate
    accuracy, precision, recall, f_score = solver.test()
