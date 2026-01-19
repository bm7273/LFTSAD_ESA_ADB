"""
Enhanced ESA Solver supporting both LFTSAD and fadsd models
fadsd requires special data loading with local and global windows
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from model.LFTSAD import LFTSAD
from model.FADSD import FADSD
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


class ESADataLoaderfadsd:
    """
    Custom data loader for fadsd that creates both local and global windows
    """
    def __init__(self, csv_path, batch_size, win_size, win_size_1, count, 
                 target_channels=None, mode='test', step=1):
        """
        Args:
            csv_path: Path to CSV file
            batch_size: Batch size
            win_size: Local window size (e.g., 10)
            win_size_1: Global window size (e.g., 20)
            count: Number of global windows (must be odd, e.g., 21)
            target_channels: List of channel names to use
            mode: 'train' or 'test'
            step: Step size for sliding window
        """
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader
        
        self.win_size = win_size
        self.win_size_1 = win_size_1
        self.count = count
        self.step = step
        self.batch_size = batch_size
        
        # Load data
        df = pd.read_csv(csv_path)
        self.timestamps = pd.to_datetime(df.iloc[:, 0])
        
        # Select channels
        if target_channels is not None:
            available_channels = [col for col in df.columns[1:] if col in target_channels]
            self.channel_names = available_channels
            self.data = df[available_channels].values.astype(np.float32)

        else:
            self.channel_names = df.columns[1:].tolist()
            self.data = df.iloc[:, 1:].values.astype(np.float32)
        
        label_cols = [f"is_anomaly_{ch}" for ch in self.channel_names]

        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in CSV: {missing[:5]} ...")

        self.labels = df[label_cols].values.astype(np.float32)

        self.n_channels = self.data.shape[1]
        print(f"Loaded data shape: {self.data.shape}")
        
        # Create dataset
        self.dataset = fadsdDataset(
            self.data, self.labels, self.timestamps, self.win_size, 
            self.win_size_1, self.count, self.step
        )
        
        # Create dataloader
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            drop_last=False
        )
    
    def get_channel_names(self):
        return self.channel_names
    
    def get_full_timestamps(self):
        return self.timestamps
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        return iter(self.loader)


class fadsdDataset(torch.utils.data.Dataset):
    """
    Dataset that returns local window, global windows, and labels for fadsd
    """
    def __init__(self, data, labels, timestamps, win_size, win_size_1, count, step):
        """
        Args:
            data: (T, C) numpy array
            timestamps: pandas Series of timestamps
            win_size: Local window size
            win_size_1: Global window size  
            count: Number of global windows (must be odd)
            step: Step size for sliding window
        """
        self.data = data
        self.timestamps = timestamps
        self.win_size = win_size
        self.win_size_1 = win_size_1
        self.count = count
        self.step = step
        self.labels = labels
        
        # Calculate required context for global windows
        half_count = count // 2
        self.global_context = half_count * win_size_1
        
        # Generate valid starting indices
        self.indices = []
        half_count = self.count // 2
        left_ctx = half_count * self.win_size_1

        # start_idx must allow ALL global windows of length win_size_1
        start_min = left_ctx
        start_max = len(data) - self.win_size_1 - left_ctx  # inclusive max start for global windows

        self.indices = list(range(start_min, start_max + 1, step))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # Local window: (win_size, C)
        local_window = self.data[start_idx:start_idx + self.win_size]
        
        # Global windows: (count, win_size_1, C)
        half_count = self.count // 2
        global_windows = []
        
        for i in range(-half_count, half_count + 1):
            offset = start_idx + i * self.win_size_1
            global_win = self.data[offset:offset + self.win_size_1]
            global_windows.append(global_win)
        
        global_windows = np.stack(global_windows, axis=0)  # (count, win_size_1, C)
        
        # Labels (all zeros for now, will be loaded separately for ESA)
        labels = self.labels[start_idx:start_idx + self.win_size]
        
        return (
            torch.FloatTensor(local_window),
            torch.FloatTensor(global_windows),
            torch.FloatTensor(labels),
            start_idx
        )


class ESASolverMultiModel:
    """
    Enhanced ESA Solver supporting both LFTSAD and fadsd models
    """
    
    def __init__(self, config):
        """
        Initialize solver with ESA data support
        
        Config should include:
            - model_type: 'LFTSAD' or 'fadsd'
            - train_csv_path: Path to training CSV (optional)
            - test_csv_path: Path to test CSV
            - labels_csv_path: Path to labels.csv (ESA format)
            - target_channels: List of channel names to use
            - batch_size, win_size, lr, num_epochs, etc.
            - use_esa_metrics: Whether to compute ESA metrics
            - beta: Beta for F-beta score
            
            For fadsd specifically:
            - p: weight for local vs global score (default 0.1)
            - select: 0 for magnitude, 1 for phase (default 1)
            - win_size_1: global window size (default 20)
            - count: number of windows (must be odd, default 21)
        """
        self.__dict__.update(config)
        
        # Validate required paths
        assert hasattr(self, 'test_csv_path'), "test_csv_path required"
        assert hasattr(self, 'labels_csv_path'), "labels_csv_path required"
        assert hasattr(self, 'model_type'), "model_type required ('LFTSAD' or 'fadsd')"
        
        # Set defaults
        self.model_type = getattr(self, 'model_type', 'LFTSAD')
        self.use_esa_metrics = getattr(self, 'use_esa_metrics', True)
        self.beta = getattr(self, 'beta', 0.5)
        self.target_channels = getattr(self, 'target_channels', None)
        self.step = getattr(self, 'step', 1)
        
        # fadsd specific defaults
        if self.model_type == 'fadsd':
            self.p = getattr(self, 'p', 0.1)
            self.select = getattr(self, 'select', 1)
            self.win_size_1 = getattr(self, 'win_size_1', 20)
            self.count = getattr(self, 'count', 21)
            assert self.count % 2 == 1, "count must be odd"
        
        # Load data
        print("="*60)
        print(f"Initializing ESA Solver with {self.model_type} model")
        print("="*60)
        
        # Create data loaders based on model type
        if self.model_type == 'fadsd':
            self._load_fadsd_data()
        else:
            self._load_lftsad_data()
        
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
        
        # Loss functions (only for LFTSAD)
        if self.model_type == 'LFTSAD':
            self.criterion = nn.MSELoss()
            self.criterion_keep = nn.MSELoss(reduction='none')
    
    def _load_lftsad_data(self):
        """Load data for LFTSAD model"""
        if hasattr(self, 'train_csv_path'):
            self.train_loader, self.train_dataset = get_esa_loader(
                csv_path=self.train_csv_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                step=self.win_size // 2,
                target_channels=self.target_channels,
                mode='train'
            )
        
        self.test_loader, self.test_dataset = get_esa_loader(
            csv_path=self.test_csv_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            target_channels=self.target_channels,
            mode='test'
        )
    
    def _load_fadsd_data(self):
        """Load data for fadsd model"""
        if hasattr(self, 'train_csv_path'):
            train_loader_obj = ESADataLoaderfadsd(
                csv_path=self.train_csv_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                win_size_1=self.win_size_1,
                count=self.count,
                target_channels=self.target_channels,
                mode='train',
                step=self.win_size // 2
            )
            self.train_loader = train_loader_obj.loader
            self.train_dataset = train_loader_obj
        
        test_loader_obj = ESADataLoaderfadsd(
            csv_path=self.test_csv_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            win_size_1=self.win_size_1,
            count=self.count,
            target_channels=self.target_channels,
            mode='test',
            step=self.step
        )
        self.test_loader = test_loader_obj.loader
        self.test_dataset = test_loader_obj
        
    def build_model(self):
        """Build model based on model_type"""
        if self.model_type == 'LFTSAD':
            self.model = LFTSAD(
                win_size=self.win_size,
                enc_in=self.n_channels,
                c_out=self.n_channels,
                d_model=getattr(self, 'd_model', 512),
                patch_size=getattr(self, 'patch_size', [5]),
                channel=self.n_channels,
                patch_seq=getattr(self, 'patch_seq', [20]),
                seq_size=getattr(self, 'seq_size', 20)
            )
            
            if torch.cuda.is_available():
                self.model.cuda()
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=getattr(self, 'lr', 1e-4)
            )
            
        elif self.model_type == 'fadsd':
            self.model = FADSD(
                p=self.p,
                select=self.select
            )
            
            if torch.cuda.is_available():
                self.model.cuda()
            
            # fadsd is training-free, no optimizer needed
            print("fadsd is training-free (no learnable parameters)")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
    def train(self):
        """Train the model (only applicable to LFTSAD)"""
        if self.model_type == 'fadsd':
            print("fadsd is training-free, skipping training")
            return
        
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
        """Test the model and compute metrics"""
        print("\n" + "="*60)
        print("Starting Testing and Evaluation")
        print("="*60)
        
        self.model.eval()
        
        if self.model_type == 'LFTSAD':
            return self._test_lftsad()
        elif self.model_type == 'fadsd':
            return self._test_fadsd()
    
    def _test_lftsad(self):
        """Test LFTSAD model"""
        all_scores = []
        all_labels = []
        all_start_indices = []
        
        with torch.no_grad():
            for i, (input_data, labels, start_indices) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)
                
                # Forward pass
                series, prior, series_seq, prior_seq = self.model(input)
                
                # Compute anomaly scores
                loss = 0
                for u in range(len(prior)):
                    loss += (
                        getattr(self, 'p_seq', 0.5) * self.criterion_keep(series_seq[u], prior_seq[u]) +
                        (1 - getattr(self, 'p_seq', 0.5)) * self.criterion_keep(series[u], prior[u])
                    )
                
                scores = loss.detach().cpu().numpy()
                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy())
                all_start_indices.extend(start_indices.numpy())
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(self.test_loader)} batches")
        
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\nCollected scores shape: {all_scores.shape}")
        print(f"Collected labels shape: {all_labels.shape}")
        
        # Reconstruct full time series
        ground_truth, anomaly_scores = self._reconstruct_from_windows(
            all_scores, all_labels, all_start_indices
        )
        
        return self._evaluate_predictions(ground_truth, anomaly_scores)
    
    def _test_fadsd(self):
        """Test fadsd model"""
        all_scores = []
        all_labels = []
        all_start_indices = []
        
        with torch.no_grad():
            for i, (input_data, data_global, labels, start_indices) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)  # (B, win_size, C)
                data_global = data_global.float().to(self.device)  # (B, count, win_size_1, C)
                
                # Forward pass - returns scalar score per sample
                scores = self.model(input, data_global)  # (B,)
                
                # Expand to window size and channels
                # scores shape: (B,) -> (B, win_size, C)
                scores_expanded = scores.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, self.win_size, self.n_channels
                )
                
                scores_np = scores_expanded.detach().cpu().numpy()
                all_scores.append(scores_np)
                all_labels.append(labels.cpu().numpy())
                all_start_indices.extend(start_indices.numpy())
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(self.test_loader)} batches")
        
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\nCollected scores shape: {all_scores.shape}")
        print(f"Collected labels shape: {all_labels.shape}")
        
        # Reconstruct full time series
        ground_truth, anomaly_scores = self._reconstruct_from_windows(
            all_scores, all_labels, all_start_indices
        )
        
        return self._evaluate_predictions(ground_truth, anomaly_scores)
    
    def _reconstruct_from_windows(self, scores, labels, start_indices):
        """
        Reconstruct full time series from overlapping windows
        
        Args:
            scores: (n_windows, win_size, n_channels)
            labels: (n_windows, win_size, n_channels)
            start_indices: list of starting indices
            
        Returns:
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

            score_sum[start_idx:end_idx] += scores[i, :window_len, :]
            score_count[start_idx:end_idx] += 1

            label_array[start_idx:end_idx] = np.maximum(
                label_array[start_idx:end_idx],
                labels[i, :window_len, :]
            )

        score_count[score_count == 0] = 1
        anomaly_scores = score_sum / score_count[:, None]

        return label_array, anomaly_scores
    
    def _evaluate_predictions(self, ground_truth, anomaly_scores):
        """
        Evaluate predictions and compute all metrics
        
        Args:
            ground_truth: (n_samples, n_channels)
            anomaly_scores: (n_samples, n_channels)
        """
        print(f"Reconstructed ground truth shape: {ground_truth.shape}")
        
        # Compute threshold
        thresh = np.percentile(anomaly_scores, 100 - self.anormly_ratio, axis=0)
        pred_binary = (anomaly_scores > thresh[None, :]).astype(int)
        thresh_str = np.array2string(thresh, formatter={'float_kind': lambda x: f"{x:.6f}"})
        print(f"Thresholds (at {getattr(self, 'anormly_ratio', 1.0)}% anomaly ratio): {thresh_str}")
        
        # Standard binary classification metrics
        print("\n" + "="*60)
        print("STANDARD METRICS (Channel-Aggregated)")
        print("="*60)
        
        gt_any = ground_truth.any(axis=1).astype(int)
        pred_any = pred_binary.any(axis=1).astype(int)
        
        accuracy = accuracy_score(gt_any, pred_any)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt_any, pred_any, average='binary', zero_division=0
        )
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f_score:.4f}")
        
        # ESA Metrics
        if self.use_esa_metrics:
            self._compute_esa_metrics(
                predictions=pred_binary,
                ground_truth=ground_truth,
                anomaly_scores=anomaly_scores
            )
        
        return accuracy, precision, recall, f_score
    
    def _compute_esa_metrics(self, predictions, ground_truth, anomaly_scores):
        """Compute ESA-specific metrics"""
        print("\n" + "="*60)
        print("ESA ANOMALY DETECTION BENCHMARK METRICS")
        print("="*60)
        
        timestamps = self.test_dataset.get_full_timestamps()
        full_range = (timestamps.iloc[0], timestamps.iloc[-1])
        
        y_true_df = self.labels_parser.get_labels_dataframe(
            channel_filter=self.channel_names
        )
        
        print(f"\nGround truth events: {len(y_true_df)}")
        print(f"Unique anomaly IDs: {y_true_df['ID'].nunique()}")
        print(f"Time range: {full_range[0]} to {full_range[1]}")
        
        # Create predictions in ESA format
        y_pred_dict = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            y_pred_channel = []
            for i in range(len(predictions)):
                y_pred_channel.append([timestamps.iloc[i], int(predictions[i, ch_idx])])
            y_pred_dict[ch_name] = y_pred_channel
        
        # Compute ESA metrics
        print("\n--- Event-wise and Affiliation-based Scores ---")
        try:
            esa_metric = ESAScores(
                betas=self.beta,
                full_range=full_range,
                select_labels={"Category": ["Anomaly"]}
            )
            
            y_pred_first = y_pred_dict[self.channel_names[0]]
            esa_results = esa_metric.score(y_true_df, y_pred_first)
            
            for metric_name, value in esa_results.items():
                print(f"{metric_name:30s}: {value:8.4f}")
                
        except Exception as e:
            print(f"Error computing ESA scores: {e}")
        
        print("\n--- Channel-Aware F-Score ---")
        try:
            channel_metric = ChannelAwareFScore(
                beta=self.beta if isinstance(self.beta, float) else self.beta,
                full_range=full_range,
                select_labels={"Category": ["Anomaly"]}
            )
            
            channel_results = channel_metric.score(y_true_df, y_pred_dict)
            
            for metric_name, value in channel_results.items():
                print(f"{metric_name:30s}: {value:8.4f}")
                
        except Exception as e:
            print(f"Error computing channel-aware metrics: {e}")
        
        print("\n--- ADTQC Latency Metrics ---")
        try:
            adtqc_metric = ADTQC(
                full_range=full_range,
                select_labels={"Category": ["Anomaly"]}
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
