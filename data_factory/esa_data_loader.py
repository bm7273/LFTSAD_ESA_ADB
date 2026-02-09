"""
ESA Anomaly Detection Benchmark Data Loader for LFTSAD
Handles multi-channel satellite telemetry data with proper timestamp parsing
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader




class ESADataset(Dataset):
    """
    Dataset class for ESA satellite telemetry data
    
    Handles:
    - Multi-channel telemetry data
    - Per-channel anomaly labels
    - Timestamp preservation
    - Sliding window segmentation
    """
    
    def __init__(self, 
                 csv_path, 
                 win_size=100,
                 step=1,
                 target_channels=None,
                 mode='train'):
        """
        Args:
            csv_path: Path to ESA CSV file
            win_size: Window size for sliding window
            step: Step size for sliding window
            target_channels: List of channel names to use (None = all channels)
            mode: 'train', 'val', or 'test'
        """
        self.win_size = win_size
        self.step = step
        self.mode = mode
        
        # Load data
        print(f"Loading ESA data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Parse timestamp
        self.timestamps = pd.to_datetime(df["timestamp"], utc=True)
        
        # Identify telemetry channels and anomaly labels
        self.all_channels = [col for col in df.columns if col.startswith('channel_')]
        self.telecommand_cols = [col for col in df.columns if col.startswith('telecommand_')]
        
        # Select target channels
        if target_channels is None:
            self.target_channels = self.all_channels
        else:
            self.target_channels = [ch for ch in target_channels if ch in self.all_channels]
        
        print(f"Total channels: {len(self.all_channels)}")
        print(f"Using channels: {len(self.target_channels)}")
        
        # Extract telemetry data
        

        if self.mode == 'train':

            validation_date_split = self.timestamps.max() - pd.DateOffset(months=3)
            self.validation_date_split = validation_date_split

            train_df = df[self.timestamps <= validation_date_split]
            val_df   = df[self.timestamps >  validation_date_split]

            train_data = train_df[self.target_channels].values.astype(np.float32)

            self.scaler.fit(train_data)

            train_data = self.scaler.transform(train_data)
            val_data   = self.scaler.transform(                
                val_df[self.target_channels].values.astype(np.float32))

            self.train = train_data
            print("train:", self.train.shape)

            self.val = val_data
            print("val:", self.val.shape)

        else:

            self.data = df[self.target_channels].values.astype(np.float32)

        
        # Extract anomaly labels (per channel)
        self.label_columns = [f'is_anomaly_{ch}' for ch in self.target_channels]
        self.labels = df[self.label_columns].values.astype(np.float32)
        
        # Create channel mapping
        self.channel_mapping = {i: ch for i, ch in enumerate(self.target_channels)}
    
        
        
        # Create sliding windows
        self._create_windows()
        
        print(f"Dataset size: {len(self)} windows")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        
    def _create_windows(self):
        """Create sliding window indices"""
        self.window_indices = []
        n_samples = len(self.data)
        
        for i in range(0, n_samples - self.win_size + 1, self.step):
            self.window_indices.append(i)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            data: (win_size, n_channels) tensor
            labels: (win_size, n_channels) tensor
            timestamp: Start timestamp of window
        """
        start_idx = self.window_indices[idx]
        end_idx = start_idx + self.win_size
        
        data = torch.FloatTensor(self.data[start_idx:end_idx])
        labels = torch.FloatTensor(self.labels[start_idx:end_idx])
        
        # Return window data, labels, and timestamp info
        return data, labels, start_idx
    
    def get_timestamp_range(self, start_idx):
        """Get timestamp range for a window"""
        end_idx = start_idx + self.win_size
        return self.timestamps[start_idx], self.timestamps[end_idx-1]
    
    def get_full_timestamps(self):
        """Get all timestamps"""
        return self.timestamps
    
    def get_channel_names(self):
        """Get channel names in order"""
        return self.target_channels


def get_esa_loader(csv_path, 
                   batch_size=32,
                   win_size=100,
                   step=1,
                   target_channels=None,
                   mode='train',
                   num_workers=4):
    """
    Create DataLoader for ESA data
    
    Args:
        csv_path: Path to ESA CSV file
        batch_size: Batch size
        win_size: Window size
        step: Step size for windows
        target_channels: List of channel names or None for all
        mode: 'train', 'val', or 'test'
        num_workers: Number of workers for DataLoader
        
    Returns:
        DataLoader and dataset
    """
    dataset = ESADataset(
        csv_path=csv_path,
        win_size=win_size,
        step=step,
        target_channels=target_channels,
        mode=mode
    )
    
    shuffle = (mode == 'train')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, dataset


class ESALabelsParser:
    """
    Parse ESA labels.csv file for event-based evaluation
    """
    
    def __init__(self, labels_csv_path):
        """
        Args:
            labels_csv_path: Path to labels.csv file
        """
        self.labels_df = pd.read_csv(labels_csv_path)
        
        # Parse timestamps
        self.labels_df['StartTime'] = pd.to_datetime(self.labels_df['StartTime'], utc=True)
        self.labels_df['EndTime'] = pd.to_datetime(self.labels_df['EndTime'], utc=True)
        
        print(f"Loaded {len(self.labels_df)} anomaly events")
        print(f"Unique anomaly IDs: {self.labels_df['ID'].nunique()}")
        
    def get_labels_dataframe(self, channel_filter=None):
        """
        Get labels DataFrame, optionally filtered by channels
        
        Args:
            channel_filter: List of channel names to include, or None for all
            
        Returns:
            pandas DataFrame in ESA format
        """
        if channel_filter is None:
            return self.labels_df.copy()
        
        # Filter by channels
        filtered_df = self.labels_df[
            self.labels_df['Channel'].isin(channel_filter)
        ].copy()
        
        return filtered_df
    
    def get_full_range(self):
        """Get full time range of data"""
        return (
            self.labels_df['StartTime'].min(),
            self.labels_df['EndTime'].max()
        )
    
    def get_anomaly_categories(self):
        """Get unique anomaly categories"""
        return self.labels_df['Category'].unique()


