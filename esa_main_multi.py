"""
Enhanced ESA main script supporting both LFTSAD and fadsd models
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from esa_solver_fadsd import ESASolverMultiModel


def setup_esa_experiment_lftsad(
    data_dir,
    dataset_name="3_months",
    target_channels=None,
    experiment_name="lftsad_esa_experiment"
):
    """
    Set up configuration for LFTSAD experiment
    
    Args:
        data_dir: Directory containing ESA data files
        dataset_name: Name of the dataset (e.g., "84_months", "42_months")
        target_channels: List of channel names or None for all
        experiment_name: Name for this experiment
        
    Returns:
        config dict
    """
    
    if target_channels is None:
        target_channels = [
            "channel_41", "channel_42", "channel_43",
            "channel_44", "channel_45", "channel_46"
        ]
    
    config = {
        # Model type
        'model_type': 'LFTSAD',
        
        # Experiment info
        'experiment_name': experiment_name,
        
        # Data paths
        'test_csv_path': os.path.join(data_dir, 'preprocessed', 'multivariate', 
                                      'ESA-Mission1-semi-supervised', f'{dataset_name}.train.csv'),
        'labels_csv_path': os.path.join(data_dir, 'ESA-Mission1', 'labels.csv'),
        
        # Optional: if you have training data
        'train_csv_path': os.path.join(data_dir, 'preprocessed', 'multivariate', 
                                        'ESA-Mission1-semi-supervised', f'{dataset_name}.train.csv'),
        
        # Channel selection
        'target_channels': target_channels,
        
        # Model architecture
        'batch_size': 32,
        'win_size': 100,
        'step': 1,
        'd_model': 512,
        'patch_size': [10],
        'patch_seq': [5],
        'seq_size': 20,
        
        # Training (if applicable)
        'num_epochs': 10,
        'lr': 1e-4,
        'p_seq': 0.5,
        
        # Evaluation
        'anormly_ratio': 1.0,
        'use_esa_metrics': True,
        'beta': 0.5,
        
        # Output
        'output_dir': f'results/{experiment_name}',
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    return config


def setup_esa_experiment_fadsd(
    data_dir,
    dataset_name="3_months",
    target_channels=None,
    experiment_name="fadsd_esa_experiment",
    win_size=10,
    win_size_1=20,
    count=21,
    p=0.1,
    select=1
):
    """
    Set up configuration for fadsd experiment
    
    Args:
        data_dir: Directory containing ESA data files
        dataset_name: Name of the dataset
        target_channels: List of channel names or None for all
        experiment_name: Name for this experiment
        win_size: Local window size (default 10)
        win_size_1: Global window size (default 20)
        count: Number of global windows, must be odd (default 21)
        p: Weight for local vs global score (default 0.1)
        select: 0 for magnitude, 1 for phase (default 1)
        
    Returns:
        config dict
    """
    
    if target_channels is None:
        target_channels = [
            "channel_41", "channel_42", "channel_43",
            "channel_44", "channel_45", "channel_46"
        ]
    
    config = {
        # Model type
        'model_type': 'fadsd',
        
        # Experiment info
        'experiment_name': experiment_name,
        
        # Data paths
        'test_csv_path': os.path.join(data_dir, 'preprocessed', 'multivariate', 
                                      'ESA-Mission1-semi-supervised', f'{dataset_name}.train.csv'),
        'labels_csv_path': os.path.join(data_dir, 'ESA-Mission1', 'labels.csv'),
        
        # Channel selection
        'target_channels': target_channels,
        
        # fadsd specific parameters
        'win_size': win_size,           # Local window size
        'win_size_1': win_size_1,       # Global window size
        'count': count,                 # Number of global windows (must be odd)
        'p': p,                         # Weight: p*local + (1-p)*global
        'select': select,               # 0=magnitude, 1=phase
        
        # Data loading
        'batch_size': 256,
        'step': 1,                      # Step=1 for full coverage
        
        # Evaluation
        'anormly_ratio': 0.9,           # Top X% as anomalies
        'use_esa_metrics': True,
        'beta': 0.5,
        
        # Output
        'output_dir': f'results/{experiment_name}',
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    return config


def run_experiment(config):
    """
    Run complete experiment with ESA metrics
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Results including all metrics
    """
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {config['experiment_name']}")
    print(f"MODEL: {config['model_type']}")
    print("="*80)
    
    # Save configuration
    config_path = os.path.join(config['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        config_serializable = {k: v for k, v in config.items() 
                              if not callable(v)}
        json.dump(config_serializable, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Create solver
    solver = ESASolverMultiModel(config)
    
    # Train if applicable (only for LFTSAD)
    if config['model_type'] == 'LFTSAD' and hasattr(solver, 'train_loader'):
        print("\nTraining phase enabled")
        solver.train()
        
        model_path = os.path.join(config['output_dir'], 'model.pth')
        torch.save(solver.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    else:
        print(f"\n{config['model_type']} is training-free, skipping training")
    
    # Test and evaluate
    accuracy, precision, recall, f_score = solver.test()
    
    # Collect results
    results = {
        'model_type': config['model_type'],
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f_score),
    }
    
    # Add model-specific parameters
    if config['model_type'] == 'fadsd':
        results.update({
            'win_size': config['win_size'],
            'win_size_1': config['win_size_1'],
            'count': config['count'],
            'p': config['p'],
            'select': config['select']
        })
    
    # Save results
    results_path = os.path.join(config['output_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    
    # Print interpretation guide
    #interpret_esa_metrics()
    
    # Set your data directory
    DATA_DIR = "data"
    
    # Example 1: Single experiment with one channel subset
    print("\n\nEXAMPLE 1: Single Experiment")
    print("="*80)
    
    parser = argparse.ArgumentParser()

    dataset_name="3_months"

    parser.add_argument('--dataset', type=str, default=f'{dataset_name}')
    parser.add_argument('--model', type=str, default='LFTSAD')

    args = parser.parse_args()

    if args.model == 'LFTSAD':

        config = setup_esa_experiment_lftsad(
            data_dir=DATA_DIR,
            dataset_name=args.dataset,
            target_channels=[
                "channel_41", "channel_42", "channel_43",
                "channel_44", "channel_45", "channel_46"
            ],
            experiment_name=f"lftsad_{args.dataset}_6ch"
        )
    
    elif args.model == 'FADSD':
    
        config = setup_esa_experiment_fadsd(
            data_dir=DATA_DIR,
            dataset_name=args.dataset,
            target_channels=[
                "channel_41", "channel_42", "channel_43",
                "channel_44", "channel_45", "channel_46"
            ],
            experiment_name=f"FADSD_{args.dataset}_6ch"
        )

        
    results = run_experiment(config)
    
    print("\n\nFinal Results Summary:")
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.4f}")
    

    