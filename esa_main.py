"""
Complete example of using LFTSAD with ESA Anomaly Detection Benchmark data

This script shows:
1. How to set up the configuration
2. How to run training and testing
3. How to interpret the ESA metrics
4. How to save and visualize results
"""
import time
import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from esa_solver_complete import ESASolver


def setup_esa_experiment(
    data_dir,
    dataset_name="3_months",
    target_channels=None,
    experiment_name="lftsad_esa_experiment"
):
    """
    Set up configuration for ESA experiment
    
    Args:
        data_dir: Directory containing ESA data files
        dataset_name: Name of the dataset (e.g., "84_months", "42_months")
        target_channels: List of channel names or None for all
        experiment_name: Name for this experiment
        
    Returns:
        config dict
    """
    
    # Default channel subset (from mission1_experiments.py)
    if target_channels is None:
        target_channels = [
            "channel_41", "channel_42", "channel_43",
            "channel_44", "channel_45", "channel_46"
        ]
    
    config = {
        # Experiment info
        'experiment_name': experiment_name,
        
        # Data paths
        'test_csv_path': os.path.join(data_dir, 'preprocessed', 'multivariate', 
                                      'ESA-Mission1-semi-supervised', f'{dataset_name}.test.csv'),
        'labels_csv_path': os.path.join(data_dir, 'ESA-Mission1', 'labels.csv'),
        
        # Optional: if you have training data
        'train_csv_path': os.path.join(data_dir, 'preprocessed', 'multivariate', 
                                        'ESA-Mission1-semi-supervised', f'{dataset_name}.train.csv'),
        
        # Channel selection
        'target_channels': target_channels,
        
        # Model architecture
        'batch_size': 32,
        'win_size': 100,
        'step': 1,  # Step=1 for test to get predictions for all timestamps
        'd_model': 512,
        'patch_size': [10],
        'patch_seq': [5],
        'seq_size': 20,
        
        # Training (if applicable)
        'num_epochs': 10,
        'lr': 1e-4,
        'p_seq': 0.5,
        'sw_max_mean': 0,
        'sw_loss': 0,
        
        # Evaluation
        'anormly_ratio': 1.0,  # Percentage of samples to mark as anomalies
        'use_esa_metrics': True,
        'beta': 0.5,  # Beta for F-beta score (0.5 favors precision)
        
        # Output
        'output_dir': f'/content/drive/MyDrive/LFTSAD_ESA_ADB/results/{experiment_name}',
    }
    
    # Create output directory
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
    print("="*80)
    
    # Save configuration
    config_path = os.path.join(config['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        # Convert non-serializable items
        config_serializable = {k: v for k, v in config.items() 
                              if not callable(v)}
        json.dump(config_serializable, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Create solver
    solver = ESASolver(config)
    
    
    # Train if training data is provided
    if hasattr(solver, 'train_loader'):
        print("\nTraining phase enabled")
        t_train_start = time.time()
        solver.train()
        t_train_end = time.time()
        t_train = t_train_end - t_train_start
        # Save model
        model_path = os.path.join(config['output_dir'], 'model.pth')
        torch.save(solver.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    else:
        print("\nNo training data provided, skipping training")
   

    

    # Test and evaluate
    if solver.use_esa_metrics:

        accuracy, precision, recall, f_score, esa_results, channel_results, adtqc, t_test = solver.test()
        
        if hasattr(solver, 'train_loader'):

            results = {
                'training time': float(t_train),
                'testing time': float(t_test),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f_score),
                **esa_results,
                **channel_results,
                **adtqc,
            }

        else: 

            results = {
                'testing time': float(t_test),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f_score),
                **esa_results,
                **channel_results,
                **adtqc,
            }
    else: 

        accuracy, precision, recall, f_score, t_test = solver.test()

        if hasattr(solver, 'train_loader'):
        # Collect results
            results = {
                'training time': float(t_train),
                'testing time': float(t_test),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f_score),
            }
        else:
            results = {
                'testing time': float(t_test),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f_score),
            }
            
    # Save results
    results_path = os.path.join(config['output_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def compare_channel_subsets(data_dir, dataset_name="84_months"):
    """
    Run experiments with different channel subsets and compare results
    
    This is useful for understanding which channels are most informative
    """
    
    # Define different channel subsets to test
    channel_subsets = {
        'subset_6ch': [
            "channel_41", "channel_42", "channel_43",
            "channel_44", "channel_45", "channel_46"
        ],
        'subset_12ch': [
            "channel_12", "channel_13", "channel_14", "channel_15",
            "channel_41", "channel_42", "channel_43", "channel_44",
            "channel_45", "channel_46", "channel_70", "channel_71"
        ],
        # Add more subsets as needed
    }
    
    all_results = {}
    
    for subset_name, channels in channel_subsets.items():
        print(f"\n{'='*80}")
        print(f"Testing channel subset: {subset_name} ({len(channels)} channels)")
        print(f"{'='*80}")
        
        config = setup_esa_experiment(
            data_dir=data_dir,
            dataset_name=dataset_name,
            target_channels=channels,
            experiment_name=f"lftsad_{dataset_name}_{subset_name}"
        )
        
        results = run_experiment(config)
        all_results[subset_name] = results
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON OF CHANNEL SUBSETS")
    print("="*80)
    
    comparison_df = pd.DataFrame(all_results).T
    print(comparison_df.to_string())
    
    # Save comparison
    comparison_path = f'results/channel_subset_comparison_{dataset_name}.csv'
    comparison_df.to_csv(comparison_path)
    print(f"\nComparison saved to: {comparison_path}")
    
    return all_results


def interpret_esa_metrics():
    """
    Guide for interpreting ESA metrics
    """
    
    guide = """
    ========================================================================
    GUIDE TO INTERPRETING ESA METRICS
    ========================================================================
    
    1. EVENT-WISE METRICS (EW_*)
    ----------------------------
    - EW_precision: Precision at the event level (not point level)
      → What fraction of detected events are true anomalies?
    
    - EW_recall: Recall at the event level
      → What fraction of true anomaly events were detected?
    
    - EW_F_0.50: F-beta score with beta=0.5 (favors precision)
      → Good for applications where false alarms are costly
    
    - alarming_precision: Precision accounting for redundant detections
      → How well does the system avoid multiple alarms for same event?
    
    2. AFFILIATION-BASED METRICS (AFF_*)
    ------------------------------------
    - AFF_precision: Precision considering temporal overlap
      → Rewards early/on-time detection
    
    - AFF_recall: Recall considering temporal affiliation
      → More forgiving than event-wise recall
    
    - AFF_F_0.50: Affiliation-based F-beta score
      → Usually higher than EW metrics due to temporal credit
    
    3. ADTQC LATENCY METRICS
    ------------------------
    - Nb_Before: Number of detections BEFORE anomaly starts
      → Early detections (good for prevention)
    
    - Nb_After: Number of detections AFTER anomaly starts
      → Late detections (still useful but less ideal)
    
    - AfterRate: Fraction of late detections
      → Lower is better (more early detections)
    
    - Total: Overall ADTQC score
      → Accounts for timing quality of detections
      → Higher is better
    
    4. CHANNEL-AWARE METRICS
    ------------------------
    - Account for which channels showed anomalies
    - Important for multi-channel satellite telemetry
    - Help understand if model detects anomalies in correct channels
    
    ========================================================================
    TYPICAL GOOD VALUES (depends on application):
    
    - EW_F_0.50 > 0.7: Good event detection
    - AFF_F_0.50 > 0.8: Good with timing consideration
    - AfterRate < 0.3: Mostly early detections
    - ADTQC Total > 0.7: Good timing quality
    
    Note: These are guidelines - actual requirements depend on your use case!
    ========================================================================
    """
    
    print(guide)
    return guide


# Main execution
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

    args = parser.parse_args()

    config = setup_esa_experiment(
        data_dir=DATA_DIR,
        dataset_name=args.dataset,
        target_channels=[
            "channel_41", "channel_42", "channel_43",
            "channel_44", "channel_45", "channel_46"
        ],
        experiment_name=f"lftsad_{args.dataset}_6ch"
    )
    
    results = run_experiment(config)
    
    print("\n\nFinal Results Summary:")
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.4f}")
    
    
    # Example 2: Compare different channel subsets
    # Uncomment to run comparison
    """
    print("\n\n\nEXAMPLE 2: Channel Subset Comparison")
    print("="*80)
    
    comparison_results = compare_channel_subsets(
        data_dir=DATA_DIR,
        dataset_name="84_months"
    )
    """
    
    
    # Example 3: Run on multiple datasets
    # Uncomment to test on different time periods
    """
    print("\n\n\nEXAMPLE 3: Multiple Datasets")
    print("="*80)
    
    datasets = ["3_months", "10_months", "21_months", "42_months", "84_months"]
    
    for dataset in datasets:
        config = setup_esa_experiment(
            data_dir=DATA_DIR,
            dataset_name=dataset,
            experiment_name=f"lftsad_{dataset}"
        )
        
        results = run_experiment(config)
        print(f"\n{dataset} - F1: {results['f1_score']:.4f}")
    """
