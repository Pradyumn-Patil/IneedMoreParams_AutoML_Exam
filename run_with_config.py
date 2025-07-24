#!/usr/bin/env python3
"""
Run AutoML experiments using configuration files.

Usage:
    python run_with_config.py --config configs/amazon_optimized.yaml --data-path /path/to/data
    python run_with_config.py --config configs/neps_comprehensive.yaml --data-path /path/to/data --dataset amazon
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_command(config, dataset=None, data_path=None, override_args=None):
    """Build command line arguments from configuration."""
    cmd = ['python', 'run.py']
    
    # Get global settings
    global_settings = config.get('global_settings', {})
    
    # Add data path
    if data_path:
        cmd.extend(['--data-path', str(data_path)])
    elif global_settings.get('data_path'):
        cmd.extend(['--data-path', str(global_settings['data_path'])])
        
    # Add dataset
    if dataset:
        cmd.extend(['--dataset', dataset])
    elif global_settings.get('dataset'):
        cmd.extend(['--dataset', global_settings['dataset']])
        
    # Add other global settings
    for key, value in global_settings.items():
        if key in ['data_path', 'dataset']:
            continue
            
        # Convert underscore to hyphen for command line args
        arg_name = key.replace('_', '-')
        
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{arg_name}')
        # Handle other values
        elif value is not None:
            cmd.extend([f'--{arg_name}', str(value)])
            
    # Add override arguments
    if override_args:
        cmd.extend(override_args)
        
    return cmd


def run_experiment(config_path, dataset=None, data_path=None, override_args=None):
    """Run a single experiment based on configuration."""
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    experiment_name = config.get('experiment_name', 'unnamed')
    logger.info(f"Running experiment: {experiment_name}")
    
    # Check if this is a multi-dataset configuration
    experiments = config.get('experiments', {})
    datasets_to_run = experiments.get('datasets', [])
    
    if dataset:
        # Run single dataset
        datasets_to_run = [dataset]
    elif not datasets_to_run and config.get('global_settings', {}).get('dataset'):
        # Single dataset config
        datasets_to_run = [config['global_settings']['dataset']]
        
    if not datasets_to_run:
        logger.error("No dataset specified. Use --dataset or specify in config.")
        return
        
    # Run experiments for each dataset
    for ds in datasets_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment for dataset: {ds}")
        logger.info(f"{'='*60}\n")
        
        # Get dataset-specific settings if available
        dataset_config = config.get('dataset_specific', {}).get(ds, {})
        
        # Build command
        cmd = build_command(config, dataset=ds, data_path=data_path, override_args=override_args)
        
        # Apply dataset-specific overrides
        for key, value in dataset_config.items():
            arg_name = key.replace('_', '-')
            # Remove existing arg if present
            try:
                idx = cmd.index(f'--{arg_name}')
                cmd.pop(idx)  # Remove flag
                if idx < len(cmd) and not cmd[idx].startswith('--'):
                    cmd.pop(idx)  # Remove value
            except ValueError:
                pass
            # Add new value
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{arg_name}')
            else:
                cmd.extend([f'--{arg_name}', str(value)])
                
        # Log the command
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run the command
        try:
            result = subprocess.run(cmd, check=True)
            logger.info(f"✓ Successfully completed experiment for {ds}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to run experiment for {ds}: {e}")
            continue
            
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed all experiments for {experiment_name}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Run AutoML experiments from configuration files')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str,
                        help='Specific dataset to run (overrides config)')
    parser.add_argument('--data-path', type=str,
                        help='Path to dataset directory (overrides config)')
    
    # Parse known args and pass the rest to run.py
    args, unknown_args = parser.parse_known_args()
    
    # Run the experiment
    run_experiment(
        config_path=args.config,
        dataset=args.dataset,
        data_path=args.data_path,
        override_args=unknown_args
    )


if __name__ == '__main__':
    main()