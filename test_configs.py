#!/usr/bin/env python3
"""
Baseline Testing Script for AutoML Text Classification

This script reads baseline configuration from configs/baseline_config.yaml
and runs all specified experiments, saving results to YAML files.
"""

import subprocess
import yaml
import time
from pathlib import Path
from datetime import datetime
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="configs/baseline_config.yaml"):
    """Load baseline configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config

def build_command(dataset, approach, config, data_path_override=None, output_path_override=None):
    """Build command arguments from config"""
    global_settings = config.get('global_settings', {})
    approach_specific = config.get('approach_specific', {}).get(approach, {})
    
    # Start with base command
    cmd = ["python", "run.py"]
    
    # Add required arguments
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--approach", approach])
    
    # Add data path (from override or config)
    if data_path_override:
        cmd.extend(["--data-path", data_path_override])
    elif global_settings.get('data_path') is not None:
        cmd.extend(["--data-path", str(global_settings['data_path'])])
    
    # Add global settings (excluding data_path since we handle it separately)
    for key, value in global_settings.items():
        if key == 'use_hpo' and value:
            cmd.append("--use-hpo")
        elif key not in ['use_hpo', 'data_path']:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Add approach-specific settings
    for key, value in approach_specific.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Set output path (from override or config)
    if output_path_override:
        base_output_path = output_path_override
    else:
        base_output_path = config.get('output', {}).get('base_path', 'results/baseline')
    
    output_path = Path(base_output_path) / f"{dataset}_{approach}"
    cmd.extend(["--output-path", str(output_path)])
    
    return cmd, output_path

def run_baseline_experiment(dataset, approach, config, data_path_override=None, output_path_override=None):
    """Run a single baseline experiment using config"""
    
    cmd, output_path = build_command(dataset, approach, config, data_path_override, output_path_override)
    
    logger.info(f"Running: {dataset} with {approach}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()
        
        runtime = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"Success: {dataset}_{approach} (Runtime: {runtime:.2f}s)")
            
            # Read the score file if it exists
            score_file = output_path / "score.yaml"
            score_data = {}
            if score_file.exists():
                with open(score_file, 'r') as f:
                    score_data = yaml.safe_load(f)
            
            return {
                "status": "success",
                "runtime": runtime,
                "val_err": score_data.get("val_err", None),
                "test_err": score_data.get("test_err", None),
                "command": ' '.join(cmd),
                "output_path": str(output_path)
            }
        else:
            logger.error(f"Failed: {dataset}_{approach}")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            
            return {
                "status": "failed",
                "return_code": result.returncode,
                "command": ' '.join(cmd),
                "error_message": result.stderr,
                "output_path": str(output_path)
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {dataset}_{approach}")
        return {
            "status": "timeout",
            "command": ' '.join(cmd),
            "error_message": "Process timed out after 1 hour",
            "output_path": str(output_path)
        }
    except Exception as e:
        logger.error(f"💥 Exception: {dataset}_{approach} - {str(e)}")
        return {
            "status": "exception",
            "error_message": str(e),
            "command": ' '.join(cmd),
            "output_path": str(output_path)
        }

def main():
    """Run all baseline experiments from config and collect results"""
    
    parser = argparse.ArgumentParser(description="Run baseline experiments from config")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/baseline_config.yaml",
        help="Path to the baseline configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the dataset directory (overrides config)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Base output path for results (overrides config)"
    )
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        return
    
    datasets = config['experiments']['datasets']
    approaches = config['experiments']['approaches']
    
    # Create results directory
    if args.output_path:
        base_output_path = args.output_path
    else:
        base_output_path = config.get('output', {}).get('base_path', 'results/baseline')
    
    results_dir = Path(base_output_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary
    all_results = {
        "experiment_info": {
            "config_file": args.config,
            "timestamp": datetime.now().isoformat(),
            "experiment_name": config.get('experiment_name', 'baseline_experiments'),
            "description": config.get('description', ''),
            "datasets": datasets,
            "approaches": approaches,
            "total_experiments": len(datasets) * len(approaches),
            "global_settings": config.get('global_settings', {}),
            "approach_specific": config.get('approach_specific', {})
        },
        "results": {}
    }
    
    logger.info(f"Starting baseline experiments from config: {args.config}")
    logger.info(f"Experiment: {config.get('experiment_name', 'baseline_experiments')}")
    logger.info(f"Total experiments: {len(datasets)} datasets × {len(approaches)} approaches = {len(datasets) * len(approaches)}")
    
    # Run all combinations
    total_start_time = time.time()
    
    for dataset in datasets:
        all_results["results"][dataset] = {}
        
        for approach in approaches:
            experiment_key = f"{dataset}_{approach}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment: {experiment_key}")
            logger.info(f"{'='*60}")
            
            result = run_baseline_experiment(dataset, approach, config, args.data_path, args.output_path)
            all_results["results"][dataset][approach] = result
            
            # Save intermediate results after each experiment
            results_file = results_dir / "baseline_results.yaml"
            with open(results_file, 'w') as f:
                yaml.safe_dump(all_results, f, default_flow_style=False, indent=2)
    
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    # Add summary statistics
    all_results["summary"] = {
        "total_runtime": total_runtime,
        "successful_experiments": sum(1 for dataset in datasets for approach in approaches 
                                    if all_results["results"][dataset][approach]["status"] == "success"),
        "failed_experiments": sum(1 for dataset in datasets for approach in approaches 
                                if all_results["results"][dataset][approach]["status"] == "failed"),
        "timeout_experiments": sum(1 for dataset in datasets for approach in approaches 
                                 if all_results["results"][dataset][approach]["status"] == "timeout"),
        "exception_experiments": sum(1 for dataset in datasets for approach in approaches 
                                   if all_results["results"][dataset][approach]["status"] == "exception")
    }
    
    # Save final results
    final_results_file = results_dir / "baseline_results.yaml"
    with open(final_results_file, 'w') as f:
        yaml.safe_dump(all_results, f, default_flow_style=False, indent=2)
    
    # Create a summary table if specified in config
    if config.get('output', {}).get('create_summary', True):
        create_summary_table(all_results, results_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE EXPERIMENTS COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    logger.info(f"Successful: {all_results['summary']['successful_experiments']}")
    logger.info(f"Failed: {all_results['summary']['failed_experiments']}")
    logger.info(f"Timeout: {all_results['summary']['timeout_experiments']}")
    logger.info(f"Exception: {all_results['summary']['exception_experiments']}")
    logger.info(f"Results saved to: {final_results_file}")

def create_summary_table(all_results, results_dir):
    """Create a summary table of results"""
    
    datasets = all_results["experiment_info"]["datasets"]
    approaches = all_results["experiment_info"]["approaches"]
    
    # Create performance summary
    summary_table = {
        "experiment_summary": {
            "config_file": all_results["experiment_info"]["config_file"],
            "experiment_name": all_results["experiment_info"]["experiment_name"],
            "timestamp": all_results["experiment_info"]["timestamp"],
            "total_experiments": all_results["experiment_info"]["total_experiments"]
        },
        "performance_summary": {},
        "runtime_summary": {},
        "status_summary": {}
    }
    
    for dataset in datasets:
        summary_table["performance_summary"][dataset] = {}
        summary_table["runtime_summary"][dataset] = {}
        summary_table["status_summary"][dataset] = {}
        
        for approach in approaches:
            result = all_results["results"][dataset][approach]
            
            summary_table["status_summary"][dataset][approach] = result["status"]
            
            if result["status"] == "success":
                val_err = result.get("val_err", None)
                test_err = result.get("test_err", None)
                runtime = result.get("runtime", None)
                
                summary_table["performance_summary"][dataset][approach] = {
                    "val_error": val_err,
                    "test_error": test_err,
                    "val_accuracy": 1 - val_err if val_err is not None else None,
                    "test_accuracy": 1 - test_err if test_err is not None else None
                }
                summary_table["runtime_summary"][dataset][approach] = f"{runtime:.2f}s" if runtime else "N/A"
            else:
                summary_table["performance_summary"][dataset][approach] = {
                    "status": result["status"],
                    "error": result.get("error_message", "")
                }
                summary_table["runtime_summary"][dataset][approach] = result["status"]
    
    # Add overall statistics
    summary_table["overall_statistics"] = all_results["summary"]
    
    # Save summary
    summary_file = results_dir / "baseline_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.safe_dump(summary_table, f, default_flow_style=False, indent=2)
    
    logger.info(f"Summary table saved to: {summary_file}")
    
    # Create a simple performance matrix for quick viewing
    create_performance_matrix(summary_table, results_dir)

def create_performance_matrix(summary_table, results_dir):
    """Create a simple performance matrix for quick viewing"""
    
    datasets = list(summary_table["performance_summary"].keys())
    approaches = list(summary_table["performance_summary"][datasets[0]].keys()) if datasets else []
    
    # Create accuracy matrix
    accuracy_matrix = {
        "validation_accuracy_matrix": {},
        "test_accuracy_matrix": {}
    }
    
    for dataset in datasets:
        accuracy_matrix["validation_accuracy_matrix"][dataset] = {}
        accuracy_matrix["test_accuracy_matrix"][dataset] = {}
        
        for approach in approaches:
            perf = summary_table["performance_summary"][dataset][approach]
            
            if isinstance(perf, dict) and "val_accuracy" in perf:
                val_acc = perf["val_accuracy"]
                test_acc = perf["test_accuracy"]
                
                accuracy_matrix["validation_accuracy_matrix"][dataset][approach] = f"{val_acc:.4f}" if val_acc is not None else "N/A"
                accuracy_matrix["test_accuracy_matrix"][dataset][approach] = f"{test_acc:.4f}" if test_acc is not None else "N/A"
            else:
                accuracy_matrix["validation_accuracy_matrix"][dataset][approach] = "FAILED"
                accuracy_matrix["test_accuracy_matrix"][dataset][approach] = "FAILED"
    
    # Save matrix
    matrix_file = results_dir / "performance_matrix.yaml"
    with open(matrix_file, 'w') as f:
        yaml.safe_dump(accuracy_matrix, f, default_flow_style=False, indent=2)
    
    logger.info(f"Performance matrix saved to: {matrix_file}")

if __name__ == "__main__":
    main()
