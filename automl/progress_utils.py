"""Progress tracking utilities for AutoML training."""

import time
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and display training progress with time estimates."""
    
    def __init__(self, total_epochs, dataset_name, approach, extra_info=""):
        self.total_epochs = total_epochs
        self.dataset_name = dataset_name
        self.approach = approach
        self.extra_info = extra_info
        self.start_time = time.time()
        self.epoch_times = []
        
        # Print header
        print("\n" + "="*80)
        print(f"TRAINING STARTED: {dataset_name.upper()} - {approach.upper()}")
        if extra_info:
            print(f"Configuration: {extra_info}")
        print(f"Total epochs: {total_epochs}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def start_epoch(self, epoch):
        """Start tracking a new epoch."""
        self.epoch_start = time.time()
        self.current_epoch = epoch
        return epoch
    
    def end_epoch(self, epoch, metrics):
        """End epoch and display progress."""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Calculate time estimates
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta = timedelta(seconds=int(eta_seconds))
        
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Format metrics
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # Progress bar
        progress = (epoch + 1) / self.total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Print progress
        print(f"\rEpoch {epoch+1}/{self.total_epochs} [{bar}] {progress*100:.1f}% | "
              f"{metric_str} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta} | "
              f"Elapsed: {elapsed_str}", end="")
        
        if epoch + 1 == self.total_epochs:
            print()  # New line at the end
    
    def finish(self, final_metrics=None):
        """Print final summary."""
        total_time = time.time() - self.start_time
        print("\n" + "="*80)
        print(f"TRAINING COMPLETED: {self.dataset_name.upper()} - {self.approach.upper()}")
        print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        if final_metrics:
            print(f"Final metrics: {final_metrics}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")


def create_batch_progress_bar(dataloader, desc="Processing"):
    """Create a progress bar for batch processing."""
    return tqdm(dataloader, desc=desc, leave=False, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')


def log_training_stats(epoch, total_epochs, train_loss, val_metrics, time_elapsed):
    """Log training statistics in a formatted way."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Epoch {epoch+1}/{total_epochs} Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Training Loss: {train_loss:.4f}")
    for metric, value in val_metrics.items():
        logger.info(f"Validation {metric}: {value:.4f}")
    logger.info(f"Epoch Time: {time_elapsed:.2f}s")
    logger.info(f"{'='*60}\n")


class DataLoadingProgress:
    """Show progress while loading and preprocessing data."""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        print(f"\n📊 Loading {dataset_name.upper()} dataset...")
        self.start_time = time.time()
    
    def update(self, step, total_steps=None):
        """Update loading progress."""
        if total_steps:
            print(f"  ✓ {step} ({total_steps:,} samples)")
        else:
            print(f"  ✓ {step}")
    
    def finish(self):
        """Finish loading."""
        elapsed = time.time() - self.start_time
        print(f"  ✓ Dataset loaded in {elapsed:.1f}s\n")


class HPOProgress:
    """Track HPO trial progress."""
    
    def __init__(self, n_trials, sampler_name):
        self.n_trials = n_trials
        self.sampler_name = sampler_name
        self.start_time = time.time()
        self.best_value = None
        
        print("\n" + "="*80)
        print(f"HYPERPARAMETER OPTIMIZATION STARTED")
        print(f"Trials: {n_trials} | Sampler: {sampler_name}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def update_trial(self, trial_number, value, params):
        """Update after each trial."""
        if self.best_value is None or value < self.best_value:
            self.best_value = value
            is_best = " 🌟 NEW BEST!"
        else:
            is_best = ""
        
        elapsed = time.time() - self.start_time
        avg_time = elapsed / (trial_number + 1)
        eta = (self.n_trials - trial_number - 1) * avg_time
        
        print(f"\nTrial {trial_number+1}/{self.n_trials} completed{is_best}")
        print(f"  Value: {value:.4f} | Best so far: {self.best_value:.4f}")
        print(f"  Params: {params}")
        print(f"  Time: {elapsed:.1f}s | ETA: {timedelta(seconds=int(eta))}")
        print("-" * 80)
    
    def finish(self):
        """Print final HPO summary."""
        total_time = time.time() - self.start_time
        print("\n" + "="*80)
        print(f"HPO COMPLETED")
        if self.best_value is not None:
            print(f"Best value: {self.best_value:.4f}")
        else:
            print("No successful trials completed")
        print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        print("="*80 + "\n")