"""
Utility functions for neuroplasticity experiments.

This module contains common functions used across different neuroplasticity
implementations, including data generation, metrics calculation, and
visualization utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import optax


def generate_data(n_samples: int = 1000, input_dim: int = 10, output_dim: int = 1, 
                 noise_std: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for neuroplasticity experiments.
    
    Args:
        n_samples: Number of training samples
        input_dim: Input dimension
        output_dim: Output dimension
        noise_std: Standard deviation of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, Y) where X is input data and Y is target data
    """
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n_samples, input_dim))
    # Create a non-linear target: sum of squares + noise
    Y = np.sum(X**2, axis=1, keepdims=True) + np.random.normal(0, noise_std, (n_samples, output_dim))
    return X, Y


def dataset(X: np.ndarray, Y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    """
    Generate batches of data for training.
    
    Args:
        X: Input data
        Y: Target data
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        
    Yields:
        Batches of (X_batch, Y_batch)
    """
    n_samples = len(X)
    while True:
        if shuffle:
            indices = np.random.choice(n_samples, size=batch_size, replace=False)
        else:
            # Sequential batching
            indices = np.arange(n_samples) % n_samples
        yield X[indices], Y[indices]


def calculate_plasticity_metrics(model_variables: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various plasticity metrics from model variables.
    
    Args:
        model_variables: Dictionary containing model variables
        
    Returns:
        Dictionary with plasticity metrics
    """
    metrics = {}
    
    # Extract plasticity variables if they exist
    if 'plasticity' in model_variables:
        plasticity_vars = model_variables['plasticity']
        
        # Calculate Hebbian activity
        hebbian_total = 0.0
        structural_total = 0.0
        activity_total = 0.0
        
        for var_name, var_value in plasticity_vars.items():
            if 'hebbian' in var_name:
                hebbian_total += float(jnp.sum(var_value))
            elif 'connection_strength' in var_name:
                structural_total += float(jnp.mean(var_value))
            elif 'activity_history' in var_name:
                activity_total += float(jnp.mean(var_value))
        
        metrics['hebbian_activity'] = hebbian_total
        metrics['structural_plasticity'] = structural_total
        metrics['neural_activity'] = activity_total
    
    return metrics


def create_optimizer(learning_rate: float = 0.001, optimizer_type: str = 'adam') -> optax.GradientTransformation:
    """
    Create an optimizer for training.
    
    Args:
        learning_rate: Learning rate for the optimizer
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optax.adam(learning_rate)
    elif optimizer_type.lower() == 'sgd':
        return optax.sgd(learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        return optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def plot_training_progress(losses: list, metrics: Dict[str, list], save_path: Optional[str] = None):
    """
    Plot training progress and plasticity metrics.
    
    Args:
        losses: List of loss values over time
        metrics: Dictionary of metric lists
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Neuroplasticity Training Progress', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Hebbian activity
    if 'hebbian_activity' in metrics:
        axes[0, 1].plot(metrics['hebbian_activity'], 'r-', linewidth=2)
        axes[0, 1].set_title('Hebbian Activity')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Activity')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot structural plasticity
    if 'structural_plasticity' in metrics:
        axes[1, 0].plot(metrics['structural_plasticity'], 'g-', linewidth=2)
        axes[1, 0].set_title('Structural Plasticity')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Plasticity')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot neural activity
    if 'neural_activity' in metrics:
        axes[1, 1].plot(metrics['neural_activity'], 'm-', linewidth=2)
        axes[1, 1].set_title('Neural Activity')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Activity')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def print_training_header():
    """Print a formatted training header."""
    print("🧠 Neuroplasticity Training")
    print("=" * 50)
    print(f"{'Step':<6} {'Loss':<12} {'Hebbian':<12} {'Structural':<12} {'Activity':<12}")
    print("-" * 70)


def print_training_step(step: int, loss: float, metrics: Dict[str, float]):
    """
    Print a formatted training step.
    
    Args:
        step: Current training step
        loss: Current loss value
        metrics: Dictionary of current metrics
    """
    hebbian = metrics.get('hebbian_activity', 0.0)
    structural = metrics.get('structural_plasticity', 0.0)
    activity = metrics.get('neural_activity', 0.0)
    
    print(f"{step:2d}    {loss:10.6f} {hebbian:10.6f} {structural:10.6f} {activity:10.6f}")


def print_final_results(loss: float, metrics: Dict[str, float]):
    """
    Print final training results.
    
    Args:
        loss: Final loss value
        metrics: Dictionary of final metrics
    """
    print("-" * 70)
    print("🎉 Training completed!")
    print(f"📊 Final loss: {loss:.6f}")
    print(f"🧠 Final plasticity metrics:")
    for key, value in metrics.items():
        print(f"   - {key.replace('_', ' ').title()}: {value:.6f}")


def create_plasticity_config(plasticity_rate: float = 0.01, 
                           target_activity: float = 1.0,
                           scaling_factor: float = 0.001) -> Dict[str, float]:
    """
    Create a configuration dictionary for plasticity parameters.
    
    Args:
        plasticity_rate: Rate of Hebbian learning
        target_activity: Target activity level for synaptic scaling
        scaling_factor: Factor for synaptic scaling
        
    Returns:
        Dictionary with plasticity configuration
    """
    return {
        'plasticity_rate': plasticity_rate,
        'target_activity': target_activity,
        'scaling_factor': scaling_factor
    }


def validate_plasticity_config(config: Dict[str, float]) -> bool:
    """
    Validate plasticity configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = ['plasticity_rate', 'target_activity', 'scaling_factor']
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required config key: {key}")
            return False
    
    if config['plasticity_rate'] <= 0 or config['plasticity_rate'] > 1:
        print("❌ plasticity_rate must be between 0 and 1")
        return False
    
    if config['target_activity'] <= 0:
        print("❌ target_activity must be positive")
        return False
    
    if config['scaling_factor'] <= 0:
        print("❌ scaling_factor must be positive")
        return False
    
    return True


def setup_random_seed(seed: int = 42):
    """
    Set up random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    jax.config.update('jax_default_prng_impl', 'unsafe_rbg')


def calculate_network_complexity(model_params: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculate network complexity metrics.
    
    Args:
        model_params: Model parameters dictionary
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = 0
    total_layers = 0
    
    for param_name, param_value in model_params.items():
        if 'weight' in param_name:
            total_params += param_value.size
            total_layers += 1
    
    return {
        'total_parameters': total_params,
        'total_layers': total_layers,
        'average_params_per_layer': total_params // max(total_layers, 1)
    }


def save_experiment_results(results: Dict[str, Any], filename: str):
    """
    Save experiment results to a file.
    
    Args:
        results: Dictionary containing experiment results
        filename: Name of the file to save results
    """
    import json
    from datetime import datetime
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {filename}")


def load_experiment_results(filename: str) -> Dict[str, Any]:
    """
    Load experiment results from a file.
    
    Args:
        filename: Name of the file to load results from
        
    Returns:
        Dictionary containing experiment results
    """
    import json
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results 