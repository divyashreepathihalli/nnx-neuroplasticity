#!/usr/bin/env python3
"""
Neuroplasticity in Flax - Simple Version

This script demonstrates neuroplasticity concepts in Flax:
1. Dynamic weight updates during forward pass
2. Adaptive connection strengths
3. Hebbian learning rules
4. Synaptic scaling
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import freeze, unfreeze

# Import utility functions
from utils import (
    generate_data, dataset, calculate_plasticity_metrics, create_optimizer,
    print_training_header, print_training_step, print_final_results,
    setup_random_seed, create_plasticity_config, validate_plasticity_config
)

# Set up random seed
setup_random_seed(42)

class PlasticLinear(nn.Module):
    """
    A linear layer with neuroplasticity features:
    - Dynamic weight updates based on activity
    - Hebbian learning
    - Synaptic scaling
    """
    
    in_features: int
    out_features: int
    plasticity_rate: float = 0.01
    
    def setup(self):
        # Standard weights and bias
        self.weight = self.param('weight', jax.random.uniform, (self.in_features, self.out_features))
        self.bias = self.param('bias', lambda rng: jnp.zeros((self.out_features,)))
        
        # Plasticity parameters
        self.activity_history = self.variable('plasticity', 'activity_history', 
                                           lambda: jnp.zeros((self.in_features, self.out_features)))
        self.connection_strength = self.variable('plasticity', 'connection_strength', 
                                               lambda: jnp.ones((self.in_features, self.out_features)))
        self.hebbian_weights = self.variable('plasticity', 'hebbian_weights', 
                                           lambda: jnp.zeros((self.in_features, self.out_features)))
    
    def hebbian_update(self, input_activity, output_activity):
        """Hebbian learning: neurons that fire together, wire together."""
        # Hebbian rule: Δw = η * input_activity * output_activity
        hebbian_update = self.plasticity_rate * input_activity[:, None] * output_activity[None, :]
        self.hebbian_weights.value += hebbian_update
    
    def synaptic_scaling(self, target_activity=1.0):
        """Synaptic scaling to maintain homeostasis."""
        current_activity = jnp.mean(self.activity_history.value)
        if current_activity > 0:
            scaling_factor = target_activity / current_activity
            self.connection_strength.value *= scaling_factor
            # Clamp scaling to reasonable bounds
            self.connection_strength.value = jnp.clip(self.connection_strength.value, 0.1, 10.0)
    
    def __call__(self, x, training=True):
        # Store input for plasticity updates
        input_activity = jnp.mean(jnp.abs(x), axis=0)
        
        # Apply connection strength scaling
        plastic_weight = self.weight * self.connection_strength.value
        
        # Forward pass
        y = x @ plastic_weight + self.bias
        
        if training:
            # Update activity history
            output_activity = jnp.mean(jnp.abs(y), axis=0)
            self.activity_history.value = 0.9 * self.activity_history.value + 0.1 * (
                input_activity[:, None] * output_activity[None, :]
            )
            
            # Apply neuroplasticity updates
            self.hebbian_update(input_activity, output_activity)
            
            # Continuous synaptic scaling (simplified)
            scaling_factor = 1.0 + 0.001 * (1.0 - jnp.mean(self.activity_history.value))
            self.connection_strength.value *= scaling_factor
            self.connection_strength.value = jnp.clip(self.connection_strength.value, 0.1, 10.0)
        
        return y

class NeuroplasticModel(nn.Module):
    """
    A neural network with neuroplasticity features.
    """
    
    input_dim: int
    hidden_dims: tuple
    output_dim: int
    
    def setup(self):
        # Create plastic layers
        prev_dim = self.input_dim
        
        # Create layers as attributes
        for i, hidden_dim in enumerate(self.hidden_dims):
            setattr(self, f'layer_{i}', PlasticLinear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer (less plastic for stability)
        self.output_layer = nn.Dense(self.output_dim)
        
        # Plasticity metrics
        self.plasticity_metrics = self.variable('metrics', 'plasticity_metrics', 
                                              lambda: jnp.zeros(4))
    
    def __call__(self, x, training=True):
        # Forward pass through plastic layers
        for i in range(len(self.hidden_dims)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x, training=training)
            x = jax.nn.relu(x)  # ReLU activation
        
        # Output layer
        x = self.output_layer(x)
        
        if training:
            # Update plasticity metrics
            hebbian_total = sum(getattr(self, f'layer_{i}').hebbian_weights.value.sum() 
                              for i in range(len(self.hidden_dims)))
            structural_total = sum(getattr(self, f'layer_{i}').connection_strength.value.mean() 
                                for i in range(len(self.hidden_dims)))
            scaling_total = sum(getattr(self, f'layer_{i}').connection_strength.value.mean() 
                             for i in range(len(self.hidden_dims)))
            activity_total = sum(getattr(self, f'layer_{i}').activity_history.value.mean() 
                              for i in range(len(self.hidden_dims)))
            
            self.plasticity_metrics.value = jnp.array([hebbian_total, structural_total, scaling_total, activity_total])
        
        return x

# Training functions
@jax.jit
def train_step(params, variables, batch, optimizer_state):
    """Single training step with plasticity monitoring."""
    x, y = batch
    
    def loss_fn(params):
        variables_new = variables.copy()
        y_pred = model.apply({'params': params, **variables_new}, x, training=True, mutable=['plasticity', 'metrics'])
        loss = jnp.mean((y - y_pred[0]) ** 2)
        return loss, y_pred[1]
    
    grads, variables = jax.grad(loss_fn, has_aux=True)(params)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    
    return params, variables, optimizer_state

@jax.jit
def eval_step(params, variables, batch):
    """Evaluation step with plasticity metrics."""
    x, y = batch
    
    y_pred = model.apply({'params': params, **variables}, x, training=False)
    loss = jnp.mean((y - y_pred) ** 2)
    
    # Get plasticity metrics
    plasticity_metrics = variables['metrics']['plasticity_metrics']
    
    return {
        'loss': loss,
        'hebbian_activity': plasticity_metrics[0],
        'structural_plasticity': plasticity_metrics[1],
        'synaptic_scaling': plasticity_metrics[2],
        'neural_activity': plasticity_metrics[3]
    }

def main():
    """Main training function with neuroplasticity monitoring."""
    print("🧠 Neuroplasticity in Flax - Simple Version")
    print("=" * 50)
    
    # Create plasticity configuration
    config = create_plasticity_config(plasticity_rate=0.01, target_activity=1.0, scaling_factor=0.001)
    if not validate_plasticity_config(config):
        print("❌ Invalid plasticity configuration!")
        return
    
    # Generate data
    print("📊 Generating synthetic data...")
    X, Y = generate_data(n_samples=1000, input_dim=10, output_dim=1)
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Model parameters
    input_dim = X.shape[1]
    hidden_dims = (32, 16)  # Smaller network for plasticity observation
    output_dim = Y.shape[1]
    
    print(f"🤖 Creating neuroplastic model: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    # Create model
    global model
    model = NeuroplasticModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim
    )
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    variables = model.init(rng, jnp.ones((1, input_dim)), training=True)
    params = variables['params']
    variables = {k: v for k, v in variables.items() if k != 'params'}
    
    # Create optimizer
    global optimizer
    optimizer = create_optimizer(learning_rate=0.001, optimizer_type='adam')
    optimizer_state = optimizer.init(params)
    
    print("✅ Neuroplastic model initialized successfully")
    print("📈 Starting training with plasticity monitoring...")
    print_training_header()
    
    # Training loop
    data_gen = dataset(X, Y, batch_size=32)
    
    for step in range(10):
        batch = next(data_gen)
        params, variables, optimizer_state = train_step(params, variables, batch, optimizer_state)
        
        # Evaluation with plasticity metrics
        if step % 2 == 0:
            metrics = eval_step(params, variables, (X, Y))
            print_training_step(step, metrics['loss'], metrics)
    
    # Final evaluation
    final_metrics = eval_step(params, variables, (X, Y))
    print_final_results(final_metrics['loss'], final_metrics)
    
    # Test prediction
    test_input = np.random.normal(0, 1, (5, input_dim))
    test_output = model.apply({'params': params, **variables}, test_input, training=False)
    print(f"🧪 Test prediction shape: {test_output.shape}")
    print(f"🧪 Sample predictions: {test_output.flatten()[:3]}")
    
    print("\n✅ Neuroplasticity in Flax demonstration completed!")

if __name__ == "__main__":
    main() 