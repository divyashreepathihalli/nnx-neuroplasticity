<div align="center">
<img src="https://raw.githubusercontent.com/google/flax/main/images/flax_logo_250px.png" alt="logo"></img>
</div>

# Neuroplasticity Experiments with NNX

This repository contains experiments demonstrating neuroplasticity concepts in neural networks using Flax NNX. The experiments showcase various forms of neural plasticity including Hebbian learning, synaptic scaling, and dynamic weight adaptation.

## 🧠 What is Neuroplasticity?

Neuroplasticity refers to the brain's ability to form and reorganize synaptic connections, especially in response to learning or experience. In neural networks, we can simulate these biological processes to create more adaptive and learning-capable models.

## 📁 Repository Structure

```
nnx-neuroplasticity/
├── neuroplasticity_simple.py          # Working Flax implementation
├── neuroplasticity_nnx_final.py      # NNX version (requires Python 3.10+)
├── neuroplasticity_nnx_simple.py     # Simple NNX example
├── neuroplasticity_nnx_example.py    # Comprehensive NNX example
├── dense_model_nnx_example.py        # Dense model with NNX
├── neuroplasticity_nnx_guide.md      # Detailed guide
├── requirements.txt                   # Dependencies
└── .gitignore                        # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (3.10+ recommended for NNX examples)
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd nnx-neuroplasticity
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

#### For Python 3.9 (Flax Linen):
```bash
python neuroplasticity_simple.py
```

#### For Python 3.10+ (NNX):
```bash
python neuroplasticity_nnx_final.py
```

## 🧬 Neuroplasticity Concepts Demonstrated

### 1. **Hebbian Learning**
- "Neurons that fire together, wire together"
- Dynamic weight updates based on input-output correlations
- Real-time synaptic strength modifications

### 2. **Synaptic Scaling**
- Homeostatic plasticity to maintain network stability
- Automatic adjustment of connection strengths
- Prevents runaway excitation or inhibition

### 3. **Activity-Dependent Plasticity**
- Weight changes based on neural activity levels
- Adaptive learning rates
- Context-sensitive modifications

### 4. **Structural Plasticity**
- Dynamic connection strength scaling
- Adaptive network architecture
- Real-time topology modifications

## 📊 Example Output

```
🧠 Neuroplasticity in Flax - Simple Version
==================================================
📊 Generating synthetic data...
Data shape: X=(1000, 10), Y=(1000, 1)
🤖 Creating neuroplastic model: 10 -> (32, 16) -> 1
✅ Neuroplastic model initialized successfully
📈 Starting training with plasticity monitoring...
----------------------------------------------------------------------
Step   Loss         Hebbian      Structural   Scaling      Activity    
----------------------------------------------------------------------
 0     74.966576 1940.521362   1.933097   1.933097  34.344379
 2     74.885422 1970.915405   1.881856   1.881856  28.444923
 4     73.942383 2062.816406   1.840620   1.840620  24.806292
 6     73.256889 2120.901123   1.807341   1.807341  21.250053
 8     72.875374 2212.901611   1.779070   1.779070  18.950569
----------------------------------------------------------------------
🎉 Neuroplasticity training completed!
📊 Final loss: 72.812416
🧠 Final plasticity metrics:
   - Hebbian activity: 2239.154541
   - Structural plasticity: 1.766980
   - Synaptic scaling: 1.766980
   - Neural activity: 17.613068
```

## 🔬 Key Features

- **Real-time Plasticity**: Weights update during forward pass
- **Multiple Plasticity Types**: Hebbian, structural, and homeostatic
- **Activity Monitoring**: Track various plasticity metrics
- **Flexible Architecture**: Easy to extend and modify
- **JAX Compatibility**: Leverages JAX's automatic differentiation

## 📚 Learning Resources

- [Neuroplasticity Guide](neuroplasticity_nnx_guide.md) - Detailed explanation of concepts
- [Flax Documentation](https://flax.readthedocs.io/) - Flax framework docs
- [JAX Documentation](https://jax.readthedocs.io/) - JAX framework docs

## 🤝 Contributing

Feel free to contribute by:
- Adding new plasticity mechanisms
- Improving existing implementations
- Adding more comprehensive examples
- Enhancing documentation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by biological neural plasticity research
- Built on Flax and JAX frameworks
- Thanks to the open-source machine learning community
