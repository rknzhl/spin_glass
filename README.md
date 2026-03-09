# Spin Glasses & Neural Networks: Multilayer Sherrington–Kirkpatrick Model

A computational study applying statistical physics to neural network loss surfaces through spin glass theory.

## Overview

This project investigates energy landscapes and phase transitions in neural networks using the multilayer Sherrington–Kirkpatrick (MSK) model. We employ:

- **Replica method** for analytical characterization of disordered systems
- **Thouless–Anderson–Palmer (TAP) equations** to analyze metastable states and phase transitions
- **Monte Carlo simulated annealing** for spin system optimization
- **Momentum-SGD with Straight-Through Estimator (STE)** for discrete weight optimization

## Key Results

- Applied replica symmetry breaking to characterize neural network loss surfaces
- Solved TAP equations numerically for metastable state analysis
- Implemented efficient discrete optimization using STE approximation
- Validated theoretical predictions through Monte Carlo simulations on MNIST

**Award**: 67th All-Russian MIPT Scientific Conference

## Project Structure

```
spin_glass/
├── spin_nn/                    # Main implementation
│   ├── model.py               # MSK model
│   ├── equations.py           # TAP equations & statistical physics
│   ├── training.py            # Training with STE
│   ├── annealing.py           # Simulated annealing
│   ├── temp_calc.py           # Critical temperature
│   ├── visualization.py       # Visualization tools
│   └── utils.py               # Utilities
├── notebooks/                 # Jupyter notebooks with experiments
├── docs/                      # Documentation
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/rknzhl/spin_glass.git
cd spin_glass
pip install -r requirements.txt
```

## Usage

### Model & Training
See `notebooks/training_MNIST.ipynb` for full training pipeline.

### Analysis
- `notebooks/tap_temp.ipynb` — TAP equations and critical temperature analysis
- `notebooks/results_MNIST.ipynb` — Results and visualization
- `notebooks/annealing.ipynb` — Simulated annealing demonstrations

## Documentation

**Main document**: `docs/PRESENTATION.pdf` — Complete technical presentation with theoretical framework, methodology, and experimental results.

For detailed information about documentation files, see `docs/README.md`
