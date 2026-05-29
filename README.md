# Cognitive Biases Generate Endogenous Volatility in Welfare

This repository contains the simulation framework, analysis pipelines, and figure generation notebooks for the paper **"Cognitive biases generate endogenous volatility in welfare"** (Dupont & Roy, 2026).

## Overview

The framework compares Expected Utility Theory (EUT), Prospect Theory (PT), and Cumulative Prospect Theory (CPT) within a coupled feedback system of human and financial capital.

## Prerequisites

This project uses `uv` for dependency management and execution. Ensure you have the standard scientific stack installed (`numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`).

## 1. Initialization (Run First)

Before running experiments, generate the shared initial states and parameter samples (Latin Hypercube Sampling).

```bash
# 1. Generate 5,000 unique agent initial states
uv run generate_initial_states.py --n-agents 5000 --grid-size 200 --seed 42

# 2. Generate parameter samples for the Global Sensitivity Analysis (LHS)
uv run generate_parameter_sample.py --n-samples 1024 --model eut --seed 42
uv run generate_parameter_sample.py --n-samples 1024 --model pt --seed 42
uv run generate_parameter_sample.py --n-samples 1024 --model cpt --seed 42
```

## 2. Core Experiments (LHS)

These simulations form the backbone of the global analysis (Figures 3, 4, 6, 7, 9, 11-14).

```bash
# Run the main simulations (this may take time)
uv run experiment.py --model eut --n-steps 5000 --max-workers 6
uv run experiment.py --model pt  --n-steps 5000 --max-workers 6
uv run experiment.py --model cpt --n-steps 5000 --max-workers 6
```

### Post-Processing: Frequency Analysis
Once the core experiments are done, run the spectral analysis to detect limit cycles.

```bash
uv run compute_trajectory_amplitudes.py --model eut --max-workers 6
uv run compute_trajectory_amplitudes.py --model pt --max-workers 6
uv run compute_trajectory_amplitudes.py --model cpt --max-workers 6
```

## 3. Specialized Experiments

These scripts run targeted parameter sweeps for specific figures.

**Bifurcation Analysis (Figure 8):**
*Requires `pt` results from Section 2 to be available.*
```bash
uv run lambda_bifurcation.py --model lambda_bifurcation
```

**Joint Wealth/Probability Sweep (Figure 10):**
```bash
uv run gamma_alpha.py --model gamma_alpha
```

## 4. Reproducing Figures (Notebook Guide)

The figures in the manuscript are generated using the Jupyter notebooks located in `notebooks/`. Execute them in the order below to ensure data dependencies are met.

| Notebook | Dependencies | Figures | Description |
| :--- | :--- | :--- | :--- |
| `health_cost.ipynb` | None | **2** | Plots the cost/benefit functions and "physics" of the model. |
| `sen_welfare.ipynb` | `experiment.py` | **3, 4, 11, 12, 13** | Computes Sen Welfare metrics, Kernel Density Estimates (KDE), and marginal parameter sensitivities for all models. |
| `sensitivity_analysis.ipynb` | `sen_welfare.ipynb` | **14** | Performs PAWN Global Sensitivity Analysis on the welfare metrics derived in the previous notebook. |
| `frequency_analysis.ipynb` | `compute_trajectory_amplitudes.py` | **5, 6, 9** | Visualizes agent trajectories, amplitude distributions, and stability phase diagrams (structural stability). |
| `attractors.ipynb` | `compute_trajectory_amplitudes.py` | **7** | Maps the spatial distribution of fixed points vs. limit cycles to visualize the "Fracture" in policy space. |
| `lambda_bifurcation.ipynb` | `lambda_bifurcation.py` | **8** | Visualizes the bifurcation tipping point (transition to limit cycles) as Loss Aversion ($\lambda$) increases. |
| `gamma_alpha.ipynb` | `gamma_alpha.py` | **10** | Heatmaps showing the tipping point caused by Probability Weighting ($\gamma$) and Wealth Preference ($\alpha$). |

---

### Data Notes
*   **Raw Data:** Stored as `.pickle` files in directories named after the model (e.g., `cpt/`).
*   **Processed Data:** The notebooks may generate intermediate data structures; it is recommended to run `sen_welfare.ipynb` before `sensitivity_analysis.ipynb`.