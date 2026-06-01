# Sinkhorn-CPD

Official implementation of **Sinkhorn-CPD: Robust Point Cloud Registration via Unbalanced Entropic Optimal Transport**.

## Overview

Sinkhorn-CPD is a robust point cloud registration method that formulates rigid alignment as an unbalanced entropic optimal transport problem. By introducing dual KL-divergence penalties on both source and target marginals, it enables symmetric outlier rejection while preserving the automatic variance annealing of CPD.

### Key Features

- **Dual-KL unbalanced formulation** — simultaneously rejects outliers on both source and target sides
- **Adaptive variance annealing** — kernel scale tracks alignment residuals, preventing premature convergence
- **Single hyperparameter** — works with $\tau=1$ across all experimental settings without retuning
- **GPU-accelerated** — implemented in PyTorch (~400 lines)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy, SciPy

## Usage

```python
from sinkhorn_cpd import SinkhornCPD

model = SinkhornCPD()
T = model.register(source, target)
```

## Citation

Paper under review. Citation info will be added upon publication.

## License

MIT
