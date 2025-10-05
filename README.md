# mini-ppl-0106 - A Minimal Probabilistic Programming Language

A from-scratch implementation of a Probabilistic Programming Language (PPL) in Python with MCMC inference capabilities. This educational framework demonstrates the internals of probabilistic programming and Bayesian inference.

- **Core Probabilistic Modeling Primitives**
  - Probability distributions: Normal, Uniform, Bernoulli, Exponential, Beta
  - Random variables with prior distributions
  - Flexible model definition with custom likelihoods

- **MCMC Inference Algorithms** (built from scratch)
  - Metropolis-Hastings sampler
  - Gibbs sampler with custom conditional distributions
  - Adaptive proposal distributions

- **Visualization & Diagnostics**
  - Posterior distribution plots
  - Trace plots for convergence diagnostics
  - Autocorrelation analysis
  - Effective sample size computation
  - R-hat convergence diagnostic

- **Bayesian Inference Utilities**
  - Posterior predictive checks
  - Model comparison (DIC, WAIC)
  - Credible intervals
  - Summary statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/SnoopyCraze/mini-ppl-0106
cd mini-ppl-0106

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Example: Bayesian Linear Regression

```python
import numpy as np
from minippl import Model, RandomVariable, Normal, MetropolisHastings
from minippl.visualize import plot_diagnostics
from minippl.utils import print_summary

# Generate data
X = np.linspace(0, 10, 100)
y = 2.5 * X + 1.0 + np.random.normal(0, 0.5, 100)

# Define model
model = Model()

# Priors
slope = RandomVariable("slope", Normal(mu=0, sigma=10))
intercept = RandomVariable("intercept", Normal(mu=0, sigma=10))
sigma = RandomVariable("sigma", Normal(mu=1, sigma=5))

model.add_variable(slope)
model.add_variable(intercept)
model.add_variable(sigma)
model.observe(y)

# Likelihood function
def likelihood(params, data):
    slope_val = params['slope']
    intercept_val = params['intercept']
    sigma_val = params['sigma']

    if sigma_val <= 0:
        return -np.inf

    y_pred = slope_val * X + intercept_val
    residuals = data - y_pred

    n = len(data)
    log_like = -0.5 * n * np.log(2 * np.pi * sigma_val**2)
    log_like -= 0.5 * np.sum(residuals**2) / (sigma_val**2)

    return log_like

model.set_likelihood(likelihood)

# Run inference
sampler = MetropolisHastings(model, proposal_std=0.1)
samples = sampler.run(n_samples=5000, burn_in=1000)

# Analyze results
print_summary(samples)
plot_diagnostics(samples)
```

### Example: Coin Flip Inference

```python
from minippl import Model, RandomVariable, Beta, MetropolisHastings

# Data: 70 heads out of 100 flips
flips = np.array([1]*70 + [0]*30)

# Model
model = Model()
bias = RandomVariable("bias", Beta(alpha=2, beta=2))
model.add_variable(bias)
model.observe(flips)

# Likelihood
def likelihood(params, data):
    p = params['bias']
    if not (0 < p < 1):
        return -np.inf
    n_heads = np.sum(data)
    n_tails = len(data) - n_heads
    return n_heads * np.log(p) + n_tails * np.log(1 - p)

model.set_likelihood(likelihood)

# Inference
sampler = MetropolisHastings(model, proposal_std=0.05)
samples = sampler.run(n_samples=10000, burn_in=2000)
```

## Examples

The `examples/` directory contains several complete demonstrations:

- **`linear_regression.py`** - Bayesian linear regression with posterior analysis
- **`coin_flip.py`** - Parameter estimation for binomial data
- **`gibbs_normal.py`** - Gibbs sampling for Normal distribution inference
- **`compare_pymc.py`** - Performance and accuracy comparison with PyMC3

Run an example:
```bash
python examples/linear_regression.py
```

## API Reference

### Distributions

All distributions inherit from the `Distribution` base class and implement:
- `logpdf(x)` - Log probability density/mass function
- `sample(size)` - Draw random samples

Available distributions:
- `Normal(mu, sigma)` - Gaussian distribution
- `Uniform(low, high)` - Uniform distribution
- `Bernoulli(p)` - Bernoulli distribution
- `Exponential(lam)` - Exponential distribution
- `Beta(alpha, beta)` - Beta distribution

### Model Definition

```python
model = Model()
model.add_variable(random_variable)
model.set_likelihood(likelihood_fn)
model.observe(data)
```

### Inference

**Metropolis-Hastings:**
```python
sampler = MetropolisHastings(model, proposal_std=0.1)
samples = sampler.run(n_samples=5000, burn_in=1000, thin=1)
```

**Gibbs Sampling:**
```python
sampler = GibbsSampler(model, conditional_samplers=samplers_dict)
samples = sampler.run(n_samples=5000, burn_in=1000)
```

### Visualization

```python
from minippl.visualize import (
    plot_posterior,
    plot_trace,
    plot_autocorrelation,
    plot_diagnostics
)

plot_diagnostics(samples)  # All-in-one diagnostic plots
plot_posterior(samples, 'parameter_name', bins=50)
plot_trace(samples, 'parameter_name')
plot_autocorrelation(samples, 'parameter_name', max_lag=50)
```

### Utilities

```python
from minippl.utils import print_summary, compute_dic, compute_waic

print_summary(samples)  # Print posterior statistics
dic = compute_dic(model, samples)  # Model comparison
waic = compute_waic(model, samples)
```

## Honest Performance Notes

- Mini PPL uses pure Python/NumPy (no compiled backends)
- Typical performance: ~1000-5000 samples/second 
- For production workloads, PyMC3/Pyro are 10-100x faster (sorry, I tried lol)

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional probability distributions
- More efficient MCMC algorithms (HMC, NUTS)
- Better proposal adaptation
- Parallel chain support
- More examples 

## Educational Reasoning

This implementation demonstrates:
1. **MCMC Fundamentals** - How Metropolis-Hastings and Gibbs sampling work
2. **Bayesian Inference** - Prior × Likelihood = Posterior
3. **Convergence Diagnostics** - Trace plots, autocorrelation, R-hat
4. **Model Comparison** - DIC and WAIC computation
5. **PPL Design** - Architecture of probabilistic programming systems

## License

MIT  

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis*
- Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*
- PyMC3 Documentation: https://docs.pymc.io
- Pyro Documentation: https://pyro.ai

## Citation

If you use this code for educational purposes, please cite:

```bibtex
@software{mini-ppl-0106-2025,
  title={mini-ppl-0106: A Minimal Probabilistic Programming Language},
  author={Zeran Johannsen},
  year={2025},
  url={https://github.com/SnoopyCraze/mini-ppl-0106}
}
```
---

