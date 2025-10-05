"""Comparison of mini PPL with PyMC for linear regression."""

import numpy as np
import matplotlib.pyplot as plt
import time

# Mini PPL implementation
print("=" * 70)
print("COMPARISON: Mini PPL vs PyMC3")
print("=" * 70)

# Generate data
np.random.seed(42)
n_points = 100
true_slope = 2.5
true_intercept = 1.0
true_sigma = 0.5

X = np.linspace(0, 10, n_points)
y_true = true_slope * X + true_intercept
y = y_true + np.random.normal(0, true_sigma, n_points)

# ==================== Mini PPL ====================
print("\n[1] Running Mini PPL...")
from minippl import Model, RandomVariable, Normal, MetropolisHastings

model_mini = Model()
slope = RandomVariable("slope", Normal(mu=0, sigma=10))
intercept = RandomVariable("intercept", Normal(mu=0, sigma=10))
sigma = RandomVariable("sigma", Normal(mu=1, sigma=5))

model_mini.add_variable(slope)
model_mini.add_variable(intercept)
model_mini.add_variable(sigma)
model_mini.observe(y)

def likelihood(params, data):
    slope_val = params['slope']
    intercept_val = params['intercept']
    sigma_val = params['sigma']

    if sigma_val <= 0:
        return -np.inf

    y_pred = slope_val * X + intercept_val
    residuals = data - y_pred
    log_like = -0.5 * n_points * np.log(2 * np.pi * sigma_val**2)
    log_like -= 0.5 * np.sum(residuals**2) / (sigma_val**2)
    return log_like

model_mini.set_likelihood(likelihood)

start_time = time.time()
sampler_mini = MetropolisHastings(model_mini, proposal_std=0.1)
samples_mini = sampler_mini.run(n_samples=5000, burn_in=1000, random_seed=42)
mini_time = time.time() - start_time

print(f"\n  Time: {mini_time:.2f} seconds")
print(f"  Acceptance rate: {sampler_mini.acceptance_rate:.3f}")
print(f"\n  Results:")
print(f"    slope:     {np.mean(samples_mini['slope']):.4f} ± {np.std(samples_mini['slope']):.4f}")
print(f"    intercept: {np.mean(samples_mini['intercept']):.4f} ± {np.std(samples_mini['intercept']):.4f}")
print(f"    sigma:     {np.mean(samples_mini['sigma']):.4f} ± {np.std(samples_mini['sigma']):.4f}")

# ==================== PyMC3 ====================
print("\n[2] Running PyMC3...")
try:
    import pymc as pm

    with pm.Model() as model_pymc:
        # Priors
        slope_pm = pm.Normal("slope", mu=0, sigma=10)
        intercept_pm = pm.Normal("intercept", mu=0, sigma=10)
        sigma_pm = pm.HalfNormal("sigma", sigma=5)

        # Linear model
        y_pred = slope_pm * X + intercept_pm

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=y_pred, sigma=sigma_pm, observed=y)

        # Inference
        start_time = time.time()
        trace = pm.sample(5000, tune=1000, random_seed=42, return_inferencedata=False,
                          progressbar=True, cores=1)
        pymc_time = time.time() - start_time

    print(f"\n  Time: {pymc_time:.2f} seconds")
    print(f"\n  Results:")
    print(f"    slope:     {np.mean(trace['slope']):.4f} ± {np.std(trace['slope']):.4f}")
    print(f"    intercept: {np.mean(trace['intercept']):.4f} ± {np.std(trace['intercept']):.4f}")
    print(f"    sigma:     {np.mean(trace['sigma']):.4f} ± {np.std(trace['sigma']):.4f}")

    # Comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    vars_to_plot = ['slope', 'intercept', 'sigma']
    for i, var in enumerate(vars_to_plot):
        # Mini PPL
        axes[i, 0].hist(samples_mini[var], bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[i, 0].set_title(f'Mini PPL: {var}')
        axes[i, 0].set_ylabel('Density')
        axes[i, 0].axvline(np.mean(samples_mini[var]), color='red', linestyle='--',
                           label=f'Mean: {np.mean(samples_mini[var]):.3f}')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # PyMC
        pymc_samples = trace[var] if var != 'sigma' else trace['sigma']
        axes[i, 1].hist(pymc_samples, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[i, 1].set_title(f'PyMC3: {var}')
        axes[i, 1].axvline(np.mean(pymc_samples), color='red', linestyle='--',
                           label=f'Mean: {np.mean(pymc_samples):.3f}')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/comparison_pymc.png", dpi=150, bbox_inches='tight')
    print("\nSaved: examples/comparison_pymc.png")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\nSpeed:")
    print(f"  Mini PPL: {mini_time:.2f}s")
    print(f"  PyMC3:    {pymc_time:.2f}s")
    print(f"  Ratio:    {pymc_time/mini_time:.2f}x")

    print(f"\nParameter estimates (Mini PPL vs PyMC3):")
    print(f"  slope:     {np.mean(samples_mini['slope']):.4f} vs {np.mean(trace['slope']):.4f}")
    print(f"  intercept: {np.mean(samples_mini['intercept']):.4f} vs {np.mean(trace['intercept']):.4f}")
    print(f"  sigma:     {np.mean(samples_mini['sigma']):.4f} vs {np.mean(trace['sigma']):.4f}")

    print(f"\nTrue values:")
    print(f"  slope:     {true_slope:.4f}")
    print(f"  intercept: {true_intercept:.4f}")
    print(f"  sigma:     {true_sigma:.4f}")

    plt.show()

except ImportError:
    print("\n  PyMC not installed. Install with: pip install pymc")
    print("  Skipping PyMC comparison.")

print("\n✓ Comparison completed!")
