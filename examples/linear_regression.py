"""Example: Bayesian Linear Regression with mini PPL."""

import numpy as np
import matplotlib.pyplot as plt
from minippl import Model, RandomVariable, Normal, MetropolisHastings
from minippl.visualize import plot_diagnostics
from minippl.utils import print_summary

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
print("Generating synthetic data...")
n_points = 100
true_slope = 2.5
true_intercept = 1.0
true_sigma = 0.5

X = np.linspace(0, 10, n_points)
y_true = true_slope * X + true_intercept
y = y_true + np.random.normal(0, true_sigma, n_points)

# Create model
print("\nBuilding Bayesian linear regression model...")
model = Model()

# Define priors
slope = RandomVariable("slope", Normal(mu=0, sigma=10))
intercept = RandomVariable("intercept", Normal(mu=0, sigma=10))
sigma = RandomVariable("sigma", Normal(mu=1, sigma=5))

# Add variables to model
model.add_variable(slope)
model.add_variable(intercept)
model.add_variable(sigma)

# Set observed data
model.observe(y)

# Define likelihood function
def likelihood(params, data):
    """Compute log likelihood for linear regression."""
    slope_val = params['slope']
    intercept_val = params['intercept']
    sigma_val = params['sigma']

    if sigma_val <= 0:
        return -np.inf

    y_pred = slope_val * X + intercept_val
    residuals = data - y_pred

    # Normal likelihood
    log_like = -0.5 * n_points * np.log(2 * np.pi * sigma_val**2)
    log_like -= 0.5 * np.sum(residuals**2) / (sigma_val**2)

    return log_like

model.set_likelihood(likelihood)

# Run MCMC inference
print("\nRunning Metropolis-Hastings MCMC...")
sampler = MetropolisHastings(model, proposal_std=0.1)
samples = sampler.run(n_samples=5000, burn_in=1000, random_seed=42)

print(f"Acceptance rate: {sampler.acceptance_rate:.3f}")

# Print summary
print_summary(samples)

# Print true values for comparison
print("\nTrue values:")
print(f"  slope:     {true_slope:.4f}")
print(f"  intercept: {true_intercept:.4f}")
print(f"  sigma:     {true_sigma:.4f}")

# Visualize results
print("\nCreating diagnostic plots...")
fig = plot_diagnostics(samples)
plt.savefig("examples/linear_regression_diagnostics.png", dpi=150, bbox_inches='tight')
print("Saved: examples/linear_regression_diagnostics.png")

# Plot data with posterior predictive
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, alpha=0.5, label='Observed data')
ax.plot(X, y_true, 'k--', linewidth=2, label='True relationship')

# Plot posterior mean prediction
mean_slope = np.mean(samples['slope'])
mean_intercept = np.mean(samples['intercept'])
y_pred_mean = mean_slope * X + mean_intercept
ax.plot(X, y_pred_mean, 'r-', linewidth=2, label='Posterior mean fit')

# Plot uncertainty
n_posterior_samples = 100
for i in np.random.choice(len(samples['slope']), n_posterior_samples):
    y_pred = samples['slope'][i] * X + samples['intercept'][i]
    ax.plot(X, y_pred, 'r-', alpha=0.02)

ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Bayesian Linear Regression')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("examples/linear_regression_fit.png", dpi=150, bbox_inches='tight')
print("Saved: examples/linear_regression_fit.png")

plt.show()

print("\n✓ Linear regression example completed!")
