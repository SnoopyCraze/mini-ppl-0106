"""Example: Gibbs sampling for Normal distribution with unknown mean and variance."""

import numpy as np
import matplotlib.pyplot as plt
from minippl import Model, RandomVariable, Normal, GibbsSampler
from minippl.visualize import plot_diagnostics
from minippl.utils import print_summary

# Set random seed
np.random.seed(42)

# Generate data from a normal distribution
print("Generating data...")
true_mu = 5.0
true_sigma = 2.0
n_obs = 50

data = np.random.normal(true_mu, true_sigma, n_obs)

print(f"Sample mean: {np.mean(data):.4f}")
print(f"Sample std:  {np.std(data):.4f}")
print(f"True mu:     {true_mu:.4f}")
print(f"True sigma:  {true_sigma:.4f}")

# Create model
print("\nBuilding Bayesian model...")
model = Model()

# Priors
mu = RandomVariable("mu", Normal(mu=0, sigma=10))
sigma = RandomVariable("sigma", Normal(mu=1, sigma=5))

model.add_variable(mu)
model.add_variable(sigma)
model.observe(data)

# Likelihood
def likelihood(params, observed_data):
    """Normal likelihood."""
    mu_val = params['mu']
    sigma_val = params['sigma']

    if sigma_val <= 0:
        return -np.inf

    n = len(observed_data)
    log_like = -0.5 * n * np.log(2 * np.pi * sigma_val**2)
    log_like -= 0.5 * np.sum((observed_data - mu_val)**2) / (sigma_val**2)

    return log_like

model.set_likelihood(likelihood)

# Custom conditional samplers for Gibbs
# In a conjugate setup, these would sample from exact conditionals
# Here we use Metropolis steps as a demonstration

def sample_mu_conditional(model, var_name):
    """Sample mu conditional on sigma."""
    # This would be exact in a conjugate setup (Normal-Normal)
    # Here we use a Metropolis step
    current_mu = model.variables['mu'].value
    current_sigma = model.variables['sigma'].value

    # Proposal
    proposal_std = 0.1
    proposed_mu = current_mu + np.random.normal(0, proposal_std)

    # Acceptance ratio
    current_vals = {'mu': current_mu, 'sigma': current_sigma}
    proposed_vals = {'mu': proposed_mu, 'sigma': current_sigma}

    log_ratio = model.log_posterior(proposed_vals) - model.log_posterior(current_vals)

    if np.log(np.random.rand()) < log_ratio:
        return proposed_mu
    else:
        return current_mu


def sample_sigma_conditional(model, var_name):
    """Sample sigma conditional on mu."""
    current_mu = model.variables['mu'].value
    current_sigma = model.variables['sigma'].value

    # Proposal
    proposal_std = 0.1
    proposed_sigma = current_sigma + np.random.normal(0, proposal_std)

    if proposed_sigma <= 0:
        return current_sigma

    # Acceptance ratio
    current_vals = {'mu': current_mu, 'sigma': current_sigma}
    proposed_vals = {'mu': current_mu, 'sigma': proposed_sigma}

    log_ratio = model.log_posterior(proposed_vals) - model.log_posterior(current_vals)

    if np.log(np.random.rand()) < log_ratio:
        return proposed_sigma
    else:
        return current_sigma


# Run Gibbs sampler
print("\nRunning Gibbs sampling...")
conditional_samplers = {
    'mu': sample_mu_conditional,
    'sigma': sample_sigma_conditional
}

gibbs = GibbsSampler(model, conditional_samplers=conditional_samplers)
samples = gibbs.run(n_samples=5000, burn_in=1000, random_seed=42)

# Results
print_summary(samples)

print(f"\nTrue values:")
print(f"  mu:    {true_mu:.4f}")
print(f"  sigma: {true_sigma:.4f}")

# Visualize
print("\nCreating diagnostic plots...")
fig = plot_diagnostics(samples)
plt.savefig("examples/gibbs_normal_diagnostics.png", dpi=150, bbox_inches='tight')
print("Saved: examples/gibbs_normal_diagnostics.png")

# Joint posterior
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(samples['mu'], samples['sigma'], alpha=0.3, s=10)
ax.scatter(true_mu, true_sigma, c='red', s=200, marker='*',
           edgecolors='black', linewidth=2, label='True values', zorder=5)
ax.scatter(np.mean(samples['mu']), np.mean(samples['sigma']), c='orange',
           s=200, marker='X', edgecolors='black', linewidth=2,
           label='Posterior mean', zorder=5)
ax.set_xlabel('mu')
ax.set_ylabel('sigma')
ax.set_title('Joint Posterior Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("examples/gibbs_normal_joint.png", dpi=150, bbox_inches='tight')
print("Saved: examples/gibbs_normal_joint.png")

plt.show()

print("\n✓ Gibbs sampling example completed!")
