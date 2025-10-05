"""Example: Coin flip inference - estimating bias parameter."""

import numpy as np
import matplotlib.pyplot as plt
from minippl import Model, RandomVariable, Beta, Bernoulli, MetropolisHastings
from minippl.visualize import plot_posterior, plot_trace
from minippl.utils import print_summary

# Set random seed
np.random.seed(42)

# Generate synthetic coin flip data
print("Generating coin flip data...")
true_bias = 0.7  # Unfair coin
n_flips = 100
flips = np.random.binomial(1, true_bias, n_flips)

print(f"Observed: {np.sum(flips)} heads out of {n_flips} flips")
print(f"True bias: {true_bias}")

# Create model
print("\nBuilding Bayesian model...")
model = Model()

# Prior: Beta(2, 2) - slightly peaked at 0.5
bias = RandomVariable("bias", Beta(alpha=2, beta=2))
model.add_variable(bias)

# Set observed data
model.observe(flips)

# Likelihood function
def likelihood(params, data):
    """Binomial likelihood for coin flips."""
    p = params['bias']

    # Ensure p is in valid range
    if not (0 < p < 1):
        return -np.inf

    # Binomial likelihood (sum of Bernoulli)
    n_heads = np.sum(data)
    n_tails = len(data) - n_heads

    log_like = n_heads * np.log(p) + n_tails * np.log(1 - p)

    return log_like

model.set_likelihood(likelihood)

# Run inference
print("\nRunning MCMC inference...")
sampler = MetropolisHastings(model, proposal_std=0.05)
samples = sampler.run(n_samples=10000, burn_in=2000, random_seed=42)

print(f"Acceptance rate: {sampler.acceptance_rate:.3f}")

# Summary
print_summary(samples)

# Analytical solution (conjugate Beta-Binomial)
# Posterior: Beta(alpha + heads, beta + tails)
n_heads = np.sum(flips)
posterior_alpha = 2 + n_heads
posterior_beta = 2 + (n_flips - n_heads)
analytical_mean = posterior_alpha / (posterior_alpha + posterior_beta)

print(f"\nAnalytical posterior mean (conjugate): {analytical_mean:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Posterior
plot_posterior(samples, 'bias', bins=50, ax=axes[0])

# Add analytical posterior
from scipy.stats import beta as beta_dist
x = np.linspace(0, 1, 1000)
analytical_posterior = beta_dist.pdf(x, posterior_alpha, posterior_beta)
axes[0].plot(x, analytical_posterior, 'g-', linewidth=2,
             label=f'Analytical (Beta({posterior_alpha}, {posterior_beta}))')
axes[0].axvline(true_bias, color='orange', linestyle='--',
                linewidth=2, label=f'True bias: {true_bias}')
axes[0].legend()

# Trace
plot_trace(samples, 'bias', ax=axes[1])

plt.tight_layout()
plt.savefig("examples/coin_flip_results.png", dpi=150, bbox_inches='tight')
print("\nSaved: examples/coin_flip_results.png")

plt.show()

print("\n✓ Coin flip example completed!")
