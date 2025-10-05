"""Unit tests for MCMC inference."""

import numpy as np
import pytest
from minippl import Model, RandomVariable, Normal, MetropolisHastings, GibbsSampler


def create_simple_model():
    """Create a simple test model."""
    model = Model()
    rv = RandomVariable("mu", Normal(0, 10))
    model.add_variable(rv)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model.observe(data)

    def likelihood(params, data):
        mu = params['mu']
        sigma = 1.0
        return np.sum(Normal(mu, sigma).logpdf(data))

    model.set_likelihood(likelihood)
    return model


def test_metropolis_hastings_basic():
    """Test basic Metropolis-Hastings functionality."""
    model = create_simple_model()
    sampler = MetropolisHastings(model, proposal_std=0.5)

    samples = sampler.run(n_samples=1000, burn_in=100, random_seed=42)

    # Check output format
    assert isinstance(samples, dict)
    assert "mu" in samples
    assert len(samples["mu"]) == 1000

    # Check acceptance rate is reasonable
    assert 0.1 < sampler.acceptance_rate < 0.9


def test_metropolis_hastings_convergence():
    """Test MH converges to correct posterior."""
    model = create_simple_model()
    sampler = MetropolisHastings(model, proposal_std=0.5)

    samples = sampler.run(n_samples=5000, burn_in=1000, random_seed=42)

    # Posterior mean should be close to sample mean (3.0)
    posterior_mean = np.mean(samples["mu"])
    assert np.abs(posterior_mean - 3.0) < 0.2


def test_gibbs_sampler_basic():
    """Test basic Gibbs sampler functionality."""
    model = create_simple_model()
    sampler = GibbsSampler(model)

    samples = sampler.run(n_samples=1000, burn_in=100, random_seed=42)

    # Check output format
    assert isinstance(samples, dict)
    assert "mu" in samples
    assert len(samples["mu"]) == 1000


def test_gibbs_sampler_convergence():
    """Test Gibbs sampler converges."""
    model = create_simple_model()
    sampler = GibbsSampler(model)

    samples = sampler.run(n_samples=5000, burn_in=1000, random_seed=42)

    # Posterior mean should be close to sample mean
    posterior_mean = np.mean(samples["mu"])
    assert np.abs(posterior_mean - 3.0) < 0.2


def test_thinning():
    """Test thinning produces fewer samples."""
    model = create_simple_model()
    sampler = MetropolisHastings(model, proposal_std=0.5)

    samples = sampler.run(n_samples=1000, burn_in=100, thin=5, random_seed=42)

    assert len(samples["mu"]) == 1000


def test_random_seed_reproducibility():
    """Test that random seed gives reproducible results."""
    model1 = create_simple_model()
    model2 = create_simple_model()

    sampler1 = MetropolisHastings(model1, proposal_std=0.5)
    sampler2 = MetropolisHastings(model2, proposal_std=0.5)

    samples1 = sampler1.run(n_samples=100, burn_in=10, random_seed=42)
    samples2 = sampler2.run(n_samples=100, burn_in=10, random_seed=42)

    # Should be identical
    assert np.allclose(samples1["mu"], samples2["mu"])


def test_multiple_variables():
    """Test inference with multiple variables."""
    model = Model()
    rv1 = RandomVariable("mu", Normal(0, 10))
    rv2 = RandomVariable("sigma", Normal(1, 5))

    model.add_variable(rv1)
    model.add_variable(rv2)

    data = np.array([1.0, 2.0, 3.0])
    model.observe(data)

    def likelihood(params, data):
        mu = params['mu']
        sigma = params['sigma']

        if sigma <= 0:
            return -np.inf

        return np.sum(Normal(mu, sigma).logpdf(data))

    model.set_likelihood(likelihood)

    sampler = MetropolisHastings(model, proposal_std=0.2)
    samples = sampler.run(n_samples=1000, burn_in=100, random_seed=42)

    # Check both variables are sampled
    assert "mu" in samples
    assert "sigma" in samples
    assert len(samples["mu"]) == 1000
    assert len(samples["sigma"]) == 1000


def test_log_posterior_trace():
    """Test that log posterior trace is recorded."""
    model = create_simple_model()
    sampler = MetropolisHastings(model, proposal_std=0.5)

    samples = sampler.run(n_samples=1000, burn_in=100, random_seed=42)

    assert hasattr(sampler, 'log_posterior_trace')
    assert len(sampler.log_posterior_trace) == 1000
    assert np.all(np.isfinite(sampler.log_posterior_trace))
