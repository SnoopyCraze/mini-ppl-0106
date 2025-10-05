"""Unit tests for probability distributions."""

import numpy as np
import pytest
from minippl.distributions import Normal, Uniform, Bernoulli, Exponential, Beta


def test_normal_creation():
    """Test Normal distribution creation."""
    dist = Normal(mu=0, sigma=1)
    assert dist.mu == 0
    assert dist.sigma == 1

    # Test invalid sigma
    with pytest.raises(ValueError):
        Normal(mu=0, sigma=-1)


def test_normal_sampling():
    """Test Normal distribution sampling."""
    dist = Normal(mu=5, sigma=2)
    samples = dist.sample(size=10000)

    assert len(samples) == 10000
    assert np.abs(np.mean(samples) - 5) < 0.1
    assert np.abs(np.std(samples) - 2) < 0.1


def test_normal_logpdf():
    """Test Normal log probability."""
    dist = Normal(mu=0, sigma=1)
    logp = dist.logpdf(0)

    # At mean, logpdf should be -0.5 * log(2*pi)
    expected = -0.5 * np.log(2 * np.pi)
    assert np.abs(logp - expected) < 1e-10


def test_uniform_creation():
    """Test Uniform distribution creation."""
    dist = Uniform(low=0, high=1)
    assert dist.low == 0
    assert dist.high == 1

    # Test invalid bounds
    with pytest.raises(ValueError):
        Uniform(low=1, high=0)


def test_uniform_sampling():
    """Test Uniform distribution sampling."""
    dist = Uniform(low=0, high=10)
    samples = dist.sample(size=10000)

    assert len(samples) == 10000
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)
    assert np.abs(np.mean(samples) - 5) < 0.2


def test_bernoulli_creation():
    """Test Bernoulli distribution creation."""
    dist = Bernoulli(p=0.5)
    assert dist.p == 0.5

    # Test invalid p
    with pytest.raises(ValueError):
        Bernoulli(p=1.5)


def test_bernoulli_sampling():
    """Test Bernoulli distribution sampling."""
    dist = Bernoulli(p=0.7)
    samples = dist.sample(size=10000)

    assert len(samples) == 10000
    assert np.all((samples == 0) | (samples == 1))
    assert np.abs(np.mean(samples) - 0.7) < 0.02


def test_exponential_creation():
    """Test Exponential distribution creation."""
    dist = Exponential(lam=1.0)
    assert dist.lam == 1.0

    # Test invalid lambda
    with pytest.raises(ValueError):
        Exponential(lam=-1)


def test_exponential_sampling():
    """Test Exponential distribution sampling."""
    dist = Exponential(lam=2.0)
    samples = dist.sample(size=10000)

    assert len(samples) == 10000
    assert np.all(samples >= 0)
    # Mean should be 1/lambda
    assert np.abs(np.mean(samples) - 0.5) < 0.02


def test_beta_creation():
    """Test Beta distribution creation."""
    dist = Beta(alpha=2, beta=2)
    assert dist.alpha == 2
    assert dist.beta == 2

    # Test invalid parameters
    with pytest.raises(ValueError):
        Beta(alpha=-1, beta=2)


def test_beta_sampling():
    """Test Beta distribution sampling."""
    dist = Beta(alpha=2, beta=5)
    samples = dist.sample(size=10000)

    assert len(samples) == 10000
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)
    # Mean should be alpha / (alpha + beta)
    expected_mean = 2 / (2 + 5)
    assert np.abs(np.mean(samples) - expected_mean) < 0.01
