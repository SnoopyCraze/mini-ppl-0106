"""Unit tests for Model class."""

import numpy as np
import pytest
from minippl import Model, RandomVariable, Normal


def test_model_creation():
    """Test Model creation."""
    model = Model()
    assert len(model.variables) == 0
    assert model.likelihood_fn is None


def test_add_variable():
    """Test adding variables to model."""
    model = Model()
    rv = RandomVariable("x", Normal(0, 1))
    model.add_variable(rv)

    assert "x" in model.variables
    assert model.variables["x"] == rv


def test_observe_data():
    """Test setting observed data."""
    model = Model()
    data = np.array([1, 2, 3, 4, 5])
    model.observe(data)

    assert model.observed_data is not None
    assert np.array_equal(model.observed_data, data)


def test_log_prior():
    """Test log prior computation."""
    model = Model()
    rv1 = RandomVariable("x", Normal(0, 1))
    rv2 = RandomVariable("y", Normal(0, 1))

    rv1.set_value(0)
    rv2.set_value(0)

    model.add_variable(rv1)
    model.add_variable(rv2)

    # At mean, each should contribute -0.5 * log(2*pi)
    log_prior = model.log_prior()
    expected = -2 * 0.5 * np.log(2 * np.pi)

    assert np.abs(log_prior - expected) < 1e-10


def test_log_likelihood():
    """Test log likelihood computation."""
    model = Model()
    rv = RandomVariable("mu", Normal(0, 1))
    rv.set_value(0)

    model.add_variable(rv)
    data = np.array([0, 0, 0])
    model.observe(data)

    def likelihood(params, data):
        mu = params['mu']
        # Simple normal likelihood with fixed sigma=1
        return np.sum(Normal(mu, 1).logpdf(data))

    model.set_likelihood(likelihood)

    log_like = model.log_likelihood()
    # Three points at mean should each contribute -0.5*log(2*pi)
    expected = -3 * 0.5 * np.log(2 * np.pi)

    assert np.abs(log_like - expected) < 1e-10


def test_log_posterior():
    """Test log posterior computation."""
    model = Model()
    rv = RandomVariable("x", Normal(0, 1))
    rv.set_value(0)

    model.add_variable(rv)
    data = np.array([0])
    model.observe(data)

    def likelihood(params, data):
        x = params['x']
        return Normal(x, 1).logpdf(data[0])

    model.set_likelihood(likelihood)

    log_post = model.log_posterior()
    # Prior at mean + likelihood at mean
    expected = -2 * 0.5 * np.log(2 * np.pi)

    assert np.abs(log_post - expected) < 1e-10


def test_get_free_variables():
    """Test getting non-observed variables."""
    model = Model()
    rv1 = RandomVariable("x", Normal(0, 1))
    rv2 = RandomVariable("y", Normal(0, 1))

    rv2.observe(5.0)

    model.add_variable(rv1)
    model.add_variable(rv2)

    free_vars = model.get_free_variables()

    assert len(free_vars) == 1
    assert "x" in free_vars
    assert "y" not in free_vars


def test_initialize_values():
    """Test initialization of variable values."""
    model = Model()
    rv1 = RandomVariable("x", Normal(0, 1))
    rv2 = RandomVariable("y", Normal(0, 1))

    model.add_variable(rv1)
    model.add_variable(rv2)

    # Initially None
    assert rv1.value is None
    assert rv2.value is None

    model.initialize_values()

    # Should be sampled
    assert rv1.value is not None
    assert rv2.value is not None
