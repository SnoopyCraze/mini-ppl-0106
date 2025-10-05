"""Probability distributions for the mini PPL."""

import numpy as np
from abc import ABC, abstractmethod
from scipy import stats


class Distribution(ABC):
    """Base class for probability distributions."""

    @abstractmethod
    def logpdf(self, x):
        """Compute log probability density/mass function."""
        pass

    @abstractmethod
    def sample(self, size=1):
        """Draw random samples from the distribution."""
        pass


class Normal(Distribution):
    """Normal (Gaussian) distribution."""

    def __init__(self, mu=0.0, sigma=1.0):
        """
        Parameters
        ----------
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation (must be positive)
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        """Compute log probability density."""
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)

    def sample(self, size=1):
        """Draw random samples."""
        return np.random.normal(self.mu, self.sigma, size=size)


class Uniform(Distribution):
    """Uniform distribution."""

    def __init__(self, low=0.0, high=1.0):
        """
        Parameters
        ----------
        low : float
            Lower bound
        high : float
            Upper bound (must be > low)
        """
        if high <= low:
            raise ValueError("high must be greater than low")
        self.low = low
        self.high = high

    def logpdf(self, x):
        """Compute log probability density."""
        return stats.uniform.logpdf(x, loc=self.low, scale=self.high - self.low)

    def sample(self, size=1):
        """Draw random samples."""
        return np.random.uniform(self.low, self.high, size=size)


class Bernoulli(Distribution):
    """Bernoulli distribution."""

    def __init__(self, p=0.5):
        """
        Parameters
        ----------
        p : float
            Probability of success (must be in [0, 1])
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def logpdf(self, x):
        """Compute log probability mass."""
        return stats.bernoulli.logpmf(x, self.p)

    def sample(self, size=1):
        """Draw random samples."""
        return np.random.binomial(1, self.p, size=size)


class Exponential(Distribution):
    """Exponential distribution."""

    def __init__(self, lam=1.0):
        """
        Parameters
        ----------
        lam : float
            Rate parameter (must be positive)
        """
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam

    def logpdf(self, x):
        """Compute log probability density."""
        return stats.expon.logpdf(x, scale=1/self.lam)

    def sample(self, size=1):
        """Draw random samples."""
        return np.random.exponential(1/self.lam, size=size)


class Beta(Distribution):
    """Beta distribution."""

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Parameters
        ----------
        alpha : float
            First shape parameter (must be positive)
        beta : float
            Second shape parameter (must be positive)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        """Compute log probability density."""
        return stats.beta.logpdf(x, self.alpha, self.beta)

    def sample(self, size=1):
        """Draw random samples."""
        return np.random.beta(self.alpha, self.beta, size=size)
