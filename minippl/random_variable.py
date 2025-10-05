"""Random variable representation for the mini PPL."""

import numpy as np


class RandomVariable:
    """Represents a random variable with a prior distribution."""

    def __init__(self, name, distribution):
        """
        Parameters
        ----------
        name : str
            Name of the random variable
        distribution : Distribution
            Prior distribution for this variable
        """
        self.name = name
        self.distribution = distribution
        self.value = None
        self.observed = False

    def set_value(self, value):
        """Set the current value of the random variable."""
        self.value = value

    def observe(self, value):
        """Mark this variable as observed with a fixed value."""
        self.value = value
        self.observed = True

    def logpdf(self, value=None):
        """Compute log probability of the value under the prior."""
        if value is None:
            value = self.value
        return self.distribution.logpdf(value)

    def sample(self):
        """Draw a sample from the prior distribution."""
        self.value = self.distribution.sample()[0]
        return self.value

    def __repr__(self):
        obs_str = " (observed)" if self.observed else ""
        return f"RandomVariable({self.name}, value={self.value}{obs_str})"
