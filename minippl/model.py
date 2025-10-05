"""Probabilistic model definition for the mini PPL."""

import numpy as np
from typing import Dict, List, Callable


class Model:
    """Container for a probabilistic model."""

    def __init__(self):
        """Initialize an empty model."""
        self.variables = {}
        self.likelihood_fn = None
        self.observed_data = None

    def add_variable(self, rv):
        """
        Add a random variable to the model.

        Parameters
        ----------
        rv : RandomVariable
            Random variable to add
        """
        self.variables[rv.name] = rv

    def set_likelihood(self, likelihood_fn):
        """
        Set the likelihood function for the model.

        Parameters
        ----------
        likelihood_fn : callable
            Function that takes variable values and returns log-likelihood
            Signature: likelihood_fn(var_dict) -> float
        """
        self.likelihood_fn = likelihood_fn

    def observe(self, data):
        """
        Set observed data for the model.

        Parameters
        ----------
        data : array-like
            Observed data
        """
        self.observed_data = np.array(data)

    def log_prior(self, var_values=None):
        """
        Compute log prior probability.

        Parameters
        ----------
        var_values : dict, optional
            Dictionary mapping variable names to values.
            If None, use current values in variables.

        Returns
        -------
        float
            Log prior probability
        """
        log_p = 0.0
        for name, rv in self.variables.items():
            if rv.observed:
                continue
            if var_values is not None:
                value = var_values.get(name, rv.value)
            else:
                value = rv.value
            log_p += rv.logpdf(value)
        return log_p

    def log_likelihood(self, var_values=None):
        """
        Compute log likelihood.

        Parameters
        ----------
        var_values : dict, optional
            Dictionary mapping variable names to values.
            If None, use current values in variables.

        Returns
        -------
        float
            Log likelihood
        """
        if self.likelihood_fn is None:
            raise ValueError("Likelihood function not set")

        if var_values is None:
            var_values = {name: rv.value for name, rv in self.variables.items()}

        return self.likelihood_fn(var_values, self.observed_data)

    def log_posterior(self, var_values=None):
        """
        Compute log posterior probability (up to normalization constant).

        Parameters
        ----------
        var_values : dict, optional
            Dictionary mapping variable names to values.
            If None, use current values in variables.

        Returns
        -------
        float
            Log posterior probability
        """
        return self.log_prior(var_values) + self.log_likelihood(var_values)

    def get_free_variables(self):
        """
        Get list of non-observed variables.

        Returns
        -------
        list
            List of variable names that are not observed
        """
        return [name for name, rv in self.variables.items() if not rv.observed]

    def initialize_values(self):
        """Initialize all non-observed variables by sampling from their priors."""
        for rv in self.variables.values():
            if not rv.observed:
                rv.sample()
