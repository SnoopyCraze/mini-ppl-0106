"""MCMC inference algorithms for the mini PPL."""

import numpy as np
from typing import Dict, List
from tqdm import tqdm


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler."""

    def __init__(self, model, proposal_std=None):
        """
        Parameters
        ----------
        model : Model
            The probabilistic model to perform inference on
        proposal_std : dict or float, optional
            Standard deviation for Gaussian proposals.
            If dict, maps variable names to std values.
            If float, uses same std for all variables.
            If None, uses adaptive values (0.1 * prior std).
        """
        self.model = model
        self.proposal_std = proposal_std
        self.samples = {}
        self.acceptance_rate = 0.0
        self.log_posterior_trace = []

    def _get_proposal_std(self, var_name):
        """Get proposal standard deviation for a variable."""
        if self.proposal_std is None:
            # Use adaptive std based on prior
            rv = self.model.variables[var_name]
            if hasattr(rv.distribution, 'sigma'):
                return rv.distribution.sigma * 0.1
            else:
                return 0.1
        elif isinstance(self.proposal_std, dict):
            return self.proposal_std.get(var_name, 0.1)
        else:
            return self.proposal_std

    def run(self, n_samples=1000, burn_in=100, thin=1, random_seed=None):
        """
        Run Metropolis-Hastings sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        burn_in : int
            Number of initial samples to discard
        thin : int
            Keep every thin-th sample
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary mapping variable names to sample arrays
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize
        self.model.initialize_values()
        free_vars = self.model.get_free_variables()

        # Storage for samples
        total_iterations = burn_in + n_samples * thin
        all_samples = {var: [] for var in free_vars}
        self.log_posterior_trace = []

        # Current state
        current_log_posterior = self.model.log_posterior()
        n_accepted = 0

        # Run MCMC
        for i in tqdm(range(total_iterations), desc="Metropolis-Hastings"):
            # Propose new values for all variables
            proposed_values = {}
            for var_name in free_vars:
                current_val = self.model.variables[var_name].value
                std = self._get_proposal_std(var_name)
                proposed_values[var_name] = current_val + np.random.normal(0, std)

            # Compute acceptance probability
            try:
                proposed_log_posterior = self.model.log_posterior(proposed_values)
                log_accept_ratio = proposed_log_posterior - current_log_posterior

                # Accept or reject
                if np.log(np.random.rand()) < log_accept_ratio:
                    # Accept
                    for var_name, val in proposed_values.items():
                        self.model.variables[var_name].set_value(val)
                    current_log_posterior = proposed_log_posterior
                    n_accepted += 1
            except (ValueError, FloatingPointError):
                # Reject if proposal is invalid
                pass

            # Store sample (after burn-in and thinning)
            if i >= burn_in and (i - burn_in) % thin == 0:
                for var_name in free_vars:
                    all_samples[var_name].append(self.model.variables[var_name].value)
                self.log_posterior_trace.append(current_log_posterior)

        # Convert to arrays
        self.samples = {var: np.array(samples) for var, samples in all_samples.items()}
        self.acceptance_rate = n_accepted / total_iterations
        self.log_posterior_trace = np.array(self.log_posterior_trace)

        return self.samples


class GibbsSampler:
    """Gibbs sampler for models with conjugate relationships."""

    def __init__(self, model, conditional_samplers=None):
        """
        Parameters
        ----------
        model : Model
            The probabilistic model to perform inference on
        conditional_samplers : dict, optional
            Dictionary mapping variable names to conditional sampling functions.
            Each function should take (model, var_name) and return a sample.
        """
        self.model = model
        self.conditional_samplers = conditional_samplers or {}
        self.samples = {}
        self.log_posterior_trace = []

    def _default_conditional_sample(self, var_name):
        """
        Default conditional sampler using Metropolis-Hastings on single variable.

        Parameters
        ----------
        var_name : str
            Name of variable to sample

        Returns
        -------
        float
            New sample for the variable
        """
        current_val = self.model.variables[var_name].value

        # Propose new value
        if hasattr(self.model.variables[var_name].distribution, 'sigma'):
            std = self.model.variables[var_name].distribution.sigma * 0.1
        else:
            std = 0.1

        proposed_val = current_val + np.random.normal(0, std)

        # Compute acceptance probability
        current_values = {name: rv.value for name, rv in self.model.variables.items()}
        proposed_values = current_values.copy()
        proposed_values[var_name] = proposed_val

        try:
            current_log_p = self.model.log_posterior(current_values)
            proposed_log_p = self.model.log_posterior(proposed_values)
            log_accept_ratio = proposed_log_p - current_log_p

            if np.log(np.random.rand()) < log_accept_ratio:
                return proposed_val
            else:
                return current_val
        except (ValueError, FloatingPointError):
            return current_val

    def run(self, n_samples=1000, burn_in=100, thin=1, random_seed=None):
        """
        Run Gibbs sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        burn_in : int
            Number of initial samples to discard
        thin : int
            Keep every thin-th sample
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary mapping variable names to sample arrays
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize
        self.model.initialize_values()
        free_vars = self.model.get_free_variables()

        # Storage for samples
        total_iterations = burn_in + n_samples * thin
        all_samples = {var: [] for var in free_vars}
        self.log_posterior_trace = []

        # Run MCMC
        for i in tqdm(range(total_iterations), desc="Gibbs Sampling"):
            # Sample each variable conditional on others
            for var_name in free_vars:
                if var_name in self.conditional_samplers:
                    # Use custom conditional sampler
                    new_val = self.conditional_samplers[var_name](self.model, var_name)
                else:
                    # Use default Metropolis step
                    new_val = self._default_conditional_sample(var_name)

                self.model.variables[var_name].set_value(new_val)

            # Store sample (after burn-in and thinning)
            if i >= burn_in and (i - burn_in) % thin == 0:
                for var_name in free_vars:
                    all_samples[var_name].append(self.model.variables[var_name].value)
                self.log_posterior_trace.append(self.model.log_posterior())

        # Convert to arrays
        self.samples = {var: np.array(samples) for var, samples in all_samples.items()}
        self.log_posterior_trace = np.array(self.log_posterior_trace)

        return self.samples
