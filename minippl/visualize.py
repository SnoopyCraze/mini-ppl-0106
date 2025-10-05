"""Visualization tools for MCMC diagnostics and posterior analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_posterior(samples, var_name=None, bins=30, show_stats=True, ax=None):
    """
    Plot posterior distribution histogram.

    Parameters
    ----------
    samples : dict or array
        If dict, mapping of variable names to sample arrays.
        If array, samples for a single variable.
    var_name : str, optional
        Name of variable to plot (if samples is dict)
    bins : int
        Number of histogram bins
    show_stats : bool
        Whether to show mean and credible interval
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    # Extract samples array
    if isinstance(samples, dict):
        if var_name is None:
            var_name = list(samples.keys())[0]
        sample_array = samples[var_name]
    else:
        sample_array = samples
        if var_name is None:
            var_name = "variable"

    # Plot histogram
    ax.hist(sample_array, bins=bins, density=True, alpha=0.7, edgecolor='black')

    # Add statistics
    if show_stats:
        mean = np.mean(sample_array)
        ci_lower = np.percentile(sample_array, 2.5)
        ci_upper = np.percentile(sample_array, 97.5)

        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
        ax.axvline(ci_lower, color='green', linestyle=':', linewidth=2,
                   label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        ax.axvline(ci_upper, color='green', linestyle=':', linewidth=2)
        ax.legend()

    ax.set_xlabel(var_name)
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior Distribution: {var_name}')
    ax.grid(True, alpha=0.3)

    if fig is not None:
        plt.tight_layout()
        return fig


def plot_trace(samples, var_name=None, ax=None):
    """
    Plot trace plot for MCMC samples.

    Parameters
    ----------
    samples : dict or array
        If dict, mapping of variable names to sample arrays.
        If array, samples for a single variable.
    var_name : str, optional
        Name of variable to plot (if samples is dict)
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = None

    # Extract samples array
    if isinstance(samples, dict):
        if var_name is None:
            var_name = list(samples.keys())[0]
        sample_array = samples[var_name]
    else:
        sample_array = samples
        if var_name is None:
            var_name = "variable"

    # Plot trace
    ax.plot(sample_array, alpha=0.7, linewidth=1)
    ax.axhline(np.mean(sample_array), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(sample_array):.3f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(var_name)
    ax.set_title(f'Trace Plot: {var_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if fig is not None:
        plt.tight_layout()
        return fig


def plot_autocorrelation(samples, var_name=None, max_lag=50, ax=None):
    """
    Plot autocorrelation function.

    Parameters
    ----------
    samples : dict or array
        If dict, mapping of variable names to sample arrays.
        If array, samples for a single variable.
    var_name : str, optional
        Name of variable to plot (if samples is dict)
    max_lag : int
        Maximum lag to compute
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    # Extract samples array
    if isinstance(samples, dict):
        if var_name is None:
            var_name = list(samples.keys())[0]
        sample_array = samples[var_name]
    else:
        sample_array = samples
        if var_name is None:
            var_name = "variable"

    # Compute autocorrelation
    acf = compute_autocorrelation(sample_array, max_lag)

    # Plot
    ax.bar(range(len(acf)), acf, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation Function: {var_name}')
    ax.grid(True, alpha=0.3)

    if fig is not None:
        plt.tight_layout()
        return fig


def compute_autocorrelation(x, max_lag):
    """
    Compute autocorrelation function.

    Parameters
    ----------
    x : array
        Time series data
    max_lag : int
        Maximum lag

    Returns
    -------
    array
        Autocorrelation values
    """
    x = np.array(x)
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for k in range(1, max_lag + 1):
        if k < len(x):
            c_k = np.dot(x[:-k], x[k:]) / len(x)
            acf[k] = c_k / c0
        else:
            acf[k] = 0.0

    return acf


def plot_diagnostics(samples, var_names=None):
    """
    Create comprehensive diagnostic plots (trace, posterior, autocorrelation).

    Parameters
    ----------
    samples : dict
        Dictionary mapping variable names to sample arrays
    var_names : list, optional
        List of variable names to plot. If None, plot all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if var_names is None:
        var_names = list(samples.keys())

    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 3, figsize=(15, 4 * n_vars))

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for i, var_name in enumerate(var_names):
        plot_trace(samples, var_name, ax=axes[i, 0])
        plot_posterior(samples, var_name, ax=axes[i, 1])
        plot_autocorrelation(samples, var_name, ax=axes[i, 2])

    plt.tight_layout()
    return fig


def compute_effective_sample_size(samples, var_name=None):
    """
    Compute effective sample size using autocorrelation.

    Parameters
    ----------
    samples : dict or array
        Samples from MCMC
    var_name : str, optional
        Variable name if samples is dict

    Returns
    -------
    float
        Effective sample size
    """
    if isinstance(samples, dict):
        if var_name is None:
            var_name = list(samples.keys())[0]
        sample_array = samples[var_name]
    else:
        sample_array = samples

    n = len(sample_array)
    max_lag = min(n - 1, 100)
    acf = compute_autocorrelation(sample_array, max_lag)

    # Sum autocorrelations until they become negligible
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] < 0.05:
            break
        tau += 2 * acf[k]

    return n / tau


def compute_rhat(chains):
    """
    Compute Gelman-Rubin convergence diagnostic (R-hat).

    Parameters
    ----------
    chains : list of arrays
        Multiple independent MCMC chains

    Returns
    -------
    float
        R-hat statistic (should be < 1.1 for convergence)
    """
    chains = [np.array(chain) for chain in chains]
    m = len(chains)  # number of chains
    n = len(chains[0])  # length of each chain

    # Chain means and overall mean
    chain_means = np.array([np.mean(chain) for chain in chains])
    overall_mean = np.mean(chain_means)

    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)

    # Within-chain variance
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chains])
    W = np.mean(chain_vars)

    # Marginal posterior variance
    var_plus = ((n - 1) / n) * W + (1 / n) * B

    # R-hat
    rhat = np.sqrt(var_plus / W)

    return rhat
