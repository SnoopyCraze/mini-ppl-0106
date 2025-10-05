"""Utility functions for Bayesian inference."""

import numpy as np


def posterior_predictive(model, samples, n_predictions=1000):
    """
    Generate posterior predictive samples.

    Parameters
    ----------
    model : Model
        The probabilistic model
    samples : dict
        Dictionary of parameter samples from MCMC
    n_predictions : int
        Number of predictive samples per parameter sample

    Returns
    -------
    array
        Posterior predictive samples
    """
    n_samples = len(samples[list(samples.keys())[0]])
    predictions = []

    for i in range(n_samples):
        # Set parameter values from this sample
        param_values = {name: samples[name][i] for name in samples.keys()}

        # Generate predictions with these parameters
        # This is model-specific and should be customized
        for _ in range(n_predictions // n_samples):
            # Sample from likelihood with these parameters
            pred = sample_from_likelihood(model, param_values)
            predictions.append(pred)

    return np.array(predictions)


def sample_from_likelihood(model, param_values):
    """
    Sample from the likelihood given parameter values.

    This is a helper that should be customized per model.

    Parameters
    ----------
    model : Model
        The probabilistic model
    param_values : dict
        Parameter values

    Returns
    -------
    float or array
        Sample from likelihood
    """
    # This is a placeholder - should be implemented per model
    raise NotImplementedError("Implement sample_from_likelihood for your specific model")


def posterior_summary(samples, var_names=None, credible_interval=0.95):
    """
    Compute summary statistics for posterior samples.

    Parameters
    ----------
    samples : dict
        Dictionary mapping variable names to sample arrays
    var_names : list, optional
        Variables to summarize. If None, summarize all.
    credible_interval : float
        Credible interval width (default 0.95 for 95% CI)

    Returns
    -------
    dict
        Summary statistics for each variable
    """
    if var_names is None:
        var_names = list(samples.keys())

    alpha = 1 - credible_interval
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    summary = {}
    for var_name in var_names:
        sample_array = samples[var_name]
        summary[var_name] = {
            'mean': np.mean(sample_array),
            'std': np.std(sample_array),
            'median': np.median(sample_array),
            f'{credible_interval*100:.0f}% CI': (
                np.percentile(sample_array, lower_percentile),
                np.percentile(sample_array, upper_percentile)
            ),
            'min': np.min(sample_array),
            'max': np.max(sample_array),
        }

    return summary


def print_summary(samples, var_names=None, credible_interval=0.95):
    """
    Print formatted summary statistics.

    Parameters
    ----------
    samples : dict
        Dictionary mapping variable names to sample arrays
    var_names : list, optional
        Variables to summarize. If None, summarize all.
    credible_interval : float
        Credible interval width (default 0.95 for 95% CI)
    """
    summary = posterior_summary(samples, var_names, credible_interval)

    print("\n" + "=" * 70)
    print("POSTERIOR SUMMARY")
    print("=" * 70)

    for var_name, stats in summary.items():
        print(f"\n{var_name}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        ci_lower, ci_upper = stats[f'{credible_interval*100:.0f}% CI']
        print(f"  {credible_interval*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    print("\n" + "=" * 70 + "\n")


def log_sum_exp(x):
    """
    Numerically stable log-sum-exp.

    Parameters
    ----------
    x : array
        Array of log values

    Returns
    -------
    float
        log(sum(exp(x)))
    """
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def compute_waic(model, samples):
    """
    Compute Watanabe-Akaike Information Criterion (WAIC).

    Parameters
    ----------
    model : Model
        The probabilistic model
    samples : dict
        Dictionary of parameter samples from MCMC

    Returns
    -------
    float
        WAIC value (lower is better)
    """
    n_samples = len(samples[list(samples.keys())[0]])

    if model.observed_data is None:
        raise ValueError("Model must have observed data to compute WAIC")

    # Compute log pointwise predictive density
    lppd = 0.0
    p_waic = 0.0

    # For each data point
    for i, y_i in enumerate(model.observed_data):
        log_likes = []

        # Evaluate likelihood at each posterior sample
        for j in range(n_samples):
            param_values = {name: samples[name][j] for name in samples.keys()}
            # This is simplified - real implementation needs pointwise likelihood
            log_like = model.log_likelihood(param_values)
            log_likes.append(log_like / len(model.observed_data))

        log_likes = np.array(log_likes)
        lppd += log_sum_exp(log_likes) - np.log(n_samples)
        p_waic += np.var(log_likes)

    waic = -2 * (lppd - p_waic)
    return waic


def compute_dic(model, samples):
    """
    Compute Deviance Information Criterion (DIC).

    Parameters
    ----------
    model : Model
        The probabilistic model
    samples : dict
        Dictionary of parameter samples from MCMC

    Returns
    -------
    float
        DIC value (lower is better)
    """
    # Mean of log likelihood
    log_likes = []
    for i in range(len(samples[list(samples.keys())[0]])):
        param_values = {name: samples[name][i] for name in samples.keys()}
        log_likes.append(model.log_likelihood(param_values))

    # Deviance at posterior mean
    mean_params = {name: np.mean(samples[name]) for name in samples.keys()}
    d_mean = -2 * model.log_likelihood(mean_params)

    # Mean deviance
    d_bar = -2 * np.mean(log_likes)

    # Effective number of parameters
    p_d = d_bar - d_mean

    # DIC
    dic = d_bar + p_d

    return dic
