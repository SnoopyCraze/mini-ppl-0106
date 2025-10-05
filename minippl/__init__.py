"""Mini Probabilistic Programming Language (PPL)

A minimal PPL implementation with MCMC inference capabilities.
"""

from .distributions import Normal, Uniform, Bernoulli, Exponential, Beta
from .random_variable import RandomVariable
from .model import Model
from .inference import MetropolisHastings, GibbsSampler
from .visualize import plot_posterior, plot_trace, plot_autocorrelation

__version__ = "0.1.0"

__all__ = [
    "Normal",
    "Uniform",
    "Bernoulli",
    "Exponential",
    "Beta",
    "RandomVariable",
    "Model",
    "MetropolisHastings",
    "GibbsSampler",
    "plot_posterior",
    "plot_trace",
    "plot_autocorrelation",
]
