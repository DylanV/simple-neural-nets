"""Utility functions."""

from typing import Callable


def finite_difference_derivative(func: Callable, x: float, h: float=1e-8) -> float:
    """Estimate the derivative of a function at a given value using the finite difference method.

    Parameters
    ----------
    func : Callable
        The function for which the derivative will be estimated.
    x : float
        The value where func's derivative will be estimated.
    h : float
        The step size for the finite difference.

    Returns
    -------
    estimated derivative : float

    """
    return (func(x + h) - func(x)) / h
