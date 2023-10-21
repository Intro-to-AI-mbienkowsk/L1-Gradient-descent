import numpy as np

DEFAULT_BETA = 0.03
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_EPSILON = .001


def g(x):
    x1, x2 = x
    term1 = 1.5 - np.exp(-x1 ** 2 - x2 ** 2)
    term2 = -0.5 * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
    return term1 + term2


def gradient_g(x):
    x1, x2 = x
    term1 = 2 * x1 * np.exp(-x1 ** 2 - x2 ** 2) + (x1 - 1) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
    term2 = 2 * x2 * np.exp(-x1 ** 2 - x2 ** 2) + (x2 + 2) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
    return np.array([term1, term2])


def f(x):
    return .25 * x ** 4


def gradient_f(x):
    return x ** 3
