from abc import ABC, abstractmethod
from constants import DEFAULT_BETA, DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS
from Plotter import Plotter
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, x0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...


class GradientSolver(Solver):
    def get_parameters(self):
        return self._parameters

    def solve(self, problem, x0, *args, **kwargs):
        if self.get_parameters()["plot"]:
            plotter = Plotter(problem)
            plotter.initialize_plot()

        x = x0
        for _ in range(self.get_parameters()["max_iterations"]):
            d = problem.calculate_gradient_value(x)
            if np.linalg.norm(d) <= self.get_parameters()["epsilon"]:
                if self.get_parameters()["debug"]:
                    print(f"Converged after {_} iterations!")
                return x
            x = x - self.get_parameters()["beta"] * d

        if self.get_parameters()["debug"]:
            print("Maximum number of iterations exceeded!")
        return x

    def __init__(self, beta=DEFAULT_BETA, epsilon=DEFAULT_EPSILON, max_iterations=DEFAULT_MAX_ITERATIONS, plot=True,
                 debug=False):
        self._parameters = {"beta": beta, "epsilon": epsilon,
                            "max_iterations": max_iterations, "debug": debug, "plot": plot}
