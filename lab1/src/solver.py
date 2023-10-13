from abc import ABC, abstractmethod
from constants import DEFAULT_BETA, DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS


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
        x = x0
        iterations = 0
        while True:
            d = problem.calculate_gradient_value(x0)
            x = x - self.get_parameters()["beta"] * d
            iterations += 1
            if iterations == 100:
                return x

    def __init__(self, parameters=None):
        if parameters is None:
            self._parameters = {"beta": DEFAULT_BETA, "epsilon": DEFAULT_EPSILON,
                                "max_iterations": DEFAULT_MAX_ITERATIONS}
        else:
            self._parameters["beta"] = DEFAULT_BETA if "beta" not in parameters else parameters["beta"]
            self._parameters["epsilon"] = DEFAULT_EPSILON if "epsilon" not in parameters else parameters["epsilon"]
            self._parameters["max_iterations"] = DEFAULT_MAX_ITERATIONS if "max_iterations" not in parameters \
                else parameters["max_iterations"]
