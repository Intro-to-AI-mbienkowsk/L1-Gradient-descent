from abc import ABC, abstractmethod
from .constants import DEFAULT_BETA, DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS
from .Plotter import Plotter
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
        x = x0 if isinstance(x0, np.ndarray) else np.array([x0])
        points_visited = [x]
        iterations = 0
        gradient_value = problem.calculate_gradient_value(x)
        while (iterations < self.get_parameters()["max_iterations"] and
               np.linalg.norm(gradient_value) > self.get_parameters()["epsilon"]):
            x = x - self.get_parameters()["beta"] * gradient_value
            gradient_value = problem.calculate_gradient_value(x)
            points_visited.append(x)
            iterations += 1

        if self.get_parameters()["debug"]:
            starting_point_feedback = f"For the starting point {x0}, the algorithm has"
            convergence_feedback = (f"converged after {iterations} iterations"
                                    if iterations < self.get_parameters()["max_iterations"]
                                    else f"exceeded the maximum number of iterations! ({iterations})")
            solution_feedback = f"at the point {[round(coord, 3) for coord in x]}."
            print(f"{starting_point_feedback} {convergence_feedback} {solution_feedback}")
        if self.get_parameters()["plot"]:
            plotter = Plotter(problem, self._parameters["plot"])
            plotter.initialize_plot()
            plotter.plot_solution_data(np.array(points_visited))
        return x

    def __init__(self, beta=DEFAULT_BETA, epsilon=DEFAULT_EPSILON, max_iterations=DEFAULT_MAX_ITERATIONS, plot="path",
                 debug=False):
        self._parameters = {"beta": beta, "epsilon": epsilon,
                            "max_iterations": max_iterations, "debug": debug, "plot": plot}
