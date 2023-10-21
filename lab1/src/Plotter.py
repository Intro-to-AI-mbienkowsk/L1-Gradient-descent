import numpy as np
from matplotlib import pyplot as plt
from problem import Problem


class Plotter:
    def __init__(self, problem: Problem):
        if problem.num_of_vars > 2:
            raise ValueError("Maximum number of dimensions supported by the plotter is 3.")
        self._problem = problem
        self._dimensions = self._problem.num_of_vars + 1
        self._ax = plt.figure().add_subplot(111, projection='3d' if self._dimensions == 3 else None)

        self._solving_path = []

    def initialize_plot(self):
        arg_values = np.arange(-5, 5, .001)
        if self._dimensions == 3:
            X, Y = np.meshgrid(arg_values, arg_values)
            Z = self._problem.calculate_function_value(np.array([X, Y]))
            self._ax.plot_surface(X, Y, Z, cmap='viridis')
            self._ax.set_xlabel('x1')
            self._ax.set_ylabel('x2')
            self._ax.set_zlabel('g(x1, x2)')

        else:
            Y = self._problem.calculate_function_value(arg_values)
            self._ax.plot(arg_values, Y)
            self._ax.set_xlabel('x')
            self._ax.set_ylabel('f(x)')

        plt.show()

    def add_point(self, x):
        self._solving_path.append(x)

    def update_plot(self):
        pass
