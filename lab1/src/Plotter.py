import numpy as np
from matplotlib import pyplot as plt
from .problem import Problem
from .constants import G_GLOBAL_MIN_ELEV, G_GLOBAL_MIN_AZIM


class Plotter:
    def __init__(self, problem: Problem, mode: str):
        if problem.num_of_vars > 2:
            raise ValueError("Maximum number of dimensions supported by the plotter is 3.")
        self._problem = problem
        self._dimensions = self._problem.num_of_vars + 1
        self._ax = plt.figure().add_subplot(111, projection='3d' if self._dimensions == 3 else None)
        self._mode = mode

        self._solving_path = []

    @property
    def ax(self):
        return self._ax

    def initialize_plot(self):
        arg_values = np.arange(-7, 7, .01)
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

    def plot_solution_data(self, points: np.ndarray):
        path_x_values = np.array([point[0] for point in points])
        path_y_values = np.array(
            [point[1] for point in points]) if self._dimensions == 3 else self._problem.calculate_function_value(
            path_x_values)

        plot_values = [path_x_values, path_y_values]
        if self._dimensions == 3:
            path_z_values = self._problem.calculate_function_value(np.array([path_x_values, path_y_values]))
            plot_values.append(path_z_values)
            self._ax.view_init(elev=G_GLOBAL_MIN_ELEV, azim=G_GLOBAL_MIN_AZIM)


        if self._mode == "path":
            # connect plot_values with lines
            self._ax.plot(*plot_values, color='red', zorder=2, marker='.', linewidth=2)
            plt.show()
        elif self._mode == "no_solution":
            plt.show()

    def add_point(self, x):
        self._solving_path.append(x)
