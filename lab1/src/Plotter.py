import numpy as np
from matplotlib import pyplot as plt
from problem import Problem


class Plotter:
    def __init__(self, problem: Problem):
        if problem.num_of_vars > 2:
            raise ValueError("Maximum number of dimensions supported by the plotter is 3.")
        self._problem = problem
        self._dimensions = self._problem.num_of_vars + 1

        self.x_values = np.arange(-5, 5, .001)
        self._fig = plt.figure()

        if self._dimensions == 3:
            self.y_values = np.arange(-5, 5, .001)
        self._solving_path = []

    def initialize_plot(self):
        if self._dimensions == 3:
            X, Y = np.meshgrid(self.x_values, self.y_values)
            Z = self._problem.calculate_function_value(np.array([X, Y]))
            ax = self._fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('f(x1, x2')
        else:
            pass

    def add_point(self, x):
        self._solving_path.append(x)


    def update_plot(self):
        plt.scatter(self.sol)


        plt.show()




