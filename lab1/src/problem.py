from typing import Callable, Tuple
from numbers import Number
import numpy as np


class Problem:
    def __init__(self, num_of_variables: int, function: callable,
                 gradient: tuple[callable] or callable):
        # todo: validation whether num_of_vars equals number of parameters in function and gradient

        # convert single-dimensional gradients to a standard tuple of functions like in higher dimensions
        if not isinstance(gradient, tuple):
            gradient = (gradient, )

        if num_of_variables < 0:
            raise ValueError("Number of variables can't be negative!")
        if len(gradient) != num_of_variables:
            raise ValueError("Number of elements in gradient has to be equal to the number of variables!")

        self._function = function
        self._gradient = gradient
        self._num_of_vars = num_of_variables

    @property
    def num_of_vars(self):
        return self._num_of_vars

    def calculate_function_value(self, x0: np.ndarray[Number] | Number) -> Number:
        """
        Calculates the value of the problem's function at a given point x0
        :param x0: point in the R^n space, n = len(x0)
        :return: the value of the function
        """
        if not isinstance(x0, np.ndarray):
            x0 = np.array(x0)

        if len(x0) != self.num_of_vars:
            raise ValueError(
                f"Wrong number of variables in x0 ({len(x0)}). Required: {self.num_of_vars}.")
        return self._function(x0)

    def calculate_gradient_value(self, x0: np.ndarray[Number] | Number) -> np.ndarray[Number]:
        """
        Calculates the value of the problem's gradient at a given point x0
        :param x0: point in the R^n space where n = len(x0)
        :return: the gradient of the function at x0, a np.array of size 1 x n
        """
        if not isinstance(x0, np.ndarray):
            x0 = np.array(x0)

        if x0.size != self.num_of_vars:
            raise ValueError(f"Wrong number of variables in x0 ({len(x0)}). Required: {self.num_of_vars}.")

        # iterate over all elements of the vector and calculate respective gradients
        if x0.ndim != 0:
            gradient = np.array([self._gradient[k](x0) for k in range(self.num_of_vars)])
        else:
            # since a 0d array is not iterable, separate case for it
            gradient = np.array([self._gradient[0](x0)])
        return gradient
