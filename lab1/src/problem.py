from typing import Callable, Tuple
from numbers import Number


class Problem:
    def __init__(self, num_of_vars: int, function: Callable[[Tuple[Number]], Number],
                 gradient: Tuple[Callable[[Tuple[Number]], Number]]):
        # todo: validation whether num_of_vars equals number of parameters in function and gradient
        if num_of_vars < 0:
            raise ValueError("Number of variables can't be negative!")
        if len(gradient) != num_of_vars:
            raise ValueError("Number of elements in gradient has to be equal to the number of variables!")

        self._function = function
        self._gradient = gradient
        self._num_of_vars = num_of_vars

    @property
    def num_of_vars(self):
        return self._num_of_vars

    def calculate_function_value(self, x0: Tuple[Number]) -> Number:
        if len(x0) != self.num_of_vars:
            raise ValueError(f"Wrong number of variables in x0 ({len(x0)}). Required: {self.num_of_vars}.")
        return self._function(x0)

    def calculate_gradient_value(self, x0: Tuple[Number]) -> Tuple[Number, ...]:
        if len(x0) != self.num_of_vars:
            raise ValueError(f"Wrong number of variables in x0 ({len(x0)}). Required: {self.num_of_vars}.")
        return tuple(self._gradient[i](x0) for i in range(self.num_of_vars))
