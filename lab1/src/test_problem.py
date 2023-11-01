from lab1.src.problem import Problem
from pytest import raises


def square(x):
    return x ** 2


def negative_cube(x):
    return -x ** 3


def difference_of_squares(x0):
    x, y = x0
    return x ** 2 - y ** 2


def sum_of_cubes(x0):
    x, y = x0
    return x ** 3 + y ** 3


def test_problem_constructor():
    problem = Problem(1, square, (negative_cube,))
    assert (problem.num_of_vars == 1)


def test_problem_constructor_negative_nov():
    with raises(ValueError):
        problem = Problem(-3, square, (negative_cube,))


def test_problem_calculate_function_value_single_val():
    problem = Problem(2, difference_of_squares, (sum_of_cubes, difference_of_squares))
    assert (problem.calculate_function_value((-1, 0)) == 1)
    assert (problem.calculate_function_value((3, 4)) == -7)


def test_problem_calculate_gradient_value_single_val():
    problem = Problem(2, difference_of_squares, (sum_of_cubes, difference_of_squares))
    assert (problem.calculate_gradient_value((-1, 0)) == (-1, 1))
    assert (problem.calculate_gradient_value((3, 4)) == (91, -7))
