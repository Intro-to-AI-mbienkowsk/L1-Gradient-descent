from lab1.src.problem import Problem
from pytest import raises

def square(x):
    return x ** 2


def negative_cube(x):
    return -x ** 3


def difference_of_squares(x, y):
    return x ** 2 - y ** 2


def sum_of_cubes(x, y):
    return x ** 3 + y ** 3


def test_problem_constructor():
    problem = Problem(1, square, negative_cube)
    assert(problem.num_of_vars == 1)


def test_problem_constructor_negative_nov():
    with raises(ValueError):
        problem = Problem(-3, square, negative_cube)

