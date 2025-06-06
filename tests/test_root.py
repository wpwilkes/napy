"""
Test suite for the `napy.root` module.
"""

import unittest
from math import cos, sin, sqrt

from napy.root import bisect, false_position, fixed_point, newton, secant


def f(x: float) -> float:
    return sqrt(x) - cos(x)


def g(x: float) -> float:
    return sqrt((x + 3) / (x**2 + 2))


def h(x: float) -> float:
    return cos(x) - x


def h_prime(x: float) -> float:
    return -sin(x) - 1


def p(x: float) -> float:
    return x**3 + 4 * x**2 - 10


def q(x: float) -> float:
    return x**4 + 2 * x**2 - x - 3


class TestBisect(unittest.TestCase):

    def test_bisect_with_f(self):
        self.assertAlmostEqual(f(bisect(f, 0, 1)), 0, 5)

    def test_bisect_with_p(self):
        self.assertAlmostEqual(p(bisect(p, 1, 2)), 0, 3)


class TestFalsePosition(unittest.TestCase):

    def test_false_position_with_h(self):
        self.assertAlmostEqual(false_position(h, 0.75, 0.25), 0.73908512457, 5)


class TestFixedPoint(unittest.TestCase):

    def test_fixed_point_with_q_and_g(self):
        self.assertAlmostEqual(q(fixed_point(g, 10)), 0, 5)


class TestNewton(unittest.TestCase):

    def test_newton_with_h(self):
        self.assertAlmostEqual(newton(h, h_prime, 0.75), 0.73908512457, 5)


class TestSecant(unittest.TestCase):

    def test_secant_with_h(self):
        self.assertAlmostEqual(secant(h, 0.75, 0.25), 0.73908512457, 5)


if __name__ == "__main__":
    unittest.main()
