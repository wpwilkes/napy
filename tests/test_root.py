"""
Test suite for the `napy.root` module.
"""

import unittest
from math import cos, sqrt

from napy.root import bisect


def f(x: float) -> float:
    return sqrt(x) - cos(x)


def p(x: float) -> float:
    return x**3 + 4*x**2 - 10


class TestBisect(unittest.TestCase):
    
    def test_bisect_with_f(self):                        
        self.assertAlmostEqual(f(bisect(f, 0, 1)), 0, 5)

    def test_bisect_with_p(self):                        
        self.assertAlmostEqual(p(bisect(p, 1, 2)), 0, 3)


if __name__ == "__main__":
    unittest.main()
