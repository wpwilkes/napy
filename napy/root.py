"""
Root finding algorithms.
"""

from typing import Callable

from napy.exceptions import ConvergenceFailure


def bisect(
    f: Callable[[float], float],
    left_endpoint: float,
    right_endpoint: float,
    max_iteration: int = 100,
    tolerance: float = 1e-5,
) -> float:
    """
    Root finding via the bisection method.

    Parameters
    ----------
    f : Callable[[float], float]
        The function to find the root of.
    left_endpoint : float
        The left endpoint of the interval to search in.
    right_endpoint : float
        The right endpoint of the interval to search in.
    max_iteration : int
        Maximum number of iterations.
    tolerance : float
        The accuracy to which the root is estimated.

    Returns
    -------
    root : float
        The estimated root of the function `f`.
    """
    if right_endpoint <= left_endpoint:
        raise ValueError(
            f"Received malformed interval: {[left_endpoint, right_endpoint]}."
        )

    if f(left_endpoint) * f(right_endpoint) >= 0:
        raise ValueError("Expected `f` to have have opposite signs at endpoints.")

    for _ in range(max_iteration):
        mid_point: float = left_endpoint + (right_endpoint - left_endpoint) / 2
        f_at_mid_point: float = f(mid_point)

        if f_at_mid_point == 0 or (right_endpoint - left_endpoint) / 2 < tolerance:
            return mid_point

        if f(left_endpoint) * f(mid_point) > 0:
            left_endpoint = mid_point
        else:
            right_endpoint = mid_point

    raise ConvergenceFailure(
        "Failed to converge with given max iteration count and tolerance."
        + f" Final estimate: root={mid_point}, f(root)={f(mid_point)}."
    )


def fixed_point(
    f: Callable[[float], float],
    point_0: float,
    max_iteration: int = 100,
    tolerance: float = 1e-5,
) -> float:
    """
    Root finding via fixed-point iteration.

    Parameters
    ----------
    f : Callable[[float], float]
        The function to find the fixed-point of.
    point_0: float
        Initial estimate of fixed point.
    tolerance : float
        The accuracy to which the fixed-point is estimated.
    max_iteration : int
        Maximum number of iterations.

    Returns
    -------
    point_n : float
        The estimated fixed-point of the function `f`.
    """
    for _ in range(max_iteration):
        point_n: float = f(point_0)
        if abs(point_n - point_0) < tolerance:
            return point_n
        point_0 = point_n

    raise ConvergenceFailure(
        "Failed to converge with given max iteration count and tolerance."
        + f" Final estimate: point_n={point_n}, f(point_n)={f(point_n)}."
    )
