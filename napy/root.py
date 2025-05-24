"""
Root finding algorithms.
"""

from typing import Callable

from napy.exceptions import ConvergenceFailure


def bisect(
    f: Callable[[float], float],
    left_endpoint: float,
    right_endpoint: float,
    tolerance: float = 1e-5,
    max_iteration: int = 100,
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
    tolerance : float
        The accuracy to which the root is estimated.
    max_iteration : int
        Maximum number of iterations.

    Returns
    -------
    root : float
        The estimated root of the function f
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
        "Failed to converge with given tolerance and max iteration count."
        + f" Final estimate: root={mid_point}, f(root)={f(mid_point)}."
    )
