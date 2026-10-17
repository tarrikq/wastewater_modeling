"""Internal helpers for functions that accept a float or a float64 array.

Convention used across ``wdtk``: the first argument of a vectorized function may be a float
or an array; other arguments may be floats or arrays that broadcast to its shape. The result
has the type and shape of the first argument.
"""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

F = TypeVar("F", float, NDArray[np.float64])
FloatOrArray = float | NDArray[np.float64]


def like(template: F, value: NDArray[np.float64] | float, name: str) -> F:
    """Return ``value`` with the type and shape of ``template``.

    Raises ``ValueError`` if ``value`` has a different shape, i.e. other arguments broadcast
    the result beyond the shape of the first argument.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != np.shape(template):
        raise ValueError(
            f"Arguments broadcast to shape {arr.shape}, larger than {name} shape "
            f"{np.shape(template)}; pass {name} as an array of the full shape."
        )
    result: float | NDArray[np.float64] = float(arr) if arr.ndim == 0 else arr
    return cast(F, result)


def require_positive(name: str, value: FloatOrArray) -> None:
    """Raise ``ValueError`` unless every element is finite and > 0."""
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")


def require_nonnegative(name: str, value: FloatOrArray) -> None:
    """Raise ``ValueError`` unless every element is finite and >= 0."""
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
