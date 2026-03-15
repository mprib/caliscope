"""Internal helpers for TOML serialization of numpy arrays.

Not part of the public API. Used by domain object to_toml()/from_toml() methods.
"""

from typing import Any

import numpy as np


def _clean_scalar(value: Any) -> Any:
    """Handle TOML 'null' string artifacts or actual None values.

    Returns None if value is None or the string 'null'.
    """
    if value is None or value == "null":
        return None
    return value


def _array_to_list(arr: np.ndarray | None) -> list | None:
    """Convert numpy array to nested list for TOML serialization."""
    return arr.tolist() if arr is not None else None


def _list_to_array(lst: Any, dtype: type[np.generic] = np.float64) -> np.ndarray | None:
    """Convert list back to numpy array from TOML deserialization.

    Handles both proper TOML null (None) and string literal "null".

    Raises:
        ValueError: If lst is not None, "null", or a valid list
    """
    if lst is None or lst == "null":
        return None
    if not isinstance(lst, list):
        raise ValueError(f"Expected list or None, got {type(lst).__name__}: {lst}")
    return np.array(lst, dtype=dtype)
