"""Misc utility functions."""
import numpy as np

def create_grid(width: int, height: int) -> np.ndarray:
    """Return row-major grid of pixel coordinates."""
    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    p = np.stack([x, y], axis=-1)
    return p.reshape([-1, 2])
