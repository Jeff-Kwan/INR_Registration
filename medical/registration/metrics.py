"""Native-voxel Jacobian diagnostic for DIR-Lab maps."""
import numpy as np


def jacobian_determinant(displacement):
    """det(I+du/dx); pixel-space determinant equals its physical conjugate."""
    rows = []
    for component in range(3):
        dz, dy, dx = np.gradient(displacement[..., component])
        rows.append([dx, dy, dz])
        rows[-1][component] = rows[-1][component] + 1
    a, b, c = rows
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])).astype(np.float32)
