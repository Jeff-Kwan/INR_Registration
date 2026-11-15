"""Native-coordinate image sampling for the registration objectives."""
from medical.registration.geometry import sample


def sample_points(image, positions, *, extent=None):
    """Linear original-image sampling with border extension, align_corners=True."""
    if positions.ndim != 2 or positions.shape[-1] != image.ndim or image.ndim not in (2, 3):
        raise ValueError('Expected scalar image and B native-coordinate points')
    return sample(image, positions.reshape(-1, *((1,) * (image.ndim - 1)), image.ndim), extent=extent).flatten()
