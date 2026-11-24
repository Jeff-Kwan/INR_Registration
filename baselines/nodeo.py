"""Native NODEO registration with the recorded input-shape compatibility patch."""
import time
from types import SimpleNamespace
import numpy as np
from .common import execution, finish, preflight, sample_xyz_field
from .sources import native_modules, upstream_imports


def run_case(case, config, device):
    preflight(case, config, 'nodeo')
    started = time.monotonic()
    with upstream_imports('nodeo'), execution(device, config['seed'], cuda_required=True) as dev:
        native, _, _ = native_modules('nodeo')
        settings = SimpleNamespace(**config['native'])
        # Native loader preserves NIfTI values; labels never enter registration.
        displacement, normalized, _ = native.registration(settings, dev, case['moving']['data'], case['fixed']['data'])
        grid = normalized[0].detach().cpu().numpy()[..., ::-1]
        mapping = (grid + 1) * (np.asarray(case['fixed']['data'].shape) - 1) / 2
        def predicted(points):
            return sample_xyz_field(mapping, points)
        checkpoint = dict(kind='provenance_only', parameter_checkpoint=False,
                          reason='Native function returns selected dense maps; field persistence is disabled')
        return finish('nodeo', case, config, predicted, checkpoint, started,
                      variant='native NODEO, shape-localized encoder and decoder',
                      preprocessing='preserve prepared image values', optimization_roi='whole_image',
                      selection='native similarity-times-Jacobian criterion over final 50 iterations',
                      shape_patch='encoder flatten size and final native-size decoder interpolation')
