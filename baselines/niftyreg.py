"""Pinned NiftyReg reg_f3d defaults, evaluated through native reg_transform."""
from pathlib import Path
import os
import subprocess
from tempfile import TemporaryDirectory
import time

import nibabel as nib
import numpy as np
from .common import finish, preflight
from .sources import verify_source, native_modules
from .physical import ras_mapping_predictor, provenance_checkpoint


def run_case(case, config, device):
    preflight(case, config, 'niftyreg')
    verify_source('niftyreg')
    registration, transform = native_modules('niftyreg')
    started = time.monotonic()
    environment = dict(os.environ, OMP_NUM_THREADS='4')
    with TemporaryDirectory(prefix='niftyreg-') as scratch:
        scratch = Path(scratch)
        control, deformation = scratch / 'control.nii.gz', scratch / 'deformation.nii.gz'
        # Defaults define the similarity, optimizer, pyramid and regularizers.
        # The output paths confine all native image artifacts to scratch.
        command = [str(registration), '-ref', case['fixed']['path'], '-flo', case['moving']['path'],
                   '-cpp', str(control), '-res', str(scratch / 'warped.nii.gz'), '-omp', '4']
        fit_started = time.monotonic()
        subprocess.run(command, cwd=scratch, env=environment, check=True)
        registration_seconds = time.monotonic() - fit_started
        subprocess.run([str(transform), '-ref', case['fixed']['path'], '-def', str(control), str(deformation),
                        '-omp', '4'], cwd=scratch, env=environment, check=True)
        image = nib.load(deformation)
        field = np.asarray(image.dataobj)
        if field.ndim == 5 and field.shape[3] == 1:
            field = field[:, :, :, 0, :]
        if not np.allclose(image.affine, case['fixed']['affine'], atol=1e-5):
            raise ValueError('Native deformation changed the fixed evaluation grid')
        predict = ras_mapping_predictor(field, case)
        return finish('niftyreg', case, config, predict, provenance_checkpoint(), started,
                      variant='reg_f3d public defaults', backend='CPU OpenMP',
                      registration_seconds=registration_seconds,
                      native_command=command, preprocessing='native image intensities', optimization_roi='whole_image',
                      prediction_boundary='border extension of native dense physical deformation')
