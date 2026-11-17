"""Physical-coordinate and mocked native adapter tests; no optimizers execute."""
from pathlib import Path
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from baselines import demons, niftyreg
from baselines.physical import sitk_image, lps_displacement_predictor, ras_mapping_predictor, image_geometry
from baselines.sources import ROOT


class ClassicalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.shape = (8, 9, 10)
        affine = np.array([[0., -2., 0., 13.], [1.5, 0., 0., -7.], [0., 0., 3., 5.], [0., 0., 0., 1.]])
        data = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        self.case = {k: dict(data=data, affine=affine, path=str(root / f'{phase}.nii.gz'))
                     for k, phase in [('fixed', 'T00'), ('moving', 'T50')]}
        self.points = np.tile([3., 4., 5.], (300, 1))
        self.shift = np.array([.3, -.2, .1])
        for phase, points in [('T00', self.points), ('T50', self.points + self.shift)]:
            np.savetxt(root / f'{phase}_landmarks_voxel.csv', points, delimiter=',', header='x,y,z', comments='')
        physical = affine[:3, :3] @ self.shift
        self.ras = np.indices(self.shape).transpose(1, 2, 3, 0) @ affine[:3, :3].T + affine[:3, 3] + physical
        self.lps = np.broadcast_to(physical * [-1, -1, 1], (*self.shape, 3)).copy()

    def config(self, name):
        return json.loads((ROOT / f'configs/{name}.json').read_text())

    def assert_result(self, result):
        self.assertLess(result['metrics']['continuous_tre_mm']['mean_mm'], 1e-5)
        self.assertFalse(result['checkpoint']['parameter_checkpoint'])
        self.assertEqual(result['metrics']['backend'].split()[0], 'CPU')

    def test_physical_conversions_match_native_sitk_transform(self):
        image = sitk_image(sitk, self.case['fixed'])
        field = sitk.GetImageFromArray(self.lps.transpose(2, 1, 0, 3).copy(), isVector=True)
        field.CopyInformation(image)
        transform = sitk.DisplacementFieldTransform(field)
        predictor = lps_displacement_predictor(self.lps, self.case)
        points = np.array([[2.2, 3.1, 4.7], [-.2, 3., 4.], [-1., 3., 4.]])
        truth = []
        for point in points:
            physical = image.TransformContinuousIndexToPhysicalPoint(point.tolist())
            mapped = transform.TransformPoint(physical)
            truth.append(image.TransformPhysicalPointToContinuousIndex(mapped))
        np.testing.assert_allclose(predictor(points), truth, atol=1e-6)
        np.testing.assert_allclose(ras_mapping_predictor(self.ras, self.case)(self.points), self.points+self.shift, atol=1e-6)
        invalid = dict(self.case['fixed'])
        invalid['affine'] = invalid['affine'].copy()
        invalid['affine'][0, 0] = .5
        with self.assertRaises(ValueError):
            image_geometry(invalid)

    def test_demons_calls_native_optimizer_and_keeps_physical_direction(self):
        field = sitk.GetImageFromArray(self.lps.transpose(2, 1, 0, 3).copy(), isVector=True)
        registration = Mock()
        registration.Execute.return_value = field
        registration.GetElapsedIterations.return_value = 50
        registration.GetMetric.return_value = 0.
        with patch.object(sitk, 'DemonsRegistrationFilter', return_value=registration):
            result = demons.run_case(self.case, self.config('demons'), 'cpu')
        registration.SetNumberOfIterations.assert_called_once_with(50)
        registration.SetStandardDeviations.assert_called_once_with(1.)
        registration.Execute.assert_called_once()
        self.assert_result(result)

    def test_niftyreg_uses_defaults_and_cleans_scratch(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                calls, scratch_paths = [], []
                def run(command, **kwargs):
                    calls.append(command)
                    scratch = Path(kwargs['cwd'])
                    scratch_paths.append(scratch)
                    if '-flo' in command:
                        (scratch / 'control.nii.gz').write_bytes(b'fake native control grid')
                        if failure:
                            raise subprocess.CalledProcessError(1, command)
                    else:
                        out = Path(command[command.index('-def') + 2])
                        nib.save(nib.Nifti1Image(self.ras[:, :, :, None, :], self.case['fixed']['affine']), out)
                    return subprocess.CompletedProcess(command, 0)
                with patch.object(niftyreg, 'verify_source'), patch.object(niftyreg, 'native_modules', return_value=[Path('/mock/reg_f3d'), Path('/mock/reg_transform')]), \
                     patch.object(niftyreg.subprocess, 'run', side_effect=run):
                    if failure:
                        with self.assertRaises(subprocess.CalledProcessError):
                            niftyreg.run_case(self.case, self.config('niftyreg'), 'cpu')
                    else:
                        self.assert_result(niftyreg.run_case(self.case, self.config('niftyreg'), 'cpu'))
                        self.assertEqual(len(calls), 2)
                        self.assertNotIn('-maxit', calls[0])
                        self.assertNotIn('-aff', calls[0])
                self.assertTrue(all(not p.exists() for p in scratch_paths))
