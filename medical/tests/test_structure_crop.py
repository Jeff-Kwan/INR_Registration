"""Synthetic crop selection checks; no training steps or saved-model recovery."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from medical.configuration import load_config, validate_config
from medical.registration.endpoint_sampling import StructureCropSampler
from medical.registration.symmetric_registration import SymmetricPairRegistration, sample_crop_indices


class TestStructureCrop(unittest.TestCase):
    def sampler(self, fraction=.5):
        moving = torch.zeros(21, 23, 25)
        fixed = torch.ones_like(moving)
        def scores(image, affine, sigma):
            score = np.zeros(tuple(image.shape), dtype=np.float32)
            score[(5, 6, 7) if image[0, 0, 0] == 0 else (15, 16, 17)] = 1
            return score
        with patch('medical.registration.endpoint_sampling.structure_scores', side_effect=scores):
            return StructureCropSampler(moving, fixed, torch.eye(4),
                {'structure_fraction': fraction, 'sigma_mm': 2.}, seed=42)

    def test_half_usual_and_equal_endpoint_structure_centers(self):
        sampler = self.sampler()
        draws = [sampler.sample_center((21, 23, 25)) for _ in range(2000)]
        counts = [sum(p is None for p in draws)]
        for point in ((7., 6., 5.), (17., 16., 15.)):
            counts.append(sum(p is not None and tuple(p.tolist()) == point for p in draws))
        self.assertEqual(sum(counts), 2000)
        for actual, expected in zip(counts, (1000, 500, 500)):
            self.assertLess(abs(actual - expected), 70)
        other = self.sampler()
        for expected in draws[:20]:
            actual = other.sample_center((21, 23, 25))
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(actual, expected)

    def test_direct_center_overrides_mask_only_for_structured_branch(self):
        image = torch.zeros(21, 23, 25)
        branch = SymmetricPairRegistration(image, image, torch.eye(4), grid_size=4,
            particles=32, crop_fraction=.4, compile_derivatives=False)
        branch.crop_center_sites = [torch.tensor([[0., 0., 0.]])]
        generator = torch.Generator().manual_seed(42)
        for center in (torch.tensor([17., 16., 15.]), torch.tensor([24., 22., 20.]), None):
            indices = sample_crop_indices(branch, 128, generator, center_xyz=center)
            xyz = torch.stack((indices % 25, (indices // 25) % 23, indices // (25 * 23)), -1)
            self.assertTrue(branch.supported(xyz).all())
            self.assertTrue((branch.origin >= 0).all())
            self.assertTrue((branch.origin + branch.domain_extent <= branch.extent).all())
            if center is None:
                torch.testing.assert_close(branch.origin, torch.zeros(3))
            elif center[0] == 17:
                self.assertTrue(((branch.origin + branch.domain_extent / 2 - center).abs() <= .5).all())

    def test_constant_fallback_and_zero_fraction(self):
        image = torch.ones(5, 6, 7)
        sampler = StructureCropSampler(image, image, torch.eye(4),
            {'structure_fraction': 1., 'sigma_mm': 2.}, seed=42)
        self.assertTrue(all(s.metadata['fallback_uniform'] for s in sampler.samplers))
        points = torch.stack([sampler.sample_center(image.shape) for _ in range(100)])
        self.assertTrue(((points >= 0) & (points <= torch.tensor([6, 5, 4]))).all())
        self.assertGreater(len(points.unique(dim=0)), 50)
        zero = self.sampler(0.)
        self.assertTrue(all(zero.sample_center(image.shape) is None for _ in range(20)))

    def test_configuration(self):
        config = load_config('medical/configs/native_structure_crop.json')
        default = load_config('medical/configs/default.json')
        clean = copy.deepcopy(config)
        clean['training'].pop('structure_crop_sampling')
        self.assertEqual(clean, default)
        for options in ({'structure_fraction': .5}, {'structure_fraction': 1.1, 'sigma_mm': 2.},
                        {'structure_fraction': .5, 'sigma_mm': 0.}):
            bad = copy.deepcopy(config)
            bad['training']['structure_crop_sampling'] = options
            with self.assertRaises(ValueError):
                validate_config(bad)
        config['objective']['registration']['crop_fraction'] = None
        with self.assertRaisesRegex(ValueError, 'cropped'):
            validate_config(config)

    def test_training_wires_shared_crop_before_any_loss(self):
        from medical.registration.training import fit_pair
        config = load_config('medical/configs/native_structure_crop.json')
        config['model'].update(width=4, layers=1)
        config['runtime'].update(device='cpu', compile_derivatives=False, compile_endpoints=False)
        config['training'].update(batch_size=128)
        config['objective']['registration'].update(compile_velocity=False, crop_fraction=.4)
        image = torch.rand(21, 23, 25)
        mask = torch.zeros_like(image, dtype=torch.bool)
        mask[0, 0, 0] = True
        class StopBeforeLoss(Exception):
            pass
        def capture(pair, batch, *args, **kwargs):
            self.assertIs(batch.coordinates, batch.target_coordinates)
            torch.testing.assert_close(batch.source, batch.target)
            self.assertEqual(pair.training_metadata['crop_sampling']['transport'], False)
            xyz = (batch.coordinates + 1) * torch.tensor([24., 22., 20.]) / 2
            self.assertTrue(((xyz >= torch.tensor([12., 12., 11.]) - 1e-5)
                            & (xyz <= torch.tensor([21., 21., 19.]) + 1e-5)).all())
            raise StopBeforeLoss
        with patch.object(StructureCropSampler, 'sample_center', return_value=torch.tensor([17., 16., 15.])), \
             patch('medical.registration.training.endpoint_loss', side_effect=capture):
            with self.assertRaises(StopBeforeLoss):
                fit_pair(image, image, config, affine=torch.eye(4), moving_mask=mask, fixed_mask=mask)
