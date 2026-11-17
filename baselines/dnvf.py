"""Run the released single-scale DNVF loop with localized data and diagnostics."""
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

import numpy as np
import torch
from .common import execution, finish, preflight, predict_for_case, sample_xyz_field
from .sources import native_modules, upstream_imports
from .precision import ieee_fp32


class PairDataset:
    """In-memory prepared pair in the native loader's channel-first XYZ layout."""
    def __init__(self, case):
        self.images = [torch.from_numpy(case[k]['data'].copy())[None] for k in ('moving', 'fixed')]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Native segmentations are used for validation logging only. Shared
        # scoring reads the real evaluation labels after fitting has finished.
        return (*self.images, torch.zeros_like(self.images[0]), torch.zeros_like(self.images[1]))


def run_case(case, config, device):
    preflight(case, config, 'dnvf')
    started = time.monotonic()
    with upstream_imports('dnvf') as root, execution(device, config['seed'], cuda_required=True), ieee_fp32():
        network, datasets, utils = native_modules('dnvf')
        shape = tuple(case['fixed']['data'].shape)
        captured, models = {}, []
        original_siren, original_int = network.Siren, network.VecInt
        iterations = config['native']['optimization_epoch']

        def siren(*args, **kwargs):
            model = original_siren(*args, **kwargs)
            models.append(model)
            return model

        class CaptureIntegration(original_int):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def forward(self, vector):
                output = super().forward(vector)
                self.calls += 1
                if self.calls == iterations:
                    captured['flow'] = output.detach().cpu()
                    captured['state'] = {k: v.detach().cpu().clone() for k, v in models[0].state_dict().items()}
                return output

        with TemporaryDirectory(prefix='dnvf-') as scratch:
            arguments = ['opt_nf_diff.py', '--save_dir', scratch + '/', '--dataset', 'OASIS']
            for key, value in config['native'].items():
                arguments.extend(['--' + key, str(value)])
            with patch.object(datasets, 'load_test_dataset', return_value=(PairDataset(case), shape)), \
                 patch.object(network, 'Siren', side_effect=siren), \
                 patch.object(network, 'VecInt', CaptureIntegration), \
                 patch.object(utils, 'dice_OASIS', return_value=0.), \
                 patch.object(sys, 'argv', arguments):
                entry = runpy.run_path(str(Path(root) / 'opt_nf_diff.py'))
                entry['main']()
            if 'flow' not in captured:
                raise RuntimeError('Native DNVF loop did not produce its final validation map')
            displacement = captured['flow'][0].movedim(0, -1).numpy()
            identity = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0)
            mapping = identity + displacement
            predicted = predict_for_case(case, config, lambda points: sample_xyz_field(mapping, points))
            checkpoint = dict(kind='network_weights', state=captured['state'],
                              snapshot='final native validation map, before its last optimizer update')
            return finish('dnvf', case, config, predicted, checkpoint, started,
                          variant='native single-scale neural velocity field with seven-step scaling and squaring',
                          precision='IEEE FP32; TF32 convolution and matmul disabled; no autocast',
                          preprocessing='preserve prepared image values; native spatial loader replaced in memory',
                          optimization_roi='whole_image',
                          native_validation='segmentation logging callback disabled; real labels scored by shared evaluator')
