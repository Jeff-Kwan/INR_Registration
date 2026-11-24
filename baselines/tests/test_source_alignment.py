"""Source-math parity on synthetic inputs. No optimizer, fit or real-data run."""
from contextlib import nullcontext,redirect_stdout
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.ndimage import map_coordinates
import torch

from baselines import ccidir,nir
from baselines.sources import upstream_imports,native_modules
from baselines.case_runner import main as case_main
from baselines.tests.test_adapters import config


class SourceAlignmentTests(unittest.TestCase):
    def test_nir_coordinates_match_native_inference_helper(self):
        with upstream_imports('nir'):
            native_modules('nir')  # Imports only; no construction/training.
            utility=importlib.import_module('util')
            shape=(7,9,11)
            points=np.linspace(-.5,.5,27,dtype=np.float32).reshape(1,3,3,3)
            volume=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            native=utility.warp_3d_volume(volume,points,shape,order=1)
            voxels=nir.native_output_voxels(points,shape,shape)
            ours=map_coordinates(volume,voxels.transpose(3,0,1,2),order=1,mode='constant',cval=0)
            np.testing.assert_allclose(ours,native,atol=1e-5)
            wrong=(points+1)*(np.asarray(shape)-1)/2
            self.assertGreater(np.max(np.abs(voxels-wrong)),.1)

    def test_ccidir_second_order_matches_native_dense_inference(self):
        class Quadratic(torch.nn.Module):
            def __init__(self, a, b):
                super().__init__()
                self.a, self.b = a, b

            def forward(self, x):
                return self.a * x.square() + self.b * x + .01

        forward, reverse = Quadratic(.05, .02), Quadratic(-.03, .04)
        reg = SimpleNamespace(network=forward, network_rev=reverse, fit=Mock())
        shape = (17, 19, 23)
        points = np.array([[8.2, 9.3, 10.4], [5.1, 7.2, 12.3]], dtype=np.float32)
        extent = np.asarray(shape[::-1]) - 1
        case = {key: {'data': np.zeros(shape, dtype=np.float32)} for key in ('fixed', 'moving')}
        with upstream_imports('ccidir'):
            _, general = native_modules('ccidir')
            zeros = torch.zeros

            def cpu_zeros(*args, **kwargs):
                if str(kwargs.get('device', '')).startswith('cuda'):
                    kwargs['device'] = 'cpu'
                return zeros(*args, **kwargs)

            def check_prediction(name, case, config, predict, checkpoint, started, **details):
                np.testing.assert_allclose(predict(points), native_prediction, atol=1e-6)
                self.assertEqual(details['inference_order'], 2)

            fake_models = SimpleNamespace(ImplicitRegistrator=Mock(return_value=reg))
            with patch.object(torch.Tensor, 'cuda', lambda t: t), patch.object(torch, 'zeros', cpu_zeros):
                coords = torch.tensor(2 * points[:, ::-1] / extent - 1, dtype=torch.float32).requires_grad_(True)
                a, _, offset = general.compute_coordinates_cycle(forward, reverse, coords, inf_order=2)
                native_prediction = ((a + .5 * offset).detach().numpy() + 1) * extent / 2
                native_prediction = native_prediction[:, ::-1]
                a1, _, offset1 = general.compute_coordinates_cycle(forward, reverse, coords, inf_order=1)
                self.assertGreater(float((a + .5 * offset - a1 - .5 * offset1).abs().max().detach()), 1e-7)
                with patch.object(ccidir, 'execution', return_value=nullcontext()), \
                     patch.object(ccidir, 'native_modules', return_value=[fake_models, general]), \
                     patch.object(ccidir, 'preflight'), \
                     patch.object(ccidir, 'finish', side_effect=check_prediction):
                    ccidir.run_case(case, config('ccidir'), 'cpu')
            reg.fit.assert_called_once()

    def test_seed_override_reaches_saved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(config('ccidir')))
            with patch('baselines.case_runner.run_case',return_value={}) as run,patch('baselines.case_runner.importlib.import_module',return_value=SimpleNamespace(run_case=Mock())),redirect_stdout(io.StringIO()):
                case_main(['--method','ccidir','--config',str(path),'--moving','moving','--fixed','fixed','--output',directory,'--seed','17'])
            self.assertEqual(run.call_args.args[2]['seed'],17)
            self.assertEqual(json.loads(path.read_text())['seed'],42)

if __name__=='__main__': unittest.main()
