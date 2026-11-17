"""CPU synthetic/mocked checks only: no optimizer, benchmark or checkpoint recovery."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from medical import tuning_experiments as launcher, tuning_method as method
from scripts.prepare_tuning_cohort import ROOT


class TestTuningLauncher(unittest.TestCase):
    def test_plan_is_complete_and_has_no_execution_side_effects(self):
        with tempfile.TemporaryDirectory() as scratch:
            output=Path(scratch)/'plan'
            with patch.object(launcher,'GPUWorker') as worker,redirect_stdout(io.StringIO()):
                plan=launcher.main(['--output',str(output)])
            worker.assert_not_called();self.assertFalse(output.exists())
            self.assertEqual(plan['total_jobs'],128)
            self.assertFalse(plan['execution'])
            self.assertEqual({j['dataset'] for j in plan['jobs']},set(launcher.DATASETS))
            for job in plan['jobs']:self.assertIn('medical.tuning_method',job['command'])

    def test_reject_data_output_and_non42_seed(self):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):launcher.main(['--output',str(ROOT/'Data/new_outputs')])
        with tempfile.TemporaryDirectory() as scratch:
            config=json.loads((ROOT/'medical/configs/default.json').read_text());config['runtime']['seed']=43
            path=Path(scratch)/'config.json';path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):launcher.make_jobs(path,Path(scratch)/'out',['oasis'])

    def test_mocked_dispatch_snapshots_config_and_summarizes(self):
        class FakeWorker:
            def __init__(self,gpu):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def run(self,command,log):
                target=Path(command[command.index('--output')+1]);target.mkdir()
                metrics={key:dict(mean=value) for key,value in dict(Dice=.8,HD95=2.,NPJ=0.,SDlogJ=.1).items()}
                metrics['timings']=dict(training_seconds=1.,recovery_seconds=.2,evaluation_seconds=.1)
                (target/'metrics.json').write_text(json.dumps(metrics))
                return dict(returncode=0)
        with tempfile.TemporaryDirectory() as scratch:
            output=Path(scratch)/'run'
            with patch.object(launcher,'GPUWorker',FakeWorker),patch.object(launcher,'verify_frozen'):
                launcher.main(['--run','--datasets','acdc','--gpus','0','1','--output',str(output)])
            summary=json.loads((output/'summary.json').read_text())
            self.assertTrue(summary['complete']);self.assertEqual(len(summary['jobs']),32)
            self.assertAlmostEqual(summary['macro_dataset_Dice'],.8)
            self.assertTrue((output/'config.json').is_file())

    def test_adapter_normalization_map_direction_and_scratch_cleanup(self):
        shape=(8,9,10);values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)+10
        labels=(np.indices(shape)[0]%3+1).astype(np.uint8)
        config=json.loads((ROOT/'medical/configs/default.json').read_text())
        case={k:dict(path=f'/synthetic/{k}',data=values.copy(),affine=np.eye(4)) for k in ('moving','fixed')}
        model=torch.nn.Linear(1,1);model.model_config={}
        xyz=torch.from_numpy(np.indices(shape).transpose(1,2,3,0).astype(np.float32))
        captured=[]
        def fake_recover(*args,**kwargs):
            captured.append(Path(args[3]));self.assertTrue(captured[-1].is_dir())
            return SimpleNamespace(inverse=xyz.permute(2,1,0,3),diagnostics={})
        for dataset in ('oasis','acdc','lpba40','ixi'):
            row=dict(dataset=dataset,pair_index=1,fixed_label='fixed.nii',moving_label='moving.nii')
            with self.subTest(dataset=dataset),patch.object(method,'membership',return_value=row),patch.object(method,'fit_pair',return_value=model) as fit,patch.object(method,'recover',side_effect=fake_recover),patch.object(method.nib,'load',return_value=SimpleNamespace(dataobj=labels)),patch.object(method,'labels_for',return_value=[1,2,3]):
                result=method.run_case(case,config,'cpu')
                actual=fit.call_args.args[0].numpy()
                expected=(values-values.min())/np.ptp(values) if dataset=='oasis' else values
                np.testing.assert_array_equal(actual,expected.transpose(2,1,0))
                self.assertEqual(result['metrics']['Dice']['mean'],1.)
                self.assertEqual(result['metrics']['NPJ']['mean'],0.)
                self.assertFalse(captured[-1].exists())
        row=dict(dataset='acdc',pair_index=1,fixed_label='fixed.nii',moving_label='moving.nii')
        with patch.object(method,'membership',return_value=row),patch.object(method,'fit_pair',return_value=model),patch.object(method,'recover',side_effect=fake_recover),patch.object(method.nib,'load',side_effect=RuntimeError('synthetic scoring failure')):
            with self.assertRaises(RuntimeError):method.run_case(case,config,'cpu')
        self.assertFalse(captured[-1].exists())
