"""Standardized metric and read-only launcher checks; no registration."""
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from medical.comparison_metrics import evaluate_arrays, labels_for
from medical.comparison_experiments import main


class ComparisonTests(unittest.TestCase):
    def test_identity_and_negative_jacobian(self):
        shape=(8,9,10)
        labels=(np.arange(np.prod(shape)).reshape(shape)%4).astype(np.uint8)
        grid=np.indices(shape).transpose(1,2,3,0).astype(np.float32)
        score=evaluate_arrays(labels,labels,grid,[1,2,3],[1.5,2.,3.],'acdc')
        self.assertEqual(score['Dice']['mean'],1)
        self.assertEqual(score['HD95']['mean'],0)
        self.assertEqual(score['NPJ']['mean'],0)
        self.assertEqual(score['SDlogJ']['mean'],0)
        grid[...,0]=shape[0]-1-grid[...,0]
        score=evaluate_arrays(labels,labels,grid,[1,2,3],[1.5,2.,3.],'acdc')
        self.assertEqual(score['NPJ']['mean'],100)
        self.assertAlmostEqual(score['SDlogJ']['mean'],0)

    def test_missing_surfaces_and_spacing(self):
        shape=(8,9,10)
        labels=np.zeros(shape,np.uint8);labels[2:4,2:5,2:6]=1
        grid=np.indices(shape).transpose(1,2,3,0).astype(np.float32)
        grid[...,0]+=1
        score=evaluate_arrays(labels,labels,grid,[1],[2.,1.,1.],'acdc')
        self.assertEqual(score['HD95']['mean'],2.)
        grid+=100
        score=evaluate_arrays(labels,labels,grid,[1],[2.,1.,1.],'acdc')
        self.assertEqual(score['Dice']['mean'],0)
        self.assertIsNone(score['HD95']['mean'])
        self.assertEqual(score['HD95']['scored_labels'],0)

    def test_frozen_label_sets(self):
        self.assertEqual(len(labels_for('ixi')),30)
        self.assertEqual(len(labels_for('lpba40')),56)
        self.assertEqual(labels_for('acdc'),[1,2,3])
        self.assertEqual(len(labels_for('neurite_oasis')),35)

    def test_plan_does_not_start_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'absent'
            with patch('medical.comparison_experiments.make_jobs',return_value=[]), \
                 patch('medical.comparison_experiments.GPUWorker') as worker,redirect_stdout(io.StringIO()):
                plan=main(['--methods','ours','--output',str(output)])
            self.assertFalse(plan['execution'])
            self.assertFalse(output.exists())
            worker.assert_not_called()

    def test_cpu_queue_runs_while_gpu_worker_is_busy(self):
        import json, threading
        from medical.configuration import load_config
        started=threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config=root/'input.json';config.write_text(json.dumps(load_config()))
            output=root/'run'
            jobs=[]
            for method,device in [('ours','cuda:0'),('demons','cpu')]:
                target=output/method
                jobs.append(dict(dataset='neurite_oasis',method=method,pair_index=1,output=str(target),device=device,seed=42,
                    missing_inputs=[],command=['python','--config',str(config),'--output',str(target)]))
            def complete(command):
                target=Path(command[command.index('--output')+1]);target.mkdir()
                (target/'metrics.json').write_text(json.dumps({k:dict(mean=1.) for k in ('Dice','HD95','NPJ','SDlogJ')}))
            def gpu(command,log):
                self.assertTrue(started.wait(5),'CPU queue failed to run concurrently')
                complete(command);return dict(returncode=0)
            def cpu(command,**kwargs):
                from types import SimpleNamespace
                started.set();complete(command);return SimpleNamespace(returncode=0)
            with patch('medical.comparison_experiments.make_jobs',return_value=jobs), \
                 patch('medical.comparison_experiments.GPUWorker.run',side_effect=gpu), \
                 patch('medical.comparison_experiments.subprocess.run',side_effect=cpu):
                main(['--run','--output',str(output),'--gpus','0','--cpu-workers','1'])
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['jobs']),2)
