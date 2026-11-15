"""Real spawned worker lifecycle with a CPU-only dummy handler; no fitting."""
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from medical.gpu_worker import GPUWorker


def dummy_case(job):
    import numpy as np
    import torch
    if job['output'].endswith('fail'): raise RuntimeError('deliberate case failure')
    model=torch.nn.Linear(3,1)
    Path(job['output']).write_text(json.dumps({'python':random.random(),'numpy':float(np.random.rand()),
        'weights':model.weight.tolist(),'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']}))


class TestGPUWorker(unittest.TestCase):
    def test_process_reuse_fresh_seed_state_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);config=root/'config.json';config.write_text('{"runtime":{"seed":42}}')
            with GPUWorker(3,handler=dummy_case) as worker:
                self.assertIsNone(worker.process)
                records=[]
                for name in ('first','fail','last'):
                    command=['python','--moving','unused','--fixed','unused','--config',str(config),'--output',str(root/name)]
                    records.append(worker.run(command,root/(name+'.log')))
                self.assertEqual([r['returncode'] for r in records],[0,1,0])
                self.assertEqual(json.loads((root/'first').read_text()),json.loads((root/'last').read_text()))
                self.assertEqual(json.loads((root/'first').read_text())['gpu'],'3')
                self.assertIn('deliberate case failure',(root/'fail.log').read_text())
                process=worker.process
            self.assertFalse(process.is_alive())
