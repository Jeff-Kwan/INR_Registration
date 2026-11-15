"""Suite planning and target moments; no training or GPU work."""
import importlib.util,json,tempfile,unittest,copy
from pathlib import Path
from unittest.mock import patch
import torch
from medical.registration.ncc import endpoint_statistics
from medical.configuration import validate_config
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('suite',ROOT/'scripts/run_method_suite.py');suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)

class TestSuite(unittest.TestCase):
 def test_full_plan_and_deltas(self):
  jobs,configs=suite.make_plan(suite.TRIALS,suite.DATASETS,Path('/tmp/not_created_suite_plan'))
  self.assertEqual(len(jobs),444)
  self.assertEqual(len({j['output'] for j in jobs}),444)
  base=configs['01_reference']
  for c in configs.values():
   validate_config(c);self.assertEqual(c['runtime']['seed'],42)
   self.assertFalse(c['objective']['registration']['compile_decoder'])
  expected=copy.deepcopy(base);expected.pop('working_grid')
  self.assertEqual(expected,configs['06_native_grid'])
 def test_target_moments_and_gradients(self):
  torch.manual_seed(42);a=torch.randn(128,dtype=torch.float64);b=torch.randn_like(a)
  x=(a*1.2+.1).requires_grad_();y=(b*.7-.2).requires_grad_()
  terms=endpoint_statistics(x,y,a,b,epsilon=1e-8,moment_weight=.01,moment_target='observed')
  expected=0
  for pred,truth in [(x,a),(y,b)]:
   vp=pred.var(unbiased=False)+1e-8;vt=truth.var(unbiased=False)+1e-8
   expected+=.5*(vp/vt+(pred.mean()-truth.mean()).square()/vt+torch.log(vt/vp)-1)
  torch.testing.assert_close(terms['moment_loss'],expected)
  ga=torch.autograd.grad(terms['moment_loss'],(x,y));gb=torch.autograd.grad(expected,(x,y))
  for u,v in zip(ga,gb):torch.testing.assert_close(u,v)
  zero=endpoint_statistics(a,b,a,b,epsilon=1e-8,moment_weight=.01,moment_target='observed')
  self.assertLess(abs(float(zero['moment_loss'])),1e-12)
 def test_plan_only_creates_nothing(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(suite,'GPUWorker',side_effect=AssertionError('GPU started')):
   output=Path(temp)/'output'
   with patch('builtins.print'):suite.main(['--output',str(output)])
   self.assertFalse(output.exists())
 def test_mock_execution_and_failure_accounting(self):
  with tempfile.TemporaryDirectory() as temp:
   output=Path(temp)/'suite'
   jobs,configs=suite.make_plan(suite.TRIALS[:1],['oasis'],output);jobs=jobs[:2]
   class Worker:
    def __init__(self,gpu):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def run(self,command,log):
     target=Path(command[command.index('--output')+1]);target.mkdir()
     Path(log).write_text('mock only')
     if target.name.startswith('0395'):return {'returncode':1}
     (target/'metrics.json').write_text(json.dumps({'DSC':{'mean':.8},'HD95':{'mean':2.},'LogJacDetStd':{'mean':.1},'recovery':[{'converged':True}]}))
     return {'returncode':0}
   with patch.object(suite,'make_plan',return_value=(jobs,configs)),patch.object(suite,'GPUWorker',Worker),patch('builtins.print'):
    with self.assertRaises(SystemExit):suite.main(['--output',str(output),'--trials',suite.TRIALS[0],'--datasets','oasis','--gpus','0','--run'])
   report=json.loads((output/'summary.json').read_text())['trials'][suite.TRIALS[0]]['oasis']
   self.assertEqual(report['successful'],1);self.assertEqual(report['intended'],2)
   self.assertEqual(report['metrics']['DSC']['scored'],1)
