"""One reusable spawned process per GPU; parent launchers retain their shared queue."""
from contextlib import redirect_stdout, redirect_stderr
import gc
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import traceback


def _run_case(job):
    from medical.case_runner import run_case
    config=json.loads(Path(job['config']).read_text())
    method=importlib.import_module(job['method']).run_case
    run_case(job['moving'],job['fixed'],config,job['output'],method,'cuda:0')


def _serve(connection, gpu, handler):
    # Spawned interpreter: bind the device before importing/initializing torch.
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MKL_NUM_THREADS','4')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    import numpy as np
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    try:
        while True:
            job=connection.recv()
            if job is None: break
            status={'returncode':0,'worker_pid':os.getpid()}
            healthy=True
            with Path(job['log']).open('x') as log, redirect_stdout(log), redirect_stderr(log):
                try:
                    config=json.loads(Path(job['config']).read_text())
                    seed=config['runtime']['seed']
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    handler(job)
                except Exception:
                    status['returncode']=1
                    traceback.print_exc()
                finally:
                    # Method stack (including optimizer/tensors) has unwound.
                    gc.collect()
                    try:
                        if torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            status['allocated_bytes']=torch.cuda.memory_allocated()
                    except Exception:
                        traceback.print_exc();status['returncode']=1;healthy=False
                        status['restart_worker']=True
            connection.send(status)
            if not healthy: break
    except EOFError:
        pass
    finally:
        connection.close()


class GPUWorker:
    """Lazy worker; construction/plan-only launchers do no CUDA work."""
    def __init__(self,gpu,*,handler=_run_case):
        self.gpu,self.handler=gpu,handler
        self.process=self.connection=None

    def __enter__(self): return self

    def run(self,command,log):
        if self.process is None or not self.process.is_alive():
            self.close()
            ctx=multiprocessing.get_context('spawn')
            self.connection,child=ctx.Pipe()
            self.process=ctx.Process(target=_serve,args=(child,self.gpu,self.handler))
            self.process.start();child.close()
        def argument(name,default=None):
            return command[command.index(name)+1] if name in command else default
        job={name:argument('--'+name) for name in ('moving','fixed','config','output')}
        job.update(method=argument('--method','medical.oasis_method'),log=str(log))
        try:
            self.connection.send(job)
            status=self.connection.recv()
            if status.get('restart_worker'): self.close()
            return status
        except (EOFError,BrokenPipeError,ConnectionResetError):
            self.close()
            return {'returncode':1,'error':'GPU worker exited; next case will start a fresh worker'}

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try: self.connection.send(None)
                except (BrokenPipeError,EOFError,OSError): pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate();self.process.join(timeout=10)
            self.connection.close()
            self.process=self.connection=None

    def __exit__(self,*args): self.close()
