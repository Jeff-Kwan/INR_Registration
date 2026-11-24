"""SIRENPath on official Learn2Reg OASIS pairs."""
import tempfile
import json
import time
import torch
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover
from medical.learn2reg_evaluation import validate_case, evaluate
from medical.models.siren import endpoint_reconstruction


def run_case(case, config, device, *, checkpoint_callback=None):
    validate_config(config)
    validate_case(case)
    started = time.perf_counter()
    arrays, normalization = [], {}
    for key in ('moving', 'fixed'):
        data = case[key]['data']
        lo, hi = float(data.min()), float(data.max())
        if hi <= lo:
            raise ValueError('Cannot normalize constant input')
        arrays.append(torch.from_numpy(((data-lo)/(hi-lo)).transpose(2, 1, 0).copy()).to(device))
        normalization[key] = {'min': lo, 'max': hi}
    affine = case['moving']['affine'].copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    preprocessing_seconds = time.perf_counter()-started
    started = time.perf_counter()
    pair = fit_pair(*arrays, config, affine=affine,
                    report=lambda row: print(json.dumps({'training': row}), flush=True),
                    checkpoint_callback=checkpoint_callback)
    training_seconds = time.perf_counter()-started
    started = time.perf_counter()
    reconstruction = endpoint_reconstruction(pair, arrays, config['runtime']['inference_chunk'])
    reconstruction_seconds = time.perf_counter()-started
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='oasis-recovery-') as scratch:
        recovered = recover(pair, arrays[0].shape, config['recovery'], scratch,
                            chunk=config['runtime']['inference_chunk'],
                            compile_derivatives=config['runtime']['compile_derivatives'],
                            max_axis=config['runtime']['recovery_max_axis'])
        if recovered.inverse.is_cuda:
            torch.cuda.synchronize(recovered.inverse.device)
        recovery_seconds = time.perf_counter()-started
        evaluation_started = time.perf_counter()
        metrics = evaluate(case, recovered.inverse.detach().permute(2, 1, 0, 3))
        metrics.update(recovery=recovered.diagnostics,
                       intensity_normalization={'method': 'independent_minmax_0_1', **normalization},
                       optimization_roi='shared crop; no anatomical-label sampling', seed=config['runtime']['seed'],
                       reconstruction=reconstruction,
                       timings={'preprocessing_seconds': preprocessing_seconds, 'training_seconds': training_seconds,
                                'reconstruction_seconds': reconstruction_seconds, 'recovery_seconds': recovery_seconds,
                                'evaluation_seconds': time.perf_counter()-evaluation_started,
                                'recovery_evaluation_seconds': time.perf_counter()-started})
    metrics['training_execution_timing'] = getattr(pair, 'training_metadata', {}).get('execution_timings')
    return {'metrics': metrics, 'checkpoint': {'format_version': 1, 'model': pair.model_config,
            'training': getattr(pair, 'training_metadata', {}),
            'state': {k: v.detach().cpu() for k, v in pair.state_dict().items()}, 'config': config}}
