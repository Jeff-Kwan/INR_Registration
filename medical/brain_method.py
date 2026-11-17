"""Our default paired-SIREN method on protocol-prepared brain benchmarks."""
import json
import tempfile
import time

import torch

from medical.brain_data import validate_case, require_sitreg_dependencies
from medical.brain_evaluation import evaluate
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover


def run_case(case, config, device, *, checkpoint_callback=None, standardized=False):
    # Dataset identity is in the validated preparation record, not a CLI guess.
    from pathlib import Path
    root = Path(case['fixed']['path']).parent.parent
    dataset = json.loads((root / 'protocol.json').read_text())['dataset']
    membership = validate_case(case, dataset)
    if dataset == 'lpba40':
        require_sitreg_dependencies()
    validate_config(config)
    cap = config['runtime'].get('recovery_max_axis')
    if cap < max(case['fixed']['data'].shape):
        raise ValueError('Brain benchmark adapter requires full native-grid voxel recovery')
    started = time.perf_counter()
    # Preserve IXI release / SITReg WM-normalized values; labels do not enter fitting.
    arrays = [torch.from_numpy(case[key]['data'].transpose(2, 1, 0).copy()).to(device)
              for key in ('moving', 'fixed')]
    affine = case['moving']['affine'].copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    preprocessing_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pair = fit_pair(*arrays, config, affine=affine,
                    report=lambda row: print(json.dumps({'training': row}), flush=True),
                    checkpoint_callback=checkpoint_callback)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'{dataset}-recovery-') as scratch:
        recovered = recover(pair, arrays[0].shape, config['recovery'], scratch,
                            chunk=config['runtime']['inference_chunk'],
                            compile_derivatives=config['runtime']['compile_derivatives'],
                            max_axis=cap)
        if recovered.inverse.is_cuda:
            torch.cuda.synchronize(recovered.inverse.device)
        recovery_seconds = time.perf_counter() - started
        evaluation_started = time.perf_counter()
        if standardized:
            from medical.comparison_metrics import evaluate as standardized_evaluate
            scores = standardized_evaluate(case, recovered.inverse.detach().permute(2, 1, 0, 3), dataset)
        else:
            scores = evaluate(case, recovered.inverse.detach().permute(2, 1, 0, 3), dataset,
                              pair_index=membership['pair_index'],
                              inverse_prediction=recovered.forward.detach().permute(2, 1, 0, 3))
        scores.update(recovery=recovered.diagnostics, seed=config['runtime']['seed'],
                      split=membership['split'], optimization_roi='shared crop; no anatomical-label sampling',
                      intensity_normalization={'method': 'preserve_protocol_prepared_values',
                                               'preparation': membership['preparation']['intensity']},
                      timings={'preprocessing_seconds': preprocessing_seconds,
                               'training_seconds': training_seconds, 'recovery_seconds': recovery_seconds,
                               'evaluation_seconds': time.perf_counter() - evaluation_started})
    scores['training_execution_timing'] = getattr(pair, 'training_metadata', {}).get('execution_timings')
    checkpoint = {'format_version': 1, 'model': pair.model_config,
                  'training': getattr(pair, 'training_metadata', {}),
                  'state': {k: v.detach().cpu() for k, v in pair.state_dict().items()},
                  'config': config, 'brain_benchmark': membership}
    return {'metrics': scores, 'checkpoint': checkpoint}
