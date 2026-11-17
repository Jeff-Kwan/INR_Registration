"""Canonical paired-SIREN fitting of prepared ACDC and Neurite-OASIS cohorts."""
import json
from pathlib import Path
import tempfile
import time

import torch
from medical.brain_data import sha256
from medical.cohort_data import validate_case
from medical.cohort_evaluation import evaluate
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover


def run_case(case, config, device, *, checkpoint_callback=None, standardized=False):
    root = Path(case['fixed']['path']).resolve().parent.parent
    protocol = json.loads((root / 'manifest.json').read_text())['protocol']
    dataset = {'acdc_kan_idir_v1': 'acdc',
               'neurite_oasis_gaussiandir_last50_consecutive49_v1': 'neurite_oasis'}[protocol]
    root, membership = validate_case(case, dataset)
    validate_config(config)
    started = time.perf_counter()
    arrays = [torch.from_numpy(case[k]['data'].transpose(2, 1, 0).copy()).to(device) for k in ('moving', 'fixed')]
    affine = case['moving']['affine'].copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    timings = {'preprocessing_seconds': time.perf_counter() - started}
    started = time.perf_counter()
    pair = fit_pair(*arrays, config, affine=affine,
                    report=lambda row: print(json.dumps({'training': row}), flush=True),
                    checkpoint_callback=checkpoint_callback)
    timings['training_seconds'] = time.perf_counter() - started
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'{dataset}-recovery-') as scratch:
        recovered = recover(pair, arrays[0].shape, config['recovery'], scratch,
                            chunk=config['runtime']['inference_chunk'],
                            compile_derivatives=config['runtime']['compile_derivatives'],
                            max_axis=config['runtime']['recovery_max_axis'])
        if recovered.inverse.is_cuda:
            torch.cuda.synchronize(recovered.inverse.device)
        timings['recovery_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        if standardized:
            from medical.comparison_metrics import evaluate as standardized_evaluate
            scores = standardized_evaluate(case, recovered.inverse.detach().permute(2, 1, 0, 3), dataset)
        else:
            scores = evaluate(case, recovered.inverse.detach().permute(2, 1, 0, 3), dataset, root, membership)
        timings['evaluation_seconds'] = time.perf_counter() - started
        scores.update(recovery=recovered.diagnostics, seed=config['runtime']['seed'], pair_index=membership['pair_index'],
                      intensity_normalization={'method': 'preserve_protocol_prepared_values'},
                      optimization_roi='shared crop; no anatomical-label sampling', timings=timings)
    scores['training_execution_timing'] = getattr(pair, 'training_metadata', {}).get('execution_timings')
    return {'metrics': scores, 'checkpoint': {'format_version': 1, 'model': pair.model_config,
            'training': getattr(pair, 'training_metadata', {}), 'state': {k: v.detach().cpu() for k, v in pair.state_dict().items()},
            'config': config, 'cohort': dataset, 'pair': membership,
            'preparation_manifest_sha256': sha256(root / 'manifest.json')}}
