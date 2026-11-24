"""SIRENPath L2R-OASIS ablations, with shared frozen recovery and scoring."""
import hashlib
from pathlib import Path
import json
import tempfile
import time
import numpy as np
import torch
from medical.configuration import validate_config
from medical.abdomen_evaluation import validate_case, evaluate, _interior_jacobian
from medical.registration.training import fit_pair
from medical.registration.recovery import recover
from medical.registration.intensity_crossfade import IntensityCrossfade
from medical.sirenpath_reconstruction import endpoint_scores, reconstruct_endpoint
from medical.models.siren import PairedSiren


def run_case(case, config, device, *, checkpoint_callback=None, raw=False, reference_checkpoint=None):
    if raw and reference_checkpoint is not None:
        raise ValueError('Choose raw images or trained reconstructions, not both')
    validate_config(config, allow_ablation_mse=True)
    validate_case(case, 'oasis')
    started = time.perf_counter()
    arrays, normalization = [], {}
    for key in ('moving', 'fixed'):
        data = case[key]['data']
        lo, hi = float(data.min()), float(data.max())
        if hi <= lo:
            raise ValueError('Cannot normalize constant input')
        arrays.append(torch.from_numpy(((data-lo)/(hi-lo)).transpose(2,1,0).copy()).to(device))
        normalization[key] = dict(min=lo, max=hi)
    affine = case['moving']['affine'].copy()
    affine[:3,:3] = affine[:3,:3][:,::-1]
    def synchronize():
        if arrays[0].is_cuda:
            torch.cuda.synchronize(arrays[0].device)
    synchronize()
    timings = dict(preprocessing_seconds=time.perf_counter()-started)
    stamp = time.perf_counter()
    predictions, reference = None, None
    if reference_checkpoint is not None:
        trained, reference = load_canonical_checkpoint(reference_checkpoint, case, config, device)
        synchronize()
        timings['checkpoint_load_seconds'] = time.perf_counter()-stamp
        stamp = time.perf_counter()
        predictions = tuple(reconstruct_endpoint(model, image, config['runtime']['inference_chunk'])
                            for model, image in zip((trained.source, trained.target), arrays))
        synchronize()
        timings['endpoint_decode_seconds'] = time.perf_counter()-stamp
        del trained
        stamp = time.perf_counter()
        pair = IntensityCrossfade(*predictions)
        synchronize()
        timings['finite_difference_setup_seconds'] = time.perf_counter()-stamp
        timings['training_seconds'] = 0.
    elif raw:
        pair = IntensityCrossfade(*arrays)
        timings['training_seconds'] = 0.
        synchronize()
        timings['finite_difference_setup_seconds'] = time.perf_counter()-stamp
    else:
        pair = fit_pair(*arrays, config, affine=affine, allow_ablation_mse=True,
                        report=lambda row: print(json.dumps({'training':row}), flush=True),
                        checkpoint_callback=checkpoint_callback)
        synchronize()
        timings['training_seconds'] = time.perf_counter()-stamp
    pair.eval().requires_grad_(False)
    stamp = time.perf_counter()
    reconstruction = endpoint_scores(pair, arrays, chunk=config['runtime']['inference_chunk'], raw=raw, predictions=predictions)
    synchronize()
    timings['reconstruction_seconds'] = time.perf_counter()-stamp
    stamp = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='sirenpath-oasis-recovery-') as scratch:
        recovered = recover(pair, arrays[0].shape, config['recovery'], scratch,
                            chunk=config['runtime']['inference_chunk'],
                            compile_derivatives=config['runtime']['compile_derivatives'],
                            max_axis=config['runtime']['recovery_max_axis'])
        synchronize()
        timings['recovery_seconds'] = time.perf_counter()-stamp
        stamp = time.perf_counter()
        mapping = recovered.inverse.permute(2,1,0,3)
        metrics = evaluate(case, mapping, 'oasis')
        jac = _interior_jacobian(mapping)
        if isinstance(jac, torch.Tensor):
            unshifted = float(jac.clamp_min(1e-9).log().std(correction=0))
        else:
            unshifted = float(np.log(np.maximum(jac,1e-9)).std())
        metrics.update(PSNR=reconstruction['PSNR'], PatchNCC=reconstruction['PatchNCC'],
                       NPJ=dict(mean=100*metrics['folding_fraction'], units='percent'),
                       SDlogJ=dict(mean=unshifted, definition='std(log(max(detJ,1e-9)))'),
                       official_L2R_SDlogJ=metrics['LogJacDetStd'], reconstruction=reconstruction,
                       method='Fitted crossfade' if reference is not None else 'Raw crossfade' if raw else 'SIRENPath',
                       control='reconstructed_crossfade' if reference is not None else 'raw_crossfade' if raw else 'trained_weight_path',
                       seed=config['runtime']['seed'], recovery=recovered.diagnostics,
                       intensity_normalization=dict(method='independent_minmax_0_1', **normalization))
        synchronize()
        timings['evaluation_seconds'] = time.perf_counter()-stamp
    timings['total_method_seconds'] = time.perf_counter()-started
    metrics['timings'] = timings
    metrics['Runtime'] = dict(mean=timings['total_method_seconds'], units='seconds',
                             scope='input normalization, fitting or finite differences, reconstruction, recovery, scoring')
    if reference is not None:
        metrics['reference_model'] = reference
        metrics['reused_training_seconds'] = reference['training_seconds']
        metrics['Runtime_with_shared_training'] = dict(
            mean=timings['total_method_seconds']+reference['training_seconds'], units='seconds',
            scope='this method plus the reused canonical fit; worker load/save overhead excluded')
    if raw or reference is not None:
        # Never serialize endpoint images, gradient volumes or deformation maps.
        checkpoint = dict(kind='provenance_only',
                          control='reconstructed_crossfade' if reference is not None else 'raw_crossfade',
                          reference_model=reference, config=config,
                          spatial_derivatives='central native voxel differences; first-order one-sided boundaries',
                          temporal_derivative='reconstructed_fixed-minus-reconstructed_moving' if reference is not None else 'fixed-minus-moving',
                          training_steps=0)
    else:
        checkpoint = dict(format_version=1, model=pair.model_config, config=config,
                          training=getattr(pair, 'training_metadata', {}),
                          state={k:v.detach().cpu() for k,v in pair.state_dict().items()})
    return dict(metrics=metrics, checkpoint=checkpoint)


def load_canonical_checkpoint(path, case, expected_config, device):
    """Validate the exact paired run-5 fit before reusing its frozen endpoints."""
    path = Path(path)
    evidence = json.loads((path.parent/'run.json').read_text())
    for key in ('moving','fixed'):
        if Path(evidence[key]).resolve() != Path(case[key]['path']).resolve():
            raise ValueError('Canonical checkpoint belongs to a different image pair')
    if (list(case['moving']['data'].shape) != evidence['shape']
            or not np.allclose(evidence['affine'], case['moving']['affine'])):
        raise ValueError('Canonical checkpoint geometry differs from this case')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('working_grid_metadata','physical_units_metadata'):
        if checkpoint.get(key) is not None:
            raise ValueError('Crossfade control supports native-voxel checkpoints only')
    if checkpoint.get('training',{}).get('spatial_units','voxel') != 'voxel':
        raise ValueError('Crossfade control supports native-voxel checkpoints only')
    validate_config(checkpoint['config'])
    if checkpoint['config'] != expected_config or evidence['config'] != expected_config:
        raise ValueError('Fitted crossfade must reuse the identical canonical run-5 recipe')
    if checkpoint['model'] != expected_config['model']:
        raise ValueError('Canonical checkpoint architecture differs from the configured model')
    pair = PairedSiren(3, **checkpoint['model']).to(device)
    pair.load_state_dict(checkpoint['state'], strict=True)
    pair.eval().requires_grad_(False)
    metrics = json.loads((path.parent/'metrics.json').read_text())
    reference = dict(checkpoint=str(path.resolve()), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     run_sha256=hashlib.sha256((path.parent/'run.json').read_bytes()).hexdigest(),
                     training_seconds=metrics['timings']['training_seconds'],
                     reconstruction_transform='none: preserve signed values and fitted NCC scale/offset')
    return pair, reference
