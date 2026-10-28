"""Run paired-SIREN pointwise PD-LK on a synthetic Cameraman deformation."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch

from cameraman.paths import ROOT
from cameraman.core.synthetic import make_pair
from cameraman.core.inr import (Siren, fit_pair, prepare_evaluation,
                                initialize_endpoint_pair, validate_training_settings)
from cameraman.core.interpolate import interpolate_weights
from cameraman.core.evaluate import psnr, save_comparison, save_animation, ground_truth_metrics
from cameraman.core.reference_cache import evaluate_with_cached_references

REPO_ROOT = ROOT.parent


def _resolve_config_path(value):
    """Config paths are repository-relative; explicit CLI paths are cwd-relative."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    default_config = ROOT / 'configs/pdlk_relaxed.json'
    p.add_argument('--config', type=Path, default=default_config, help='JSON defaults; CLI flags override')
    p.add_argument('--image', type=Path, default=ROOT / 'data/cameraman.jpg')
    p.add_argument('--output', type=Path, help='New or empty run directory')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0')
    integers = dict(size=256, steps=5000, batch_size=8192, width=128, layers=3,
                    deformation_seed=42, elastic_steps=7, frames=33, intervals=8,
                    inference_chunk=0, log_every=10, flow_iterations=500, substeps=4,
                    seed=42, lk_points=1960)
    for name, value in integers.items():
        p.add_argument('--' + name.replace('_', '-'), type=int, default=value)
    numbers = dict(amplitude=10., elastic_sigma=12., elastic_min_jacobian=.2,
                   min_transform_epe=0., omega=30., lr=1e-4, lr_final_factor=1.,
                   lk_weight=.1, lk_ridge=1e-6, lk_gradient_weight=4.,
                   lk_affine_rho=.4, grad_clip=0., min_psnr=25., smoothness=.01,
                   ridge=1e-6, flow_tolerance=2e-5)
    for name, value in numbers.items():
        help_text = 'rho=c/beta; 0 is exact standard PD-LK' if name == 'lk_affine_rho' else None
        p.add_argument('--' + name.replace('_', '-'), type=float, default=value, help=help_text)
    choices = dict(deformation=('shear', 'elastic'), elastic_boundary=('taper', 'normal'),
                   recovery_method=('analytical', 'finite'), lr_schedule=('exponential', 'linear'),
                   lk_weight_schedule=('constant', 'linear', 'cosine'),
                   endpoint_initialization=('shared', 'independent'))
    for name, values in choices.items():
        p.add_argument('--' + name.replace('_', '-'), choices=values, default=values[0])
    p.add_argument('--target-init-seed', type=int)
    p.add_argument('--lk-differentiate-solve', action=argparse.BooleanOptionalAction, default=False,
                   help='Differentiate the local solve; default uses the envelope gradient')

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=default_config)
    selected, _ = config_parser.parse_known_args(argv)
    selected_config = selected.config
    if not selected_config.is_absolute() and not selected_config.exists():
        selected_config = REPO_ROOT / selected_config
    try:
        defaults = json.loads(selected_config.read_text())
    except (OSError, ValueError) as error:
        p.error(f'Cannot read config {selected_config}: {error}')
    allowed = {action.dest for action in p._actions} - {'help', 'config'}
    if not isinstance(defaults, dict):
        p.error('Config must be a JSON object')
    if defaults.keys() - allowed:
        p.error(f'Unknown config keys: {sorted(defaults.keys() - allowed)}')
    for name in ('image', 'output'):
        if name in defaults and defaults[name] is not None:
            defaults[name] = _resolve_config_path(defaults[name])
    p.set_defaults(**defaults)
    args = p.parse_args(argv)
    args.config = selected_config.resolve()
    for name, values in choices.items():
        if getattr(args, name) not in values:
            p.error(f'{name} must be one of {values}')
    for name in integers:
        value = getattr(args, name)
        minimum = 0 if name in ('inference_chunk', 'seed', 'deformation_seed') else 1
        if type(value) is not int or value < minimum:
            p.error(f'{name} must be an integer >= {minimum}')
    for name in numbers:
        value = getattr(args, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            p.error(f'{name} must be a finite number')
    try:
        validate_training_settings(vars(args))
    except ValueError as error:
        p.error(str(error))
    if args.size < 8 or args.frames < 2:
        p.error('size must be >=8 and frames >=2')
    if any(getattr(args, key) <= 0 for key in ('omega', 'elastic_sigma', 'smoothness', 'ridge', 'flow_tolerance')):
        p.error('omega, elastic_sigma, smoothness, ridge and flow_tolerance must be positive')
    if not 0 < args.elastic_min_jacobian < 1 or args.min_transform_epe < 0:
        p.error('Require 0<elastic_min_jacobian<1 and min_transform_epe>=0')
    if args.deformation == 'elastic' and args.amplitude < 0:
        p.error('Elastic amplitude must be nonnegative')
    if args.target_init_seed is None:
        args.target_init_seed = args.seed + 1
    if type(args.target_init_seed) is not int or args.target_init_seed < 0:
        p.error('target_init_seed must be a nonnegative integer')
    if args.output is None:
        args.output = ROOT / 'runs/pdlk'
    return args


def prepare_output_directory(output):
    """Refuse to overwrite an existing experiment, including partial runs."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu or a CUDA-enabled environment')
    prepare_output_directory(args.output)
    print(f'Device: {device}', flush=True)
    source, target, grid, displacement, _, mask, image_path, margin, geometry = make_pair(
        args.image, args.size, args.amplitude, args.frames, device, generate_oracle=False,
        deformation=args.deformation, elastic_sigma=args.elastic_sigma, deformation_seed=args.deformation_seed,
        elastic_steps=args.elastic_steps, elastic_min_jacobian=args.elastic_min_jacobian, return_geometry=True,
        elastic_boundary=args.elastic_boundary, min_transform_epe=args.min_transform_epe)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), image=str(image_path), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda, inr='siren', architecture='siren',
                  evaluation_margin_pixels=margin, deformation_diagnostics=geometry.get('diagnostics', {}),
                  coordinates='native pixel (x,y); INR input=2*pixel/(size-1)-1',
                  sampling='bilinear; border padding; align_corners=True',
                  integration='piecewise constant Eulerian velocity; midpoint RK2',
                  training_objective='MSE_source + MSE_target + lk_weight * minimized_pointwise_PD_LK',
                  parameterization='c=lk_gradient_weight; eta=lk_ridge; rho=lk_affine_rho; rho=0 is standard PD-LK',
                  lk_sampling='one shared uniform time; independent full-domain normalized xy queries',
                  lk_derivatives='eager analytical sine chain rule, rescaled to native pixels',
                  optimizer='Adam; zero weight decay; fused on CUDA',
                  precision='float32; CUDA training may not be bitwise deterministic')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    coordinates = (2 * grid / (args.size - 1) - 1).reshape(-1, 2)
    factory = lambda: Siren(args.width, args.layers, args.omega)
    initial, source_model, target_model = initialize_endpoint_pair(
        factory, device, args.endpoint_initialization, args.target_init_seed)
    initial_state = {key: value.detach().cpu().clone() for key, value in source_model.state_dict().items()}
    target_initial_state = {key: value.detach().cpu().clone() for key, value in target_model.state_dict().items()}
    print(f'Jointly fitting paired SIRENs with PD-LK (rho={args.lk_affine_rho:g})', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    histories = {'joint': fit_pair(
        source_model, target_model, coordinates, source, target,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, log_every=args.log_every,
        lr_final_factor=args.lr_final_factor, lr_schedule=args.lr_schedule,
        lk_weight=args.lk_weight, lk_points=args.lk_points, lk_ridge=args.lk_ridge,
        lk_gradient_weight=args.lk_gradient_weight, lk_affine_rho=args.lk_affine_rho,
        lk_differentiate_solve=args.lk_differentiate_solve,
        lk_weight_schedule=args.lk_weight_schedule, grad_clip=args.grad_clip)}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    print(f"Training time: {training_seconds:.3f} s", flush=True)
    prepare_evaluation(initial, source_model, target_model)
    inr_sequence = interpolate_weights(initial, source_model.state_dict(), target_model.state_dict(), coordinates, args.size, args.frames)
    sequences, experiments, maps, tensors = evaluate_with_cached_references(
        inr_sequence, source, target, grid, displacement, mask, (ROOT / "runs") / "_reference_cache",
        smoothness=args.smoothness, iterations=args.flow_iterations, tolerance=args.flow_tolerance,
        ridge=args.ridge, substeps=args.substeps, geometry=geometry,
        inr_models=(source_model, target_model) if args.recovery_method == "analytical" else None,
        intervals=args.intervals, inference_chunk=args.inference_chunk)
    reconstructions = {"source": sequences["INR"][0], "target": sequences["INR"][-1]}
    scores = {name: psnr(reconstructions[name], image) for name, image in (("source", source), ("target", target))}
    reliable = min(scores.values()) >= args.min_psnr
    if not reliable:
        warnings.warn(f"Endpoint PSNR below {args.min_psnr} dB: {scores}. Increase --steps before interpreting INR results.")
    results = {"training_seconds": training_seconds, "endpoint_psnr_db": scores, "endpoint_quality_pass": reliable, "experiments": experiments,
               "ground_truth": ground_truth_metrics(source, target, grid, displacement, mask, geometry)}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    (args.output / "training_history.json").write_text(json.dumps(histories, indent=2))
    torch.save({"initial": initial_state, "initial_source": initial_state, "initial_target": target_initial_state,
                "source": {k: v.cpu() for k, v in source_model.state_dict().items()},
                "target": {k: v.cpu() for k, v in target_model.state_dict().items()}, "config": config}, args.output / "checkpoints.pt")
    torch.save({"source": source.cpu(), "target": target.cpu(), "grid": grid.cpu(),
                "ground_truth_displacement": displacement.cpu(), "mask": mask.cpu(),
                "reconstructions": {k: v.cpu() for k, v in reconstructions.items()},
                "sequences": {k: v.cpu() for k, v in sequences.items()}, "recovery": tensors,
                "deformation": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in geometry.items()}}, args.output / "results.pt")
    save_comparison(args.output / "comparison.png", source, target, grid, displacement, sequences, maps)
    save_animation(args.output / "trajectories.gif", sequences)
    print(f"Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
