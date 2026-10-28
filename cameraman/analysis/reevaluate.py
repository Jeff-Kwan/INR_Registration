"""Overwrite saved evaluation outputs from checkpoints, without retraining."""
from cameraman.paths import ROOT
import argparse
import copy
import json
from pathlib import Path
import tempfile

import torch

from cameraman.core.inr import Siren
from cameraman.core.interpolate import interpolate_weights
from cameraman.core.evaluate import psnr, save_comparison, save_animation, ground_truth_metrics
from cameraman.core.reference_cache import evaluate_with_cached_references


def load_saved(path):
    # Early runs saved torch.__version__ as TorchVersion rather than a plain str.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def is_archived(run):
    return any((parent / '.archived').is_file() for parent in (run, *run.parents))


def discover_runs(root):
    return sorted({path.parent for path in root.rglob('checkpoints.pt')
                   if not is_archived(path.parent)
                   and not any(part.startswith('.reevaluate-') for part in path.parts)})


@torch.no_grad()
def reevaluate(run, frames, device, recovery_method="analytical", intervals=8, inference_chunk=0):
    if is_archived(run):
        raise ValueError(f'Archived experiment has no active model implementation: {run}')
    config = json.loads((run / "config.json").read_text())
    if config.get("inr", "siren") != "siren" or config.get("architecture", "siren") != "siren":
        raise ValueError("Saved-model evaluation supports paired SIRENs only")
    if recovery_method not in ("analytical", "finite"):
        raise ValueError("Recovery must be analytical or finite")
    if frames < 2 or intervals < 1 or inference_chunk < 0:
        raise ValueError("Require frames >=2, intervals >=1 and inference_chunk >=0")
    checkpoint = load_saved(run / "checkpoints.pt")
    old = load_saved(run / "results.pt")
    size = config["size"]
    model = Siren(config["width"], config["layers"], config.get("omega", 30.))
    model = model.to(device)
    # Strict loading checks architecture before any files are overwritten.
    model.load_state_dict(checkpoint["source"])
    model.load_state_dict(checkpoint["target"])
    model.eval()
    source, target, grid, displacement, mask = [old[key].to(device) for key in
        ("source", "target", "grid", "ground_truth_displacement", "mask")]
    coordinates = (2 * grid / (size-1) - 1).reshape(-1, 2)
    source_state = {key: value.to(device) for key, value in checkpoint["source"].items()}
    target_state = {key: value.to(device) for key, value in checkpoint["target"].items()}
    source_model, target_model = copy.deepcopy(model), copy.deepcopy(model)
    source_model.load_state_dict(source_state)
    target_model.load_state_dict(target_state)
    inr_sequence = interpolate_weights(model, source_state, target_state, coordinates, size, frames)
    geometry = old.get("deformation", {"kind": "shear"})
    if config.get("deformation") == "elastic" and geometry.get("kind") != "elastic":
        raise ValueError("Saved elastic geometry is missing; cannot regenerate its oracle")
    geometry = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in geometry.items()}
    sequences, experiments, maps, tensors = evaluate_with_cached_references(
        inr_sequence, source, target, grid, displacement, mask, (ROOT / "runs") / "_reference_cache",
        smoothness=config.get("smoothness", .01), iterations=config.get("flow_iterations", 500),
        tolerance=config.get("flow_tolerance", 1e-5), ridge=config.get("ridge", 1e-6), substeps=config.get("substeps", 4), geometry=geometry,
        inr_models=(source_model, target_model) if recovery_method == "analytical" else None,
        intervals=intervals, inference_chunk=inference_chunk)
    reconstructions = {"source": sequences["INR"][0], "target": sequences["INR"][-1]}
    scores = {name: psnr(reconstructions[name], image) for name, image in (("source", source), ("target", target))}
    metrics = {"endpoint_psnr_db": scores, "endpoint_quality_pass": min(scores.values()) >= config.get("min_psnr", 30.),
               "experiments": experiments,
               "ground_truth": ground_truth_metrics(source, target, grid, displacement, mask, geometry)}
    old.update(sequences={key: value.cpu() for key, value in sequences.items()}, recovery=tensors,
               reconstructions={key: value.cpu() for key, value in reconstructions.items()})
    settings = dict(frames=frames, recovery_method=recovery_method, intervals=intervals,
                    inference_chunk=inference_chunk)
    config.update(settings)
    checkpoint["config"].update(settings)
    if "torch_version" in checkpoint["config"]:
        checkpoint["config"]["torch_version"] = str(checkpoint["config"]["torch_version"])
    # Prepare all outputs before replacing the existing evaluation files.
    with tempfile.TemporaryDirectory(dir=run, prefix=".reevaluate-") as folder:
        staging = Path(folder)
        (staging / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        (staging / "config.json").write_text(json.dumps(config, indent=2))
        torch.save(checkpoint, staging / "checkpoints.pt")
        torch.save(old, staging / "results.pt")
        save_comparison(staging / "comparison.png", source, target, grid, displacement, sequences, maps)
        save_animation(staging / "trajectories.gif", sequences)
        for path in staging.iterdir():
            path.replace(run / path.name)
    return {"run": str(run), "frames": frames, "inr_epe_pixels": experiments["D_inr_2d"]["epe_pixels"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Run folders; omit to discover checkpoints under --root")
    parser.add_argument("--root", type=Path, default=(ROOT / 'runs'))
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recovery-method", choices=("analytical", "finite"), default="analytical")
    parser.add_argument("--intervals", type=int, default=8)
    parser.add_argument("--inference-chunk", type=int, default=0)
    args = parser.parse_args()
    if args.intervals < 1 or args.inference_chunk < 0:
        parser.error("--intervals must be positive and --inference-chunk nonnegative")
    if args.frames < 2:
        parser.error("--frames must be >=2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    runs = args.runs or discover_runs(args.root)
    if not runs:
        parser.error("No saved checkpoints found")
    results, failures = [], []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] Re-evaluating {run} with {args.frames} frames on {device}", flush=True)
        try:
            result = reevaluate(run, args.frames, device, args.recovery_method, args.intervals, args.inference_chunk)
            results.append(result)
            print(f"Saved {run}: INR EPE={result['inr_epe_pixels']:.4f} px", flush=True)
        except Exception as error:
            failures.append({"run": str(run), "error": str(error)})
            print(f"FAILED {run}: {error}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "reevaluation_summary.json").write_text(json.dumps({"completed": results, "failed": failures}, indent=2))
    print(f"Completed {len(results)}/{len(runs)} runs", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
