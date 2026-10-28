"""Content-addressed oracle/cross-fade assets and their A/B/C recovery results."""
import hashlib
import json
from pathlib import Path
import torch
from cameraman.core.synthetic import oracle_frame
from cameraman.core.interpolate import crossfade
from cameraman.core.evaluate import evaluate_recovery, RECOVERY_CASES, foreground_dice_metrics


@torch.no_grad()
def evaluate_with_cached_references(inr_sequence, source, target, grid, displacement, mask,
                                    cache_dir, geometry=None, inr_models=None, intervals=8, inference_chunk=0, **flow_options):
    """Reuse references only when images, geometry, sampling and solver agree."""
    cache_dir = Path(cache_dir)
    frames = len(inr_sequence)
    metadata = {"version": 2, "frames": frames, "flow": flow_options,
                "torch": str(torch.__version__), "device": str(source.device),
                "sampling": "bilinear-border-align_corners_true", "integration": "midpoint-rk2"}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for value in (source, target, grid, displacement, mask):
        value = value.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    if geometry is not None and geometry.get("kind") == "elastic":
        digest.update(f"elastic-svf-v1-steps={geometry['integration_steps']}".encode())
        digest.update(geometry["velocity"].detach().cpu().contiguous().numpy().tobytes())
    path = cache_dir / (digest.hexdigest() + ".pt")
    if path.exists():
        print(f"Loading cached oracle/cross-fade: {path.name[:12]}", flush=True)
        assets = torch.load(path, map_location="cpu", weights_only=True)
        updated = False
        for label, result in assets["experiments"].items():
            if "foreground_dice" not in result:
                inverse = assets["recovery"][label]["inverse_map"].to(source.device)
                result.update(foreground_dice_metrics(source, target, inverse, mask))
                updated = True
            if "epe_pixels_full" not in result:
                forward = assets["recovery"][label]["forward_map"].to(source.device)
                result["epe_pixels_full"] = (forward-grid-displacement).norm(dim=-1).mean().item()
                updated = True
        if updated:
            torch.save(assets, path)
    else:
        print("Building shared oracle/cross-fade cache", flush=True)
        times = torch.linspace(0, 1, frames, device=source.device)
        references = {"Oracle": torch.stack([oracle_frame(source, grid, displacement, t, geometry) for t in times]),
                      "Cross-fade": crossfade(source, target, frames)}
        experiments, maps, tensors = evaluate_recovery(references, grid, displacement, source, target, mask,
                                                       cases=RECOVERY_CASES[:3], **flow_options)
        assets = {"metadata": metadata, "sequences": {k: v.cpu() for k, v in references.items()},
                  "experiments": experiments, "maps": {k: v.cpu() for k, v in maps.items()}, "recovery": tensors}
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(assets, temporary)
        temporary.replace(path)
    sequences = {"Oracle": assets["sequences"]["Oracle"].to(source.device), "INR": inr_sequence,
                 "Cross-fade": assets["sequences"]["Cross-fade"].to(source.device)}
    derivatives = None
    if inr_models is not None:
        from cameraman.core.analytical_recovery import midpoint_derivatives
        derivatives = midpoint_derivatives(*inr_models, grid, intervals, inference_chunk)
    experiments, maps, tensors = evaluate_recovery(sequences, grid, displacement, source, target, mask,
                                                   cases=RECOVERY_CASES[3:], inr_derivatives=derivatives, **flow_options)
    return (sequences, {**assets["experiments"], **experiments},
            {**{k: v.to(source.device) for k, v in assets["maps"].items()}, **maps},
            {**assets["recovery"], **tensors})
