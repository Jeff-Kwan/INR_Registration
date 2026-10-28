"""Metrics use pixel-valued forward maps and a fixed interior mask."""
import math
import numpy as np
import torch
from cameraman.core.synthetic import sample
from cameraman.core.flow import estimate_velocity, integrate


RECOVERY_CASES = (
    ("A_oracle_horizontal", "Oracle", True),
    ("B_oracle_2d", "Oracle", False),
    ("C_crossfade_2d", "Cross-fade", False),
    ("D_inr_2d", "INR", False),
    ("D_inr_horizontal", "INR", True),
)


@torch.no_grad()
def evaluate_recovery(sequences, grid, displacement, source, target, mask, *,
                      smoothness=.01, iterations=500, tolerance=1e-5, ridge=1e-6, substeps=4,
                      cases=RECOVERY_CASES, inr_derivatives=None):
    """Run the fixed A/B/C/D comparisons; all maps follow source-to-target convention."""
    experiments, maps, tensors = {}, {}, {}
    oracle = sequences["Oracle"]
    for label, sequence_name, horizontal in cases:
        print(f"Recovering {label}", flush=True)
        sequence = sequences[sequence_name]
        if sequence_name == "INR" and inr_derivatives is not None:
            from cameraman.core.flow import estimate_velocity_from_derivatives
            velocity, brightness, diagnostics = estimate_velocity_from_derivatives(
                *inr_derivatives, horizontal=horizontal, smoothness=smoothness,
                iterations=iterations, tolerance=tolerance, ridge=ridge)
            diagnostics.update(recovery_method="analytical", intervals=len(inr_derivatives[0]))
        else:
            velocity, brightness, diagnostics = estimate_velocity(sequence, horizontal, smoothness,
                                                                   iterations, tolerance, ridge)
            diagnostics.update(recovery_method="finite", intervals=len(sequence)-1)
        forward = integrate(velocity, grid, substeps)
        inverse = integrate(velocity, grid, substeps, inverse=True)
        result = metrics(forward, inverse, grid + displacement, source, target, mask, brightness)
        result.update(diagnostics)
        squared_error = (sequence - oracle).square()[:, mask]
        result["trajectory_mse_vs_oracle"] = squared_error.mean().item()
        result["interior_trajectory_mse_vs_oracle"] = squared_error[1:-1].mean().item() if len(sequence) > 2 else None
        experiments[label] = result
        tensors[label] = {"velocity": velocity.cpu(), "forward_map": forward.cpu(), "inverse_map": inverse.cpu()}
        if not horizontal:
            maps[label] = forward
        print(f"  EPE={result['epe_pixels']:.4f} px; warped MSE={result['warped_image_mse']:.6g}", flush=True)
    return experiments, maps, tensors


def psnr(prediction, reference):
    mse = (prediction - reference).square().mean().item()
    return -10 * math.log10(max(mse, 1e-12))


def jacobian(mapping):
    dy, dx = torch.gradient(mapping, dim=(0, 1))
    return dx[..., 0] * dy[..., 1] - dy[..., 0] * dx[..., 1]


def binary_dice(prediction, reference, mask=None):
    if mask is not None:
        prediction, reference = prediction[mask], reference[mask]
    denominator = prediction.sum() + reference.sum()
    return 1. if denominator.item() == 0 else (2*(prediction & reference).sum()/denominator).item()


def foreground_dice_metrics(source, target, inverse, mask):
    """Threshold original endpoint images, then transport the source labels.

    Nearest-neighbor, zero exterior sampling preserves binary labels. Labels
    never enter training. Primary Dice uses the same fixed interior as EPE.
    """
    source_mask, target_mask = source < 50/255, target < 50/255
    warped = sample(source_mask.float(), inverse, mode="nearest", padding_mode="zeros") > .5
    return {"foreground_dice": binary_dice(warped, target_mask, mask),
            "foreground_dice_full": binary_dice(warped, target_mask)}


def ground_truth_metrics(source, target, grid, displacement, mask, geometry=None):
    inverse = geometry["inverse_map"] if geometry and geometry.get("kind") == "elastic" else grid-displacement
    return {"epe_pixels": 0., "epe_pixels_full": 0., **foreground_dice_metrics(source, target, inverse, mask)}


def metrics(mapping, inverse, ground_truth, source, target, mask, brightness):
    error = (mapping - ground_truth).norm(dim=-1)
    determinant = jacobian(mapping)
    warped = sample(source, inverse)
    return {**foreground_dice_metrics(source, target, inverse, mask),
        "epe_pixels": error[mask].mean().item(),
        "epe_pixels_full": error.mean().item(),
        "epe_p95_pixels": torch.quantile(error[mask], 0.95).item(),
        "warped_image_mse": (warped - target).square()[mask].mean().item(),
        "brightness_residual_rms": brightness[:, mask].square().mean().sqrt().item(),
        "jacobian_min": determinant[mask].min().item(),
        "jacobian_mean": determinant[mask].mean().item(),
        "folding_fraction": (determinant[mask] <= 0).float().mean().item(),
    }


def save_comparison(path, source, target, grid, displacement, sequences, mappings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def array(tensor):
        return tensor.detach().cpu().numpy()

    fig, axes = plt.subplots(7, 5, figsize=(16, 20), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")

    def image(ax, tensor, title, **kwargs):
        ax.imshow(array(tensor), **kwargs)
        ax.set_title(title, fontsize=10)

    def geometry(ax, mapping, title):
        points = array(mapping)
        stride = max(1, source.shape[0] // 16)
        for i in range(0, source.shape[0], stride):
            ax.plot(points[i, :, 0], points[i, :, 1], "b-", linewidth=.5)
            ax.plot(points[:, i, 0], points[:, i, 1], "b-", linewidth=.5)
        ax.set_xlim(0, source.shape[1] - 1)
        ax.set_ylim(source.shape[0] - 1, 0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)

    image(axes[0, 0], source, "Source: cameraman", cmap="gray", vmin=0, vmax=1)
    image(axes[0, 1], target, "Target: inverse-sampled deformation", cmap="gray", vmin=0, vmax=1)
    geometry(axes[0, 2], grid + displacement, "Ground truth forward grid")
    image(axes[0, 3], displacement[..., 0], "Ground truth horizontal displacement (px)", cmap="coolwarm")
    image(axes[0, 4], displacement[..., 1], "Ground truth vertical displacement (px)", cmap="coolwarm")
    for row, (name, sequence) in enumerate(sequences.items(), start=1):
        for col, index in enumerate(np.linspace(0, len(sequence) - 1, 5).round().astype(int)):
            image(axes[row, col], sequence[index], f"{name}: τ={index / (len(sequence)-1):.2f}", cmap="gray", vmin=0, vmax=1)
    truth = grid + displacement
    for row, (name, mapping) in enumerate(mappings.items(), start=4):
        geometry(axes[row, 0], mapping, f"{name}: recovered grid")
        image(axes[row, 1], (mapping-grid)[..., 0], "Horizontal displacement (px)", cmap="coolwarm")
        image(axes[row, 2], (mapping-grid)[..., 1], "Vertical displacement (px)", cmap="coolwarm")
        error = (mapping-truth).norm(dim=-1)
        plot = axes[row, 3].imshow(array(error), cmap="magma", vmin=0)
        axes[row, 3].set_title("Endpoint error (px)", fontsize=10)
        fig.colorbar(plot, ax=axes[row, 3], fraction=.046)
        image(axes[row, 4], jacobian(mapping), "Jacobian determinant", cmap="coolwarm", vmin=0, vmax=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_animation(path, sequences, duration=100):
    from PIL import Image, ImageDraw
    frames = []
    names = list(sequences)
    size = next(iter(sequences.values())).shape[-1]
    for k in range(len(sequences[names[0]])):
        canvas = Image.new("RGB", (size * len(names), size + 24), "white")
        draw = ImageDraw.Draw(canvas)
        for j, name in enumerate(names):
            frame = (sequences[name][k].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(frame).convert("RGB"), (j * size, 24))
            draw.text((j * size + 4, 4), name, fill="black")
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
