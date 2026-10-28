"""Compare metrics and metadata from saved registration runs."""
from cameraman.paths import ROOT
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=(ROOT / 'runs/Comparisons/comparison'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.runs), 6, figsize=(18, 3*len(args.runs)), squeeze=False,
                              constrained_layout=True)
    report = ["# Run comparison", "", "| Run | Mode | Source/target PSNR (dB) | INR EPE (px) | Trajectory MSE | Folding fraction |",
              "|---|---|---|---|---|---|"]
    records = []
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        data = torch.load(run / "results.pt", map_location="cpu", weights_only=True)
        sequence = data["sequences"]["INR"]
        error = (data["recovery"]["D_inr_2d"]["forward_map"] - data["grid"] - data["ground_truth_displacement"]).norm(dim=-1)
        records.append((run, config, metrics, sequence, error))
    maximum = max(float(record[-1].max()) for record in records)
    for row, (run, config, metrics, sequence, error) in enumerate(records):
        values = metrics["experiments"]["D_inr_2d"]
        scores = metrics["endpoint_psnr_db"]
        if config.get('lk_weight', 0) > 0:
            rho = config['lk_affine_rho']
            mode = f"PD-LK rho={rho:g}"
        else:
            mode = config.get('experiment', 'parameter_l1' if config.get('weight_l1', 0) > 0 else 'endpoint_only')
        report.append(f"| {run.name} | {mode} | "
                      f"{scores['source']:.2f} / {scores['target']:.2f} | {values['epe_pixels']:.4f} | "
                      f"{values['trajectory_mse_vs_oracle']:.6g} | {values['folding_fraction']:.6g} |")
        for col in range(5):
            index = round(col*(len(sequence)-1)/4)
            axes[row, col].imshow(sequence[index].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{run.name}: t={index/(len(sequence)-1):.2f}", fontsize=9)
        plot = axes[row, 5].imshow(error.numpy(), cmap="magma", vmin=0, vmax=maximum)
        axes[row, 5].set_title(f"EPE: {values['epe_pixels']:.3f} px")
        fig.colorbar(plot, ax=axes[row, 5], fraction=.046)
        for ax in axes[row]:
            ax.axis("off")
    # Flag differences that compromise comparisons; do not silently imply matched runs.
    keys = ("image", "size", "amplitude", "frames", "steps", "seed", "width", "layers", "omega", "lr", "batch_size",
            "inr", "architecture", "lr_schedule", "lr_final_factor",
            "endpoint_initialization", "target_init_seed", "grad_clip", "lk_points",
            "lk_weight_schedule", "lk_differentiate_solve", "recovery_method", "intervals",
            "deformation", "deformation_seed", "elastic_sigma", "elastic_steps", "elastic_min_jacobian",
            "smoothness", "ridge", "flow_iterations", "flow_tolerance", "substeps")
    mismatches = [key for key in keys if any(record[1].get(key) != records[0][1].get(key) for record in records[1:])]
    if mismatches:
        report.extend(["", "Settings differ across runs: " + ", ".join(mismatches) + "."])
    report.extend(["", "Images use [0,1]; all error maps share one color scale. Metrics use each run's valid mask.",
                   "Inspect endpoint PSNR, geometry and EPE alongside appearance; smooth frames alone do not establish recovery."])
    fig.savefig(args.output / "trajectories_and_epe.png", dpi=140)
    plt.close(fig)
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
