# Local analytical versus global PCG velocity: LPBA S02

The local and recovered velocity fields differ substantially for both maintained MSE+PD-LK and NCC+fixed-moments(.01)+PD-LK checkpoints. This compares **velocities derived from SIREN derivatives**: PCG does not produce or modify image gradients. No training, RK2 integration, map evaluation or anatomical scoring was performed.

## Main comparison

Values pool the same 17,322 original foreground-union sites across all 16 actual recovery midpoints. Speeds are RMS vector magnitudes in mm per unit interpolation time; these LPBA voxels are 1mm. Field cosine is the dot product of concatenated point/time vectors divided by their norms, not the median pointwise direction cosine.

| Checkpoint | Local strict PD-LK speed | PCG speed | Speed ratio | Field cosine | Opposite directions |
|---|---:|---:|---:|---:|---:|
| MSE+PD-LK | 13.98 | 4.91 | 2.85 | .381 | 17.36% |
| NCC+fixed moments .01+PD-LK | 18.80 | 5.98 | 3.14 | .353 | 15.12% |

Opposite means local·PCG<0 where both speeds exceed .1mm per unit time. Relative vector RMSE, normalized by PCG RMS, is **2.63/2.94** for MSE/NCC. Median-of-interval pointwise direction medians are **.657/.714**; most nonzero points retain some directional agreement, while magnitude outliers and spatial variability reduce pooled agreement. The strong-original-edge subset gives the same broad finding: field cosines **.405/.353**, speed ratios **2.68/3.11**. Disagreement is not confined to flat background.

For sampled adjacent foreground voxel pairs, mean-over-time RMS velocity differences are **14.37 vs .437** for local PD-LK vs PCG (MSE), and **21.38 vs .795** (NCC). Local adjacent changes are about **33×/27×** larger; the larger local speed accounts for part, but not all, of that difference.

## Separate global coupling from the PD-LK objective

The local PD-LK velocity minimizes brightness plus gradient-transport residuals using g, q, H and m, with c=4, eta=1e-6, rho=0. PCG minimizes global brightness error plus neighbor smoothing and a ridge, using only g and q. Therefore two changes are present: a different local objective and spatial coupling.

An additional local brightness-only ridge solution, `v=-g*q/(|g|²+1e-6)`, isolates the coupling comparison. It uses the same brightness/ridge terms as PCG but no neighbors.

| Checkpoint | Local brightness-only / PCG RMS speed | Field cosine | Opposite directions | Local / PCG neighbor-change ratio |
|---|---:|---:|---:|---:|
| MSE+PD-LK | 1.045 | .507 | 12.76% | 8.89 |
| NCC+moments+PD-LK | 1.015 | .536 | 6.78% | 6.11 |

Even with almost equal RMS speed, the local brightness-only field differs in direction and is much rougher than PCG. Spatial coupling materially changes the velocity rather than merely rescaling a local field. The local brightness equation alone leaves tangential motion underdetermined; the local ridge picks a minimum-norm solution, while global smoothness couples that choice to surrounding voxels.

Within each checkpoint's own intensity units, foreground brightness residual RMS is:

| Checkpoint | Local PD-LK | Local brightness-only | PCG |
|---|---:|---:|---:|
| MSE | .03948 | .000946 | .07630 |
| NCC+moments | .11660 | .001092 | .10922 |

The local PD-LK differential residual vector RMS is **.04041/.11988**, versus PCG **.10199/.30071** for MSE/NCC. These local fits trade off different terms. PCG accepts brightness residual to produce spatially coherent flow; a smaller pointwise residual does not establish better deformation recovery. Residual magnitudes across the two checkpoints should not be compared directly because NCC uses different learned intensity units.

## Protocol and limitations

Used the existing MSE+PD-LK S02 checkpoint from `results/mc_registration/20261110T052912_070532Z/baseline/lpba/siren_pdlk/S02/endpoints.pt` and the fixed-moment NCC weight .01 checkpoint from `results/ncc_moment_screen/20261110T132700_429803Z/training/w001/S02/endpoints.pt`. These are maintained PD-LK-trained references, not the recent PD-LK-free normal/edge pilots.

PCG ran on the full native 160×192×160 grid at `(i+.5)/16`, using the original smoothing .01, ridge1e-6, maximum500 iterations and true-residual tolerance2e-5. Differential local derivatives were evaluated in chunks at the same 65,536 sampled interior voxel centres (seed20261111), excluding the outer border. The local solves used the exact sampled dense g/q arrays supplied to PCG; only H/m came from separate analytical queries. Thus first-derivative numerical differences cannot explain the velocity disagreement. As a consistency check, eager differential versus compiled dense g/q queries differed by at most 4.77e-6/2.81e-5 in absolute native derivative units for MSE/NCC.

Foreground is the union of nonzero original normalized endpoint intensities. Strong edges are the top quartile of the maximum original endpoint central-difference gradient magnitude within that foreground: 4,331 sampled sites. Masks are shared between checkpoints. Direction metrics exclude nearly stationary vectors as stated above. Neighbor roughness uses the first4,096 sampled centres and their positive-axis neighbors, restricted to foreground centres. No labels entered the probe.

MSE PCG converged in **15/16** intervals (maximum relative residual **2.01950e-5**, a small tolerance miss); NCC converged in **16/16** (**1.95057e-5** maximum). All velocity fields were finite. The probe took **35.06s** on the existing GPU. No hyperparameters were changed. These are instantaneous field differences on one pair, not evidence of registration quality or stability of integrating the local fields.

Evidence: `results/local_pcg_velocity_probe/20261110T161014_670128Z`, containing the executed runner, protocol, per-interval scalar comparisons and pooled summary. The [compact record](local_pcg_velocity_probe_20261111.json) includes denominators and preservation hashes. Checkpoints and both source images matched their pre-probe SHA256 hashes. No model weights, images, velocity fields, maps or trajectories were written. No maintained source or preset was modified, and no commit/push occurred.

A subsequent [Gaussian smoothing probe](gaussian_local_pcg_probe_20261111.md) qualifies the raw-field finding: sigma4mm local PD-LK velocities have .944/.932 field cosine with PCG at three selected times, with .330/.364 relative vector error. Much of the raw discrepancy is local fluctuation; equivalent final registration or inference speed has not been established.
