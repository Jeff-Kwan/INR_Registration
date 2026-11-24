# Canonical SIRENPath runtime profile on Learn2Reg OASIS

Canonical SIRENPath took **119.96 s** on a first run and **81.44 s** with the compilation cache populated. Training accounted for **89.2%** and **88.0%** of case wall time, respectively. Autograd backward took about **65.8%** of the sampled training replay. Dense analytical SIREN derivatives took **74.6%** of measured frozen recovery.

## Scope

Learn2Reg OASIS validation pair 17, moving subject 0412 and fixed subject 0411, uses a $160\times224\times192$ grid of 6,881,280 voxels. Measurements were taken on one RTX 5090 with Python 3.12.15, PyTorch 2.14.1+cu132 and Triton 3.8.0. All three fits used the same canonical config and seed 42, with 5,000 captured FP32 updates and the default eight-interval frozen recovery. Case wall time starts before the runner loads the pair and ends after it writes the checkpoint and metrics. Interpreter startup is outside this interval.

The two uninstrumented runs give the case and training wall-time breakdowns. The first used a fresh TorchInductor cache directory, and the cached repeat reused it in a new process. A separate instrumented repeat used the same cache and added CUDA events around segments of the captured training update. It also timed recovery phases with GPU synchronization before and after each phase. Its case wall time was 82.65 s. All time shares below use the denominator named in their section.

## Case wall time

| Stage | First run | Cached repeat |
| --- | ---: | ---: |
| Input loading, protocol checks and output overhead | 0.45 s (0.4%) | 0.45 s (0.5%) |
| Normalization and device transfer | 0.12 s (0.1%) | 0.12 s (0.1%) |
| Training, including setup | 107.05 s (89.2%) | 71.65 s (88.0%) |
| Endpoint decoding and diagnostics | 0.76 s (0.6%) | 0.76 s (0.9%) |
| Frozen map recovery | 11.28 s (9.4%) | 8.16 s (10.0%) |
| Registration scoring | 0.30 s (0.2%) | 0.30 s (0.4%) |
| **Total** | **119.96 s** | **81.44 s** |

Endpoint decoding and diagnostics include the two whole-volume fitted intensities, PSNR and Patch NCC. In the instrumented repeat, decoding the two endpoints took 0.713 s and evaluating Patch NCC took 0.039 s. Frozen map recovery is the registration inference stage. Registration scoring applies the fixed-to-moving map to evaluation labels and computes the Learn2Reg OASIS metrics.

## Training wall time and GPU replay

| Training wall time | First run | Cached repeat |
| --- | ---: | ---: |
| First three updates and CUDA Graph setup | 39.65 s (37.0%) | 3.78 s (5.3%) |
| Remaining training, including replay, host work and logging | 67.40 s (63.0%) | 67.86 s (94.7%) |

The setup bucket includes the first three optimizer updates, compilation or cache loading, and graph capture. It does not separately identify bare compiler time.

The table below averages 98 event samples at every 50th update from step 101 through step 4951. These measurements cover the captured work from drawing the inputs to the end of the AdamW step. Host scheduling before replay and reporting after replay are outside this interval.

| CUDA Graph segment | Mean per update | Share |
| --- | ---: | ---: |
| Input sampling and gathers | 0.106 ms | 0.80% |
| Gradient zeroing | 0.017 ms | 0.12% |
| Compiled joint objective forward | 4.368 ms | 32.76% |
| Autograd backward | 8.776 ms | 65.81% |
| Gradient norm and clipping | 0.033 ms | 0.25% |
| Fused AdamW update | 0.035 ms | 0.26% |
| **Measured replay interval** | **13.334 ms** | **100%** |

### Components of the compiled objective

An Nsight Systems trace sampled 20 consecutive replays at updates 1001–1020 of a fresh fit of the same pair with the canonical config and seed. Three small CUDA marker kernels bracketed the forward and backward passes without splitting the compiled objective. Their own execution is outside the reported times. Each percentage uses the mean GPU elapsed time between the corresponding markers.

| Component | Forward | Share of forward | Backward | Share of backward |
| --- | ---: | ---: | ---: | ---: |
| Endpoint fitting and moment matching | 2.485 ms | 57.0% | 2.564 ms | 29.2% |
| Analytical SIREN derivatives | 1.551 ms | 35.6% | 2.673 ms | 30.4% |
| Local velocity solve, transport residuals and velocity smoothing | 0.053 ms | 1.2% | 0.043 ms | 0.5% |
| GW-NCC similarity and image sampling | 0.077 ms | 1.8% | 0.042 ms | 0.5% |
| Particle RK2 and boundary penalty | 0.054 ms | 1.2% | 1.021 ms | 11.6% |
| Shared endpoint and derivative kernels | 0.000 ms | 0.0% | 1.673 ms | 19.1% |
| Other fused kernels across components | 0.024 ms | 0.5% | 0.486 ms | 5.5% |
| Endpoint parameter penalty | 0.007 ms | 0.1% | 0.013 ms | 0.2% |
| Unattributed kernels, memory operations and gaps | 0.112 ms | 2.6% | 0.265 ms | 3.0% |
| **Total** | **4.361 ms** | **100%** | **8.780 ms** | **100%** |

The attribution follows TorchInductor's generated launch order and the source locations attached to its forward and backward graphs. The Triton launch sequence matched in all 20 replays. External CUDA calls were matched between these launches. For consecutive GEMMs, the analysis uses one main launch per matrix multiplication and its following split-K reduction where present. Kernels with provenance spanning multiple components are counted once in a shared row. The resulting pass totals agree within 0.2% with the broader CUDA event measurements above.

Endpoint fitting includes the two SIREN evaluations, NCC, the moment penalty and endpoint diagnostics computed inside the training graph. The derivative row covers both sampled times and includes temporal parameter preparation. The local velocity solve row covers assembly and solution of the pooled systems, evaluation of their residuals, and smoothing of the fields. GW-NCC covers sampling observed intensities at the particle endpoints, accumulating and smoothing local statistics, and computing the similarity. Particle RK2 covers velocity sampling along trajectories and their boundary penalty.

The largest shared backward cost comes from six matrix multiplications that form endpoint weight gradients and add gradients already accumulated through the derivative path. Their measured time is listed separately because one launch does both. The registration path also pays for RK2 backpropagation through trilinear velocity sampling. The kernels directly attributable to the local transport loss (LT) and GW-NCC are small, while both losses send gradients through the derivatives and the velocity solve. The rows describe where the canonical replay executed its work.

## Frozen recovery

| Frozen recovery stage | Time | Share |
| --- | ---: | ---: |
| Dense SIREN derivatives | 6.705 s | 74.61% |
| Global PCG velocity solve | 1.777 s | 19.78% |
| Forward and inverse midpoint RK2 | 0.490 s | 5.45% |
| Remaining overhead | 0.015 s | 0.17% |
| **Total recovery** | **8.987 s** | **100%** |

Each of the eight intervals evaluates $g$ and $q$ on the full evaluation grid in 105 chunks of 65,536 queries. The derivative timer includes parameter preparation, coordinate generation and compiled evaluation. PCG includes velocity initialization or warm start, iterations and convergence checks. In the instrumented repeat all eight intervals met the relative tolerance with 230–330 iterations. The RK2 timer includes the first use of its Triton kernel. The first forward integration took 0.480 s and the other 15 forward and inverse integrations together took 0.010 s.

The instrumented recovery totals describe that run. Across the three same-seed fits, Dice ranged from 0.8166 to 0.8190 and HD95 from 1.6221 to 1.7812 voxels. The measurements come from one case and one GPU. Case-specific PCG convergence and compilation cache state can both affect the proportions.
