# External baseline protocols

We compared eight methods on 309 directed MRI pairs per method. Neurite-OASIS (49 pairs), LPBA40 (45) and IXI (115) are brain MRI cohorts. ACDC (100) registers cardiac end-diastole and end-systole in both directions. We used one fit per pair with seed 42. For ACDC we transferred the selected presets without a cardiac hyperparameter search. Its prepared images are 128 × 128 voxels in plane, 19–32 slices deep, with 1.5 × 1.5 × 3.15 mm spacing. This geometry matters for sampling windows, network dimensions and penalties measured in voxels.

## Recipe choices and local adaptations

**ConvexAdam.** We chose the public image-only preset using MIND-SSC local self-similarity descriptors. We applied the same preset to brain and cardiac MRI, omitting the authors’ per-task self-configuration search. The fit used whole images without anatomical masks or segmentation-derived features.

**FireANTs-SyN.** We took symmetric normalization (SyN) directly from the authors’ OASIS brain test preset. On ACDC this meant a 5 × 5 × 5 local correlation window, gradient smoothing σ = 0.5, warp smoothing σ = 0.25, and up to 200, 100 and 50 iterations at scales 4, 2 and 1. We retained those voxel-based settings despite the thicker cardiac slices and started SyN directly on the prepared images. We used the GPU PyTorch backend because the optional fused extension was unavailable.

**Demons.** We chose the single-resolution classic Demons example in SimpleITK 2.5.7 as the reference recipe for all four MRI cohorts. We histogram-matched the moving image to the fixed image with 1,024 levels and seven match points. We ran four CPU work units and used each image’s orientation, origin and spacing to convert the physical displacement to moving-image voxel coordinates for evaluation.

**NiftyReg.** We selected the public B-spline registration defaults in NiftyReg v2.1.1 for all four cohorts and used a CPU/OpenMP build with four threads per fit. To score the native result, we expanded the B-spline control grid on the original fixed-image grid and converted the resulting physical positions to moving-image voxels.

**ccIDIR.** For brain and cardiac MRI, we passed each prepared image pair directly to the authors’ registration model and sampled the whole image in both directions. We chose the 2,500-update general-image reproduction setting and disabled the cycle-loss schedule. The pinned release supplies a general-image workflow. We kept the cycle-consistency weight at 0.001 and measured cycle errors with unit spacing along all axes. On ACDC, a one-voxel error therefore has the same cycle penalty in plane and through plane, although the physical distances are 1.5 and 3.15 mm.

**NIR-D.** We used the released single-field model with the paper’s brain MRI objective on all four cohorts, including ACDC. We used local normalized cross-correlation and a negative-Jacobian penalty weighted 1,000. To translate the paper’s Fourier settings into the released normalized coordinates, we chose Gaussian frequencies with standard deviation 3, a 2π factor in the Fourier phase and sine activation frequency 30. We preserved each cohort’s prepared intensities, including LPBA40’s white-matter normalization, and used full 32-bit precision to keep correlation gradients finite.

For ACDC we retained the brain sampler’s three-voxel stride on every axis. A 19- or 24-slice volume then supplies only seven or eight through-plane samples. We kept the sampled coordinates and used the largest odd correlation window that fit each axis, up to 9. The resulting window was 9 × 9 × 7 at those two depths and 9 × 9 × 9 on the other prepared ACDC volumes.

**NODEO.** The released network used a 160 × 192 × 144 brain grid and expected 864 features at its encoder’s first dense layer. We computed that feature count from each input and interpolated the decoder’s final output to the exact input dimensions. This accommodated the 160 × 192 × 224 Neurite-OASIS and IXI grids, the 160 × 192 × 160 LPBA40 grid and variable ACDC slice counts. On the original grid, initialization and forward output matched the authors’ implementation exactly. On ACDC we kept the brain preset’s 21-voxel-wide local correlation window and one pass of 15-voxel-wide averaging for smoothing.

**DNVF.** We chose the released OASIS brain single-scale recipe for all four cohorts. On ACDC we kept its 200 updates, factor-two velocity-grid downsampling on every axis and negative-Jacobian penalty weight of 50. We supplied one prepared image pair in memory and used placeholder labels for the training-time segmentation diagnostics. We disabled TensorFloat-32 in convolution and matrix multiplication to preserve precision in local correlation. The saved network weights match the deformation evaluated immediately before the last optimizer update.

## Common evaluation

Anatomical labels entered only after fitting. We sampled the moving labels on the original fixed grid by nearest neighbor, assigning background outside the image. We measured foreground Dice overlap, 95th-percentile Hausdorff distance in the prepared image geometry, the percentage of nonpositive Jacobian determinants, and the standard deviation of the log Jacobian determinant. IXI distances use its released unit-voxel geometry.
