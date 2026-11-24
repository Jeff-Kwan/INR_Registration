# Analytical SIREN derivatives in SIRENPath

At each query $(x,t)$, SIRENPath evaluates four derivatives of a scalar intensity field: the voxel-space gradient $g=\nabla_x I$, the time derivative $q=\partial_t I$, the spatial Hessian $H=\nabla_x^2 I$, and the mixed derivative $m=\nabla_x\partial_t I$. The canonical CUDA implementation computes all four with one forward sweep carrying first derivatives and one reverse sweep accumulating curvature. Both sweeps use the analytical derivatives of the sine and affine layers.

## Field and coordinate convention

Let $x=(x_1,x_2,x_3)$ be continuous voxel coordinates in XYZ order. For axis lengths $n_i$, define $S=\operatorname{diag}(2/(n_i-1))$ and $h_0=Sx-\mathbf{1}$. This puts the first and last voxel centers at $-1$ and $+1$. The code accepts the normalized coordinates $h_0$ and the diagonal of $S$ separately. It reverses the stored ZYX volume shape when constructing the XYZ scale.

Every weight and bias follows the same linear path from the moving SIREN at $t=0$ to the fixed SIREN at $t=1$. For example, $W_\ell(t)=W_\ell^{(0)}+t\Delta W_\ell$, with $\Delta W_\ell=W_\ell^{(1)}-W_\ell^{(0)}$. Using column vectors and suppressing $t$ on the parameters, the network is

$$
z_\ell=W_\ell h_{\ell-1}+b_\ell,\qquad
h_\ell=\sin(\omega_\ell z_\ell),\qquad
I=w^\top h_L+b_{\rm out}.
$$

The canonical network has $L=4$ sine layers of width 512 and $\omega_\ell=30$. The output weights $w$ and bias $b_{\rm out}$ also vary with time. Time differentiation holds the query voxel fixed.

## Forward sweep: spatial and temporal tangents

Let $D_\ell=\partial h_\ell/\partial x$ and $u_\ell=\partial h_\ell/\partial t$. Initialize $D_0=S$ and $u_0=0$. At each layer the code first forms the preactivation tangents

$$
J_\ell=W_\ell D_{\ell-1},\qquad
r_\ell=W_\ell u_{\ell-1}+\Delta W_\ell h_{\ell-1}+\Delta b_\ell .
$$

Set $c_\ell=\omega_\ell\cos(\omega_\ell z_\ell)$ and $s_\ell=-\omega_\ell^2 h_\ell$, the first and second derivatives of the activation with respect to $z_\ell$. Then $D_\ell=\operatorname{diag}(c_\ell)J_\ell$ and $u_\ell=c_\ell\odot r_\ell$, where $\odot$ is elementwise multiplication. At the scalar output,

$$
q=w^\top u_L+\Delta w^\top h_L+\Delta b_{\rm out}.
$$

The last two terms account for time variation of the output layer itself.

## Reverse sweep: gradient, Hessian and mixed derivative

Let $a_\ell=\partial I/\partial h_\ell$ be the scalar output’s sensitivity to each hidden unit. Start with $a_L=w$ and sweep backwards, computing $v_\ell=a_\ell\odot c_\ell$ and $a_{\ell-1}=W_\ell^\top v_\ell$. Let $B_\ell=\operatorname{diag}(a_\ell\odot s_\ell)$. The quantities accumulated by the CUDA training path are

$$
\begin{aligned}
g &= S^\top a_0,\\
H &= \sum_{\ell=1}^{L} J_\ell^\top B_\ell J_\ell,\\
m &= D_L^\top\Delta w
   + \sum_{\ell=1}^{L}\left(
       J_\ell^\top B_\ell r_\ell
       + D_{\ell-1}^\top\Delta W_\ell^\top v_\ell
     \right).
\end{aligned}
$$

The two terms inside the mixed-derivative sum have distinct origins. The first differentiates the sine slope as its phase changes with time. The second differentiates the spatial affine map itself because its weights change with time. The output affine contributes $D_L^\top\Delta w$ for the same reason. A changing bias enters $r_\ell$, but has zero direct spatial derivative. The spatial tangents already contain $S$, so the Hessian carries one voxel scaling factor per spatial index.

## Computing the derivatives on CUDA

### Query and tensor layout

The canonical FP32 calculation evaluates four spatial queries in each of $8^3$ cells, giving $N=2048$ points at one time. The two sampled times use the same endpoint parameter differences. Each time has its own interpolated weights and its own derivative evaluation. The query dimensions flatten into an $(N,3)$ coordinate array. The four returned tensors have shapes $(N,3)$ for $g$, $(N,)$ for $q$, $(N,3,3)$ for $H$, and $(N,3)$ for $m$.

At a hidden layer of width 512, the forward sweep packs five arrays into a $(5,N,512)$ tensor. The first array holds the activation. The other four hold its derivatives with respect to $x_1,x_2,x_3,t$. All five use the same next-layer weight matrix. In a compiled CUDA module from this environment, the compiler views the tensor as a $(10240,512)$ matrix and issues a single dense matrix multiplication by the transposed $(512,512)$ weights. The bias applies only to the activation. The time lane additionally receives the product of the previous activation with the weight difference, plus the bias difference. At the first layer, the three spatial tangents come directly from the three columns of the input weights scaled by the voxel factors. They are broadcast across points.

### Triton kernel for layer curvature

The derivative routines are written as PyTorch tensor operations. TorchInductor, the backend of `torch.compile`, generates Triton GPU kernels for elementwise work and reductions and can dispatch dense products to separate matrix-multiplication routines. The following details come from inspecting a compiled width-512 FP32 module on this machine. Kernel boundaries and tile sizes can change with the compiler and input shape.

In one generated curvature kernel, a Triton program handles a tile of query points and all 512 hidden units for each point. For a point index $p$ and hidden unit $k$, the offset within a tangent lane is $512p+k$. Adjacent lanes begin $2048\times512$ elements apart. The program loads the three spatial preactivation tangents and the time tangent at those offsets. It computes the sine slope $30\cos(30z)$ and curvature $-900\sin(30z)$ from the phase, then multiplies the curvature by the incoming scalar-output sensitivity $a$.

For example, the contribution to $H_{12}$ at one point is the sum over hidden units of $(a_k s_k)J_{k1}J_{k2}$. The same weighted tangent $(a_k s_k)J_{k1}$ also contributes to $H_{11}$, $H_{13}$ and the first mixed component when multiplied by $J_{k1}$, $J_{k3}$ and $r_k$, respectively. The generated kernel reuses these products and performs FP32 reductions over the hidden-unit axis with `tl.sum`. Across the three spatial lanes this gives six distinct spatial contractions and three space–time contractions per layer. Symmetry supplies the lower triangle of $H$. The changing-weight terms in $m$ are accumulated separately as shown in the reverse-sweep equation.

This layout avoids materializing a $512\times512$ diagonal curvature matrix for each point or propagating a separate spatial Hessian through every hidden unit. The forward sweep retains the per-layer slopes, curvatures and tangents needed in reverse. The reverse sweep propagates one scalar-output adjoint vector per point through the weights.

### Compilation, replay and differentiation

For a CUDA FP32 first weight matrix of shape $(512,3)$, the training objective selects the layer-curvature algorithm above. The objective is compiled as a full graph with static shapes and the highest FP32 matrix-multiplication precision setting. After warmup, explicit CUDA Graph capture records the compiled forward and backward computations within the training update. Replay submits the captured operations with fresh sample values while retaining fixed tensor shapes. Triton fusion reduces intermediate memory traffic inside individual kernels. CUDA Graph replay reduces host launch overhead across kernels.

The analytical reverse sweep computes derivatives of *intensity*. Training backpropagation differentiates those derivative values with respect to the two endpoint networks. Ordinary autograd can follow both the interpolated weights and their differences because the sweeps use differentiable tensor operations throughout. The general second-order implementation computes the same four outputs by carrying spatial and temporal derivatives of the adjoint backwards; it provides a separate numerical check on the layer-curvature calculation.

### First-order evaluation on a dense grid

When only $g$ and $q$ are requested, the compiled routine stores each hidden layer’s input and sine slope. It initializes $q$ with the output-layer term $\Delta w^\top h_L+\Delta b_{\rm out}$ and adds $v_\ell^\top(\Delta W_\ell h_{\ell-1}+\Delta b_\ell)$ during the reverse sweep. The same sweep yields $g=S^\top a_0$. Its working state contains no second-order tangents. For a frozen network at one time, parameters are prepared once, voxel coordinates are generated on the GPU in chunks of up to 65,536 points, and each chunk writes directly into preallocated gradient and time-derivative arrays.

## Where the derivatives enter SIRENPath

### Training: two transport residuals and a local velocity

For a point moving with voxel velocity $v$, the instantaneous change in intensity is $q+g^\top v$. The change in its spatial gradient along the same path is $m+Hv$. The first residual is scalar and the second has three components. These expressions follow by differentiating $I(x(t),t)$ and $\nabla_x I(x(t),t)$ along the path, with $\dot{x}=v$.

At each of two sampled times (one in each half of the unit interval), a training cell shares one velocity across four query sites. Let $\langle\cdot\rangle$ denote their arithmetic mean. The local transport energy in each cell and its minimizer satisfy

$$
\begin{aligned}
E(v)&=\left\langle(q+g^\top v)^2+4\|m+Hv\|^2\right\rangle
       +10^{-6}\|v\|^2,\\
\left[\left\langle gg^\top+4H^\top H\right\rangle+10^{-6}\mathrm{Id}_3\right]v^*
  &=-\left\langle gq+4H^\top m\right\rangle .
\end{aligned}
$$

The implementation forms and solves this $3\times3$ system once per cell and time. The derivatives enter the training objective twice:

1. **Local transport loss (LT).** The code evaluates $E(v^*)$ with the unsmoothed cell velocity and averages it across cells and both times. Its canonical objective weight is 0.1.
2. **Trajectory-based registration loss.** Each grid of solved velocities is smoothed with a three-tap Gaussian of width $\sigma=0.5$ grid cells. For a smoothed, interpolated field $u$, the spatial midpoint step from particle position $p$ is

   $$
   R_h(p;u)=p+h\,u\!\left(p+\tfrac{h}{2}u(p)\right).
   $$

   With fields $u_1,u_2$ in time order, the forward trajectory is $p_f=R_{1/2}(R_{1/2}(p;u_1);u_2)$. It pairs the observed moving intensity at $p$ with the observed fixed intensity at $p_f$. The reverse comparison starts at the fixed image and applies the fields in reverse order with $h=-1/2$. Global and Gaussian-windowed local correlations of these intensity pairs, together with a full-image boundary penalty, give the bidirectional registration loss. Its weight decays from 0.03 to 0.01.

The endpoint correlation and moment terms use SIREN intensity values directly, and the endpoint parameter penalty uses the endpoint parameters directly. The registration term depends on $g,q,H,m$ through the solve, smoothing and particle positions. Backpropagation follows that entire chain to the endpoint parameters. Consequently, an ablation that sets the **LT weight** to zero still needs all four derivatives when the registration term is active.

### Frozen inference: a global brightness-flow solve

Inference freezes the fitted endpoint parameters and recomputes $g,q$ on the full recovery domain at the eight interval midpoints $t_k=(k+\tfrac12)/8$, $k=0,\ldots,7$. The inference data term uses only the brightness residual $q+g^\top v$. Neighboring voxels are coupled by a smoothness penalty. At one midpoint, with voxel index $i$, the linear system is

$$
\begin{aligned}
g_i(g_i^\top v_i)+0.01(Lv)_i+10^{-6}v_i&=-g_iq_i,\\
(Lv)_i&=\sum_{j\sim i}w_{ij}(v_i-v_j).
\end{aligned}
$$

Here $j\sim i$ ranges over valid axis-adjacent neighbors. The weights are one on the native voxel grid and inverse squared sampling steps on a capped recovery grid. Thus $g_i,q_i$ determine each voxel’s data term and right-hand side, while $L$ couples velocities across the image. Preconditioned conjugate gradients applies the operator using pointwise dot products and a neighbor stencil, warm-starting each subsequent interval from the preceding solution. This solve requires neither $H$ nor $m$.

Starting from the identity voxel map, inference applies the same midpoint update $R_h$ eight times with each interval velocity, using $h=1/64$. Forward integration takes the eight fields in chronological order. Inverse integration takes them in reverse order with $h=-1/64$. The inverse map $\phi_{1\to0}$ gives the moving-image sample position for each fixed voxel $x$, so the registered intensity there is $I_{\rm moving}(\phi_{1\to0}(x))$.

## Numerical checks

The CPU reference evaluates an ordinary SIREN at interpolated parameters and repeatedly differentiates its scalar output with autograd. Tests compare $g,q,H,m$ from both analytical second-order implementations to this independent reference in double precision. They also contract each result with a test vector and compare gradients with respect to both sets of endpoint parameters, the query coordinates and time. This checks that values agree and that training can differentiate through them. Separate CUDA tests compare the compiled general derivative routine and its gradients with the double-precision CPU reference.
