# Cameraman historical methods

Cameraman supplied a fast synthetic environment with known elastic motion,
oracle intermediate frames and a true deformation map. Prior work tested
endpoint-only fitting, shift-tolerant patch H1, patch LK, pointwise differential
LK, temporal stationarity, locally affine relaxation, metamorphosis, and a
direct space-time SIREN. The retained `cameraman/runs/` folders remain local and
ignored; their manifests preserve individual run settings.

These studies motivated the current focus on paired endpoint SIRENs and
pointwise PD-LK. Endpoint reconstruction gates matter: several old beta-sweep
runs missed the then-current 30 dB endpoint threshold; saved gate flags reflect
that policy. The current threshold is 25 dB. Those sparse comparisons alone
still do not establish a method ranking. Oracle-flow results also retain brightness ambiguity; compare
them with the true map and learned-trajectory errors separately.

The maintained comparison configs are `cameraman/configs/pdlk_standard.json`
(`rho=0`) and `cameraman/configs/pdlk_relaxed.json` (`rho=0.4`). The 2D suite is
not a substitute for multi-seed 3D evaluation.

Small reference objectives live in `cameraman/historical.py`: shifted central
patch H1, finite-frame local motion explainability, pointwise metamorphosis and
temporal stationarity. Patch LK's normal equations are retained in
`cameraman/core/lk_loss.py`. These have no training dispatch, samplers, pyramid
machinery or alternative architecture stack. Saved SIREN runs can be evaluated
with the current analysis tool; alternate INR architectures are unsupported.
