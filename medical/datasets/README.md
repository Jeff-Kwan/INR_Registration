# Medical dataset setup

Medical images and labels stay local and are excluded from Git. Add one setup
script here for each future dataset. Keep acquisition and preprocessing specific
to that dataset, record checksums and provenance, and never invent a conversion
recipe from the shape of an existing output.

## LPBA40

The existing prepared LPBA40 tree stays at
`Data/LPBA40/LPBA40_delineation/`, inside this repository but ignored by Git. The
maintained config points there through `dataset.root`. Current inputs have
shape `160×192×160`, affine `diag(-1,-1,1,1)`, and stored MRI intensity in `[0,1]`.
At runtime the loader normalizes nonzero foreground 1st/99th percentiles; it
never writes back to those files.

The raw-release preprocessing that produced this prepared grid is not recorded.
Until its provenance is recovered, `setup_lpba40.py` supports only a prepared
compatible source. It validates a complete input set and grid before writing,
copies `.nii.gz` files byte-for-byte, records hashes, rejects unsafe archives,
and refuses an existing destination. It does not download the public raw release
or guess resampling, cropping or intensity steps.

From the repository root, inspect the interface with:

```bash
python -m medical.datasets.setup_lpba40 --help
```

Given an authorized prepared archive and a new destination, the tool can be run
with `--source` or with an HTTPS `--url` and expected `--sha256`. Do not place
private links, archives or prepared images inside a tracked path. Do not use it
to change the internal prepared-data arrangement.
