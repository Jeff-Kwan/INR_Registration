# ACDC acquisition

Source: the [official ACDC site](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
and its linked [Human Heart Project repository](https://humanheart-project.creatis.insa-lyon.fr/database/).
Public collection ID: `637218c173e9f0047faa00fb`; training folder ID:
`63721d7073e9f0047faa0525`.

Acquire the 100-subject labeled training release, including original full 4D cine
sequences, ED/ES images, endpoint labels and `Info.cfg`. Store unchanged under
ignored `Data/ACDC/training/patientXXX/`. The separate 50-subject challenge testing
release is outside this acquisition. No spatial/intensity preprocessing is applied.

The local `download_manifest.json` records source item IDs, byte counts and locally
computed SHA-256 hashes. These hashes record acquired content, rather than checksums
independently published by the source. `verification.json` is written after all
NIfTI payloads load, all cine values are finite, endpoint image/label geometries agree and endpoint/cine voxel spacings agree,
endpoint images exactly match the corresponding 1-based `Info.cfg` frame indices
in the 4D cine, and endpoint label values are within 0–3.

License: CC BY-NC-SA 4.0, with the source's restriction to non-commercial scientific
research. Retain the downloaded `LICENSE_TERMS.md` and `MANDATORY_CITATION.md`.
Required citation: Bernard O, Lalande A, Zotti C, Cervenansky F, et al.
**Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation
and Diagnosis: Is the Problem Solved?** IEEE TMI 37(11):2514–2525, 2018.
[DOI](https://doi.org/10.1109/TMI.2018.2837502).

Acquisition and input validation only. No training, recovery, evaluation experiment,
protocol implementation or result claim is included in this step.

Acquisition completed 2026-11-13: 100 subjects, 702 source files,
1,643,087,908 bytes. All cine arrays are finite and both supplied endpoint images
exactly match their indexed cine frames for every subject. Endpoint labels match
their corresponding endpoint geometry and contain only values 0–3.

Source-header caveat: 35 subjects have endpoint affines differing from the full
cine affine, despite exact array equality and matching voxel spacings. Both affines
and per-endpoint comparison flags are retained in `verification.json`. Raw files
are preserved. Any future loader must resolve this explicitly before mixing world
coordinates between these source representations. No alignment correction or
resampling was performed during acquisition.
