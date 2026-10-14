# Registered saved-field gates

`woof verify-fields` compares saved outputs without starting a model backend:

```sh
woof verify-fields igw --candidate-dir candidate --reference-dir reference \
  --registration policy.json --candidate-config candidate/config.json \
  --candidate-initializer candidate/initializer.py --out-report report.json
```

Exit status is 0 for pass, 1 for a completed numerical failure, and 2 for
blocked coverage or admission. A report includes per-field, per-lead scores,
artifact hashes, policy hashes and reasons for blocked admission. A blocked
comparison can still contain useful scores; it supplies no acceptance claim.

Each score is FP64 `RMS(candidate_signal - reference_signal) /
RMS(reference_signal)` over every stored cell. Fields and times have separate
gates; units and leads are never pooled. A zero reference signal passes only
when the difference is zero, unless the registration independently declares
`zero_reference_rms_delta_max` in that field's units. An undefined normalized
score is JSON `null`, accompanied by the absolute RMS difference.

Supported inventories:

| Case | Saved file | Required coverage |
| --- | --- | --- |
| `gabls1` | `gabls1_head.npz` | Final `u,v,theta,e,uw,vw,wt,kv`; all 3240 samples of `ust,hfx_kin,tstar,theta_s`; matching `z,z_face` |
| `cbl_dry` | `cbl_dry_1.npz` | Final horizontal-mean `u,v,theta,speed,z`, plus TKE when saved; matching saved initial profiles |
| `igw` | `wrfout_igw.nc` | Complete `U,V,W,T` arrays at all 11 leads from 0 to 3000 seconds |
| `hill2d` | `wrfout_hill2d.nc` | Complete `U,V,W,T` arrays at all 11 hourly leads from 0 to 36000 seconds |

CBL coverage is profiles only: it cannot certify three-dimensional fields or
intermediate times. GABLS surface samples describe the pre-step state at
0, 10, ..., 32390 seconds; final profiles are at 32400 seconds. NetCDF values
and `Times` use the Rust NetCDF bridge. Times, dimensions, DX/DY, PHB/MUB/HGT,
available ZNU/ZNW/P_TOP coordinates and saved initial U/V/W/T/PH/MU must match.
Missing, empty, differently shaped or non-finite arrays block the claim.
Physical configurations must be JSON objects; profile lengths and native
staggered field shapes must match their admitted level and grid dimensions.

CBL initial height is admission geometry. Its final height is an evolved
output and receives its own registered `z` gate, rather than blocking the wind
and thermal scores when height changes differently between backends.

GABLS theta is final theta minus the independently pinned reference initial
theta, stored as a FLOAT64 `.npy` profile. Surface theta is cooling relative
to the independently registered initial surface theta. CBL theta is final
minus its matching saved initial theta. IGW/Hill theta is stored `T` minus
`reference_thb - 300`, using the **same** independently pinned FLOAT32 `.npy`
reference base array for both outputs. The base can be a matching vertical
profile or a full array with the same shape as T (including terrain-following
Hill geometry). U signals subtract the signed
background, 20 m/s for IGW and 10 m/s for Hill. Raw final theta scores in
GABLS/CBL are diagnostics and do not enter the verdict.

Registration is an independently declared policy, not a fit to candidate
results. No nonzero backend tolerance is supplied by this command. The
`exact_stored_output` mode explicitly chooses exact equality of the saved
signals and requires zero limits; that choice does not establish an acceptable
physics roundoff tolerance for other campaigns.

A fixed CBL seed makes initialization repeatable; it does not establish
nonchaotic evolution or justify a nonzero acceptance band.

The JSON structure below is illustrative: replace every digest and declaration
with campaign evidence before sealing it. Fields in `limits` must cover the
case inventory above (`theta` is the thermal signal name for all cases).

```json
{
  "schema": "woof-backend-fields-v1",
  "case": "igw",
  "mode": "exact_stored_output",
  "declared_by": "campaign policy owner",
  "declaration": "Exact stored-output equality chosen before candidate comparison",
  "limits": {
    "U": {"nrms_max": 0}, "V": {"nrms_max": 0},
    "W": {"nrms_max": 0}, "theta": {"nrms_max": 0}
  },
  "reference": {
    "producer": "cpu",
    "provenance": "Independent reference campaign record",
    "artifacts": {
      "wrfout_igw.nc": "SHA256",
      "config.json": "FILE_SHA256",
      "initializer.py": "SHA256"
    },
    "config_file": "config.json",
    "configuration_sha256": "CANONICAL_CONFIG_JSON_SHA256",
    "initializer_file": "initializer.py",
    "initializer_sha256": "SHA256",
    "base": {"file": "thb.npy", "sha256": "SHA256"}
  },
  "candidate_initializer_sha256": "INDEPENDENTLY_PINNED_CANDIDATE_INITIALIZER_SHA256"
}
```

GABLS requires `producer: "numpy_cpu"` and
`reference.surface_theta_initial`. Other references declare `cpu` or
`numpy_cpu` or `woof_cuda`. These gates compare the declared backend reference;
they do not establish that backend as a universal numerical authority.
The producer/source declaration is provenance supplied by the
campaign owner; output arrays alone cannot establish how the reference was
produced. Reference configuration and initializer files must be readable and
included in `reference.artifacts`. Candidate configuration is canonical JSON
hashed against the same physical configuration digest. Candidate and reference
initializer digests may differ when the independently registered backend
adapters differ. Supplying initializer files checks their hashes, while saved
initial arrays are checked wherever the format contains them.

After declaring the policy, seal its canonical hash and limits hash:

```python
import json
from pathlib import Path
from woof.verify.backend_fields import seal_registration

spec = json.loads(Path("policy-spec.json").read_text())
Path("policy.json").write_text(json.dumps(seal_registration(spec), indent=2) + "\n")
```

Canonical hashing uses sorted compact UTF-8 JSON without non-finite values.
`limits_sha256` hashes the limits object; `registration_sha256` hashes the
complete registration with that field omitted. File digests are SHA-256 of
their stored bytes. Missing or mismatched policy, limits, reference hashes,
base or provenance blocks admission. Sealing records a declaration; it does
not prove that limits were independently justified or selected in advance.
