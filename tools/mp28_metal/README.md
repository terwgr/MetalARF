# Thompson aerosol Metal arithmetic

`generate.py` translates the source networks and their shared binary64 math
from the CUDA translation units. `generate_reflectivity.py` translates the
v4.6.1 aerosol reflector in `thompson_aerosol_state.cu`.

The MP28 WRF 3.9 caller selects the generic `refl10cm_thompson_column` in
`refl.cu` and the dry expression in `refl.py`. `generate_fork_reflectivity.py`
translates those two bodies into separate MP28 entrypoints. Their binary64
slopes, 400 kg/m³ graupel constants, snow-moment expression order and radar
arithmetic remain the generic caller's source contract. The MP8 dispatcher
keeps its existing entrypoint.

The generic reflector's ordinary `powf` uses the scalar reference power
implemented by `p3_reference_pow_bits`. The aerosol source units explicitly
call their own `thompson_aa_powf`; its fused polynomial has a distinct
rounding contract. Both power helpers retain their respective callers.

`compare_calls.py compare` runs the production MP28 wrapper through the
retained CUDA-source host backend and Metal in separate processes. Each side
keeps its own state between calls. Prepared restart inputs are preserved by
default; `--seed-condensate` adds process coverage explicitly, and
`--reflectivity` requests the actual output-due reflector on every call.
Required inputs are `--version`, `--prepared`, `--restart` and `--reference`;
`--out` saves the comparison report.

The comparison covers every state array, returned surface diagnostic, radar
handoff and live scratch array. It also reports two dead scratch slots:
`mp_z8w`, whose Metal caller forms layer thickness directly, and the final
`mp_thompson_aero_condensation_rate`, whose last consumer has already run.
Metal's working timestep array is checked against the scalar timestep.
Physical state, live scratch, missing fields and backend constants must
match bit for bit; dead scratch differences are recorded separately.
