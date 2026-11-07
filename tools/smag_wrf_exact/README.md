# Regional exact Smagorinsky coefficients and tendencies

The Metal exact implementation covers the `diff_opt=2`, `km_opt=4`
coefficient stage reached through `launch_wrf_smag2d_km`: horizontal
deformation, terrain-limited mixing length, and the cold-zero logical
boundary row. Ordinary Metal uses its existing shader and dispatch.

`generate.py` preprocesses the retained `smag2d.cu` with
`GPUWM_WRF_EXACT` and `GPUWM_WRF_EXACT_C_DIFFUSION`. It retains the
reachable source statements and their expression trees. The translation
changes binary32 storage to `Exact32`, pointer address spaces and the
thread-index ABI. Integer binary32 operators round each operation to
nearest, ties to even, retaining subnormals. Loads and stores move words
without floating arithmetic. The generated family is linked only into
the exact-policy library.

The strict source forms height before differencing reciprocal vertical
metrics, calculates base and perturbation slopes separately, preserves
the expanded bottom interpolation sum, and retains the distinct D12
product grouping. Strain squares the deformation difference before the
quarter factor; mixing length divides `dxm * dy` by the map factor.
These expressions cannot be replaced by algebraically equivalent forms.

`compare.py` runs the production launcher on retained compiled-WRF
diffusion fixtures and arithmetic cases. An independently compiled C++
closure of the strict source supplies the additional reference. It
compares every deformation and coefficient word, including signed zeros
and subnormals, with no tolerance. Both current and time-t inputs and
one- and three-dimensional base geopotential storage are exercised.

```sh
python tools/smag_wrf_exact/generate.py
python tools/generate_metal_kernel_signatures.py \
  --replace-fragment woof/metal/signature_fragments/wrf_exact_smag_coefficients.json
# Rebuild the bundle through the compute queue before running:
GPUWM_WRF_EXACT=1 WOOF_METAL_VERIFY_SYNC=1 \
  python tools/smag_wrf_exact/compare.py --out coefficient-words.json
```

## Stress and scalar-flux tendencies

The same generator translates the strict `wrf_smag_hd_u`, `wrf_smag_hd_v`,
`wrf_smag_hd_w`, `wrf_smag_flux_s` and `wrf_smag_hd_s` closure into
`wrf_exact_smag_tendencies`, with the literal-zero boundary-strip kernel. The
translation keeps the reciprocal ALT then vapour loading, the tensor averages,
the sequential terrain corrections, the strict theta-minus-300 basis and the
expanded scalar boundary interpolation. Exact W takes the original direct
source path, not the cached-W route used by ordinary Metal. Ordinary dispatch
and the default library are unchanged.

`compare_tendencies.py` runs `launch_wrf_smag2d_hd` and `_zero_open_strips`
unchanged on the retained compiled-WRF horizontal fixtures and on synthetic
terrain/map, one- and three-dimensional `phb`, current and time-t, cancellation
and subnormal cases, with the host-compiled strict source as reference. It
compares every tendency, both scalar face-flux workspaces and the strip modes
as literal words.

```sh
GPUWM_WRF_EXACT=1 WOOF_METAL_VERIFY_SYNC=1 \
  python tools/smag_wrf_exact/compare_tendencies.py --out tendency-words.json
```

This stage does not admit a full exact forecast. The full physics guard
remains closed. Radiation/heating accumulation and other forecast contracts still need their
own arithmetic and caller evidence. The coordinate-surface `diff_opt=1`
coefficient path remains refused by this launcher.


## PBL-off vertical and surface diffusion

The generated `wrf_exact_smag_vertical` module retains the strict source
closure for u/v/w vertical stresses, vertical scalar fluxes and the six
surface branches. U/v retain reciprocal-before-difference accumulation;
wall stress retains the source averaging order and density grouping;
surface heat stores the heat-capacity division before multiplying by gravity.
Integer binary32 arithmetic retains separate rounding, division, square root
and gradual underflow. Ordinary dispatch and shaders remain unchanged.

`compare_vertical.py` calls `launch_wrf_smag2d_vertical` for both time levels,
physical and periodic axes, one- and three-dimensional base fields, scalar
and theta rows, all isfflx arms, inactive surface writers, and isolated
surface calls. Small underflow cases exercise stresses, drag and wind-speed
squares. The reference is the host-compiled strict source; every tendency
and HFX word is compared directly, including retained subnormals and zeros.
Separate cases use the production theta-carrier adoption, which sets the
exact-state base theta to 300 K in both storage layouts. Nonconstant base
theta cases are retained-source arithmetic coverage only.

```sh
python tools/smag_wrf_exact/generate.py
python tools/generate_metal_kernel_signatures.py \
  --add-fragment woof/metal/signature_fragments/wrf_exact_smag_vertical.json
# Rebuild the bundle through the compute queue before running:
GPUWM_WRF_EXACT=1 WOOF_METAL_VERIFY_SYNC=1 \
  python tools/smag_wrf_exact/compare_vertical.py --out vertical-words.json
```

The caller preserves its existing coefficient selection, including explicit
`w_km`. This module provides the local source arithmetic contract; it does
not establish whole-forecast parity. The full physics guard stays
closed.


## N2, three-dimensional Smagorinsky and prognostic TKE

The exact turbulence module translates the strict calculate_N2, smag3d_km,
tke_km and tke_rhs closures. Their pow, exp and log function boundaries use
the pinned glibc integer implementations. TKE RHS keeps each sequential
shear increment, buoyancy subtraction, dissipation and positivity limit,
including the budget export order. The caller's mass assembly retains the
CuPy add-ufunc flush-to-zero boundary. The separate bound_tke entrypoint
uses CuPy clip's ordered comparisons, retaining input NaNs and signed zero.

`compare_turbulence.py` drives the production km_opt=2/3 launchers with
current and time-t state, both mixing-length branches, moist and dry N2,
optional condensates and surface fluxes, all boundary-axis combinations,
and enabled/disabled RHS budget exports. Cases use the production strict
theta-carrier adoption (`thb=300`) in both storage layouts. Bound cases
cover signed zero, subnormal words, infinities and NaNs.

```sh
GPUWM_WRF_EXACT=1 WOOF_METAL_VERIFY_SYNC=1 \
  python tools/smag_wrf_exact/compare_turbulence.py --out turbulence-words.json
```

The reference for the turbulence stages is the host-compiled strict CUDA
source, not execution on NVIDIA hardware. The bound reference expresses
the production clip's ordered comparisons. These local contracts do not
admit a full exact forecast; the global exact-mode refusal stays closed.
