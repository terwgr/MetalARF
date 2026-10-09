# CRTM as the numerical reference for the ABI infrared operator

`abi_crtm_reference.f90` is a small Fortran driver around JCSDA's Community
Radiative Transfer Model (CRTM v3.1.1). It is a **reference instrument**,
not a data path: the forward operator the ensemble filter runs is Rust
(`rw_goes forward`, `tools/rustwx/crates/rw-goes/src/forward.rs`), and this
driver exists so the Rust operator's error can be named by term against an
established model on the same columns (`docs/arwen-global-abi-operator.md`).

## What it does

Reads a `gpuwm-da.abi-columns.v2` stream (`woof.globe.abi_reference.write_columns`:
the analysed columns of the global model at the observation points, top of
atmosphere first, with the model's vertical coordinate), runs `CRTM_K_Matrix`
for the requested ABI channels on every column (clear sky, no aerosol, water
vapor and a US-standard ozone fill as absorbers, CRTM's climatology by
latitude and season for the layers it adds above the model top), and writes
a `gpuwm-da.abi-crtm.v1` stream: brightness temperature, the surface
emissivity CRTM used, the radiance, the skin-temperature Jacobian, the layer
temperature and water-vapor Jacobians and the **nadir** layer optical depths
of the user layers (CRTM's `RTSolution%Layer_Optical_Depth`; the emission
solver applies the secant of the zenith angle itself).

```
abi_crtm_reference SENSOR_ID COEFF_DIR COLUMNS.bin OUT.bin EMIS_MODE IRLAND_FILE CH1 [CH2 ...]
  SENSOR_ID    abi_g16 / abi_g17 / abi_g18 (the SpcCoeff and TauCoeff pair must be in COEFF_DIR)
  COEFF_DIR    one flat directory (trailing slash) with every coefficient file CRTM_Init loads
  EMIS_MODE    0: CRTM's surface models (Nalli IR water by wind speed, IR land by the class table);
               1: the per-column, per-channel emissivity carried in the columns file
  IRLAND_FILE  IGBP.IRland.EmisCoeff.bin (the model's MODIFIED_IGBP_MODIS_NOAH class is the table
               index) or NPOESS.IRland.EmisCoeff.bin
```

## Building (Linux x86-64, 2026-09-06)

```
git clone --depth 1 --branch v3.1.1 https://github.com/JCSDA/CRTMv3.git      # bd262087acbd49866fac557444a3ab2bba9169b1
cd CRTMv3 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON \
      -DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch -fallow-invalid-boz"     # gfortran 15.2.0, netCDF-Fortran 4.6.2
make -j8 crtm                                                                  # libcrtm.so, module/crtm/GNU/15.2.0
gfortran -O2 -fallow-argument-mismatch -I$B/module/crtm/GNU/15.2.0 abi_crtm_reference.f90 \
      -o abi_crtm_reference -L$B/src -lcrtm -Wl,-rpath,$B/src $(nf-config --flibs)
```

`-DOPENMP=OFF` does not configure (v3.1.1's `src/CMakeLists.txt` links
`OpenMP::OpenMP_Fortran` unconditionally). With OpenMP on, the K-matrix
segfaults inside `CRTM_Atmosphere_AddLayerCopy` when more than one thread
runs; run with `OMP_NUM_THREADS=1` (27,713 columns, two channels, K-matrix:
2.1 s).

## Coefficients (public, no account)

The whole `fix_REL-3.1.1.2.tgz` (7,852,108,208 bytes, ETag
`1d405a9b0-620eb9e69fda2`, Last-Modified Fri 30 Aug 2024 19:30:07 GMT, sha256
`c2e289f690d82a3aa82d2239cbb567cd514fa0f476a8b498ceba11670685ca66`) from
`https://bin.ssec.wisc.edu/pub/s4/CRTM/` was streamed once (23.5 MB/s) and
only the files below were kept; the driver reads them from one flat
directory of symlinks.

| file | bytes | sha256 |
|---|---:|---|
| abi_g16.SpcCoeff.bin | 872 | 8d8a8792ccfcf5bf15910fea3cc77c7003d129e767366bfaf530bfe171b1b383 |
| abi_g16.TauCoeff.bin (ODPS) | 184,588 | c0c6af627088c0d4eafce5b262449c603ee4f832a3e60db34583fe4d34f6863e |
| abi_g17.SpcCoeff.bin | 872 | 5487d1b0237ce27dcf4b2c16c8b352b8aa7c1842375522ffe1130ec112903348 |
| abi_g17.TauCoeff.bin (ODPS) | 185,732 | 87b3a6dec768574553b92f4ad8811e5b6fe7af2ee3b1ad8d63fbf103be9c05f2 |
| abi_g18.SpcCoeff.bin | 872 | 48bd0a9829106c260dc005ff61479eae27602de652877a509587cbcee3679748 |
| abi_g18.TauCoeff.bin (ODPS) | 199,356 | d04c5ed599e09b171aadc01e85e3ad5b59c2ea17e3f2fed2a6a75408f8f4a6df |
| Nalli.IRwater.EmisCoeff.bin | 14,886,206 | 912eccb52eac205e99b1fc3c674afba0f2637ccfe7c578663c369a13e6fce968 |
| IGBP.IRland.EmisCoeff.bin | 5,726 | 9107c5beae7ef027742775c62b99ad5877dbeca55334af998aeaef2bb0fb8a0b |
| NPOESS.IRland.EmisCoeff.bin | 5,728 | e7348d3439ec9a8c564a09cc9d958de86bb76282e6d37d7ce995e3616006f498 |
| NPOESS.IRsnow.EmisCoeff.bin | 1,336 | 9aafe0e7d7e1eddb0c888c11157dedc81aa600bcb0d8720714cbf8b9e4a9f2bb |
| NPOESS.IRice.EmisCoeff.bin | 1,091 | cc4c18bc74b379f8693afd727ebae39ec4be2eaf91c0d2e670e9e0b1da098c06 |

GOES-19 has a public SpcCoeff (`abi_g19_v8.02_for_Ivette.tgz`, 1,428 bytes,
sha256 of the LE file `85e7f112c0e5bfc4190272cff943179fc8a2e2d626390f198dbd05be59914e37`)
but no TauCoeff, so the reference runs use the GOES-16 pair; the GOES-18 pair
measures the sensor term (band 13 -0.045 K, band 8 -0.21 K on the case).
The IGBP land table's 20 classes are, in order, the model's
MODIFIED_IGBP_MODIS_NOAH categories; classes 15 (snow and ice) and 17
(water) carry no land emissivity and are handled as snow and as grasslands
respectively.

## Where it is not

Not vendored: CRTM is 200 MB of Fortran under a public-domain licence with
its own build system, and the reference is run by hand on a Linux host, never in a release
or a test. Its version, commit, build flags, coefficient hashes and the
optical-depth convention are recorded in every fast-model table it trains
(`reference` block) and in the receipts.
