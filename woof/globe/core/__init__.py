"""The engine physics this package carries rather than depends on.

WHY THIS DIRECTORY EXISTS.  Everything else under `woof/globe/` is this
model: its dynamics, its assimilation, its doors.  This directory is not.
Every module here is the ENGINE's, cut in from the source tree the model was
developed and graded in, because the PUBLISHED engine's copy of it is a
different piece of physics, or because it compiles a kernel that is.  The
last paragraph names the two that are here for the second reason.

Measured 2026-09-09 against a published woof 2.7.0: `RRTMGPRadiation` has no
effective-size bounding at all, `GrellFreitas()` takes no arguments,
`NewTiedtke()` takes no column chunk, `sfclay` dropped `vegfra` and gained
`ustm`, the YSU launcher lost the free-atmosphere mixing-length flag,
`ysu_contract` is absent outright, `landuse` no longer assigns the ice soil
category (30,233 columns of the T255 statics turn into mixed forest on silty
clay loam without it), and nine of the eleven modules carried here differ.
Eight of the fifteen carried kernel files differ with them: the seven
translation units `gf.cu`, `ntiedtke.cu`, `sfclay.cu`, `ysu.cu`, `noah.cu`,
`morrison.cu` and `rrtmgp_rte.cu`, and the header `glibc_flt32.cuh`
(re-measured on the Windows desktop 2026-09-10, line endings normalised; an
earlier count of nine here was the whole engine kernel directory, not the
fifteen files carried).  A forecast
against that engine either raises `TypeError` several minutes in or, with the
arguments adapted away, integrates under physics no receipt in the run
describes.

So the modules and kernels the model's own physics path EXECUTES and that
differ from the published engine are here, and everything else the package
reaches stays on the engine and is pinned in
`woof/globe/data/engine-seam.json`.  A carry makes the bytes this package's
own; a pin says when somebody else's bytes moved.  The two are different
mechanisms for different problems and neither substitutes for the other.

NOTHING HERE IS EDITED BY HAND.  `tools/resync_from_owner.py` cuts this
directory from the source tree with a three-way merge and rewires exactly the
imports that name a carried module; every other `woof.` import inside these
files names a module that stays on the engine and is left verbatim.  The
kernel sources are carried BYTE FOR BYTE, comments included, because the
loader hands the assembled string both to nvrtc and to the manifest that
digests it.

Two modules here match the published engine's apart from the import rewrites
the carve makes, and are carried anyway: `noah` and `morrison`.  Their kernels
are two of the eight that differ, and the loader binds its own directory, so a
module left on the engine would compile the engine's `.cu`.
"""
