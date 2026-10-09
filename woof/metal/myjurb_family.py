"""Fail-closed entry point for the not-yet-portable MYJURB column kernel.

There is one CUDA export in this family. Its live arithmetic depends on the
glibc-compatible `gfk_exp` and `gfk_pow` implementations in
`glibc_flt32.cuh`; their core operations are binary64. Metal Shading Language
doesn't provide a binary64 type. A float `exp`/`pow` substitution or a
binary64-to-float intermediate cast would change the declared arithmetic, so
this module deliberately has no launcher until MYJURB can call a centrally
reviewed exact math implementation from MSL. The runtime supports descriptor-
aware int32 resources, so the `kpbl` output type is representable; that does
not resolve the transcendental dependency.
"""


def submit_myjurb(*args, **kwargs):
    """Refuse MYJURB until its CUDA math can use the shared exact MSL path."""
    raise NotImplementedError(
        "Metal MYJURB is unavailable: myjurb_column calls glibc-compatible "
        "gfk_exp/gfk_pow and the centrally reviewed exact MSL math interface "
        "is not available to this family; no float fallback or downcast is used")
