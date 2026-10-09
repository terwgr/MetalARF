// needs integration: source helper fragment for the SLUCM canopy kernels.
// Keep this out of the built library until the urban kernel family is
// admitted with its complete source and arithmetic contract.
#include <metal_stdlib>
#include "soft_gfk_pow.metalh"

using namespace metal;

// urban_ucm.cu's gfk_pow is the same pinned glibc powf path used by the
// source helper. Keep the bit interface so its RN-even result is explicit.
static inline uint urban_ucm_powf_bits(uint x, uint y)
{
    return soft_gfk_pow_bits(x, y);
}

// Remaining canopy dependencies: u_atan/u_log10 widen FP32 to binary64,
// call CUDA's binary64 atan/log10, then narrow RN-even. These must be wired
// to the central software-binary64 transcendental helpers at integration;
// MSL atan/log10 substitutions would change the source operation graph.
