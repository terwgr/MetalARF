#include <metal_stdlib>
#include "thompson_aerosol_exner_libm.metalh"
#include "thompson_aerosol_exner_bindings.metalh"

using namespace metal;

kernel void thompson_aa_exner(
    constant ThompsonAaExnerResources &r [[buffer(0)]],
    constant ThompsonAaExnerParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint argument = soft32_div(as_type<uint>(r.pressure[idx]),
                                     as_type<uint>(100000.0f));
    const uint exponent = soft32_div(as_type<uint>(287.0f),
                                     as_type<uint>(1004.5f));
    r.pii_out[idx] = as_type<float>(thompson_exner_powf_bits(argument, exponent));
}
