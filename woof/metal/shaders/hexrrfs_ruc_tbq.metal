#include <metal_stdlib>
using namespace metal;
#include "hexrrfs_ruc_tbq_math.metalh"
#include "hexrrfs_ruc_tbq_init_bindings.metalh"
#include "hexrrfs_ruc_qsn_bindings.metalh"
#pragma clang fp contract(off)
// CUDA ruc_tbq_init advances cq by a rounded binary32 addition at each index.
// tc_exp uses the source's fused binary64 expf graph.
kernel void hexrrfs_ruc_tbq_init(constant HexrrfsRucTbqInitResources &r [[buffer(0)]],
                                 constant HexrrfsRucTbqInitParams &p [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n) || gid != 0)
        return;
    F cq = tc_sub(F(173.15f), F(.05f));
    F r61 = tc_mul(F(6.1153f), F(.62198f));
    for (int k = 0; k < 5001; ++k) {
        cq = tc_add(cq, F(.05f));
        F evs = tc_exp(tc_div(tc_mul(F(17.67f), tc_sub(cq, F(273.15f))), tc_sub(cq, F(29.65f))));
        F eis = tc_exp(tc_sub(F(22.514f), tc_div(F(6.15e3f), cq)));
        r.tbq[k] = as_type<float>(tc_mul(r61, cq >= F(273.15f) ? evs : eis).bits);
    }
}
kernel void hexrrfs_ruc_qsn(constant HexrrfsRucQsnResources &r [[buffer(0)]],
                            constant HexrrfsRucQsnParams &p [[buffer(1)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    r.qsn[c] = as_type<float>(rrfs_ruc_qsn(f32_word(as_type<uint>(r.tn[c])), r.tbq).bits);
}
