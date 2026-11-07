#include <metal_stdlib>
#include "soft_word32.metalh"
#include "fsbm_pack_state_bindings.metalh"
#include "fsbm_unpack_state_bindings.metalh"
#include "fsbm_frozen_fraction_bindings.metalh"
using namespace metal;
kernel void fsbm_pack_state(constant FsbmPackStateResources &r [[buffer(0)]], constant FsbmPackStateParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.nz * p.ny * p.nx)) return;
    int i = gid % p.nx, j = (gid / p.nx) % p.ny, k = gid / (p.nx * p.ny);
    ulong src = gid, dst = ((ulong(j + 1) * (p.nz + 1) + k) * (p.nx + 2) + i + 1);
    ulong cells = ulong(p.nz) * p.ny * p.nx, frame = ulong(p.ny + 2) * (p.nz + 1) * (p.nx + 2);
    r.packed_w[dst] = r.src_w[src];
    if (k == p.nz - 1) r.packed_w[dst + p.nx + 2] = r.src_w[src + ulong(p.ny) * p.nx];
    if (p.staggered) { ulong x = (ulong(k) * p.ny + j) * (p.nx + 1) + i; F a = F(r.src_u[x]), b = F(r.src_u[x + 1]); r.packed_u[dst] = float(F(0.5f) * (a + b)); } else r.packed_u[dst] = r.src_u[src];
    if (p.staggered) { ulong x = (ulong(k) * (p.ny + 1) + j) * p.nx + i; F a = F(r.src_v[x]), b = F(r.src_v[x + p.nx]); r.packed_v[dst] = float(F(0.5f) * (a + b)); } else r.packed_v[dst] = r.src_v[src];
    r.packed_th_old[dst] = r.src_th_old[src];
    for (int b = 0; b < 142; ++b) r.packed_chem[ulong(b + 1) * frame + dst] = r.bins[ulong(b) * cells + src];
    r.packed_dz8w[dst] = r.src_dz8w[src];
    r.packed_rho[dst] = r.src_rho[src];
    r.packed_p[dst] = r.src_p[src];
    r.packed_pii[dst] = r.src_pii[src];
    r.packed_th[dst] = r.src_th[src];
    r.packed_qv[dst] = r.src_qv[src];
    r.packed_qv_old[dst] = r.src_qv_old[src];
    r.packed_qc[dst] = r.src_qc[src];
    r.packed_qr[dst] = r.src_qr[src];
    r.packed_qi[dst] = r.src_qi[src];
    r.packed_qs[dst] = r.src_qs[src];
    r.packed_qg[dst] = r.src_qg[src];
    r.packed_qnc[dst] = r.src_qnc[src];
    r.packed_qnr[dst] = r.src_qnr[src];
    r.packed_qni[dst] = r.src_qni[src];
    r.packed_qns[dst] = r.src_qns[src];
    r.packed_qng[dst] = r.src_qng[src];
    r.packed_qna[dst] = r.src_qna[src];
    if (k == 0) r.packed_xland[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_xland[ulong(j) * p.nx + i];
    if (k == 0) r.packed_rainnc[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_rainnc[ulong(j) * p.nx + i];
    if (k == 0) r.packed_rainncv[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_rainncv[ulong(j) * p.nx + i];
    if (k == 0) r.packed_snownc[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_snownc[ulong(j) * p.nx + i];
    if (k == 0) r.packed_snowncv[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_snowncv[ulong(j) * p.nx + i];
    if (k == 0) r.packed_graupelnc[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_graupelnc[ulong(j) * p.nx + i];
    if (k == 0) r.packed_graupelncv[ulong(j + 1) * (p.nx + 2) + i + 1] = r.src_graupelncv[ulong(j) * p.nx + i];
}
kernel void fsbm_unpack_state(constant FsbmUnpackStateResources &r [[buffer(0)]], constant FsbmUnpackStateParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.nz * p.ny * p.nx)) return;
    int i = gid % p.nx, j = (gid / p.nx) % p.ny, k = gid / (p.nx * p.ny);
    ulong dst = gid, src = ((ulong(j + 1) * (p.nz + 1) + k) * (p.nx + 2) + i + 1);
    ulong cells = ulong(p.nz) * p.ny * p.nx, frame = ulong(p.ny + 2) * (p.nz + 1) * (p.nx + 2);
    for (int b = 0; b < 142; ++b) r.bins[ulong(b) * cells + dst] = r.packed_chem[ulong(b + 1) * frame + src];
    r.src_th[dst] = r.packed_th[src];
    r.src_qv[dst] = r.packed_qv[src];
    r.src_qc[dst] = r.packed_qc[src];
    r.src_qr[dst] = r.packed_qr[src];
    r.src_qi[dst] = r.packed_qi[src];
    r.src_qs[dst] = r.packed_qs[src];
    r.src_qg[dst] = r.packed_qg[src];
    r.src_th_old[dst] = r.packed_th_old[src];
    r.src_qv_old[dst] = r.packed_qv_old[src];
    r.src_qnc[dst] = r.packed_qnc[src];
    r.src_qnr[dst] = r.packed_qnr[src];
    r.src_qni[dst] = r.packed_qni[src];
    r.src_qns[dst] = r.packed_qns[src];
    r.src_qng[dst] = r.packed_qng[src];
    r.src_qna[dst] = r.packed_qna[src];
    if (k == 0) r.src_rainnc[ulong(j) * p.nx + i] = r.packed_rainnc[ulong(j + 1) * (p.nx + 2) + i + 1];
    if (k == 0) r.src_rainncv[ulong(j) * p.nx + i] = r.packed_rainncv[ulong(j + 1) * (p.nx + 2) + i + 1];
    if (k == 0) r.src_snownc[ulong(j) * p.nx + i] = r.packed_snownc[ulong(j + 1) * (p.nx + 2) + i + 1];
    if (k == 0) r.src_snowncv[ulong(j) * p.nx + i] = r.packed_snowncv[ulong(j + 1) * (p.nx + 2) + i + 1];
    if (k == 0) r.src_graupelnc[ulong(j) * p.nx + i] = r.packed_graupelnc[ulong(j + 1) * (p.nx + 2) + i + 1];
    if (k == 0) r.src_graupelncv[ulong(j) * p.nx + i] = r.packed_graupelncv[ulong(j + 1) * (p.nx + 2) + i + 1];
}
kernel void fsbm_frozen_fraction(constant FsbmFrozenFractionResources &r [[buffer(0)]], constant FsbmFrozenFractionParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    F total = F(r.total[gid]), frozen = F(r.snow[gid]) + F(r.graupel[gid]);
    F denom = total > frozen ? total : frozen;
    if (denom < F(1.e-30f)) denom = F(1.e-30f);
    r.sr[gid] = total > F(0.0f) ? float(aa_div_impl(frozen, denom)) : 0.0f;
}
