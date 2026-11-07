// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_nest_uncouple_off_bindings.metalh"
using namespace metal;
namespace wrf_exact_nest_uncouple_off {
enum LbcFieldKind {
    LBC_U = 0, LBC_V = 1, LBC_THETA = 2,
    LBC_PHI = 3, LBC_MU = 4, LBC_QV = 5,
    LBC_W = 6, LBC_SCALAR = 7
};
static
Exact32 hybrid_point_current(Exact32 c1, Exact32 c2,
                          device const Exact32* mub2d, device const Exact32* mup,
                          int j, int i, int nx)
{
    ulong q = (ulong)j*nx + i;
    Exact32 base = exact_add(exact_mul(c1, mub2d[q]), c2);
    Exact32 perturbation = exact_mul(c1, mup[q]);
    return exact_add(base, perturbation);
}
static
Exact32 u_face_hybrid_current(Exact32 c1, Exact32 c2,
                           device const Exact32* mub2d, device const Exact32* mup,
                           int j, int i, int ny, int nx)
{
    if (i == 0) {
        ulong point = (ulong)j*nx;
        Exact32 base = exact_add(exact_mul(c1, mub2d[point]), c2);
        Exact32 perturbation = exact_mul(c1, mup[point]);
        Exact32 sum = exact_add(base, base);
        sum = exact_add(sum, perturbation);
        sum = exact_add(sum, perturbation);
        return exact_mul(0.5f, sum);
    }
    if (i == nx)
        return hybrid_point_current(c1, c2, mub2d, mup, j, nx - 1, nx);
    ulong right = (ulong)j*nx + i;
    ulong left = right - 1;
    Exact32 base_right = exact_add(exact_mul(c1, mub2d[right]), c2);
    Exact32 base_left = exact_add(exact_mul(c1, mub2d[left]), c2);
    Exact32 sum = exact_add(base_right, base_left);
    sum = exact_add(sum, exact_mul(c1, mup[right]));
    sum = exact_add(sum, exact_mul(c1, mup[left]));
    return exact_mul(0.5f, sum);
}
static
Exact32 v_face_hybrid_current(Exact32 c1, Exact32 c2,
                           device const Exact32* mub2d, device const Exact32* mup,
                           int j, int i, int ny, int nx)
{
    if (j == 0) {
        ulong point = (ulong)i;
        Exact32 base = exact_add(exact_mul(c1, mub2d[point]), c2);
        Exact32 perturbation = exact_mul(c1, mup[point]);
        Exact32 sum = exact_add(base, base);
        sum = exact_add(sum, perturbation);
        sum = exact_add(sum, perturbation);
        return exact_mul(0.5f, sum);
    }
    if (j == ny)
        return hybrid_point_current(c1, c2, mub2d, mup, ny - 1, i, nx);
    ulong north = (ulong)j*nx + i;
    ulong south = north - nx;
    Exact32 base_north = exact_add(exact_mul(c1, mub2d[north]), c2);
    Exact32 base_south = exact_add(exact_mul(c1, mub2d[south]), c2);
    Exact32 sum = exact_add(base_north, base_south);
    sum = exact_add(sum, exact_mul(c1, mup[north]));
    sum = exact_add(sum, exact_mul(c1, mup[south]));
    return exact_mul(0.5f, sum);
}
void uncouple_feedback_field(
    device Exact32*  target,
    device const Exact32*  coupled,
    device const Exact32*  mub2d,
    device const Exact32*  mup,
    device const Exact32*  thb,
    device const Exact32*  c1h,
    device const Exact32*  c2h,
    device const Exact32*  c1f,
    device const Exact32*  c2f,
    device const Exact32*  msft,
    device const Exact32*  msfu,
    device const Exact32*  msfv,
    int has_msf, int thb_3d, int kind,
    int i_lo, int i_hi, int j_lo, int j_hi,
    int nz, int ny, int nx, int mny, int mnx, int tid)
{
    int ni = i_hi - i_lo + 1;
    int nj = j_hi - j_lo + 1;

    if (tid >= nz*nj*ni) return;
    int k = tid/(nj*ni);
    int rem = tid - k*nj*ni;
    int j = j_lo + rem/ni;
    int i = i_lo + rem%ni;
    ulong idx = ((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i));
    Exact32 value = coupled[idx];
    Exact32 ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            value = exact_mul(value, msfu[(ulong)j*nx + i]);
        target[idx] = exact_div(value, ch);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            value = exact_mul(value, msfv[(ulong)j*nx + i]);
        target[idx] = exact_div(value, ch);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            value = exact_mul(value, msft[(ulong)j*nx + i]);
        target[idx] = exact_div(value, ch);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        target[idx] = exact_div(value, ch);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    Exact32 result = exact_div(value, ch);
    if (kind == LBC_THETA) {
        ulong h = thb_3d ? ((ulong)(k) * (mny) * (mnx) + (ulong)(j) * (mnx) + (i)) : (ulong)k;
        result = exact_sub(exact_add(result, 300.0f), thb[h]);
    }
    target[idx] = result;
}
}
kernel void wrf_exact_nest_uncouple_off_uncouple_feedback_field(constant WrfExactNestUncoupleOffUncoupleFeedbackFieldResources &r [[buffer(0)]],constant WrfExactNestUncoupleOffUncoupleFeedbackFieldParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(q.nz*(q.j_hi-q.j_lo+1)*(q.i_hi-q.i_lo+1))) return;
 wrf_exact_nest_uncouple_off::uncouple_feedback_field(reinterpret_cast<device Exact32*>(r.target), reinterpret_cast<device const Exact32*>(r.coupled), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.thb), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), q.has_msf, q.thb_3d, q.kind, q.i_lo, q.i_hi, q.j_lo, q.j_hi, q.nz, q.ny, q.nx, q.mny, q.mnx,int(gid));
}
