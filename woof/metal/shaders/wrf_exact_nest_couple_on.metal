// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_nest_couple_on_bindings.metalh"
using namespace metal;
namespace wrf_exact_nest_couple_on {
enum LbcFieldKind {
    LBC_U = 0, LBC_V = 1, LBC_THETA = 2,
    LBC_PHI = 3, LBC_MU = 4, LBC_QV = 5,
    LBC_W = 6, LBC_SCALAR = 7
};
static
int frame_rows(int d, int ny) { return (ny - 1 - d > d) ? 2 : 1; }
static
int frame_cols(int d, int nx) { return (nx - 1 - d > d) ? 2 : 1; }
static
bool frame_point(int p, int ny, int nx, int width,
                 thread int* j, thread int* i, thread int* distance)
{
    for (int d = 0; d < width; ++d) {
        int ni = nx - 2*d;
        int nj = max(ny - 2*d - 2, 0);
        int rows = frame_rows(d, ny);
        int ring = rows*ni + frame_cols(d, nx)*nj;
        if (p >= ring) {
            p -= ring;
            continue;
        }
        *distance = d;
        if (p < ni) {
            *j = d;
            *i = d + p;
        } else if (p < rows*ni) {
            *j = ny - 1 - d;
            *i = d + p - ni;
        } else if (p < rows*ni + nj) {
            *j = d + 1 + p - rows*ni;
            *i = d;
        } else {
            *j = d + 1 + p - rows*ni - nj;
            *i = nx - 1 - d;
        }
        return true;
    }
    return false;
}
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
void couple_nest_field(
    device const Exact32*  target,
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
    device Exact32*  out,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int mny, int mnx, int tid)
{

    if (tid >= nz*ny*nx) return;
    int k = tid/(ny*nx);
    int rem = tid - k*ny*nx;
    int j = rem/nx;
    int i = rem - j*nx;
    if (kind == LBC_MU) {
        out[tid] = target[tid];
        return;
    }
    Exact32 ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = exact_div(ch, msfu[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = exact_div(ch, msfv[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            ch = exact_div(ch, msft[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    Exact32 value = target[tid];
    if (kind == LBC_THETA) {
    }
    out[tid] = exact_mul(ch, value);
}
void couple_nest_field_window(
    device const Exact32*  target,
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
    device Exact32*  out,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int mny, int mnx,
    int j0, int i0, int nj, int ni, int tid)
{

    if (tid >= nz*nj*ni) return;
    int k = tid/(nj*ni);
    int rem = tid - k*nj*ni;
    int j = j0 + rem/ni;
    int i = i0 + rem%ni;
    tid = ((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i));
    if (kind == LBC_MU) {
        out[tid] = target[tid];
        return;
    }
    Exact32 ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = exact_div(ch, msfu[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = exact_div(ch, msfv[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            ch = exact_div(ch, msft[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    Exact32 value = target[tid];
    if (kind == LBC_THETA) {
    }
    out[tid] = exact_mul(ch, value);
}
void couple_nest_field_frame(
    device const Exact32*  target,
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
    device Exact32*  out,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int mny, int mnx,
    int width, int points, int tid)
{

    if (tid >= nz*points) return;
    int k = tid/points;
    int j, i, distance;
    if (!frame_point(tid - k*points, ny, nx, width, &j, &i, &distance)) return;
    tid = ((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i));
    if (kind == LBC_MU) {
        out[tid] = target[tid];
        return;
    }
    Exact32 ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = exact_div(ch, msfu[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = exact_div(ch, msfv[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            ch = exact_div(ch, msft[(ulong)j*nx + i]);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[tid] = exact_mul(ch, target[tid]);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    Exact32 value = target[tid];
    if (kind == LBC_THETA) {
    }
    out[tid] = exact_mul(ch, value);
}
}
kernel void wrf_exact_nest_couple_on_couple_nest_field(constant WrfExactNestCoupleOnCoupleNestFieldResources &r [[buffer(0)]],constant WrfExactNestCoupleOnCoupleNestFieldParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(q.nz*q.ny*q.nx)) return;
 wrf_exact_nest_couple_on::couple_nest_field(reinterpret_cast<device const Exact32*>(r.target), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.thb), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device Exact32*>(r.out), q.has_msf, q.thb_3d, q.kind, q.nz, q.ny, q.nx, q.mny, q.mnx,int(gid));
}
kernel void wrf_exact_nest_couple_on_couple_nest_field_window(constant WrfExactNestCoupleOnCoupleNestFieldWindowResources &r [[buffer(0)]],constant WrfExactNestCoupleOnCoupleNestFieldWindowParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(q.nz*q.nj*q.ni)) return;
 wrf_exact_nest_couple_on::couple_nest_field_window(reinterpret_cast<device const Exact32*>(r.target), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.thb), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device Exact32*>(r.out), q.has_msf, q.thb_3d, q.kind, q.nz, q.ny, q.nx, q.mny, q.mnx, q.j0, q.i0, q.nj, q.ni,int(gid));
}
kernel void wrf_exact_nest_couple_on_couple_nest_field_frame(constant WrfExactNestCoupleOnCoupleNestFieldFrameResources &r [[buffer(0)]],constant WrfExactNestCoupleOnCoupleNestFieldFrameParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(q.nz*q.points)) return;
 wrf_exact_nest_couple_on::couple_nest_field_frame(reinterpret_cast<device const Exact32*>(r.target), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.thb), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device Exact32*>(r.out), q.has_msf, q.thb_3d, q.kind, q.nz, q.ny, q.nx, q.mny, q.mnx, q.width, q.points,int(gid));
}
