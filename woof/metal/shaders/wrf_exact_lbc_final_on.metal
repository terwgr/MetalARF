// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_lbc_final_on_bindings.metalh"
using namespace metal;
namespace wrf_exact_lbc_final_on_impl {
static inline bool exact_isnan(Exact32 value){return soft32_is_nan(value.bits);}
enum LbcFieldKind {
    LBC_U = 0, LBC_V = 1, LBC_THETA = 2,
    LBC_PHI = 3, LBC_MU = 4, LBC_QV = 5,
    LBC_W = 6, LBC_SCALAR = 7
};
static
int min4(int a, int b, int c, int d)
{
    return min(min(a, b), min(c, d));
}
static
int frame_rows(int d, int ny) { return (ny - 1 - d > d) ? 2 : 1; }
static
int frame_cols(int d, int nx) { return (nx - 1 - d > d) ? 2 : 1; }
static
int frame_offset(int j, int i, int ny, int nx, int width)
{
    int d = min4(j, ny - 1 - j, i, nx - 1 - i);
    if (d < 0 || d >= width) return -1;
    int offset = 0;
    for (int q = 0; q < d; ++q) {
        offset += frame_rows(q, ny)*(nx - 2*q)
                  + frame_cols(q, nx)*max(ny - 2*q - 2, 0);
    }
    int ni = nx - 2*d;
    int nj = max(ny - 2*d - 2, 0);
    int rows = frame_rows(d, ny);
    if (j == d) return offset + i - d;
    if (j == ny - 1 - d) return offset + ni + i - d;
    if (i == d) return offset + rows*ni + j - d - 1;
    return offset + rows*ni + nj + j - d - 1;
}
static
Exact32 current_mu(device const Exact32* mub2d, device const Exact32* mup,
                int j, int i, int nx)
{
    ulong q = (ulong)j*nx + i;
    return exact_add(mub2d[q], mup[q]);
}
static
Exact32 old_mu(device const Exact32* old_mup_frame,
            device const Exact32* mub2d, device const Exact32* new_mup,
            int j, int i, int ny, int nx, int spec_zone)
{
    int frame = frame_offset(j, i, ny, nx, spec_zone);
    Exact32 mup = frame >= 0 ? old_mup_frame[frame]
                          : new_mup[(ulong)j*nx + i];
    return exact_add(mub2d[(ulong)j*nx + i], mup);
}
static
Exact32 u_face_mu_current(device const Exact32* mub2d, device const Exact32* mup,
                       int j, int i, int ny, int nx)
{
    if (i == 0) return current_mu(mub2d, mup, j, 0, nx);
    if (i == nx) return current_mu(mub2d, mup, j, nx - 1, nx);
    Exact32 right = current_mu(mub2d, mup, j, i, nx);
    Exact32 left = current_mu(mub2d, mup, j, i - 1, nx);
    return exact_mul(0.5f, exact_add(right, left));
}
static
Exact32 v_face_mu_current(device const Exact32* mub2d, device const Exact32* mup,
                       int j, int i, int ny, int nx)
{
    if (j == 0) return current_mu(mub2d, mup, 0, i, nx);
    if (j == ny) return current_mu(mub2d, mup, ny - 1, i, nx);
    Exact32 north = current_mu(mub2d, mup, j, i, nx);
    Exact32 south = current_mu(mub2d, mup, j - 1, i, nx);
    return exact_mul(0.5f, exact_add(north, south));
}
static
Exact32 u_face_mu_old(device const Exact32* old_mup_frame,
                   device const Exact32* mub2d, device const Exact32* mup,
                   int j, int i, int ny, int nx, int spec_zone)
{
    if (i == 0)
        return old_mu(old_mup_frame, mub2d, mup, j, 0,
                      ny, nx, spec_zone);
    if (i == nx)
        return old_mu(old_mup_frame, mub2d, mup, j, nx - 1,
                      ny, nx, spec_zone);
    Exact32 right = old_mu(old_mup_frame, mub2d, mup, j, i,
                        ny, nx, spec_zone);
    Exact32 left = old_mu(old_mup_frame, mub2d, mup, j, i - 1,
                       ny, nx, spec_zone);
    return exact_mul(0.5f, exact_add(right, left));
}
static
Exact32 v_face_mu_old(device const Exact32* old_mup_frame,
                   device const Exact32* mub2d, device const Exact32* mup,
                   int j, int i, int ny, int nx, int spec_zone)
{
    if (j == 0)
        return old_mu(old_mup_frame, mub2d, mup, 0, i,
                      ny, nx, spec_zone);
    if (j == ny)
        return old_mu(old_mup_frame, mub2d, mup, ny - 1, i,
                      ny, nx, spec_zone);
    Exact32 north = old_mu(old_mup_frame, mub2d, mup, j, i,
                        ny, nx, spec_zone);
    Exact32 south = old_mu(old_mup_frame, mub2d, mup, j - 1, i,
                        ny, nx, spec_zone);
    return exact_mul(0.5f, exact_add(north, south));
}
static
Exact32 installed_value(device const Exact32* value, device const Exact32* tendency,
                     ulong index, Exact32 dtbc)
{
    return exact_add(value[index], exact_mul(dtbc, tendency[index]));
}
static
bool boundary_index(int k, int j, int i, int ny, int nx, int spec_zone,
                    int width, device const Exact32* west, device const Exact32* west_t,
                    device const Exact32* east, device const Exact32* east_t,
                    device const Exact32* south, device const Exact32* south_t,
                    device const Exact32* north, device const Exact32* north_t,
                    Exact32 dtbc, thread Exact32* value, bool interpolate)
{
    int ds = j;
    int dn = ny - 1 - j;
    int dw = i;
    int de = nx - 1 - i;
    device const Exact32* bvalue = nullptr;
    device const Exact32* btend = nullptr;
    ulong b = 0;
    if (ds < spec_zone && i >= ds && i < nx - ds) {
        b = ((ulong)k*width + ds)*nx + i;
        bvalue = south; btend = south_t;
    } else if (dn < spec_zone && i >= dn && i < nx - dn) {
        b = ((ulong)k*width + dn)*nx + i;
        bvalue = north; btend = north_t;
    } else if (dw < spec_zone && j >= dw + 1 && j < ny - dw - 1) {
        b = ((ulong)k*ny + j)*width + dw;
        bvalue = west; btend = west_t;
    } else if (de < spec_zone && j >= de + 1 && j < ny - de - 1) {
        b = ((ulong)k*ny + j)*width + de;
        bvalue = east; btend = east_t;
    } else {
        return false;
    }
    *value = interpolate ? installed_value(bvalue, btend, b, dtbc)
                         : btend[b];
    return true;
}
static
Exact32 coupled_old_target(int kind, int k, int j, int i,
                        device const Exact32* target,
                        device const Exact32* old_mup_frame,
                        device const Exact32* mub2d, device const Exact32* mup,
                        device const Exact32* thb, device const Exact32* c1h, device const Exact32* c2h,
                        device const Exact32* c1f, device const Exact32* c2f,
                        device const Exact32* msft, device const Exact32* msfu, device const Exact32* msfv,
                        int has_msf, int thb_3d, int spec_zone,
                        int ny, int nx, int mny, int mnx)
{
    ulong idx = ((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i));
    Exact32 mass;
    Exact32 ch;
    if (kind == LBC_U) {
        mass = u_face_mu_old(old_mup_frame, mub2d, mup, j, i,
                             mny, mnx, spec_zone);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 result = exact_mul(ch, target[idx]);
        return has_msf ? exact_div(result, msfu[(ulong)j*nx + i])
                       : result;
    }
    if (kind == LBC_V) {
        mass = v_face_mu_old(old_mup_frame, mub2d, mup, j, i,
                             mny, mnx, spec_zone);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 result = exact_mul(ch, target[idx]);
        return has_msf ? exact_div(result, msfv[(ulong)j*nx + i])
                       : result;
    }
    mass = old_mu(old_mup_frame, mub2d, mup, j, i,
                  mny, mnx, spec_zone);
    if (kind == LBC_W) {
        ch = exact_add(exact_mul(c1f[k], mass), c2f[k]);
        Exact32 result = exact_mul(ch, target[idx]);
        return has_msf ? exact_div(result, msft[(ulong)j*nx + i])
                       : result;
    }
    if (kind == LBC_PHI) {
        ch = exact_add(exact_mul(c1f[k], mass), c2f[k]);
        return exact_mul(ch, target[idx]);
    }
    ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
    if (kind == LBC_THETA) {
        return exact_mul(ch, target[idx]);
    }
    return exact_mul(ch, target[idx]);
}
void finalize_state_field(
    device Exact32*  target,
    device const Exact32*  old_mup_frame,
    device const Exact32*  mub2d,
    device const Exact32*  mup,
    device const Exact32*  thb,
    device const Exact32*  scalar_unused,
    device const Exact32*  c1h,
    device const Exact32*  c2h,
    device const Exact32*  c1f,
    device const Exact32*  c2f,
    device const Exact32*  msft,
    device const Exact32*  msfu,
    device const Exact32*  msfv,
    device const Exact32*  west,
    device const Exact32*  west_t,
    device const Exact32*  east,
    device const Exact32*  east_t,
    device const Exact32*  south,
    device const Exact32*  south_t,
    device const Exact32*  north,
    device const Exact32*  north_t,
    Exact32 dtbc, int width, int spec_zone, int has_msf, int thb_3d,
    int kind, int nz, int ny, int nx, int mny, int mnx,
    device const Exact32*  muts_wrf, int tid)
{

    if (tid >= nz*ny*nx) return;
    int k = tid/(ny*nx);
    int rem = tid - k*ny*nx;
    int j = rem/nx;
    int i = rem - j*nx;
    if (muts_wrf != nullptr) {
        Exact32 bfield;
        if (!boundary_index(k, j, i, ny, nx, spec_zone, width,
                            west, west_t, east, east_t, south, south_t,
                            north, north_t, dtbc, &bfield, true))
            return;
        Exact32 mass;
        if (kind == LBC_U) {
            ulong r = (ulong)j*mnx + min(i, mnx - 1);
            ulong l = (ulong)j*mnx + max(i - 1, 0);
            mass = exact_mul(0.5f, exact_add(muts_wrf[r], muts_wrf[l]));
        } else if (kind == LBC_V) {
            ulong n = (ulong)min(j, mny - 1)*mnx + i;
            ulong s2 = (ulong)max(j - 1, 0)*mnx + i;
            mass = exact_mul(0.5f, exact_add(muts_wrf[n], muts_wrf[s2]));
        } else {
            mass = muts_wrf[(ulong)j*mnx + i];
        }
        bool full = kind == LBC_W || kind == LBC_PHI;
        Exact32 xmu = exact_add(exact_mul(full ? c1f[k] : c1h[k], mass),
                             full ? c2f[k] : c2h[k]);
        Exact32 num = bfield;
        if (has_msf) {
            if (kind == LBC_U) num = exact_mul(msfu[(ulong)j*nx + i], num);
            else if (kind == LBC_V) num = exact_mul(msfv[(ulong)j*nx + i], num);
            else if (kind == LBC_W) num = exact_mul(msft[(ulong)j*nx + i], num);
        }
        target[tid] = exact_div(num, xmu);
        return;
    }
    Exact32 coupled = coupled_old_target(
        kind, k, j, i, target, old_mup_frame, mub2d, mup, thb,
        c1h, c2h, c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
        spec_zone, ny, nx, mny, mnx);
    Exact32 installed;
    if (boundary_index(k, j, i, ny, nx, spec_zone, width,
                       west, west_t, east, east_t, south, south_t,
                       north, north_t, dtbc, &installed, true)) {
        coupled = installed;
    }
    Exact32 mass;
    Exact32 ch;
    if (kind == LBC_U) {
        mass = u_face_mu_current(mub2d, mup, j, i, mny, mnx);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 remapped = exact_mul(coupled, msfu[(ulong)j*nx + i]);
        target[tid] = exact_div(remapped, ch);
    } else if (kind == LBC_V) {
        mass = v_face_mu_current(mub2d, mup, j, i, mny, mnx);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 remapped = exact_mul(coupled, msfv[(ulong)j*nx + i]);
        target[tid] = exact_div(remapped, ch);
    } else if (kind == LBC_PHI) {
        mass = current_mu(mub2d, mup, j, i, mnx);
        ch = exact_add(exact_mul(c1f[k], mass), c2f[k]);
        target[tid] = exact_div(coupled, ch);
    } else if (kind == LBC_W) {
        mass = current_mu(mub2d, mup, j, i, mnx);
        ch = exact_add(exact_mul(c1f[k], mass), c2f[k]);
        Exact32 remapped = has_msf
            ? exact_mul(coupled, msft[(ulong)j*nx + i]) : coupled;
        target[tid] = exact_div(remapped, ch);
    } else {
        mass = current_mu(mub2d, mup, j, i, mnx);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 result = exact_div(coupled, ch);
        if (kind == LBC_THETA) {
        } else if (!exact_isnan(result)) {
            result = exact_max(result, 0.0f);
        }
        target[tid] = result;
    }
}
}
kernel void wrf_exact_lbc_final_on(constant WrfExactLbcFinalOnResources &r [[buffer(0)]],constant WrfExactLbcFinalOnParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(ulong(gid)>=ulong(q.nz)*ulong(q.ny)*ulong(q.nx)) return;
 wrf_exact_lbc_final_on_impl::finalize_state_field(reinterpret_cast<device Exact32*>(r.target), reinterpret_cast<device const Exact32*>(r.old_mup_frame), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.thb), reinterpret_cast<device const Exact32*>(r.scalar_unused), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device const Exact32*>(r.west), reinterpret_cast<device const Exact32*>(r.west_t), reinterpret_cast<device const Exact32*>(r.east), reinterpret_cast<device const Exact32*>(r.east_t), reinterpret_cast<device const Exact32*>(r.south), reinterpret_cast<device const Exact32*>(r.south_t), reinterpret_cast<device const Exact32*>(r.north), reinterpret_cast<device const Exact32*>(r.north_t), Exact32(q.dtbc), q.width, q.spec_zone, q.has_msf, q.thb_3d, q.kind, q.nz, q.ny, q.nx, q.mny, q.mnx, (q.has_muts ? reinterpret_cast<device const Exact32*>(r.muts_wrf) : nullptr),int(gid));
}
