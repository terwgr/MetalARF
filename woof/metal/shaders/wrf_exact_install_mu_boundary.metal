// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_install_mu_boundary_bindings.metalh"
using namespace metal;
namespace wrf_exact_lbc_install {
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
void install_mu_boundary(
    device Exact32*  mup,
    device Exact32*  old_mup_frame,
    device const Exact32*  west,
    device const Exact32*  west_t,
    device const Exact32*  east,
    device const Exact32*  east_t,
    device const Exact32*  south,
    device const Exact32*  south_t,
    device const Exact32*  north,
    device const Exact32*  north_t,
    Exact32 dtbc, int width, int spec_zone, int ny, int nx, int frame_count, int p)
{

    if (p >= frame_count) return;
    int j, i, d;
    if (!frame_point(p, ny, nx, spec_zone, &j, &i, &d)) return;
    ulong idx = (ulong)j*nx + i;
    old_mup_frame[p] = mup[idx];
    Exact32 value;
    if (boundary_index(0, j, i, ny, nx, spec_zone, width,
                       west, west_t, east, east_t, south, south_t,
                       north, north_t, dtbc, &value, true)) {
        mup[idx] = value;
    }
}
}
kernel void wrf_exact_install_mu_boundary(constant WrfExactInstallMuBoundaryResources &r [[buffer(0)]],constant WrfExactInstallMuBoundaryParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(q.frame_count)) return;
 wrf_exact_lbc_install::install_mu_boundary(reinterpret_cast<device Exact32*>(r.mup), reinterpret_cast<device Exact32*>(r.old_mup_frame), reinterpret_cast<device const Exact32*>(r.west), reinterpret_cast<device const Exact32*>(r.west_t), reinterpret_cast<device const Exact32*>(r.east), reinterpret_cast<device const Exact32*>(r.east_t), reinterpret_cast<device const Exact32*>(r.south), reinterpret_cast<device const Exact32*>(r.south_t), reinterpret_cast<device const Exact32*>(r.north), reinterpret_cast<device const Exact32*>(r.north_t), Exact32(q.dtbc), q.width, q.spec_zone, q.ny, q.nx, q.frame_count,int(gid));
}
