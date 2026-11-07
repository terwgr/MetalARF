// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_lbc_off_bindings.metalh"
using namespace metal;
namespace wrf_exact_lbc_off {
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
Exact32 current_mu(device const Exact32* mub2d, device const Exact32* mup,
                int j, int i, int nx)
{
    ulong q = (ulong)j*nx + i;
    return exact_add(mub2d[q], mup[q]);
}
static
Exact32 u_face_mu_wrf(device const Exact32* mub2d, device const Exact32* mup, int j, int i, int nx)
{
    ulong r = (ulong)j*nx + min(i, nx - 1);
    ulong l = (ulong)j*nx + max(i - 1, 0);
    return exact_mul(0.5f, exact_add(exact_add(exact_add(mup[r], mup[l]),
                                               mub2d[r]), mub2d[l]));
}
static
Exact32 v_face_mu_wrf(device const Exact32* mub2d, device const Exact32* mup, int j, int i, int ny,
                   int nx)
{
    ulong n = (ulong)min(j, ny - 1)*nx + i;
    ulong s = (ulong)max(j - 1, 0)*nx + i;
    return exact_mul(0.5f, exact_add(exact_add(exact_add(mup[n], mup[s]),
                                               mub2d[n]), mub2d[s]));
}
static
Exact32 coupled_current(int kind, int k, int j, int i,
                     device const Exact32* mub2d, device const Exact32* mup,
                     device const Exact32* u, device const Exact32* v, device const Exact32* w,
                     device const Exact32* thp, device const Exact32* thb,
                     device const Exact32* php, device const Exact32* scalar,
                     device const Exact32* c1h, device const Exact32* c2h,
                     device const Exact32* c1f, device const Exact32* c2f,
                     device const Exact32* msft, device const Exact32* msfu, device const Exact32* msfv,
                     int has_msf, int thb_3d,
                     int ny, int nx)
{
    int mny = ny - (kind == LBC_V);
    int mnx = nx - (kind == LBC_U);
    ulong idx = ((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i));
    if (kind == LBC_MU) return mup[(ulong)j*mnx + i];
    Exact32 mass;
    Exact32 ch;
    if (kind == LBC_U) {
        mass = u_face_mu_wrf(mub2d, mup, j, i, mnx);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 result = exact_mul(ch, u[idx]);
        return has_msf ? exact_div(result, msfu[(ulong)j*nx + i])
                       : result;
    }
    if (kind == LBC_V) {
        mass = v_face_mu_wrf(mub2d, mup, j, i, mny, mnx);
        ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
        Exact32 result = exact_mul(ch, v[idx]);
        return has_msf ? exact_mul(result,
                                   msfv[(ulong)j*nx + i])
                       : result;
    }
    mass = current_mu(mub2d, mup, j, i, mnx);
    if (kind == LBC_W) {
        ch = exact_add(exact_mul(c1f[k], mass), c2f[k]);
        return exact_mul(ch, w[idx]);
    }
    if (kind == LBC_PHI) {
        ch = exact_add(exact_mul(c1f[k], mass), c2f[k]);
        return exact_mul(ch, php[idx]);
    }
    ch = exact_add(exact_mul(c1h[k], mass), c2h[k]);
    if (kind == LBC_THETA) {
        ulong h = thb_3d ? ((ulong)(k) * (mny) * (mnx) + (ulong)(j) * (mnx) + (i)) : (ulong)k;
        Exact32 total_theta = exact_add(thb[h], thp[idx]);
        return exact_mul(ch, exact_sub(total_theta, 300.0f));
    }
    return exact_mul(ch, scalar[idx]);
}
static
Exact32 lbc_value(device const Exact32* value, device const Exact32* tendency, ulong index, Exact32 dtbc)
{
    return value[index] + dtbc * tendency[index];
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
void state_specified_relaxation(
    device Exact32*  field_tend,
    device const Exact32*  held,
    device const Exact32*  mub2d,
    device const Exact32*  mup,
    device const Exact32*  u,
    device const Exact32*  v,
    device const Exact32*  w,
    device const Exact32*  thp,
    device const Exact32*  thb,
    device const Exact32*  php,
    device const Exact32*  scalar,
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
    device const Exact32*  fcx,
    device const Exact32*  gcx,
    Exact32 dtbc, int width, int spec_zone, int relax_zone,
    int apply_relax, int clear_specified, int add_held, int divide_msf,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int frame_count, int relax_sides, int tid)
{

    if (tid >= nz*frame_count) return;
    int k = tid/frame_count;
    int p = tid - k*frame_count;
    int j, i, mapped_d;
    if (!frame_point(p, ny, nx, max(spec_zone, relax_zone),
                     &j, &i, &mapped_d)) return;
    const bool own_s = (relax_sides & 1) != 0;
    const bool own_n = (relax_sides & 2) != 0;
    const bool own_w = (relax_sides & 4) != 0;
    const bool own_e = (relax_sides & 8) != 0;
    ulong idx = ((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i));
    Exact32 result = field_tend[idx];
    Exact32 specified;
    if (boundary_index(k, j, i, ny, nx, spec_zone, width,
                       west, west_t, east, east_t, south, south_t,
                       north, north_t, dtbc, &specified, false)) {
        result = clear_specified ? 0.0f : specified;
    } else if (apply_relax) {
        int ds = j;
        int dn = ny - 1 - j;
        int dw = i;
        int de = nx - 1 - i;
        Exact32 f0, f1, f2, f3, f4;
        if (own_s && ds >= spec_zone && ds < relax_zone
            && (!own_w || i >= ds) && (!own_e || i < nx - ds)) {
            int d = ds;
            ulong b0 = ((ulong)k*width + d)*nx + i;
            ulong bm = ((ulong)k*width + d)*nx + max(i - 1, 0);
            ulong bp = ((ulong)k*width + d)*nx + min(i + 1, nx - 1);
            ulong bo = ((ulong)k*width + d - 1)*nx + i;
            ulong bi = ((ulong)k*width + d + 1)*nx + i;
            f0 = lbc_value(south, south_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(south, south_t, bm, dtbc)
                 - coupled_current(kind, k, j, max(i - 1, 0), mub2d, mup,
                                   u, v, w, thp, thb, php, scalar, c1h, c2h,
                                   c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
                                   ny, nx);
            f2 = lbc_value(south, south_t, bp, dtbc)
                 - coupled_current(kind, k, j, min(i + 1, nx - 1),
                                   mub2d, mup, u, v, w, thp, thb, php, scalar,
                                   c1h, c2h, c1f, c2f, msft, msfu, msfv,
                                   has_msf, thb_3d, ny, nx);
            f3 = lbc_value(south, south_t, bo, dtbc)
                 - coupled_current(kind, k, j - 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(south, south_t, bi, dtbc)
                 - coupled_current(kind, k, j + 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            (result) = exact_sub(exact_add((result), exact_mul((fcx[d]), (f0))), exact_mul((gcx[d]), exact_sub(exact_add(exact_add(exact_add((f1), (f2)), (f3)), (f4)), exact_mul(4.0f, (f0)))));
        } else if (own_n && dn >= spec_zone && dn < relax_zone
                   && (!own_w || i >= dn) && (!own_e || i < nx - dn)) {
            int d = dn;
            ulong b0 = ((ulong)k*width + d)*nx + i;
            ulong bm = ((ulong)k*width + d)*nx + max(i - 1, 0);
            ulong bp = ((ulong)k*width + d)*nx + min(i + 1, nx - 1);
            ulong bo = ((ulong)k*width + d - 1)*nx + i;
            ulong bi = ((ulong)k*width + d + 1)*nx + i;
            f0 = lbc_value(north, north_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(north, north_t, bm, dtbc)
                 - coupled_current(kind, k, j, max(i - 1, 0), mub2d, mup,
                                   u, v, w, thp, thb, php, scalar, c1h, c2h,
                                   c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
                                   ny, nx);
            f2 = lbc_value(north, north_t, bp, dtbc)
                 - coupled_current(kind, k, j, min(i + 1, nx - 1),
                                   mub2d, mup, u, v, w, thp, thb, php, scalar,
                                   c1h, c2h, c1f, c2f, msft, msfu, msfv,
                                   has_msf, thb_3d, ny, nx);
            f3 = lbc_value(north, north_t, bo, dtbc)
                 - coupled_current(kind, k, j + 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(north, north_t, bi, dtbc)
                 - coupled_current(kind, k, j - 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            (result) = exact_sub(exact_add((result), exact_mul((fcx[d]), (f0))), exact_mul((gcx[d]), exact_sub(exact_add(exact_add(exact_add((f1), (f2)), (f3)), (f4)), exact_mul(4.0f, (f0)))));
        } else if (own_w && dw >= spec_zone && dw < relax_zone
                   && (!own_s || j >= dw + 1)
                   && (!own_n || j < ny - dw - 1)) {
            int d = dw;
            int jm = max(j - 1, 0);
            int jp = min(j + 1, ny - 1);
            ulong b0 = ((ulong)k*ny + j)*width + d;
            ulong bm = ((ulong)k*ny + jm)*width + d;
            ulong bp = ((ulong)k*ny + jp)*width + d;
            ulong bo = ((ulong)k*ny + j)*width + d - 1;
            ulong bi = ((ulong)k*ny + j)*width + d + 1;
            f0 = lbc_value(west, west_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(west, west_t, bm, dtbc)
                 - coupled_current(kind, k, jm, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f2 = lbc_value(west, west_t, bp, dtbc)
                 - coupled_current(kind, k, jp, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f3 = lbc_value(west, west_t, bo, dtbc)
                 - coupled_current(kind, k, j, i - 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(west, west_t, bi, dtbc)
                 - coupled_current(kind, k, j, i + 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            (result) = exact_sub(exact_add((result), exact_mul((fcx[d]), (f0))), exact_mul((gcx[d]), exact_sub(exact_add(exact_add(exact_add((f1), (f2)), (f3)), (f4)), exact_mul(4.0f, (f0)))));
        } else if (own_e && de >= spec_zone && de < relax_zone
                   && (!own_s || j >= de + 1)
                   && (!own_n || j < ny - de - 1)) {
            int d = de;
            int jm = max(j - 1, 0);
            int jp = min(j + 1, ny - 1);
            ulong b0 = ((ulong)k*ny + j)*width + d;
            ulong bm = ((ulong)k*ny + jm)*width + d;
            ulong bp = ((ulong)k*ny + jp)*width + d;
            ulong bo = ((ulong)k*ny + j)*width + d - 1;
            ulong bi = ((ulong)k*ny + j)*width + d + 1;
            f0 = lbc_value(east, east_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(east, east_t, bm, dtbc)
                 - coupled_current(kind, k, jm, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f2 = lbc_value(east, east_t, bp, dtbc)
                 - coupled_current(kind, k, jp, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f3 = lbc_value(east, east_t, bo, dtbc)
                 - coupled_current(kind, k, j, i + 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(east, east_t, bi, dtbc)
                 - coupled_current(kind, k, j, i - 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            (result) = exact_sub(exact_add((result), exact_mul((fcx[d]), (f0))), exact_mul((gcx[d]), exact_sub(exact_add(exact_add(exact_add((f1), (f2)), (f3)), (f4)), exact_mul(4.0f, (f0)))));
        }
    }
    if (divide_msf) result = exact_div(result, msft[(ulong)j*nx + i]);
    if (add_held) result = exact_add(result, held[idx]);
    field_tend[idx] = result;
}
}
kernel void wrf_exact_lbc_off_state_specified_relaxation(constant WrfExactLbcOffStateSpecifiedRelaxationResources &r [[buffer(0)]],constant WrfExactLbcOffStateSpecifiedRelaxationParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(ulong(gid)>=ulong(p.nz)*ulong(p.frame_count)) return;
 wrf_exact_lbc_off::state_specified_relaxation(reinterpret_cast<device Exact32*>(r.field_tend), reinterpret_cast<device const Exact32*>(r.held), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.u), reinterpret_cast<device const Exact32*>(r.v), reinterpret_cast<device const Exact32*>(r.w), reinterpret_cast<device const Exact32*>(r.thp), reinterpret_cast<device const Exact32*>(r.thb), reinterpret_cast<device const Exact32*>(r.php), reinterpret_cast<device const Exact32*>(r.scalar), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device const Exact32*>(r.west), reinterpret_cast<device const Exact32*>(r.west_t), reinterpret_cast<device const Exact32*>(r.east), reinterpret_cast<device const Exact32*>(r.east_t), reinterpret_cast<device const Exact32*>(r.south), reinterpret_cast<device const Exact32*>(r.south_t), reinterpret_cast<device const Exact32*>(r.north), reinterpret_cast<device const Exact32*>(r.north_t), reinterpret_cast<device const Exact32*>(r.fcx), reinterpret_cast<device const Exact32*>(r.gcx), Exact32(p.dtbc), p.width, p.spec_zone, p.relax_zone, p.apply_relax, p.clear_specified, p.add_held, p.divide_msf, p.has_msf, p.thb_3d, p.kind, p.nz, p.ny, p.nx, p.frame_count, p.relax_sides,int(gid));
}
