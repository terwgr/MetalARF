#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_slow_geopotential_bindings.metalh"
using namespace metal;
#define I3S(k,j,i,ny,nx) (((ulong)(k)*(ny)+(j))*(nx)+(i))
#define G Exact32(9.81f)
#define rn_add exact_add
#define rn_sub exact_sub
#define rn_mul exact_mul
#define rn_div exact_div
#define __fdiv_rn exact_div
#define fminf exact_min
#define fmaxf exact_max
namespace sg0 {
static inline
Exact32 base_value(device const Exact32*  field, int k, ulong c,
                ulong st, int base3d)
{
    return field[base3d ? (ulong)k * st + c : (ulong)k];
}
static inline
Exact32 mut_value(device const Exact32*  mub2d,
               device const Exact32*  mup, ulong c)
{
    return rn_add(mub2d[c], mup[c]);
}
static inline
Exact32 wrf_face_mu(device const Exact32*  mup,
                 device const Exact32*  mub2d, ulong ca, ulong cb)
{
    return rn_mul(Exact32(0.5f), rn_add(rn_add(rn_add(mup[ca], mup[cb]), mub2d[ca]),
                               mub2d[cb]));
}
static inline
Exact32 ph_total(device const Exact32*  php, device const Exact32*  phb,
              int k, ulong c, ulong st, int base3d)
{
    return rn_add(base_value(phb, k, c, st, base3d),
                  php[(ulong)k * st + c]);
}
static inline
Exact32 avg_mut(device const Exact32*  mup,
             device const Exact32*  mub2d, ulong ca, ulong cb)
{
    return wrf_face_mu(mup, mub2d, ca, cb);
}
static
Exact32 fcx_value(int face, int j, int k, ulong st, int ny, int nx,
               device const Exact32*  u,
               device const Exact32*  mup,
               device const Exact32*  mub2d,
               device const Exact32*  c1f,
               device const Exact32*  c2f,
               device const Exact32*  msfu, int has_msf,
               int top, Exact32 cfn, Exact32 cfn1)
{
    int fidx = face % nx;
    int ia = fidx, ib = (fidx - 1 + nx) % nx;
    ulong ca = (ulong)j * nx + ia, cb = (ulong)j * nx + ib;
    Exact32 mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    ulong f = I3S(k - 1, j, fidx, ny, nx + 1);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        ulong fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(ulong)j * (nx + 1) + fidx]);
    return value;
}
static
Exact32 fcy_value(int row, int i, int k, ulong st, int ny, int nx,
               device const Exact32*  v,
               device const Exact32*  mup,
               device const Exact32*  mub2d,
               device const Exact32*  c1f,
               device const Exact32*  c2f,
               device const Exact32*  msfv, int has_msf,
               int top, Exact32 cfn, Exact32 cfn1)
{
    int ridx = row % ny;
    int ja = ridx, jb = (ridx - 1 + ny) % ny;
    ulong ca = (ulong)ja * nx + i, cb = (ulong)jb * nx + i;
    Exact32 mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    ulong f = I3S(k - 1, ridx, i, ny + 1, nx);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        ulong fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(ulong)ridx * nx + i]);
    return value;
}
static
Exact32 supplied_fcx_value(int face, int j, int k, int ny, int nx,
                        device const Exact32*  u,
                        device const Exact32*  mux,
                        device const Exact32*  c1f,
                        device const Exact32*  c2f,
                        device const Exact32*  msfu, int has_msf,
                        int top, Exact32 cfn, Exact32 cfn1)
{
    int fidx = face % nx;
    Exact32 mass = rn_add(rn_mul(c1f[k], mux[(ulong)j * (nx + 1) + fidx]),
                       c2f[k]);
    ulong f = I3S(k - 1, j, fidx, ny, nx + 1);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        ulong fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(ulong)j * (nx + 1) + fidx]);
    return value;
}
static
Exact32 supplied_fcy_value(int row, int i, int k, int ny, int nx,
                        device const Exact32*  v,
                        device const Exact32*  muy,
                        device const Exact32*  c1f,
                        device const Exact32*  c2f,
                        device const Exact32*  msfv, int has_msf,
                        int top, Exact32 cfn, Exact32 cfn1)
{
    int ridx = row % ny;
    Exact32 mass = rn_add(rn_mul(c1f[k], muy[(ulong)ridx * nx + i]), c2f[k]);
    ulong f = I3S(k - 1, ridx, i, ny + 1, nx);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        ulong fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(ulong)ridx * nx + i]);
    return value;
}
static inline
Exact32 wrf_phi_field_difference(device const Exact32*  field, int axis,
                              int i, int j, int k, int offset,
                              int ny, int nx, ulong st, int profile)
{
    if (profile) return Exact32(0.0f);
    int pos = axis ? i : j;
    int length = axis ? nx : ny;
    int pp = (pos + offset + length) % length;
    int pm = (pos - offset + length) % length;
    ulong cp = axis ? (ulong)j * nx + pp : (ulong)pp * nx + i;
    ulong cm = axis ? (ulong)j * nx + pm : (ulong)pm * nx + i;
    return rn_sub(field[(ulong)k * st + cp], field[(ulong)k * st + cm]);
}
static inline
Exact32 x_difference(int i, int offset, int j, int k, ulong st, int nx,
                  device const Exact32*  php,
                  device const Exact32*  phb, int base3d)
{
    int ip = (i + offset + nx) % nx;
    int im = (i - offset + nx) % nx;
    return rn_sub(ph_total(php, phb, k, (ulong)j * nx + ip, st, base3d),
                  ph_total(php, phb, k, (ulong)j * nx + im, st, base3d));
}
static inline
Exact32 y_difference(int j, int offset, int i, int k, ulong st, int ny, int nx,
                  device const Exact32*  php,
                  device const Exact32*  phb, int base3d)
{
    int jp = (j + offset + ny) % ny;
    int jm = (j - offset + ny) % ny;
    return rn_sub(ph_total(php, phb, k, (ulong)jp * nx + i, st, base3d),
                  ph_total(php, phb, k, (ulong)jm * nx + i, st, base3d));
}
static inline
Exact32 centered_x(int i, int j, int k, ulong st, int nx,
                device const Exact32*  php,
                device const Exact32*  phb, int base3d,
                Exact32 w1, Exact32 w2, Exact32 w3, Exact32 denom)
{
    Exact32 t1 = rn_mul(w1, x_difference(i, 1, j, k, st, nx, php, phb, base3d));
    Exact32 t2 = rn_mul(w2, x_difference(i, 2, j, k, st, nx, php, phb, base3d));
    Exact32 t3 = rn_mul(w3, x_difference(i, 3, j, k, st, nx, php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}
static inline
Exact32 centered_y(int j, int i, int k, ulong st, int ny, int nx,
                device const Exact32*  php,
                device const Exact32*  phb, int base3d,
                Exact32 w1, Exact32 w2, Exact32 w3, Exact32 denom)
{
    Exact32 t1 = rn_mul(w1, y_difference(j, 1, i, k, st, ny, nx,
                                      php, phb, base3d));
    Exact32 t2 = rn_mul(w2, y_difference(j, 2, i, k, st, ny, nx,
                                      php, phb, base3d));
    Exact32 t3 = rn_mul(w3, y_difference(j, 3, i, k, st, ny, nx,
                                      php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}

void slow_geopotential_vertical(device Exact32*  rph_t,
                                device const Exact32*  ww,
                                device const Exact32*  w,
                                device const Exact32*  php,
                                device const Exact32*  phb,
                                device const Exact32*  mup,
                                device const Exact32*  mub2d,
                                device const Exact32*  rdnw,
                                device const Exact32*  fnm,
                                device const Exact32*  fnp,
                                device const Exact32*  c1f,
                                device const Exact32*  c2f,
                                device const Exact32*  msft,
                                int has_msf, int base3d,
                                int nz, int ny, int nx, uint tid)
{

    ulong st = (ulong)ny * nx;
    if (tid >= (ulong)nz * st) return;
    int k = (int)(tid / st) + 1;
    ulong c = tid - (ulong)(k - 1) * st;
    ulong ix = (ulong)k * st + c;
    int top = (k == nz);
    Exact32 tendency = top ? Exact32(0.0f) : rph_t[ix];
    if (!top) {
        Exact32 ph_km = ph_total(php, phb, k - 1, c, st, base3d);
        Exact32 ph_k = ph_total(php, phb, k, c, st, base3d);
        Exact32 ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
        Exact32 wd_lo = rn_mul(rn_mul(rn_mul(Exact32(0.5f),
                                         rn_add(ww[ix], ww[ix - st])),
                                  rdnw[k - 1]),
                            rn_sub(ph_k, ph_km));
        Exact32 wd_hi = rn_mul(rn_mul(rn_mul(Exact32(0.5f),
                                         rn_add(ww[ix + st], ww[ix])),
                                  rdnw[k]),
                            rn_sub(ph_kp, ph_k));
        Exact32 omega = rn_add(rn_mul(fnm[k], wd_hi),
                            rn_mul(fnp[k], wd_lo));
        tendency = rn_sub(tendency, omega);
    }
    Exact32 mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
    Exact32 gw = rn_mul(mass, rn_mul(G, w[ix]));
    if (has_msf) gw = rn_div(gw, msft[c]);
    rph_t[ix] = rn_add(tendency, gw);
}

void slow_geopotential_open(device Exact32* rph_t,device const Exact32* ww,device const Exact32* w,
    device const Exact32* u,device const Exact32* v,device const Exact32* php,device const Exact32* phb,
    device const Exact32* mup,device const Exact32* mub,device const Exact32* rdnw,device const Exact32* fnm,device const Exact32* fnp,
    device const Exact32* c1f,device const Exact32* c2f,Exact32 cfn,Exact32 cfn1,
    device const Exact32* msft,device const Exact32* msfu,device const Exact32* msfv,
    device const Exact32* mux,device const Exact32* muy,Exact32 quarter_rdx,Exact32 quarter_rdy,
    int mapped,int open_x,int open_y,int supplied,int add_vertical,int order,int base3d,
    int nz,int ny,int nx, uint tid)
{
    ulong st=(ulong)ny*nx;
    if(tid>=(ulong)nz*st)return;
    int k=int(tid/st)+1;ulong c=tid-(ulong)(k-1)*st,ix=(ulong)k*st+c;
    int j=int(c/nx),i=int(c-(ulong)j*nx),top=k==nz;
    Exact32 tendency=(top&&add_vertical)?Exact32(0.0f):rph_t[ix];
    Exact32 mass=rn_add(rn_mul(c1f[k],mut_value(mub,mup,c)),c2f[k]);
    if(add_vertical) {
        if(!top) {
            Exact32 dl=rn_sub(rn_add(rn_sub(php[ix],php[ix-st]),base_value(phb,k,c,st,base3d)),base_value(phb,k-1,c,st,base3d));
            Exact32 dh=rn_sub(rn_add(rn_sub(php[ix+st],php[ix]),base_value(phb,k+1,c,st,base3d)),base_value(phb,k,c,st,base3d));
            Exact32 lo=rn_mul(rn_mul(rn_mul(Exact32(0.5f),rn_add(ww[ix],ww[ix-st])),rdnw[k-1]),dl);
            Exact32 hi=rn_mul(rn_mul(rn_mul(Exact32(0.5f),rn_add(ww[ix+st],ww[ix])),rdnw[k]),dh);
            tendency=rn_sub(tendency,rn_add(rn_mul(fnm[k],hi),rn_mul(fnp[k],lo)));
        }
        Exact32 gw=rn_mul(rn_mul(mass,G),w[ix]);
        tendency=rn_add(tendency,mapped?rn_div(gw,msft[c]):gw);
    }
    for(int axis=0;axis<2;++axis) {
        int pos=axis?i:j,length=axis?nx:ny,boundary=axis?open_x:open_y;
        if(boundary&&(pos==0||pos==length-1))continue;
        int stencil=(order==2||(boundary&&(pos==1||pos==length-2)))?2:
                    ((boundary&&(pos==2||pos==length-3))?4:6);
        Exact32 rd=axis?quarter_rdx:quarter_rdy;
        if(top)rd=rn_mul(Exact32(2.0f),rd);
        Exact32 coefficient=mapped?rn_div(rd,msft[c]):rd;
        Exact32 f0,f1;
        if(axis) {
            f0=supplied?supplied_fcx_value(i,j,k,ny,nx,u,mux,c1f,c2f,msfu,mapped,top,cfn,cfn1):fcx_value(i,j,k,st,ny,nx,u,mup,mub,c1f,c2f,msfu,mapped,top,cfn,cfn1);
            f1=supplied?supplied_fcx_value(i+1,j,k,ny,nx,u,mux,c1f,c2f,msfu,mapped,top,cfn,cfn1):fcx_value(i+1,j,k,st,ny,nx,u,mup,mub,c1f,c2f,msfu,mapped,top,cfn,cfn1);
        } else {
            f0=supplied?supplied_fcy_value(j,i,k,ny,nx,v,muy,c1f,c2f,msfv,mapped,top,cfn,cfn1):fcy_value(j,i,k,st,ny,nx,v,mup,mub,c1f,c2f,msfv,mapped,top,cfn,cfn1);
            f1=supplied?supplied_fcy_value(j+1,i,k,ny,nx,v,muy,c1f,c2f,msfv,mapped,top,cfn,cfn1):fcy_value(j+1,i,k,st,ny,nx,v,mup,mub,c1f,c2f,msfv,mapped,top,cfn,cfn1);
        }
        Exact32 contribution;
        if(stencil==2) {
            int pm=(pos-1+length)%length,pp=(pos+1)%length;
            ulong cm=axis?(ulong)j*nx+pm:(ulong)pm*nx+i;
            ulong cp=axis?(ulong)j*nx+pp:(ulong)pp*nx+i;
            Exact32 dl=rn_sub(rn_add(rn_sub(base_value(phb,k,c,st,base3d),base_value(phb,k,cm,st,base3d)),php[ix]),php[(ulong)k*st+cm]);
            Exact32 dh=rn_sub(rn_add(rn_sub(base_value(phb,k,cp,st,base3d),base_value(phb,k,c,st,base3d)),php[(ulong)k*st+cp]),php[ix]);
            contribution=rn_mul(coefficient,rn_add(rn_mul(f1,dh),rn_mul(f0,dl)));
        } else {
            Exact32 p1=wrf_phi_field_difference(php,axis,i,j,k,1,ny,nx,st,0);
            Exact32 p2=wrf_phi_field_difference(php,axis,i,j,k,2,ny,nx,st,0);
            Exact32 b1=wrf_phi_field_difference(phb,axis,i,j,k,1,ny,nx,st,!base3d);
            Exact32 b2=wrf_phi_field_difference(phb,axis,i,j,k,2,ny,nx,st,!base3d),raw,reciprocal;
            if(stencil==4) {
                raw=rn_sub(rn_add(rn_sub(rn_mul(Exact32(8.0f),p1),p2),rn_mul(Exact32(8.0f),b1)),b2);
                reciprocal=__fdiv_rn(Exact32(1.0f),Exact32(12.0f));
            } else {
                Exact32 p3=wrf_phi_field_difference(php,axis,i,j,k,3,ny,nx,st,0);
                Exact32 b3=wrf_phi_field_difference(phb,axis,i,j,k,3,ny,nx,st,!base3d);
                raw=rn_add(rn_sub(rn_mul(Exact32(45.0f),p1),rn_mul(Exact32(9.0f),p2)),p3);
                raw=rn_add(rn_sub(rn_add(raw,rn_mul(Exact32(45.0f),b1)),rn_mul(Exact32(9.0f),b2)),b3);
                reciprocal=__fdiv_rn(Exact32(1.0f),Exact32(60.0f));
            }
            contribution=rn_mul(coefficient,rn_mul(rn_mul(rn_add(f1,f0),reciprocal),raw));
        }
        tendency=rn_sub(tendency,contribution);
    }
    for(int axis=0;axis<2;++axis) {
        int pos=axis?i:j,length=axis?nx:ny,boundary=axis?open_x:open_y;
        if(!boundary||(pos!=0&&pos!=length-1))continue;
        int west_top=axis&&pos==0&&top;
        int kz=west_top?k:min(k,nz-1);
        Exact32 wl,wh,hm=west_top?Exact32(0.0f):fnm[kz],hp=west_top?Exact32(0.0f):fnp[kz];
        if(axis) {
            ulong face=(ulong)j*(nx+1)+i,plane=(ulong)ny*(nx+1);
            wh=west_top?Exact32(0.0f):rn_add(u[(ulong)kz*plane+face+1],u[(ulong)kz*plane+face]);
            wl=rn_add(u[(ulong)(kz-1)*plane+face+1],u[(ulong)(kz-1)*plane+face]);
        } else {
            ulong face=(ulong)j*nx+i,plane=(ulong)(ny+1)*nx;
            wh=rn_add(v[(ulong)kz*plane+face+nx],v[(ulong)kz*plane+face]);
            wl=rn_add(v[(ulong)(kz-1)*plane+face+nx],v[(ulong)(kz-1)*plane+face]);
        }
        Exact32 speed=rn_mul(Exact32(0.5f),rn_add(rn_mul(hm,wh),rn_mul(hp,wl)));
        speed=pos==0?fminf(speed,Exact32(0.0f)):fmaxf(speed,Exact32(0.0f));
        ulong ci=axis?(pos==0?c+1:c-1):(pos==0?c+nx:c-nx);
        Exact32 diff=pos==0?rn_sub(php[(ulong)k*st+ci],php[ix]):rn_sub(php[ix],php[(ulong)k*st+ci]);
        Exact32 rd=rn_mul(Exact32(4.0f),axis?quarter_rdx:quarter_rdy);
        tendency=rn_sub(tendency,rn_mul(rn_mul(rd,mass),rn_mul(speed,diff)));
    }
    rph_t[ix]=tendency;
}

void slow_geopotential(device Exact32*  rph_t,
                       device const Exact32*  ww,
                       device const Exact32*  w,
                       device const Exact32*  u,
                       device const Exact32*  v,
                       device const Exact32*  php,
                       device const Exact32*  phb,
                       device const Exact32*  mup,
                       device const Exact32*  mub2d,
                       device const Exact32*  rdnw,
                       device const Exact32*  fnm,
                       device const Exact32*  fnp,
                       device const Exact32*  c1f,
                       device const Exact32*  c2f,
                       Exact32 cfn, Exact32 cfn1,
                       device const Exact32*  msft,
                       device const Exact32*  msfu,
                       device const Exact32*  msfv,
                       Exact32 quarter_rdx, Exact32 quarter_rdy,
                       int has_msf, int boundary_x, int boundary_y,
                       int specified, int order, int add_vertical,
                       int base3d, int nz, int ny, int nx, uint tid)
{

    ulong st = (ulong)ny * nx;
    if (tid >= (ulong)nz * st) return;
    int k = (int)(tid / st) + 1;
    ulong c = tid - (ulong)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (ulong)j * nx);
    ulong ix = (ulong)k * st + c;
    int top = (k == nz);
    Exact32 tendency = (top && add_vertical) ? Exact32(0.0f) : rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(Exact32(2.0f), quarter_rdx);
        quarter_rdy = rn_mul(Exact32(2.0f), quarter_rdy);
    }
    if (add_vertical) {
        if (!top) {
            Exact32 ph_km = ph_total(php, phb, k - 1, c, st, base3d);
            Exact32 ph_k = ph_total(php, phb, k, c, st, base3d);
            Exact32 ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
            Exact32 wd_lo = rn_mul(rn_mul(rn_mul(Exact32(0.5f),
                                             rn_add(ww[ix], ww[ix - st])),
                                      rdnw[k - 1]),
                                rn_sub(ph_k, ph_km));
            Exact32 wd_hi = rn_mul(rn_mul(rn_mul(Exact32(0.5f),
                                             rn_add(ww[ix + st], ww[ix])),
                                      rdnw[k]),
                                rn_sub(ph_kp, ph_k));
            Exact32 omega = rn_add(rn_mul(fnm[k], wd_hi),
                                rn_mul(fnp[k], wd_lo));
            tendency = rn_sub(tendency, omega);
        }
        Exact32 mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
        Exact32 gw = rn_mul(mass, rn_mul(G, w[ix]));
        if (has_msf) gw = rn_div(gw, msft[c]);
        tendency = rn_add(tendency, gw);
    }
    if (order == 2) {
        Exact32 dphx;
        if (boundary_x && i == 0) {
            dphx = Exact32(0.0f);
        } else {
            int im = (i - 1 + nx) % nx;
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (ulong)j * nx + im, st, base3d));
        }
        Exact32 fx = rn_mul(fcx_value(i, j, k, st, ny, nx, u, mup, mub2d,
                                   c1f, c2f, msfu, 0,
                                   top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(ulong)j * (nx + 1) + i]);
        Exact32 fxp;
        if (!(boundary_x && i == nx - 1)) {
            int ip = (i + 1) % nx;
            Exact32 dph = rn_sub(ph_total(php, phb, k,
                                       (ulong)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(fcx_value(i + 1, j, k, st, ny, nx, u, mup, mub2d,
                                   c1f, c2f, msfu, 0,
                                   top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(ulong)j * (nx + 1)
                                       + ((i + 1) % nx)]);
        } else {
            fxp = Exact32(0.0f);
        }
        Exact32 hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        if (boundary_x && (i == 0 || i == nx - 1)) hx = Exact32(0.0f);
        tendency = rn_sub(tendency, hx);
        Exact32 dphy;
        if (boundary_y && j == 0) {
            dphy = Exact32(0.0f);
        } else {
            int jm = (j - 1 + ny) % ny;
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (ulong)jm * nx + i, st, base3d));
        }
        Exact32 fy = rn_mul(fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                                   c1f, c2f, msfv, 0,
                                   top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(ulong)j * nx + i]);
        Exact32 fyp;
        if (!(boundary_y && j == ny - 1)) {
            int jp = (j + 1) % ny;
            Exact32 dph = rn_sub(ph_total(php, phb, k,
                                       (ulong)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d,
                                   c1f, c2f, msfv, 0,
                                   top, cfn, cfn1), dph);
            if (has_msf)
                fyp = rn_mul(fyp, msfv[(ulong)((j + 1) % ny) * nx + i]);
        } else {
            fyp = Exact32(0.0f);
        }
        Exact32 hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        if (boundary_y && (j == 0 || j == ny - 1)) hy = Exact32(0.0f);
        tendency = rn_sub(tendency, hy);
    } else {
        Exact32 hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
                Exact32 d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (ulong)j * nx + im, st, base3d));
                Exact32 d1 = rn_sub(ph_total(php, phb, k,
                                          (ulong)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                Exact32 f0 = rn_mul(fcx_value(i, j, k, st, ny, nx, u, mup,
                                          mub2d, c1f, c2f, msfu, has_msf,
                                          top, cfn, cfn1), d0);
                Exact32 f1 = rn_mul(fcx_value(i + 1, j, k, st, ny, nx, u, mup,
                                          mub2d, c1f, c2f, msfu, has_msf,
                                          top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = Exact32(0.0f);
            }
        } else {
            Exact32 csx = rn_add(fcx_value(i, j, k, st, ny, nx, u, mup, mub2d,
                                        c1f, c2f, msfu, has_msf,
                                        top, cfn, cfn1),
                              fcx_value(i + 1, j, k, st, ny, nx, u, mup,
                                        mub2d, c1f, c2f, msfu, has_msf,
                                        top, cfn, cfn1));
            Exact32 diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   Exact32(45.0f), Exact32(-9.0f), Exact32(1.0f), Exact32(60.0f));
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }
        Exact32 hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
                Exact32 d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (ulong)jm * nx + i, st, base3d));
                Exact32 d1 = rn_sub(ph_total(php, phb, k,
                                          (ulong)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                Exact32 f0 = rn_mul(fcy_value(j, i, k, st, ny, nx, v, mup,
                                          mub2d, c1f, c2f, msfv, has_msf,
                                          top, cfn, cfn1), d0);
                Exact32 f1 = rn_mul(fcy_value(j + 1, i, k, st, ny, nx, v, mup,
                                          mub2d, c1f, c2f, msfv, has_msf,
                                          top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                Exact32 csy = rn_add(
                    fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                              c1f, c2f, msfv, has_msf,
                              top, cfn, cfn1),
                    fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d,
                              c1f, c2f, msfv, has_msf,
                              top, cfn, cfn1));
                Exact32 diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, Exact32(8.0f), Exact32(-1.0f), Exact32(0.0f), Exact32(12.0f));
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = Exact32(0.0f);
            }
        } else {
            Exact32 csy = rn_add(fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                                        c1f, c2f, msfv, has_msf,
                                        top, cfn, cfn1),
                              fcy_value(j + 1, i, k, st, ny, nx, v, mup,
                                        mub2d, c1f, c2f, msfv, has_msf,
                                        top, cfn, cfn1));
            Exact32 diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   Exact32(45.0f), Exact32(-9.0f), Exact32(1.0f), Exact32(60.0f));
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;
}

void slow_geopotential_faces(device Exact32*  rph_t,
                             device const Exact32*  u,
                             device const Exact32*  v,
                             device const Exact32*  php,
                             device const Exact32*  phb,
                             device const Exact32*  mux,
                             device const Exact32*  muy,
                             device const Exact32*  c1f,
                             device const Exact32*  c2f,
                             Exact32 cfn, Exact32 cfn1,
                             device const Exact32*  msft,
                             device const Exact32*  msfu,
                             device const Exact32*  msfv,
                             Exact32 quarter_rdx, Exact32 quarter_rdy,
                             int has_msf, int boundary_x, int boundary_y,
                             int specified, int order, int base3d,
                             int nz, int ny, int nx, uint tid)
{

    ulong st = (ulong)ny * nx;
    if (tid >= (ulong)nz * st) return;
    int k = (int)(tid / st) + 1;
    ulong c = tid - (ulong)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (ulong)j * nx);
    ulong ix = (ulong)k * st + c;
    int top = (k == nz);
    Exact32 tendency = rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(Exact32(2.0f), quarter_rdx);
        quarter_rdy = rn_mul(Exact32(2.0f), quarter_rdy);
    }
    if (order == 2) {
        Exact32 dphx;
        if (boundary_x && i == 0) {
            dphx = Exact32(0.0f);
        } else {
            int im = (i - 1 + nx) % nx;
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (ulong)j * nx + im, st, base3d));
        }
        Exact32 fx = rn_mul(supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(ulong)j * (nx + 1) + i]);
        Exact32 fxp;
        if (!(boundary_x && i == nx - 1)) {
            int ip = (i + 1) % nx;
            Exact32 dph = rn_sub(ph_total(php, phb, k,
                                       (ulong)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(ulong)j * (nx + 1)
                                       + ((i + 1) % nx)]);
        } else {
            fxp = Exact32(0.0f);
        }
        Exact32 hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        if (boundary_x && (i == 0 || i == nx - 1)) hx = Exact32(0.0f);
        tendency = rn_sub(tendency, hx);
        Exact32 dphy;
        if (boundary_y && j == 0) {
            dphy = Exact32(0.0f);
        } else {
            int jm = (j - 1 + ny) % ny;
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (ulong)jm * nx + i, st, base3d));
        }
        Exact32 fy = rn_mul(supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(ulong)j * nx + i]);
        Exact32 fyp;
        if (!(boundary_y && j == ny - 1)) {
            int jp = (j + 1) % ny;
            Exact32 dph = rn_sub(ph_total(php, phb, k,
                                       (ulong)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fyp = rn_mul(fyp, msfv[(ulong)((j + 1) % ny) * nx + i]);
        } else {
            fyp = Exact32(0.0f);
        }
        Exact32 hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        if (boundary_y && (j == 0 || j == ny - 1)) hy = Exact32(0.0f);
        tendency = rn_sub(tendency, hy);
    } else {
        Exact32 hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
                Exact32 d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (ulong)j * nx + im, st, base3d));
                Exact32 d1 = rn_sub(ph_total(php, phb, k,
                                          (ulong)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                Exact32 f0 = rn_mul(supplied_fcx_value(
                                     i, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d0);
                Exact32 f1 = rn_mul(supplied_fcx_value(
                                     i + 1, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = Exact32(0.0f);
            }
        } else {
            Exact32 csx = rn_add(
                supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1),
                supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1));
            Exact32 diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   Exact32(45.0f), Exact32(-9.0f), Exact32(1.0f), Exact32(60.0f));
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }
        Exact32 hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
                Exact32 d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (ulong)jm * nx + i, st, base3d));
                Exact32 d1 = rn_sub(ph_total(php, phb, k,
                                          (ulong)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                Exact32 f0 = rn_mul(supplied_fcy_value(
                                     j, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d0);
                Exact32 f1 = rn_mul(supplied_fcy_value(
                                     j + 1, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                Exact32 csy = rn_add(
                    supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1),
                    supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1));
                Exact32 diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, Exact32(8.0f), Exact32(-1.0f), Exact32(0.0f), Exact32(12.0f));
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = Exact32(0.0f);
            }
        } else {
            Exact32 csy = rn_add(
                supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1),
                supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1));
            Exact32 diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   Exact32(45.0f), Exact32(-9.0f), Exact32(1.0f), Exact32(60.0f));
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;
}

}
namespace sg1 {
static inline
Exact32 base_value(device const Exact32*  field, int k, ulong c,
                ulong st, int base3d)
{
    return field[base3d ? (ulong)k * st + c : (ulong)k];
}
static inline
Exact32 mut_value(device const Exact32*  mub2d,
               device const Exact32*  mup, ulong c)
{
    return rn_add(mub2d[c], mup[c]);
}
static inline
Exact32 wrf_face_mu(device const Exact32*  mup,
                 device const Exact32*  mub2d, ulong ca, ulong cb)
{
    return rn_mul(Exact32(0.5f), rn_add(rn_add(rn_add(mup[ca], mup[cb]), mub2d[ca]),
                               mub2d[cb]));
}
static inline
Exact32 ph_total(device const Exact32*  php, device const Exact32*  phb,
              int k, ulong c, ulong st, int base3d)
{
    return rn_add(base_value(phb, k, c, st, base3d),
                  php[(ulong)k * st + c]);
}
static inline
Exact32 avg_mut(device const Exact32*  mup,
             device const Exact32*  mub2d, ulong ca, ulong cb)
{
    return wrf_face_mu(mup, mub2d, ca, cb);
}
static
Exact32 fcx_value(int face, int j, int k, ulong st, int ny, int nx,
               device const Exact32*  u,
               device const Exact32*  mup,
               device const Exact32*  mub2d,
               device const Exact32*  c1f,
               device const Exact32*  c2f,
               device const Exact32*  msfu, int has_msf,
               int top, Exact32 cfn, Exact32 cfn1)
{
    int fidx = face % nx;
    int ia = fidx, ib = (fidx - 1 + nx) % nx;
    ulong ca = (ulong)j * nx + ia, cb = (ulong)j * nx + ib;
    Exact32 mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    ulong f = I3S(k - 1, j, fidx, ny, nx + 1);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        ulong fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(ulong)j * (nx + 1) + fidx]);
    return value;
}
static
Exact32 fcy_value(int row, int i, int k, ulong st, int ny, int nx,
               device const Exact32*  v,
               device const Exact32*  mup,
               device const Exact32*  mub2d,
               device const Exact32*  c1f,
               device const Exact32*  c2f,
               device const Exact32*  msfv, int has_msf,
               int top, Exact32 cfn, Exact32 cfn1)
{
    int ridx = row % ny;
    int ja = ridx, jb = (ridx - 1 + ny) % ny;
    ulong ca = (ulong)ja * nx + i, cb = (ulong)jb * nx + i;
    Exact32 mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    ulong f = I3S(k - 1, ridx, i, ny + 1, nx);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        ulong fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(ulong)ridx * nx + i]);
    return value;
}
static
Exact32 supplied_fcx_value(int face, int j, int k, int ny, int nx,
                        device const Exact32*  u,
                        device const Exact32*  mux,
                        device const Exact32*  c1f,
                        device const Exact32*  c2f,
                        device const Exact32*  msfu, int has_msf,
                        int top, Exact32 cfn, Exact32 cfn1)
{
    int fidx = face % nx;
    Exact32 mass = rn_add(rn_mul(c1f[k], mux[(ulong)j * (nx + 1) + fidx]),
                       c2f[k]);
    ulong f = I3S(k - 1, j, fidx, ny, nx + 1);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        ulong fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(ulong)j * (nx + 1) + fidx]);
    return value;
}
static
Exact32 supplied_fcy_value(int row, int i, int k, int ny, int nx,
                        device const Exact32*  v,
                        device const Exact32*  muy,
                        device const Exact32*  c1f,
                        device const Exact32*  c2f,
                        device const Exact32*  msfv, int has_msf,
                        int top, Exact32 cfn, Exact32 cfn1)
{
    int ridx = row % ny;
    Exact32 mass = rn_add(rn_mul(c1f[k], muy[(ulong)ridx * nx + i]), c2f[k]);
    ulong f = I3S(k - 1, ridx, i, ny + 1, nx);
    Exact32 wind;
    if (top) {
        ulong fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        ulong fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    Exact32 value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(ulong)ridx * nx + i]);
    return value;
}
static inline
Exact32 wrf_phi_field_difference(device const Exact32*  field, int axis,
                              int i, int j, int k, int offset,
                              int ny, int nx, ulong st, int profile)
{
    if (profile) return Exact32(0.0f);
    int pos = axis ? i : j;
    int length = axis ? nx : ny;
    int pp = (pos + offset + length) % length;
    int pm = (pos - offset + length) % length;
    ulong cp = axis ? (ulong)j * nx + pp : (ulong)pp * nx + i;
    ulong cm = axis ? (ulong)j * nx + pm : (ulong)pm * nx + i;
    return rn_sub(field[(ulong)k * st + cp], field[(ulong)k * st + cm]);
}
static inline
Exact32 x_difference(int i, int offset, int j, int k, ulong st, int nx,
                  device const Exact32*  php,
                  device const Exact32*  phb, int base3d)
{
    int ip = (i + offset + nx) % nx;
    int im = (i - offset + nx) % nx;
    return rn_sub(ph_total(php, phb, k, (ulong)j * nx + ip, st, base3d),
                  ph_total(php, phb, k, (ulong)j * nx + im, st, base3d));
}
static inline
Exact32 y_difference(int j, int offset, int i, int k, ulong st, int ny, int nx,
                  device const Exact32*  php,
                  device const Exact32*  phb, int base3d)
{
    int jp = (j + offset + ny) % ny;
    int jm = (j - offset + ny) % ny;
    return rn_sub(ph_total(php, phb, k, (ulong)jp * nx + i, st, base3d),
                  ph_total(php, phb, k, (ulong)jm * nx + i, st, base3d));
}
static inline
Exact32 centered_x(int i, int j, int k, ulong st, int nx,
                device const Exact32*  php,
                device const Exact32*  phb, int base3d,
                Exact32 w1, Exact32 w2, Exact32 w3, Exact32 denom)
{
    Exact32 t1 = rn_mul(w1, x_difference(i, 1, j, k, st, nx, php, phb, base3d));
    Exact32 t2 = rn_mul(w2, x_difference(i, 2, j, k, st, nx, php, phb, base3d));
    Exact32 t3 = rn_mul(w3, x_difference(i, 3, j, k, st, nx, php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}
static inline
Exact32 centered_y(int j, int i, int k, ulong st, int ny, int nx,
                device const Exact32*  php,
                device const Exact32*  phb, int base3d,
                Exact32 w1, Exact32 w2, Exact32 w3, Exact32 denom)
{
    Exact32 t1 = rn_mul(w1, y_difference(j, 1, i, k, st, ny, nx,
                                      php, phb, base3d));
    Exact32 t2 = rn_mul(w2, y_difference(j, 2, i, k, st, ny, nx,
                                      php, phb, base3d));
    Exact32 t3 = rn_mul(w3, y_difference(j, 3, i, k, st, ny, nx,
                                      php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}

void slow_geopotential_vertical(device Exact32*  rph_t,
                                device const Exact32*  ww,
                                device const Exact32*  w,
                                device const Exact32*  php,
                                device const Exact32*  phb,
                                device const Exact32*  mup,
                                device const Exact32*  mub2d,
                                device const Exact32*  rdnw,
                                device const Exact32*  fnm,
                                device const Exact32*  fnp,
                                device const Exact32*  c1f,
                                device const Exact32*  c2f,
                                device const Exact32*  msft,
                                int has_msf, int base3d,
                                int nz, int ny, int nx, uint tid)
{

    ulong st = (ulong)ny * nx;
    if (tid >= (ulong)nz * st) return;
    int k = (int)(tid / st) + 1;
    ulong c = tid - (ulong)(k - 1) * st;
    ulong ix = (ulong)k * st + c;
    int top = (k == nz);
    Exact32 tendency = top ? Exact32(0.0f) : rph_t[ix];
    if (!top) {
        Exact32 ph_km = ph_total(php, phb, k - 1, c, st, base3d);
        Exact32 ph_k = ph_total(php, phb, k, c, st, base3d);
        Exact32 ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
        Exact32 wd_lo = rn_mul(rn_mul(rn_mul(Exact32(0.5f),
                                         rn_add(ww[ix], ww[ix - st])),
                                  rdnw[k - 1]),
                            rn_sub(ph_k, ph_km));
        Exact32 wd_hi = rn_mul(rn_mul(rn_mul(Exact32(0.5f),
                                         rn_add(ww[ix + st], ww[ix])),
                                  rdnw[k]),
                            rn_sub(ph_kp, ph_k));
        Exact32 omega = rn_add(rn_mul(fnm[k], wd_hi),
                            rn_mul(fnp[k], wd_lo));
        tendency = rn_sub(tendency, omega);
    }
    Exact32 mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
    Exact32 gw = rn_mul(rn_mul(mass, G), w[ix]);
    if (has_msf) gw = rn_div(gw, msft[c]);
    rph_t[ix] = rn_add(tendency, gw);
}

void slow_geopotential_open(device Exact32* rph_t,device const Exact32* ww,device const Exact32* w,
    device const Exact32* u,device const Exact32* v,device const Exact32* php,device const Exact32* phb,
    device const Exact32* mup,device const Exact32* mub,device const Exact32* rdnw,device const Exact32* fnm,device const Exact32* fnp,
    device const Exact32* c1f,device const Exact32* c2f,Exact32 cfn,Exact32 cfn1,
    device const Exact32* msft,device const Exact32* msfu,device const Exact32* msfv,
    device const Exact32* mux,device const Exact32* muy,Exact32 quarter_rdx,Exact32 quarter_rdy,
    int mapped,int open_x,int open_y,int supplied,int add_vertical,int order,int base3d,
    int nz,int ny,int nx, uint tid)
{
    ulong st=(ulong)ny*nx;
    if(tid>=(ulong)nz*st)return;
    int k=int(tid/st)+1;ulong c=tid-(ulong)(k-1)*st,ix=(ulong)k*st+c;
    int j=int(c/nx),i=int(c-(ulong)j*nx),top=k==nz;
    Exact32 tendency=(top&&add_vertical)?Exact32(0.0f):rph_t[ix];
    Exact32 mass=rn_add(rn_mul(c1f[k],mut_value(mub,mup,c)),c2f[k]);
    if(add_vertical) {
        if(!top) {
            Exact32 dl=rn_sub(rn_add(rn_sub(php[ix],php[ix-st]),base_value(phb,k,c,st,base3d)),base_value(phb,k-1,c,st,base3d));
            Exact32 dh=rn_sub(rn_add(rn_sub(php[ix+st],php[ix]),base_value(phb,k+1,c,st,base3d)),base_value(phb,k,c,st,base3d));
            Exact32 lo=rn_mul(rn_mul(rn_mul(Exact32(0.5f),rn_add(ww[ix],ww[ix-st])),rdnw[k-1]),dl);
            Exact32 hi=rn_mul(rn_mul(rn_mul(Exact32(0.5f),rn_add(ww[ix+st],ww[ix])),rdnw[k]),dh);
            tendency=rn_sub(tendency,rn_add(rn_mul(fnm[k],hi),rn_mul(fnp[k],lo)));
        }
        Exact32 gw=rn_mul(rn_mul(mass,G),w[ix]);
        tendency=rn_add(tendency,mapped?rn_div(gw,msft[c]):gw);
    }
    for(int axis=0;axis<2;++axis) {
        int pos=axis?i:j,length=axis?nx:ny,boundary=axis?open_x:open_y;
        if(boundary&&(pos==0||pos==length-1))continue;
        int stencil=(order==2||(boundary&&(pos==1||pos==length-2)))?2:
                    ((boundary&&(pos==2||pos==length-3))?4:6);
        Exact32 rd=axis?quarter_rdx:quarter_rdy;
        if(top)rd=rn_mul(Exact32(2.0f),rd);
        Exact32 coefficient=mapped?rn_div(rd,msft[c]):rd;
        Exact32 f0,f1;
        if(axis) {
            f0=supplied?supplied_fcx_value(i,j,k,ny,nx,u,mux,c1f,c2f,msfu,mapped,top,cfn,cfn1):fcx_value(i,j,k,st,ny,nx,u,mup,mub,c1f,c2f,msfu,mapped,top,cfn,cfn1);
            f1=supplied?supplied_fcx_value(i+1,j,k,ny,nx,u,mux,c1f,c2f,msfu,mapped,top,cfn,cfn1):fcx_value(i+1,j,k,st,ny,nx,u,mup,mub,c1f,c2f,msfu,mapped,top,cfn,cfn1);
        } else {
            f0=supplied?supplied_fcy_value(j,i,k,ny,nx,v,muy,c1f,c2f,msfv,mapped,top,cfn,cfn1):fcy_value(j,i,k,st,ny,nx,v,mup,mub,c1f,c2f,msfv,mapped,top,cfn,cfn1);
            f1=supplied?supplied_fcy_value(j+1,i,k,ny,nx,v,muy,c1f,c2f,msfv,mapped,top,cfn,cfn1):fcy_value(j+1,i,k,st,ny,nx,v,mup,mub,c1f,c2f,msfv,mapped,top,cfn,cfn1);
        }
        Exact32 contribution;
        if(stencil==2) {
            int pm=(pos-1+length)%length,pp=(pos+1)%length;
            ulong cm=axis?(ulong)j*nx+pm:(ulong)pm*nx+i;
            ulong cp=axis?(ulong)j*nx+pp:(ulong)pp*nx+i;
            Exact32 dl=rn_sub(rn_add(rn_sub(base_value(phb,k,c,st,base3d),base_value(phb,k,cm,st,base3d)),php[ix]),php[(ulong)k*st+cm]);
            Exact32 dh=rn_sub(rn_add(rn_sub(base_value(phb,k,cp,st,base3d),base_value(phb,k,c,st,base3d)),php[(ulong)k*st+cp]),php[ix]);
            contribution=rn_mul(coefficient,rn_add(rn_mul(f1,dh),rn_mul(f0,dl)));
        } else {
            Exact32 p1=wrf_phi_field_difference(php,axis,i,j,k,1,ny,nx,st,0);
            Exact32 p2=wrf_phi_field_difference(php,axis,i,j,k,2,ny,nx,st,0);
            Exact32 b1=wrf_phi_field_difference(phb,axis,i,j,k,1,ny,nx,st,!base3d);
            Exact32 b2=wrf_phi_field_difference(phb,axis,i,j,k,2,ny,nx,st,!base3d),raw,reciprocal;
            if(stencil==4) {
                raw=rn_sub(rn_add(rn_sub(rn_mul(Exact32(8.0f),p1),p2),rn_mul(Exact32(8.0f),b1)),b2);
                reciprocal=__fdiv_rn(Exact32(1.0f),Exact32(12.0f));
            } else {
                Exact32 p3=wrf_phi_field_difference(php,axis,i,j,k,3,ny,nx,st,0);
                Exact32 b3=wrf_phi_field_difference(phb,axis,i,j,k,3,ny,nx,st,!base3d);
                raw=rn_add(rn_sub(rn_mul(Exact32(45.0f),p1),rn_mul(Exact32(9.0f),p2)),p3);
                raw=rn_add(rn_sub(rn_add(raw,rn_mul(Exact32(45.0f),b1)),rn_mul(Exact32(9.0f),b2)),b3);
                reciprocal=__fdiv_rn(Exact32(1.0f),Exact32(60.0f));
            }
            contribution=rn_mul(coefficient,rn_mul(rn_mul(rn_add(f1,f0),reciprocal),raw));
        }
        tendency=rn_sub(tendency,contribution);
    }
    for(int axis=0;axis<2;++axis) {
        int pos=axis?i:j,length=axis?nx:ny,boundary=axis?open_x:open_y;
        if(!boundary||(pos!=0&&pos!=length-1))continue;
        int west_top=axis&&pos==0&&top;
        int kz=west_top?k:min(k,nz-1);
        Exact32 wl,wh,hm=west_top?Exact32(0.0f):fnm[kz],hp=west_top?Exact32(0.0f):fnp[kz];
        if(axis) {
            ulong face=(ulong)j*(nx+1)+i,plane=(ulong)ny*(nx+1);
            wh=west_top?Exact32(0.0f):rn_add(u[(ulong)kz*plane+face+1],u[(ulong)kz*plane+face]);
            wl=rn_add(u[(ulong)(kz-1)*plane+face+1],u[(ulong)(kz-1)*plane+face]);
        } else {
            ulong face=(ulong)j*nx+i,plane=(ulong)(ny+1)*nx;
            wh=rn_add(v[(ulong)kz*plane+face+nx],v[(ulong)kz*plane+face]);
            wl=rn_add(v[(ulong)(kz-1)*plane+face+nx],v[(ulong)(kz-1)*plane+face]);
        }
        Exact32 speed=rn_mul(Exact32(0.5f),rn_add(rn_mul(hm,wh),rn_mul(hp,wl)));
        speed=pos==0?fminf(speed,Exact32(0.0f)):fmaxf(speed,Exact32(0.0f));
        ulong ci=axis?(pos==0?c+1:c-1):(pos==0?c+nx:c-nx);
        Exact32 diff=pos==0?rn_sub(php[(ulong)k*st+ci],php[ix]):rn_sub(php[ix],php[(ulong)k*st+ci]);
        Exact32 rd=rn_mul(Exact32(4.0f),axis?quarter_rdx:quarter_rdy);
        tendency=rn_sub(tendency,rn_mul(rn_mul(rd,mass),rn_mul(speed,diff)));
    }
    rph_t[ix]=tendency;
}

void slow_geopotential(device Exact32*  rph_t,
                       device const Exact32*  ww,
                       device const Exact32*  w,
                       device const Exact32*  u,
                       device const Exact32*  v,
                       device const Exact32*  php,
                       device const Exact32*  phb,
                       device const Exact32*  mup,
                       device const Exact32*  mub2d,
                       device const Exact32*  rdnw,
                       device const Exact32*  fnm,
                       device const Exact32*  fnp,
                       device const Exact32*  c1f,
                       device const Exact32*  c2f,
                       Exact32 cfn, Exact32 cfn1,
                       device const Exact32*  msft,
                       device const Exact32*  msfu,
                       device const Exact32*  msfv,
                       Exact32 quarter_rdx, Exact32 quarter_rdy,
                       int has_msf, int boundary_x, int boundary_y,
                       int specified, int order, int add_vertical,
                       int base3d, int nz, int ny, int nx, uint tid)
{

    ulong st = (ulong)ny * nx;
    if (tid >= (ulong)nz * st) return;
    int k = (int)(tid / st) + 1;
    ulong c = tid - (ulong)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (ulong)j * nx);
    ulong ix = (ulong)k * st + c;
    int top = (k == nz);
    Exact32 tendency = (top && add_vertical) ? Exact32(0.0f) : rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(Exact32(2.0f), quarter_rdx);
        quarter_rdy = rn_mul(Exact32(2.0f), quarter_rdy);
    }
    if (add_vertical) {
        if (!top) {
            Exact32 ph_km = ph_total(php, phb, k - 1, c, st, base3d);
            Exact32 ph_k = ph_total(php, phb, k, c, st, base3d);
            Exact32 ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
            Exact32 wd_lo = rn_mul(rn_mul(rn_mul(Exact32(0.5f), rn_add(ww[ix], ww[ix - st])), rdnw[k - 1]), rn_sub(rn_add(rn_sub(php[ix], php[ix - st]), base_value(phb, k, c, st, base3d)), base_value(phb, k - 1, c, st, base3d)));
            Exact32 wd_hi = rn_mul(rn_mul(rn_mul(Exact32(0.5f), rn_add(ww[ix + st], ww[ix])), rdnw[k]), rn_sub(rn_add(rn_sub(php[ix + st], php[ix]), base_value(phb, k + 1, c, st, base3d)), base_value(phb, k, c, st, base3d)));
            Exact32 omega = rn_add(rn_mul(fnm[k], wd_hi), rn_mul(fnp[k], wd_lo));
            tendency = rn_sub(tendency, omega);
        }
        Exact32 mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
        Exact32 gw = rn_mul(rn_mul(mass, G), w[ix]);
        if (has_msf) gw = rn_div(gw, msft[c]);
        tendency = rn_add(tendency, gw);
    }
    for (int axis = 0; axis < 2; ++axis) {
        int pos = axis ? i : j;
        int length = axis ? nx : ny;
        int boundary = axis ? boundary_x : boundary_y;
        Exact32 quarter_rd = axis ? quarter_rdx : quarter_rdy;
        Exact32 coefficient = has_msf ? rn_div(quarter_rd, msft[c]) : quarter_rd;
        int stencil = order == 2 ? 2 : 6;
        if (boundary && (pos == 0 || pos == length - 1)) continue;
        if (order != 2 && specified && (pos < 3 || pos >= length - 3)) {
            if (pos == 1 || pos == length - 2) stencil = 2;
            else if (!axis && (pos == 2 || pos == length - 3)) stencil = 4;
            else continue;
        }
        Exact32 face0 = axis ? fcx_value(i, j, k, st, ny, nx, u, mup, mub2d, c1f, c2f, msfu, has_msf, top, cfn, cfn1)
                          : fcy_value(j, i, k, st, ny, nx, v, mup, mub2d, c1f, c2f, msfv, has_msf, top, cfn, cfn1);
        Exact32 face1 = axis ? fcx_value(i + 1, j, k, st, ny, nx, u, mup, mub2d, c1f, c2f, msfu, has_msf, top, cfn, cfn1)
                          : fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d, c1f, c2f, msfv, has_msf, top, cfn, cfn1);
        Exact32 contribution;
        if (stencil == 2) {
            int pm = (pos - 1 + length) % length;
            int pp = (pos + 1) % length;
            ulong cm = axis ? (ulong)j * nx + pm : (ulong)pm * nx + i;
            ulong cp = axis ? (ulong)j * nx + pp : (ulong)pp * nx + i;
            Exact32 d0 = rn_sub(rn_add(rn_sub(base_value(phb, k, c, st, base3d), base_value(phb, k, cm, st, base3d)), php[(ulong)k * st + c]), php[(ulong)k * st + cm]);
            Exact32 d1 = rn_sub(rn_add(rn_sub(base_value(phb, k, cp, st, base3d), base_value(phb, k, c, st, base3d)), php[(ulong)k * st + cp]), php[(ulong)k * st + c]);
            Exact32 sum = rn_add(rn_mul(face1, d1), rn_mul(face0, d0));
            contribution = rn_mul(coefficient, sum);
        } else {
            Exact32 p1 = wrf_phi_field_difference(php, axis, i, j, k, 1, ny, nx, st, 0);
            Exact32 p2 = wrf_phi_field_difference(php, axis, i, j, k, 2, ny, nx, st, 0);
            Exact32 b1 = wrf_phi_field_difference(phb, axis, i, j, k, 1, ny, nx, st, !base3d);
            Exact32 b2 = wrf_phi_field_difference(phb, axis, i, j, k, 2, ny, nx, st, !base3d);
            Exact32 raw;
            Exact32 reciprocal;
            if (stencil == 4) {
                raw = rn_sub(rn_mul(Exact32(8.0f), p1), p2);
                raw = rn_sub(rn_add(raw, rn_mul(Exact32(8.0f), b1)), b2);
                reciprocal = __fdiv_rn(Exact32(1.0f), Exact32(12.0f));
            } else {
                Exact32 p3 = wrf_phi_field_difference(php, axis, i, j, k, 3, ny, nx, st, 0);
                Exact32 b3 = wrf_phi_field_difference(phb, axis, i, j, k, 3, ny, nx, st, !base3d);
                raw = rn_add(rn_sub(rn_mul(Exact32(45.0f), p1), rn_mul(Exact32(9.0f), p2)), p3);
                raw = rn_add(raw, rn_mul(Exact32(45.0f), b1));
                raw = rn_add(rn_sub(raw, rn_mul(Exact32(9.0f), b2)), b3);
                reciprocal = __fdiv_rn(Exact32(1.0f), Exact32(60.0f));
            }
            Exact32 weighted_faces = rn_mul(rn_add(face1, face0), reciprocal);
            contribution = rn_mul(coefficient, rn_mul(weighted_faces, raw));
        }
        tendency = rn_sub(tendency, contribution);
    }
    rph_t[ix] = tendency;
}

void slow_geopotential_faces(device Exact32*  rph_t,
                             device const Exact32*  u,
                             device const Exact32*  v,
                             device const Exact32*  php,
                             device const Exact32*  phb,
                             device const Exact32*  mux,
                             device const Exact32*  muy,
                             device const Exact32*  c1f,
                             device const Exact32*  c2f,
                             Exact32 cfn, Exact32 cfn1,
                             device const Exact32*  msft,
                             device const Exact32*  msfu,
                             device const Exact32*  msfv,
                             Exact32 quarter_rdx, Exact32 quarter_rdy,
                             int has_msf, int boundary_x, int boundary_y,
                             int specified, int order, int base3d,
                             int nz, int ny, int nx, uint tid)
{

    ulong st = (ulong)ny * nx;
    if (tid >= (ulong)nz * st) return;
    int k = (int)(tid / st) + 1;
    ulong c = tid - (ulong)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (ulong)j * nx);
    ulong ix = (ulong)k * st + c;
    int top = (k == nz);
    Exact32 tendency = rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(Exact32(2.0f), quarter_rdx);
        quarter_rdy = rn_mul(Exact32(2.0f), quarter_rdy);
    }
    if (order == 2) {
        Exact32 dphx;
        if (boundary_x && i == 0) {
            dphx = Exact32(0.0f);
        } else {
            int im = (i - 1 + nx) % nx;
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (ulong)j * nx + im, st, base3d));
        }
        Exact32 fx = rn_mul(supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(ulong)j * (nx + 1) + i]);
        Exact32 fxp;
        if (!(boundary_x && i == nx - 1)) {
            int ip = (i + 1) % nx;
            Exact32 dph = rn_sub(ph_total(php, phb, k,
                                       (ulong)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(ulong)j * (nx + 1)
                                       + ((i + 1) % nx)]);
        } else {
            fxp = Exact32(0.0f);
        }
        Exact32 hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        if (boundary_x && (i == 0 || i == nx - 1)) hx = Exact32(0.0f);
        tendency = rn_sub(tendency, hx);
        Exact32 dphy;
        if (boundary_y && j == 0) {
            dphy = Exact32(0.0f);
        } else {
            int jm = (j - 1 + ny) % ny;
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (ulong)jm * nx + i, st, base3d));
        }
        Exact32 fy = rn_mul(supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(ulong)j * nx + i]);
        Exact32 fyp;
        if (!(boundary_y && j == ny - 1)) {
            int jp = (j + 1) % ny;
            Exact32 dph = rn_sub(ph_total(php, phb, k,
                                       (ulong)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fyp = rn_mul(fyp, msfv[(ulong)((j + 1) % ny) * nx + i]);
        } else {
            fyp = Exact32(0.0f);
        }
        Exact32 hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        if (boundary_y && (j == 0 || j == ny - 1)) hy = Exact32(0.0f);
        tendency = rn_sub(tendency, hy);
    } else {
        Exact32 hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
                Exact32 d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (ulong)j * nx + im, st, base3d));
                Exact32 d1 = rn_sub(ph_total(php, phb, k,
                                          (ulong)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                Exact32 f0 = rn_mul(supplied_fcx_value(
                                     i, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d0);
                Exact32 f1 = rn_mul(supplied_fcx_value(
                                     i + 1, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = Exact32(0.0f);
            }
        } else {
            Exact32 csx = rn_add(
                supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1),
                supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1));
            Exact32 diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   Exact32(45.0f), Exact32(-9.0f), Exact32(1.0f), Exact32(60.0f));
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }
        Exact32 hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
                Exact32 d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (ulong)jm * nx + i, st, base3d));
                Exact32 d1 = rn_sub(ph_total(php, phb, k,
                                          (ulong)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                Exact32 f0 = rn_mul(supplied_fcy_value(
                                     j, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d0);
                Exact32 f1 = rn_mul(supplied_fcy_value(
                                     j + 1, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                Exact32 csy = rn_add(
                    supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1),
                    supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1));
                Exact32 diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, Exact32(8.0f), Exact32(-1.0f), Exact32(0.0f), Exact32(12.0f));
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = Exact32(0.0f);
            }
        } else {
            Exact32 csy = rn_add(
                supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1),
                supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1));
            Exact32 diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   Exact32(45.0f), Exact32(-9.0f), Exact32(1.0f), Exact32(60.0f));
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;
}

}

#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
#define O(name) reinterpret_cast<device Exact32 *>(r.name)
#define CALL(ns) \
    if (q.mode == 2) ns::slow_geopotential_vertical(O(rph_t),A(ww),A(w),A(php),A(phb),A(mup),A(mub2d),A(rdnw),A(fnm),A(fnp),A(c1f),A(c2f),A(msft),q.has_msf,q.base3d,q.nz,q.ny,q.nx,tid); \
    else if ((q.open_x || q.open_y) && !q.specified) ns::slow_geopotential_open(O(rph_t),A(ww),A(w),A(u),A(v),A(php),A(phb),A(mup),A(mub2d),A(rdnw),A(fnm),A(fnp),A(c1f),A(c2f),Exact32(q.cfn),Exact32(q.cfn1),A(msft),A(msfu),A(msfv),A(mux),A(muy),Exact32(q.quarter_rdx),Exact32(q.quarter_rdy),q.has_msf,q.open_x,q.open_y,q.mode==1,q.add_vertical,q.order,q.base3d,q.nz,q.ny,q.nx,tid); \
    else if (q.mode == 1) ns::slow_geopotential_faces(O(rph_t),A(u),A(v),A(php),A(phb),A(mux),A(muy),A(c1f),A(c2f),Exact32(q.cfn),Exact32(q.cfn1),A(msft),A(msfu),A(msfv),Exact32(q.quarter_rdx),Exact32(q.quarter_rdy),q.has_msf,q.boundary_x,q.boundary_y,q.specified,q.order,q.base3d,q.nz,q.ny,q.nx,tid); \
    else ns::slow_geopotential(O(rph_t),A(ww),A(w),A(u),A(v),A(php),A(phb),A(mup),A(mub2d),A(rdnw),A(fnm),A(fnp),A(c1f),A(c2f),Exact32(q.cfn),Exact32(q.cfn1),A(msft),A(msfu),A(msfv),Exact32(q.quarter_rdx),Exact32(q.quarter_rdy),q.has_msf,q.boundary_x,q.boundary_y,q.specified,q.order,q.add_vertical,q.base3d,q.nz,q.ny,q.nx,tid)
kernel void wrf_exact_slow_geopotential(constant WrfExactSlowGeopotentialResources &r [[buffer(0)]], constant WrfExactSlowGeopotentialParams &q [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (ulong(tid)>=ulong(q.nz)*q.ny*q.nx) return;
    if (q.bigstep) { CALL(sg1); } else { CALL(sg0); }
}
#undef CALL
#undef O
#undef A
