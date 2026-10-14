#include <metal_stdlib>
#include "case_boundary_mass_flux_bindings.metalh"

using namespace metal;

// Keep the default FP32 expression order aligned with
// woof.core.dycore._boundary_mass_tendency_flat.  The host sums these
// independent per-level boundary contributions in FP64.
#pragma clang fp contract(off)

kernel void case_boundary_mass_flux(
    constant CaseBoundaryMassFluxResources &r [[buffer(0)]],
    constant CaseBoundaryMassFluxParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const ulong row_width=ulong(p.nedge);
    const ulong count=ulong(p.nz)*row_width;
    if(ulong(flat)>=count)return;
    if(p.nedge!=p.ny+p.nx)return;

    const int k=int(ulong(flat)/row_width);
    const int local=int(ulong(flat)%row_width);
    float tendency=0.0f;

    if(local<p.ny) {
        if(p.open_x) {
            const int j=local;
            const ulong west_mu=ulong(j)*ulong(p.nx);
            const ulong east_mu=west_mu+ulong(p.nx-1);
            const ulong west_face=(ulong(k)*ulong(p.ny)+ulong(j))
                                  *ulong(p.nx+1);
            const ulong east_face=west_face+ulong(p.nx);
            float west_reference=(r.c1h[k]*r.mu[west_mu]+r.c2h[k])
                                 *r.u[west_face];
            float east_reference=(r.c1h[k]*r.mu[east_mu]+r.c2h[k])
                                 *r.u[east_face];
            if(p.has_msf) {
                west_reference/=r.msfu[ulong(j)*ulong(p.nx+1)];
                east_reference/=r.msfu[ulong(j)*ulong(p.nx+1)
                                       +ulong(p.nx)];
            }
            const float west=r.u_pp[west_face]+west_reference;
            const float east=r.u_pp[east_face]+east_reference;
            tendency=(r.dnw[k]*p.rdx)*(east-west);
        }
    } else if(p.open_y) {
        const int i=local-p.ny;
        const ulong south_mu=ulong(i);
        const ulong north_mu=ulong(p.ny-1)*ulong(p.nx)+ulong(i);
        const ulong south_face=ulong(k)*ulong(p.ny+1)*ulong(p.nx)
                               +ulong(i);
        const ulong north_face=south_face+ulong(p.ny)*ulong(p.nx);
        float south_reference=(r.c1h[k]*r.mu[south_mu]+r.c2h[k])
                              *r.v[south_face];
        float north_reference=(r.c1h[k]*r.mu[north_mu]+r.c2h[k])
                              *r.v[north_face];
        if(p.has_msf) {
            south_reference/=r.msfv[ulong(i)];
            north_reference/=r.msfv[ulong(p.ny)*ulong(p.nx)+ulong(i)];
        }
        const float south=r.v_pp[south_face]+south_reference;
        const float north=r.v_pp[north_face]+north_reference;
        tendency=(r.dnw[k]*p.rdy)*(north-south);
    }

    r.out[flat]=tendency;
}
