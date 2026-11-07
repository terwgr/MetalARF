#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#define E2(k,e,ne) ((k)*(ne)+(e))
#define C2(k,c,nc) ((k)*(nc)+(c))
#define DYNAMICS_ELEMENT_LOOP(total) for(int element=int(gid % p.dispatch_x); element<(total);element+=p.dispatch_x)

#include "hexv841_rrfs_dynamics_vector_momentum_rrfs_f32_bindings.metalh"
kernel void hexv841_rrfs_dynamics_vector_momentum_rrfs_f32(constant Hexv841RrfsDynamicsVectorMomentumRrfsF32Resources &r [[buffer(0)]],
    constant Hexv841RrfsDynamicsVectorMomentumRrfsF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges2 = p.max_edges2;
    device const float *u = r.u;
    device const float *rho_edge = r.rho_edge;
    device const float *pv_edge = r.pv_edge;
    device const float *kinetic = r.kinetic;
    device const float *mass_divergence = r.mass_divergence;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *edges_on_edge = r.edges_on_edge;
    device const int *n_edges_on_edge = r.n_edges_on_edge;
    device const float *weights_on_edge = r.weights_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *initial_tendency = r.initial_tendency;
    const int has_initial = p.has_initial;
    device float *result = r.result;
    DYNAMICS_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float q = 0.0f;
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            const float work_pv = mpas_mul(0.5f, mpas_add(
                pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
            q = mpas_add(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], u[E2(k, neighbor, nedges)]), work_pv));
        }
        const float gradient = mpas_mul(mpas_sub(
            kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]),
            inv_dc_edge[edge]);
        const float horizontal = mpas_mul(rho_edge[index], mpas_sub(q, gradient));
        result[index] = mpas_sub(has_initial ? mpas_add(initial_tendency[index], horizontal) : horizontal,
            mpas_mul(mpas_mul(u[index], 0.5f),
                mpas_add(mass_divergence[C2(k, c0, ncells)],
                    mass_divergence[C2(k, c1, ncells)])));
    }
}


#include "hexv841_rrfs_dynamics_euler_w_accumulate_rrfs_f32_bindings.metalh"
kernel void hexv841_rrfs_dynamics_euler_w_accumulate_rrfs_f32(constant Hexv841RrfsDynamicsEulerWAccumulateRrfsF32Resources &r [[buffer(0)]],
    constant Hexv841RrfsDynamicsEulerWAccumulateRrfsF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nl = p.nl;
    const int nc = p.nc;
    const int solve = p.solve;
    device const float *pp = r.pp;
    device const float *dpdz = r.dpdz;
    device const float *cqw = r.cqw;
    device const float *rdzu = r.rdzu;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device float *tend = r.tend;
    DYNAMICS_ELEMENT_LOOP((nl-1)*solve) {
        const int c=element%solve, k=1+element/solve;
        const float pressure=mpas_mul(rdzu[k],mpas_sub(pp[C2(k,c,nc)],pp[C2(k-1,c,nc)]));
        const float buoyancy=mpas_add(mpas_mul(fzm[k],dpdz[C2(k,c,nc)]),
                                      mpas_mul(fzp[k],dpdz[C2(k-1,c,nc)]));
        tend[C2(k,c,nc)]=mpas_sub(tend[C2(k,c,nc)],mpas_mul(cqw[C2(k,c,nc)],mpas_sub(pressure,buoyancy)));
    }
}
