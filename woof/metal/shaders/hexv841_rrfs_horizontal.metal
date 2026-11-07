#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#include "hex_rrfs_native_powf.metalh"


#define HORIZONTAL_ELEMENT_LOOP(total) \
    for (int element = int(gid % p.dispatch_x); \
         element < (total); element += p.dispatch_x)

static inline int lidx_v841(int level, int entity, int count) {
    return level * count + entity;
}

static inline float cell_sign_v841(
    int cell, int edge, device const int *cells_on_edge)
{
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}

static inline float vertex_sign_v841(
    int vertex_id, int edge, device const int *vertices_on_edge)
{
    return vertices_on_edge[2 * edge] == vertex_id ? -1.0f : 1.0f;
}

static inline int kite_slot_v841(
    int vertex_id, int cell, device const int *cells_on_vertex, int vertex_degree)
{
    for (int slot = 0; slot < vertex_degree; ++slot) {
        if (cells_on_vertex[vertex_id * vertex_degree + slot] == cell) return slot;
    }
    return -1;
}


#include "hexv841_rrfs_horizontal_vertex_diagnostics_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_vertex_diagnostics_v841_f32(constant Hexv841RrfsHorizontalVertexDiagnosticsV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalVertexDiagnosticsV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *normal_velocity = r.normal_velocity;
    device const float *ke_edge = r.ke_edge;
    device const int *edges_on_vertex = r.edges_on_vertex;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *inv_area_triangle = r.inv_area_triangle;
    device const float *f_vertex = r.f_vertex;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int vertex_degree = p.vertex_degree;
    device float *vorticity = r.vorticity;
    device float *ke_vertex = r.ke_vertex;
    device float *pv_vertex = r.pv_vertex;
    HORIZONTAL_ELEMENT_LOOP(nlev * nvertices) {
        const int vertex_id = element % nvertices;
        const int k = element / nvertices;
        float vort = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex_id * vertex_degree + slot];
            if (edge < 0) continue;
            vort = mpas_add(vort, mpas_mul(mpas_mul(
                vertex_sign_v841(vertex_id, edge, vertices_on_edge), dc_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, ke_edge[lidx_v841(k, edge, nedges)]);
        }
        vort = mpas_mul(vort, inv_area_triangle[vertex_id]);
        const float r = mpas_mul(0.25f, inv_area_triangle[vertex_id]);
        ke = mpas_mul(ke, r);
        const int index = lidx_v841(k, vertex_id, nvertices);
        vorticity[index] = vort;
        ke_vertex[index] = ke;
        pv_vertex[index] = mpas_add(f_vertex[vertex_id], vort);
    }
}

#ifndef HEX_HOIST_EOE
#define HEX_HOIST_EOE 10
#endif


#include "hexv841_rrfs_horizontal_tangential_velocity_hoist_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_tangential_velocity_hoist_v841_f32(constant Hexv841RrfsHorizontalTangentialVelocityHoistV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalTangentialVelocityHoistV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *normal_velocity = r.normal_velocity;
    device const int *edges_on_edge = r.edges_on_edge;
    device const int *n_edges_on_edge = r.n_edges_on_edge;
    device const float *weights_on_edge = r.weights_on_edge;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int max_edges2 = p.max_edges2;
    device float *tangential_velocity = r.tangential_velocity;
    const int lpt = p.lpt;
    const int hex_rows = nlev;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = int(gid % p.dispatch_x);
    if (hex_element >= nedges * hex_chunks) return;
    const int edge = hex_element % nedges;
    const int hex_k0 = (hex_element / nedges) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int count = n_edges_on_edge[edge];
    if (count > HEX_HOIST_EOE) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int offset = edge * max_edges2 + slot;
                const int neighbor = edges_on_edge[offset];
                value = mpas_add(value, mpas_mul(weights_on_edge[offset],
                    normal_velocity[lidx_v841(k, neighbor, nedges)]));
            }
            tangential_velocity[lidx_v841(k, edge, nedges)] = value;
        }
        return;
    }
    int nb[HEX_HOIST_EOE];
    float wt[HEX_HOIST_EOE];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
        nb[slot] = slot < count ? edges_on_edge[edge * max_edges2 + slot] : edge;
        wt[slot] = slot < count ? weights_on_edge[edge * max_edges2 + slot] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
            if (slot < count) {
                value = mpas_add(value, mpas_mul(wt[slot],
                    normal_velocity[lidx_v841(k, nb[slot], nedges)]));
            }
        }
        tangential_velocity[lidx_v841(k, edge, nedges)] = value;
    }
}


#include "hexv841_rrfs_horizontal_cell_diagnostics_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_cell_diagnostics_v841_f32(constant Hexv841RrfsHorizontalCellDiagnosticsV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalCellDiagnosticsV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *normal_velocity = r.normal_velocity;
    device const float *ke_edge = r.ke_edge;
    device const float *ke_vertex = r.ke_vertex;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_cell = r.vertices_on_cell;
    device const int *cells_on_vertex = r.cells_on_vertex;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *kite_areas = r.kite_areas;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int max_edges = p.max_edges;
    const int vertex_degree = p.vertex_degree;
    device float *divergence = r.divergence;
    device float *kinetic_energy = r.kinetic_energy;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const float inv_area = inv_area_cell[cell];
        const float ke_fact = 0.625f;
        float div = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            div = mpas_add(div, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
        for (int slot = 0; slot < count; ++slot) {
            const int vertex_id = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex_id, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                    kite_areas[vertex_id * vertex_degree + kite]);
                correction = mpas_mul(correction,
                    ke_vertex[lidx_v841(k, vertex_id, nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        divergence[index] = div;
        kinetic_energy[index] = ke;
    }
}


#include "hexv841_rrfs_horizontal_laplacian_divergence_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_laplacian_divergence_v841_f32(constant Hexv841RrfsHorizontalLaplacianDivergenceV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalLaplacianDivergenceV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *delsq_u = r.delsq_u;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device float *delsq_divergence = r.delsq_divergence;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(inv_area_cell[cell], dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), delsq_u[lidx_v841(k, edge, nedges)]));
        }
        delsq_divergence[lidx_v841(k, cell, ncells)] = value;
    }
}



#include "hexv841_rrfs_horizontal_laplacian_vorticity_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_laplacian_vorticity_v841_f32(constant Hexv841RrfsHorizontalLaplacianVorticityV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalLaplacianVorticityV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *delsq_u = r.delsq_u;
    device const int *edges_on_vertex = r.edges_on_vertex;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *inv_area_triangle = r.inv_area_triangle;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int vertex_degree = p.vertex_degree;
    device float *delsq_vorticity = r.delsq_vorticity;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * nvertices) return;
    const int vertex_id = element % nvertices;
    const int k = element / nvertices;
    { 
        float value = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex_id * vertex_degree + slot];
            if (edge < 0) continue;
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(inv_area_triangle[vertex_id], dc_edge[edge]),
                vertex_sign_v841(vertex_id, edge, vertices_on_edge)), delsq_u[lidx_v841(k, edge, nedges)]));
        }
        delsq_vorticity[lidx_v841(k, vertex_id, nvertices)] = value;
    }
}



#include "hexv841_rrfs_horizontal_theta_filter_lap2_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_theta_filter_lap2_v841_f32(constant Hexv841RrfsHorizontalThetaFilterLap2V841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalThetaFilterLap2V841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *theta_m = r.theta_m;
    device const float *rho_edge = r.rho_edge;
    device const float *kdiff = r.kdiff;
    device const float *del2 = r.del2;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device float *delsq_theta = r.delsq_theta;
    device float *tendency = r.tendency;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(inv_area_cell[cell],
                cell_sign_v841(cell, edge, cells_on_edge)), dv_edge[edge]),
                inv_dc_edge[edge]);
            const float flux = mpas_mul(mpas_mul(factor, mpas_sub(
                theta_m[lidx_v841(k, cell1, ncells)],
                theta_m[lidx_v841(k, cell0, ncells)])),
                rho_edge[lidx_v841(k, edge, nedges)]);
            lap = mpas_add(lap, flux);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, 0.5f), mpas_add(
                    kdiff[lidx_v841(k, cell0, ncells)],
                    kdiff[lidx_v841(k, cell1, ncells)])), del2[edge]));
        }
        const int tid = lidx_v841(k, cell, ncells);
        delsq_theta[tid] = lap;
        tendency[tid] = tend;
    }
}



#include "hexv841_rrfs_horizontal_theta_filter_lap4_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_theta_filter_lap4_v841_f32(constant Hexv841RrfsHorizontalThetaFilterLap4V841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalThetaFilterLap4V841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *delsq_theta = r.delsq_theta;
    device const float *del4 = r.del4;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const float h4 = p.h4;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int max_edges = p.max_edges;
    device float *tendency = r.tendency;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(del4[edge],
                mpas_mul(h4, inv_area_cell[cell])), dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), inv_dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_theta[lidx_v841(k, cell1, ncells)],
                delsq_theta[lidx_v841(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}



#include "hexv841_rrfs_horizontal_w_filter_lap2_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_w_filter_lap2_v841_f32(constant Hexv841RrfsHorizontalWFilterLap2V841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalWFilterLap2V841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *vertical_velocity = r.vertical_velocity;
    device const float *rho_edge = r.rho_edge;
    device const float *kdiff = r.kdiff;
    device const float *del2 = r.del2;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device float *delsq_w = r.delsq_w;
    device float *tendency = r.tendency;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        if (k == 0) {
            delsq_w[tid] = 0.0f;
            tendency[tid] = 0.0f;
            continue;
        }
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f,
                inv_area_cell[cell]),
                cell_sign_v841(cell, edge, cells_on_edge)), dv_edge[edge]),
                inv_dc_edge[edge]);
            const float flux = mpas_mul(mpas_mul(factor, mpas_add(
                rho_edge[lidx_v841(k, edge, nedges)],
                rho_edge[lidx_v841(k - 1, edge, nedges)])), mpas_sub(
                    vertical_velocity[lidx_v841(k, cell1, ncells)],
                    vertical_velocity[lidx_v841(k, cell0, ncells)]));
            lap = mpas_add(lap, flux);
            const float sum_k = mpas_add(mpas_add(mpas_add(
                kdiff[lidx_v841(k, cell0, ncells)],
                kdiff[lidx_v841(k, cell1, ncells)]),
                kdiff[lidx_v841(k - 1, cell0, ncells)]),
                kdiff[lidx_v841(k - 1, cell1, ncells)]);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, del2[edge]), 0.25f), sum_k));
        }
        delsq_w[tid] = lap;
        tendency[tid] = tend;
    }
}



#include "hexv841_rrfs_horizontal_w_filter_lap4_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_w_filter_lap4_v841_f32(constant Hexv841RrfsHorizontalWFilterLap4V841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalWFilterLap4V841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *delsq_w = r.delsq_w;
    device const float *del4 = r.del4;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const float h4 = p.h4;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int max_edges = p.max_edges;
    device float *tendency = r.tendency;
    
    HORIZONTAL_ELEMENT_LOOP((nlev - 1) * ncells) {
        const int cell = element % ncells;
        const int k = 1 + element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(del4[edge],
                mpas_mul(h4, inv_area_cell[cell])), dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), inv_dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_w[lidx_v841(k, cell1, ncells)],
                delsq_w[lidx_v841(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}



#include "hexv841_rrfs_horizontal_momentum_filter_lap4_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_momentum_filter_lap4_v841_f32(constant Hexv841RrfsHorizontalMomentumFilterLap4V841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalMomentumFilterLap4V841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *rho_edge = r.rho_edge;
    device const float *delsq_divergence = r.delsq_divergence;
    device const float *delsq_vorticity = r.delsq_vorticity;
    device const float *del4 = r.del4;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *inv_dv_edge = r.inv_dv_edge;
    const float h4 = p.h4;
    const float div_factor = p.div_factor;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    device float *tendency = r.tendency;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const float inv_dc = inv_dc_edge[edge];
    const float inv_dv = fct_min(inv_dv_edge[edge], 4.0f * inv_dc);
    const float scale = mpas_mul(del4[edge], h4);
    
    const float r_dc = mpas_mul(mpas_mul(scale, div_factor), inv_dc);
    const float r_dv = mpas_mul(scale, inv_dv);
    { 
        const int tid = lidx_v841(k, edge, nedges);
        const float diffusion = mpas_mul(rho_edge[tid], mpas_sub(
            mpas_mul(mpas_sub(
                delsq_divergence[lidx_v841(k, cell1, ncells)],
                delsq_divergence[lidx_v841(k, cell0, ncells)]), r_dc),
            mpas_mul(mpas_sub(
                delsq_vorticity[lidx_v841(k, vertex1, nvertices)],
                delsq_vorticity[lidx_v841(k, vertex0, nvertices)]), r_dv)));
        tendency[tid] = mpas_sub(tendency[tid], diffusion);
    }
}



#include "hexv841_rrfs_horizontal_momentum_filter_seed_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_momentum_filter_seed_v841_f32(constant Hexv841RrfsHorizontalMomentumFilterSeedV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalMomentumFilterSeedV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *rho_edge = r.rho_edge;
    device const float *divergence = r.divergence;
    device const float *vorticity = r.vorticity;
    device const float *kdiff = r.kdiff;
    device const float *del2 = r.del2;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *inv_dv_edge = r.inv_dv_edge;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    device float *delsq_u = r.delsq_u;
    device float *tendency = r.tendency;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    const int c0 = cells_on_edge[2*edge], c1 = cells_on_edge[2*edge+1];
    const int v0 = vertices_on_edge[2*edge], v1 = vertices_on_edge[2*edge+1];
    const float inv_dc = inv_dc_edge[edge];
    const float inv_dv = fct_min(inv_dv_edge[edge], mpas_mul(4.0f, inv_dc));
    { 
        const int tid=lidx_v841(k,edge,nedges);
        const float lap=mpas_sub(mpas_mul(mpas_sub(divergence[lidx_v841(k,c1,ncells)],divergence[lidx_v841(k,c0,ncells)]),inv_dc),mpas_mul(mpas_sub(vorticity[lidx_v841(k,v1,nvertices)],vorticity[lidx_v841(k,v0,nvertices)]),inv_dv));
        const float kedge=mpas_mul(0.5f,mpas_add(kdiff[lidx_v841(k,c0,ncells)],kdiff[lidx_v841(k,c1,ncells)]));
        delsq_u[tid]=lap;
        
        
        tendency[tid]=mpas_add(tendency[tid],mpas_mul(mpas_mul(mpas_mul(rho_edge[tid],kedge),lap),del2[edge]));
    }
}



#include "hexv841_rrfs_horizontal_smagorinsky_rrfs_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_smagorinsky_rrfs_f32(constant Hexv841RrfsHorizontalSmagorinskyRrfsF32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalSmagorinskyRrfsF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *u = r.u;
    device const float *v = r.v;
    device const int *edges = r.edges;
    device const int *counts = r.counts;
    device const float *defc_a = r.defc_a;
    device const float *defc_b = r.defc_b;
    const float strain_scale = p.strain_scale;
    const float ceiling = p.ceiling;
    const int nl = p.nl;
    const int nc = p.nc;
    const int ne = p.ne;
    const int me = p.me;
    device float *kdiff = r.kdiff;
    HORIZONTAL_ELEMENT_LOOP(nl * nc) {
        const int c = element % nc, k = element / nc;
        float diag = 0.0f, off = 0.0f;
        for (int j = 0; j < counts[c]; ++j) {
            const int o = c * me + j, e = edges[o];
            const float un = u[k * ne + e], vn = v[k * ne + e];
            diag = mpas_sub(mpas_add(diag, mpas_mul(defc_a[o], un)), mpas_mul(defc_b[o], vn));
            off = mpas_add(mpas_add(off, mpas_mul(defc_b[o], un)), mpas_mul(defc_a[o], vn));
        }
        kdiff[k * nc + c] = fct_min(mpas_mul(strain_scale,
            mpas_sqrt(mpas_add(mpas_mul(diag,diag),mpas_mul(off,off)))),ceiling);
    }
}


#include "hexv841_rrfs_horizontal_pv_cell_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_pv_cell_v841_f32(constant Hexv841RrfsHorizontalPvCellV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalPvCellV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *pv_vertex = r.pv_vertex;
    device const int *vertices_on_cell = r.vertices_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_vertex = r.cells_on_vertex;
    device const float *kite_areas = r.kite_areas;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nvertices = p.nvertices;
    const int max_edges = p.max_edges;
    const int vertex_degree = p.vertex_degree;
    device float *pv_cell = r.pv_cell;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int vertex_id = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex_id, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float contribution = mpas_mul(
                    kite_areas[vertex_id * vertex_degree + kite],
                    pv_vertex[lidx_v841(k, vertex_id, nvertices)]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                value = mpas_add(value, contribution);
            }
        }
        pv_cell[lidx_v841(k, cell, ncells)] = value;
    }
}


#include "hexv841_rrfs_horizontal_pv_apvm_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_pv_apvm_v841_f32(constant Hexv841RrfsHorizontalPvApvmV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalPvApvmV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *normal_velocity = r.normal_velocity;
    device const float *tangential_velocity = r.tangential_velocity;
    device const float *pv_vertex = r.pv_vertex;
    device const float *pv_cell = r.pv_cell;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *inv_dv_edge = r.inv_dv_edge;
    const float scale = p.scale;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    device float *pv_edge = r.pv_edge;
    device float *grad_pv_normal = r.grad_pv_normal;
    device float *grad_pv_tangential = r.grad_pv_tangential;
    HORIZONTAL_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int vertex0 = vertices_on_edge[2 * edge];
        const int vertex1 = vertices_on_edge[2 * edge + 1];
        const int cell0 = cells_on_edge[2 * edge];
        const int cell1 = cells_on_edge[2 * edge + 1];
        const int index = lidx_v841(k, edge, nedges);
        const float grad_t = mpas_mul(mpas_sub(
            pv_vertex[lidx_v841(k, vertex1, nvertices)],
            pv_vertex[lidx_v841(k, vertex0, nvertices)]), inv_dv_edge[edge]);
        const float grad_n = mpas_mul(mpas_sub(
            pv_cell[lidx_v841(k, cell1, ncells)],
            pv_cell[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        grad_pv_tangential[index] = grad_t;
        grad_pv_normal[index] = grad_n;
        pv_edge[index] = mpas_sub(pv_edge[index], mpas_mul(scale, mpas_add(
            mpas_mul(tangential_velocity[index], grad_t),
            mpas_mul(normal_velocity[index], grad_n))));
    }
}


#include "hexv841_rrfs_horizontal_mass_flux_divergence_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_mass_flux_divergence_v841_f32(constant Hexv841RrfsHorizontalMassFluxDivergenceV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalMassFluxDivergenceV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *rho_u = r.rho_u;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device float *mass_divergence = r.mass_divergence;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                rho_u[lidx_v841(k, edge, nedges)]));
        }
        mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(
            value, inv_area_cell[cell]);
    }
}


#include "hexv841_rrfs_horizontal_pressure_gradient_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_pressure_gradient_v841_f32(constant Hexv841RrfsHorizontalPressureGradientV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalPressureGradientV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *pressure_p = r.pressure_p;
    device const float *dpdz = r.dpdz;
    device const float *cqu = r.cqu;
    device const float *zz = r.zz;
    device const float *zxu = r.zxu;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    device float *tendency = r.tendency;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    { 
        const int index = lidx_v841(k, edge, nedges);
        float normal = mpas_mul(mpas_sub(
            pressure_p[lidx_v841(k, cell1, ncells)],
            pressure_p[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        normal = mpas_div(normal, mpas_mul(0.5f, mpas_add(
            zz[lidx_v841(k, cell1, ncells)],
            zz[lidx_v841(k, cell0, ncells)])));
        const float terrain = mpas_mul(mpas_mul(0.5f, zxu[index]), mpas_add(
            dpdz[lidx_v841(k, cell0, ncells)],
            dpdz[lidx_v841(k, cell1, ncells)]));
        tendency[index] = mpas_mul(mpas_sub(0.0f, cqu[index]),
            mpas_sub(normal, terrain));
    }
}

#ifndef HEX_HOIST_CELL
#define HEX_HOIST_CELL 10
#endif
#define HEX_CELL_CHUNK(rows) \
    const int hex_rows = (rows); \
    const int hex_chunks = (hex_rows + lpt - 1) / lpt; \
    const int hex_element = int(gid % p.dispatch_x); \
    if (hex_element >= ncells * hex_chunks) return; \
    const int cell = hex_element % ncells; \
    const int hex_k0 = (hex_element / ncells) * lpt; \
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);


#include "hexv841_rrfs_horizontal_cell_diagnostics_chunk_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_cell_diagnostics_chunk_v841_f32(constant Hexv841RrfsHorizontalCellDiagnosticsChunkV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalCellDiagnosticsChunkV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *normal_velocity = r.normal_velocity;
    device const float *ke_edge = r.ke_edge;
    device const float *ke_vertex = r.ke_vertex;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_cell = r.vertices_on_cell;
    device const int *cells_on_vertex = r.cells_on_vertex;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *kite_areas = r.kite_areas;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int max_edges = p.max_edges;
    const int vertex_degree = p.vertex_degree;
    device float *divergence = r.divergence;
    device float *kinetic_energy = r.kinetic_energy;
    const int lpt = p.lpt;
    HEX_CELL_CHUNK(nlev)
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    const float ke_fact = 0.625f;
    if (count > HEX_HOIST_CELL) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float div = 0.0f;
            float ke = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int edge = edges_on_cell[cell * max_edges + slot];
                div = mpas_add(div, mpas_mul(mpas_mul(
                    cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                    normal_velocity[lidx_v841(k, edge, nedges)]));
                ke = mpas_add(ke, mpas_mul(
                    0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
            }
            div = mpas_mul(div, inv_area);
            ke = mpas_mul(ke, inv_area);
            ke = mpas_mul(ke, ke_fact);
            for (int slot = 0; slot < count; ++slot) {
                const int vertex_id = vertices_on_cell[cell * max_edges + slot];
                const int kite = kite_slot_v841(
                    vertex_id, cell, cells_on_vertex, vertex_degree);
                if (kite >= 0) {
                    float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                        kite_areas[vertex_id * vertex_degree + kite]);
                    correction = mpas_mul(correction,
                        ke_vertex[lidx_v841(k, vertex_id, nvertices)]);
                    correction = mpas_mul(correction, inv_area);
                    ke = mpas_add(ke, correction);
                }
            }
            const int index = lidx_v841(k, cell, ncells);
            divergence[index] = div;
            kinetic_energy[index] = ke;
        }
        return;
    }
    int ed[HEX_HOIST_CELL];
    float sdv[HEX_HOIST_CELL];
    int vx[HEX_HOIST_CELL];
    float kc[HEX_HOIST_CELL];
    bool kv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = slot < count;
        const int edge = live ? edges_on_cell[cell * max_edges + slot] : 0;
        ed[slot] = edge;
        sdv[slot] = live ? mpas_mul(cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]) : 0.0f;
        const int vertex_id = live ? vertices_on_cell[cell * max_edges + slot] : 0;
        vx[slot] = vertex_id;
        const int kite = live ? kite_slot_v841(vertex_id, cell, cells_on_vertex, vertex_degree) : -1;
        kv[slot] = kite >= 0;
        kc[slot] = kite >= 0 ? mpas_mul(mpas_sub(1.0f, ke_fact),
            kite_areas[vertex_id * vertex_degree + kite]) : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float div = 0.0f;
        float ke = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count) {
                div = mpas_add(div, mpas_mul(sdv[slot],
                    normal_velocity[lidx_v841(k, ed[slot], nedges)]));
                ke = mpas_add(ke, mpas_mul(
                    0.25f, ke_edge[lidx_v841(k, ed[slot], nedges)]));
            }
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count && kv[slot]) {
                float correction = mpas_mul(kc[slot],
                    ke_vertex[lidx_v841(k, vx[slot], nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        divergence[index] = div;
        kinetic_energy[index] = ke;
    }
}


#include "hexv841_rrfs_horizontal_pv_cell_chunk_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_pv_cell_chunk_v841_f32(constant Hexv841RrfsHorizontalPvCellChunkV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalPvCellChunkV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *pv_vertex = r.pv_vertex;
    device const int *vertices_on_cell = r.vertices_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_vertex = r.cells_on_vertex;
    device const float *kite_areas = r.kite_areas;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nvertices = p.nvertices;
    const int max_edges = p.max_edges;
    const int vertex_degree = p.vertex_degree;
    device float *pv_cell = r.pv_cell;
    const int lpt = p.lpt;
    HEX_CELL_CHUNK(nlev)
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    if (count > HEX_HOIST_CELL) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int vertex_id = vertices_on_cell[cell * max_edges + slot];
                const int kite = kite_slot_v841(
                    vertex_id, cell, cells_on_vertex, vertex_degree);
                if (kite >= 0) {
                    float contribution = mpas_mul(
                        kite_areas[vertex_id * vertex_degree + kite],
                        pv_vertex[lidx_v841(k, vertex_id, nvertices)]);
                    contribution = mpas_mul(contribution, inv_area);
                    value = mpas_add(value, contribution);
                }
            }
            pv_cell[lidx_v841(k, cell, ncells)] = value;
        }
        return;
    }
    int vx[HEX_HOIST_CELL];
    float ka[HEX_HOIST_CELL];
    bool kv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = slot < count;
        const int vertex_id = live ? vertices_on_cell[cell * max_edges + slot] : 0;
        vx[slot] = vertex_id;
        const int kite = live ? kite_slot_v841(vertex_id, cell, cells_on_vertex, vertex_degree) : -1;
        kv[slot] = kite >= 0;
        ka[slot] = kite >= 0 ? kite_areas[vertex_id * vertex_degree + kite] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count && kv[slot]) {
                float contribution = mpas_mul(ka[slot],
                    pv_vertex[lidx_v841(k, vx[slot], nvertices)]);
                contribution = mpas_mul(contribution, inv_area);
                value = mpas_add(value, contribution);
            }
        }
        pv_cell[lidx_v841(k, cell, ncells)] = value;
    }
}


#include "hexv841_rrfs_horizontal_mass_flux_divergence_chunk_v841_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_mass_flux_divergence_chunk_v841_f32(constant Hexv841RrfsHorizontalMassFluxDivergenceChunkV841F32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalMassFluxDivergenceChunkV841F32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const float *rho_u = r.rho_u;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device float *mass_divergence = r.mass_divergence;
    const int lpt = p.lpt;
    HEX_CELL_CHUNK(nlev)
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    if (count > HEX_HOIST_CELL) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int edge = edges_on_cell[cell * max_edges + slot];
                value = mpas_add(value, mpas_mul(mpas_mul(
                    cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                    rho_u[lidx_v841(k, edge, nedges)]));
            }
            mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(value, inv_area);
        }
        return;
    }
    int ed[HEX_HOIST_CELL];
    float sdv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = slot < count;
        const int edge = live ? edges_on_cell[cell * max_edges + slot] : 0;
        ed[slot] = edge;
        sdv[slot] = live ? mpas_mul(cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]) : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count) {
                value = mpas_add(value, mpas_mul(sdv[slot],
                    rho_u[lidx_v841(k, ed[slot], nedges)]));
            }
        }
        mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(value, inv_area);
    }
}
#include "hexv841_rrfs_horizontal_mixing_scaling_f32_bindings.metalh"
kernel void hexv841_rrfs_horizontal_mixing_scaling_f32(constant Hexv841RrfsHorizontalMixingScalingF32Resources &r [[buffer(0)]],
    constant Hexv841RrfsHorizontalMixingScalingF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *mesh_density = r.mesh_density;
    const int nedges = p.nedges;
    const int scale_with_mesh = p.scale_with_mesh;
    device float *del2 = r.del2;
    device float *del4 = r.del4;
    const int edge = int(gid % p.dispatch_x);
    if (edge >= nedges) return;
    if (!scale_with_mesh) {
        del2[edge] = 1.0f;
        del4[edge] = 1.0f;
        return;
    }
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const float mean_density = 0.5f * (mesh_density[cell0] + mesh_density[cell1]);
    del2[edge] = 1.0f / rrfs_native_powf(mean_density, 0.25f);
    del4[edge] = 1.0f / rrfs_native_powf(mean_density, 0.75f);
}
