// Hex v8.4.1 Metal transcription of CUDA kernel regional_relaxzone_filter_edge_v841 (woof/hex/cuda_regional_v841.py:327).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_relaxzone_filter_edge_v841_bindings.metalh"
#pragma clang fp contract(off)

#define RC2(k,c,nc) ((k)*(nc) + (c))
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RQI3(t,k,c,nlev,nc) (((t)*((nlev)+1) + (k))*(nc) + (c))
#define RCES(c,s,me) ((c)*(me) + (s))
#define RADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
#define REGIONAL_N_RELAX_ZONE 5
#define REGIONAL_N_SPEC_ZONE 2
#define MPAS_FTZ_FALLBACK_ENABLED 1


kernel void hexv841_regional_relaxzone_filter_edge_v841(
    constant Hexv841RegionalRelaxzoneFilterEdgeV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalRelaxzoneFilterEdgeV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    const int nvertices = prm__.nvertices;
    const int max_edges = prm__.max_edges;
    const int vertex_degree = prm__.vertex_degree;
    const int nrelax = prm__.nrelax;
    const float ten_dt = prm__.ten_dt;
    const float divdamp = prm__.divdamp;
    const float relax_zone = prm__.relax_zone;
    device const int *relax_edges = rsc__.relax_edges;
    device const int *bdy_mask_edge = rsc__.bdy_mask_edge;
    device const float *mesh_scaling_edge = rsc__.mesh_scaling_edge;
    device const int *n_edges_on_cell = rsc__.n_edges_on_cell;
    device const int *edges_on_cell = rsc__.edges_on_cell;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const int *vertices_on_edge = rsc__.vertices_on_edge;
    device const int *edges_on_vertex = rsc__.edges_on_vertex;
    device const float *edge_sign_on_cell = rsc__.edge_sign_on_cell;
    device const float *edge_sign_on_vertex = rsc__.edge_sign_on_vertex;
    device const float *dc_edge = rsc__.dc_edge;
    device const float *dv_edge = rsc__.dv_edge;
    device const float *inv_dc_edge = rsc__.inv_dc_edge;
    device const float *inv_dv_edge = rsc__.inv_dv_edge;
    device const float *inv_area_cell = rsc__.inv_area_cell;
    device const float *inv_area_triangle = rsc__.inv_area_triangle;
    device const float *ru = rsc__.ru;
    device const float *ru_driving = rsc__.ru_driving;
    device float *tend_ru = rsc__.tend_ru;

    const int element = int(tid);
    if (element >= (nlev * nrelax)) return;
    {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int edge = relax_edges[slot];
        const float mask = (float)bdy_mask_edge[edge];
        const float dc = dc_edge[edge];
        const float dc_squared = mpas_mul(dc, dc);
        const float filter_coef = mpas_div(
            mpas_div(mpas_mul(dc_squared, mpas_sub(mask, 1.0f)),
                     relax_zone),
            mpas_mul(ten_dt, mesh_scaling_edge[edge]));
        const int cell1 = cells_on_edge[2 * edge];
        const int cell2 = cells_on_edge[2 * edge + 1];
        const int vertex1 = vertices_on_edge[2 * edge];
        const int vertex2 = vertices_on_edge[2 * edge + 1];
        const float r_dc = inv_dc_edge[edge];
        const float r_dv = mpas_min(inv_dv_edge[edge],
            mpas_mul(4.0f, inv_dc_edge[edge]));
        const int count1 = n_edges_on_cell[cell1];
        const int count2 = n_edges_on_cell[cell2];
        const float inv_area1 = inv_area_cell[cell1];
        const float inv_area2 = inv_area_cell[cell2];
        float divergence1 = 0.0f;
        float divergence2 = 0.0f;
        float vorticity1 = 0.0f;
        float vorticity2 = 0.0f;
        for (int s = 0; s < count1; ++s) {
            const int ed = edges_on_cell[RCES(cell1, s, max_edges)];
            float sign = mpas_mul(inv_area1, dv_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_cell[RCES(cell1, s, max_edges)]);
            const int j = RE2(k, ed, nedges);
            divergence1 = mpas_add(divergence1, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < count2; ++s) {
            const int ed = edges_on_cell[RCES(cell2, s, max_edges)];
            float sign = mpas_mul(inv_area2, dv_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_cell[RCES(cell2, s, max_edges)]);
            const int j = RE2(k, ed, nedges);
            divergence2 = mpas_add(divergence2, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < vertex_degree; ++s) {
            const int ed = edges_on_vertex[vertex1 * vertex_degree + s];
            float sign = mpas_mul(inv_area_triangle[vertex1], dc_edge[ed]);
            sign = mpas_mul(sign,
                edge_sign_on_vertex[vertex1 * vertex_degree + s]);
            const int j = RE2(k, ed, nedges);
            vorticity1 = mpas_add(vorticity1, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < vertex_degree; ++s) {
            const int ed = edges_on_vertex[vertex2 * vertex_degree + s];
            float sign = mpas_mul(inv_area_triangle[vertex2], dc_edge[ed]);
            sign = mpas_mul(sign,
                edge_sign_on_vertex[vertex2 * vertex_degree + s]);
            const int j = RE2(k, ed, nedges);
            vorticity2 = mpas_add(vorticity2, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        const int i = RE2(k, edge, nedges);
        const float divergence_part = mpas_mul(
            mpas_mul(divdamp, mpas_sub(divergence2, divergence1)), r_dc);
        const float vorticity_part = mpas_mul(
            mpas_sub(vorticity2, vorticity1), r_dv);
        tend_ru[i] = mpas_add(tend_ru[i], mpas_mul(
            filter_coef, mpas_sub(divergence_part, vorticity_part)));
    }
}
