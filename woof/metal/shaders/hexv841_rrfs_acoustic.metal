#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

#include "hexv841_rrfs_acoustic_acoustic_cofrz_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_cofrz_rrfs(constant Hexv841RrfsAcousticAcousticCofrzRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticCofrzRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const float dts = p.dts;
    const float epssm = p.epssm;
    device const float *rdzw = r.rdzw;
    device float *cofrz = r.cofrz;
    const int k = int(gid % p.dispatch_x);
    if (k < nlev) cofrz[k] = mpas_mul(
        mpas_mul(mpas_mul(0.5f, dts), mpas_add(1.0f, epssm)), rdzw[k]);
}


#include "hexv841_rrfs_acoustic_acoustic_coefficients_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_coefficients_rrfs(constant Hexv841RrfsAcousticAcousticCoefficientsRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticCoefficientsRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const float gravity = p.gravity;
    const float rgas = p.rgas;
    const float cp = p.cp;
    device const float *zz = r.zz;
    device const float *cqw = r.cqw;
    device const float *pressure = r.pressure;
    device const float *theta = r.theta;
    device const float *rho_base = r.rho_base;
    device const float *rtheta_base = r.rtheta_base;
    device const float *pressure_base = r.pressure_base;
    device const float *rtheta_p = r.rtheta_p;
    device const float *qtot = r.qtot;
    device const float *rdzw = r.rdzw;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzu = r.rdzu;
    const int nsolve = p.nsolve;
    device float *cofwr = r.cofwr;
    device float *cofwz = r.cofwz;
    device float *coftz = r.coftz;
    device float *cofwt = r.cofwt;
    device float *cofrz = r.cofrz;
    device float *a_tri = r.a_tri;
    device float *b_tri = r.b_tri;
    device float *c_tri = r.c_tri;
    device float *alpha_tri = r.alpha_tri;
    device float *gamma_tri = r.gamma_tri;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    const float dtseps = 0.5f * dts * (1.0f + epssm);
    const float rcv = rgas / (cp - rgas);
    const float csquared = cp * rcv;

    {
        const int i = C2(k, cell, ncells);
        cofwt[i] = 0.5f * dtseps * rcv * zz[i] * gravity * rho_base[i]
            / (1.0f + qtot[i]) * pressure[i]
            / ((rtheta_base[i] + rtheta_p[i]) * pressure_base[i]);
        a_tri[i] = 0.0f;
        b_tri[i] = 1.0f;
        c_tri[i] = 0.0f;
        alpha_tri[i] = 0.0f;
        gamma_tri[i] = 0.0f;
        coftz[C2(k, cell, ncells)] = 0.0f;
    }
    if (k == 0) coftz[C2(nlev, cell, ncells)] = 0.0f;

    if (k > 0) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float interp_zz = fzm[k] * zz[i] + fzp[k] * zz[im];
        cofwr[i] = 0.5f * dtseps * gravity * interp_zz;
        const float interp_p = fzm[k] * pressure[i] + fzp[k] * pressure[im];
        cofwz[i] = dtseps * csquared * interp_zz * rdzu[k]
            * cqw[i] * interp_p;
        coftz[i] = dtseps * (fzm[k] * theta[i] + fzp[k] * theta[im]);
    }
}

#include "hexv841_rrfs_acoustic_acoustic_matrix_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_matrix_rrfs(constant Hexv841RrfsAcousticAcousticMatrixRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticMatrixRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const float gravity = p.gravity;
    const float rgas = p.rgas;
    const float cp = p.cp;
    device const float *zz = r.zz;
    device const float *cqw = r.cqw;
    device const float *pressure = r.pressure;
    device const float *theta = r.theta;
    device const float *rho_base = r.rho_base;
    device const float *rtheta_base = r.rtheta_base;
    device const float *pressure_base = r.pressure_base;
    device const float *rtheta_p = r.rtheta_p;
    device const float *qtot = r.qtot;
    device const float *rdzw = r.rdzw;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzu = r.rdzu;
    const int nsolve = p.nsolve;
    device float *cofwr = r.cofwr;
    device float *cofwz = r.cofwz;
    device float *coftz = r.coftz;
    device float *cofwt = r.cofwt;
    device float *cofrz = r.cofrz;
    device float *a_tri = r.a_tri;
    device float *b_tri = r.b_tri;
    device float *c_tri = r.c_tri;
    device float *alpha_tri = r.alpha_tri;
    device float *gamma_tri = r.gamma_tri;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (k > 0) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const int ip = C2(k + 1, cell, ncells);
        a_tri[i] = -cofwz[i] * coftz[im] * rdzw[k - 1] * zz[im]
            + cofwr[i] * cofrz[k - 1]
            - cofwt[im] * coftz[im] * rdzw[k - 1];
        b_tri[i] = 1.0f
            + cofwz[i] * (coftz[i] * rdzw[k] * zz[i]
                + coftz[i] * rdzw[k - 1] * zz[im])
            - coftz[i] * (cofwt[i] * rdzw[k]
                - cofwt[im] * rdzw[k - 1])
            + cofwr[i] * (cofrz[k] - cofrz[k - 1]);
        c_tri[i] = -cofwz[i] * coftz[ip] * rdzw[k] * zz[i]
            - cofwr[i] * cofrz[k]
            + cofwt[i] * coftz[ip] * rdzw[k];
    }
}

#include "hexv841_rrfs_acoustic_acoustic_factor_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_factor_rrfs(constant Hexv841RrfsAcousticAcousticFactorRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticFactorRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const float gravity = p.gravity;
    const float rgas = p.rgas;
    const float cp = p.cp;
    device const float *zz = r.zz;
    device const float *cqw = r.cqw;
    device const float *pressure = r.pressure;
    device const float *theta = r.theta;
    device const float *rho_base = r.rho_base;
    device const float *rtheta_base = r.rtheta_base;
    device const float *pressure_base = r.pressure_base;
    device const float *rtheta_p = r.rtheta_p;
    device const float *qtot = r.qtot;
    device const float *rdzw = r.rdzw;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzu = r.rdzu;
    const int nsolve = p.nsolve;
    device float *cofwr = r.cofwr;
    device float *cofwz = r.cofwz;
    device float *coftz = r.coftz;
    device float *cofwt = r.cofwt;
    device float *cofrz = r.cofrz;
    device float *a_tri = r.a_tri;
    device float *b_tri = r.b_tri;
    device float *c_tri = r.c_tri;
    device float *alpha_tri = r.alpha_tri;
    device float *gamma_tri = r.gamma_tri;
    const int cell = int(gid % p.dispatch_x);
    if (cell >= nsolve) return;
    
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float denominator = b_tri[i] - a_tri[i] * gamma_tri[im];
        alpha_tri[i] = 1.0f / denominator;
        gamma_tri[i] = c_tri[i] * alpha_tri[i];
    }
}


#include "hexv841_rrfs_acoustic_acoustic_ru_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_ru_rrfs(constant Hexv841RrfsAcousticAcousticRuRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticRuRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int ncells = p.ncells;
    const int small_step = p.small_step;
    const float dts = p.dts;
    const int edge_start = p.edge_start;
    const int edge_end = p.edge_end;
    const float gravity = p.gravity;
    const float rgas = p.rgas;
    const float cp = p.cp;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *zz = r.zz;
    device const float *exner = r.exner;
    device const float *cqu = r.cqu;
    device const float *zxu = r.zxu;
    device const float *tend_ru = r.tend_ru;
    device const float *spec_edge = r.spec_edge;
    const int nsolve = p.nsolve;
    device const float *rho_pp = r.rho_pp;
    device const float *rtheta_pp = r.rtheta_pp;
    device float *ru_p = r.ru_p;
    device float *ru_avg = r.ru_avg;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    if (edge < edge_start || edge >= edge_end) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    if (c0 >= nsolve && c1 >= nsolve) return;
    const float half_value = 0.5f;
    const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
    const float csquared = mpas_mul(cp, rcv);
    {
        const int index = E2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, tend_ru[index]);
            ru_p[index] = value;
            ru_avg[index] = value;
        } else {
            const float normal = mpas_mul(mpas_sub(
                rtheta_pp[C2(k, c1, ncells)],
                rtheta_pp[C2(k, c0, ncells)]), inv_dc_edge[edge]);
            const float normalized = mpas_div(normal, mpas_mul(half_value, mpas_add(
                zz[C2(k, c1, ncells)], zz[C2(k, c0, ncells)])));
            float pgrad = mpas_mul(cqu[index], half_value);
            pgrad = mpas_mul(pgrad, csquared);
            pgrad = mpas_mul(pgrad, mpas_add(exner[C2(k, c0, ncells)],
                exner[C2(k, c1, ncells)]));
            pgrad = mpas_mul(pgrad, normalized);
            const float terrain = mpas_mul(mpas_mul(mpas_mul(
                half_value, zxu[index]), gravity), mpas_add(
                    rho_pp[C2(k, c0, ncells)],
                    rho_pp[C2(k, c1, ncells)]));
            pgrad = mpas_add(pgrad, terrain);
            ru_p[index] = mpas_add(ru_p[index], mpas_mul(
                dts, mpas_sub(tend_ru[index], mpas_mul(mpas_sub(1.0f, spec_edge[edge]), pgrad))));
            ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
        }
    }
}


#include "hexv841_rrfs_acoustic_acoustic_prepare_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_prepare_rrfs(constant Hexv841RrfsAcousticAcousticPrepareRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticPrepareRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int small_step = p.small_step;
    const int nsolve = p.nsolve;
    const int cell_start = p.cell_start;
    const int cell_end = p.cell_end;
    device float *rw_p = r.rw_p;
    device float *rtheta_pp = r.rtheta_pp;
    device float *rtheta_pp_old = r.rtheta_pp_old;
    device float *rho_pp = r.rho_pp;
    device float *ww_avg = r.ww_avg;
    const int element = int(gid % p.dispatch_x);
    if (element >= (nlev + 1) * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= ncells) return;
    if (k < nlev) {
        const int index = C2(k, cell, ncells);
        if (small_step == 1) {
            if (cell >= cell_start && cell < cell_end) rtheta_pp_old[index] = 0.0f;
            if (cell < nsolve) {
                rtheta_pp[index] = 0.0f;
                rho_pp[index] = 0.0f;
            }
        } else {
            if (cell >= cell_start && cell < cell_end) rtheta_pp_old[index] = rtheta_pp[index];
        }
    }
    if (small_step == 1 && cell < nsolve) {
         {
            const int index = C2(k, cell, ncells);
            rw_p[index] = 0.0f;
            ww_avg[index] = 0.0f;
        }
    }
}


#include "hexv841_rrfs_acoustic_acoustic_rs_ts_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_rs_ts_rrfs(constant Hexv841RrfsAcousticAcousticRsTsRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticRsTsRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    const float dts = p.dts;
    const float epssm = p.epssm;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *edge_sign_on_cell = r.edge_sign_on_cell;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *spec_cell = r.spec_cell;
    const int nsolve = p.nsolve;
    device const float *theta_m = r.theta_m;
    device const float *rdzw = r.rdzw;
    device const float *cofrz = r.cofrz;
    device const float *coftz = r.coftz;
    device const float *ru_p = r.ru_p;
    device const float *rw_p = r.rw_p;
    device const float *rho_pp = r.rho_pp;
    device const float *rtheta_pp = r.rtheta_pp;
    device const float *tend_rho = r.tend_rho;
    device const float *tend_rt = r.tend_rt;
    device float *rs = r.rs;
    device float *ts = r.ts;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (spec_cell[cell] != 0.0f) {
         {
            rs[C2(k, cell, ncells)] = tend_rho[C2(k, cell, ncells)];
            ts[C2(k, cell, ncells)] = tend_rt[C2(k, cell, ncells)];
        }
        return;
    }
    const int count = n_edges_on_cell[cell];
    {
        const int index = C2(k, cell, ncells);
        float r = 0.0f;
        float t = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[slot * ncells + cell];
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            const float flux = mpas_mul(mpas_mul(mpas_mul(mpas_mul(
                edge_sign_on_cell[slot * ncells + cell], dts),
                dv_edge[edge]), ru_p[E2(k, edge, nedges)]), inv_area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                theta_m[C2(k, c1, ncells)],
                theta_m[C2(k, c0, ncells)])));
        }
        const float resm = mpas_div(mpas_sub(1.0f, epssm), mpas_add(1.0f, epssm));
        r = mpas_sub(mpas_add(mpas_add(rho_pp[index], mpas_mul(dts, tend_rho[index])), r),
            mpas_mul(mpas_mul(cofrz[k], resm), mpas_sub(
                rw_p[C2(k + 1, cell, ncells)], rw_p[index])));
        t = mpas_sub(mpas_add(mpas_add(rtheta_pp[index], mpas_mul(dts, tend_rt[index])), t),
            mpas_mul(mpas_mul(resm, rdzw[k]), mpas_sub(
                mpas_mul(coftz[C2(k + 1, cell, ncells)], rw_p[C2(k + 1, cell, ncells)]),
                mpas_mul(coftz[index], rw_p[index]))));
        rs[index] = r;
        ts[index] = t;
    }
}



#include "hexv841_rrfs_acoustic_acoustic_column_pre_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_column_pre_rrfs(constant Hexv841RrfsAcousticAcousticColumnPreRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticColumnPreRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const int nsolve = p.nsolve;
    device const float *spec_cell = r.spec_cell;
    device const float *zz = r.zz;
    device const float *rho_zz = r.rho_zz;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzw = r.rdzw;
    device const float *dss = r.dss;
    device const float *w = r.w;
    device const float *rw = r.rw;
    device const float *rw_save = r.rw_save;
    device const float *tend_rw = r.tend_rw;
    device const float *rs = r.rs;
    device const float *ts = r.ts;
    device const float *cofwr = r.cofwr;
    device const float *cofwz = r.cofwz;
    device const float *coftz = r.coftz;
    device const float *cofwt = r.cofwt;
    device const float *cofrz = r.cofrz;
    device const float *a_tri = r.a_tri;
    device const float *alpha_tri = r.alpha_tri;
    device const float *gamma_tri = r.gamma_tri;
    device float *rw_p = r.rw_p;
    device float *rho_pp = r.rho_pp;
    device float *rtheta_pp = r.rtheta_pp;
    device float *ww_avg = r.ww_avg;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (spec_cell[cell] != 0.0f) {
        {
            const int i = C2(k, cell, ncells);
            rho_pp[i] = mpas_add(rho_pp[i], mpas_mul(dts, rs[i]));
            rtheta_pp[i] = mpas_add(rtheta_pp[i], mpas_mul(dts, ts[i]));
            rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, tend_rw[i]));
            ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(mpas_mul(0.5f, mpas_add(1.0f, epssm)), rw_p[i]));
        }
        return;
    }
    if (k == 0) return;
    const float resm = (1.0f - epssm) / (1.0f + epssm);
    {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(
            mpas_mul(0.5f, mpas_sub(1.0f, epssm)), rw_p[i]));
        rw_p[i] = mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(
            rw_p[i], mpas_mul(dts, tend_rw[i])), mpas_mul(cofwz[i],
                mpas_add(mpas_sub(mpas_mul(zz[i], ts[i]),
                    mpas_mul(zz[im], ts[im])), mpas_mul(resm,
                        mpas_sub(mpas_mul(zz[i], rtheta_pp[i]),
                            mpas_mul(zz[im], rtheta_pp[im])))))),
            mpas_mul(cofwr[i], mpas_add(mpas_add(rs[i], rs[im]),
                mpas_mul(resm, mpas_add(rho_pp[i], rho_pp[im]))))),
            mpas_mul(cofwt[i], mpas_add(ts[i],
                mpas_mul(resm, rtheta_pp[i])))),
            mpas_mul(cofwt[im], mpas_add(ts[im],
                mpas_mul(resm, rtheta_pp[im]))));
    }
}

#include "hexv841_rrfs_acoustic_acoustic_column_solve_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_column_solve_rrfs(constant Hexv841RrfsAcousticAcousticColumnSolveRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticColumnSolveRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const int nsolve = p.nsolve;
    device const float *spec_cell = r.spec_cell;
    device const float *zz = r.zz;
    device const float *rho_zz = r.rho_zz;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzw = r.rdzw;
    device const float *dss = r.dss;
    device const float *w = r.w;
    device const float *rw = r.rw;
    device const float *rw_save = r.rw_save;
    device const float *tend_rw = r.tend_rw;
    device const float *rs = r.rs;
    device const float *ts = r.ts;
    device const float *cofwr = r.cofwr;
    device const float *cofwz = r.cofwz;
    device const float *coftz = r.coftz;
    device const float *cofwt = r.cofwt;
    device const float *cofrz = r.cofrz;
    device const float *a_tri = r.a_tri;
    device const float *alpha_tri = r.alpha_tri;
    device const float *gamma_tri = r.gamma_tri;
    device float *rw_p = r.rw_p;
    device float *rho_pp = r.rho_pp;
    device float *rtheta_pp = r.rtheta_pp;
    device float *ww_avg = r.ww_avg;
    const int cell = int(gid % p.dispatch_x);
    if (cell >= nsolve || spec_cell[cell] != 0.0f) return;

    
    
    
    float previous = rw_p[cell];
    for (int k = 1; k < nlev; k += 4) {
        float value[4], lower[4], inverse[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k + j < nlev) {
                const int i = C2(k + j, cell, ncells);
                value[j] = rw_p[i];
                lower[j] = a_tri[i];
                inverse[j] = alpha_tri[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k + j < nlev) {
                previous = mpas_mul(mpas_sub(value[j],
                    mpas_mul(lower[j], previous)), inverse[j]);
                rw_p[C2(k + j, cell, ncells)] = previous;
            }
        }
    }
    previous = rw_p[C2(nlev, cell, ncells)];
    for (int k = nlev - 1; k >= 0; k -= 4) {
        float value[4], upper[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k - j >= 0) {
                const int i = C2(k - j, cell, ncells);
                value[j] = rw_p[i];
                upper[j] = gamma_tri[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k - j >= 0) {
                previous = mpas_sub(value[j], mpas_mul(upper[j], previous));
                rw_p[C2(k - j, cell, ncells)] = previous;
            }
        }
    }
}

#include "hexv841_rrfs_acoustic_acoustic_column_post_w_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_column_post_w_rrfs(constant Hexv841RrfsAcousticAcousticColumnPostWRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticColumnPostWRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const int nsolve = p.nsolve;
    device const float *spec_cell = r.spec_cell;
    device const float *zz = r.zz;
    device const float *rho_zz = r.rho_zz;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzw = r.rdzw;
    device const float *dss = r.dss;
    device const float *w = r.w;
    device const float *rw = r.rw;
    device const float *rw_save = r.rw_save;
    device const float *tend_rw = r.tend_rw;
    device const float *rs = r.rs;
    device const float *ts = r.ts;
    device const float *cofwr = r.cofwr;
    device const float *cofwz = r.cofwz;
    device const float *coftz = r.coftz;
    device const float *cofwt = r.cofwt;
    device const float *cofrz = r.cofrz;
    device const float *a_tri = r.a_tri;
    device const float *alpha_tri = r.alpha_tri;
    device const float *gamma_tri = r.gamma_tri;
    device float *rw_p = r.rw_p;
    device float *rho_pp = r.rho_pp;
    device float *rtheta_pp = r.rtheta_pp;
    device float *ww_avg = r.ww_avg;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (k == 0 || spec_cell[cell] != 0.0f) return;
    {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rw_save[i], rw[i]);
        float damping = mpas_mul(dts, dss[i]);
        damping = mpas_mul(damping, mpas_add(mpas_mul(fzm[k], zz[i]), mpas_mul(fzp[k], zz[im])));
        damping = mpas_mul(damping, mpas_add(mpas_mul(fzm[k], rho_zz[i]), mpas_mul(fzp[k], rho_zz[im])));
        damping = mpas_mul(damping, w[i]);
        rw_p[i] = mpas_sub(mpas_div(mpas_sub(mpas_add(rw_p[i], delta_saved), damping),
            mpas_add(1.0f, mpas_mul(dts, dss[i]))), delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(mpas_mul(
            0.5f, mpas_add(1.0f, epssm)), rw_p[i]));
    }
}

#include "hexv841_rrfs_acoustic_acoustic_column_post_rt_rrfs_bindings.metalh"
kernel void hexv841_rrfs_acoustic_acoustic_column_post_rt_rrfs(constant Hexv841RrfsAcousticAcousticColumnPostRtRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsAcousticAcousticColumnPostRtRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dts = p.dts;
    const float epssm = p.epssm;
    const int nsolve = p.nsolve;
    device const float *spec_cell = r.spec_cell;
    device const float *zz = r.zz;
    device const float *rho_zz = r.rho_zz;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *rdzw = r.rdzw;
    device const float *dss = r.dss;
    device const float *w = r.w;
    device const float *rw = r.rw;
    device const float *rw_save = r.rw_save;
    device const float *tend_rw = r.tend_rw;
    device const float *rs = r.rs;
    device const float *ts = r.ts;
    device const float *cofwr = r.cofwr;
    device const float *cofwz = r.cofwz;
    device const float *coftz = r.coftz;
    device const float *cofwt = r.cofwt;
    device const float *cofrz = r.cofrz;
    device const float *a_tri = r.a_tri;
    device const float *alpha_tri = r.alpha_tri;
    device const float *gamma_tri = r.gamma_tri;
    device float *rw_p = r.rw_p;
    device float *rho_pp = r.rho_pp;
    device float *rtheta_pp = r.rtheta_pp;
    device float *ww_avg = r.ww_avg;
    const int element = int(gid % p.dispatch_x);
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (spec_cell[cell] != 0.0f) return;
    {
        const int i = C2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(cofrz[k], mpas_sub(
            rw_p[C2(k + 1, cell, ncells)], rw_p[i])));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(rdzw[k], mpas_sub(
            mpas_mul(coftz[C2(k + 1, cell, ncells)],
                rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(coftz[i], rw_p[i]))));
    }
}
