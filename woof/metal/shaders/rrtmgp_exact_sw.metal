// Binary32 operator order from the retained RRTMGP CUDA shortwave solver.
#include <metal_stdlib>
#include "hexv842_mpas_thompson_column_math.metalh"
#include "rrtmgp_exact_sw_2stream_workspace_bindings.metalh"
#include "rrtmgp_exact_sw_flux_reduce_bindings.metalh"
using namespace metal;

static inline F sw_div(F a, F b) { return f32_word(soft32_div(a.bits, b.bits)); }
static inline F sw_max(F a, F b) {
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a == b) return f32_word(a.bits & b.bits);
    return a > b ? a : b;
}
static inline F sw_min(F a, F b) {
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a == b) return f32_word(a.bits | b.bits);
    return a < b ? a : b;
}

kernel void rrtmgp_exact_sw_2stream_workspace(
    constant RrtmgpExactSw2StreamWorkspaceResources &r [[buffer(0)]],
    constant RrtmgpExactSw2StreamWorkspaceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ncol * p.gpt_tile) || p.warp_fold) return;
    const int col = int(gid) / p.gpt_tile, gl = int(gid) - col * p.gpt_tile;
    const int gpt = p.gpt0 + gl;
    if (gpt >= p.ngpt) return;
    const int nlev = p.nlay + 1, base = gl * nlev;
    const int top = p.top_at_1 ? 0 : p.nlay, top_lay = p.top_at_1 ? 0 : p.nlay - 1;
    const int sfc = p.top_at_1 ? p.nlay : 0, sfc_lay = p.top_at_1 ? p.nlay - 1 : 0;
    const int spec = col * p.ngpt + gpt;
    const F eps = 0x1p-23f, min_k = F(1.0e4f) * eps;
    const F min_mu = f32_word(soft32_sqrt(eps.bits));
    device float *optics = r.work_optics + gid * uint(p.nlay * 4);
    device float *direct = r.work_direct + gid * uint(nlev);
    device float *adding = r.work_adding + gid * uint(nlev * 2);
    direct[top] = float(F(r.inc_flux[spec]) * F(r.mu0[col * p.nlay + top_lay]));
    for (int j = 0; j < p.nlay; ++j) {
        const int lay = p.top_at_1 ? j : p.nlay - j - 1;
        const int in = p.top_at_1 ? lay : lay + 1, out = p.top_at_1 ? lay + 1 : lay;
        const int idx = (col * p.nlay + lay) * p.ngpt + gpt;
        F ts, ws, gs;
        if (p.fz_on) {
            const int band = (col * p.nlay + lay) * p.fz_nband + as_type<int>(r.fz_gpt_bands[gpt]);
            F tc = r.fz_cld_tau[band];
            if (p.fz_have_mask) tc = tc * F(r.fz_mask[idx]);
            const F wc = r.fz_cld_ssa[band], gc = r.fz_cld_asy[band];
            const F floorv = F(3.0f) * F(1.17549435e-38f);
            const F tau1 = r.tau[idx], ssa1 = r.ssa[idx];
            const F tau12 = tau1 + tc;
            const F scatter = tau1 * ssa1 + tc * wc;
            const F g12 = sw_div(tau1 * ssa1 * F(0.0f) + tc * wc * gc, sw_max(floorv, scatter));
            const F ssa12 = sw_div(scatter, sw_max(floorv, tau12));
            const F f = g12 * g12, wf = ssa12 * f;
            ts = (F(1.0f) - wf) * tau12;
            ws = sw_div(ssa12 - wf, sw_max(floorv, F(1.0f) - wf));
            gs = sw_div(g12 - f, sw_max(floorv, F(1.0f) - f));
        } else {
            ts = r.tau[idx]; ws = r.ssa[idx]; gs = p.zero_g ? F(0.0f) : F(r.asym[idx]);
        }
        const F gamma1 = (F(8.0f) - ws * (F(5.0f) + F(3.0f) * gs)) * F(0.25f);
        const F gamma2 = F(3.0f) * (ws * (F(1.0f) - gs)) * F(0.25f);
        const F kval = f32_word(soft32_sqrt(sw_max((gamma1 - gamma2) * (gamma1 + gamma2), min_k).bits));
        const F ex = native_tc_math::thompson_aa_expf(-ts * kval), ex2 = ex * ex;
        F rt = sw_div(1.0f, kval * (F(1.0f) + ex2) + gamma1 * (F(1.0f) - ex2));
        const F rdif = rt * gamma2 * (F(1.0f) - ex2);
        const F tdif = rt * F(2.0f) * kval * ex;
        const F mu = sw_max(min_mu, F(r.mu0[col * p.nlay + lay])), kmu = kval * mu;
        const F dterm = fabs(F(1.0f) - kmu * kmu) >= eps ? F(1.0f) - kmu * kmu : eps;
        rt = sw_div(ws * rt, dterm);
        const F gamma3 = (F(2.0f) - F(3.0f) * mu * gs) * F(0.25f), gamma4 = F(1.0f) - gamma3;
        const F alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
        const F alpha2 = gamma1 * gamma3 + gamma2 * gamma4;
        const F kg3 = kval * gamma3, kg4 = kval * gamma4;
        const F tnoscat = native_tc_math::thompson_aa_expf(-sw_div(ts, mu));
        F rdir = rt * ((F(1.0f) - kmu) * (alpha2 + kg3)
            - (F(1.0f) + kmu) * (alpha2 - kg3) * ex2
            - F(2.0f) * (kg3 - alpha2 * kmu) * ex * tnoscat);
        F tdir = -rt * ((F(1.0f) + kmu) * (alpha1 + kg4) * tnoscat
            - (F(1.0f) - kmu) * (alpha1 - kg4) * ex2 * tnoscat
            - F(2.0f) * (kg4 + alpha1 * kmu) * ex);
        rdir = sw_max(0.0f, sw_min(rdir, F(1.0f) - tnoscat));
        tdir = sw_max(0.0f, sw_min(tdir, F(1.0f) - tnoscat - rdir));
        F src_up = rdir * F(direct[in]), src_dn = tdir * F(direct[in]);
        direct[out] = float(tnoscat * F(direct[in]));
        if (F(r.mu0[col * p.nlay + lay]) <= F(0.0f)) src_up = src_dn = F(0.0f);
        optics[4 * lay] = float(rdif); optics[4 * lay + 1] = float(tdif);
        optics[4 * lay + 2] = float(src_dn); optics[4 * lay + 3] = float(src_up);
    }
    const F source = F(r.mu0[col * p.nlay + sfc_lay]) > F(0.0f)
        ? F(direct[sfc]) * F(r.sfc_alb_dir[spec]) : F(0.0f);
    adding[2 * sfc] = r.sfc_alb_dif[spec]; adding[2 * sfc + 1] = float(source);
    for (int j = 0; j < p.nlay; ++j) {
        const int lay = p.top_at_1 ? p.nlay - j - 1 : j;
        const int prev = p.top_at_1 ? lay + 1 : lay, next = p.top_at_1 ? lay : lay + 1;
        const F x = optics[4 * lay], y = optics[4 * lay + 1];
        const F z = optics[4 * lay + 2], w = optics[4 * lay + 3];
        const F ax = adding[2 * prev], ay = adding[2 * prev + 1];
        const F dn = sw_div(1.0f, F(1.0f) - x * ax);
        adding[2 * next] = float(x + y * y * ax * dn);
        adding[2 * next + 1] = float(w + y * dn * (ay + ax * z));
    }
    F dnv = 0.0f, upv = dnv * F(adding[2 * top]) + F(adding[2 * top + 1]);
    r.part_up[(base + top) * p.ncol + col] = float(upv);
    r.part_dn[(base + top) * p.ncol + col] = float(dnv);
    r.part_dir[(base + top) * p.ncol + col] = direct[top];
    for (int j = 0; j < p.nlay; ++j) {
        const int lev = p.top_at_1 ? j + 1 : p.nlay - j - 1;
        const int lay = p.top_at_1 ? lev - 1 : lev;
        const F x = optics[4 * lay], y = optics[4 * lay + 1], z = optics[4 * lay + 2];
        const F ax = adding[2 * lev], ay = adding[2 * lev + 1];
        const F dn = sw_div(1.0f, F(1.0f) - x * ax);
        dnv = (y * dnv + x * ay + z) * dn;
        upv = dnv * ax + ay;
        r.part_up[(base + lev) * p.ncol + col] = float(upv);
        r.part_dn[(base + lev) * p.ncol + col] = float(dnv);
        r.part_dir[(base + lev) * p.ncol + col] = direct[lev];
    }
}

kernel void rrtmgp_exact_sw_flux_reduce(
    constant RrtmgpExactSwFluxReduceResources &r [[buffer(0)]],
    constant RrtmgpExactSwFluxReduceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ncol * p.nlev)) return;
    const int lev = int(gid) / p.ncol, col = int(gid) - lev * p.ncol;
    const int output = col * p.nlev + lev;
    F up = p.gpt0 == 0 ? F(0.0f) : F(r.flux_up[output]);
    F down = p.gpt0 == 0 ? F(0.0f) : F(r.flux_dn[output]);
    F direct = p.gpt0 == 0 ? F(0.0f) : F(r.flux_dir[output]);
    for (int g = 0; g < p.ntile; ++g) {
        const int input = (g * p.nlev + lev) * p.ncol + col;
        up = up + F(r.part_up[input]);
        down = down + F(r.part_dn[input]) + F(r.part_dir[input]);
        direct = direct + F(r.part_dir[input]);
    }
    r.flux_up[output] = float(up);
    r.flux_dn[output] = float(down);
    r.flux_dir[output] = float(direct);
}
