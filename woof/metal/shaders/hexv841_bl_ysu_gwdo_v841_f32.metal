// Hex v8.4.1 Metal transcription of CUDA kernel bl_ysu_gwdo_v841_f32 (woof/hex/cuda_gwdo_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "pressure_power.metalh"
#include "hexv841_bl_ysu_gwdo_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
#define GWDO_NLEV 55


kernel void hexv841_bl_ysu_gwdo_v841_f32(
    constant Hexv841BlYsuGwdoV841F32Resources &r [[buffer(0)]],
    constant Hexv841BlYsuGwdoV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dt = p.dt;
    const float len_disp = p.len_disp;
    const float gravity = p.gravity;
    const float cp = p.cp;
    const float rd = p.rd;
    const float fv = p.fv;
    const float pi = p.pi;
    device const float *u = r.u;
    device const float *v = r.v;
    device const float *temperature = r.temperature;
    device const float *qv = r.qv;
    device const float *p_mass_dry_hyd = r.p_mass_dry_hyd;
    device const float *p_interface_dry_hyd = r.p_interface_dry_hyd;
    device const float *exner = r.exner;
    device const float *zmid = r.zmid;
    device const float *rublten_in = r.rublten_in;
    device const float *rvblten_in = r.rvblten_in;
    device const float *mesh_density = r.mesh_density;
    device const float *var2d = r.var2d;
    device const float *oc1 = r.oc1;
    device const float *oa1 = r.oa1;
    device const float *oa2 = r.oa2;
    device const float *oa3 = r.oa3;
    device const float *oa4 = r.oa4;
    device const float *ol1 = r.ol1;
    device const float *ol2 = r.ol2;
    device const float *ol3 = r.ol3;
    device const float *ol4 = r.ol4;
    device float *rublten_out = r.rublten_out;
    device float *rvblten_out = r.rvblten_out;
    device float *dtaux3d = r.dtaux3d;
    device float *dtauy3d = r.dtauy3d;
    device float *dusfcg = r.dusfcg;
    device float *dvsfcg = r.dvsfcg;
    device float *rubldiff = r.rubldiff;
    device float *rvbldiff = r.rvbldiff;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    if (nlev != GWDO_NLEV) {
        atomicExch(invalid, 1);
        return;
    }

    const float md = mesh_density[cell];
    bool bad = !isfinite(dt) || dt <= 0.0f || !isfinite(len_disp)
        || len_disp <= 0.0f || !isfinite(md) || md <= 0.0f || md > 1.0f
        || !isfinite(var2d[cell]) || var2d[cell] < 0.0f
        || !isfinite(oc1[cell]) || oc1[cell] < 0.0f;
    const float oa_input[4] = {oa1[cell], oa2[cell], oa3[cell], oa4[cell]};
    const float ol_input[4] = {ol1[cell], ol2[cell], ol3[cell], ol4[cell]};
    for (int j = 0; j < 4; ++j) {
        bad = bad || !isfinite(oa_input[j]) || oa_input[j] < -1.0f
            || oa_input[j] > 1.0f || !isfinite(ol_input[j])
            || ol_input[j] < 0.0f || ol_input[j] > 1.0f;
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        bad = bad || !isfinite(u[i]) || !isfinite(v[i])
            || !isfinite(temperature[i]) || temperature[i] <= 0.0f
            || !isfinite(qv[i]) || qv[i] < 0.0f
            || !isfinite(p_mass_dry_hyd[i]) || p_mass_dry_hyd[i] <= 0.0f
            || !isfinite(exner[i]) || exner[i] <= 0.0f
            || !isfinite(zmid[i]) || !isfinite(rublten_in[i])
            || !isfinite(rvblten_in[i]);
        const int ii = k * ncells + cell;
        const int ii_above = (k + 1) * ncells + cell;
        bad = bad || !isfinite(p_interface_dry_hyd[ii])
            || p_interface_dry_hyd[ii] <= 0.0f;
        if (k + 1 < nlev) {
            bad = bad || !(p_mass_dry_hyd[i] > p_mass_dry_hyd[i + ncells])
                || !(zmid[i + ncells] > zmid[i]);
        }
        if (k == nlev - 1) {
            bad = bad || !isfinite(p_interface_dry_hyd[ii_above])
                || p_interface_dry_hyd[ii_above] <= 0.0f;
        }
        bad = bad || !(p_interface_dry_hyd[ii]
            > p_interface_dry_hyd[ii_above]);
    }
    if (bad) {
        atomicExch(invalid, 1);
        return;
    }

    float vtj[GWDO_NLEV];
    float vtk[GWDO_NLEV];
    float rho[GWDO_NLEV];
    float delta_p[GWDO_NLEV];
    float u1[GWDO_NLEV];
    float v1[GWDO_NLEV];
    float usqj[GWDO_NLEV];
    float bnv2[GWDO_NLEV];
    float velco[GWDO_NLEV - 1];
    float taup[GWDO_NLEV + 1];
    float taufb[GWDO_NLEV + 1];
    float taud[GWDO_NLEV];
    for (int k = 0; k < nlev; ++k) {
        usqj[k] = 0.0f;
        bnv2[k] = 0.0f;
        taud[k] = 0.0f;
    }
    for (int k = 0; k <= nlev; ++k) {
        taup[k] = 0.0f;
        taufb[k] = 0.0f;
    }

    const float dx = mpas_div(len_disp, fp32_pressure_power(md, 1.0f, 0.25f));
    const float diagonal = mpas_sqrt(
        mpas_add(mpas_mul(dx, dx), mpas_mul(dx, dx)));
    const float dxy4[4] = {dx, dx, diagonal, diagonal};
    const float dxy4p[4] = {dx, dx, diagonal, diagonal};
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const int ip = (k + 1) * ncells + cell;
        const float virtual_factor = mpas_add(1.0f, mpas_mul(fv, qv[i]));
        vtj[k] = mpas_mul(temperature[i], virtual_factor);
        vtk[k] = mpas_div(vtj[k], exner[i]);
        rho[k] = mpas_div(
            mpas_mul(mpas_div(1.0f, rd), p_mass_dry_hyd[i]), vtj[k]);
        delta_p[k] = mpas_sub(
            p_interface_dry_hyd[i], p_interface_dry_hyd[ip]);
        u1[k] = u[i];
        v1[k] = v[i];
    }

    const float zlowtop = mpas_mul(2.0f, var2d[cell]);
    int klowtop = 0;
    for (int k = 1; k < nlev; ++k) {
        const float height = mpas_sub(
            zmid[k * ncells + cell], zmid[cell]);
        if (zlowtop > 0.0f && klowtop == 0 && height >= zlowtop)
            klowtop = k + 2;
    }
    int kbl = klowtop;
    if (kbl > nlev) kbl = nlev;
    if (kbl < 2) kbl = 2;
    const float delks = mpas_div(
        1.0f, mpas_sub(p_interface_dry_hyd[cell],
            p_interface_dry_hyd[(kbl - 1) * ncells + cell]));
    const float delks1 = mpas_div(
        1.0f, mpas_sub(p_mass_dry_hyd[cell],
            p_mass_dry_hyd[(kbl - 1) * ncells + cell]));
    float ubar = 0.0f;
    float vbar = 0.0f;
    float rhobar = 0.0f;
    for (int k = 0; k < nlev; ++k) {
        if (k + 1 < kbl) {
            const float weight = mpas_mul(delta_p[k], delks);
            ubar = mpas_add(ubar, mpas_mul(weight, u1[k]));
            vbar = mpas_add(vbar, mpas_mul(weight, v1[k]));
            rhobar = mpas_add(rhobar, mpas_mul(weight, rho[k]));
        }
    }

    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    const float fdir = mpas_div(8.0f, mpas_mul(2.0f, pi));
    const float wdir = mpas_add(precise::atan2(ubar, vbar), pi);
    const int nearest = (int)floor(mpas_add(mpas_mul(fdir, wdir), 0.5f));
    const int nwd = nwdir[nearest % 8];
    const int direction = (nwd - 1) % 4;
    const int sign = 1 - 2 * ((nwd - 1) / 4);
    const float oa = mpas_mul((float)sign, oa_input[direction]);
    const float ol = ol_input[direction];
    const float ol_perp[4] = {
        ol_input[1], ol_input[0], ol_input[3], ol_input[2]};
    const float olp = ol_perp[direction];
    float od = mpas_div(olp, mpas_max(ol, 1.0e-5f));
    od = mpas_max(mpas_min(od, 10.0f), 0.1f);
    const float dxy = dxy4[direction];
    const float dxyp = dxy4p[direction];

    for (int k = 0; k < nlev - 1; ++k) {
        const int i = k * ncells + cell;
        const int ia = i + ncells;
        const float ti = mpas_div(
            2.0f, mpas_add(temperature[i], temperature[ia]));
        const float rdz = mpas_div(1.0f, mpas_sub(zmid[ia], zmid[i]));
        const float du = mpas_sub(u1[k], u1[k + 1]);
        const float dv = mpas_sub(v1[k], v1[k + 1]);
        const float dw2 = mpas_add(mpas_mul(du, du), mpas_mul(dv, dv));
        const float shr2 = mpas_mul(
            mpas_mul(mpas_max(dw2, 1.0f), rdz), rdz);
        const float stability = mpas_add(
            mpas_div(gravity, cp), mpas_mul(rdz, mpas_sub(vtj[k + 1], vtj[k])));
        const float bvf2 = mpas_mul(mpas_mul(gravity, stability), ti);
        usqj[k] = mpas_max(mpas_div(bvf2, shr2), -100.0f);
        bnv2[k] = mpas_div(
            mpas_mul(mpas_mul(mpas_mul(2.0f, gravity), rdz),
                mpas_sub(vtk[k + 1], vtk[k])),
            mpas_add(vtk[k + 1], vtk[k]));
    }

    float ulow = mpas_sqrt(mpas_add(mpas_mul(ubar, ubar), mpas_mul(vbar, vbar)));
    ulow = mpas_max(ulow, 1.0f);
    const float rulow = mpas_div(1.0f, ulow);
    for (int k = 0; k < nlev - 1; ++k) {
        float aligned = mpas_add(
            mpas_mul(mpas_add(u1[k], u1[k + 1]), ubar),
            mpas_mul(mpas_add(v1[k], v1[k + 1]), vbar));
        velco[k] = mpas_mul(mpas_mul(0.5f, aligned), rulow);
        if (velco[k] < 1.0f && velco[k] > 0.0f) velco[k] = 1.0f;
    }
    bool ldrag = velco[0] <= 0.0f;
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl && velco[kf - 1] <= 0.0f) ldrag = true;
    }

    const float first_weight = mpas_mul(
        mpas_sub(p_mass_dry_hyd[cell], p_mass_dry_hyd[ncells + cell]),
        delks1);
    bnv2[0] = mpas_mul(first_weight, bnv2[0]);
    usqj[0] = mpas_mul(first_weight, usqj[0]);
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl) {
            const int k = kf - 1;
            const float pressure_weight = mpas_mul(
                mpas_sub(p_mass_dry_hyd[k * ncells + cell],
                    p_mass_dry_hyd[(k + 1) * ncells + cell]), delks1);
            bnv2[0] = mpas_add(bnv2[0], mpas_mul(bnv2[k], pressure_weight));
            usqj[0] = mpas_add(usqj[0], mpas_mul(usqj[k], pressure_weight));
        }
    }
    ldrag = ldrag || bnv2[0] <= 0.0f || ulow == 1.0f || var2d[cell] <= 0.0f;
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl) usqj[kf - 1] = usqj[0];
    }

    float bnv = 0.0f;
    float fr = 0.0f;
    float xn = 0.0f;
    float yn = 0.0f;
    if (!ldrag) {
        bnv = mpas_sqrt(bnv2[0]);
        fr = mpas_mul(mpas_mul(mpas_mul(bnv, rulow), var2d[cell]), od);
        fr = mpas_min(fr, 10.0f);
        xn = mpas_mul(ubar, rulow);
        yn = mpas_mul(vbar, rulow);
    }

    float coefm = 0.0f;
    float taub = 0.0f;
    if (!ldrag) {
        float efact = fp32_pressure_power(mpas_add(oa, 2.0f), 1.0f, mpas_mul(0.8f, fr));
        efact = mpas_min(mpas_max(efact, 0.0f), 10.0f);
        coefm = fp32_pressure_power(mpas_add(1.0f, ol), 1.0f, mpas_add(oa, 1.0f));
        const float xlinv = mpas_div(coefm, dx);
        const float tem = mpas_mul(mpas_mul(fr, fr), oc1[cell]);
        const float gfobnv = mpas_div(
            tem, mpas_mul(mpas_add(tem, 0.5f), bnv));
        taub = mpas_mul(xlinv, rhobar);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(mpas_mul(taub, gfobnv), efact);
    }
    for (int kf = 1; kf <= nlev; ++kf) {
        if (kf <= kbl) taup[kf - 1] = taub;
    }

    bool icrilv = false;
    for (int kf = 2; kf < nlev; ++kf) {
        const int k = kf - 1;
        const int kp1 = k + 1;
        float brvf = 0.0f;
        if (kf >= kbl) {
            icrilv = icrilv || usqj[k] < 0.25f || velco[k] <= 0.0f;
            brvf = mpas_sqrt(mpas_max(bnv2[k], 1.0e-5f));
        }
        if (kf >= kbl && !ldrag && !icrilv && taup[k] > 0.0f) {
            const float temv = mpas_div(1.0f, velco[k]);
            float tem1 = mpas_div(coefm, dxy);
            tem1 = mpas_mul(tem1, mpas_add(rho[kp1], rho[k]));
            tem1 = mpas_mul(tem1, brvf);
            tem1 = mpas_mul(tem1, velco[k]);
            tem1 = mpas_mul(tem1, 0.5f);
            float hd = mpas_sqrt(mpas_div(taup[k], tem1));
            const float fro = mpas_mul(mpas_mul(brvf, hd), temv);
            const float tem2 = mpas_sqrt(usqj[k]);
            const float tem = mpas_add(1.0f, mpas_mul(tem2, fro));
            const float rim = mpas_div(
                mpas_mul(usqj[k], mpas_sub(1.0f, fro)), mpas_mul(tem, tem));
            if (rim <= 0.25f) {
                const float temc = mpas_add(2.0f, mpas_div(1.0f, tem2));
                hd = mpas_div(
                    mpas_mul(velco[k],
                        mpas_sub(mpas_mul(2.0f, mpas_sqrt(temc)), temc)),
                    brvf);
                taup[kp1] = mpas_mul(tem1, mpas_mul(hd, hd));
            } else {
                taup[kp1] = taup[k];
            }
        }
    }

    if (!ldrag) {
        int kblk = 0;
        float fbdpe = 0.0f;
        float zblk = 0.0f;
        for (int kf = nlev; kf >= 2; --kf) {
            if (kblk == 0 && kf <= kbl) {
                const int k = kf - 1;
                float contribution = mpas_mul(
                    bnv2[k], mpas_sub(zmid[(kbl - 1) * ncells + cell],
                        zmid[k * ncells + cell]));
                contribution = mpas_mul(contribution, delta_p[k]);
                contribution = mpas_div(contribution, gravity);
                contribution = mpas_div(contribution, rho[k]);
                fbdpe = mpas_add(fbdpe, contribution);
                const float fbdke = mpas_mul(0.5f,
                    mpas_add(mpas_mul(u1[k], u1[k]), mpas_mul(v1[k], v1[k])));
                if (fbdpe >= fbdke) {
                    kblk = kf < kbl ? kf : kbl;
                    zblk = mpas_sub(zmid[(kblk - 1) * ncells + cell], zmid[cell]);
                }
            }
        }
        if (kblk != 0) {
            const float fbdcd = mpas_max(
                mpas_sub(2.0f, mpas_div(1.0f, od)), 0.0f);
            float value = mpas_mul(0.5f, rhobar);
            value = mpas_mul(value, coefm);
            value = mpas_div(value, mpas_mul(dx, dx));
            value = mpas_mul(value, fbdcd);
            value = mpas_mul(value, dxyp);
            value = mpas_mul(value, olp);
            value = mpas_mul(value, zblk);
            value = mpas_mul(value, mpas_mul(ulow, ulow));
            taufb[0] = value;
            const float tautem = mpas_div(taufb[0], (float)(kblk - 1));
            for (int kf = 2; kf <= kblk; ++kf)
                taufb[kf - 1] = mpas_sub(taufb[kf - 2], tautem);
            for (int k = 0; k <= nlev; ++k)
                taup[k] = mpas_add(taup[k], taufb[k]);
        }
    }

    for (int k = 0; k < nlev; ++k) {
        taud[k] = mpas_div(
            mpas_mul(mpas_sub(taup[k + 1], taup[k]), gravity), delta_p[k]);
    }
    float dtfac = 1.0f;
    for (int kf = 1; kf < nlev; ++kf) {
        const int k = kf - 1;
        if (kf <= kbl && taud[k] != 0.0f) {
            const float limiter = mpas_abs(
                mpas_div(velco[k], mpas_mul(dt, taud[k])));
            dtfac = mpas_min(dtfac, limiter);
        }
    }

    float surface_u = 0.0f;
    float surface_v = 0.0f;
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        taud[k] = mpas_mul(taud[k], dtfac);
        const float du = mpas_mul(taud[k], xn);
        const float dv = mpas_mul(taud[k], yn);
        dtaux3d[i] = du;
        dtauy3d[i] = dv;
        surface_u = mpas_add(surface_u, mpas_mul(du, delta_p[k]));
        surface_v = mpas_add(surface_v, mpas_mul(dv, delta_p[k]));
        rublten_out[i] = mpas_add(rublten_in[i], du);
        rvblten_out[i] = mpas_add(rvblten_in[i], dv);
        rubldiff[i] = mpas_sub(rublten_out[i], rublten_in[i]);
        rvbldiff[i] = mpas_sub(rvblten_out[i], rvblten_in[i]);
        if (!isfinite(du) || !isfinite(dv) || !isfinite(rublten_out[i])
                || !isfinite(rvblten_out[i]) || !isfinite(rubldiff[i])
                || !isfinite(rvbldiff[i])) bad = true;
    }
    dusfcg[cell] = mpas_mul(mpas_div(-1.0f, gravity), surface_u);
    dvsfcg[cell] = mpas_mul(mpas_div(-1.0f, gravity), surface_v);
    if (!isfinite(dusfcg[cell]) || !isfinite(dvsfcg[cell]) || bad)
        atomicExch(invalid, 1);
}
