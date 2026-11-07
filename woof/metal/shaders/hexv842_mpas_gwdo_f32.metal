// MPAS v8.4.2 YSU GWDO column operator.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_gwdo_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


static inline float td_div(float a, float b) { return cu_div(a,b); }
static inline int td_nint(float x) { return int(round(x)); }
constant uint TD_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};
#define TD_ATAN_HI(i) as_type<float>(TD_ATANF_TAB[(i)])
#define TD_ATAN_LO(i) as_type<float>(TD_ATANF_TAB[4 + (i)])
#define TD_ATAN_T(i)  as_type<float>(TD_ATANF_TAB[8 + (i)])

static inline float td_atanf(float x) {
    uint hx = as_type<uint>(x);
    uint ix = hx & 0x7FFFFFFFu;
    int signed_hx = int(hx);
    int id;
    if (ix >= 0x4C000000u) {
        if (ix > 0x7F800000u) return x + x;
        if (signed_hx > 0) return TD_ATAN_HI(3) + TD_ATAN_LO(3);
        return -TD_ATAN_HI(3) - TD_ATAN_LO(3);
    }
    if (ix < 0x3EE00000u) {
        if (ix < 0x31000000u) return x;
        id = -1;
    } else {
        x = as_type<float>(ix);
        if (ix < 0x3F980000u) {
            if (ix < 0x3F300000u) {
                id = 0;
                x = td_div(2.0f * x - 1.0f, 2.0f + x);
            } else {
                id = 1;
                x = td_div(x - 1.0f, x + 1.0f);
            }
        } else if (ix < 0x401C0000u) {
            id = 2;
            x = td_div(x - 1.5f, 1.0f + 1.5f * x);
        } else {
            id = 3;
            x = td_div(-1.0f, x);
        }
    }
    float z = x * x;
    float w = z * z;
    float s1 = w * TD_ATAN_T(10);
    s1 = w * (TD_ATAN_T(8) + s1);
    s1 = w * (TD_ATAN_T(6) + s1);
    s1 = w * (TD_ATAN_T(4) + s1);
    s1 = w * (TD_ATAN_T(2) + s1);
    s1 = z * (TD_ATAN_T(0) + s1);
    float s2 = w * TD_ATAN_T(9);
    s2 = w * (TD_ATAN_T(7) + s2);
    s2 = w * (TD_ATAN_T(5) + s2);
    s2 = w * (TD_ATAN_T(3) + s2);
    s2 = w * (TD_ATAN_T(1) + s2);
    float s = s1 + s2;
    if (id < 0) return x - x * s;
    float r = TD_ATAN_HI(id) - ((x * s - TD_ATAN_LO(id)) - x);
    return (signed_hx < 0) ? -r : r;
}

#undef TD_ATAN_HI
#undef TD_ATAN_LO
#undef TD_ATAN_T

static inline float td_atan2f(float y, float x) {
    const float tiny = 1.0e-30f;
    const float pi_o_4 = as_type<float>(0x3F490FDBu);
    const float pi_o_2 = as_type<float>(0x3FC90FDBu);
    const float pi = as_type<float>(0x40490FDBu);
    const float pi_lo = as_type<float>(0xB3BBBD2Eu);
    int hx = as_type<int>(x), hy = as_type<int>(y);
    int ix = hx & 0x7FFFFFFF, iy = hy & 0x7FFFFFFF;
    if (ix > 0x7F800000 || iy > 0x7F800000) return x + y;
    if (hx == 0x3F800000) return td_atanf(y);
    int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if (iy == 0) {
        if (m <= 1) return y;
        return (m == 2) ? pi + tiny : -pi - tiny;
    }
    if (ix == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    if (ix == 0x7F800000) {
        if (iy == 0x7F800000) {
            switch (m) {
            case 0: return pi_o_4 + tiny;
            case 1: return -pi_o_4 - tiny;
            case 2: return 3.0f * pi_o_4 + tiny;
            default: return -3.0f * pi_o_4 - tiny;
            }
        }
        switch (m) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return pi + tiny;
        default: return -pi - tiny;
        }
    }
    if (iy == 0x7F800000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    int k = (iy - ix) >> 23;
    float z;
    if (k > 60) z = pi_o_2 + 0.5f * pi_lo;
    else if (hx < 0 && k < -60) z = 0.0f;
    else z = td_atanf(as_type<float>(as_type<uint>(td_div(y, x)) & 0x7FFFFFFFu));
    switch (m) {
    case 0: return z;
    case 1: return -z;
    case 2: return pi - (z - pi_lo);
    default: return (z - pi_lo) - pi;
    }
}

kernel void hexv842_mpas_gwdo_f32(
    constant Hexv842MpasGwdoF32Resources &r [[buffer(0)]],
    constant Hexv842MpasGwdoF32Params &p [[buffer(1)]],
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

    const int cell = int(tid);
    if (cell >= ncells) return;
    const float md = mesh_density[cell];
    const float oa_input[4] = {oa1[cell], oa2[cell], oa3[cell], oa4[cell]};
    const float ol_input[4] = {ol1[cell], ol2[cell], ol3[cell], ol4[cell]};

    device float *vtj = r.scratch + (0 * ncells + cell) * (nlev + 1);
    device float *vtk = r.scratch + (1 * ncells + cell) * (nlev + 1);
    device float *rho = r.scratch + (2 * ncells + cell) * (nlev + 1);
    device float *delta_p = r.scratch + (3 * ncells + cell) * (nlev + 1);
    device float *u1 = r.scratch + (4 * ncells + cell) * (nlev + 1);
    device float *v1 = r.scratch + (5 * ncells + cell) * (nlev + 1);
    device float *usqj = r.scratch + (6 * ncells + cell) * (nlev + 1);
    device float *bnv2 = r.scratch + (7 * ncells + cell) * (nlev + 1);
    device float *velco = r.scratch + (8 * ncells + cell) * (nlev + 1);
    device float *taup = r.scratch + (9 * ncells + cell) * (nlev + 1);
    device float *taufb = r.scratch + (10 * ncells + cell) * (nlev + 1);
    device float *taud = r.scratch + (11 * ncells + cell) * (nlev + 1);
    for (int k = 0; k < nlev; ++k) {
        usqj[k] = 0.0f;
        bnv2[k] = 0.0f;
        taud[k] = 0.0f;
    }
    for (int k = 0; k <= nlev; ++k) {
        taup[k] = 0.0f;
        taufb[k] = 0.0f;
    }

    const float dx = mpas_div(len_disp, gfk_pow(md, 0.25f));
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
        u1[k] = mpas_sub(mpas_mul(u[i], 1.0f), mpas_mul(v[i], 0.0f));
        v1[k] = mpas_add(mpas_mul(u[i], 0.0f), mpas_mul(v[i], 1.0f));
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
    const float wdir = mpas_add(td_atan2f(ubar, vbar), pi);
    const int nearest = td_nint(mpas_mul(fdir, wdir));
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
        float efact = gfk_pow(mpas_add(oa, 2.0f), mpas_mul(0.8f, fr));
        efact = mpas_min(mpas_max(efact, 0.0f), 10.0f);
        coefm = gfk_pow(mpas_add(1.0f, ol), mpas_add(oa, 1.0f));
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
                taup[kp1] = mpas_mul(mpas_mul(tem1, hd), hd);
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
        dtaux3d[i] = mpas_add(mpas_mul(du, 1.0f), mpas_mul(dv, 0.0f));
        dtauy3d[i] = mpas_add(mpas_mul(-du, 0.0f), mpas_mul(dv, 1.0f));
        surface_u = mpas_add(surface_u, mpas_mul(du, delta_p[k]));
        surface_v = mpas_add(surface_v, mpas_mul(dv, delta_p[k]));
        rublten_out[i] = mpas_add(mpas_add(rublten_in[i], mpas_mul(du, 1.0f)), mpas_mul(dv, 0.0f));
        rvblten_out[i] = mpas_add(mpas_sub(rvblten_in[i], mpas_mul(du, 0.0f)), mpas_mul(dv, 1.0f));
        rubldiff[i] = mpas_sub(rublten_out[i], rublten_in[i]);
        rvbldiff[i] = mpas_sub(rvblten_out[i], rvblten_in[i]);
    }
    const float stress_u = mpas_mul(mpas_div(-1.0f, gravity), surface_u);
    const float stress_v = mpas_mul(mpas_div(-1.0f, gravity), surface_v);
    dusfcg[cell] = mpas_add(mpas_mul(stress_u, 1.0f), mpas_mul(stress_v, 0.0f));
    dvsfcg[cell] = mpas_add(mpas_mul(-stress_u, 0.0f), mpas_mul(stress_v, 1.0f));
}
