#include <metal_stdlib>
#include "p3_kloopmain_bindings.metalh"
#include "p3_constants.metalh"
#include "p3_math.metalh"
#include "soft_gfk_tgamma.metalh"
using namespace metal;
// P3's k_loop_main process rates, conservation limits, and prognostic update
// (module_mp_p3.F:2449-3927; p3.cu:p3_step_kloopmain_levels). One invocation
// owns a whole column and keeps the source level order. Native MSL math is
// the ordinary-FP32 translation; WRF-exact binary64/libm behavior is not
// claimed by this stage.
#pragma clang fp contract(off)

constant int P3_ISIZE=50, P3_DENSIZE=5, P3_RIMSIZE=4, P3_RCOLLSIZE=30, P3_TABSIZE=14;
constant float P3_MW=0x1.26e9780000000p-6f;
constant float P3_RR=0x1.0a32ca0000000p+3f, P3_BACT=0x1.4efb0a0000000p-1f;
constant float P3_INV_RM1=0x1.312d000000000p+24f, P3_SIG1=2.0f, P3_NANEW1=0x1.1e1a300000000p+28f;
constant float P3_INV_RM2=0x1.7799d80000000p+19f, P3_SIG2=2.5f, P3_NANEW2=0.0f;
static inline float p3_div_rn(float a,float b){return a/b;}
#define ITAB(jj, ii, i2, idx) (itab[(((((jj) * P3_RIMSIZE + (ii)) * P3_ISIZE + (i2)) * P3_TABSIZE) + (idx))])
#define ICOLL(jj, ii, i2, j2, idx) (icoll[(((((jj) * P3_RIMSIZE + (ii)) * P3_ISIZE + (i2)) * P3_RCOLLSIZE + (j2)) * 2) + (idx)])
#define RTAB(tb, a, b) ((tb)[(a) * 10 + (b)])

static inline float p3_erf(float x)
{
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * ax);
    const float tau = t * exp(-ax * ax - 1.26551223f + t * (
        1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (
        -0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (
        1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return sign * (1.0f - tau);
}

static inline float p3_gam(float x)
{
    return soft_gfk_tgammaf(x);
}

static inline int p3_int(float x) { return int(x); }

static inline float p3_polysvp1(float t, int i_type)
{
    if (i_type == 1 && t < 273.15f) {
        if (t >= 195.8f) {
            float dt = t - 273.15f;
            float a0i = 6.11147274f,      a1i = 0.503160820f,
                  a2i = 0.188439774e-1f,  a3i = 0.420895665e-3f,
                  a4i = 0.615021634e-5f,  a5i = 0.602588177e-7f,
                  a6i = 0.385852041e-9f,  a7i = 0.146898966e-11f,
                  a8i = 0.252751365e-14f;
            float out = a0i + dt * (a1i + dt * (a2i + dt * (a3i + dt * (a4i
                        + dt * (a5i + dt * (a6i + dt * (a7i + a8i * dt)))))));
            return out * 100.0f;
        }
        return pow(10.0f, -9.09718f * (273.16f / t - 1.0f)
                            - 3.56654f * p3_log10(273.16f / t)
                            + 0.876793f * (1.0f - p3_div_rn(t, 273.16f))
                            + p3_log10(6.1071f)) * 100.0f;
    }
    if (t >= 202.0f) {
        float dt = t - 273.15f;
        float a0 = 6.11239921f,       a1 = 0.443987641f,
              a2 = 0.142986287e-1f,   a3 = 0.264847430e-3f,
              a4 = 0.302950461e-5f,   a5 = 0.206739458e-7f,
              a6 = 0.640689451e-10f,  a7 = -0.952447341e-13f,
              a8 = -0.976195544e-15f;
        float out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt * (a4
                    + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt)))))));
        return out * 100.0f;
    }
    return pow(10.0f, -7.90298f * (373.16f / t - 1.0f)
                        + 5.02808f * p3_log10(373.16f / t)
                        - 1.3816e-7f * (pow(10.0f, 11.344f
                                        * (1.0f - p3_div_rn(t, 373.16f))) - 1.0f)
                        + 8.1328e-3f * (pow(10.0f, -3.49149f
                                        * (373.16f / t - 1.0f)) - 1.0f)
                        + p3_log10(1013.246f)) * 100.0f;
}

static inline float p3_qv_sat(float t_atm, float p_atm, int i_wrt,
                              device const float *pc)
{
    float e_pres = p3_polysvp1(t_atm, i_wrt);
    return P3_EP_2 * e_pres / fmax(1.0e-3f, p_atm - e_pres);
}

static inline float p3_access_lookup_table(device const float * itab,
                                        int dumjj, int dumii, int dumi,
                                        int index, float dum1, float dum4,
                                        float dum5)
{
    float iproc1, gproc1, tmp1, tmp2;
    iproc1 = ITAB(dumjj-1, dumii-1, dumi-1, index-1)
        + (dum1 - float(dumi)) * (ITAB(dumjj-1, dumii-1, dumi, index-1)
                                  - ITAB(dumjj-1, dumii-1, dumi-1, index-1));
    gproc1 = ITAB(dumjj-1, dumii, dumi-1, index-1)
        + (dum1 - float(dumi)) * (ITAB(dumjj-1, dumii, dumi, index-1)
                                  - ITAB(dumjj-1, dumii, dumi-1, index-1));
    tmp1 = iproc1 + (dum4 - float(dumii)) * (gproc1 - iproc1);
    iproc1 = ITAB(dumjj, dumii-1, dumi-1, index-1)
        + (dum1 - float(dumi)) * (ITAB(dumjj, dumii-1, dumi, index-1)
                                  - ITAB(dumjj, dumii-1, dumi-1, index-1));
    gproc1 = ITAB(dumjj, dumii, dumi-1, index-1)
        + (dum1 - float(dumi)) * (ITAB(dumjj, dumii, dumi, index-1)
                                  - ITAB(dumjj, dumii, dumi-1, index-1));
    tmp2 = iproc1 + (dum4 - float(dumii)) * (gproc1 - iproc1);
    return tmp1 + (dum5 - float(dumjj)) * (tmp2 - tmp1);
}

static inline float p3_coll_pair(device const float * icoll,
                                              int jj, int ii, int dumi,
                                              int dumj, int index,
                                              float dum1, float dum3)
{
    float dproc1 = ICOLL(jj, ii, dumi-1, dumj-1, index-1)
        + (dum1 - float(dumi)) * (ICOLL(jj, ii, dumi, dumj-1, index-1)
                                  - ICOLL(jj, ii, dumi-1, dumj-1, index-1));
    float dproc2 = ICOLL(jj, ii, dumi-1, dumj, index-1)
        + (dum1 - float(dumi)) * (ICOLL(jj, ii, dumi, dumj, index-1)
                                  - ICOLL(jj, ii, dumi-1, dumj, index-1));
    return dproc1 + (dum3 - float(dumj)) * (dproc2 - dproc1);
}

static inline float p3_access_lookup_table_coll(device const float * icoll,
                                             int dumjj, int dumii, int dumj,
                                             int dumi, int index, float dum1,
                                             float dum3, float dum4, float dum5)
{
    float iproc1 = p3_coll_pair(icoll, dumjj-1, dumii-1, dumi, dumj, index,
                                dum1, dum3);
    float gproc1 = p3_coll_pair(icoll, dumjj-1, dumii, dumi, dumj, index,
                                dum1, dum3);
    float tmp1 = iproc1 + (dum4 - float(dumii)) * (gproc1 - iproc1);
    iproc1 = p3_coll_pair(icoll, dumjj, dumii-1, dumi, dumj, index,
                          dum1, dum3);
    gproc1 = p3_coll_pair(icoll, dumjj, dumii, dumi, dumj, index, dum1, dum3);
    float tmp2 = iproc1 + (dum4 - float(dumii)) * (gproc1 - iproc1);
    return tmp1 + (dum5 - float(dumjj)) * (tmp2 - tmp1);
}

static inline void p3_find_lt_1a(float qitot, float nitot, float qirim, float rhop,
                              thread int* dumi, thread int* dumjj, thread int* dumii,
                              thread float * dum1, thread float * dum4, thread float * dum5)
{
    float d1 = (p3_log10(qitot / nitot) + 18.0f) * 3.444606f - 10.0f;
    int di = p3_int(d1);
    d1 = fmin(d1, float(P3_ISIZE));
    d1 = fmax(d1, 1.0f);
    di = max(1, di);
    di = min(P3_ISIZE - 1, di);

    float d4 = (qirim / qitot) * 3.0f + 1.0f;
    int dii = p3_int(d4);
    d4 = fmin(d4, float(P3_RIMSIZE));
    d4 = fmax(d4, 1.0f);
    dii = max(1, dii);
    dii = min(P3_RIMSIZE - 1, dii);

    float d5;
    if (rhop <= 650.0f) d5 = (rhop - 50.0f) * 0.005f + 1.0f;
    else                d5 = (rhop - 650.0f) * 0.004f + 4.0f;
    int djj = p3_int(d5);
    d5 = fmin(d5, float(P3_DENSIZE));
    d5 = fmax(d5, 1.0f);
    djj = max(1, djj);
    djj = min(P3_DENSIZE - 1, djj);

    *dumi = di; *dumjj = djj; *dumii = dii;
    *dum1 = d1; *dum4 = d4; *dum5 = d5;
}

static inline void p3_find_lt_1b(float qr, float nr, thread int* dumj,
                                 thread float * dum3,
                                 device const float *pc)
{
    if (qr >= P3_QSMALL && nr > 0.0f) {
        float dumlr = pow(qr / (P3_PI * P3_RHOW * nr), P3_THRD);
        float d3 = (p3_log10(1.0f * dumlr) + 5.0f) * 10.70415f;
        int dj = p3_int(d3);
        d3 = fmin(d3, float(P3_RCOLLSIZE));
        d3 = fmax(d3, 1.0f);
        dj = max(1, dj);
        dj = min(P3_RCOLLSIZE - 1, dj);
        *dumj = dj; *dum3 = d3;
    } else {
        *dumj = 1; *dum3 = 1.0f;
    }
}

static inline void p3_find_lt_3(float mu_r, float lamr, thread int* dumii, thread int* dumjj,
                             thread float * rdumii, thread float * rdumjj,
                             device const float *pc)
{
    float dum1 = (mu_r + 1.0f) / lamr;
    float rii, inv_dum3;
    int dii;
    if (dum1 <= 195.0e-6f) {
        inv_dum3 = 0.1f;
        rii = (dum1 * 1.0e6f + 5.0f) * inv_dum3;
        rii = fmax(rii, 1.0f);
        rii = fmin(rii, 20.0f);
        dii = p3_int(rii);
        dii = max(dii, 1);
        dii = min(dii, 20);
    } else {
        inv_dum3 = P3_THRD * 0.1f;
        rii = (dum1 * 1.0e6f - 195.0f) * inv_dum3 + 20.0f;
        rii = fmax(rii, 20.0f);
        rii = fmin(rii, 300.0f);
        dii = p3_int(rii);
        dii = max(dii, 20);
        dii = min(dii, 299);
    }
    (void)inv_dum3;
    float rjj = mu_r + 1.0f;
    rjj = fmax(rjj, 1.0f);
    rjj = fmin(rjj, 10.0f);
    int djj = p3_int(rjj);
    djj = max(djj, 1);
    djj = min(djj, 9);
    *dumii = dii; *dumjj = djj; *rdumii = rii; *rdumjj = rjj;
}

static inline float p3_rain_table(device const float * tb,
                                               int dumii, int dumjj,
                                               float rdumii, float rdumjj)
{
    float dum1 = RTAB(tb, dumii-1, dumjj-1)
        + (rdumii - float(dumii)) * (RTAB(tb, dumii, dumjj-1)
                                     - RTAB(tb, dumii-1, dumjj-1));
    float dum2 = RTAB(tb, dumii-1, dumjj)
        + (rdumii - float(dumii)) * (RTAB(tb, dumii, dumjj)
                                     - RTAB(tb, dumii-1, dumjj));
    return dum1 + (rdumjj - float(dumjj)) * (dum2 - dum1);
}

static inline void p3_get_cloud_dsd2(thread float * nc_grd, float qc_grd, float rho,
                                  float iscf, thread float * mu_c_o, thread float * lamc_o,
                                  thread float * cdist_o, thread float * cdist1_o,
                                  device const float *pc)
{
    float qc = qc_grd * iscf;
    if (qc >= P3_QSMALL) {
        float nc = *nc_grd * iscf;
        nc = fmax(nc, P3_NSMALL);
        float mu_c = 0.0005714f * (nc * 1.0e-6f * rho) + 0.2714f;
        mu_c = 1.0f / (mu_c * mu_c) - 1.0f;            // :6662, integer **2
        mu_c = fmax(mu_c, 2.0f);
        mu_c = fmin(mu_c, 15.0f);
        // iparam = 3 (:193), so nu is 0 and dnu is unread (:6666-6670).
        float lamc = pow(P3_CONS1 * nc * (mu_c + 3.0f) * (mu_c + 2.0f)
                           * (mu_c + 1.0f) / qc, P3_THRD);
        float lammin = (mu_c + 1.0f) * 2.5e4f;
        float lammax = (mu_c + 1.0f) * 1.0e6f;
        if (lamc < lammin) {
            lamc = lammin;
            nc = 6.0f * ((lamc * lamc) * lamc) * qc
                 / (P3_PI * P3_RHOW * (mu_c + 3.0f) * (mu_c + 2.0f)
                    * (mu_c + 1.0f));                  // :6681, integer **3
        } else if (lamc > lammax) {
            lamc = lammax;
            nc = 6.0f * ((lamc * lamc) * lamc) * qc
                 / (P3_PI * P3_RHOW * (mu_c + 3.0f) * (mu_c + 2.0f)
                    * (mu_c + 1.0f));                  // :6684, integer **3
        }
        *cdist_o = nc * (mu_c + 1.0f) / lamc;
        *cdist1_o = nc / p3_gam(mu_c + 1.0f);
        *nc_grd = nc / iscf;
        *mu_c_o = mu_c; *lamc_o = lamc;
    } else {
        *mu_c_o = 0.0f; *lamc_o = 0.0f; *cdist_o = 0.0f; *cdist1_o = 0.0f;
    }
}

static inline void p3_get_rain_dsd2(thread float * nr_grd, float qr_grd, float ispf,
                                 thread float * mu_r_o, thread float * lamr_o,
                                 thread float * cdistr_o, thread float * logn0r_o,
                                 device const float *pc)
{
    float qr = qr_grd * ispf;
    if (qr >= P3_QSMALL) {
        float nr = *nr_grd * ispf;
        nr = fmax(nr, P3_NSMALL);
        float mu_r = P3_MU_R_CONSTANT;
        float lamr = pow(P3_CONS1 * nr * (mu_r + 3.0f) * (mu_r + 2.0f)
                           * (mu_r + 1.0f) / qr, P3_THRD);
        float lammax = (mu_r + 1.0f) * 1.0e5f;
        float lammin = (mu_r + 1.0f) * P3_INV_DRMAX;
        if (lamr < lammin) {
            lamr = lammin;
            nr = p3_div_rn(exp(3.0f * log(lamr) + log(qr)
                       + log(p3_gam(mu_r + 1.0f))
                       - log(p3_gam(mu_r + 4.0f))), P3_CONS1);
        } else if (lamr > lammax) {
            lamr = lammax;
            nr = p3_div_rn(exp(3.0f * log(lamr) + log(qr)
                       + log(p3_gam(mu_r + 1.0f))
                       - log(p3_gam(mu_r + 4.0f))), P3_CONS1);
        }
        *logn0r_o = p3_log10(nr) + (mu_r + 1.0f) * p3_log10(lamr)
                    - p3_log10(p3_gam(mu_r + 1.0f));
        *cdistr_o = nr / p3_gam(mu_r + 1.0f);
        *nr_grd = nr / ispf;
        *mu_r_o = mu_r; *lamr_o = lamr;
    } else {
        *mu_r_o = 0.0f; *lamr_o = 0.0f; *cdistr_o = 0.0f; *logn0r_o = 0.0f;
    }
}

static inline void p3_calc_bulk_rho_rime(float qi_tot, thread float * qi_rim,
                                      thread float * bi_rim, thread float * rho_rime_o,
                                      device const float *pc)
{
    float qir = *qi_rim, bir = *bi_rim, rho_rime;
    if (bir >= 1.0e-15f) {
        rho_rime = qir / bir;
        if (rho_rime < P3_RHO_RIMEMIN) {
            rho_rime = P3_RHO_RIMEMIN;
            bir = p3_div_rn(qir, rho_rime);
        } else if (rho_rime > P3_RHO_RIMEMAX) {
            rho_rime = P3_RHO_RIMEMAX;
            bir = p3_div_rn(qir, rho_rime);
        }
    } else {
        qir = 0.0f; bir = 0.0f; rho_rime = 0.0f;
    }
    if (qir > qi_tot && rho_rime > 0.0f) {
        qir = qi_tot;
        bir = qir / rho_rime;
    }
    if (qir < P3_QSMALL) { qir = 0.0f; bir = 0.0f; }
    *qi_rim = qir; *bi_rim = bir; *rho_rime_o = rho_rime;
}

static inline float p3_impose_max_ni(float nitot, float inv_rho,
                                     device const float *pc)
{
    if (nitot >= 1.0e-20f) {
        float dum = P3_MAX_TOTAL_NI * inv_rho / nitot;
        nitot = nitot * fmin(dum, 1.0f);
    }
    return nitot;
}

static inline void p3_activate_droplets(int log_predictNc, float sup_cld, int it,
                                     float t, float pres, float qc, float nc,
                                     float qv_cld, float inv_rho, float iscf,
                                     float scf, float odt,
                                     thread float * qcnuc, thread float * ncnuc,
                                     device const float *pc, float xxlv)
{
    if (!log_predictNc && sup_cld > 1.0e-6f && it > 1) {
        float dum = P3_NCCNST * inv_rho * P3_CONS7 - qc;
        dum = fmax(0.0f, dum * iscf);
        float dumqvs = p3_qv_sat(t, pres, 0, pc);
        float dqsdT = xxlv * dumqvs / (P3_RV * t * t);
        float ab = 1.0f + dqsdT * xxlv * P3_INV_CP;
        dum = fmax(0.0f, fmin(dum, (qv_cld - dumqvs) / ab));
        *qcnuc = dum * odt * scf;
    }
    if (log_predictNc) {
        if (sup_cld > 1.0e-6f) {
            float dum1 = 1.0f / pow(P3_BACT, 0.5f);
            float sigvl = 0.0761f - 1.55e-4f * (t - 273.15f);
            float aact = 2.0f * P3_MW / (P3_RHOW * P3_RR * t) * sigvl;
            float sm1 = 2.0f * dum1 * pow(aact * P3_THRD * P3_INV_RM1, 1.5f);
            float sm2 = 2.0f * dum1 * pow(aact * P3_THRD * P3_INV_RM2, 1.5f);
            float uu1 = 2.0f * log(sm1 / sup_cld) / (4.242f * log(P3_SIG1));
            float uu2 = 2.0f * log(sm2 / sup_cld) / (4.242f * log(P3_SIG2));
            // derf is the DOUBLE erf in the authority (:3324-3325).
            dum1 = P3_NANEW1 * 0.5f * (1.0f - p3_erf(uu1));
            float dum2 = P3_NANEW2 * 0.5f * (1.0f - p3_erf(uu2));
            dum2 = fmin(P3_NANEW1 + P3_NANEW2, dum1 + dum2);
            dum2 = (dum2 - nc * iscf) * odt * scf;
            dum2 = fmax(0.0f, dum2);
            *ncnuc = dum2;
            if (it <= 1) *qcnuc = 0.0f;
            else         *qcnuc = *ncnuc * P3_CONS7;
        }
    }
}
static inline int p3_step_kloopmain_levels(constant P3KloopmainResources &r, constant P3KloopmainParams &p, uint column)
{
    device const float *pc = r.constants;
    const int ncol = p.ny * p.nx; const int nk = p.nz;
    const float dt = p.dt; const int it = p.it;
    const int log_predictNc = p.log_predict_nc;
    const float clbfact_dep = p.clbfact_dep, clbfact_sub = p.clbfact_sub;
    const float xxlv = p.xxlv, xxls = p.xxls, xlf = p.xlf;
    const int i = int(column), k_start = 0, k_stride = 1;
    device float *FLG = r.flags;
    device const float *itab = r.itab, *icoll = r.icoll, *revap = r.revap;
    device float *rho_buf=r.rho, *inv_rho_buf=r.inv_rho, *qvs_buf=r.qvs, *qvi_buf=r.qvi;
    device float *sup_buf=r.sup, *supi_buf=r.supi, *rhofacr_buf=r.rhofacr, *rhofaci_buf=r.rhofaci;
    device float *acn_buf=r.acn, *temperature_buf=r.temperature, *tmparr1_buf=r.tmparr1, *qv_cld_buf=r.qv_cld;
    // goto 333 (:2439) -- nothing to do in this column.
    if (FLG[i] == 0.0f && FLG[ncol + i] == 0.0f) return 0;

    device float * qc=r.qc;
    device float * nc=r.nc;
    device float * qr=r.qr;
    device float * nr=r.nr;
    device float * qi=r.qi;
    device float * qir=r.qir;
    device float * ni=r.ni;
    device float * qib=r.qib;
    device float * th=r.theta;
    device float * qv=r.qv;
    device const float * qv_old=r.qv_old;
    device const float * th_old=r.th_old;
    device float * ssat=r.ssat;
    device const float * pres=r.pressure;

    const float odt = 1.0f / dt;                                     // :2261
    // SCPF off-branch (:1889-1899) -- constant 1 / 0, kept named so the
    // Sundqvist branch has somewhere to attach.
    const float iscf = 1.0f, scf = 1.0f, spf = 1.0f, ispf = 1.0f;
    const float spf_clr = 0.0f;
    // :2290, assigned once outside the loops.  It is only ever READ when
    // qccol > 0, which implies the if/else below just wrote it, so the
    // authority's carry across columns is unobservable and this per-column
    // initialisation is equivalent.
    float rhorime_c = 400.0f;
    int hydrometeors_present = 0;

    for (int k = k_start; k < nk; k += k_stride) {
        const float t = temperature_buf[(k) * ncol + i];
        const float rho = rho_buf[(k) * ncol + i];
        const float inv_rho = inv_rho_buf[(k) * ncol + i];
        const float qvs = qvs_buf[(k) * ncol + i];
        const float qvi = qvi_buf[(k) * ncol + i];
        const float sup = sup_buf[(k) * ncol + i];
        const float supi = supi_buf[(k) * ncol + i];
        const float rhofaci = rhofaci_buf[(k) * ncol + i];
        const float acn = acn_buf[(k) * ncol + i];
        const float qv_cld = qv_cld_buf[(k) * ncol + i];
        const float invexn = 1.0f / tmparr1_buf[(k) * ncol + i];
        const float t_old = th_old[(k) * ncol + i] * tmparr1_buf[(k) * ncol + i];

        // goto 555 (:2462-2464): dry, subsaturated, no hydrometeors.
        int log_exitlevel = !(qc[(k) * ncol + i] >= P3_QSMALL || qr[(k) * ncol + i] >= P3_QSMALL
                              || qi[(k) * ncol + i] >= P3_QSMALL);
        if (log_exitlevel && ((t < 273.15f && supi < -0.05f)
                              || (t >= 273.15f && sup < -0.05f))) continue;

        // process rates (:2467-2485), nCat = 1 so all are scalars
        float qcacc = 0.0f, qrevp = 0.0f, qccon = 0.0f;
        float qcaut = 0.0f, qcevp = 0.0f, qrcon = 0.0f;
        float ncacc = 0.0f, ncnuc = 0.0f, ncslf = 0.0f;
        float ncautc = 0.0f, qcnuc = 0.0f, nrslf = 0.0f;
        float nrevp = 0.0f, ncautr = 0.0f;
        float qchetc = 0.0f, qisub = 0.0f, nrshdr = 0.0f;
        float qcheti = 0.0f, qrcol = 0.0f, qcshd = 0.0f;
        float qrhetc = 0.0f, qimlt = 0.0f, qccol = 0.0f;
        float qrheti = 0.0f, qinuc = 0.0f, nimlt = 0.0f;
        float nchetc = 0.0f, nccol = 0.0f, ncshdc = 0.0f;
        float ncheti = 0.0f, nrcol = 0.0f, nislf = 0.0f;
        float nrhetc = 0.0f, ninuc = 0.0f, qidep = 0.0f;
        float nrheti = 0.0f, nisub = 0.0f, qwgrth = 0.0f;
        float qrmul = 0.0f, nimul = 0.0f;
        int log_wetgrowth = 0;

        // goto 444 (:2523-2528)
        log_exitlevel = !(qc[(k) * ncol + i] >= P3_QSMALL || qr[(k) * ncol + i] >= P3_QSMALL
                          || qi[(k) * ncol + i] >= P3_QSMALL);

        float dqsdT = 0.0f, epsi = 0.0f, epsi_tot = 0.0f;
        float f1pr04 = 0.0f, f1pr05 = 0.0f, f1pr09 = 0.0f;
        float f1pr10 = 0.0f, f1pr14 = 0.0f;
        float f1pr02 = 0.0f, f1pr03 = 0.0f;
        float f1pr07 = -99.0f, f1pr08 = -99.0f;
        float mu_c = 0.0f, lamc = 0.0f, cdist = 0.0f, cdist1 = 0.0f;
        float mu_r = 0.0f, lamr = 0.0f, cdistr = 0.0f, logn0r = 0.0f;
        float abi = 1.0f, ab = 1.0f, oabi = 1.0f, oxx = 0.0f, aaa = 0.0f;
        float dum, dum1, dum2, tmp1, tmp2;

        if (!log_exitlevel) {
            float mu = 1.496e-6f * pow(t, 1.5f) / (t + 120.0f);    // :2531
            float dv = 8.794e-5f * pow(t, 1.81f) / pres[(k) * ncol + i];    // :2532
            float sc = mu / (rho * dv);                              // :2533
            dum = 1.0f / (P3_RV * (t * t));                          // :2534 int **2
            dqsdT = xxlv * qvs * dum;                                // :2535
            float dqsidT = xxls * qvi * dum;                          // :2536
            ab = 1.0f + dqsdT * xxlv * P3_INV_CP;                    // :2537
            abi = 1.0f + dqsidT * xxls * P3_INV_CP;                  // :2538
            float kap = 1.414e3f * mu;                               // :2539
            float eii;
            if (t < 253.15f)      eii = 0.001f;                      // :2547
            else if (t < 273.15f) eii = 0.001f + p3_div_rn((t - 253.15f)
                                         * (0.3f - 0.001f), 20.0f);
            else                  eii = 0.3f;

            {   float ncg = nc[(k) * ncol + i];
                p3_get_cloud_dsd2(&ncg, qc[(k) * ncol + i], rho, iscf,
                                  &mu_c, &lamc, &cdist, &cdist1, pc);
                nc[(k) * ncol + i] = ncg; }
            {   float nrg = nr[(k) * ncol + i];
                p3_get_rain_dsd2(&nrg, qr[(k) * ncol + i], ispf,
                                 &mu_r, &lamr, &cdistr, &logn0r, pc);
                nr[(k) * ncol + i] = nrg; }

            epsi_tot = 0.0f;                                         // :2564
            ni[(k) * ncol + i] = p3_impose_max_ni(ni[(k) * ncol + i], inv_rho, pc); // :2566

            float eii_fact = 1.0f;
            if (qi[(k) * ncol + i] >= P3_QSMALL) {                            // :2570
                ni[(k) * ncol + i] = fmax(ni[(k) * ncol + i], P3_NSMALL);
                nr[(k) * ncol + i] = fmax(nr[(k) * ncol + i], P3_NSMALL);
                float rhop;
                {   float qq = qir[(k) * ncol + i], bb = qib[(k) * ncol + i];
                    p3_calc_bulk_rho_rime(qi[(k) * ncol + i], &qq, &bb, &rhop, pc);
                    qir[(k) * ncol + i] = qq; qib[(k) * ncol + i] = bb; }
                int dumi, dumjj, dumii, dumj;
                float d1, d4, d5, d3;
                p3_find_lt_1a(qi[(k) * ncol + i], ni[(k) * ncol + i], qir[(k) * ncol + i], rhop,
                              &dumi, &dumjj, &dumii, &d1, &d4, &d5);
                p3_find_lt_1b(qr[(k) * ncol + i], nr[(k) * ncol + i], &dumj, &d3, pc);
                f1pr02 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                                d1, d4, d5);
                f1pr03 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 3,
                                                d1, d4, d5);
                f1pr04 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 4,
                                                d1, d4, d5);
                f1pr05 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 5,
                                                d1, d4, d5);
                f1pr09 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                                d1, d4, d5);
                f1pr10 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                                d1, d4, d5);
                f1pr14 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 10,
                                                d1, d4, d5);
                if (qr[(k) * ncol + i] >= P3_QSMALL) {
                    f1pr07 = p3_access_lookup_table_coll(icoll, dumjj, dumii,
                                                         dumj, dumi, 1,
                                                         d1, d3, d4, d5);
                    f1pr08 = p3_access_lookup_table_coll(icoll, dumjj, dumii,
                                                         dumj, dumi, 2,
                                                         d1, d3, d4, d5);
                } else {
                    f1pr07 = -99.0f;
                    f1pr08 = -99.0f;
                }
                ni[(k) * ncol + i] = fmin(ni[(k) * ncol + i], f1pr09 * qi[(k) * ncol + i]);     // :2649
                ni[(k) * ncol + i] = fmax(ni[(k) * ncol + i], f1pr10 * qi[(k) * ncol + i]);     // :2650
                if (qir[(k) * ncol + i] > 0.0f) {                              // :2665
                    tmp1 = qir[(k) * ncol + i] / qi[(k) * ncol + i];
                    if (tmp1 < 0.6f)      eii_fact = 1.0f;
                    else if (tmp1 < 0.9f) eii_fact = 1.0f
                                                     - p3_div_rn((tmp1 - 0.6f), 0.3f);
                    else                  eii_fact = 0.0f;
                } else {
                    eii_fact = 1.0f;
                }
            }

            // collection of droplets (:2697-2710)
            if (qi[(k) * ncol + i] >= P3_QSMALL && qc[(k) * ncol + i] >= P3_QSMALL
                && t <= 273.15f) {
                qccol = rhofaci * f1pr04 * qc[(k) * ncol + i] * P3_ECI * rho
                        * ni[(k) * ncol + i] * iscf;
                nccol = rhofaci * f1pr04 * nc[(k) * ncol + i] * P3_ECI * rho
                        * ni[(k) * ncol + i] * iscf;
            }
            if (qi[(k) * ncol + i] >= P3_QSMALL && qc[(k) * ncol + i] >= P3_QSMALL
                && t > 273.15f) {
                qcshd = rhofaci * f1pr04 * qc[(k) * ncol + i] * P3_ECI * rho
                        * ni[(k) * ncol + i] * iscf;
                nccol = rhofaci * f1pr04 * nc[(k) * ncol + i] * P3_ECI * rho
                        * ni[(k) * ncol + i] * iscf;
                ncshdc = qcshd * 1.923e6f;
            }

            // collection of rain (:2725-2748)
            if (qi[(k) * ncol + i] >= P3_QSMALL && qr[(k) * ncol + i] >= P3_QSMALL
                && t <= 273.15f) {
                qrcol = pow(10.0f, f1pr08 + logn0r) * rho * rhofaci
                        * P3_ERI * ni[(k) * ncol + i] * iscf * (spf - spf_clr);
                nrcol = pow(10.0f, f1pr07 + logn0r) * rho * rhofaci
                        * P3_ERI * ni[(k) * ncol + i] * iscf * (spf - spf_clr);
            }
            if (qi[(k) * ncol + i] >= P3_QSMALL && qr[(k) * ncol + i] >= P3_QSMALL
                && t > 273.15f) {
                nrcol = pow(10.0f, f1pr07 + logn0r) * rho * rhofaci
                        * P3_ERI * ni[(k) * ncol + i] * iscf * (spf - spf_clr);
            }

            // self-collection of ice (:2840-2842)
            if (qi[(k) * ncol + i] >= P3_QSMALL) {
                nislf = f1pr03 * rho * eii * eii_fact * rhofaci
                        * ni[(k) * ncol + i] * ni[(k) * ncol + i] * iscf;
            }

            // melting (:2851-2865)
            if (qi[(k) * ncol + i] >= P3_QSMALL && t > 273.15f) {
                float qsat0 = 0.622f * P3_E0 / (pres[(k) * ncol + i] - P3_E0);
                dum = 0.0f;
                qimlt = (p3_div_rn((f1pr05 + f1pr14 * pow(sc, P3_THRD)
                          * pow(rhofaci * rho / mu, 0.5f))
                         * ((t - 273.15f) * kap
                            - rho * xxlv * dv * (qsat0 - qv_cld))
                         * 2.0f * P3_PI, xlf) + dum) * ni[(k) * ncol + i];
                qimlt = fmax(qimlt, 0.0f);
                nimlt = qimlt * (ni[(k) * ncol + i] / qi[(k) * ncol + i]);
            }

            // wet growth (:2873-2894)
            if (qi[(k) * ncol + i] >= P3_QSMALL
                && (qc[(k) * ncol + i] + qr[(k) * ncol + i]) >= 1.0e-6f && t < 273.15f) {
                float qsat0 = 0.622f * P3_E0 / (pres[(k) * ncol + i] - P3_E0);
                qwgrth = ((f1pr05 + f1pr14 * pow(sc, P3_THRD)
                           * pow(rhofaci * rho / mu, 0.5f))
                          * 2.0f * P3_PI
                          * (rho * xxlv * dv * (qsat0 - qv_cld)
                             - (t - 273.15f) * kap)
                          / (xlf + P3_CPW * (t - 273.15f))) * ni[(k) * ncol + i];
                qwgrth = fmax(qwgrth, 0.0f);
                dum = fmax(0.0f, (qccol + qrcol) - qwgrth);
                if (dum >= 1.0e-10f) {
                    nrshdr = nrshdr + dum * 1.923e6f;
                    if ((qccol + qrcol) >= 1.0e-10f) {
                        dum1 = 1.0f / (qccol + qrcol);
                        qcshd = qcshd + dum * qccol * dum1;
                        qccol = qccol - dum * qccol * dum1;
                        qrcol = qrcol - dum * qrcol * dum1;
                    }
                    log_wetgrowth = 1;
                }
            }

            // inverse supersaturation relaxation timescale (:2900-2906)
            if (qi[(k) * ncol + i] >= P3_QSMALL && t < 273.15f) {
                epsi = ((f1pr05 + f1pr14 * pow(sc, P3_THRD)
                         * pow(rhofaci * rho / mu, 0.5f))
                        * 2.0f * P3_PI * rho * dv) * ni[(k) * ncol + i];
                epsi_tot = epsi_tot + epsi;
            } else {
                epsi = 0.0f;
            }

            // rime density, Cober and List 1993 (:2925-2969)
            if (qccol >= P3_QSMALL && t < 273.15f) {
                float vtrmi1 = f1pr02 * rhofaci;
                float iTc = 1.0f / fmin(-0.001f, t - 273.15f);
                if (qc[(k) * ncol + i] >= P3_QSMALL) {
                    float Vt_qc = acn * p3_gam(4.0f + P3_BCN + mu_c)
                                  / (pow(lamc, P3_BCN)
                                     * p3_gam(mu_c + 4.0f));
                    float D_c = (mu_c + 4.0f) / lamc;
                    float V_impact = fabs(vtrmi1 - Vt_qc);
                    float Ri = -(0.5e6f * D_c) * V_impact * iTc;
                    Ri = fmax(1.0f, fmin(Ri, 12.0f));
                    if (Ri <= 8.0f) {
                        rhorime_c = (0.051f + 0.114f * Ri
                                     - 0.0055f * (Ri * Ri)) * 1000.0f;
                    } else {
                        rhorime_c = 611.0f + 72.25f * (Ri - 8.0f);
                    }
                }
            } else {
                rhorime_c = 400.0f;
            }

            // immersion freezing of droplets (:2984-3015)
            //
            // OVERFLOW RESCUE, and the divergence is exactly the overflowing
            // case.  WRF's left-to-right single-precision chain
            // cons6*gamma*tmp1*dum**2 reaches 4.8e45 on the partial product
            // at t = 198.99 K / cdist1 = 3.394e5 / mu_c = 12.576 /
            // lamc = 1.6846e4 -- MEASURED, in the cell that took a real 6 h
            // forecast non-finite at step 284 -- while the mathematical
            // result 4.4e19 is an ordinary float.  The +Inf then meets the
            // conservation limiter at :3571-3583, ratio = sources/sinks is
            // 0, and Inf*0 is the NaN that reached theta and vapour.
            // Fortran does not fix the association of a*b*c*d, so WRF is
            // UNDEFINED here rather than authoritative; P3's own authors
            // left the double form commented out at :2999-3002.  The float
            // chain runs first and unchanged, so every representable value
            // is still WRF's bit for bit.  Mirrors
            // gpuwm/core/p3.py _rescue_overflowed_product exactly.
            if (qc[(k) * ncol + i] >= P3_QSMALL && t <= 269.15f) {
                float d = 1.0f / lamc;
                dum = (d * d) * d;                                   // :2988 int **3
                tmp1 = cdist1 * exp(P3_AIMM * (273.15f - t));
                float gam_q = p3_gam(7.0f + mu_c);
                float gam_n = p3_gam(mu_c + 4.0f);
                float Q_nuc = P3_CONS6 * gam_q * tmp1;
                Q_nuc = Q_nuc * dum;
                Q_nuc = Q_nuc * dum;                                 // :2995 int **2
                float N_nuc = P3_CONS5 * gam_n * tmp1 * dum;
                if (!isfinite(Q_nuc) || !isfinite(N_nuc)) {
                    // The argument is built in FLOAT32 and only the exp is
                    // double -- exactly the authority's own commented-out
                    // form, dexp(dble(aimm*(273.15-t(i,k)))) at :2999, and
                    // exactly what the rain branch below already does.  A
                    // bare `273.15` here would also be an unsuffixed double
                    // literal, which tests/test_p3_cuda.py refuses in this
                    // file on purpose.
                    float dcdist1 = cdist1;
                    float ddum = dum;
                    const ulong exp_aimm = p3_exp64(P3_AIMM * (273.15f - t));
                    if (!isfinite(Q_nuc)) {
                        ulong qwide = soft64_mul(p3_soft64(P3_CONS6), p3_soft64(gam_q));
                        qwide = soft64_mul(qwide, p3_soft64(dcdist1));
                        qwide = soft64_mul(qwide, exp_aimm);
                        qwide = soft64_mul(qwide, p3_soft64(ddum));
                        qwide = soft64_mul(qwide, p3_soft64(ddum));
                        Q_nuc = p3_float32(qwide);
                    }
                    if (!isfinite(N_nuc)) {
                        ulong nwide = soft64_mul(p3_soft64(P3_CONS5), p3_soft64(gam_n));
                        nwide = soft64_mul(nwide, p3_soft64(dcdist1));
                        nwide = soft64_mul(nwide, exp_aimm);
                        nwide = soft64_mul(nwide, p3_soft64(ddum));
                        N_nuc = p3_float32(nwide);
                    }
                }
                qcheti = Q_nuc;
                ncheti = N_nuc;
            }

            // immersion freezing of rain (:3021-3043) -- the authority's
            // own double excursion, kept double.
            if (qr[(k) * ncol + i] * ispf >= P3_QSMALL && t <= 269.15f) {
                ulong tmpdbl1 = p3_exp64(log(cdistr) + log(p3_gam(7.0f + mu_r))
                                         - 6.0f * log(lamr));
                ulong tmpdbl2 = p3_exp64(log(cdistr) + log(p3_gam(mu_r + 4.0f))
                                         - 3.0f * log(lamr));
                ulong tmpdbl3 = p3_exp64(P3_AIMM * (273.15f - t));
                qrheti = P3_CONS6 * p3_float32(soft64_mul(tmpdbl1, tmpdbl3)) * spf;
                nrheti = P3_CONS5 * p3_float32(soft64_mul(tmpdbl2, tmpdbl3)) * spf;
                // Same class as the droplet branch above: WRF's
                // sngl(tmpdbl1*tmpdbl3) (:3037-3038) can round a double to
                // +Inf for the same cold-cloud reason and reach the same
                // limiter.  The double product is in hand, so the rescue is
                // one more multiply.
                if (!isfinite(qrheti)) {
                    ulong qwide = soft64_mul(p3_soft64(P3_CONS6), tmpdbl1);
                    qwide = soft64_mul(qwide, tmpdbl3);
                    qwide = soft64_mul(qwide, p3_soft64(spf));
                    qrheti = p3_float32(qwide);
                }
                if (!isfinite(nrheti)) {
                    ulong nwide = soft64_mul(p3_soft64(P3_CONS5), tmpdbl2);
                    nwide = soft64_mul(nwide, tmpdbl3);
                    nwide = soft64_mul(nwide, p3_soft64(spf));
                    nrheti = p3_float32(nwide);
                }
            }

            // condensation / evaporation / deposition / sublimation
            float epsr;
            if (qr[(k) * ncol + i] * ispf >= P3_QSMALL) {                     // :3114
                int dumii_r, dumjj_r;
                float rdumii, rdumjj;
                p3_find_lt_3(mu_r, lamr, &dumii_r, &dumjj_r,
                             &rdumii, &rdumjj, pc);
                dum = p3_rain_table(revap, dumii_r, dumjj_r, rdumii, rdumjj);
                epsr = 2.0f * P3_PI * cdistr * rho * dv
                       * (P3_F1R * p3_gam(mu_r + 2.0f) / lamr
                          + P3_F2R * pow(rho / mu, 0.5f)
                            * pow(sc, P3_THRD) * dum);
            } else {
                epsr = 0.0f;
            }
            float epsc = (qc[(k) * ncol + i] >= P3_QSMALL)
                         ? 2.0f * P3_PI * rho * dv * cdist : 0.0f;

            float xx;
            if (t < 273.15f) {
                oabi = 1.0f / abi;
                xx = epsc + epsr + epsi_tot
                     * (1.0f + xxls * P3_INV_CP * dqsdT) * oabi;
            } else {
                oabi = 1.0f / abi;
                xx = epsc + epsr;
            }

            float dumqvi = qvi;                                      // :3144

            dum = -P3_CP / P3_G * (t - t_old) * odt;                 // :3171
            if (t < 273.15f) {
                aaa = (qv[(k) * ncol + i] - qv_old[(k) * ncol + i]) * odt
                      - dqsdT * (-dum * P3_G * P3_INV_CP)
                      - (qvs - dumqvi) * (1.0f + xxls * P3_INV_CP * dqsdT)
                        * oabi * epsi_tot;
            } else {
                aaa = (qv[(k) * ncol + i] - qv_old[(k) * ncol + i]) * odt
                      - dqsdT * (-dum * P3_G * P3_INV_CP);
            }

            xx = fmax(1.0e-20f, xx);                                // :3182
            oxx = 1.0f / xx;

            float ssat_cld = ssat[(k) * ncol + i];                            // :3186
            float ssat_r = ssat[(k) * ncol + i];
            float sup_cld = sup, sup_r = sup, supi_cld = supi;

            if (qc[(k) * ncol + i] >= P3_QSMALL) {                            // :3202
                qccon = (aaa * epsc * oxx
                         + (ssat_cld * scf - aaa * oxx) * odt * epsc * oxx
                          * (1.0f - p3_float32(p3_exp64(-(xx * dt))))) / ab;
            }
            if (qr[(k) * ncol + i] >= P3_QSMALL) {                            // :3204
                qrcon = (aaa * epsr * oxx
                         + (ssat_r * spf - aaa * oxx) * odt * epsr * oxx
                          * (1.0f - p3_float32(p3_exp64(-(xx * dt))))) / ab;
            }

            if (sup_cld < -0.001f && qc[(k) * ncol + i] < 1.0e-12f)
                qccon = -qc[(k) * ncol + i] * odt;
            if (sup_r < -0.001f && qr[(k) * ncol + i] < 1.0e-12f)
                qrcon = -qr[(k) * ncol + i] * odt;

            if (qccon < 0.0f) { qcevp = -qccon; qccon = 0.0f; }
            else              { qccon = fmin(qccon, qv[(k) * ncol + i] * odt); }

            if (qrcon < 0.0f) {
                qrevp = -qrcon;
                nrevp = qrevp * (nr[(k) * ncol + i] / qr[(k) * ncol + i]);
                qrcon = 0.0f;
            } else {
                qrcon = fmin(qrcon, qv[(k) * ncol + i] * odt);
            }

            if (qi[(k) * ncol + i] >= P3_QSMALL && t < 273.15f) {             // :3231
                qidep = (aaa * epsi * oxx
                         + (ssat_cld * scf - aaa * oxx) * odt * epsi * oxx
                           * (1.0f - p3_float32(p3_exp64(-(xx * dt))))) * oabi
                        + (qvs - dumqvi) * epsi * oabi;
            }
            if (supi_cld < -0.001f && qi[(k) * ncol + i] < 1.0e-12f)
                qidep = -qi[(k) * ncol + i] * odt;
            if (qidep < 0.0f) {
                qisub = -qidep;
                qisub = qisub * clbfact_sub;
                qisub = fmin(qisub, qi[(k) * ncol + i] * odt);
                nisub = qisub * (ni[(k) * ncol + i] / qi[(k) * ncol + i]);
                qidep = 0.0f;
            } else {
                qidep = qidep * clbfact_dep;
                qidep = fmin(qidep, qv[(k) * ncol + i] * odt);
            }
        }
        // ---- 444 continue (:3257) ----------------------------------

        float sup_cld = sup;
        float supi_cld = supi;

        // deposition/condensation-freezing nucleation, Cooper 1986 (:3264)
        if (t < 258.15f && supi_cld >= 0.05f) {
            dum = 0.005f * exp(0.304f * (273.15f - t)) * 1000.0f * inv_rho;
            dum = fmin(dum, 100.0e3f * inv_rho * scf);
            float N_nuc = fmax(0.0f, (dum - ni[(k) * ncol + i]) * odt);
            if (N_nuc >= 1.0e-20f) {
                float Q_nuc = fmax(0.0f, (dum - ni[(k) * ncol + i]) * P3_MI0 * odt);
                qinuc = Q_nuc;
                ninuc = N_nuc;
            }
        }

        // SEAM 2 -- droplet activation, both log_predictNc branches.
        p3_activate_droplets(log_predictNc, sup_cld, it, t, pres[(k) * ncol + i],
                             qc[(k) * ncol + i], nc[(k) * ncol + i], qv_cld, inv_rho, iscf, scf,
                             odt, &qcnuc, &ncnuc, pc, xxlv);

        // first-step saturation adjustment (:3348-3356)
        if (it <= 1) {
            float dumt = th[(k) * ncol + i] * pow(pres[(k) * ncol + i] * 1.0e-5f,
                                           P3_RD * P3_INV_CP);
            float dumqv = qv_cld;
            float dumqvs = p3_qv_sat(dumt, pres[(k) * ncol + i], 0, pc);
            float dums = dumqv - dumqvs;
            qccon = dums / (1.0f + (xxlv * xxlv) * dumqvs
                            / (P3_CP * P3_RV * (dumt * dumt)))
                    * odt * scf;                         // :3353, integer **2
            qccon = fmax(0.0f, qccon);
            if (qccon <= 1.0e-7f) qccon = 0.0f;
        }

        // autoconversion, iparam = 3, KK2000 (:3396-3402)
        if (qc[(k) * ncol + i] * iscf >= 1.0e-8f) {
            dum = qc[(k) * ncol + i] * iscf;
            qcaut = 1350.0f * pow(dum, 2.47f)
                    * pow(nc[(k) * ncol + i] * iscf * 1.0e-6f * rho, -1.79f) * scf;
            ncautr = qcaut * P3_CONS3;
            ncautc = qcaut * nc[(k) * ncol + i] / qc[(k) * ncol + i];
            if (qcaut == 0.0f) ncautc = 0.0f;
            if (ncautc == 0.0f) qcaut = 0.0f;
        }

        // self-collection of droplets, iparam = 3 (:3421-3435)
        if (qc[(k) * ncol + i] >= P3_QSMALL) ncslf = 0.0f;

        // accretion, iparam = 3 (:3458-3462)
        if (qr[(k) * ncol + i] >= P3_QSMALL && qc[(k) * ncol + i] >= P3_QSMALL) {
            dum2 = spf - spf_clr;
            qcacc = 67.0f * pow(qc[(k) * ncol + i] * iscf * qr[(k) * ncol + i] * ispf, 1.15f)
                    * dum2;
            ncacc = qcacc * nc[(k) * ncol + i] / qc[(k) * ncol + i];
            if (qcacc == 0.0f) ncacc = 0.0f;
            if (ncacc == 0.0f) qcacc = 0.0f;
        }

        // rain self-collection / breakup, iparam = 3 (:3478-3504)
        if (qr[(k) * ncol + i] >= P3_QSMALL) {
            dum1 = 280.0e-6f;
            nr[(k) * ncol + i] = fmax(nr[(k) * ncol + i], P3_NSMALL);
            dum2 = pow(qr[(k) * ncol + i] / (P3_PI * P3_RHOW * nr[(k) * ncol + i]), P3_THRD);
            if (dum2 < dum1) dum = 1.0f;
            else             dum = 2.0f - exp(2300.0f * (dum2 - dum1));
            nrslf = dum * 5.78f * nr[(k) * ncol + i] * ispf * qr[(k) * ncol + i] * ispf
                    * rho * spf;
        }

        // ---- conservation of mass (:3515-3644) ---------------------
        float ratio;
        bool cloud_exhausted = false;
        {
            float dumqvs = p3_qv_sat(t, pres[(k) * ncol + i], 0, pc);
            float qcon_satadj = (qv_cld - dumqvs)
                / (1.0f + (xxlv * xxlv) * dumqvs
                   / (P3_CP * P3_RV * (t * t))) * odt * scf;  // :3516 int **2
            tmp1 = qccon + qrcon + qcnuc;
            if (tmp1 > 0.0f && qcon_satadj < 0.0f) {
                qccon = 0.0f; qrcon = 0.0f; qcnuc = 0.0f; ncnuc = 0.0f;
            } else {
                if (tmp1 > 0.0f && tmp1 > qcon_satadj) {
                    ratio = fmax(0.0f, qcon_satadj) / tmp1;
                    ratio = fmin(1.0f, ratio);
                    qccon = qccon * ratio;
                    qrcon = qrcon * ratio;
                    qcnuc = qcnuc * ratio;
                    ncnuc = ncnuc * ratio;
                } else if (qcevp + qrevp > 0.0f) {
                    ratio = fmax(0.0f, -qcon_satadj) / (qcevp + qrevp);
                    ratio = fmin(1.0f, ratio);
                    qcevp = qcevp * ratio;
                    qrevp = qrevp * ratio;
                    nrevp = nrevp * ratio;
                }
            }

            float qv_tmp = qv_cld + (-qcnuc - qccon - qrcon + qcevp + qrevp)
                           * dt;
            float t_tmp = t + (qcnuc + qccon + qrcon - qcevp - qrevp)
                          * xxlv * P3_INV_CP * dt;
            float dumqvi = p3_qv_sat(t_tmp, pres[(k) * ncol + i], 1, pc);
            float qdep_satadj = (qv_tmp - dumqvi)
                / (1.0f + (xxls * xxls) * dumqvi
                   / (P3_CP * P3_RV * (t_tmp * t_tmp))) * odt * scf;
            tmp1 = qidep + qinuc;
            if (tmp1 > 0.0f && qdep_satadj < 0.0f) {
                qidep = 0.0f; qinuc = 0.0f; ninuc = 0.0f;
            } else {
                if (tmp1 > 0.0f && tmp1 > qdep_satadj) {
                    ratio = fmax(0.0f, qdep_satadj) / tmp1;
                    ratio = fmin(1.0f, ratio);
                    qidep = qidep * ratio;
                    qinuc = qinuc * ratio;
                    ninuc = ninuc * ratio;
                }
                dum = fmax(qisub, 1.0e-20f);
                qisub = qisub * fmin(1.0f, fmax(0.0f, -qdep_satadj)
                                            / fmax(qisub, 1.0e-20f));
                nisub = nisub * fmin(1.0f, qisub / dum);
            }
        }

        float sinks, sources;
        sinks = (qcaut + qcacc + qccol + qcevp + qchetc + qcheti + qcshd) * dt;
        sources = qc[(k) * ncol + i] + (qccon + qcnuc) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3571
            cloud_exhausted = true;
            ratio = sources / sinks;
            qcaut *= ratio; qcacc *= ratio; qcevp *= ratio; qccol *= ratio;
            qcheti *= ratio; qcshd *= ratio; ncautc *= ratio; ncacc *= ratio;
            nccol *= ratio; ncheti *= ratio;
        }

        sinks = (qrevp + qrcol + qrhetc + qrheti + qrmul) * dt;
        sources = qr[(k) * ncol + i] + (qrcon + qcaut + qcacc + qimlt + qcshd) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3593
            ratio = sources / sinks;
            qrevp *= ratio; qrcol *= ratio; qrheti *= ratio; qrmul *= ratio;
            nrevp *= ratio; nrcol *= ratio; nrheti *= ratio;
        }

        sinks = (qisub + qimlt) * dt;
        sources = qi[(k) * ncol + i] + (qidep + qinuc + qrcol + qccol + qrhetc
                               + qrheti + qchetc + qcheti + qrmul) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3609
            ratio = sources / sinks;
            qisub *= ratio; qimlt *= ratio; nisub *= ratio; nimlt *= ratio;
        }

        sinks = (qccon + qrcon + qcnuc + qidep + qinuc) * dt;
        sources = qv[(k) * ncol + i] + (qcevp + qrevp + qisub) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3633
            ratio = sources / sinks;
            qccon *= ratio; qrcon *= ratio; qcnuc *= ratio; qidep *= ratio;
            qinuc *= ratio; ninuc *= ratio; ncnuc *= ratio;
        }

        // ---- update prognostic variables (:3756-3921) ---------------
        float rimevolume = 0.0f, rimefraction = 0.0f;
        if (qi[(k) * ncol + i] >= P3_QSMALL) {                                // :3756
            tmp1 = 1.0f / qi[(k) * ncol + i];
            rimevolume = qib[(k) * ncol + i] * tmp1;
            rimefraction = qir[(k) * ncol + i] * tmp1;
        }

        qc[(k) * ncol + i] = qc[(k) * ncol + i] + (-qchetc - qcheti - qccol - qcshd) * dt;
        if (log_predictNc) {                                         // :3770
            nc[(k) * ncol + i] = nc[(k) * ncol + i] + (-nccol - nchetc - ncheti) * dt;
        }
        qr[(k) * ncol + i] = qr[(k) * ncol + i] + (-qrcol + qimlt - qrhetc - qrheti + qcshd
                                 - qrmul) * dt;
        nr[(k) * ncol + i] = nr[(k) * ncol + i] + (-nrcol - nrhetc - nrheti
                                 + P3_NMLTRATIO * nimlt + nrshdr + ncshdc)
                                * dt;

        qib[(k) * ncol + i] = qib[(k) * ncol + i] - (qisub + qimlt) * dt * rimevolume;
        qir[(k) * ncol + i] = qir[(k) * ncol + i] - (qisub + qimlt) * dt * rimefraction;
        qi[(k) * ncol + i] = qi[(k) * ncol + i] - (qisub + qimlt) * dt;

        dum = (qrcol + qccol + qrhetc + qrheti + qchetc + qcheti + qrmul) * dt;
        qi[(k) * ncol + i] = qi[(k) * ncol + i] + (qidep + qinuc) * dt + dum;
        qir[(k) * ncol + i] = qir[(k) * ncol + i] + dum;
        qib[(k) * ncol + i] = qib[(k) * ncol + i] + (qrcol * P3_INV_RHO_RIMEMAX
                                   + qccol / rhorime_c
                                   + (qrhetc + qrheti + qchetc + qcheti
                                      + qrmul) * P3_INV_RHO_RIMEMAX) * dt;
        ni[(k) * ncol + i] = ni[(k) * ncol + i] + (ninuc - nimlt - nisub - nislf
                                 + nrhetc + nrheti + nchetc + ncheti + nimul)
                                * dt;

        if (qir[(k) * ncol + i] < 0.0f) { qir[(k) * ncol + i] = 0.0f; qib[(k) * ncol + i] = 0.0f; }

        if (log_wetgrowth) {                                         // :3835
            qir[(k) * ncol + i] = qi[(k) * ncol + i];
            qib[(k) * ncol + i] = qir[(k) * ncol + i] * P3_INV_RHO_RIMEMAX;
        }

        if (qi[(k) * ncol + i] >= P3_QSMALL && qib[(k) * ncol + i] >= P3_BSMALL
            && qimlt > 0.0f) {                                       // :3841
            tmp1 = qir[(k) * ncol + i] / qib[(k) * ncol + i];
            tmp2 = qi[(k) * ncol + i] + qimlt * dt;
            qib[(k) * ncol + i] = qir[(k) * ncol + i] / (tmp1 + (917.0f - tmp1) * qimlt
                                       * dt / tmp2);
        }

        qv[(k) * ncol + i] = qv[(k) * ncol + i] + (-qidep + qisub - qinuc) * dt;
        th[(k) * ncol + i] = th[(k) * ncol + i] + invexn * ((qidep - qisub + qinuc)
                                          * xxls * P3_INV_CP
                                          + (qrcol + qccol + qchetc + qcheti
                                             + qrhetc + qrheti + qrmul - qimlt)
                                            * xlf * P3_INV_CP) * dt;

        // warm-phase updates (:3869-3885)
        qc[(k) * ncol + i] = qc[(k) * ncol + i] + (-qcacc - qcaut + qcnuc + qccon - qcevp) * dt;
        qr[(k) * ncol + i] = qr[(k) * ncol + i] + (qcacc + qcaut + qrcon - qrevp) * dt;
        if (log_predictNc) {                                         // :3872
            nc[(k) * ncol + i] = nc[(k) * ncol + i] + (-ncacc - ncautc + ncslf + ncnuc) * dt;
        } else {
            nc[(k) * ncol + i] = P3_NCCNST * inv_rho;                         // :3875
        }
        nr[(k) * ncol + i] = nr[(k) * ncol + i] + (ncautr - nrslf - nrevp) * dt;       // iparam=3
        qv[(k) * ncol + i] = qv[(k) * ncol + i] + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt;
        th[(k) * ncol + i] = th[(k) * ncol + i] + invexn * ((qcnuc + qccon + qrcon - qcevp
                                           - qrevp) * xxlv * P3_INV_CP)
                                * dt;

        // clipping (:3889-3918)
        if (cloud_exhausted || qc[(k) * ncol + i] < P3_QSMALL) {
            qv[(k) * ncol + i] = qv[(k) * ncol + i] + qc[(k) * ncol + i];
            th[(k) * ncol + i] = th[(k) * ncol + i] - invexn * qc[(k) * ncol + i] * xxlv * P3_INV_CP;
            qc[(k) * ncol + i] = 0.0f;
            nc[(k) * ncol + i] = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (qr[(k) * ncol + i] < P3_QSMALL) {
            qv[(k) * ncol + i] = qv[(k) * ncol + i] + qr[(k) * ncol + i];
            th[(k) * ncol + i] = th[(k) * ncol + i] - invexn * qr[(k) * ncol + i] * xxlv * P3_INV_CP;
            qr[(k) * ncol + i] = 0.0f;
            nr[(k) * ncol + i] = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (qi[(k) * ncol + i] < P3_QSMALL) {
            qv[(k) * ncol + i] = qv[(k) * ncol + i] + qi[(k) * ncol + i];
            th[(k) * ncol + i] = th[(k) * ncol + i] - invexn * qi[(k) * ncol + i] * xxls * P3_INV_CP;
            qi[(k) * ncol + i] = 0.0f;
            ni[(k) * ncol + i] = 0.0f;
            qir[(k) * ncol + i] = 0.0f;
            qib[(k) * ncol + i] = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        qv[(k) * ncol + i] = fmax(0.0f, qv[(k) * ncol + i]);
        ni[(k) * ncol + i] = p3_impose_max_ni(ni[(k) * ncol + i], inv_rho, pc);
        // 555 continue
    }

    // second compute_SCPF (:3968-3970): refresh the qv snapshot.
    for (int k = k_start; k < nk; k += k_stride) qv_cld_buf[(k) * ncol + i] = qv[k * ncol + i];
    return hydrometeors_present;
}
kernel void p3_kloopmain(constant P3KloopmainResources &r [[buffer(0)]],
                         constant P3KloopmainParams &p [[buffer(1)]],
                         uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx); if(column>=ncol)return;
    const int present=p3_step_kloopmain_levels(r,p,column);
    r.flags[ncol+column]=present?1.0f:0.0f;
}
