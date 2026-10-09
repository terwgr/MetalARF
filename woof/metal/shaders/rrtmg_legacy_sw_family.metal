#include <metal_stdlib>
#include "rrtmg_legacy_sw_inatm_bindings.metalh"
#include "rrtmg_legacy_sw_cldprmc_bindings.metalh"
#include "rrtmg_legacy_sw_setcoef_bindings.metalh"
#include "rrtmg_legacy_sw_laysolfr_bindings.metalh"
#include "rrtmg_legacy_sw_taumol_bindings.metalh"
#include "rrtmg_legacy_sw_sfluxzen_bindings.metalh"
#include "rrtmg_legacy_sw_spcvmc_bindings.metalh"
#include "rrtmg_legacy_sw_accum_bindings.metalh"
#include "rrtmg_legacy_sw_post_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_exp.metalh"
#include "rrtmg_legacy_sw_tables.metalh"

using namespace metal;

// Legacy RRTMG SW (WRF ra_sw_physics = 4): Metal twin of the batched _b
// kernels of woof/core/kernels/rrtmg_sw.cu.  The CUDA unit rounds every
// FP32 add/sub/mul as a single IEEE .rn instruction with gradual underflow
// (the direct-beam transmittance chain really produces subnormals), and
// its glibc logf/expf run their binary64 steps as separate __d*_rn ops.
// Here AD/SU/MU/DV/SQ give soft32's values (RN-even, subnormals kept,
// independent of the GPU's FP32 denormal mode; see rsw_add) and soft64
// for the libm's double steps (mul then add, no FMA, as in the CUDA rsw_
// transcription -- NOT the fused gfk_ variants).

typedef float real;

#define RSW_F(bits) as_type<float>(bits)
#define RSW_U(x) as_type<uint>(x)

// The GPU's FP32 add/sub/mul/div/sqrt (safe math mode, precise fp32
// functions, no contraction) are RN-even like soft32 but flush subnormal
// operands and results to zero.  Each operation therefore takes the
// hardware result when both operands and the result are finite with
// magnitude >= 2^-125, and the soft32 result otherwise (zeros, subnormals,
// the boundary band, Inf and NaN).  Every value is the soft32 value.
static inline bool rsw_hw_ok(float v)
{ const float m = fabs(v); return m >= 0x1p-125f && m <= 0x1.fffffep127f; }
static inline float rsw_add(float a, float b)
{ const float r = a + b;
  return (rsw_hw_ok(a) && rsw_hw_ok(b) && rsw_hw_ok(r))
      ? r : RSW_F(soft32_add(RSW_U(a), RSW_U(b))); }
static inline float rsw_sub(float a, float b)
{ const float r = a - b;
  return (rsw_hw_ok(a) && rsw_hw_ok(b) && rsw_hw_ok(r))
      ? r : RSW_F(soft32_sub(RSW_U(a), RSW_U(b))); }
static inline float rsw_mul(float a, float b)
{ const float r = a * b;
  return (rsw_hw_ok(a) && rsw_hw_ok(b) && rsw_hw_ok(r))
      ? r : RSW_F(soft32_mul(RSW_U(a), RSW_U(b))); }
static inline float rsw_div(float a, float b)
{ const float r = a / b;
  return (rsw_hw_ok(a) && rsw_hw_ok(b) && rsw_hw_ok(r))
      ? r : RSW_F(soft32_div(RSW_U(a), RSW_U(b))); }
static inline float rsw_sqrt(float a)
{ const float r = sqrt(a);
  return (a > 0.0f && rsw_hw_ok(a) && rsw_hw_ok(r))
      ? r : RSW_F(soft32_sqrt(RSW_U(a))); }
#define AD(a, b) rsw_add((float)(a), (float)(b))
#define SU(a, b) rsw_sub((float)(a), (float)(b))
#define MU(a, b) rsw_mul((float)(a), (float)(b))
#define DV(a, b) rsw_div((float)(a), (float)(b))
#define SQ(a) rsw_sqrt((float)(a))
#define EQ0(x) ((RSW_U(x) & 0x7fffffffu) == 0u)

#define RSW_NGPT 112
#define RSW_NBND 14
#define RSW_MXMOL 38
#define RSW_NMOL 7

// glibc 2.39 logf as rsw_log: binary64 steps as separate RN mul/add.
static inline real rsw_log(real x)
{
    uint ix = RSW_U(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return RSW_F(0xff800000u);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) != 0u || ix * 2u >= 0xff000000u)
            return RSW_F(0x7fc00000u);
        ix = RSW_U(MU(x, 8388608.0f));
        ix -= 23u << 23;
    }
    const uint tmp = ix - 0x3f330000u;
    const uint i = (tmp >> 19) & 15u;
    const int k = int(tmp) >> 23;
    const uint iz = ix - (tmp & 0xff800000u);
    const ulong z = soft64_from_f32_bits(iz);
    const ulong r = soft64_sub(soft64_mul(z, SOFT_GFK_LOGF_INVC[i]),
                               0x3ff0000000000000ul);
    const ulong y0 = soft64_add(SOFT_GFK_LOGF_LOGC[i],
        soft64_mul(soft64_from_i32(k), SOFT_GFK_LOGF_LN2));
    const ulong r2 = soft64_mul(r, r);
    ulong y = soft64_add(soft64_mul(SOFT_GFK_LOGF_A1, r), SOFT_GFK_LOGF_A2);
    y = soft64_add(soft64_mul(SOFT_GFK_LOGF_A0, r2), y);
    y = soft64_add(soft64_mul(y, r2), soft64_add(y0, r));
    return RSW_F(soft64_to_f32_bits(y));
}

static inline ulong rsw_exp2_core(ulong xd)
{
    const ulong shift = 0x4338000000000000ul;          // 0x1.8p+52
    ulong kd = soft64_add(xd, shift);
    const ulong ki = kd;
    kd = soft64_sub(kd, shift);
    const ulong r = soft64_sub(xd, kd);
    ulong t = SOFT_GFK_EXP2F_TAB[uint(ki & 31ul)];
    t += ki << 47u;
    const ulong s = t;
    const ulong z = soft64_add(soft64_mul(0x3ebc6af84b912394ul, r),
                               0x3f2ebfce50fac4f3ul);
    const ulong r2 = soft64_mul(r, r);
    ulong y = soft64_add(soft64_mul(0x3f962e42ff0c52d6ul, r),
                         0x3ff0000000000000ul);
    y = soft64_add(soft64_mul(z, r2), y);
    return soft64_mul(y, s);
}

static inline real rsw_exp(real x)
{
    const uint abstop = (RSW_U(x) >> 20) & 0x7ffu;
    if (abstop >= (RSW_U(88.0f) >> 20)) {
        if (RSW_U(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return AD(x, x);
        if (x > RSW_F(0x42b17218u)) return RSW_F(0x7f800000u);
        if (x < -RSW_F(0x42cff1b4u)) return 0.0f;
    }
    const ulong xd = soft64_from_f32_bits(RSW_U(x));
    const ulong z = soft64_mul(0x40471547652b82feul, xd);
    return RSW_F(soft64_to_f32_bits(rsw_exp2_core(z)));
}

#include "rrtmg_legacy_sw_bodies.metalh"

kernel void rrtmg_legacy_sw_inatm(
    constant RrtmgLegacySwInatmResources &r [[buffer(0)]],
    constant RrtmgLegacySwInatmParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long idx = long(tid);
    if (idx >= long(p.ncol) * p.nl) return;
    const int col = int(idx / p.nl);
    const int L = int(idx % p.nl);
    const ulong o = ulong(col) * ulong(p.nl);
    rsw_inatm_layers_body(L, p.nl, p.grav, p.avogad,
        r.plev + ulong(col) * ulong(p.nl + 1),
        r.h2ovmr + o, r.co2vmr + o, r.o3vmr + o, r.n2ovmr + o,
        r.ch4vmr + o, r.o2vmr + o, r.pdp + o, r.coldry + o,
        r.wkl + ulong(col) * RSW_MXMOL * ulong(p.nl));
}

kernel void rrtmg_legacy_sw_cldprmc(
    constant RrtmgLegacySwCldprmcResources &r [[buffer(0)]],
    constant RrtmgLegacySwCldprmcParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long W = long(p.nl) * RSW_NGPT;
    const long idx = long(tid);
    if (idx >= long(p.ncol) * W) return;
    const int col = int(idx / W);
    const int rr = int(idx % W);
    const int ig = rr % RSW_NGPT;
    const int lay = rr / RSW_NGPT;
    const ulong og = ulong(col) * ulong(W);
    const ulong o = ulong(col) * ulong(p.nl);
    rsw_cldprmc_body(ig, lay, p.nl, p.inflag, p.iceflag, p.liqflag,
        r.tab, r.ngb, r.cldfmc + og, r.ciwpmc + og, r.clwpmc + og,
        r.cswpmc + og, r.reicmc + o, r.relqmc + o, r.resnmc + o,
        r.taormc + og, r.taucmc + og, r.ssacmc + og, r.asmcmc + og,
        r.fsfcmc + og, r.err + og);
}

kernel void rrtmg_legacy_sw_setcoef(
    constant RrtmgLegacySwSetcoefResources &r [[buffer(0)]],
    constant RrtmgLegacySwSetcoefParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long idx = long(tid);
    if (idx >= long(p.ncol) * p.nl) return;
    const int col = int(idx / p.nl);
    const int lay = int(idx % p.nl);
    const ulong o = ulong(col) * ulong(p.nl);
    const ulong om = ulong(col) * RSW_MXMOL * ulong(p.nl);
    rsw_setcoef_body(lay, p.nl, r.tab, r.pavel + o, r.tavel + o,
        r.coldry + o, r.wkl + om, r.jp + o, r.jt + o, r.jt1 + o,
        r.indself + o, r.indfor + o, r.tflag + o, r.lflag + o,
        r.colh2o + o, r.colco2 + o, r.colo3 + o, r.coln2o + o,
        r.colch4 + o, r.colo2 + o, r.colmol + o, r.co2mult + o,
        r.selffac + o, r.selffrac + o, r.forfac + o, r.forfrac + o,
        r.fac00 + o, r.fac01 + o, r.fac10 + o, r.fac11 + o);
}

// laytrop = integer sum of the tropopause flags, then the rsw_laysolfr_b
// integer scan (last matching layer wins, including the upper wrap).
kernel void rrtmg_legacy_sw_laysolfr(
    constant RrtmgLegacySwLaysolfrResources &r [[buffer(0)]],
    constant RrtmgLegacySwLaysolfrParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (int(tid) >= p.ncol) return;
    const int col = int(tid);
    const int nlayers = p.nl;
    device const int *row = r.jp + ulong(col) * ulong(nlayers);
    device const int *flag = r.tflag + ulong(col) * ulong(nlayers);
    int trop = 0;
    for (int lay = 0; lay < nlayers; ++lay) trop += flag[lay];
    r.laytrop[col] = trop;
    for (int band = 0; band < RSW_NBND; ++band) {
        const int lr = r.spec[band];
        const int upper = r.spec[RSW_NBND + band];
        const int has_lr = r.spec[2 * RSW_NBND + band];
        int value = upper ? nlayers : trop;
        if (!has_lr) value = trop;
        else if (upper) {
            for (int lay = trop + 1; lay <= nlayers; ++lay) {
                const int previous = (lay - 2 + nlayers) % nlayers;
                if (row[previous] < lr && row[lay - 1] >= lr) value = lay;
            }
        } else {
            for (int lay = 1; lay <= trop; ++lay) {
                const int next = lay < nlayers ? lay : nlayers - 1;
                if (row[lay - 1] < lr && row[next] >= lr)
                    value = lay + 1 < trop ? lay + 1 : trop;
            }
        }
        r.laysolfr[ulong(col) * RSW_NBND + ulong(band)] = value;
    }
}

kernel void rrtmg_legacy_sw_taumol(
    constant RrtmgLegacySwTaumolResources &r [[buffer(0)]],
    constant RrtmgLegacySwTaumolParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long W = long(p.nl) * RSW_NGPT;
    const long idx = long(tid);
    if (idx >= long(p.ncol) * W) return;
    const int col = int(idx / W);
    const int rr = int(idx % W);
    const int L = rr / RSW_NGPT;
    const int iw = rr % RSW_NGPT;
    const ulong o = ulong(col) * ulong(p.nl);
    const ulong og = ulong(col) * ulong(W);
    rsw_taumol_body(L, iw, p.nl, r.laytrop[col], r.tab, r.ngb,
        r.jp + o, r.jt + o, r.jt1 + o,
        r.colh2o + o, r.colco2 + o, r.colch4 + o, r.colo2 + o,
        r.colo3 + o, r.colmol + o,
        r.fac00 + o, r.fac01 + o, r.fac10 + o, r.fac11 + o,
        r.selffac + o, r.selffrac + o, r.indself + o,
        r.forfac + o, r.forfrac + o, r.indfor + o,
        r.taug + og, r.taur + og, true);
}

kernel void rrtmg_legacy_sw_sfluxzen(
    constant RrtmgLegacySwSfluxzenResources &r [[buffer(0)]],
    constant RrtmgLegacySwSfluxzenParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long idx = long(tid);
    if (idx >= long(p.ncol) * RSW_NGPT) return;
    const int col = int(idx / RSW_NGPT);
    const int iw = int(idx % RSW_NGPT);
    const ulong o = ulong(col) * ulong(p.nl);
    rsw_sfluxzen_body(iw, p.nl, r.laytrop[col], r.tab, r.ngb,
        r.laysolfr + ulong(col) * RSW_NBND,
        r.jp + o, r.jt + o, r.jt1 + o,
        r.colh2o + o, r.colco2 + o, r.colch4 + o, r.colo2 + o,
        r.colo3 + o, r.fac00 + o, r.fac01 + o, r.fac10 + o,
        r.fac11 + o, r.sfluxzen + ulong(col) * RSW_NGPT);
}

kernel void rrtmg_legacy_sw_spcvmc(
    constant RrtmgLegacySwSpcvmcResources &r [[buffer(0)]],
    constant RrtmgLegacySwSpcvmcParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long idx = long(tid);
    if (idx >= long(p.ncol) * RSW_NGPT) return;
    const int col = int(idx / RSW_NGPT);
    const int iw1 = int(idx % RSW_NGPT);
    const ulong n1 = ulong(p.nl) + 1ul;
    const ulong ob = ulong(col) * RSW_NBND;
    const ulong og = ulong(col) * RSW_NGPT;
    const ulong ogl = ulong(col) * ulong(p.nl) * RSW_NGPT;
    const ulong obl = ulong(col) * ulong(p.nl) * RSW_NBND;
    const ulong og1 = ulong(col) * n1 * RSW_NGPT;
    rsw_spcvmc_body(iw1, p.nl, r.tab, r.ngb, r.palbd + ob, r.palbp + ob,
        r.pcldfmc + ogl, r.ptaucmc + ogl, r.pasycmc + ogl,
        r.pomgcmc + ogl, r.ptaormc + ogl,
        r.ptaua + obl, r.pasya + obl, r.pomga + obl,
        r.prmu0[col], r.adjflux + ob, r.sfluxzen + og,
        r.taug + ogl, r.taur + ogl, r.zincflx + og,
        r.zcd + og1, r.zcu + og1, r.zfd + og1, r.zfu + og1,
        r.ztdbt_nodel + og1, r.ztdbtc_nodel + og1,
        r.wk + ulong(idx), ulong(p.ncol) * RSW_NGPT, true);
}

// rsw_spc_accum_body over the interleaved (n1, 112) slabs: one thread per
// (column, output level), g-points summed sequentially in iw order.
kernel void rrtmg_legacy_sw_accum(
    constant RrtmgLegacySwAccumResources &r [[buffer(0)]],
    constant RrtmgLegacySwAccumParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n1 = p.nl + 1;
    const long idx = long(tid);
    if (idx >= long(p.ncol) * n1) return;
    const int col = int(idx / n1);
    const int K = int(idx % n1);
    const int J = p.nl - K;
    const ulong og = ulong(col) * RSW_NGPT;
    const ulong og1 = ulong(col) * ulong(n1) * RSW_NGPT + ulong(J) * RSW_NGPT;
    real bbfd = 0.0f, bbfu = 0.0f, bbcd = 0.0f, bbcu = 0.0f;
    real bbfddir = 0.0f, bbcddir = 0.0f;
    real uvfd = 0.0f, uvcd = 0.0f, uvfddir = 0.0f, uvcddir = 0.0f;
    real nifd = 0.0f, nicd = 0.0f, nifddir = 0.0f, nicddir = 0.0f;
    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        const ulong q = og1 + ulong(iw);
        const int ibm = r.ngb[iw] - 15;
        const real w = r.zincflx[og + ulong(iw)];
        bbfu = AD(bbfu, MU(w, r.zfu[q]));
        bbfd = AD(bbfd, MU(w, r.zfd[q]));
        bbcu = AD(bbcu, MU(w, r.zcu[q]));
        bbcd = AD(bbcd, MU(w, r.zcd[q]));
        bbfddir = AD(bbfddir, MU(w, r.ztdbt_nodel[q]));
        bbcddir = AD(bbcddir, MU(w, r.ztdbtc_nodel[q]));
        if (ibm >= 10 && ibm <= 13) {
            uvcd = AD(uvcd, MU(w, r.zcd[q]));
            uvfd = AD(uvfd, MU(w, r.zfd[q]));
            uvcddir = AD(uvcddir, MU(w, r.ztdbtc_nodel[q]));
            uvfddir = AD(uvfddir, MU(w, r.ztdbt_nodel[q]));
        } else if (ibm == 14 || ibm <= 9) {
            nicd = AD(nicd, MU(w, r.zcd[q]));
            nifd = AD(nifd, MU(w, r.zfd[q]));
            nicddir = AD(nicddir, MU(w, r.ztdbtc_nodel[q]));
            nifddir = AD(nifddir, MU(w, r.ztdbt_nodel[q]));
        }
    }
    const ulong o = ulong(idx);
    r.pbbfd[o] = bbfd;  r.pbbfu[o] = bbfu;
    r.pbbcd[o] = bbcd;  r.pbbcu[o] = bbcu;
    r.pbbfddir[o] = bbfddir;  r.pbbcddir[o] = bbcddir;
    r.puvfd[o] = uvfd;  r.puvcd[o] = uvcd;
    r.puvfddir[o] = uvfddir;  r.puvcddir[o] = uvcddir;
    r.pnifd[o] = nifd;  r.pnicd[o] = nicd;
    r.pnifddir[o] = nifddir;  r.pnicddir[o] = nicddir;
}

kernel void rrtmg_legacy_sw_post(
    constant RrtmgLegacySwPostResources &r [[buffer(0)]],
    constant RrtmgLegacySwPostParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long idx = long(tid);
    if (idx >= long(p.ncol) * p.nl) return;
    const int col = int(idx / p.nl);
    const int i = int(idx % p.nl);
    const ulong o = ulong(col) * ulong(p.nl);
    const ulong o1 = ulong(col) * ulong(p.nl + 1);
    rsw_post_body(i, p.nl, p.heatfac, r.pdp + o, r.pbbfu + o1,
        r.pbbfd + o1, r.pbbcu + o1, r.pbbcd + o1, r.swhr + o, r.swhrc + o);
}
