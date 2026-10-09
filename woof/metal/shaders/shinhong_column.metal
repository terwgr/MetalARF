#include <metal_stdlib>
#include "../generated/shinhong_column_bindings.metalh"
#include "../generated/shinhong_validate_outputs_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)
static inline float sh_div(float a, float b) { return a / b; }

constant float G=9.81f, RD=287.0f, RV=461.6f, CP=1004.5f, XLV=2.5e6f;
#define SHINHONG_KMAX 128
#define SHINHONG_K2 (SHINHONG_KMAX + 2)
#define SHWS_SLOTS 50
#define SHWS_LANES 32
#define SHWS_AT(base,idx,kp) (ShCol{(base) + ulong(idx) * ulong(kp) * ulong(SHWS_LANES)})
#define SHWS_LANE_BASE(ws,kp) ((ws) + ulong(group_index) * ulong(SHWS_SLOTS) * ulong(kp) * ulong(SHWS_LANES) + ulong(lane_index))
struct ShCol { device float *p; device float &operator[](int k) const { return p[ulong(k)*ulong(SHWS_LANES)]; } };
struct ShColC { device const float *p; ShColC(thread const ShCol &a) : p(a.p) {} float operator[](int k) const { return p[ulong(k)*ulong(SHWS_LANES)]; } };
// Shin-Hong scale-aware PBL column scheme (bl_pbl_physics=11), WRF v4.6.1.
//
// Transcribed statement for statement from the float32 CPU authority
// gpuwm/verify/shinhong_ref.py::np_shinhong_column, which is itself bitwise
// (max_ulp 0, both arms) against the byte-frozen WRF v4.6.1
// phys/module_bl_shinhong.F (sha256 99f44dbe..., pinned in
// gpuwm/data/shinhong/oracle/oracle-sha256sums.txt).  The :NNNN line numbers
// below are that Fortran file's; they are carried over from the authority so
// every statement here can be walked back to the module it mirrors.  One CUDA
// thread owns one complete surface-to-top column; element k (0-based) of
// column col is [k*st + col] with st = ny*nx; the working arrays use a tiled global workspace,
// indexed 1-based like the Fortran, and bounded by SHINHONG_KMAX.
//
// ARM A ONLY.  The oracle records two arms; this kernel implements arm A
// (ctopo = ctopo2 = 1), the only arm WRF's Registry default (topo_wind=0) and
// this engine can reach.  With ctopo == 1 the momentum surface-drag diagonal
// (:1387) multiplies by exactly 1.0f, and with ctopo2 == 1 the u10/v10 blend
// (:1472-1473) returns its input word for word -- both are bitwise identities,
// proved on the CPU gate by
// tests/test_shinhong_wrf461_parity.py::test_the_topo_wind_arm_is_real_and_this_far_from_arm_a
// (which also pins how far arm B actually sits from arm A: ~1.9e9 ULP on the
// momentum tendencies, because they cross zero between the arms).  The ctopo
// multiply and the u10/v10 blend are therefore omitted here, and the kernel
// has no u10/v10 outputs at all.
//
// Two deliberate divergences carried over from the authority, both
// defined-behaviour guards against reads WRF itself never defines:
//   * efxpbl (:1557) reads q2xk(kpbl+1), one past the end when kpbl == kte;
//     guarded here (kpbl < kte), same as the authority.
//   * zfacent is only assigned below kpbl (:1006) but the TKE staging loop
//     (:1550-1552) reads it at every level; WRF reads stack garbage there and
//     never uses the result.  This port keeps those lanes at 0.0f.
//
// libm and pow spellings (pow-probe.txt, measured on the oracle toolchain):
//   * x**2. and x**pfac(2.0) fold to x*x -- spelled as multiplies;
//   * every other real-exponent power stays powf.  Device powf is NOT glibc's
//     correctly rounded powf, so those sites carry ULPs into the pinned
//     tables in tests/test_shinhong_wrf461_parity.py rather than being hidden
//     by a tolerance.  Same for expf and tanhf, and for the FMA contraction
//     nvrtc's default --fmad=true performs (measured in isolation on the
//     partition probe: pq has no transcendental at all and still sits 2 ULP
//     from glibc through pure contraction).
//   * the four x**(1/3) sites were measured BOTH ways against the oracle
//     (pow(x, h1) vs cbrtf(x), the -Ofast spelling; ysu.cu precedent), one
//     site at a time on the full fixture:
//       wstar  :750  cbrtf moved wstar differing lanes 24 -> 60   -> powf
//       wscale :757  cbrtf within noise (du 414 -> 416 lanes)     -> powf
//       wscalek:945  cbrtf closed dv's worst column 1491308 -> 46603 ULP and
//                    exch_h 9 -> 8, at the cost of tke 2005 -> 2022 -> cbrtf
//       ckp   :1980  cbrtf same el max (14), more lanes 1840 -> 1903 -> powf
//     Each site keeps whichever spelling measured closer; the one cbrtf site
//     is marked at its line below.
//     THE wscalek ROW IS COMPILER-SPECIFIC AND THE OTHERS ARE NOT.  Those
//     numbers were measured under NVRTC 13.0.48 (the system CUDA v13.0
//     toolkit, which CuPy resolved on the certification box until
//     2026-08-04).  Under 12.8.61/12.8.93/12.9.41/12.9.86 the same cbrtf
//     spelling still buys exch_h 9 -> 8 and buys NOTHING on dv or tke: the
//     device cube root changed between the CUDA 12 and CUDA 13 lines, and
//     the dv/tke halves of that measurement were properties of the 13.0 code
//     generator rather than of this source.  powf reads 1491308 / 9 / 2005
//     under every build measured, so the toolchain did not move generally --
//     only the one intrinsic did.  The spelling STAYS cbrtf: it still wins
//     exch_h on both lines, and it is 1 ULP from a correctly rounded cube
//     root on this device where pow(x, h1) is 3 (device probe 2026-08-06,
//     300k samples).  tests/test_shinhong_wrf461_parity.py carries one ULP
//     row per compiler and refuses a build it has never measured.
//   * pow(x, 0.0f) == 1.0f for every x on this device, including +/-0 and
//     subnormal x (device probe, 2026-08-03); no special case needed, and
//     the zfac**(pfac_q-pfac) multiply at :968 is kept (zfac is clamped to
//     [zfmin, 1] anyway, so the base is never 0 there).
//   * sqrtf is correctly rounded on both sides and carries nothing.
//
// FTZ/DAZ: CuPy appends -ftz=true unconditionally, and sm_120 additionally
// flushes FP32 subnormals in ALL arithmetic (DAZ) with --ftz=false
// ineffective.  The fixture's designed probes and what they measured:
//   * br = +1.4e-45 (case 7) on the `br .gt. 0` sfcflg compare (:706): the
//     one branch DAZ actually flips.  Countermeasure and measurements at the
//     compare itself, below (sh_positive_float).
//   * subnormal ust (case 13): ust**3. underflows to +0 in float32 on BOTH
//     sides, so WRF's own prfac2 0/0 (:1010) fires here too and the NaN heat
//     column reproduces -- with a different NaN payload than the CSV can
//     carry, so the gate pins NaN GEOMETRY, never NaN bits.
//   * subnormal qfx (case 13), subnormal/signed-zero moisture (case 24) and
//     subnormal u10 (case 11): no branch moves; the columns sit inside the
//     ordinary arithmetic table (measured per-case worst: case 11 at 2 ULP,
//     case 13 at 1 ULP outside its NaN lanes, case 24 in the dqv family).
//     What -ftz DOES reach is the OUTPUT side: every lane where WRF wrote a
//     subnormal tendency, the kernel writes exactly zero (dqv 30 / dqc 24 /
//     dqi 18 lanes, pinned both ways by the gate's subnormal test).
//
// min/max are NumPy-semantics (NaN propagates), NOT fmaxf/fminf: CUDA's
// fmaxf(NaN, x) returns x, which would erase the case-13 NaN geometry at
// the amax1/amin1 sites (:2118 q2 floor, :1032-1034 xkz clamps).  gfortran
// -O0 max/min propagate the NaN on the lanes the fixture reaches, and the
// authority transcribes them as np.maximum/np.minimum for exactly that
// reason.

#define SHINHONG_KMAX 128
#define SHINHONG_K2 (SHINHONG_KMAX + 2)

// Interleaving levels across a warp keeps workspace accesses coalesced.
// 1-based access to a (nz, ny, nx) global field: SH_G(a, 1) is the surface.
#define SH_G(a, k) a[(ulong)((k) - 1) * st + col]
// The scheme's own per-level names, as the authority spells them.
#define SH_UX(k)    SH_G(u, k)
#define SH_VX(k)    SH_G(v, k)
#define SH_THX(k)   SH_G(theta, k)
// Temperature: the engine hands the kernel theta and exner, so tx is formed
// as theta*exner where the scheme reads temperature (same convention as
// ysu.cu; the parity loader reconstructs theta = t/pi2d so thx is bitwise
// WRF's, and the (t/pi)*pi round-trip cost at the tx sites lands in the
// pinned tables).
#define SH_TX(k)    (SH_G(theta, k) * SH_G(exner, k))
#define SH_QVX(k)   SH_G(qv, k)
#define SH_QCX(k)   SH_G(qc, k)
#define SH_QIX(k)   SH_G(qi, k)
#define SH_P2DX(k)  SH_G(pfield, k)
#define SH_PI2DX(k) SH_G(exner, k)
#define SH_DZ8W(k)  SH_G(dz, k)
#define SH_TKEX(k)  SH_G(tke, k)

// NumPy-semantics min/max: NaN in, NaN out (see header).  For non-NaN input
// these are the plain ternaries.
static inline float sh_max(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a > b ? a : b;
}

// Validate the native output bundle using one shared status word. The
// floating-output bit order matches _SHINHONG_OUTPUTS in core/shinhong.py.
kernel void shinhong_validate_outputs(
    constant ShinhongValidateOutputsResources &r [[buffer(0)]],
    constant ShinhongValidateOutputsParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    device atomic_uint *status = reinterpret_cast<device atomic_uint *>(r.status);
    uint flags = 0u;
    if (index < uint(p.count3)) {
        if (!isfinite(r.du[index])) flags |= 1u << 0;
        if (!isfinite(r.dv[index])) flags |= 1u << 1;
        if (!isfinite(r.dtheta[index])) flags |= 1u << 2;
        if (!isfinite(r.dqv[index])) flags |= 1u << 3;
        if (!isfinite(r.dqc[index])) flags |= 1u << 4;
        if (!isfinite(r.dqi[index])) flags |= 1u << 5;
        if (!isfinite(r.exch_h[index])) flags |= 1u << 6;
        if (!isfinite(r.tke[index])) flags |= 1u << 7;
        if (!isfinite(r.el[index])) flags |= 1u << 8;
    }
    if (index < uint(p.count2)) {
        if (!isfinite(r.hpbl[index])) flags |= 1u << 9;
        if (!isfinite(r.wstar[index])) flags |= 1u << 11;
        if (!isfinite(r.delta[index])) flags |= 1u << 12;
    }
    if (flags != 0u)
        atomic_fetch_or_explicit(status, flags, memory_order_relaxed);
}
static inline float sh_min(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a < b ? a : b;
}

// Positive IEEE-754 float test that preserves subnormal inputs without
// requiring device double support or arithmetic on the subnormal value.
static inline bool sh_positive_float(float x) {
    const uint bits = as_type<uint>(x);
    return bits != 0u && bits <= 0x7f800000u;
}

// ------------------------------------------------------------------------
// The five partition functions (:2297-2427).  (doh)**b1 with b1 = 2.0 folds
// to a multiply (pow-probe.txt); the fractional b2 powers stay powf.  The
// h != 0 early-out is exact -- both signed zeros of h return 1.0 -- and IS
// reachable from the scheme (cslen of a non-convective column).
// ------------------------------------------------------------------------

static inline float sh_pu(float d, float h) {
    // pu (:2297-2319): nonlocal/local momentum-transport partition.
    const float a1 = 1.0f, a2 = 0.070f, a3 = 1.0f, a4 = 0.142f, a5 = 0.071f;
    const float b2 = 0.6666667f;
    float pu;
    if (h != 0.0f) {
        float doh = d / h;
        float t2 = doh * doh;
        float tb = pow(doh, b2);
        float num = a1 * t2 + a2 * tb;
        float den = a3 * t2 + a4 * tb + a5;
        pu = num / den;
    } else {
        pu = 1.0f;
    }
    pu = sh_max(pu, 0.0f);
    pu = sh_min(pu, 1.0f);
    return pu;
}

static inline float sh_pq(float d, float h) {
    // pq (:2323-2345): moisture-transport partition (b1 = 2.0 only).
    const float a1 = 1.0f, a2 = -0.098f, a3 = 1.0f, a4 = 0.106f, a5 = 0.5f;
    float pq;
    if (h != 0.0f) {
        float doh = d / h;
        float t2 = doh * doh;
        float num = a1 * t2 + a2;
        float den = a3 * t2 + a4;
        pq = a5 * num / den + (1.0f - a5);
    } else {
        pq = 1.0f;
    }
    pq = sh_max(pq, 0.0f);
    pq = sh_min(pq, 1.0f);
    return pq;
}

static inline float sh_pthnl(float d, float h) {
    // pthnl (:2349-2372): nonlocal heat-transport partition (b2 = 0.875).
    const float a1 = 1.000f, a2 = 0.936f, a3 = -1.110f;
    const float a4 = 1.000f, a5 = 0.312f, a6 = 0.329f, a7 = 0.243f;
    const float b2 = 0.875f;
    float pthnl;
    if (h != 0.0f) {
        float doh = d / h;
        float t2 = doh * doh;
        float tb = pow(doh, b2);
        float num = a1 * t2 + a2 * tb + a3;
        float den = a4 * t2 + a5 * tb + a6;
        pthnl = a7 * num / den + (1.0f - a7);
    } else {
        pthnl = 1.0f;
    }
    pthnl = sh_max(pthnl, 0.0f);
    pthnl = sh_min(pthnl, 1.0f);
    return pthnl;
}

static inline float sh_pthl(float d, float h) {
    // pthl (:2376-2399): local heat-transport partition (b2 = 0.5).
    // (doh)**b2 stays a powf here too: the authority spells what WRF calls,
    // and this port spells what the authority spells.
    const float a1 = 1.000f, a2 = 0.870f, a3 = -0.913f;
    const float a4 = 1.000f, a5 = 0.153f, a6 = 0.278f, a7 = 0.280f;
    const float b2 = 0.5f;
    float pthl;
    if (h != 0.0f) {
        float doh = d / h;
        float t2 = doh * doh;
        float tb = pow(doh, b2);
        float num = a1 * t2 + a2 * tb + a3;
        float den = a4 * t2 + a5 * tb + a6;
        pthl = a7 * num / den + (1.0f - a7);
    } else {
        pthl = 1.0f;
    }
    pthl = sh_max(pthl, 0.0f);
    pthl = sh_min(pthl, 1.0f);
    return pthl;
}

static inline float sh_ptke(float d, float h) {
    // ptke (:2403-2426): TKE-transport partition (same curve as pu).
    const float a1 = 1.000f, a2 = 0.070f, a3 = 1.000f, a4 = 0.142f,
               a5 = 0.071f;
    const float b2 = 0.6666667f;
    float ptke;
    if (h != 0.0f) {
        float doh = d / h;
        float t2 = doh * doh;
        float tb = pow(doh, b2);
        float num = a1 * t2 + a2 * tb;
        float den = a3 * t2 + a4 * tb + a5;
        ptke = num / den;
    } else {
        ptke = 1.0f;
    }
    ptke = sh_max(ptke, 0.0f);
    ptke = sh_min(ptke, 1.0f);
    return ptke;
}

// ------------------------------------------------------------------------
// The tridiagonal solver (tridi1n :1619 / tridin_ysu :1710) with the
// sequence-association shift applied: both solvers receive the actual
// argument al, dimensioned (its:ite, kts:kte), into a dummy cl dimensioned
// (its:ite, kts+1:kte+1), so cl(i,k) inside the solver is the caller's
// al(i,k-1) -- transcribed as al[k-1] wherever the Fortran says cl(i,k).
// One factorization serves every right-hand side: per RHS the two solvers
// emit the identical operation sequence, and tridin_ysu's per-sweep
// au recomputation is idempotent (see shinhong_ref._tridi_shared).
// Arrays are 1-based; lau is the LU upper band the Fortran calls au.
// ------------------------------------------------------------------------

static inline void sh_tridi_factor(ShColC al, ShColC ad, ShColC cu,
                                ShCol fkk, ShCol lau, int n) {
    fkk[1] = 1.0f / ad[1];
    lau[1] = fkk[1] * cu[1];
    for (int k = 2; k <= n - 1; ++k) {
        fkk[k] = 1.0f / (ad[k] - al[k - 1] * lau[k - 1]);   // cl(i,k) == al[k-1]
        lau[k] = fkk[k] * cu[k];
    }
    fkk[n] = 1.0f / (ad[n] - al[n - 1] * lau[n - 1]);
}

static inline void sh_tridi_solve(ShColC al, ShColC fkk,
                               ShColC lau, ShCol f, int n) {
    f[1] = fkk[1] * f[1];
    for (int k = 2; k <= n - 1; ++k)
        f[k] = fkk[k] * (f[k] - al[k - 1] * f[k - 1]);
    f[n] = fkk[n] * (f[n] - al[n - 1] * f[n - 1]);
    for (int k = n - 1; k >= 1; --k)
        f[k] = f[k] - lau[k] * f[k + 1];
}

// ------------------------------------------------------------------------
// mixlen (:1776-2012), one column, lmh = lmxl = 1.  Writes s2, ri, el
// (1-based, live entries 2..kte); gh is internal.  WRF also returns gh; the
// caller never reads it, so it uses a private workspace slot here.
// ------------------------------------------------------------------------

static inline void sh_mixlen(device const float *u, device const float *v, device const float *theta,
                          device const float *exner, device const float *qv, device const float *qc,
                          int st, int col,
                          ShColC q2, ShColC z, float ustar, float corf,
                          float epshol, float hpbl, int lpbl, bool pblflg,
                          float hgamu, float hgamv, float hgamq,
                          ShColC mf, ShColC ufxpbl,
                          ShColC vfxpbl, ShColC qfxpbl, int kte,
                          ShCol s2, ShCol ri, ShCol el, device float *wsb, int wskp) {
    // Parameters (:1789-1826).  The a1/a2x/b1/b2/c1 decimals are
    // double-precision literals assigned to default REAL: one rounding of
    // the decimal to float32, which is what a C float literal is.  Every
    // derived parameter is folded per operation in float32, in the
    // association order of the expression tree -- the same chains the
    // authority spells; nvrtc folds float chains per-op (ysu.cu's
    // RV / RD - 1.0f measurement is the precedent).
    const float blckdr = 0.0063f, cn = 0.75f, eps1 = 1.0e-12f, epsl = 0.32f;
    const float epsru = 1.0e-7f, epsrs = 1.0e-7f;
    const float el0max = 1000.0f, el0min = 1.0f;
    const float elfc = 0.23f * 0.5f;              // elfc = 0.23*0.5 (:1791)
    const float alph = 0.30f;
    const float beta = 1.0f / 273.0f;             // beta = 1./273. (:1792)
    const float g_qnse = 9.81f;                   // mixlen local g, not the dummy
    const float btg = beta * g_qnse;              // btg = beta*g (:1792)
    const float a1 = 0.659888514560862645f;
    const float a2x = 0.6574209922667784586f;
    const float b1 = 11.87799326209552761f;
    const float b2 = 7.226971804046074028f;
    const float c1 = 0.000830955950095854396f;
    // adnh = 9.*a1*a2x*a2x*(12.*a1+3.*b2)*btg*btg (:1796)
    const float adnh = 9.0f * a1 * a2x * a2x * (12.0f * a1 + 3.0f * b2)
                    * btg * btg;
    // adnm = 18.*a1*a1*a2x*(b2-3.*a2x)*btg (:1797)
    const float adnm = 18.0f * a1 * a1 * a2x * (b2 - 3.0f * a2x) * btg;
    // bdnh = 3.*a2x*(7.*a1+b2)*btg, bdnm = 6.*a1*a1 (:1798)
    const float bdnh = 3.0f * a2x * (7.0f * a1 + b2) * btg;
    const float bdnm = 6.0f * a1 * a1;
    // aeqh, aeqm (:1802-1805)
    const float aeqh = 9.0f * a1 * a2x * a2x * b1 * btg * btg
                    + 9.0f * a1 * a2x * a2x * (12.0f * a1 + 3.0f * b2)
                    * btg * btg;
    const float aeqm = 3.0f * a1 * a2x * b1
                    * (3.0f * a2x + 3.0f * b2 * c1 + 18.0f * a1 * c1 - b2)
                    * btg
                    + 18.0f * a1 * a1 * a2x * (b2 - 3.0f * a2x) * btg;
    // requ = -aeqh/aeqm; epsgm = requ*epsgh (:1809-1810)
    const float requ = -(aeqh / aeqm);
    const float epsgh = 1.0e-9f;
    const float epsgm = requ * epsgh;
    // ubryl/ubry/ubry3 (:1814-1817)
    const float ubryl = (18.0f * requ * a1 * a1 * a2x * b2 * c1 * btg
                        + 9.0f * a1 * a2x * a2x * b2 * btg * btg)
                     / (requ * adnm + adnh);
    const float ubry = (1.0f + epsrs) * ubryl;
    const float ubry3 = 3.0f * ubry;
    // aubh/aubm/bubh/bubm/cubr/rcubr (:1818-1822)
    const float aubh = 27.0f * a1 * a2x * a2x * b2 * btg * btg - adnh * ubry3;
    const float aubm = 54.0f * a1 * a1 * a2x * b2 * c1 * btg - adnm * ubry3;
    const float bubh = (9.0f * a1 * a2x + 3.0f * a2x * b2) * btg
                    - bdnh * ubry3;
    const float bubm = 18.0f * a1 * a1 * c1 - bdnm * ubry3;
    const float cubr = 1.0f - ubry3;
    const float rcubr = 1.0f / cubr;
    const float elcbl = 0.77f;
    const float karman = 0.4f;
    // ep1: WRF's EP_1 forming, see the kernel body's comment.
    const float ep1 = RV / RD - 1.0f;

    ShCol q1 = SHWS_AT(wsb, 0, wskp);
    ShCol dth = SHWS_AT(wsb, 1, wskp);
    ShCol gh = SHWS_AT(wsb, 2, wskp);
    ShCol en2 = SHWS_AT(wsb, 3, wskp);
    ShCol elm = SHWS_AT(wsb, 4, wskp);
    ShCol rel = SHWS_AT(wsb, 5, wskp);
    for (int k = 0; k < wskp; ++k)
        q1[k] = dth[k] = gh[k] = en2[k] = elm[k] = rel[k] = 0.0f;

    const float elocp = 2.72e6f / CP;             // :1880
    const float ct = 0.0f;                        // :1881 (inout, zeroed here)
    for (int k = 2; k <= kte; ++k)               // :1887-1889
        dth[k] = SH_THX(k) - SH_THX(k - 1);
    for (int k = 3; k <= kte; ++k) {             // :1891-1896
        if (dth[k] > 0.0f && dth[k - 1] <= 0.0f) {
            dth[k] = dth[k] + ct;                // ct == 0.; dth[k] > 0 so exact
            break;
        }
    }
    for (int k = kte; k >= 2; --k) {             // :1900-1926
        float rdz = 2.0f / (z[k + 1] - z[k - 1]);
        float du = SH_UX(k) - SH_UX(k - 1);
        float dv = SH_VX(k) - SH_VX(k - 1);
        float s2l = (du * du + dv * dv) * rdz * rdz;   // integer **2 -> multiply
        if (pblflg && k <= lpbl) {
            float suk = (SH_UX(k) - SH_UX(k - 1)) * rdz;
            float svk = (SH_VX(k) - SH_VX(k - 1)) * rdz;
            s2l = (suk - hgamu / hpbl - ufxpbl[k]) * suk
                + (svk - hgamv / hpbl - vfxpbl[k]) * svk;
        }
        s2l = sh_max(s2l, epsgm);
        s2[k] = s2l;
        float tem = (SH_TX(k) + SH_TX(k - 1)) * 0.5f;
        float thm = (SH_THX(k) + SH_THX(k - 1)) * 0.5f;
        float a = thm * ep1;                      // p608 == ep1
        float b = (elocp / tem - 1.0f - ep1) * thm;
        float ghl = (dth[k] * ((SH_QVX(k) + SH_QVX(k - 1) + SH_QCX(k)
                               + SH_QCX(k - 1)) * (0.5f * ep1) + 1.0f)
                    + (SH_QVX(k) - SH_QVX(k - 1) + SH_QCX(k) - SH_QCX(k - 1))
                    * a
                    + (SH_QCX(k) - SH_QCX(k - 1)) * b) * rdz;
        if (pblflg && k <= lpbl)                 // :1918-1920
            ghl = ghl - mf[k] - (hgamq / hpbl + qfxpbl[k]) * a;
        if (fabs(ghl) <= epsgh)
            ghl = epsgh;
        en2[k] = ghl * g_qnse / thm;
        gh[k] = ghl;
        ri[k] = en2[k] / s2l;
    }
    for (int k = kte; k >= 2; --k) {             // :1930-1950
        float s2l = s2[k];
        float ghl = gh[k];
        if (ghl >= epsgh) {
            if (s2l / ghl <= requ) {
                elm[k] = epsl;
            } else {
                float aubr = (aubm * s2l + aubh * ghl) * ghl;
                float bubr = bubm * s2l + bubh * ghl;
                float qol2st = (-0.5f * bubr
                               + sqrt(bubr * bubr * 0.25f - aubr * cubr))
                            * rcubr;
                float eloq2x = 1.0f / qol2st;
                elm[k] = sh_max(sqrt(eloq2x * q2[k]), epsl);
            }
        } else {
            float aden = (adnm * s2l + adnh * ghl) * ghl;
            float bden = bdnm * s2l + bdnh * ghl;
            float qol2un = -0.5f * bden + sqrt(bden * bden * 0.25f - aden);
            float eloq2x = 1.0f / (qol2un + epsru);
            elm[k] = sh_max(sqrt(eloq2x * q2[k]), epsl);
        }
    }
    for (int k = lpbl; k >= 1; --k)              // :1952-1954  (lmh == 1)
        q1[k] = sqrt(q2[k]);
    float szq = 0.0f;
    float sq = 0.0f;
    for (int k = kte; k >= 2; --k) {             // :1956-1962
        float qdzl = (q1[k] + q1[k - 1]) * (z[k] - z[k - 1]);
        szq = (z[k] + z[k - 1] - z[1] - z[1]) * qdzl + szq;
        sq = qdzl + sq;
    }
    float el0 = sh_min(alph * szq * 0.5f / sq, el0max);   // :1966
    el0 = sh_max(el0, el0min);
    int lpblm = min(lpbl + 1, kte);              // :1971
    for (int k = kte; k >= lpblm; --k) {         // :1972-1975
        el[k] = (z[k + 1] - z[k - 1]) * elfc;
        rel[k] = el[k] / elm[k];
    }
    epshol = sh_min(epshol, 0.0f);               // :1979
    float ckp = elcbl * pow(1.0f - 8.0f * epshol, 1.0f / 3.0f);   // :1980
    if (lpbl > 1) {                              // :1981-1992
        for (int k = lpbl; k >= 2; --k) {
            float vkrmz = (z[k] - z[1]) * karman;
            if (pblflg) {
                vkrmz = ckp * (z[k] - z[1]) * karman;
                el[k] = vkrmz / (vkrmz / el0 + 1.0f);
            } else {
                el[k] = vkrmz / (vkrmz / el0 + 1.0f);
            }
            rel[k] = el[k] / elm[k];
        }
    }
    for (int k = lpbl - 1; k >= 3; --k) {        // :1994-1997  (lmh+2 == 3)
        float srel = sh_min(((rel[k - 1] + rel[k + 1]) * 0.5f + rel[k]) * 0.5f,
                           rel[k]);
        el[k] = sh_max(srel * elm[k], epsl);
    }
    float fcor = sh_max(corf, eps1);              // :2001 f = max(corf,eps1)
    float rlambda = fcor / (blckdr * ustar);      // ustar == 0 -> inf -> el == 0
    for (int k = kte; k >= 2; --k) {             // :2003-2010
        if (en2[k] >= 0.0f) {
            float vkrmz = (z[k] - z[1]) * karman;
            float rlb = rlambda + 1.0f / vkrmz;
            float rln = sh_div(sqrt(2.0f * en2[k] / q2[k]), cn);
            el[k] = 1.0f / (rlb + rln);
        }
    }
}

// ------------------------------------------------------------------------
// prodq2 (:2016-2125), one column.  Mutates q2 in place.  Note dtturbl is
// dt, NOT dt2 (:1578 passes the wrapper's dt).  WRF also passes s2 and ri
// here; s2 feeds only a dead local and ri is never read, so both stay out.
// ------------------------------------------------------------------------

static inline void sh_prodq2(float dtturbl, float ustar,
                          device const float *u, device const float *v, device const float *theta,
                          int st, int col, ShColC thvx,
                          ShCol q2, ShColC el, ShColC z,
                          ShColC akm, ShColC akh,
                          float hgamu, float hgamv, float hgamq, float delxy,
                          float hpbl, bool pblflg, int kpbl,
                          ShColC mf, ShColC ufxpbl,
                          ShColC vfxpbl, ShColC qfxpbl, int kte) {
    const float g_qnse = 9.81f;                   // prodq2 local g (:2029 block)
    const float c0 = 0.55f, ceps = 16.6f;
    const float epsq2l = 0.01f;
    const float ep1 = RV / RD - 1.0f;
    const float rc02 = 2.0f / (c0 * c0);          // :2073
    for (int k = 2; k <= kte; ++k) {             // :2077-2119
        float deltaz = 0.5f * (z[k + 1] - z[k - 1]);
        float q2l = q2[k];
        float suk = (SH_UX(k) - SH_UX(k - 1)) / deltaz;
        float svk = (SH_VX(k) - SH_VX(k - 1)) / deltaz;
        float gthvk = (thvx[k] - thvx[k - 1]) / deltaz;
        float govrthvk = g_qnse / (0.5f * (thvx[k] + thvx[k - 1]));
        float akml = akm[k];
        float akhl = akh[k];
        float thm = (SH_THX(k) + SH_THX(k - 1)) * 0.5f;
        float pru, prv;
        if (pblflg && k <= kpbl) {               // :2092-2098
            pru = (akml * (suk - hgamu / hpbl - ufxpbl[k])) * suk;
            prv = (akml * (svk - hgamv / hpbl - vfxpbl[k])) * svk;
        } else {
            pru = akml * suk * suk;
            prv = akml * svk * svk;
        }
        float pr = pru + prv;
        float bpr;
        if (pblflg && k <= kpbl) {               // :2103-2107
            bpr = (akhl * (gthvk - mf[k]
                           - (hgamq / hpbl + qfxpbl[k]) * ep1 * thm))
                * govrthvk;
        } else {
            bpr = akhl * gthvk * govrthvk;
        }
        float disel = sh_min(delxy, ceps * el[k]);   // :2111
        float dis = pow(q2l, 1.5f) / disel;         // (q2l)**1.5 stays powf
        q2l = q2l + 2.0f * (pr - bpr - dis) * dtturbl;
        q2[k] = sh_max(q2l, epsq2l);                // amax1: NaN propagates
    }
    q2[1] = sh_max(rc02 * ustar * ustar, epsq2l);   // :2123
}

// ------------------------------------------------------------------------
// vdifq (:2129-2231), one column, lmh = 1.  Mutates q2.  The akq/cm/cr/dtoz/
// rsq2 arrays run kts+2..kte in the Fortran (:2172-2176); the loops below
// are transcribed index-exactly against that (trap: the lower boundary
// handling at :2212-2225 reaches akq(3)/cm(3)/cr(3) only).  WRF also passes
// el here and never reads it.
// ------------------------------------------------------------------------

static inline void sh_vdifq(float dtdif, ShCol q2, ShColC z, ShColC akhk,
                         ShColC ptke1, ShColC hgame, float hpbl,
                         bool pblflg, int kpbl, float efxpbl, int kte, device float *wsb, int wskp) {
    const float c_k = 1.0f;
    const int lmh = 1;
    ShCol zfacentk = SHWS_AT(wsb, 6, wskp);
    ShCol dtoz = SHWS_AT(wsb, 7, wskp);
    ShCol akq = SHWS_AT(wsb, 8, wskp);
    ShCol cr = SHWS_AT(wsb, 9, wskp);
    ShCol cm = SHWS_AT(wsb, 10, wskp);
    ShCol rsq2 = SHWS_AT(wsb, 11, wskp);
    for (int k = 0; k < wskp; ++k)
        zfacentk[k] = dtoz[k] = akq[k] = cr[k] = cm[k] = rsq2[k] = 0.0f;

    for (int k = 2; k <= kte; ++k) {             // :2183-2186
        float zak = 0.5f * (z[k] + z[k - 1]);     // za(k-1) of shinhong2d
        zfacentk[k] = pow(zak / hpbl, 3.0f);    // ((zak/hpbl))**3.0 -- a powf
    }
    for (int k = kte; k >= 3; --k) {             // :2188-2193 (kte .. kts+2)
        dtoz[k] = (dtdif + dtdif) / (z[k + 1] - z[k - 1]);
        akq[k] = c_k * (akhk[k] / (z[k + 1] - z[k - 1])
                        + akhk[k - 1] / (z[k] - z[k - 2]));
        akq[k] = akq[k] * ptke1[k];
        cr[k] = -(dtoz[k] * akq[k]);
    }
    float akqs = c_k * akhk[2] / (z[3] - z[1]);   // :2195-2196
    akqs = akqs * ptke1[2];
    cm[kte] = dtoz[kte] * akq[kte] + 1.0f;       // :2197-2198
    rsq2[kte] = q2[kte];
    for (int k = kte - 1; k >= 3; --k) {         // :2200-2210
        float cf = -(dtoz[k] * akq[k + 1] / cm[k + 1]);
        cm[k] = -(cr[k + 1] * cf) + (akq[k + 1] + akq[k]) * dtoz[k] + 1.0f;
        rsq2[k] = -(rsq2[k + 1] * cf) + q2[k];
        if (pblflg && k < kpbl) {
            rsq2[k] = rsq2[k]
                    - dtoz[k] * (2.0f * hgame[k] / hpbl) * akq[k + 1]
                    * (z[k + 1] - z[k])
                    + dtoz[k] * (2.0f * hgame[k - 1] / hpbl) * akq[k]
                    * (z[k] - z[k - 1]);
            rsq2[k] = rsq2[k]
                    - dtoz[k] * 2.0f * efxpbl * zfacentk[k + 1]
                    + dtoz[k] * 2.0f * efxpbl * zfacentk[k];
        }
    }
    float dtozs = (dtdif + dtdif) / (z[3] - z[1]);   // :2212
    float cf = -(dtozs * akq[lmh + 2] / cm[lmh + 2]);   // :2213
    if (pblflg && (lmh + 1) < kpbl) {            // :2215-2225
        q2[2] = dtozs * akqs * q2[1] - rsq2[3] * cf + q2[2]
              - dtozs * (2.0f * hgame[2] / hpbl) * akq[3] * (z[3] - z[2])
              + dtozs * (2.0f * hgame[1] / hpbl) * akqs * (z[2] - z[1]);
        q2[2] = q2[2] - dtozs * 2.0f * efxpbl * zfacentk[3]
              + dtozs * 2.0f * efxpbl * zfacentk[2];
        q2[2] = q2[2] / ((akq[3] + akqs) * dtozs - cr[3] * cf + 1.0f);
    } else {
        q2[2] = (dtozs * akqs * q2[1] - rsq2[3] * cf + q2[2])
              / ((akq[3] + akqs) * dtozs - cr[3] * cf + 1.0f);
    }
    for (int k = 3; k <= kte; ++k)               // :2227-2229 (lmh+2 .. kte)
        q2[k] = (-(cr[k] * q2[k - 1]) + rsq2[k]) / cm[k];
}

// ------------------------------------------------------------------------
// The column: shinhong (:9-262) around shinhong2d (:266-1615), arm A.
// ------------------------------------------------------------------------

kernel void shinhong_column(
    constant ShinhongColumnResources &r [[buffer(0)]],
    constant ShinhongColumnParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]],
    uint lane_index [[thread_position_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]) {

    const int nz = int(p.nz), ny = int(p.ny), nx = int(p.nx);
    const int col0 = int(p.col0), wskp = int(p.wskp);
    const int tke_diag = int(p.tke_diag);
    const float dt = p.dt, dx = p.dx, dy = p.dy;
    device const float *u=r.u, *v=r.v, *theta=r.theta, *qv=r.qv, *qc=r.qc, *qi=r.qi;
    device const float *pfield=r.p, *p_interface=r.p_interface, *exner=r.exner, *dz=r.dz, *tke=r.tke;
    device const float *psfc=r.psfc, *znt=r.znt, *ust=r.ust, *hfx=r.hfx, *qfx=r.qfx;
    device const float *wspd=r.wspd, *br=r.br, *psim=r.psim, *psih=r.psih, *xland=r.xland;
    device const float *u10=r.u10, *v10=r.v10, *corf=r.corf;
    device float *du=r.du, *dv=r.dv, *dtheta=r.dtheta, *dqv=r.dqv, *dqc=r.dqc, *dqi=r.dqi;
    device float *exch_h=r.exch_h, *tke_out=r.tke_out, *el=r.el, *hpbl_out=r.hpbl;
    device int *kpbl_out=r.kpbl;
    device float *wstar_out=r.wstar, *delta_out=r.delta, *ws=r.ws;
    int col = col0 + int(flat);
    int st = ny * nx;
    if (col >= st || nz > SHINHONG_KMAX) return;
    device float *wsb = SHWS_LANE_BASE(ws, wskp);
    const int kte = nz;
    const int klpbl = kte;                       // klpbl = kte (:531 region)

    // shinhong2d parameters (:321-346), WRF literals.
    const float xkzminm = 0.1f, xkzminh = 0.01f, xkzmax = 1000.0f;
    const float rimin = -100.0f, rlam = 30.0f, prmin = 0.25f, prmax = 4.0f;
    const float brcr_ub = 0.0f, brcr_sb = 0.25f, cori = 1.0e-4f;
    const float afac = 6.8f, bfac = 6.8f, pfac = 2.0f, pfac_q = 2.0f;
    const float phifac = 8.0f, sfcfrac = 0.1f;
    const float d1 = 0.02f, d2 = 0.05f, d3 = 0.001f;
    const float h1 = 0.33333333f;   // rounds to float32 1/3 (pow-probe.txt)
    const float h2 = 0.6666667f;
    const float zfmin = 1.0e-8f, aphi5 = 5.0f, aphi16 = 16.0f;
    const float tmin = 1.0e-2f, gamcrt = 3.0f, gamcrq = 2.0e-3f;
    const float epsq2l = 0.01f, c_1 = 1.0f, gamcre = 0.224f;
    const float mltop = 1.0f, sfcfracn1 = 0.075f, nlfrac = 0.7f;
    const float enlfrac = -0.4f;
    const float a11 = 1.0f, a12 = -1.15f, ezfac = 1.5f;
    const float cpent = -0.4f, rigsmax = 100.0f, entfmin = 1.0f,
               entfmax = 5.0f;
    // ep1 is WRF's EP_1, and WRF forms it in float32: module_model_constants
    // declares `REAL, PARAMETER :: EP_1 = R_v/R_d - 1.` so the quotient is a
    // float32 divide of two float32 values.  CUDA_DEFINES["RVOVRD"] is RV/RD
    // computed in Python doubles and rounded once to float32, which lands 1
    // ULP below WRF's quotient at 1.608 and therefore 2 ULP below it here at
    // 0.608.  Spelling the divide keeps the whole thv/Richardson/PBL-diagnosis
    // chain on WRF's constant (ysu.cu precedent, measured there).
    const float karman = 0.4f, ep1 = RV / RD - 1.0f;

    ShCol thvx = SHWS_AT(wsb, 12, wskp);
    ShCol zq = SHWS_AT(wsb, 13, wskp);
    ShCol za = SHWS_AT(wsb, 14, wskp);
    ShCol delp = SHWS_AT(wsb, 15, wskp);
    ShCol dza = SHWS_AT(wsb, 16, wskp);
    ShCol q2x = SHWS_AT(wsb, 17, wskp);
    ShCol hgame2d = SHWS_AT(wsb, 18, wskp);
    ShCol tflux_e = SHWS_AT(wsb, 19, wskp);
    ShCol qflux_e = SHWS_AT(wsb, 20, wskp);
    ShCol tvflux_e = SHWS_AT(wsb, 21, wskp);
    ShCol mf = SHWS_AT(wsb, 22, wskp);
    ShCol entfacmf = SHWS_AT(wsb, 23, wskp);
    ShCol entfac = SHWS_AT(wsb, 24, wskp);
    ShCol xkzm = SHWS_AT(wsb, 25, wskp);
    ShCol xkzh = SHWS_AT(wsb, 26, wskp);
    ShCol xkzq = SHWS_AT(wsb, 27, wskp);
    ShCol xkzml = SHWS_AT(wsb, 28, wskp);
    ShCol xkzhl = SHWS_AT(wsb, 29, wskp);
    ShCol zfacent = SHWS_AT(wsb, 30, wskp);
    ShCol al = SHWS_AT(wsb, 31, wskp);
    ShCol ad = SHWS_AT(wsb, 32, wskp);
    ShCol au = SHWS_AT(wsb, 33, wskp);
    ShCol fkk = SHWS_AT(wsb, 34, wskp);
    ShCol lau = SHWS_AT(wsb, 35, wskp);
    // r1/r2/r3 are the solver right-hand sides: heat uses r1 (f1 :1141);
    // moisture uses r1/r2/r3 (f3 components 1..3, :1230-1245); momentum
    // reuses r1/r2 (f1/f2, :1389-1390).
    ShCol r1 = SHWS_AT(wsb, 36, wskp);
    ShCol r2 = SHWS_AT(wsb, 37, wskp);
    ShCol r3 = SHWS_AT(wsb, 38, wskp);
    for (int k = 0; k < wskp; ++k) {
        thvx[k] = zq[k] = za[k] = delp[k] = dza[k] = 0.0f;
        q2x[k] = hgame2d[k] = 0.0f;
        tflux_e[k] = qflux_e[k] = tvflux_e[k] = 0.0f;
        mf[k] = entfacmf[k] = entfac[k] = 0.0f;
        xkzm[k] = xkzh[k] = xkzq[k] = xkzml[k] = xkzhl[k] = zfacent[k] = 0.0f;
        al[k] = ad[k] = au[k] = fkk[k] = lau[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }

    const float psfcv = psfc[col], zntv = znt[col], ustv = ust[col];
    const float hfxv = hfx[col], qfxv = qfx[col], wspdv = wspd[col];
    const float brv = br[col], psimv = psim[col], psihv = psih[col];
    const float xlandv = xland[col], corfv = corf[col];
    const float u10v = u10[col], v10v = v10[col];

    const float cont = CP / G;                    // :544
    const float conpr = bfac * karman * sfcfrac;  // :548
    // conq/conw/conwrc (:545-547) feed only the dusfc/dvsfc/dtsfc/dqsfc
    // accumulators, which the wrapper never reads; not transcribed.

    // thx is the model's theta itself (see SH_THX above); thvx (:563-568).
    for (int k = 1; k <= kte; ++k) {
        float tvcon = 1.0f + ep1 * SH_QVX(k);
        thvx[k] = SH_THX(k) * tvcon;
    }
    float tvcon = 1.0f + ep1 * SH_QVX(1);         // :570-574
    float rhox = psfcv / (RD * SH_TX(1) * tvcon);
    float govrth = G / SH_THX(1);

    zq[1] = 0.0f;                                // :579-587 (zq(:,1) = 0)
    for (int k = 1; k <= kte; ++k)
        zq[k + 1] = SH_DZ8W(k) + zq[k];
    for (int k = 1; k <= kte; ++k) {             // :589-595 (dzq never read)
        za[k] = 0.5f * (zq[k] + zq[k + 1]);
        delp[k] = p_interface[(ulong)(k - 1) * st + col]
                - p_interface[(ulong)k * st + col];
    }
    dza[1] = za[1];                              // :597-605
    for (int k = 2; k <= kte; ++k)
        dza[k] = za[k] - za[k - 1];

    float wspd1 = sqrt(SH_UX(1) * SH_UX(1) + SH_VX(1) * SH_VX(1))
               + 1.0e-9f;                        // :616

    float dt2 = 2.0f * dt;                        // :624-626
    float rdt = 1.0f / dt2;

    float hgamu = 0.0f, hgamv = 0.0f, delta = 0.0f, efxpbl = 0.0f;
    float epshol = 0.0f, deltaoh = 0.0f, rigs = 0.0f, enlfrac2 = 0.0f;
    float cslen = 0.0f;
    for (int k = 1; k <= kte; ++k)               // :665-669
        q2x[k] = 2.0f * SH_TKEX(k);
    // :671-679 -- el_pbl is zeroed UNCONDITIONALLY, tke_diag on or off, and
    // tke passes through untouched when the diagnostic is off (:1478).
    for (int k = 1; k <= kte; ++k) {
        SH_G(el, k) = 0.0f;
        SH_G(tke_out, k) = SH_TKEX(k);
    }
    // xkzom/xkzoh (:688-693) hold xkzminm/xkzminh on kts..klpbl-1 and zero at
    // klpbl == kte.  Every read below sits at k <= kte-1 (the k < kpbl loop,
    // the free atmosphere, and the three entrainment overwrites all stop at
    // kte-1), so the constants stand in for the arrays.

    float hgamt = 0.0f, hgamq = 0.0f;             // :702-715
    int kpbl = 1;
    float hpbl = zq[1];
    float zl1 = za[1];
    float thermal = thvx[1];
    bool pblflg = true;
    bool sfcflg = true;
    float sflux = sh_div(hfxv / rhox, CP) + qfxv / rhox * ep1 * SH_THX(1);
    // :706 `if(br(i).gt.0.0) sfcflg = .false.` must preserve subnormal
    // comparisons.  The IEEE bit check below tests positivity without
    // arithmetic or native double support.
    if (sh_positive_float(brv))
        sfcflg = false;

    // ---- first guess of the PBL height (:719-749) ----
    bool stable = false;
    float brup = brv;
    float brcr = brcr_ub;
    float brdn = 0.0f;
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable) {
            brdn = brup;
            float spdk2 = sh_max(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    {
        int k = kpbl;
        float brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    // ---- surface-layer scales (:751-780) ----
    float fm = psimv;
    float fh = psihv;
    float zol1 = sh_max(brv * fm * fm / fh, rimin);
    if (sfcflg) zol1 = sh_min(zol1, -zfmin);
    else zol1 = sh_max(zol1, zfmin);
    float hol1 = zol1 * hpbl / zl1 * sfcfrac;
    epshol = hol1;
    float phim, phih, wstar, wstar3;
    if (sfcflg) {
        phim = pow(1.0f - aphi16 * hol1, -(1.0f / 4.0f));
        phih = pow(1.0f - aphi16 * hol1, -(1.0f / 2.0f));
        float bfx0 = sh_max(sflux, 0.0f);
        // hfx0/qfx0 (:766-767) are computed by WRF and never read; omitted.
        wstar3 = govrth * bfx0 * hpbl;
        wstar = pow(wstar3, h1);
    } else {
        phim = 1.0f + aphi5 * hol1;
        phih = phim;
        wstar = 0.0f;
        wstar3 = 0.0f;
    }
    float ust3 = pow(ustv, 3.0f);                // ust**3. stays powf
    float wscale = pow(ust3 + phifac * karman * wstar3 * 0.5f, h1);
    wscale = sh_min(wscale, ustv * aphi16);
    wscale = sh_max(wscale, sh_div(ustv, aphi5));

    // ---- countergradient terms (:785-800) ----
    if (sfcflg && sflux > 0.0f) {
        float gamfac = bfac / rhox / wscale;
        hgamt = sh_min(sh_div(gamfac * hfxv, CP), gamcrt);
        hgamq = sh_min(gamfac * qfxv, gamcrq);
        float vpert = sh_div((hgamt + ep1 * SH_THX(1) * hgamq), bfac) * afac;
        thermal = thermal + (sh_max(vpert, 0.0f)
                             * sh_min(za[1] / (sfcfrac * hpbl), 1.0f));
        hgamt = sh_max(hgamt, 0.0f);
        hgamq = sh_max(hgamq, 0.0f);
        float brint = -15.9f * ustv * ustv / wspdv * wstar3
                   / pow(wscale, 4.0f);         // wscale**4. stays powf
        hgamu = brint * SH_UX(1);
        hgamv = brint * SH_VX(1);
    } else {
        pblflg = false;
    }

    // ---- enhance the PBL height by the thermal excess (:804-853) ----
    if (pblflg) {
        kpbl = 1;
        hpbl = zq[1];
        stable = false;
        brup = brv;
        brcr = brcr_ub;
    }
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable && pblflg) {
            brdn = brup;
            float spdk2 = sh_max(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    if (pblflg) {
        int k = kpbl;
        float brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
        // :844-851 -- still inside the outer if(pblflg) block entered above,
        // so csfac/cslen are computed even when pblflg was just falsified.
        float csfac;
        if (wstar != 0.0f) {
            float uwst = fabs(ustv / wstar - 0.5f);
            float uwstx = -80.0f * uwst + 14.0f;
            csfac = 0.5f * (tanh(uwstx) + 3.0f);
        } else {
            csfac = 1.0f;   // dead in WRF: pblflg requires sflux>0 => wstar>0
        }
        cslen = csfac * hpbl;
    }

    // ---- stable boundary layer (:857-912) ----
    if (!sfcflg && hpbl < zq[2]) {
        brup = brv;
        stable = false;
    } else {
        stable = true;
    }
    float brcr_sbro = 0.0f;
    if (!stable && (xlandv - 1.5f) >= 0.0f) {    // :867-874
        float wspd10 = u10v * u10v + v10v * v10v;
        wspd10 = sqrt(wspd10);
        float ross = wspd10 / (cori * zntv);
        brcr_sbro = sh_min(0.16f * pow(1.0e-7f * ross, -0.18f), 0.3f);
    }
    if (!stable) {                               // :876-884
        if ((xlandv - 1.5f) >= 0.0f) brcr = brcr_sbro;
        else brcr = brcr_sb;
    }
    for (int k = 2; k <= klpbl; ++k) {           // :886-896
        if (!stable) {
            brdn = brup;
            float spdk2 = sh_max(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    if (!sfcflg && hpbl < zq[2]) {               // :898-912
        int k = kpbl;
        float brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    // ---- scale dependency, nonlocal momentum and moisture (:916-926) ----
    float delxy = sqrt(dx * dy);
    float pu1 = sh_pu(delxy, cslen);              // cslen == 0 columns take the
    float pq1 = sh_pq(delxy, cslen);              // h == 0 early-out and get 1.
    if (pblflg) {
        hgamu = hgamu * pu1;
        hgamv = hgamv * pu1;
        hgamq = hgamq * pq1;
    }

    // ---- entrainment parameters (:930-985) ----
    float prpbl = 0.0f;
    float wm2 = 0.0f;
    float we = 0.0f;
    float bfxpbl = 0.0f;                          // :629-637
    float hfxpbl = 0.0f;
    float qfxpbl = 0.0f;
    float ufxpbl = 0.0f;
    float vfxpbl = 0.0f;
    float dthvx = 0.0f;
    if (pblflg) {
        int kt = kpbl - 1;                       // k = kpbl - 1 (:931)
        prpbl = 1.0f;
        float wm3 = wstar3 + 5.0f * ust3;
        wm2 = pow(wm3, h2);
        bfxpbl = sh_div(-0.15f * thvx[1], G) * wm3 / hpbl;
        dthvx = sh_max(thvx[kt + 1] - thvx[kt], tmin);
        float dthx = sh_max(SH_THX(kt + 1) - SH_THX(kt), tmin);
        float dqx = sh_min(SH_QVX(kt + 1) - SH_QVX(kt), 0.0f);
        we = sh_max(bfxpbl / dthvx, -sqrt(wm2));
        hfxpbl = we * dthx;                      // :943 -- overwritten at
        pq1 = sh_pq(delxy, cslen);               // :1114; ported as written
        qfxpbl = we * dqx * pq1;
        pu1 = sh_pu(delxy, cslen);
        float dux = SH_UX(kt + 1) - SH_UX(kt);
        float dvx = SH_VX(kt + 1) - SH_VX(kt);
        if (dux > tmin)
            ufxpbl = sh_max(prpbl * we * dux * pu1, -(ustv * ustv));
        else if (dux < -tmin)
            ufxpbl = sh_min(prpbl * we * dux * pu1, ustv * ustv);
        else
            ufxpbl = 0.0f;
        if (dvx > tmin)
            vfxpbl = sh_max(prpbl * we * dvx * pu1, -(ustv * ustv));
        else if (dvx < -tmin)
            vfxpbl = sh_min(prpbl * we * dvx * pu1, ustv * ustv);
        else
            vfxpbl = 0.0f;
        float delb = govrth * d3 * hpbl;
        delta = sh_min(d1 * hpbl + d2 * wm2 / delb, 100.0f);
        delb = govrth * dthvx;
        deltaoh = d1 * hpbl + d2 * wm2 / delb;
        deltaoh = sh_max(ezfac * deltaoh, hpbl - za[kpbl - 1] - 1.0f);
        deltaoh = sh_min(deltaoh, hpbl);
        if (dux != 0.0f || dvx != 0.0f) {
            // dux**2. and dvx**2. fold to multiplies (pow-probe, negative
            // bases included).
            rigs = govrth * dthvx * deltaoh / (dux * dux + dvx * dvx);
        } else {
            rigs = rigsmax;
        }
        rigs = sh_max(sh_min(rigs, rigsmax), rimin);
        float cenlfrac;
        if (rigs > 0.0f && fabs(rigs + cpent) <= 1.0e-6f)
            cenlfrac = entfmax;
        else
            cenlfrac = rigs / (rigs + cpent);
        cenlfrac = sh_min(cenlfrac, entfmax);
        enlfrac2 = sh_max(wm3 / wstar3 * cenlfrac, entfmin);
        enlfrac2 = enlfrac2 * enlfrac;
    }

    for (int k = 1; k <= klpbl; ++k) {           // :987-998
        if (pblflg) {
            float tmf = (zq[k + 1] - hpbl) / deltaoh;
            entfacmf[k] = sqrt(tmf * tmf);      // sqrt((...)**2.), **2. folds
        }
        if (pblflg && k >= kpbl) {
            float te = (zq[k + 1] - hpbl) / deltaoh;
            entfac[k] = te * te;
        } else {
            entfac[k] = 1.0e30f;
        }
    }

    // ---- diffusion coefficients below the PBL top (:1002-1036) ----
    for (int k = 1; k <= klpbl; ++k) {
        if (k < kpbl) {
            float zfac = sh_min(sh_max(
                1.0f - (zq[k + 1] - zl1) / (hpbl - zl1), zfmin), 1.0f);
            zfacent[k] = pow(1.0f - zfac, 3.0f);   // (1.-zfac)**3. -- powf
            // The ONE cbrtf site (see the header's per-site table): powf
            // left dv's worst column at 1491308 ULP; under NVRTC 13.0.48
            // cbrtf lands closer to the oracle here (dv 46603, exch_h 8) for
            // tke 2005 -> 2022.  Under the CUDA 12 line only the exch_h
            // 9 -> 8 half survives -- the row is compiler-specific and the
            // header says why.  Do not "simplify" this back to powf.
            float wscalek = pow(
                ust3 + phifac * karman * wstar3 * (1.0f - zfac),
                0.3333333333333333f);
            float prfac, prfac2, prnumfac;
            if (sfcflg) {
                prfac = conpr;
                // :1010 -- WRF's own 0/0 when both cubes are exactly zero
                // (fixture case 13); the NaN column downstream is the point.
                prfac2 = 15.9f * wstar3 / ust3
                       / (1.0f + 4.0f * karman * wstar3 / ust3);
                float m = sh_max(zq[k + 1] - sfcfrac * hpbl, 0.0f);
                prnumfac = -3.0f * (m * m) / (hpbl * hpbl);   // **2. folds
            } else {
                prfac = 0.0f;
                prfac2 = 0.0f;
                prnumfac = 0.0f;
                float phim8z = 1.0f + aphi5 * zol1 * zq[k + 1] / zl1;
                wscalek = ustv / phim8z;
                wscalek = sh_max(wscalek, 0.001f);
            }
            float prnum0 = phih / phim + prfac;
            prnum0 = sh_max(sh_min(prnum0, prmax), prmin);
            xkzm[k] = wscalek * karman * zq[k + 1] * (zfac * zfac);   // **pfac folds
            float prnum = 1.0f + (prnum0 - 1.0f) * exp(prnumfac);
            // zfac**(pfac_q-pfac) == zfac**0. == 1.0 for every zfac
            // (pow-probe.txt, including 0; device pow(x, 0) is 1 for all x);
            // the multiply is kept.
            xkzq[k] = xkzm[k] / prnum * pow(zfac, pfac_q - pfac);
            prnum0 = prnum0 / (1.0f + prfac2 * karman * sfcfrac);
            prnum = 1.0f + (prnum0 - 1.0f) * exp(prnumfac);
            xkzh[k] = xkzm[k] / prnum;
            xkzm[k] = xkzm[k] + xkzminm;         // + xkzom(k), k <= kte-1
            xkzh[k] = xkzh[k] + xkzminh;         // + xkzoh(k)
            xkzq[k] = xkzq[k] + xkzminh;
            xkzm[k] = sh_min(xkzm[k], xkzmax);
            xkzh[k] = sh_min(xkzh[k], xkzmax);
            xkzq[k] = sh_min(xkzq[k], xkzmax);
        }
    }

    // ---- free atmosphere (:1040-1086) ----
    for (int k = 1; k <= kte - 1; ++k) {
        if (k >= kpbl) {
            float duk = SH_UX(k + 1) - SH_UX(k);
            float dvk = SH_VX(k + 1) - SH_VX(k);
            float ss = (duk * duk + dvk * dvk) / (dza[k + 1] * dza[k + 1])
                    + 1.0e-9f;
            float govrthv = G / (0.5f * (thvx[k + 1] + thvx[k]));
            float ri = govrthv * (thvx[k + 1] - thvx[k]) / (ss * dza[k + 1]);
            // :1049-1058 in cloud (imvdif == 1, nwmass == 3 always):
            // qx(i,kqc+k-1) is qc(k) and qx(i,kqi+k-1) is qi(k) in the
            // ndiff-packed array (kqc = 1+kte, kqi = 1+2*kte).
            if ((SH_QCX(k) + SH_QIX(k)) > 0.01e-3f
                    && (SH_QCX(k + 1) + SH_QIX(k + 1)) > 0.01e-3f) {
                float qmean = 0.5f * (SH_QVX(k) + SH_QVX(k + 1));
                float tmean = 0.5f * (SH_TX(k) + SH_TX(k + 1));
                float alpha = sh_div(XLV * qmean, RD) / tmean;
                float chi = sh_div(sh_div(XLV * XLV * qmean, CP), RV) / tmean / tmean;
                ri = (1.0f + alpha) * (
                    ri - sh_div(G * G / ss / tmean, CP)
                    * ((chi - alpha) / (1.0f + chi)));
            }
            float zk = karman * zq[k + 1];
            float rlamdz = sh_min(sh_max(0.1f * dza[k + 1], rlam), 300.0f);
            rlamdz = sh_min(dza[k + 1], rlamdz);
            float tt = zk * rlamdz / (rlamdz + zk);
            float rl2 = tt * tt;                  // integer **2 -> multiply
            float dk = rl2 * sqrt(ss);
            if (ri < 0.0f) {
                ri = sh_max(ri, rimin);
                float sri = sqrt(-ri);
                xkzm[k] = dk * (1.0f + 8.0f * (-ri)
                                / (1.0f + 1.746f * sri));
                xkzh[k] = dk * (1.0f + 8.0f * (-ri)
                                / (1.0f + 1.286f * sri));
            } else {
                tt = 1.0f + 5.0f * ri;
                xkzh[k] = dk / (tt * tt);        // integer **2 -> multiply
                float prnum = 1.0f + 2.1f * ri;
                prnum = sh_min(prnum, prmax);
                xkzm[k] = xkzh[k] * prnum;
            }
            xkzm[k] = xkzm[k] + xkzminm;         // + xkzom(k), k <= kte-1
            xkzh[k] = xkzh[k] + xkzminh;
            xkzm[k] = sh_min(xkzm[k], xkzmax);
            xkzh[k] = sh_min(xkzh[k], xkzmax);
            xkzml[k] = xkzm[k];
            xkzhl[k] = xkzh[k];
        }
    }

    // ---- prescribed nonlocal heat transport (:1090-1130) ----
    deltaoh = deltaoh / hpbl;                    // :1091, every column
    delxy = sqrt(dx * dy);
    float mlfrac = mltop - deltaoh;
    // ezfrac (:1097) is computed by WRF and never read; omitted.
    // :1098 -- zfacmf(i,1) is first computed CLAMPED and feeds sfcfracn; the
    // k-loop below then RE-computes it UNCLAMPED (:1119), overwriting index 1,
    // and the mf profile uses the unclamped value.
    float zfacmf1 = sh_min(sh_max(zq[2] / hpbl, zfmin), 1.0f);
    float sfcfracn = sh_max(sfcfracn1, zfacmf1);
    float sflux0 = (a11 + a12 * sfcfracn) * sflux;
    float snlflux0 = nlfrac * sflux0;
    float amf1 = snlflux0 / sfcfracn;
    float amf2 = 0.0f;
    float bmf2 = 0.0f;
    if (pblflg) {
        amf2 = -(snlflux0 / (mlfrac - sfcfracn));
        bmf2 = -(mlfrac * amf2);
    }
    float amf3;
    if (deltaoh == 0.0f && enlfrac2 == 0.0f)     // :1108, exact compares
        amf3 = 0.0f;
    else
        amf3 = snlflux0 * enlfrac2 / deltaoh;
    float bmf3 = -(amf3 * mlfrac);
    // :1114-1116 -- hfxpbl is OVERWRITTEN here (the :943 we*dthx value is
    // discarded) and scaled by pthnl once ...
    hfxpbl = amf3 + bmf3;
    float pth1 = sh_pthnl(delxy, cslen);
    hfxpbl = hfxpbl * pth1;
    for (int k = 1; k <= klpbl; ++k) {
        float zfacmf = sh_max(zq[k + 1] / hpbl, zfmin);   // :1119, unclamped
        if (pblflg && k < kpbl) {
            if (zfacmf <= sfcfracn)
                mf[k] = amf1 * zfacmf;
            else if (zfacmf <= mlfrac)
                mf[k] = amf2 * zfacmf + bmf2;
            // ... and the WHOLE mf -- entrainment term included -- is scaled
            // by pth1 AGAIN at :1127, so the entrainment piece is
            // double-scaled.  WRF's arithmetic, ported as written.
            mf[k] = mf[k] + hfxpbl * exp(-entfacmf[k]);
            mf[k] = mf[k] * pth1;
        }
    }

    // ---- heat: matrix assembly (:1134-1187), solve, recover ----
    ad[1] = 1.0f;
    r1[1] = SH_THX(1) - 300.0f + sh_div(hfxv, cont) / delp[1] * dt2;
    SH_G(exch_h, 1) = 0.0f;   // exch_h(:,1) is never written by the scheme
    for (int k = 1; k <= kte - 1; ++k) {
        float dtodsd = dt2 / delp[k];
        float dtodsu = dt2 / delp[k + 1];
        float dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        float rdz = 1.0f / dza[k + 1];
        float tem1 = dsig * xkzh[k] * rdz;
        if (pblflg && k < kpbl) {
            float dsdzt = tem1 * (-(mf[k] / xkzh[k]));
            r1[k] = r1[k] + dtodsd * dsdzt;
            r1[k + 1] = SH_THX(k + 1) - 300.0f - dtodsu * dsdzt;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            // :1161-1164 -- the entrainment-zone overwrite of xkzh ...
            xkzh[k] = -(we * dza[kpbl] * exp(-entfac[k]));
            xkzh[k] = sqrt(xkzh[k] * xkzhl[k]);
            xkzh[k] = sh_max(xkzh[k], xkzminh);
            xkzh[k] = sh_min(xkzh[k], xkzmax);
            r1[k + 1] = SH_THX(k + 1) - 300.0f;
        } else {
            r1[k + 1] = SH_THX(k + 1) - 300.0f;
        }
        // :1169 -- ... after which tem1 is RECOMPUTED from the updated
        // coefficient before au/al.  Sequence kept exactly.
        tem1 = dsig * xkzh[k] * rdz;
        float dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2);
        al[k] = -(dtodsu * dsdz2);
        // :1176-1178 -- LOCAL heat partition: h = hpbl (not cslen) and delxy
        // scaled by zfacdx; the NONLOCAL calls above use h = cslen, plain
        // delxy.
        float zfacdx = 0.2f * hpbl / zq[k + 1];
        float delxy_l = sqrt(dx * dy) * sh_max(zfacdx, 1.0f);
        float pthl1 = sh_pthl(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pthl1;
            al[k] = al[k] * pthl1;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
        SH_G(exch_h, k + 1) = xkzh[k];           // :1185, after the overwrite
    }
    sh_tridi_factor(al, ad, au, fkk, lau, kte);  // tridi1n :1198
    sh_tridi_solve(al, fkk, lau, r1, kte);
    for (int k = kte; k >= 1; --k) {             // :1202-1213
        float ttend = (r1[k] - SH_THX(k) + 300.0f) * rdt * SH_PI2DX(k);
        float ttnpk = 0.0f + ttend;               // ttnp zeroed at :612
        // The wrapper's own epilogue (:253): rthblten = ttnp/pi3d.  The
        // divide does NOT round-trip -- (x*pi)/pi is not the float32
        // identity -- and the oracle's rthblten is the divided word, so the
        // kernel performs the same multiply-then-divide.
        SH_G(dtheta, k) = ttnpk / SH_PI2DX(k);
        if (k == kte)
            tflux_e[k] = ttend * SH_DZ8W(k);
        else
            tflux_e[k] = tflux_e[k + 1] + ttend * SH_DZ8W(k);
    }

    // ---- moisture, clouds, ice: assembly (:1217-1320), solve, recover ----
    for (int k = 0; k < wskp; ++k) {
        al[k] = ad[k] = au[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }
    ad[1] = 1.0f;
    r1[1] = SH_QVX(1) + qfxv * G / delp[1] * dt2;
    r2[1] = SH_QCX(1);                           // :1238-1245, ic = 2, 3
    r3[1] = SH_QIX(1);
    for (int k = 1; k <= kte - 1; ++k) {         // :1247-1253
        if (k >= kpbl)
            xkzq[k] = xkzh[k];
    }
    for (int k = 1; k <= kte - 1; ++k) {         // :1255-1293
        float dtodsd = dt2 / delp[k];
        float dtodsu = dt2 / delp[k + 1];
        float dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        float rdz = 1.0f / dza[k + 1];
        float tem1 = dsig * xkzq[k] * rdz;
        if (pblflg && k < kpbl) {
            float dsdzq = tem1 * (-(qfxpbl * zfacent[k] / xkzq[k]));
            r1[k] = r1[k] + dtodsd * dsdzq;
            r1[k + 1] = SH_QVX(k + 1) - dtodsu * dsdzq;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzq[k] = -(we * dza[kpbl] * exp(-entfac[k]));
            xkzq[k] = sqrt(xkzq[k] * xkzhl[k]);
            xkzq[k] = sh_max(xkzq[k], xkzminh);
            xkzq[k] = sh_min(xkzq[k], xkzmax);
            r1[k + 1] = SH_QVX(k + 1);
        } else {
            r1[k + 1] = SH_QVX(k + 1);
        }
        tem1 = dsig * xkzq[k] * rdz;             // recompute, as at :1275
        float dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2);
        al[k] = -(dtodsu * dsdz2);
        float zfacdx = 0.2f * hpbl / zq[k + 1];   // :1282-1284, h = hpbl
        float delxy_l = sqrt(dx * dy) * sh_max(zfacdx, 1.0f);
        float pq1_l = sh_pq(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pq1_l;
            al[k] = al[k] * pq1_l;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
    }
    for (int k = 1; k <= kte - 1; ++k) {         // :1295-1304
        r2[k + 1] = SH_QCX(k + 1);
        r3[k + 1] = SH_QIX(k + 1);
    }
    // :1324 -- one tridin_ysu call, ndiff = 3: all three components share
    // the same au/al/ad built from xkzq (one LU factorization).
    sh_tridi_factor(al, ad, au, fkk, lau, kte);
    sh_tridi_solve(al, fkk, lau, r1, kte);
    sh_tridi_solve(al, fkk, lau, r2, kte);
    sh_tridi_solve(al, fkk, lau, r3, kte);
    for (int k = kte; k >= 1; --k) {             // :1328-1340
        float qtend = (r1[k] - SH_QVX(k)) * rdt;
        SH_G(dqv, k) = 0.0f + qtend;             // qtnp zeroed at :613
        if (k == kte)
            qflux_e[k] = qtend * SH_DZ8W(k);
        else
            qflux_e[k] = qflux_e[k + 1] + qtend * SH_DZ8W(k);
        tvflux_e[k] = tflux_e[k] + qflux_e[k] * ep1 * SH_THX(k);
    }
    for (int k = 1; k <= kte; ++k) {             // :1342-1356
        if (pblflg && k < kpbl) {
            float hgame_c = c_1 * 0.2f * 2.5f * (G / thvx[k]) * wstar
                         / (0.25f * (q2x[k + 1] + q2x[k]));
            hgame_c = sh_min(hgame_c, gamcre);
            // the k == kte arm (:1347) is unreachable: k < kpbl <= kte
            hgame2d[k] = hgame_c * 0.5f * (tvflux_e[k] + tvflux_e[k + 1])
                       * hpbl;
            hgame2d[k] = sh_max(hgame2d[k], 0.0f);
        }
    }
    for (int k = kte; k >= 1; --k)               // :1358-1370, ic = 2
        SH_G(dqc, k) = 0.0f + (r2[k] - SH_QCX(k)) * rdt;
    for (int k = kte; k >= 1; --k)               // :1358-1370, ic = 3
        SH_G(dqi, k) = 0.0f + (r3[k] - SH_QIX(k)) * rdt;

    // ---- momentum: assembly (:1374-1436), solve, recover ----
    for (int k = 0; k < wskp; ++k) {
        al[k] = ad[k] = au[k] = 0.0f;
        r1[k] = r2[k] = 0.0f;
    }
    float s = wspd1 / wspdv;                      // no floor: WRF has none, and
                                                 // the fixture's 1e-10 wspd
                                                 // probe would measure one
    // :1387 -- both arms run this; the ctopo multiply is the arm-A identity
    // omitted per the header comment.
    ad[1] = 1.0f + ustv * ustv / wspd1 * rhox * G / delp[1] * dt2 * (s * s);
    r1[1] = SH_UX(1);
    r2[1] = SH_VX(1);
    for (int k = 1; k <= kte - 1; ++k) {
        float dtodsd = dt2 / delp[k];
        float dtodsu = dt2 / delp[k + 1];
        float dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        float rdz = 1.0f / dza[k + 1];
        float tem1 = dsig * xkzm[k] * rdz;
        if (pblflg && k < kpbl) {
            float dsdzu = tem1 * (-(hgamu / hpbl)
                                 - ufxpbl * zfacent[k] / xkzm[k]);
            float dsdzv = tem1 * (-(hgamv / hpbl)
                                 - vfxpbl * zfacent[k] / xkzm[k]);
            r1[k] = r1[k] + dtodsd * dsdzu;
            r1[k + 1] = SH_UX(k + 1) - dtodsu * dsdzu;
            r2[k] = r2[k] + dtodsd * dsdzv;
            r2[k + 1] = SH_VX(k + 1) - dtodsu * dsdzv;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzm[k] = prpbl * xkzh[k];
            xkzm[k] = sqrt(xkzm[k] * xkzml[k]);
            xkzm[k] = sh_max(xkzm[k], xkzminm);
            xkzm[k] = sh_min(xkzm[k], xkzmax);
            r1[k + 1] = SH_UX(k + 1);
            r2[k + 1] = SH_VX(k + 1);
        } else {
            r1[k + 1] = SH_UX(k + 1);
            r2[k + 1] = SH_VX(k + 1);
        }
        tem1 = dsig * xkzm[k] * rdz;             // recompute, as at :1419
        float dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2);
        al[k] = -(dtodsu * dsdz2);
        float zfacdx = 0.2f * hpbl / zq[k + 1];   // :1426-1428, h = hpbl
        float delxy_l = sqrt(dx * dy) * sh_max(zfacdx, 1.0f);
        float pu1_l = sh_pu(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pu1_l;
            al[k] = al[k] * pu1_l;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
    }
    sh_tridi_factor(al, ad, au, fkk, lau, kte);  // tridi1n :1450
    sh_tridi_solve(al, fkk, lau, r1, kte);
    sh_tridi_solve(al, fkk, lau, r2, kte);
    for (int k = kte; k >= 1; --k) {             // :1454-1463
        SH_G(du, k) = 0.0f + (r1[k] - SH_UX(k)) * rdt;
        SH_G(dv, k) = 0.0f + (r2[k] - SH_VX(k)) * rdt;
    }

    // ---- 10 m wind blend (:1471-1474): with ctopo2 = 1 the blend returns
    // its input bitwise (the arm-A identity, see the header); no outputs.

    // ---- SGS TKE diagnostics (:1478-1608) ----
    if (tke_diag == 1) {
        ShCol q2xk = SHWS_AT(wsb, 39, wskp);
        ShCol ptke1 = SHWS_AT(wsb, 40, wskp);
        ShCol akmk = SHWS_AT(wsb, 41, wskp);
        ShCol akhk = SHWS_AT(wsb, 42, wskp);
        ShCol mfk = SHWS_AT(wsb, 43, wskp);
        ShCol ufxpblk = SHWS_AT(wsb, 44, wskp);
        ShCol vfxpblk = SHWS_AT(wsb, 45, wskp);
        ShCol qfxpblk = SHWS_AT(wsb, 46, wskp);
        ShCol s2k = SHWS_AT(wsb, 47, wskp);
        ShCol rigk = SHWS_AT(wsb, 48, wskp);
        ShCol elk = SHWS_AT(wsb, 49, wskp);
        for (int k = 0; k < wskp; ++k) {
            q2xk[k] = akmk[k] = akhk[k] = mfk[k] = 0.0f;
            ufxpblk[k] = vfxpblk[k] = qfxpblk[k] = 0.0f;
            s2k[k] = rigk[k] = elk[k] = 0.0f;
            ptke1[k] = 1.0f;                     // :1509
        }
        for (int k = 1; k <= kte - 1; ++k) {     // :1530-1536
            if (pblflg && k <= kpbl) {           // k .le. kpbl, and za(k),
                float zfacdx = 0.2f * hpbl / za[k];   // not zq -- unlike the
                float delxy_l = sqrt(dx * dy)        // tri loops
                             * sh_max(zfacdx, 1.0f);
                ptke1[k + 1] = sh_ptke(delxy_l, hpbl);
            }
        }
        for (int k = 2; k <= kte; ++k) {         // :1546-1553
            akmk[k] = xkzm[k - 1];
            akhk[k] = xkzh[k - 1];
            mfk[k] = mf[k - 1] / xkzh[k - 1];
            // WRF reads zfacent at k-1 >= kpbl here, where it never assigned
            // it (stack garbage); every consumer is guarded k <= kpbl so the
            // garbage is dead.  This port's zfacent is 0 there instead.
            ufxpblk[k] = ufxpbl * zfacent[k - 1] / xkzm[k - 1];
            vfxpblk[k] = vfxpbl * zfacent[k - 1] / xkzm[k - 1];
            qfxpblk[k] = qfxpbl * zfacent[k - 1] / xkzq[k - 1];
        }
        for (int k = 1; k <= kte; ++k)
            q2xk[k] = q2x[k];
        if (pblflg && kpbl < kte) {
            // :1555-1559.  DOCUMENTED DIVERGENCE: WRF reads q2xk(kpbl+1)
            // with no guard, which is one past the end when kpbl == kte
            // (:1557).  The port computes efxpbl only for kpbl < kte -- the
            // defined behaviour -- and the registry warning covers it
            // (Phase D).  The fixture never reaches the undefined arm (case
            // 22 runs diag off).
            int kk = kpbl - 1;
            float dex = 0.25f * (q2xk[kk + 2] - q2xk[kk]);
            efxpbl = we * dex;
        }
        delxy = sqrt(dx * dy);                  // :1561
        sh_mixlen(u, v, theta, exner, qv, qc, st, col,
                  q2xk, zq, ustv, corfv, epshol, hpbl, kpbl, pblflg,
                  hgamu, hgamv, hgamq, mfk, ufxpblk, vfxpblk, qfxpblk, kte,
                  s2k, rigk, elk, wsb, wskp);
        sh_prodq2(dt, ustv, u, v, theta, st, col, thvx,
                  q2xk, elk, zq, akmk, akhk,
                  hgamu, hgamv, hgamq, delxy, hpbl, pblflg, kpbl,
                  mfk, ufxpblk, vfxpblk, qfxpblk, kte);
        sh_vdifq(dt, q2xk, zq, akhk, ptke1, hgame2d, hpbl, pblflg, kpbl,
                 efxpbl, kte, wsb, wskp);
        for (int k = 1; k <= kte; ++k) {         // :1601-1605
            q2x[k] = sh_max(q2xk[k], epsq2l);    // amax1: NaN propagates
            SH_G(tke_out, k) = 0.5f * q2x[k];
            if (k != 1)
                SH_G(el, k) = elk[k];            // el(kts) stays the :673 zero
        }
    }

    hpbl_out[col] = hpbl;
    kpbl_out[col] = kpbl;                        // the 1-based Fortran index
    wstar_out[col] = wstar;
    delta_out[col] = delta;
}
