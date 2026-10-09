#include <metal_stdlib>
#include "../generated/uwpbl_columns_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
using namespace metal;
#pragma clang fp contract(off)
// uwpbl_common.cuh -- shared device vocabulary of the UW moist-turbulence PBL
// port (WRF v4.7.1 bl_pbl_physics = 9, module_bl_camuwpbl_driver.F and the
// CAM modules it calls).  ArWen's own work, Apache-2.0.
//
// The CAM code computes in binary64 (real(r8)).  Three things make the port
// graded bit for bit against the gfortran -O0 oracle
// (tools/uwpbl_wrf471_oracle):
//
//  1. R8.  Every binary64 operation goes through a rounding-pinned intrinsic
//     (__dadd_rn, __dsub_rn, __dmul_rn, __ddiv_rn, __dsqrt_rn).  NVRTC compiles
//     this project with its default --fmad=true, so a plain `a*b + c` on
//     doubles WOULD be contracted into an FMA that the SSE2 gfortran build
//     never emits; the intrinsics are guaranteed never merged.  R8 wraps a
//     double so that ordinary infix code (`a*b + c`) calls them, which keeps
//     the transcription readable next to the Fortran.
//  2. Fortran's own semantics, measured on the oracle's toolchain (gfortran
//     15.2.0 -O0, glibc 2.43, x86-64; tools/uwpbl_wrf471_oracle/README.md):
//       MAX(a,b) = (a > b || isnan(a)) ? a : b   (ties return b: max(+0,-0)=-0)
//       MIN(a,b) = (a < b || isnan(a)) ? a : b
//       x**n, integer constant n: n = 2 is x*x inline, any other n is libgcc's
//         __powidf2 (square-and-multiply, 1/y for n < 0) -- uw_powi below.
//       x**y, real y (including 2._r8, 3._r8, 0.5_r8): glibc pow, never folded
//         at -O0 -- uw_pow below.
//       A literal without _r8 is a default (single-precision) REAL: its value
//         is the float32 rounding of the decimal, then widened exactly.
//         Write it as uw_r8(0x3eb0c6f7a0000000ul), never uw_r8(0x3eb0c6f7a0b5ed8dul).
//  3. The transcendentals are glibc's words, not CUDA's: exp/log/pow from
//     glibc_flt64.cuh.  cos/acos are correctly rounded (glibc's own binary64
//     cos/acos are LGPL IBM code and are not transcribed); where glibc does
//     not round correctly the port differs, and that residual is measured,
//     not absorbed.
//
// Column storage.  One thread owns one column.  Every per-column array lives
// in a global workspace laid out slot-major with the column index fastest
// (element k of a view at slot s is base[(s + k - 1) * stride + col]), so a
// warp touching the same k of 32 columns reads 32 consecutive doubles.
// Views are 1-based like the Fortran they transcribe: a(1) .. a(n).
#ifndef UWPBL_COMMON_CUH
#define UWPBL_COMMON_CUH

struct R8 { ulong bits; R8() : bits(0ul) {} R8(ulong b) : bits(b) {} };
static inline R8 uw_r8(ulong b) { return R8{b}; }
static inline R8 operator+(R8 a,R8 b){return uw_r8(soft64_add(a.bits,b.bits));}
static inline R8 operator-(R8 a,R8 b){return uw_r8(soft64_sub(a.bits,b.bits));}
static inline R8 operator*(R8 a,R8 b){return uw_r8(soft64_mul(a.bits,b.bits));}
static inline R8 operator/(R8 a,R8 b){return uw_r8(soft64_div(a.bits,b.bits));}
static inline R8 operator-(R8 a){return uw_r8(soft64_negate_bits(a.bits));}
static inline thread R8& operator+=(thread R8& a,R8 b){a=a+b;return a;}
static inline thread R8& operator-=(thread R8& a,R8 b){a=a-b;return a;}
static inline thread R8& operator*=(thread R8& a,R8 b){a=a*b;return a;}
static inline thread R8& operator/=(thread R8& a,R8 b){a=a/b;return a;}
static inline device R8& operator+=(device R8& a,R8 b){a=a+b;return a;}
static inline device R8& operator-=(device R8& a,R8 b){a=a-b;return a;}
static inline device R8& operator*=(device R8& a,R8 b){a=a*b;return a;}
static inline device R8& operator/=(device R8& a,R8 b){a=a/b;return a;}
static inline bool operator<(R8 a,R8 b){return soft64_less(a.bits,b.bits);}
static inline bool operator>(R8 a,R8 b){return soft64_greater(a.bits,b.bits);}
static inline bool operator<=(R8 a,R8 b){return soft64_less_equal(a.bits,b.bits);}
static inline bool operator>=(R8 a,R8 b){return soft64_greater_equal(a.bits,b.bits);}
static inline bool operator==(R8 a,R8 b){return soft64_equal(a.bits,b.bits);}
static inline bool operator!=(R8 a,R8 b){return !soft64_equal(a.bits,b.bits);}
static inline bool uw_isnan(R8 a){return soft64_is_nan(a.bits);}
static inline R8 uw_max(R8 a,R8 b){return (a>b||uw_isnan(a))?a:b;}
static inline R8 uw_min(R8 a,R8 b){return (a<b||uw_isnan(a))?a:b;}
static inline R8 uw_max(R8 a,R8 b,R8 c){return uw_max(uw_max(a,b),c);}
static inline R8 uw_min(R8 a,R8 b,R8 c){return uw_min(uw_min(a,b),c);}
static inline int uw_imax(int a,int b){return max(a,b);}
static inline int uw_imin(int a,int b){return min(a,b);}
static inline R8 uw_abs(R8 a){return uw_r8(soft64_abs_bits(a.bits));}
static inline R8 uw_sqrt(R8 a){return uw_r8(soft64_sqrt(a.bits));}
static inline R8 uw_sign(R8 a,R8 b){return uw_r8((a.bits&0x7ffffffffffffffful)|(b.bits&0x8000000000000000ul));}
static inline int uw_int(R8 a){return soft64_to_i32_rz(a.bits).value;}
static inline R8 uw_aint(R8 a){return uw_r8(soft64_from_i32(uw_int(a)));}
static inline R8 uw_real(int i){return uw_r8(soft64_from_i32(i));}
static inline R8 uw_widen(float x){return uw_r8(soft64_from_f32_bits(as_type<uint>(x)));}
static inline float uw_narrow(R8 a){return as_type<float>(soft64_to_f32_bits(a.bits));}
static inline R8 uw_powi(R8 x,int m){uint n=uint(m<0?-m:m);R8 y=(n&1u)?x:uw_r8(0x3ff0000000000000ul);while(n>>=1u){x=x*x;if(n&1u)y=y*x;}return m<0?uw_r8(0x3ff0000000000000ul)/y:y;}
static inline R8 uw_sq(R8 x){return x*x;}
static inline R8 uw_exp(R8 x){return uw_r8(soft64_plm_exp(x.bits));}
static inline R8 uw_log(R8 x){return uw_r8(soft64_plm_log(x.bits));}
static inline R8 uw_pow(R8 x,R8 y){return uw_r8(soft64_plm_pow(x.bits,y.bits));}
// Metal has no binary64 trig primitive; the closure's bounded cubic angles are
// evaluated in FP32 and widened back to the software binary64 format.
static inline R8 uw_cos(R8 x){return uw_widen(cos(uw_narrow(x)));}
static inline R8 uw_acos(R8 x){return uw_widen(acos(uw_narrow(x)));}
// ---- column views and the per-thread workspace ------------------------------
struct V {                        // real(r8) column array, 1-based
    device R8* p;                    // address of element 1 for this column
    int s;                        // stride between consecutive elements
    device R8& operator()(int k) const {
        return *reinterpret_cast<device R8*>(p + (long)(k - 1) * s);
    }
};
struct VI {                       // integer column array, 1-based
    device int* p;
    int s;
    device int& operator()(int k) const {
        return p[(long)(k - 1) * s];
    }
};
typedef VI VL;                    // logical arrays are stored as int 0/1

struct Ws {
    device R8* r8base;               // this column's slot 0 in the r8 pool
    device int* i4base;                  // this column's slot 0 in the i4 pool
    int stride;                   // number of columns sharing the pools
    int r8top, r8cap, i4top, i4cap;
    device atomic_int* err;              // first failing column's status
    V r8(int n) {
        if (r8top + n > r8cap) { atomic_store_explicit(err, 1, memory_order_relaxed); r8top = 0; }
        V v{r8base + (long)r8top * stride, stride};
        r8top += n;
        return v;
    }
    VI i4(int n) {
        if (i4top + n > i4cap) { atomic_store_explicit(err, 1, memory_order_relaxed); i4top = 0; }
        VI v{i4base + (long)i4top * stride, stride};
        i4top += n;
        return v;
    }
};
// Automatic arrays: `WsMark m(ws);` at the top of a routine releases every
// view the routine took when it returns.
struct WsMark {
    thread Ws& w;
    int r8, i4;
    explicit WsMark(thread Ws& ws) : w(ws), r8(ws.r8top), i4(ws.i4top) {}
    ~WsMark() { w.r8top = r8; w.i4top = i4; }
};

// ---- constants: the oracle's own run-time words (gpuwm/core/uwpbl_constants.py
// documents each; hex so no decimal conversion intervenes) --------------------
#define UW_CPAIR 0x408f651eb851eb85ul    // physconst cpair  = 1004.64
#define UW_GRAVIT 0x40239cc100e6afcdul    // physconst gravit = 9.80616
#define UW_RAIR 0x4071f0ad4eae9222ul    // physconst rair   = RGAS/MWDAIR
#define UW_ZVIR 0x3fe3730a75507baeul    // physconst zvir   = RWV/RDAIR - 1
#define UW_LATVAP 0x414314c400000000ul   // physconst latvap = 2.501e6
#define UW_LATICE 0x41145e1000000000ul   // physconst latice = 3.337e5
#define UW_KARMAN 0x3fd999999999999aul    // physconst karman = 0.4
#define UW_EPSILO 0x3fe3e72edbda50a2ul    // physconst epsilo = MWWV/MWDAIR
#define UW_RH2O 0x407cd81301343d8aul    // physconst rh2o   = RGAS/MWWV
#define UW_TMELT 0x4071126666666666ul    // physconst tmelt  = 273.15
#define UW_QMIN_Q 0x3d719799812dea11ul   // constituents qmin(1) = 1.E-12_r8
#define UW_B123 0x4009d339a3d1e83dul    // eddy_diff b123 = b1**(2/3), glibc pow

// wv_saturation's estbl(1..250), uploaded from uwpbl_constants.ESTBL by the
// launcher before the first launch (UW_ESTBL[i-1] is Fortran estbl(i)).


#endif  // UWPBL_COMMON_CUH

constant ulong UW_ESTBL[250] = {
    0x3f57051e74897e06ul,
    0x3f5c32944cd69aa3ul,
    0x3f613afe36d4dad8ul,
    0x3f6502ae1ffca58dul,
    0x3f6990311948a9a4ul,
    0x3f6f08fc6837f60cul,
    0x3f72cc853eee88d8ul,
    0x3f76b9f1be9e83a9ul,
    0x3f7b6ae5b788690dul,
    0x3f808146b7db5825ul,
    0x3f83d4f1c92c759aul,
    0x3f87c8480761b1c0ul,
    0x3f8c76f2739d5647ul,
    0x3f91007f1dee49aeul,
    0x3f9445bf45eea764ul,
    0x3f9820a197157c36ul,
    0x3f9ca9d32a7cc0f5ul,
    0x3fa0fedd06e56c2dul,
    0x3fa41e7ba3a9fd3eul,
    0x3fa7c67c52e32d33ul,
    0x3fac0c4e03755845ul,
    0x3fb0843979bb8a32ul,
    0x3fb36b734c8dd50ful,
    0x3fb6cbc79fc4f6cbul,
    0x3fbab75e40adbd34ul,
    0x3fbf42d9f189d925ul,
    0x3fc242d33a7fac5cul,
    0x3fc54d2789e97b9bul,
    0x3fc8cf6f33dc4ceful,
    0x3fccdaa422cb831ful,
    0x3fd0c0f7ac848260ul,
    0x3fd36d74b723fa16ul,
    0x3fd67ef0cbd0c90ful,
    0x3fda0315e583bbdeul,
    0x3fde093b4c9a5d56ul,
    0x3fe1514a58f63d52ul,
    0x3fe3f132e839ca42ul,
    0x3fe6ef1d79ea8d96ul,
    0x3fea57136420ccfbul,
    0x3fee3682ef418d62ul,
    0x3ff14e32157132baul,
    0x3ff3ccb088e5a027ul,
    0x3ff6a0009e7c3097ul,
    0x3ff9d26cb4a28a23ul,
    0x3ffd6f5f8ba978c2ul,
    0x4000c1c02454a544ul,
    0x40030e6869a255b3ul,
    0x4005a5675c32f995ul,
    0x40088f4b5f29e45dul,
    0x400bd585ac66de2ful,
    0x400f827f1723e04aul,
    0x4011d0d738750ca3ul,
    0x40141fd855f5c762ul,
    0x4016b5316b8e9295ul,
    0x4019987f08e214daul,
    0x401cd21acac53092ul,
    0x40203595bde4fc94ul,
    0x402236db30a0c6f8ul,
    0x40247258e3338132ul,
    0x4026ee0c6c12c652ul,
    0x4029b0807257f204ul,
    0x402cc0d80972f333ul,
    0x4030136d6994037ful,
    0x4031f580f9517815ul,
    0x40340b42ede9374eul,
    0x403659b68554283eul,
    0x4038e64dd7229753ul,
    0x403bb6f22a4162a7ul,
    0x403ed20cd15e4a50ul,
    0x40411f484a490673ul,
    0x40430201d797093bul,
    0x4045154534616563ul,
    0x40475d792855b8e5ul,
    0x4049df5f11d1cb24ul,
    0x404ca0193191308ful,
    0x404fa53153379af0ul,
    0x40517a4feb66ad98ul,
    0x40534a698faf931bul,
    0x4055465bb586d125ul,
    0x405771df8c7a335eul,
    0x4059d1ab0658eac2ul,
    0x405cb24d636462baul,
    0x405fd4baa11ce6deul,
    0x40619ea2333f3ec2ul,
    0x4063782f2e1518caul,
    0x4065794d736a811ful,
    0x4067a4539b81ddb9ul,
    0x4069fba34eeb974eul,
    0x406c81a68ada5a0aul,
    0x406f38cc764bd602ul,
    0x407111c2df348588ul,
    0x4072a22037fd1898ul,
    0x40744eb1a19fdaf7ul,
    0x407618a42779ddb9ul,
    0x4078011b8b5300aaul,
    0x407a092ef407ae19ul,
    0x407c31e53d30f294ul,
    0x407e7c30e1000060ul,
    0x40807475ba999ae0ul,
    0x4081bc6859dd2e97ul,
    0x4083163cf541bcadul,
    0x408484eef36eea99ul,
    0x40860bcdab8d3cebul,
    0x4087ac3541ed6732ul,
    0x40896791c30785beul,
    0x408b3f5fabea86c1ul,
    0x408d352c7510a97eul,
    0x408f4a971f8fc8c6ul,
    0x4090c0a8624dba77ul,
    0x4091ed8e93a71cb9ul,
    0x40932ce9a4575f00ul,
    0x40947faefef268b1ul,
    0x4095e6de4448e1bcul,
    0x40976381989f6d70ul,
    0x4098f6adf1ea0971ul,
    0x409aa18367045b6aul,
    0x409c652d7fdf6fdaul,
    0x409e42e3869d240eul,
    0x40a01df46cc8975dul,
    0x40a128c69f0fbdf4ul,
    0x40a242969a37a666ul,
    0x40a36c1925f528c6ul,
    0x40a4a609bc3f5ba0ul,
    0x40a5f12ab44d7737ul,
    0x40a74e456decda9dul,
    0x40a8be2a7d2a6382ul,
    0x40aa41b1d64a2d51ul,
    0x40abd9bafa08bac0ul,
    0x40ad872d22207716ul,
    0x40af4af76e0e6997ul,
    0x40b09308880872d0ul,
    0x40b18cbcbd2d7a12ul,
    0x40b2931c463b9118ul,
    0x40b3a6af606a2c91ul,
    0x40b4c802ad3bac4cul,
    0x40b5f7a74965b107ul,
    0x40b73632e3c650a4ul,
    0x40b8843fd4636b2ful,
    0x40b9e26d337162feul,
    0x40bb515ef05e77e2ul,
    0x40bcd1bde8e006c0ul,
    0x40be6437fffef143ul,
    0x40c004c01a90381ful,
    0x40c0e1275d844d0dul,
    0x40c1c7b08769f3a2ul,
    0x40c2b8bd07728271ul,
    0x40c3b4b108e225f3ul,
    0x40c4bbf37e771ae2ul,
    0x40c5ceee2dc71a73ul,
    0x40c6ee0dba91abdbul,
    0x40c819c1b2061606ul,
    0x40c9527c95fbaea6ul,
    0x40ca98b3e81b4b04ul,
    0x40cbece034f899eful,
    0x40cd4f7d1f1a3650ul,
    0x40cec10969ef4688ul,
    0x40d021038258c130ul,
    0x40d0e97d8a994236ul,
    0x40d1ba370149a585ul,
    0x40d29375bff26eb8ul,
    0x40d375814b5884d7ul,
    0x40d460a2d8b51e11ul,
    0x40d5552552e0bbfful,
    0x40d653555f70c0f2ul,
    0x40d75b8163c729e3ul,
    0x40d86df98a13fe83ul,
    0x40d98b0fc6480c59ul,
    0x40dab317daf88781ul,
    0x40dbe6675e332eabul,
    0x40dd2555be429766ul,
    0x40de703c466249adul,
    0x40dfc77623625944ul,
    0x40e095b0341d16ceul,
    0x40e14e2d09451948ul,
    0x40e20d6207861557ul,
    0x40e2d3809ef8f491ul,
    0x40e3a0bb3b991b2dul,
    0x40e475454761d728ul,
    0x40e551532c62b45ful,
    0x40e6351a56caa10bul,
    0x40e720d136e9d162ul,
    0x40e814af432a517cul,
    0x40e910ecf9ff3b04ul,
    0x40ea15c3e3ca82ddul,
    0x40eb236e94b9589dul,
    0x40ec3a28ae971276ul,
    0x40ed5a2ee296a3ebul,
    0x40ee83bef312a065ul,
    0x40efb717b543cb19ul,
    0x40f07a3c89771e68ul,
    0x40f11e1206029264ul,
    0x40f1c72d5c229952ul,
    0x40f275b02462d9e4ul,
    0x40f329bc84c03bb0ul,
    0x40f3e3753168443dul,
    0x40f4a2fd6d73ea1cul,
    0x40f568790b9dea19ul,
    0x40f6340c6ef4aaa2ul,
    0x40f705dc8b87bb4aul,
    0x40f7de0ee7110054ul,
    0x40f8bcc999999984ul,
    0x40f9a2334e1a960aul,
    0x40fa8e73431987acul,
    0x40fb81b14b410954ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul,
    0xc0f869f000000000ul
};
// WRF module_cam_wv_saturation.F:82-98,235-298,512-639,757-774.
#ifndef UW_WVSAT_CUH
#define UW_WVSAT_CUH
static inline R8 uw_estblf(R8 td) {
 R8 e=uw_max(uw_min(td,uw_r8(0x4077728f5c28f5c3ul)),uw_r8(0x4065a51eb851eb85ul));
 int i=uw_int(e-uw_r8(0x4065a51eb851eb85ul))+1; R8 ai=uw_aint(e-uw_r8(0x4065a51eb851eb85ul));
 return (uw_r8(0x4065a51eb851eb85ul)+ai-e+uw_r8(0x3ff0000000000000ul))*R8(UW_ESTBL[i-1])-(uw_r8(0x4065a51eb851eb85ul)+ai-e)*R8(UW_ESTBL[i]);
}
static inline void uw_aqsat(V t,V p,V es,V qs,int kstart,int kend) {
 R8 omeps=uw_r8(0x3ff0000000000000ul)-R8(UW_EPSILO);
 for(int k=kstart;k<=kend;++k) {
  es(k)=uw_estblf(t(k)); qs(k)=R8(UW_EPSILO)*es(k)/(p(k)-omeps*es(k));
  qs(k)=uw_min(uw_r8(0x3ff0000000000000ul),qs(k));
  if(qs(k)<uw_r8(0x0000000000000000ul)) {qs(k)=uw_r8(0x3ff0000000000000ul);es(k)=p(k);}
 }
}
static inline int uw_fqsatd(R8 t,R8 p,thread R8& es,thread R8& qs,thread R8& gam) {
 R8 omeps=uw_r8(0x3ff0000000000000ul)-R8(UW_EPSILO);
 es=uw_estblf(t);qs=R8(UW_EPSILO)*es/(p-omeps*es);qs=uw_min(uw_r8(0x3ff0000000000000ul),qs);
 if(qs<uw_r8(0x0000000000000000ul)) {qs=uw_r8(0x3ff0000000000000ul);es=p;}
 R8 trinv=uw_r8(0x3ff0000000000000ul)/uw_r8(0x4034000000000000ul),tc=t-R8(UW_TMELT);
 bool lflg=tc>=uw_r8(0xc034000000000000ul)&&tc<uw_r8(0x0000000000000000ul);
 R8 weight=uw_min(-(tc*trinv),uw_r8(0x3ff0000000000000ul));
 R8 hlatsb=R8(UW_LATVAP)+weight*R8(UW_LATICE);
 R8 hlatvp=R8(UW_LATVAP)-uw_r8(0x40a2820000000000ul)*tc;
 R8 hltalt=t<R8(UW_TMELT)?hlatsb:hlatvp;
 R8 tterm=lflg?uw_r8(0x3fe0249d680eb712ul)+tc*(uw_r8(0xc015e43bd0c77471ul)+tc*(uw_r8(0xbfd784a8afdde5e6ul)+tc*(uw_r8(0xbf82596de59b3a85ul)+tc*uw_r8(0xbf14656e0e30f6fbul)))):uw_r8(0x0000000000000000ul);
 R8 desdt=hltalt*es/(R8(UW_RH2O)*t*t)+tterm*trinv;
 gam=hltalt*qs*p*desdt/(R8(UW_CPAIR)*es*(p-omeps*es));
 if(qs==uw_r8(0x3ff0000000000000ul))gam=uw_r8(0x0000000000000000ul);return 1;
}
#endif

// WRF diffusion_solver: non-molecular WRF_PORT path. Views are 1-based.
#ifndef UW_VDIFF_CUH
#define UW_VDIFF_CUH
struct UwVdiffFields { bool u,v,s; bool q[5]; };
static inline void uw_vd_lu_decomp(int pver,R8 ksrf,V kv,V tmpi,V rpdel,R8 ztodt,R8 cc_top,V ca,V cc,V dnom,V ze,int ntop,int nbot) {
 for(int k=nbot-1;k>=ntop;--k) {ca(k)=kv(k+1)*tmpi(k+1)*rpdel(k);cc(k+1)=kv(k+1)*tmpi(k+1)*rpdel(k+1);}
 ca(nbot)=uw_r8(0x0000000000000000ul);
 dnom(nbot)=uw_r8(0x3ff0000000000000ul)/(uw_r8(0x3ff0000000000000ul)+cc(nbot)+ksrf*ztodt*R8(UW_GRAVIT)*rpdel(nbot));
 ze(nbot)=cc(nbot)*dnom(nbot);
 for(int k=nbot-1;k>=ntop+1;--k) {dnom(k)=uw_r8(0x3ff0000000000000ul)/(uw_r8(0x3ff0000000000000ul)+ca(k)+cc(k)-ca(k)*ze(k+1));ze(k)=cc(k)*dnom(k);}
 dnom(ntop)=uw_r8(0x3ff0000000000000ul)/(uw_r8(0x3ff0000000000000ul)+ca(ntop)+cc_top-ca(ntop)*ze(ntop+1));
}
static inline void uw_vd_lu_solve(int pver,V q,V ca,V ze,V dnom,int ntop,int nbot,R8 cd_top) {
 // zf is stored in q: each original value is consumed before being replaced.
 // This preserves each arithmetic operation of the automatic zf array.
 q(nbot)=q(nbot)*dnom(nbot);
 for(int k=nbot-1;k>=ntop+1;--k)q(k)=(q(k)+ca(k)*q(k+1))*dnom(k);
 q(ntop)=(q(ntop)+cd_top+ca(ntop)*q(ntop+1))*dnom(ntop);
 for(int k=ntop+1;k<=nbot;++k)q(k)=q(k)+ze(k)*q(k-1);
}
static inline void uw_compute_vdiff(int pver,int ncnst,V pmid,V pint,V rpdel,V t,R8 ztodt,R8 taux,R8 tauy,R8 shflx,thread const R8* cflx,int ntop,int nbot,V kvh,V kvm,V kvq,V cgs,V cgh,V zi,R8 ksrftms,thread const R8* qmincg,thread const UwVdiffFields& fl,V u,V v,thread V* q,V dse,thread R8& tautmsx,thread R8& tautmsy,V dtk,thread R8& topflx,thread int& errflag,thread R8& tauresx,thread R8& tauresy,int itaures,thread Ws& ws) {
 errflag=0;bool momentum=fl.u||fl.v;
 if(momentum&&!fl.s) {errflag=1;return;}
 WsMark mark(ws);
 V rhoi=ws.r8(pver+1),tmpi2=ws.r8(pver+1),tmpi1=ws.r8(pver+1),du=ws.r8(pver+1),dv=ws.r8(pver+1);
 V ca=ws.r8(pver),cc=ws.r8(pver),dnom=ws.r8(pver),ze=ws.r8(pver),qtm=ws.r8(pver);
 rhoi(1)=pint(1)/(R8(UW_RAIR)*t(1));
 for(int k=2;k<=pver;++k) {R8 tint=uw_r8(0x3fe0000000000000ul)*(t(k)+t(k-1));rhoi(k)=pint(k)/(R8(UW_RAIR)*tint);tmpi2(k)=ztodt*uw_sq(R8(UW_GRAVIT)*rhoi(k))/(pmid(k)-pmid(k-1));}
 rhoi(pver+1)=pint(pver+1)/(R8(UW_RAIR)*t(pver));
 R8 rrho=R8(UW_RAIR)*t(pver)/pmid(pver),tmp1=ztodt*R8(UW_GRAVIT)*rpdel(pver);
 if(momentum) {
  du(1)=uw_r8(0x0000000000000000ul);dv(1)=uw_r8(0x0000000000000000ul);du(pver+1)=-u(pver);dv(pver+1)=-v(pver);
  for(int k=2;k<=pver;++k){du(k)=u(k)-u(k-1);dv(k)=v(k)-v(k-1);}
  R8 speed=uw_max(uw_sqrt(uw_pow(u(pver),uw_r8(0x4000000000000000ul))+uw_pow(v(pver),uw_r8(0x4000000000000000ul))),uw_r8(0x3ff0000000000000ul));
  R8 tau=uw_sqrt(uw_pow(taux,uw_r8(0x4000000000000000ul))+uw_pow(tauy,uw_r8(0x4000000000000000ul)));
  R8 ksrf=uw_max(tau/speed,uw_r8(0x3f1a36e2eb1c432dul))+ksrftms;
  R8 usum=uw_r8(0x0000000000000000ul),vsum=uw_r8(0x0000000000000000ul);
  for(int k=1;k<=pver;++k){usum=usum+(uw_r8(0x3ff0000000000000ul)/R8(UW_GRAVIT))*u(k)/rpdel(k);vsum=vsum+(uw_r8(0x3ff0000000000000ul)/R8(UW_GRAVIT))*v(k)/rpdel(k);}
  R8 ramda=ztodt/uw_r8(0x40bc200000000000ul);
  u(pver)=u(pver)+tmp1*tauresx*ramda;v(pver)=v(pver)+tmp1*tauresy*ramda;
  uw_vd_lu_decomp(pver,ksrf,kvm,tmpi2,rpdel,ztodt,uw_r8(0x0000000000000000ul),ca,cc,dnom,ze,ntop,nbot);
  uw_vd_lu_solve(pver,u,ca,ze,dnom,ntop,nbot,uw_r8(0x0000000000000000ul));uw_vd_lu_solve(pver,v,ca,ze,dnom,ntop,nbot,uw_r8(0x0000000000000000ul));
  tautmsx=-(ksrftms*u(pver));tautmsy=-(ksrftms*v(pver));
  R8 usout=uw_r8(0x0000000000000000ul),vsout=uw_r8(0x0000000000000000ul);
  for(int k=1;k<=pver;++k){usout=usout+(uw_r8(0x3ff0000000000000ul)/R8(UW_GRAVIT))*u(k)/rpdel(k);vsout=vsout+(uw_r8(0x3ff0000000000000ul)/R8(UW_GRAVIT))*v(k)/rpdel(k);}
  R8 tx=(usout-usum)/ztodt,ty=(vsout-vsum)/ztodt;
  if(itaures==1){tauresx=taux+tautmsx+tauresx-tx;tauresy=tauy+tautmsy+tauresy-ty;}
  tmpi1(1)=uw_r8(0x0000000000000000ul);
  tmpi1(pver+1)=uw_r8(0x3fe0000000000000ul)*ztodt*R8(UW_GRAVIT)*((-u(pver)+du(pver+1))*tx+(-v(pver)+dv(pver+1))*ty);
  for(int k=2;k<=pver;++k){R8 dout_u=u(k)-u(k-1),dout_v=v(k)-v(k-1);tmpi1(k)=uw_r8(0x3fd0000000000000ul)*tmpi2(k)*kvm(k)*(uw_sq(dout_u)+uw_sq(dout_v)+dout_u*du(k)+dout_v*dv(k));}
  for(int k=1;k<=pver;++k){dtk(k)=(tmpi1(k+1)+tmpi1(k))*rpdel(k);dse(k)=dse(k)+dtk(k);}
 }
 if(fl.s){
  for(int k=1;k<=pver;++k)dse(k)=dse(k)+ztodt*rpdel(k)*R8(UW_GRAVIT)*(rhoi(k+1)*kvh(k+1)*cgh(k+1)-rhoi(k)*kvh(k)*cgh(k));
  dse(pver)=dse(pver)+tmp1*shflx;
  uw_vd_lu_decomp(pver,uw_r8(0x0000000000000000ul),kvh,tmpi2,rpdel,ztodt,uw_r8(0x0000000000000000ul),ca,cc,dnom,ze,ntop,nbot);
  uw_vd_lu_solve(pver,dse,ca,ze,dnom,ntop,nbot,uw_r8(0x0000000000000000ul));
 }
 bool need_decomp=true;
 for(int m=1;m<=ncnst;++m)if(fl.q[m-1]){
  for(int k=1;k<=pver;++k)qtm(k)=q[m-1](k);
  for(int k=1;k<=pver;++k)q[m-1](k)=q[m-1](k)+ztodt*rpdel(k)*R8(UW_GRAVIT)*(cflx[m-1]*rrho)*(rhoi(k+1)*kvh(k+1)*cgs(k+1)-rhoi(k)*kvh(k)*cgs(k));
  bool lqtst=true;for(int k=1;k<=pver;++k)if(!(q[m-1](k)>=qmincg[m-1]))lqtst=false;
  for(int k=1;k<=pver;++k)if(!lqtst)q[m-1](k)=qtm(k);
  q[m-1](pver)=q[m-1](pver)+tmp1*cflx[m-1];
  if(need_decomp){uw_vd_lu_decomp(pver,uw_r8(0x0000000000000000ul),kvq,tmpi2,rpdel,ztodt,uw_r8(0x0000000000000000ul),ca,cc,dnom,ze,ntop,nbot);need_decomp=false;}
  uw_vd_lu_solve(pver,q[m-1],ca,ze,dnom,ntop,nbot,uw_r8(0x0000000000000000ul));
 }
}
#endif

// Generated literal transcription. F: comments refer to module_cam_bl_eddy_diff.F.
// Constant expression subtrees are folded in source grouping to exact hex words.
#ifndef UWPBL_ZISOCL_CUH
#define UWPBL_ZISOCL_CUH
#define uw_cpair R8(UW_CPAIR)
#define uw_rair R8(UW_RAIR)
#define uw_zvir R8(UW_ZVIR)
#define uw_latvap R8(UW_LATVAP)
#define uw_latice R8(UW_LATICE)
#define uw_g R8(UW_GRAVIT)
#define uw_vk R8(UW_KARMAN)
#define uw_b123 R8(UW_B123)
#define uw_latsub (uw_latvap + uw_latice)
#define uw_ccon ((uw_r8(0x4021000000000000ul) * uw_r8(0x3fb999999999999aul)) * uw_vk)

static inline R8 uw_compute_cubic(R8 a, R8 b, R8 c) {
    R8 qq;
    R8 rr;
    R8 dd;
    R8 theta;
    R8 aa;
    R8 bb;
    R8 x1;
    R8 x2;
    R8 x3;
    qq = ((uw_sq(a) - (uw_r8(0x4008000000000000ul) * b)) / uw_r8(0x4022000000000000ul)); // F:4038 qq = (a**2-3._r8*b)/9._r8
    rr = ((((uw_r8(0x4000000000000000ul) * uw_powi(a, 3)) - ((uw_r8(0x4022000000000000ul) * a) * b)) + (uw_r8(0x403b000000000000ul) * c)) / uw_r8(0x404b000000000000ul)); // F:4039 rr = (2._r8*a**3 - 9._r8*a*b + 27._r8*c)/54._r8
    dd = (uw_sq(rr) - uw_powi(qq, 3)); // F:4041 dd = rr**2 - qq**3
    if ((dd <= uw_r8(0x0000000000000000ul))) { // F:4042 if( dd .le. 0._r8 ) then
        theta = uw_acos((rr / uw_pow(qq, uw_r8(0x3ff8000000000000ul)))); // F:4043 theta = acos(rr/qq**(3._r8/2._r8))
        x1 = ((-((uw_r8(0x4000000000000000ul) * uw_sqrt(qq)) * uw_cos((theta / uw_r8(0x4008000000000000ul))))) - (a / uw_r8(0x4008000000000000ul))); // F:4044 x1 = -2._r8*sqrt(qq)*cos(theta/3._r8) - a/3._r8
        x2 = ((-((uw_r8(0x4000000000000000ul) * uw_sqrt(qq)) * uw_cos(((theta + uw_r8(0x401921fb00000000ul)) / uw_r8(0x4008000000000000ul))))) - (a / uw_r8(0x4008000000000000ul))); // F:4045 x2 = -2._r8*sqrt(qq)*cos((theta+2._r8*3.141592)/3._r8) - a/3._r8
        x3 = ((-((uw_r8(0x4000000000000000ul) * uw_sqrt(qq)) * uw_cos(((theta - uw_r8(0x401921fb00000000ul)) / uw_r8(0x4008000000000000ul))))) - (a / uw_r8(0x4008000000000000ul))); // F:4046 x3 = -2._r8*sqrt(qq)*cos((theta-2._r8*3.141592)/3._r8) - a/3._r8
        return uw_max(uw_max(uw_max(x1, x2), x3), uw_r8(0x3f847ae147ae147bul)); // F:4047 compute_cubic = max(max(max(x1,x2),x3),xmin)
    } else {
        if ((rr >= uw_r8(0x0000000000000000ul))) { // F:4050 if( rr .ge. 0._r8 ) then
            aa = (-uw_pow((uw_sqrt((uw_sq(rr) - uw_powi(qq, 3))) + rr), uw_r8(0x3fd5555555555555ul))); // F:4051 aa = -(sqrt(rr**2-qq**3)+rr)**(1._r8/3._r8)
        } else {
            aa = uw_pow((uw_sqrt((uw_sq(rr) - uw_powi(qq, 3))) - rr), uw_r8(0x3fd5555555555555ul)); // F:4053 aa =  (sqrt(rr**2-qq**3)-rr)**(1._r8/3._r8)
        }
        if ((aa == uw_r8(0x0000000000000000ul))) { // F:4055 if( aa .eq. 0._r8 ) then
            bb = uw_r8(0x0000000000000000ul); // F:4056 bb = 0._r8
        } else {
            bb = (qq / aa); // F:4058 bb = qq/aa
        }
        return uw_max(((aa + bb) - (a / uw_r8(0x4008000000000000ul))), uw_r8(0x3f847ae147ae147bul)); // F:4060 compute_cubic = max((aa+bb)-a/3._r8,xmin)
    }
}
static inline void uw_exacol(int pver, V ri, R8 bflxs, R8 minpblh, V zi, VI ktop, VI kbase, thread int& ncvfin, thread Ws& ws) {
    const int ncvmax = pver;
    const int nbot_turb = pver;
    WsMark uw_mark(ws);
    int k;
    int ncv;
    R8 rimaxentr;
    V riex = ws.r8((pver + 1));
    ncvfin = 0; // F:3114 ncvfin(i) = 0
    for (ncv = 1; ncv <= ncvmax; ncv += 1) { // F:3115 do ncv = 1, ncvmax
        ktop(ncv) = 0; // F:3116 ktop(i,ncv)  = 0
        kbase(ncv) = 0; // F:3117 kbase(i,ncv) = 0
    }
    rimaxentr = uw_r8(0x0000000000000000ul); // F:3125 rimaxentr = 0._r8
    for (int uw_k = 2; uw_k <= pver; ++uw_k) { // F:3129 riex(2:pver) = ri(i,2:pver)
        riex(uw_k) = ri(uw_k);
    }
    riex((pver + 1)) = (rimaxentr - bflxs); // F:3134 riex(pver+1) = rimaxentr - bflxs(i)
    ncv = 0; // F:3136 ncv = 0
    k = (pver + 1); // F:3137 k   = pver + 1
    while ((k > 2)) { // F:3139 do while ( k .gt. ntop_turb + 1 )
        if ((riex(k) < rimaxentr)) { // F:3144 if( riex(k) .lt. rimaxentr ) then
            ncv = (ncv + 1); // F:3148 ncv = ncv + 1
            kbase(ncv) = uw_imin((k + 1), (pver + 1)); // F:3153 kbase(i,ncv) = min(k+1,pver+1)
            while (((riex(k) < rimaxentr) && (k > 2))) { // F:3157 do while( riex(k) .lt. rimaxentr .and. k .gt. ntop_turb + 1 )
                k = (k - 1); // F:3158 k = k - 1
            }
            ktop(ncv) = k; // F:3164 ktop(i,ncv) = k
        } else {
            k = (k - 1); // F:3170 k = k - 1
        }
    }
    ncvfin = ncv; // F:3176 ncvfin(i) = ncv
    return; // F:3180 return
}
static inline void uw_zisocl(int pver, V z, V zi, V n2, V s2, V bprod, V sprod, R8 bflxs, R8 tkes, thread int& ncvfin, VI kbase, VI ktop, VL belongcv, V ricl, V ghcl, V shcl, V smcl, V lbrk, V wbrk, V ebrk, thread bool& extend, thread bool& extend_up, thread bool& extend_dn, thread Ws& ws) {
    const int ncvmax = pver;
    const int nbot_turb = pver;
    WsMark uw_mark(ws);
    bool bottom;
    int ncv;
    int incv;
    int k;
    int kb;
    int kt;
    int ncvinit;
    int cntu;
    int cntd;
    int kbinc;
    int ktinc;
    R8 wint;
    R8 dwinc;
    R8 dw_surf;
    R8 dzinc;
    R8 gh;
    R8 sh;
    R8 sm;
    R8 gh_surf;
    R8 sh_surf;
    R8 sm_surf;
    R8 l2n2;
    R8 l2s2;
    R8 dl2n2;
    R8 dl2s2;
    R8 dl2n2_surf;
    R8 dl2s2_surf;
    R8 lint;
    R8 dlint;
    R8 dlint_surf;
    R8 lbulk;
    R8 lz;
    R8 ricll;
    R8 trma;
    R8 trmb;
    R8 trmc;
    R8 det;
    R8 zbot;
    R8 l2rat;
    R8 gg;
    R8 tunlramp;
    for (k = 1; k <= ncvmax; k += 1) { // F:3308 do k = 1, ncvmax
        ricl(k) = uw_r8(0x0000000000000000ul); // F:3309 ricl(i,k) = 0._r8
        ghcl(k) = uw_r8(0x0000000000000000ul); // F:3310 ghcl(i,k) = 0._r8
        shcl(k) = uw_r8(0x0000000000000000ul); // F:3311 shcl(i,k) = 0._r8
        smcl(k) = uw_r8(0x0000000000000000ul); // F:3312 smcl(i,k) = 0._r8
        lbrk(k) = uw_r8(0x0000000000000000ul); // F:3313 lbrk(i,k) = 0._r8
        wbrk(k) = uw_r8(0x0000000000000000ul); // F:3314 wbrk(i,k) = 0._r8
        ebrk(k) = uw_r8(0x0000000000000000ul); // F:3315 ebrk(i,k) = 0._r8
    }
    extend = false; // F:3317 extend    = .false.
    extend_up = false; // F:3318 extend_up = .false.
    extend_dn = false; // F:3319 extend_dn = .false.
    ncv = 1; // F:3325 ncv = 1
    while ((ncv <= ncvfin)) { // F:3327 do while( ncv .le. ncvfin(i) )
        ncvinit = ncv; // F:3329 ncvinit = ncv
        cntu = 0; // F:3330 cntu    = 0
        cntd = 0; // F:3331 cntd    = 0
        kb = kbase(ncv); // F:3332 kb      = kbase(i,ncv)
        kt = ktop(ncv); // F:3333 kt      = ktop(i,ncv)
        lbulk = (zi(kt) - zi(kb)); // F:3359 lbulk      = zi(i,kt) - zi(i,kb)
        dlint_surf = uw_r8(0x0000000000000000ul); // F:3360 dlint_surf = 0._r8
        dl2n2_surf = uw_r8(0x0000000000000000ul); // F:3361 dl2n2_surf = 0._r8
        dl2s2_surf = uw_r8(0x0000000000000000ul); // F:3362 dl2s2_surf = 0._r8
        dw_surf = uw_r8(0x0000000000000000ul); // F:3363 dw_surf    = 0._r8
        if ((kb == (pver + 1))) { // F:3364 if( kb .eq. pver+1 ) then
            if ((bflxs > uw_r8(0x0000000000000000ul))) { // F:3366 if( bflxs(i) .gt. 0._r8 ) then
                gg = ((((uw_r8(0x3fe0000000000000ul) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, uw_r8(0x3ff8000000000000ul))); // F:3373 gg    = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                gh = (gg / (uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))); // F:3374 gh    = gg/(alph5-gg*alph3)
                gh = uw_min(uw_max(gh, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3376 gh    = min(max(gh,-3.5334_r8),0.0233_r8)
                sh = (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))); // F:3377 sh    = alph5/(1._r8+alph3*gh)
                sm = (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh))); // F:3378 sm    = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
                ricll = uw_min((-((sm / sh) * (bprod((pver + 1)) / sprod((pver + 1))))), uw_r8(0x3fc851eb851eb852ul)); // F:3379 ricll = min(-(sm/sh)*(bprod(i,pver+1)/sprod(i,pver+1)),ricrit)
                dlint_surf = z(pver); // F:3387 dlint_surf = z(i,pver)
                dl2n2_surf = (-(((uw_vk * uw_sq(z(pver))) * bprod((pver + 1))) / (sh * uw_sqrt(tkes)))); // F:3388 dl2n2_surf = -vk*(z(i,pver)**2)*bprod(i,pver+1)/(sh*sqrt(tkes(i)))
                dl2s2_surf = (((uw_vk * uw_sq(z(pver))) * sprod((pver + 1))) / (sm * uw_sqrt(tkes))); // F:3389 dl2s2_surf =  vk*(z(i,pver)**2)*sprod(i,pver+1)/(sm*sqrt(tkes(i)))
                dw_surf = ((tkes / uw_r8(0x4017333333333333ul)) * z(pver)); // F:3390 dw_surf    = (tkes(i)/b1)*z(i,pver)
            } else {
                lbulk = (zi(kt) - z(pver)); // F:3396 lbulk = zi(i,kt) - z(i,pver)
            }
        }
        lint = dlint_surf; // F:3406 lint = dlint_surf
        l2n2 = dl2n2_surf; // F:3407 l2n2 = dl2n2_surf
        l2s2 = dl2s2_surf; // F:3408 l2s2 = dl2s2_surf
        wint = dw_surf; // F:3409 wint = dw_surf
        l2n2 = uw_r8(0x0000000000000000ul); // F:3411 l2n2 = 0._r8
        l2s2 = uw_r8(0x0000000000000000ul); // F:3412 l2s2 = 0._r8
        if ((kt < (kb - 1))) { // F:3421 if( kt .lt. kb - 1 ) then
            for (k = (kb - 1); k >= (kt + 1); k += -1) { // F:3423 do k = kb - 1, kt + 1, -1
                tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3426 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                lz = uw_pow((uw_pow((uw_vk * zi(k)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3434 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                dzinc = (z((k - 1)) - z(k)); // F:3439 dzinc = z(i,k-1) - z(i,k)
                l2n2 = (l2n2 + (((lz * lz) * n2(k)) * dzinc)); // F:3440 l2n2  = l2n2 + lz*lz*n2(i,k)*dzinc
                l2s2 = (l2s2 + (((lz * lz) * s2(k)) * dzinc)); // F:3441 l2s2  = l2s2 + lz*lz*s2(i,k)*dzinc
                lint = (lint + dzinc); // F:3442 lint  = lint + dzinc
            }
            ricll = uw_min((l2n2 / uw_max(l2s2, uw_r8(0x3d719799812dea11ul))), uw_r8(0x3fc851eb851eb852ul)); // F:3451 ricll = min(l2n2/max(l2s2,ntzero),ricrit)
            trma = ((uw_r8(0x406a8f03ff93f46aul) * ricll) + (uw_r8(0x4027333333333333ul) * (uw_r8(0xc01174bc6a7ef9dbul) - (uw_r8(0xc0111f3168d8b188ul) * ricll)))); // F:3452 trma  = alph3*alph4*ricll+2._r8*b1*(alph2-alph4*alph5*ricll)
            trmb = ((ricll * uw_r8(0xc04466dc5d638866ul)) + (uw_r8(0x4027333333333333ul) * ((-(uw_r8(0x3fe65aee631f8a09ul) * ricll)) + uw_r8(0x3fe1cc63f141205cul)))); // F:3453 trmb  = ricll*(alph3+alph4)+2._r8*b1*(-alph5*ricll+alph1)
            trmc = ricll; // F:3454 trmc  = ricll
            det = uw_max(((trmb * trmb) - ((uw_r8(0x4010000000000000ul) * trma) * trmc)), uw_r8(0x0000000000000000ul)); // F:3455 det   = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = ((((-trmb) + uw_sqrt(det)) / uw_r8(0x4000000000000000ul)) / trma); // F:3456 gh    = (-trmb + sqrt(det))/2._r8/trma
            gh = uw_min(uw_max(gh, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3458 gh    = min(max(gh,-3.5334_r8),0.0233_r8)
            sh = (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))); // F:3459 sh    = alph5/(1._r8+alph3*gh)
            sm = (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh))); // F:3460 sm    = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
            wint = ((wint - (sh * l2n2)) + (sm * l2s2)); // F:3461 wint  = wint - sh*l2n2 + sm*l2s2
        } else {
            lint = dlint_surf; // F:3472 lint = dlint_surf
            l2n2 = dl2n2_surf; // F:3473 l2n2 = dl2n2_surf
            l2s2 = dl2s2_surf; // F:3474 l2s2 = dl2s2_surf
            wint = dw_surf; // F:3475 wint = dw_surf
        }
        l2n2 = (-uw_min((-l2n2), ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sh)))); // F:3499 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
        l2s2 = uw_min(l2s2, ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sm))); // F:3500 l2s2 =  min( l2s2, tkemax*lint/(b1*sm))
        extend = false; // F:3530 extend = .false.
        tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3535 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
        lz = uw_pow((uw_pow((uw_vk * zi(kt)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3543 lz = ( (vk*zi(i,kt))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
        dzinc = (z((kt - 1)) - z(kt)); // F:3549 dzinc = z(i,kt-1)-z(i,kt)
        dl2n2 = (((lz * lz) * n2(kt)) * dzinc); // F:3550 dl2n2 = lz*lz*n2(i,kt)*dzinc
        dl2s2 = (((lz * lz) * s2(kt)) * dzinc); // F:3551 dl2s2 = lz*lz*s2(i,kt)*dzinc
        dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3552 dwinc = -sh*dl2n2 + sm*dl2s2
        while ((((-dl2n2) > (-((uw_r8(0xbfa47ae147ae147bul) * l2n2) / uw_r8(0x3ff0a3d70a3d70a4ul)))) && ((kt - 1) > 1))) { // F:3563 do while ( -dl2n2 .gt. (-rinc*l2n2/(1._r8-rinc)) .and. kt-1 .gt. ntop_turb )
            lint = (lint + dzinc); // F:3574 lint = lint + dzinc
            l2n2 = (l2n2 + dl2n2); // F:3575 l2n2 = l2n2 + dl2n2
            l2n2 = (-uw_min((-l2n2), ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sh)))); // F:3576 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
            l2s2 = (l2s2 + dl2s2); // F:3577 l2s2 = l2s2 + dl2s2
            wint = (wint + dwinc); // F:3578 wint = wint + dwinc
            kt = (kt - 1); // F:3582 kt        = kt - 1
            extend = true; // F:3583 extend    = .true.
            extend_up = true; // F:3584 extend_up = .true.
            if ((kt == 1)) { // F:3585 if( kt .eq. ntop_turb ) then
                atomic_store_explicit(ws.err, 3590, memory_order_relaxed); return; // F:3590 stop
            }
            ktinc = (kbase(((ncv + cntu) + 1)) - 1); // F:3599 ktinc = kbase(i,ncv+cntu+1) - 1
            if ((kt == ktinc)) { // F:3601 if( kt .eq. ktinc ) then
                for (k = (kbase(((ncv + cntu) + 1)) - 1); k >= (ktop(((ncv + cntu) + 1)) + 1); k += -1) { // F:3603 do k = kbase(i,ncv+cntu+1) - 1, ktop(i,ncv+cntu+1) + 1, -1
                    tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3606 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                    lz = uw_pow((uw_pow((uw_vk * zi(k)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3614 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                    dzinc = (z((k - 1)) - z(k)); // F:3620 dzinc = z(i,k-1)-z(i,k)
                    dl2n2 = (((lz * lz) * n2(k)) * dzinc); // F:3621 dl2n2 = lz*lz*n2(i,k)*dzinc
                    dl2s2 = (((lz * lz) * s2(k)) * dzinc); // F:3622 dl2s2 = lz*lz*s2(i,k)*dzinc
                    dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3623 dwinc = -sh*dl2n2 + sm*dl2s2
                    lint = (lint + dzinc); // F:3625 lint = lint + dzinc
                    l2n2 = (l2n2 + dl2n2); // F:3626 l2n2 = l2n2 + dl2n2
                    l2n2 = (-uw_min((-l2n2), ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sh)))); // F:3627 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                    l2s2 = (l2s2 + dl2s2); // F:3628 l2s2 = l2s2 + dl2s2
                    wint = (wint + dwinc); // F:3629 wint = wint + dwinc
                }
                kt = ktop(((ncv + cntu) + 1)); // F:3633 kt        = ktop(i,ncv+cntu+1)
                ncvfin = (ncvfin - 1); // F:3634 ncvfin(i) = ncvfin(i) - 1
                cntu = (cntu + 1); // F:3635 cntu      = cntu + 1
            }
            tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3643 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
            lz = uw_pow((uw_pow((uw_vk * zi(kt)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3651 lz = ( (vk*zi(i,kt))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            dzinc = (z((kt - 1)) - z(kt)); // F:3657 dzinc = z(i,kt-1)-z(i,kt)
            dl2n2 = (((lz * lz) * n2(kt)) * dzinc); // F:3658 dl2n2 = lz*lz*n2(i,kt)*dzinc
            dl2s2 = (((lz * lz) * s2(kt)) * dzinc); // F:3659 dl2s2 = lz*lz*s2(i,kt)*dzinc
            dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3660 dwinc = -sh*dl2n2 + sm*dl2s2
        }
        if ((cntu > 0)) { // F:3670 if( cntu .gt. 0 ) then
            for (incv = 1; incv <= (ncvfin - ncv); incv += 1) { // F:3671 do incv = 1, ncvfin(i) - ncv
                kbase((ncv + incv)) = kbase(((ncv + cntu) + incv)); // F:3672 kbase(i,ncv+incv) = kbase(i,ncv+cntu+incv)
                ktop((ncv + incv)) = ktop(((ncv + cntu) + incv)); // F:3673 ktop(i,ncv+incv)  = ktop(i,ncv+cntu+incv)
            }
        }
        if ((kb != (pver + 1))) { // F:3681 if( kb .ne. pver + 1 ) then
            tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3686 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
            lz = uw_pow((uw_pow((uw_vk * zi(kb)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3694 lz = ( (vk*zi(i,kb))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            dzinc = (z((kb - 1)) - z(kb)); // F:3700 dzinc = z(i,kb-1)-z(i,kb)
            dl2n2 = (((lz * lz) * n2(kb)) * dzinc); // F:3701 dl2n2 = lz*lz*n2(i,kb)*dzinc
            dl2s2 = (((lz * lz) * s2(kb)) * dzinc); // F:3702 dl2s2 = lz*lz*s2(i,kb)*dzinc
            dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3703 dwinc = -sh*dl2n2 + sm*dl2s2
            while ((((-dl2n2) > (-((uw_r8(0xbfa47ae147ae147bul) * l2n2) / uw_r8(0x3ff0a3d70a3d70a4ul)))) && (kb != (pver + 1)))) { // F:3716 do while( ( -dl2n2 .gt. (-rinc*l2n2/(1._r8-rinc)) ) .and.(kb.ne.pver+1))
                lint = (lint + dzinc); // F:3721 lint = lint + dzinc
                l2n2 = (l2n2 + dl2n2); // F:3722 l2n2 = l2n2 + dl2n2
                l2n2 = (-uw_min((-l2n2), ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sh)))); // F:3723 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                l2s2 = (l2s2 + dl2s2); // F:3724 l2s2 = l2s2 + dl2s2
                wint = (wint + dwinc); // F:3725 wint = wint + dwinc
                kb = (kb + 1); // F:3729 kb        =  kb + 1
                extend = true; // F:3730 extend    = .true.
                extend_dn = true; // F:3731 extend_dn = .true.
                kbinc = 0; // F:3744 kbinc = 0
                if ((ncv > 1)) { // F:3745 if( ncv .gt. 1 ) kbinc = ktop(i,ncv-1) + 1
                    kbinc = (ktop((ncv - 1)) + 1); // F:3745 if( ncv .gt. 1 ) kbinc = ktop(i,ncv-1) + 1
                }
                if ((kb == kbinc)) { // F:3746 if( kb .eq. kbinc ) then
                    for (k = (ktop((ncv - 1)) + 1); k <= (kbase((ncv - 1)) - 1); k += 1) { // F:3748 do k =  ktop(i,ncv-1) + 1, kbase(i,ncv-1) - 1
                        tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3751 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                        lz = uw_pow((uw_pow((uw_vk * zi(k)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3759 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                        dzinc = (z((k - 1)) - z(k)); // F:3765 dzinc = z(i,k-1)-z(i,k)
                        dl2n2 = (((lz * lz) * n2(k)) * dzinc); // F:3766 dl2n2 = lz*lz*n2(i,k)*dzinc
                        dl2s2 = (((lz * lz) * s2(k)) * dzinc); // F:3767 dl2s2 = lz*lz*s2(i,k)*dzinc
                        dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3768 dwinc = -sh*dl2n2 + sm*dl2s2
                        lint = (lint + dzinc); // F:3770 lint = lint + dzinc
                        l2n2 = (l2n2 + dl2n2); // F:3771 l2n2 = l2n2 + dl2n2
                        l2n2 = (-uw_min((-l2n2), ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sh)))); // F:3772 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                        l2s2 = (l2s2 + dl2s2); // F:3773 l2s2 = l2s2 + dl2s2
                        wint = (wint + dwinc); // F:3774 wint = wint + dwinc
                    }
                    kb = kbase((ncv - 1)); // F:3781 kb        = kbase(i,ncv-1)
                    ncv = (ncv - 1); // F:3782 ncv       = ncv - 1
                    ncvfin = (ncvfin - 1); // F:3783 ncvfin(i) = ncvfin(i) -1
                    cntd = (cntd + 1); // F:3784 cntd      = cntd + 1
                }
                if ((kb == (pver + 1))) { // F:3792 if( kb .eq. pver + 1 ) then
                    if ((bflxs > uw_r8(0x0000000000000000ul))) { // F:3794 if( bflxs(i) .gt. 0._r8 ) then
                        gg = ((((uw_r8(0x3fe0000000000000ul) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, uw_r8(0x3ff8000000000000ul))); // F:3796 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                        gh_surf = (gg / (uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))); // F:3797 gh_surf = gg/(alph5-gg*alph3)
                        gh_surf = uw_min(uw_max(gh_surf, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3799 gh_surf = min(max(gh_surf,-3.5334_r8),0.0233_r8)
                        sh_surf = (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh_surf))); // F:3800 sh_surf = alph5/(1._r8+alph3*gh_surf)
                        sm_surf = (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh_surf)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh_surf))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh_surf))); // F:3801 sm_surf = (alph1 + alph2*gh_surf)/(1._r8+alph3*gh_surf)/(1._r8+alph4*gh_surf)
                        dlint_surf = z(pver); // F:3804 dlint_surf = z(i,pver)
                        dl2n2_surf = (-(((uw_vk * uw_pow(z(pver), uw_r8(0x4000000000000000ul))) * bprod((pver + 1))) / (sh_surf * uw_sqrt(tkes)))); // F:3805 dl2n2_surf = -vk*(z(i,pver)**2._r8)*bprod(i,pver+1)/(sh_surf*sqrt(tkes(i)))
                        dl2s2_surf = (((uw_vk * uw_pow(z(pver), uw_r8(0x4000000000000000ul))) * sprod((pver + 1))) / (sm_surf * uw_sqrt(tkes))); // F:3806 dl2s2_surf =  vk*(z(i,pver)**2._r8)*sprod(i,pver+1)/(sm_surf*sqrt(tkes(i)))
                        dw_surf = ((tkes / uw_r8(0x4017333333333333ul)) * z(pver)); // F:3807 dw_surf = (tkes(i)/b1)*z(i,pver)
                    } else {
                        dlint_surf = uw_r8(0x0000000000000000ul); // F:3809 dlint_surf = 0._r8
                        dl2n2_surf = uw_r8(0x0000000000000000ul); // F:3810 dl2n2_surf = 0._r8
                        dl2s2_surf = uw_r8(0x0000000000000000ul); // F:3811 dl2s2_surf = 0._r8
                        dw_surf = uw_r8(0x0000000000000000ul); // F:3812 dw_surf = 0._r8
                    }
                    lint = (lint + dlint_surf); // F:3821 lint = lint + dlint_surf
                    l2n2 = (l2n2 + dl2n2_surf); // F:3822 l2n2 = l2n2 + dl2n2_surf
                    l2n2 = (-uw_min((-l2n2), ((uw_r8(0x4034000000000000ul) * lint) / (uw_r8(0x4017333333333333ul) * sh)))); // F:3823 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                    l2s2 = (l2s2 + dl2s2_surf); // F:3824 l2s2 = l2s2 + dl2s2_surf
                    wint = (wint + dw_surf); // F:3825 wint = wint + dw_surf
                } else {
                    tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3830 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                    lz = uw_pow((uw_pow((uw_vk * zi(kb)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3838 lz = ( (vk*zi(i,kb))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                    dzinc = (z((kb - 1)) - z(kb)); // F:3844 dzinc = z(i,kb-1)-z(i,kb)
                    dl2n2 = (((lz * lz) * n2(kb)) * dzinc); // F:3845 dl2n2 = lz*lz*n2(i,kb)*dzinc
                    dl2s2 = (((lz * lz) * s2(kb)) * dzinc); // F:3846 dl2s2 = lz*lz*s2(i,kb)*dzinc
                    dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3847 dwinc = -sh*dl2n2 + sm*dl2s2
                }
            }
            if (((kb == (pver + 1)) && (ncv != 1))) { // F:3853 if( (kb.eq.pver+1) .and. (ncv.ne.1) ) then
                atomic_store_explicit(ws.err, 3858, memory_order_relaxed); return; // F:3858 stop
            }
        }
        if ((cntd > 0)) { // F:3869 if( cntd .gt. 0 ) then
            for (incv = 1; incv <= (ncvfin - ncv); incv += 1) { // F:3870 do incv = 1, ncvfin(i) - ncv
                kbase((ncv + incv)) = kbase((ncvinit + incv)); // F:3871 kbase(i,ncv+incv) = kbase(i,ncvinit+incv)
                ktop((ncv + incv)) = ktop((ncvinit + incv)); // F:3872 ktop(i,ncv+incv)  = ktop(i,ncvinit+incv)
            }
        }
        if ((wint < uw_r8(0x3f847ae147ae147bul))) { // F:3878 if( wint .lt. 0.01_r8 ) then
            wint = uw_r8(0x3f847ae147ae147bul); // F:3879 wint = 0.01_r8
        }
        if (extend) { // F:3892 if( extend ) then
            ktop(ncv) = kt; // F:3894 ktop(i,ncv)  = kt
            kbase(ncv) = kb; // F:3895 kbase(i,ncv) = kb
            lbulk = (zi(kt) - zi(kb)); // F:3901 lbulk      = zi(i,kt) - zi(i,kb)
            dlint_surf = uw_r8(0x0000000000000000ul); // F:3902 dlint_surf = 0._r8
            dl2n2_surf = uw_r8(0x0000000000000000ul); // F:3903 dl2n2_surf = 0._r8
            dl2s2_surf = uw_r8(0x0000000000000000ul); // F:3904 dl2s2_surf = 0._r8
            dw_surf = uw_r8(0x0000000000000000ul); // F:3905 dw_surf    = 0._r8
            if ((kb == (pver + 1))) { // F:3906 if( kb .eq. pver + 1 ) then
                if ((bflxs > uw_r8(0x0000000000000000ul))) { // F:3907 if( bflxs(i) .gt. 0._r8 ) then
                    gg = ((((uw_r8(0x3fe0000000000000ul) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, uw_r8(0x3ff8000000000000ul))); // F:3909 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                    gh = (gg / (uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))); // F:3910 gh = gg/(alph5-gg*alph3)
                    gh = uw_min(uw_max(gh, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3912 gh = min(max(gh,-3.5334_r8),0.0233_r8)
                    sh = (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))); // F:3913 sh = alph5/(1._r8+alph3*gh)
                    sm = (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh))); // F:3914 sm = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
                    dlint_surf = z(pver); // F:3917 dlint_surf = z(i,pver)
                    dl2n2_surf = (-(((uw_vk * uw_pow(z(pver), uw_r8(0x4000000000000000ul))) * bprod((pver + 1))) / (sh * uw_sqrt(tkes)))); // F:3918 dl2n2_surf = -vk*(z(i,pver)**2._r8)*bprod(i,pver+1)/(sh*sqrt(tkes(i)))
                    dl2s2_surf = (((uw_vk * uw_pow(z(pver), uw_r8(0x4000000000000000ul))) * sprod((pver + 1))) / (sm * uw_sqrt(tkes))); // F:3919 dl2s2_surf =  vk*(z(i,pver)**2._r8)*sprod(i,pver+1)/(sm*sqrt(tkes(i)))
                    dw_surf = ((tkes / uw_r8(0x4017333333333333ul)) * z(pver)); // F:3920 dw_surf    = (tkes(i)/b1)*z(i,pver)
                } else {
                    lbulk = (zi(kt) - z(pver)); // F:3922 lbulk = zi(i,kt) - z(i,pver)
                }
            }
            lint = dlint_surf; // F:3925 lint = dlint_surf
            l2n2 = dl2n2_surf; // F:3926 l2n2 = dl2n2_surf
            l2s2 = dl2s2_surf; // F:3927 l2s2 = dl2s2_surf
            wint = dw_surf; // F:3928 wint = dw_surf
            l2n2 = uw_r8(0x0000000000000000ul); // F:3930 l2n2 = 0._r8
            l2s2 = uw_r8(0x0000000000000000ul); // F:3931 l2s2 = 0._r8
            for (k = (kt + 1); k <= (kb - 1); k += 1) { // F:3940 do k = kt + 1, kb - 1
                tunlramp = uw_r8(0x3fc051eb851eb852ul); // F:3942 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                lz = uw_pow((uw_pow((uw_vk * zi(k)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:3950 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                dzinc = (z((k - 1)) - z(k)); // F:3955 dzinc = z(i,k-1) - z(i,k)
                lint = (lint + dzinc); // F:3956 lint = lint + dzinc
                l2n2 = (l2n2 + (((lz * lz) * n2(k)) * dzinc)); // F:3957 l2n2 = l2n2 + lz*lz*n2(i,k)*dzinc
                l2s2 = (l2s2 + (((lz * lz) * s2(k)) * dzinc)); // F:3958 l2s2 = l2s2 + lz*lz*s2(i,k)*dzinc
            }
            ricll = uw_min((l2n2 / uw_max(l2s2, uw_r8(0x3d719799812dea11ul))), uw_r8(0x3fc851eb851eb852ul)); // F:3961 ricll = min(l2n2/max(l2s2,ntzero),ricrit)
            trma = ((uw_r8(0x406a8f03ff93f46aul) * ricll) + (uw_r8(0x4027333333333333ul) * (uw_r8(0xc01174bc6a7ef9dbul) - (uw_r8(0xc0111f3168d8b188ul) * ricll)))); // F:3962 trma = alph3*alph4*ricll+2._r8*b1*(alph2-alph4*alph5*ricll)
            trmb = ((ricll * uw_r8(0xc04466dc5d638866ul)) + (uw_r8(0x4027333333333333ul) * ((-(uw_r8(0x3fe65aee631f8a09ul) * ricll)) + uw_r8(0x3fe1cc63f141205cul)))); // F:3963 trmb = ricll*(alph3+alph4)+2._r8*b1*(-alph5*ricll+alph1)
            trmc = ricll; // F:3964 trmc = ricll
            det = uw_max(((trmb * trmb) - ((uw_r8(0x4010000000000000ul) * trma) * trmc)), uw_r8(0x0000000000000000ul)); // F:3965 det = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = ((((-trmb) + uw_sqrt(det)) / uw_r8(0x4000000000000000ul)) / trma); // F:3966 gh = (-trmb + sqrt(det))/2._r8/trma
            gh = uw_min(uw_max(gh, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3968 gh = min(max(gh,-3.5334_r8),0.0233_r8)
            sh = (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))); // F:3969 sh = alph5 / (1._r8+alph3*gh)
            sm = (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh))); // F:3970 sm = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
            wint = uw_max(((wint - (sh * l2n2)) + (sm * l2s2)), uw_r8(0x3f847ae147ae147bul)); // F:3974 wint = max( wint - sh*l2n2 + sm*l2s2, 0.01_r8 )
        }
        lbrk(ncv) = lint; // F:3982 lbrk(i,ncv) = lint
        wbrk(ncv) = (wint / lint); // F:3983 wbrk(i,ncv) = wint/lint
        ebrk(ncv) = (uw_r8(0x4017333333333333ul) * wbrk(ncv)); // F:3984 ebrk(i,ncv) = b1*wbrk(i,ncv)
        ebrk(ncv) = uw_min(ebrk(ncv), uw_r8(0x4034000000000000ul)); // F:3985 ebrk(i,ncv) = min(ebrk(i,ncv),tkemax)
        ricl(ncv) = ricll; // F:3986 ricl(i,ncv) = ricll
        ghcl(ncv) = gh; // F:3987 ghcl(i,ncv) = gh
        shcl(ncv) = sh; // F:3988 shcl(i,ncv) = sh
        smcl(ncv) = sm; // F:3989 smcl(i,ncv) = sm
        ncv = (ncv + 1); // F:3996 ncv = ncv + 1
    }
    for (ncv = (ncvfin + 1); ncv <= ncvmax; ncv += 1) { // F:4004 do ncv = ncvfin(i) + 1, ncvmax
        ktop(ncv) = 0; // F:4005 ktop(i,ncv)  = 0
        kbase(ncv) = 0; // F:4006 kbase(i,ncv) = 0
    }
    for (k = 1; k <= (pver + 1); k += 1) { // F:4014 do k = 1, pver + 1
        belongcv(k) = false; // F:4015 belongcv(i,k) = .false.
    }
    for (ncv = 1; ncv <= ncvfin; ncv += 1) { // F:4018 do ncv = 1, ncvfin(i)
        for (k = ktop(ncv); k <= kbase(ncv); k += 1) { // F:4019 do k = ktop(i,ncv), kbase(i,ncv)
            belongcv(k) = true; // F:4020 belongcv(i,k) = .true.
        }
    }
    return; // F:4024 return
}
#endif

// Generated literal transcription. F: comments refer to module_cam_bl_eddy_diff.F.
// Constant expression subtrees are folded in source grouping to exact hex words.
#ifndef UWPBL_CALEDDY_CUH
#define UWPBL_CALEDDY_CUH
static inline void uw_caleddy(int pver, V sl, V qt, V ql, V slv, V u, V v, V pi, V z, V zi, R8 qflx, R8 shflx, V slslope, V qtslope, V chu, V chs, V cmu, V cms, V sfuh, V sflh, V n2, V s2, V ri, R8 rrho, thread R8& pblh, R8 ustar, V kvh_in, V kvm_in, V kvh, V kvm, thread R8& tpert, thread R8& qpert, V qrlin, V kvf, V tke, bool wstarent, V bprod, V sprod, R8 minpblh, thread R8& wpert, thread R8& tkes, V turbtype_f, V sm_aw, V kbase_o, V ktop_o, thread R8& ncvfin_o, V kbase_mg, V ktop_mg, thread R8& ncvfin_mg, V kbase_f, V ktop_f, thread R8& ncvfin_f, V wet_CL, V web_CL, V jtbu_CL, V jbbu_CL, V evhc_CL, V jt2slv_CL, V n2ht_CL, V n2hb_CL, V lwp_CL, V opt_depth_CL, V radinvfrac_CL, V radf_CL, V wstar_CL, V wstar3fact_CL, V ebrk, V wbrk, V lbrk, V ricl, V ghcl, V shcl, V smcl, V gh_a, V sh_a, V sm_a, V ri_a, V leng, V wcap, thread R8& pblhp, V cld, thread R8& ipbl, thread R8& kpblh, V wsedl, thread Ws& ws) {
    const int ncvmax = pver;
    const int nbot_turb = pver;
    WsMark uw_mark(ws);
    VL belongcv = ws.i4((pver + 1));
    VL belongst = ws.i4((pver + 1));
    bool in_CL;
    bool extend;
    bool extend_up;
    bool extend_dn;
    int k;
    int ks;
    int ncvfin;
    int ncvf;
    int ncv;
    int ncvnew;
    int ncvsurf;
    VI kbase = ws.i4(ncvmax);
    VI ktop = ws.i4(ncvmax);
    int kb;
    int kt;
    int ktblw;
    VI turbtype = ws.i4((pver + 1));
    int ktopbl;
    R8 bflxs;
    R8 rcap;
    R8 jtzm;
    R8 jtsl;
    R8 jtqt;
    R8 jtbu;
    R8 jtu;
    R8 jtv;
    R8 jt2slv;
    R8 radf;
    R8 jbzm;
    R8 jbsl;
    R8 jbqt;
    R8 jbbu;
    R8 jbu;
    R8 jbv;
    R8 ch;
    R8 cm;
    R8 n2ht;
    R8 n2hb;
    R8 n2htsrcl;
    R8 gh;
    R8 sh;
    R8 sm;
    R8 lbulk;
    R8 dzht;
    R8 dzhb;
    R8 rootp;
    R8 evhc;
    R8 kentr;
    R8 lwp;
    R8 opt_depth;
    R8 radinvfrac;
    R8 wet;
    R8 web;
    R8 vyt;
    R8 vyb;
    R8 vut;
    R8 vub;
    R8 fact;
    R8 trma;
    R8 trmb;
    R8 trmc;
    R8 trmp;
    R8 trmq;
    R8 qq;
    R8 det;
    R8 gg;
    R8 dzhb5;
    R8 dzht5;
    V qrlw = ws.r8(pver);
    V cldeff = ws.r8(pver);
    R8 qleff;
    R8 tunlramp;
    R8 leng_imsi;
    R8 tke_imsi;
    R8 kvh_imsi;
    R8 kvm_imsi;
    R8 alph4exs;
    R8 ghmin;
    R8 sedfact;
    R8 cet;
    R8 ceb;
    R8 wstar;
    R8 wstar3;
    R8 wstar3fact;
    R8 rmin;
    R8 fmin;
    R8 rcrit;
    R8 fcrit;
    bool noroot;
    for (int uw_k = 1; uw_k <= pver; ++uw_k) { // F:1637 qrlw(:ncol,:pver) = qrlin(:ncol,:pver)
        qrlw(uw_k) = qrlin(uw_k);
    }
    for (k = 1; k <= pver; k += 1) { // F:1645 do k = 1, pver
        cldeff(k) = cld(k); // F:1650 cldeff(i,k) = cld(i,k)
    }
    alph4exs = uw_r8(0xc0188240b780346eul); // F:1659 alph4exs = alph4
    ghmin = uw_r8(0xc00c4467381d7dbful); // F:1660 ghmin    = -3.5334_r8
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1677 wet_CL(i,:ncvmax)        = 0._r8
        wet_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1678 web_CL(i,:ncvmax)        = 0._r8
        web_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1679 jtbu_CL(i,:ncvmax)       = 0._r8
        jtbu_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1680 jbbu_CL(i,:ncvmax)       = 0._r8
        jbbu_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1681 evhc_CL(i,:ncvmax)       = 0._r8
        evhc_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1682 jt2slv_CL(i,:ncvmax)     = 0._r8
        jt2slv_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1683 n2ht_CL(i,:ncvmax)       = 0._r8
        n2ht_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1684 n2hb_CL(i,:ncvmax)       = 0._r8
        n2hb_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1685 lwp_CL(i,:ncvmax)        = 0._r8
        lwp_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1686 opt_depth_CL(i,:ncvmax)  = 0._r8
        opt_depth_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1687 radinvfrac_CL(i,:ncvmax) = 0._r8
        radinvfrac_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1688 radf_CL(i,:ncvmax)       = 0._r8
        radf_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1689 wstar_CL(i,:ncvmax)      = 0._r8
        wstar_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1690 wstar3fact_CL(i,:ncvmax) = 0._r8
        wstar3fact_CL(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1691 ricl(i,:ncvmax)          = 0._r8
        ricl(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1692 ghcl(i,:ncvmax)          = 0._r8
        ghcl(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1693 shcl(i,:ncvmax)          = 0._r8
        shcl(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1694 smcl(i,:ncvmax)          = 0._r8
        smcl(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1695 ebrk(i,:ncvmax)          = 0._r8
        ebrk(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1696 wbrk(i,:ncvmax)          = 0._r8
        wbrk(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1697 lbrk(i,:ncvmax)          = 0._r8
        lbrk(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1698 gh_a(i,:pver+1)          = 0._r8
        gh_a(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1699 sh_a(i,:pver+1)          = 0._r8
        sh_a(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1700 sm_a(i,:pver+1)          = 0._r8
        sm_a(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1701 ri_a(i,:pver+1)          = 0._r8
        ri_a(uw_k) = uw_r8(0x0000000000000000ul);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1702 sm_aw(i,:pver+1)         = 0._r8
        sm_aw(uw_k) = uw_r8(0x0000000000000000ul);
    }
    ipbl = uw_r8(0x0000000000000000ul); // F:1703 ipbl(i)                  = 0._r8
    kpblh = uw_real(pver); // F:1704 kpblh(i)                 = real(pver,r8)
    for (k = 1; k <= (pver + 1); k += 1) { // F:1713 do k = 1, pver + 1
        kvh(k) = uw_r8(0x0000000000000000ul); // F:1720 kvh(i,k) = 0._r8
        kvm(k) = uw_r8(0x0000000000000000ul); // F:1721 kvm(i,k) = 0._r8
        wcap(k) = uw_r8(0x0000000000000000ul); // F:1724 wcap(i,k) = 0._r8
        leng(k) = uw_r8(0x0000000000000000ul); // F:1725 leng(i,k) = 0._r8
        tke(k) = uw_r8(0x0000000000000000ul); // F:1726 tke(i,k)  = 0._r8
        turbtype(k) = 0; // F:1727 turbtype(i,k) = 0
    }
    for (k = 2; k <= pver; k += 1) { // F:1739 do k = 2, pver
        bprod(k) = (-(kvh_in(k) * n2(k))); // F:1741 bprod(i,k) = -kvh_in(i,k) * n2(i,k)
        sprod(k) = (kvm_in(k) * s2(k)); // F:1742 sprod(i,k) =  kvm_in(i,k) * s2(i,k)
    }
    bprod(1) = uw_r8(0x0000000000000000ul); // F:1762 bprod(i,1) = 0._r8
    sprod(1) = uw_r8(0x0000000000000000ul); // F:1763 sprod(i,1) = 0._r8
    ch = ((chu((pver + 1)) * (uw_r8(0x3ff0000000000000ul) - sflh(pver))) + (chs((pver + 1)) * sflh(pver))); // F:1764 ch = chu(i,pver+1) * ( 1._r8 - sflh(i,pver) ) + chs(i,pver+1) * sflh(i,pver)
    cm = ((cmu((pver + 1)) * (uw_r8(0x3ff0000000000000ul) - sflh(pver))) + (cms((pver + 1)) * sflh(pver))); // F:1765 cm = cmu(i,pver+1) * ( 1._r8 - sflh(i,pver) ) + cms(i,pver+1) * sflh(i,pver)
    bflxs = (((ch * shflx) * rrho) + ((cm * qflx) * rrho)); // F:1766 bflxs(i) = ch * shflx(i) * rrho(i) + cm * qflx(i) * rrho(i)
    bprod((pver + 1)) = bflxs; // F:1768 bprod(i,pver+1) = bflxs(i)
    sprod((pver + 1)) = (uw_powi(ustar, 3) / (uw_vk * z(pver))); // F:1772 sprod(i,pver+1) = (ustar(i)**3)/(vk*z(i,pver))
    uw_exacol(pver, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin, ws); // F:1787 call exacol( pcols, pver, ncol, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin )
    for (k = 1; k <= ncvmax; k += 1) { // F:1792 do k = 1, ncvmax
        kbase_o(k) = uw_real(kbase(k)); // F:1793 kbase_o(i,k) = real(kbase(i,k),r8)
        ktop_o(k) = uw_real(ktop(k)); // F:1794 ktop_o(i,k)  = real(ktop(i,k),r8)
        ncvfin_o = uw_real(ncvfin); // F:1795 ncvfin_o(i)  = real(ncvfin(i),r8)
    }
    tkes = uw_pow(uw_max((((uw_r8(0x4017333333333333ul) * uw_vk) * z(pver)) * (bprod((pver + 1)) + sprod((pver + 1)))), uw_r8(0x3e7ad7f29abcaf48ul)), uw_r8(0x3fe5555555555555ul)); // F:1822 tkes(i) = max(b1*vk*z(i,pver)*(bprod(i,pver+1)+sprod(i,pver+1)), 1.e-7_r8)**(2._r8/3._r8)
    tkes = uw_min(tkes, uw_r8(0x4034000000000000ul)); // F:1823 tkes(i) = min(tkes(i), tkemax)
    tke((pver + 1)) = tkes; // F:1824 tke(i,pver+1)  = tkes(i)
    wcap((pver + 1)) = (tkes / uw_r8(0x4017333333333333ul)); // F:1825 wcap(i,pver+1) = tkes(i)/b1
    ncvsurf = 0; // F:1856 ncvsurf = 0
    if ((ncvfin > 0)) { // F:1857 if( ncvfin(i) .gt. 0 ) then
        uw_zisocl(pver, z, zi, n2, s2, bprod, sprod, bflxs, tkes, ncvfin, kbase, ktop, belongcv, ricl, ghcl, shcl, smcl, lbrk, wbrk, ebrk, extend, extend_up, extend_dn, ws); // F:1858 call zisocl( pcols  , pver     , i        , z      , zi       , n2       , s2      , bprod  , sprod    , bflxs    , tkes    , ncvfin , kbase    , ktop     , belongcv, ricl   , ghcl     , shcl     , smcl    , lbrk   , wbrk     , ebrk     , extend , extend_up, extend_dn )
        if ((kbase(1) == (pver + 1))) { // F:1865 if( kbase(i,1) .eq. pver + 1 ) ncvsurf = 1
            ncvsurf = 1; // F:1865 if( kbase(i,1) .eq. pver + 1 ) ncvsurf = 1
        }
    } else {
        for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1867 belongcv(i,:) = .false.
            belongcv(uw_k) = false;
        }
    }
    for (k = 1; k <= ncvmax; k += 1) { // F:1873 do k = 1, ncvmax
        kbase_mg(k) = uw_widen((float)(kbase(k))); // F:1874 kbase_mg(i,k) = real(kbase(i,k))
        ktop_mg(k) = uw_widen((float)(ktop(k))); // F:1875 ktop_mg(i,k)  = real(ktop(i,k))
        ncvfin_mg = uw_widen((float)(ncvfin)); // F:1876 ncvfin_mg(i)  = real(ncvfin(i))
    }
    ncv = 1; // F:1921 ncv  = 1
    ncvf = ncvfin; // F:1922 ncvf = ncvfin(i)
    for (k = nbot_turb; k >= 2; k += -1) { // F:1926 do k = nbot_turb, ntop_turb + 1, -1
        if (((((ql(k) > uw_r8(0x3ee4f8b588e368f1ul)) && (ql((k - 1)) < uw_r8(0x3ee4f8b588e368f1ul))) && (qrlw(k) < uw_r8(0x0000000000000000ul))) && (ri(k) >= uw_r8(0x3fc851eb851eb852ul)))) { // F:1928 if( ql(i,k) .gt. qmin .and. ql(i,k-1) .lt. qmin .and. qrlw(i,k) .lt. 0._r8 .and. ri(i,k) .ge. ricrit ) then
            if ((true && belongcv((k + 1)))) { // F:1938 if( choice_srcl .eq. 'nonamb' .and. belongcv(i,k+1) ) then
                continue; // F:1939 go to 220
            }
            ch = (((uw_r8(0x3ff0000000000000ul) - sfuh(k)) * chu(k)) + (sfuh(k) * chs(k))); // F:1942 ch = ( 1._r8 - sfuh(i,k) ) * chu(i,k) + sfuh(i,k) * chs(i,k)
            cm = (((uw_r8(0x3ff0000000000000ul) - sfuh(k)) * cmu(k)) + (sfuh(k) * cms(k))); // F:1943 cm = ( 1._r8 - sfuh(i,k) ) * cmu(i,k) + sfuh(i,k) * cms(i,k)
            n2htsrcl = ((ch * slslope(k)) + (cm * qtslope(k))); // F:1945 n2htsrcl = ch * slslope(i,k) + cm * qtslope(i,k)
            if ((n2htsrcl <= uw_r8(0x0000000000000000ul))) { // F:1947 if( n2htsrcl .le. 0._r8 ) then
                in_CL = false; // F:1955 in_CL = .false.
                while ((ncv <= ncvf)) { // F:1957 do while ( ncv .le. ncvf )
                    if ((ktop(ncv) <= k)) { // F:1958 if( ktop(i,ncv) .le. k ) then
                        if ((kbase(ncv) > k)) { // F:1959 if( kbase(i,ncv) .gt. k ) then
                            in_CL = true; // F:1960 in_CL = .true.
                        }
                        break; // F:1962 exit
                    } else {
                        ncv = (ncv + 1); // F:1964 ncv = ncv + 1
                    }
                }
                if ((!in_CL)) { // F:1968 if( .not. in_CL ) then
                    ncvfin = (ncvfin + 1); // F:1972 ncvfin(i)       =  ncvfin(i) + 1
                    ncvnew = ncvfin; // F:1973 ncvnew          =  ncvfin(i)
                    ktop(ncvnew) = k; // F:1974 ktop(i,ncvnew)  =  k
                    kbase(ncvnew) = (k + 1); // F:1975 kbase(i,ncvnew) =  k+1
                    belongcv(k) = true; // F:1976 belongcv(i,k)   = .true.
                    belongcv((k + 1)) = true; // F:1977 belongcv(i,k+1) = .true.
                    if ((k < pver)) { // F:1994 if( k .lt. pver ) then
                        wbrk(ncvnew) = uw_r8(0x0000000000000000ul); // F:1996 wbrk(i,ncvnew) = 0._r8
                        ebrk(ncvnew) = uw_r8(0x0000000000000000ul); // F:1997 ebrk(i,ncvnew) = 0._r8
                        lbrk(ncvnew) = uw_r8(0x0000000000000000ul); // F:1998 lbrk(i,ncvnew) = 0._r8
                        ghcl(ncvnew) = uw_r8(0x0000000000000000ul); // F:1999 ghcl(i,ncvnew) = 0._r8
                        shcl(ncvnew) = uw_r8(0x0000000000000000ul); // F:2000 shcl(i,ncvnew) = 0._r8
                        smcl(ncvnew) = uw_r8(0x0000000000000000ul); // F:2001 smcl(i,ncvnew) = 0._r8
                        ricl(ncvnew) = uw_r8(0x0000000000000000ul); // F:2002 ricl(i,ncvnew) = 0._r8
                    } else {
                        if ((bflxs > uw_r8(0x0000000000000000ul))) { // F:2006 if( bflxs(i) .gt. 0._r8 ) then
                            ebrk(ncvnew) = tkes; // F:2010 ebrk(i,ncvnew) = tkes(i)
                            lbrk(ncvnew) = z(pver); // F:2011 lbrk(i,ncvnew) = z(i,pver)
                            wbrk(ncvnew) = (tkes / uw_r8(0x4017333333333333ul)); // F:2012 wbrk(i,ncvnew) = tkes(i) / b1
                            for (ks = 1; ks <= ncvmax; ks += 1) { // F:2030 do ks = 1, ncvmax
                                ;
                            }
                            atomic_store_explicit(ws.err, 2036, memory_order_relaxed); return; // F:2036 stop
                        } else {
                            ebrk(ncvnew) = uw_r8(0x0000000000000000ul); // F:2040 ebrk(i,ncvnew) = 0._r8
                            lbrk(ncvnew) = uw_r8(0x0000000000000000ul); // F:2041 lbrk(i,ncvnew) = 0._r8
                            wbrk(ncvnew) = uw_r8(0x0000000000000000ul); // F:2042 wbrk(i,ncvnew) = 0._r8
                        }
                        gg = ((((uw_r8(0x3fe0000000000000ul) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, uw_r8(0x3ff8000000000000ul))); // F:2056 gg = 0.5_r8 * vk * z(i,pver) * bprod(i,pver+1) / ( tkes(i)**(3._r8/2._r8) )
                        if ((uw_abs((uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))) <= uw_r8(0x3e7ad7f29abcaf48ul))) { // F:2057 if( abs(alph5-gg*alph3) .le. 1.e-7_r8 ) then
                            gh = ghmin; // F:2060 gh = ghmin
                        } else {
                            gh = (gg / (uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))); // F:2062 gh = gg / ( alph5 - gg * alph3 )
                        }
                        gh = uw_min(uw_max(gh, ghmin), uw_r8(0x3f97dbf487fcb924ul)); // F:2066 gh = min(max(gh,ghmin),0.0233_r8)
                        ghcl(ncvnew) = gh; // F:2067 ghcl(i,ncvnew) =  gh
                        shcl(ncvnew) = uw_max(uw_r8(0x0000000000000000ul), (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh)))); // F:2068 shcl(i,ncvnew) =  max(0._r8,alph5/(1._r8+alph3*gh))
                        smcl(ncvnew) = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (alph4exs * gh)))); // F:2069 smcl(i,ncvnew) =  max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
                        ricl(ncvnew) = (-((smcl(ncvnew) / shcl(ncvnew)) * (bprod((pver + 1)) / sprod((pver + 1))))); // F:2070 ricl(i,ncvnew) = -(smcl(i,ncvnew)/shcl(i,ncvnew))*(bprod(i,pver+1)/sprod(i,pver+1))
                        ncvsurf = ncvnew; // F:2075 ncvsurf = ncvnew
                    }
                }
            }
        }
    }
    for (k = 1; k <= ncvmax; k += 1) { // F:2108 do k = 1, ncvmax
        kbase_f(k) = uw_widen((float)(kbase(k))); // F:2109 kbase_f(i,k) = real(kbase(i,k))
        ktop_f(k) = uw_widen((float)(ktop(k))); // F:2110 ktop_f(i,k)  = real(ktop(i,k))
        ncvfin_f = uw_widen((float)(ncvfin)); // F:2111 ncvfin_f(i)  = real(ncvfin(i))
    }
    ktblw = 0; // F:2134 ktblw = 0
    for (ncv = 1; ncv <= ncvfin; ncv += 1) { // F:2135 do ncv = 1, ncvfin(i)
        kt = ktop(ncv); // F:2137 kt = ktop(i,ncv)
        kb = kbase(ncv); // F:2138 kb = kbase(i,ncv)
        if (((kb == (pver + 1)) && (bflxs <= uw_r8(0x0000000000000000ul)))) { // F:2140 if( kb .eq. (pver+1) .and. bflxs(i) .le. 0._r8 ) then
            lbulk = (zi(kt) - z(pver)); // F:2141 lbulk = zi(i,kt) - z(i,pver)
        } else {
            lbulk = (zi(kt) - zi(kb)); // F:2143 lbulk = zi(i,kt) - zi(i,kb)
        }
        for (k = uw_imin(kb, pver); k >= kt; k += -1) { // F:2152 do k = min(kb,pver), kt, -1
            tunlramp = (uw_r8(0x3fc5c28f5c28f5c3ul) * (uw_r8(0x3ff0000000000000ul) - (uw_r8(0x3fe0000000000000ul) * uw_exp(uw_min(uw_r8(0x0000000000000000ul), ricl(ncv)))))); // F:2158 tunlramp = ctunl*tunl*(1._r8-(1._r8-1._r8/ctunl)*exp(min(0._r8,ricl(i,ncv))))
            tunlramp = uw_min(uw_max(tunlramp, uw_r8(0x3fb5c28f5c28f5c3ul)), uw_r8(0x3fc5c28f5c28f5c3ul)); // F:2159 tunlramp = min(max(tunlramp,tunl),ctunl*tunl)
            leng(k) = uw_pow((uw_pow((uw_vk * zi(k)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:2167 leng(i,k) = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            wcap(k) = (uw_sq(leng(k)) * ((-(shcl(ncv) * n2(k))) + (smcl(ncv) * s2(k)))); // F:2172 wcap(i,k) = (leng(i,k)**2) * (-shcl(i,ncv)*n2(i,k)+smcl(i,ncv)*s2(i,k))
        }
        if ((kb < (pver + 1))) { // F:2178 if( kb .lt. pver+1 ) then
            jbzm = (z((kb - 1)) - z(kb)); // F:2180 jbzm = z(i,kb-1) - z(i,kb)
            jbsl = (sl((kb - 1)) - sl(kb)); // F:2181 jbsl = sl(i,kb-1) - sl(i,kb)
            jbqt = (qt((kb - 1)) - qt(kb)); // F:2182 jbqt = qt(i,kb-1) - qt(i,kb)
            jbbu = (n2(kb) * jbzm); // F:2183 jbbu = n2(i,kb) * jbzm
            jbbu = uw_max(jbbu, uw_r8(0x3f50624dd2f1a9fcul)); // F:2184 jbbu = max(jbbu,jbumin)
            jbu = (u((kb - 1)) - u(kb)); // F:2185 jbu  = u(i,kb-1) - u(i,kb)
            jbv = (v((kb - 1)) - v(kb)); // F:2186 jbv  = v(i,kb-1) - v(i,kb)
            ch = (((uw_r8(0x3ff0000000000000ul) - sflh((kb - 1))) * chu(kb)) + (sflh((kb - 1)) * chs(kb))); // F:2187 ch   = (1._r8 -sflh(i,kb-1))*chu(i,kb) + sflh(i,kb-1)*chs(i,kb)
            cm = (((uw_r8(0x3ff0000000000000ul) - sflh((kb - 1))) * cmu(kb)) + (sflh((kb - 1)) * cms(kb))); // F:2188 cm   = (1._r8 -sflh(i,kb-1))*cmu(i,kb) + sflh(i,kb-1)*cms(i,kb)
            n2hb = (((ch * jbsl) + (cm * jbqt)) / jbzm); // F:2189 n2hb = (ch*jbsl + cm*jbqt)/jbzm
            vyb = ((n2hb * jbzm) / jbbu); // F:2190 vyb  = n2hb*jbzm/jbbu
            vub = uw_min(uw_r8(0x3ff0000000000000ul), ((uw_sq(jbu) + uw_sq(jbv)) / (jbbu * jbzm))); // F:2191 vub  = min(1._r8,(jbu**2+jbv**2)/(jbbu*jbzm) )
        } else {
            jbbu = uw_r8(0x0000000000000000ul); // F:2196 jbbu = 0._r8
            n2hb = uw_r8(0x0000000000000000ul); // F:2197 n2hb = 0._r8
            vyb = uw_r8(0x0000000000000000ul); // F:2198 vyb  = 0._r8
            vub = uw_r8(0x0000000000000000ul); // F:2199 vub  = 0._r8
            web = uw_r8(0x0000000000000000ul); // F:2200 web  = 0._r8
        }
        jtzm = (z((kt - 1)) - z(kt)); // F:2208 jtzm = z(i,kt-1) - z(i,kt)
        jtsl = (sl((kt - 1)) - sl(kt)); // F:2209 jtsl = sl(i,kt-1) - sl(i,kt)
        jtqt = (qt((kt - 1)) - qt(kt)); // F:2210 jtqt = qt(i,kt-1) - qt(i,kt)
        jtbu = (n2(kt) * jtzm); // F:2211 jtbu = n2(i,kt)*jtzm
        jtbu = uw_max(jtbu, uw_r8(0x3f50624dd2f1a9fcul)); // F:2212 jtbu = max(jtbu,jbumin)
        jtu = (u((kt - 1)) - u(kt)); // F:2213 jtu  = u(i,kt-1) - u(i,kt)
        jtv = (v((kt - 1)) - v(kt)); // F:2214 jtv  = v(i,kt-1) - v(i,kt)
        ch = (((uw_r8(0x3ff0000000000000ul) - sfuh(kt)) * chu(kt)) + (sfuh(kt) * chs(kt))); // F:2215 ch   = (1._r8 -sfuh(i,kt))*chu(i,kt) + sfuh(i,kt)*chs(i,kt)
        cm = (((uw_r8(0x3ff0000000000000ul) - sfuh(kt)) * cmu(kt)) + (sfuh(kt) * cms(kt))); // F:2216 cm   = (1._r8 -sfuh(i,kt))*cmu(i,kt) + sfuh(i,kt)*cms(i,kt)
        n2ht = (((ch * jtsl) + (cm * jtqt)) / jtzm); // F:2217 n2ht = (ch*jtsl + cm*jtqt)/jtzm
        vyt = ((n2ht * jtzm) / jtbu); // F:2218 vyt  = n2ht*jtzm/jtbu
        vut = uw_min(uw_r8(0x3ff0000000000000ul), ((uw_sq(jtu) + uw_sq(jtv)) / (jtbu * jtzm))); // F:2219 vut  = min(1._r8,(jtu**2+jtv**2)/(jtbu*jtzm))
        evhc = uw_r8(0x3ff0000000000000ul); // F:2229 evhc   = 1._r8
        jt2slv = uw_r8(0x0000000000000000ul); // F:2230 jt2slv = 0._r8
        qleff = uw_max(ql((kt - 1)), ql(kt)); // F:2253 qleff  = max( ql(i,kt-1), ql(i,kt) )
        jt2slv = (slv(uw_imax((kt - 2), 1)) - slv(kt)); // F:2254 jt2slv = slv(i,max(kt-2,1)) - slv(i,kt)
        jt2slv = uw_max(jt2slv, ((uw_r8(0x3f50624dd2f1a9fcul) * slv((kt - 1))) / uw_g)); // F:2255 jt2slv = max( jt2slv, jbumin*slv(i,kt-1)/g )
        evhc = (uw_r8(0x3ff0000000000000ul) + (((uw_r8(0x4038000000000000ul) * uw_latvap) * qleff) / jt2slv)); // F:2256 evhc   = 1._r8 + a2l * a3l * latvap * qleff / jt2slv
        evhc = uw_min(evhc, uw_r8(0x4024000000000000ul)); // F:2257 evhc   = min( evhc, evhcmax )
        lwp = uw_r8(0x0000000000000000ul); // F:2268 lwp        = 0._r8
        opt_depth = uw_r8(0x0000000000000000ul); // F:2269 opt_depth  = 0._r8
        radinvfrac = uw_r8(0x0000000000000000ul); // F:2270 radinvfrac = 0._r8
        radf = uw_r8(0x0000000000000000ul); // F:2271 radf       = 0._r8
        lwp = ((ql(kt) * (pi((kt + 1)) - pi(kt))) / uw_g); // F:2304 lwp         = ql(i,kt) * ( pi(i,kt+1) - pi(i,kt) ) / g
        opt_depth = (uw_r8(0x4063800000000000ul) * lwp); // F:2305 opt_depth   = 156._r8 * lwp
        radinvfrac = ((opt_depth * (uw_r8(0x4010000000000000ul) + opt_depth)) / ((uw_r8(0x4018000000000000ul) * (uw_r8(0x4010000000000000ul) + opt_depth)) + uw_sq(opt_depth))); // F:2306 radinvfrac  = opt_depth * ( 4._r8 + opt_depth ) / ( 6._r8 * ( 4._r8 + opt_depth ) + opt_depth**2 )
        radf = uw_max((((radinvfrac * qrlw(kt)) / (pi(kt) - pi((kt + 1)))) * (zi(kt) - zi((kt + 1)))), uw_r8(0x0000000000000000ul)); // F:2307 radf        = max( radinvfrac * qrlw(i,kt) / ( pi(i,kt) - pi(i,kt+1) ) * ( zi(i,kt) - zi(i,kt+1) ), 0._r8 )
        lwp = ((ql((kt - 1)) * (pi(kt) - pi((kt - 1)))) / uw_g); // F:2309 lwp         = ql(i,kt-1) * ( pi(i,kt) - pi(i,kt-1) ) / g
        opt_depth = (uw_r8(0x4063800000000000ul) * lwp); // F:2310 opt_depth   = 156._r8 * lwp
        radinvfrac = ((opt_depth * (uw_r8(0x4010000000000000ul) + opt_depth)) / ((uw_r8(0x4018000000000000ul) * (uw_r8(0x4010000000000000ul) + opt_depth)) + uw_sq(opt_depth))); // F:2311 radinvfrac  = opt_depth * ( 4._r8 + opt_depth ) / ( 6._r8 * ( 4._r8 + opt_depth) + opt_depth**2 )
        radf = (radf + uw_max((((radinvfrac * qrlw((kt - 1))) / (pi((kt - 1)) - pi(kt))) * (zi((kt - 1)) - zi(kt))), uw_r8(0x0000000000000000ul))); // F:2312 radf        = radf + max( radinvfrac * qrlw(i,kt-1) / ( pi(i,kt-1) - pi(i,kt) ) * ( zi(i,kt-1) - zi(i,kt) ), 0._r8 )
        radf = (uw_max(radf, uw_r8(0x0000000000000000ul)) * chs(kt)); // F:2314 radf        = max( radf, 0._r8 ) * chs(i,kt)
        dzht = (zi(kt) - z(kt)); // F:2339 dzht   = zi(i,kt)  - z(i,kt)
        dzhb = (z((kb - 1)) - zi(kb)); // F:2340 dzhb   = z(i,kb-1) - zi(i,kb)
        wstar3 = (radf * dzht); // F:2341 wstar3 = radf * dzht
        for (k = (kt + 1); k <= (kb - 1); k += 1) { // F:2342 do k = kt + 1, kb - 1
            wstar3 = (wstar3 + (bprod(k) * (z((k - 1)) - z(k)))); // F:2343 wstar3 =  wstar3 + bprod(i,k) * ( z(i,k-1) - z(i,k) )
        }
        if (((kb == (pver + 1)) && (bflxs > uw_r8(0x0000000000000000ul)))) { // F:2351 if( kb .eq. (pver+1) .and. bflxs(i) .gt. 0._r8 ) then
            wstar3 = (wstar3 + (bflxs * dzhb)); // F:2352 wstar3 = wstar3 + bflxs(i) * dzhb
        }
        wstar3 = uw_max((uw_r8(0x4004000000000000ul) * wstar3), uw_r8(0x0000000000000000ul)); // F:2355 wstar3 = max( 2.5_r8 * wstar3, 0._r8 )
        if ((wstar3 > uw_r8(0x0000000000000000ul))) { // F:2408 if( wstar3 .gt. 0._r8 ) then
            cet = ((uw_r8(0x3fc999999999999aul) * evhc) / (jtbu * lbulk)); // F:2409 cet = a1i * evhc / ( jtbu * lbulk )
            if ((kb == (pver + 1))) { // F:2410 if( kb .eq. pver + 1 ) then
                wstar3fact = uw_max((uw_r8(0x3ff0000000000000ul) + ((((uw_r8(0x4004000000000000ul) * cet) * n2ht) * jtzm) * dzht)), uw_r8(0x3fe0000000000000ul)); // F:2411 wstar3fact = max( 1._r8 + 2.5_r8 * cet * n2ht * jtzm * dzht, wstar3factcrit )
            } else {
                ceb = (uw_r8(0x3fc999999999999aul) / (jbbu * lbulk)); // F:2413 ceb = a1i / ( jbbu * lbulk )
                wstar3fact = uw_max(((uw_r8(0x3ff0000000000000ul) + ((((uw_r8(0x4004000000000000ul) * cet) * n2ht) * jtzm) * dzht)) + ((((uw_r8(0x4004000000000000ul) * ceb) * n2hb) * jbzm) * dzhb)), uw_r8(0x3fe0000000000000ul)); // F:2414 wstar3fact = max( 1._r8 + 2.5_r8 * cet * n2ht * jtzm * dzht + 2.5_r8 * ceb * n2hb * jbzm * dzhb, wstar3factcrit )
            }
            wstar3 = (wstar3 / wstar3fact); // F:2417 wstar3 = wstar3 / wstar3fact
        } else {
            wstar3fact = uw_r8(0x0000000000000000ul); // F:2419 wstar3fact = 0._r8
            cet = uw_r8(0x0000000000000000ul); // F:2420 cet        = 0._r8
            ceb = uw_r8(0x0000000000000000ul); // F:2421 ceb        = 0._r8
        }
        fact = ((((evhc * ((-vyt) + vut)) * dzht) + (((((-vyb) + vub) * dzhb) * leng(kb)) / leng(kt))) / lbulk); // F:2465 fact = ( evhc * ( -vyt + vut ) * dzht + ( -vyb + vub ) * dzhb * leng(i,kb) / leng(i,kt) ) / lbulk
        trma = uw_r8(0x3ff0000000000000ul); // F:2475 trma = 1._r8
        trmp = (((ebrk(ncv) * (lbrk(ncv) / lbulk)) / uw_r8(0x4008000000000000ul)) + uw_r8(0x3d719799812dea11ul)); // F:2476 trmp = ebrk(i,ncv) * ( lbrk(i,ncv) / lbulk ) / 3._r8 + ntzero
        trmq = ((uw_r8(0x4007333333333333ul) * (leng(kt) / lbulk)) * ((radf * dzht) + ((uw_r8(0x3fc999999999999aul) * fact) * wstar3))); // F:2477 trmq = 0.5_r8 * b1 * ( leng(i,kt)  / lbulk ) * ( radf * dzht + a1i * fact * wstar3 )
        rmin = uw_sqrt(trmp); // F:2483 rmin  = sqrt(trmp)
        fmin = ((rmin * ((rmin * rmin) - (uw_r8(0x4008000000000000ul) * trmp))) - (uw_r8(0x4000000000000000ul) * trmq)); // F:2484 fmin  = rmin * ( rmin * rmin - 3._r8 * trmp ) - 2._r8 * trmq
        wstar = uw_pow(wstar3, uw_r8(0x3fd5555555555555ul)); // F:2485 wstar = wstar3**onet
        rcrit = (uw_r8(0x3fe0000000000000ul) * wstar); // F:2486 rcrit = ccrit * wstar
        fcrit = ((rcrit * ((rcrit * rcrit) - (uw_r8(0x4008000000000000ul) * trmp))) - (uw_r8(0x4000000000000000ul) * trmq)); // F:2487 fcrit = rcrit * ( rcrit * rcrit - 3._r8 * trmp ) - 2._r8 * trmq
        noroot = (((rmin < rcrit) && (fcrit > uw_r8(0x0000000000000000ul))) || ((rmin >= rcrit) && (fmin > uw_r8(0x0000000000000000ul)))); // F:2499 noroot = ( ( rmin .lt. rcrit ) .and. ( fcrit .gt. 0._r8 ) ) .or. ( ( rmin .ge. rcrit ) .and. ( fmin  .gt. 0._r8 ) )
        if (noroot) { // F:2501 if( noroot ) then
            trma = (uw_r8(0x3ff0000000000000ul) - ((((uw_r8(0x4017333333333333ul) * (leng(kt) / lbulk)) * uw_r8(0x3fc999999999999aul)) * fact) / uw_r8(0x3fc0000000000000ul))); // F:2502 trma = 1._r8 - b1 * ( leng(i,kt) / lbulk ) * a1i * fact / ccrit**3
            trma = uw_max(trma, uw_r8(0x3fe0000000000000ul)); // F:2503 trma = max( trma, 0.5_r8 )
            trmp = (trmp / trma); // F:2504 trmp = trmp / trma
            trmq = ((((uw_r8(0x4007333333333333ul) * (leng(kt) / lbulk)) * radf) * dzht) / trma); // F:2505 trmq = 0.5_r8 * b1 * ( leng(i,kt) / lbulk ) * radf * dzht / trma
        }
        qq = (uw_sq(trmq) - uw_powi(trmp, 3)); // F:2510 qq = trmq**2 - trmp**3
        if ((qq >= uw_r8(0x0000000000000000ul))) { // F:2511 if( qq .ge. 0._r8 ) then
            rootp = (uw_pow((trmq + uw_sqrt(qq)), uw_r8(0x3fd5555555555555ul)) + uw_pow(uw_max((trmq - uw_sqrt(qq)), uw_r8(0x0000000000000000ul)), uw_r8(0x3fd5555555555555ul))); // F:2512 rootp = ( trmq + sqrt(qq) )**(1._r8/3._r8) + ( max( trmq - sqrt(qq), 0._r8 ) )**(1._r8/3._r8)
        } else {
            rootp = ((uw_r8(0x4000000000000000ul) * uw_sqrt(trmp)) * uw_cos((uw_acos((trmq / uw_sqrt(uw_powi(trmp, 3)))) / uw_r8(0x4008000000000000ul)))); // F:2514 rootp = 2._r8 * sqrt(trmp) * cos( acos( trmq / sqrt(trmp**3) ) / 3._r8 )
        }
        if (noroot) { // F:2520 if( noroot )  wstar3 = ( rootp / ccrit )**3
            wstar3 = uw_powi((rootp / uw_r8(0x3fe0000000000000ul)), 3); // F:2520 if( noroot )  wstar3 = ( rootp / ccrit )**3
        }
        wet = (cet * wstar3); // F:2521 wet = cet * wstar3
        if ((kb < (pver + 1))) { // F:2522 if( kb .lt. pver + 1 ) web = ceb * wstar3
            web = (ceb * wstar3); // F:2522 if( kb .lt. pver + 1 ) web = ceb * wstar3
        }
        ebrk(ncv) = uw_sq(rootp); // F:2553 ebrk(i,ncv) = rootp**2
        ebrk(ncv) = uw_min(ebrk(ncv), uw_r8(0x4034000000000000ul)); // F:2554 ebrk(i,ncv) = min(ebrk(i,ncv),tkemax)
        wbrk(ncv) = (ebrk(ncv) / uw_r8(0x4017333333333333ul)); // F:2555 wbrk(i,ncv) = ebrk(i,ncv)/b1
        if ((ebrk(ncv) <= uw_r8(0x0000000000000000ul))) { // F:2567 if( ebrk(i,ncv) .le. 0._r8 ) then
            belongcv(kt) = false; // F:2572 belongcv(i,kt) = .false.
            belongcv(kb) = false; // F:2573 belongcv(i,kb) = .false.
        }
        for (k = (kb - 1); k >= (kt + 1); k += -1) { // F:2591 do k = kb - 1, kt + 1, -1
            rcap = ((uw_r8(0x4017333333333333ul) + (wcap(k) / wbrk(ncv))) / uw_r8(0x401b333333333333ul)); // F:2592 rcap = ( b1 * ae + wcap(i,k) / wbrk(i,ncv) ) / ( b1 * ae + 1._r8 )
            rcap = uw_min(uw_max(rcap, uw_r8(0x3fb999999999999aul)), uw_r8(0x4000000000000000ul)); // F:2593 rcap = min( max(rcap,rcapmin), rcapmax )
            tke(k) = (ebrk(ncv) * rcap); // F:2594 tke(i,k) = ebrk(i,ncv) * rcap
            tke(k) = uw_min(tke(k), uw_r8(0x4034000000000000ul)); // F:2595 tke(i,k) = min( tke(i,k), tkemax )
            kvh(k) = ((leng(k) * uw_sqrt(tke(k))) * shcl(ncv)); // F:2596 kvh(i,k) = leng(i,k) * sqrt(tke(i,k)) * shcl(i,ncv)
            kvm(k) = ((leng(k) * uw_sqrt(tke(k))) * smcl(ncv)); // F:2597 kvm(i,k) = leng(i,k) * sqrt(tke(i,k)) * smcl(i,ncv)
            bprod(k) = (-(kvh(k) * n2(k))); // F:2598 bprod(i,k) = -kvh(i,k) * n2(i,k)
            sprod(k) = (kvm(k) * s2(k)); // F:2599 sprod(i,k) =  kvm(i,k) * s2(i,k)
            turbtype(k) = 2; // F:2600 turbtype(i,k) = 2
            sm_aw(k) = (smcl(ncv) / uw_r8(0x3fe1cc63f141205cul)); // F:2601 sm_aw(i,k) = smcl(i,ncv)/alph1
        }
        kentr = (wet * jtzm); // F:2605 kentr = wet * jtzm
        kvh(kt) = kentr; // F:2606 kvh(i,kt) = kentr
        kvm(kt) = kentr; // F:2607 kvm(i,kt) = kentr
        bprod(kt) = ((-(kentr * n2ht)) + radf); // F:2608 bprod(i,kt) = -kentr * n2ht + radf
        sprod(kt) = (kentr * s2(kt)); // F:2609 sprod(i,kt) =  kentr * s2(i,kt)
        turbtype(kt) = 4; // F:2610 turbtype(i,kt) = 4
        trmp = uw_r8(0xbfeb4b4b4b4b4b4bul); // F:2611 trmp = -b1 * ae / ( 1._r8 + b1 * ae )
        trmq = (-(((((bprod(kt) + sprod(kt)) * uw_r8(0x4017333333333333ul)) * leng(kt)) / uw_r8(0x401b333333333333ul)) / uw_pow(ebrk(ncv), uw_r8(0x3ff8000000000000ul)))); // F:2612 trmq = -(bprod(i,kt)+sprod(i,kt))*b1*leng(i,kt)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
        rcap = uw_pow(uw_compute_cubic(uw_r8(0x0000000000000000ul), trmp, trmq), uw_r8(0x4000000000000000ul)); // F:2613 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
        rcap = uw_min(uw_max(rcap, uw_r8(0x3fb999999999999aul)), uw_r8(0x4000000000000000ul)); // F:2614 rcap = min( max(rcap,rcapmin), rcapmax )
        tke(kt) = (ebrk(ncv) * rcap); // F:2615 tke(i,kt)  = ebrk(i,ncv) * rcap
        tke(kt) = uw_min(tke(kt), uw_r8(0x4034000000000000ul)); // F:2616 tke(i,kt)  = min( tke(i,kt), tkemax )
        sm_aw(kt) = (smcl(ncv) / uw_r8(0x3fe1cc63f141205cul)); // F:2617 sm_aw(i,kt) = smcl(i,ncv) / alph1
        if ((kb < (pver + 1))) { // F:2625 if( kb .lt. pver + 1 ) then
            kentr = (web * jbzm); // F:2627 kentr = web * jbzm
            if ((kb != ktblw)) { // F:2629 if( kb .ne. ktblw ) then
                kvh(kb) = kentr; // F:2631 kvh(i,kb) = kentr
                kvm(kb) = kentr; // F:2632 kvm(i,kb) = kentr
                bprod(kb) = (-(kvh(kb) * n2hb)); // F:2633 bprod(i,kb) = -kvh(i,kb)*n2hb
                sprod(kb) = (kvm(kb) * s2(kb)); // F:2634 sprod(i,kb) =  kvm(i,kb)*s2(i,kb)
                turbtype(kb) = 3; // F:2635 turbtype(i,kb) = 3
                trmp = uw_r8(0xbfeb4b4b4b4b4b4bul); // F:2636 trmp = -b1*ae/(1._r8+b1*ae)
                trmq = (-(((((bprod(kb) + sprod(kb)) * uw_r8(0x4017333333333333ul)) * leng(kb)) / uw_r8(0x401b333333333333ul)) / uw_pow(ebrk(ncv), uw_r8(0x3ff8000000000000ul)))); // F:2637 trmq = -(bprod(i,kb)+sprod(i,kb))*b1*leng(i,kb)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
                rcap = uw_pow(uw_compute_cubic(uw_r8(0x0000000000000000ul), trmp, trmq), uw_r8(0x4000000000000000ul)); // F:2638 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
                rcap = uw_min(uw_max(rcap, uw_r8(0x3fb999999999999aul)), uw_r8(0x4000000000000000ul)); // F:2639 rcap = min( max(rcap,rcapmin), rcapmax )
                tke(kb) = (ebrk(ncv) * rcap); // F:2640 tke(i,kb)  = ebrk(i,ncv) * rcap
                tke(kb) = uw_min(tke(kb), uw_r8(0x4034000000000000ul)); // F:2641 tke(i,kb)  = min( tke(i,kb),tkemax )
            } else {
                kvh(kb) = (kvh(kb) + kentr); // F:2645 kvh(i,kb) = kvh(i,kb) + kentr
                kvm(kb) = (kvm(kb) + kentr); // F:2646 kvm(i,kb) = kvm(i,kb) + kentr
                dzhb5 = (z((kb - 1)) - zi(kb)); // F:2649 dzhb5 = z(i,kb-1) - zi(i,kb)
                dzht5 = (zi(kb) - z(kb)); // F:2650 dzht5 = zi(i,kb) - z(i,kb)
                bprod(kb) = (((dzht5 * bprod(kb)) - ((dzhb5 * kentr) * n2hb)) / (dzhb5 + dzht5)); // F:2651 bprod(i,kb) = ( dzht5*bprod(i,kb) - dzhb5*kentr*n2hb )     / ( dzhb5 + dzht5 )
                sprod(kb) = (((dzht5 * sprod(kb)) + ((dzhb5 * kentr) * s2(kb))) / (dzhb5 + dzht5)); // F:2652 sprod(i,kb) = ( dzht5*sprod(i,kb) + dzhb5*kentr*s2(i,kb) ) / ( dzhb5 + dzht5 )
                trmp = uw_r8(0xbfeb4b4b4b4b4b4bul); // F:2653 trmp = -b1*ae/(1._r8+b1*ae)
                trmq = (-(((((kentr * (s2(kb) - n2hb)) * uw_r8(0x4017333333333333ul)) * leng(kb)) / uw_r8(0x401b333333333333ul)) / uw_pow(ebrk(ncv), uw_r8(0x3ff8000000000000ul)))); // F:2654 trmq = -kentr*(s2(i,kb)-n2hb)*b1*leng(i,kb)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
                rcap = uw_pow(uw_compute_cubic(uw_r8(0x0000000000000000ul), trmp, trmq), uw_r8(0x4000000000000000ul)); // F:2655 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
                rcap = uw_min(uw_max(rcap, uw_r8(0x3fb999999999999aul)), uw_r8(0x4000000000000000ul)); // F:2656 rcap = min( max(rcap,rcapmin), rcapmax )
                tke_imsi = (ebrk(ncv) * rcap); // F:2657 tke_imsi = ebrk(i,ncv) * rcap
                tke_imsi = uw_min(tke_imsi, uw_r8(0x4034000000000000ul)); // F:2658 tke_imsi = min( tke_imsi, tkemax )
                tke(kb) = (((dzht5 * tke(kb)) + (dzhb5 * tke_imsi)) / (dzhb5 + dzht5)); // F:2659 tke(i,kb)  = ( dzht5*tke(i,kb) + dzhb5*tke_imsi ) / ( dzhb5 + dzht5 )
                tke(kb) = uw_min(tke(kb), uw_r8(0x4034000000000000ul)); // F:2660 tke(i,kb)  = min(tke(i,kb),tkemax)
                turbtype(kb) = 5; // F:2661 turbtype(i,kb) = 5
            }
        } else {
            rcap = ((uw_r8(0x4017333333333333ul) + (wcap(kb) / wbrk(ncv))) / uw_r8(0x401b333333333333ul)); // F:2671 rcap = (b1*ae + wcap(i,kb)/wbrk(i,ncv))/(b1*ae + 1._r8)
            rcap = uw_min(uw_max(rcap, uw_r8(0x3fb999999999999aul)), uw_r8(0x4000000000000000ul)); // F:2672 rcap = min( max(rcap,rcapmin), rcapmax )
            tke(kb) = (ebrk(ncv) * rcap); // F:2673 tke(i,kb) = ebrk(i,ncv) * rcap
            tke(kb) = uw_min(tke(kb), uw_r8(0x4034000000000000ul)); // F:2674 tke(i,kb) = min( tke(i,kb),tkemax )
        }
        sm_aw(kb) = (smcl(ncv) / uw_r8(0x3fe1cc63f141205cul)); // F:2682 sm_aw(i,kb) = smcl(i,ncv)/alph1
        wcap(kt) = (((bprod(kt) + sprod(kt)) * leng(kt)) / uw_sqrt(uw_max(tke(kt), uw_r8(0x3eb0c6f7a0b5ed8dul)))); // F:2691 wcap(i,kt) = (bprod(i,kt)+sprod(i,kt))*leng(i,kt)/sqrt(max(tke(i,kt),1.e-6_r8))
        if ((kb < (pver + 1))) { // F:2692 if( kb .lt. pver + 1 ) then
            wcap(kb) = (((bprod(kb) + sprod(kb)) * leng(kb)) / uw_sqrt(uw_max(tke(kb), uw_r8(0x3eb0c6f7a0b5ed8dul)))); // F:2693 wcap(i,kb) = (bprod(i,kb)+sprod(i,kb))*leng(i,kb)/sqrt(max(tke(i,kb),1.e-6_r8))
        }
        ktblw = kt; // F:2700 ktblw = kt
        wet_CL(ncv) = wet; // F:2704 wet_CL(i,ncv)        = wet
        web_CL(ncv) = web; // F:2705 web_CL(i,ncv)        = web
        jtbu_CL(ncv) = jtbu; // F:2706 jtbu_CL(i,ncv)       = jtbu
        jbbu_CL(ncv) = jbbu; // F:2707 jbbu_CL(i,ncv)       = jbbu
        evhc_CL(ncv) = evhc; // F:2708 evhc_CL(i,ncv)       = evhc
        jt2slv_CL(ncv) = jt2slv; // F:2709 jt2slv_CL(i,ncv)     = jt2slv
        n2ht_CL(ncv) = n2ht; // F:2710 n2ht_CL(i,ncv)       = n2ht
        n2hb_CL(ncv) = n2hb; // F:2711 n2hb_CL(i,ncv)       = n2hb
        lwp_CL(ncv) = lwp; // F:2712 lwp_CL(i,ncv)        = lwp
        opt_depth_CL(ncv) = opt_depth; // F:2713 opt_depth_CL(i,ncv)  = opt_depth
        radinvfrac_CL(ncv) = radinvfrac; // F:2714 radinvfrac_CL(i,ncv) = radinvfrac
        radf_CL(ncv) = radf; // F:2715 radf_CL(i,ncv)       = radf
        wstar_CL(ncv) = wstar; // F:2716 wstar_CL(i,ncv)      = wstar
        wstar3fact_CL(ncv) = wstar3fact; // F:2717 wstar3fact_CL(i,ncv) = wstar3fact
    }
    if ((ncvsurf > 0)) { // F:2728 if( ncvsurf .gt. 0 ) then
        ktopbl = ktop(ncvsurf); // F:2730 ktopbl(i) = ktop(i,ncvsurf)
        pblh = zi(ktopbl); // F:2731 pblh(i)   = zi(i, ktopbl(i))
        pblhp = pi(ktopbl); // F:2732 pblhp(i)  = pi(i, ktopbl(i))
        wpert = uw_max((uw_r8(0x3ff0000000000000ul) * uw_sqrt(ebrk(ncvsurf))), uw_r8(0x3eb0c6f7a0b5ed8dul)); // F:2733 wpert(i)  = max(wfac*sqrt(ebrk(i,ncvsurf)),wpertmin)
        tpert = uw_max(((uw_abs(((shflx * rrho) / uw_cpair)) * uw_r8(0x3ff0000000000000ul)) / wpert), uw_r8(0x0000000000000000ul)); // F:2734 tpert(i)  = max(abs(shflx(i)*rrho(i)/cpair)*tfac/wpert(i),0._r8)
        qpert = uw_max(((uw_abs((qflx * rrho)) * uw_r8(0x3ff0000000000000ul)) / wpert), uw_r8(0x0000000000000000ul)); // F:2735 qpert(i)  = max(abs(qflx(i)*rrho(i))*tfac/wpert(i),0._r8)
        if ((bflxs > uw_r8(0x0000000000000000ul))) { // F:2737 if( bflxs(i) .gt. 0._r8 ) then
            turbtype((pver + 1)) = 2; // F:2738 turbtype(i,pver+1) = 2
        } else {
            turbtype((pver + 1)) = 3; // F:2740 turbtype(i,pver+1) = 3
        }
        ipbl = uw_r8(0x3ff0000000000000ul); // F:2743 ipbl(i)  = 1._r8
        kpblh = (ktopbl - uw_r8(0x3ff0000000000000ul)); // F:2744 kpblh(i) = ktopbl(i) - 1._r8
    }
    belongst(1) = false; // F:2755 belongst(i,1) = .false.
    for (k = 2; k <= pver; k += 1) { // F:2756 do k = 2, pver
        belongst(k) = ((ri(k) < uw_r8(0x3fc851eb851eb852ul)) && (!belongcv(k))); // F:2757 belongst(i,k) = ( ri(i,k) .lt. ricrit ) .and. ( .not. belongcv(i,k) )
        if ((belongst(k) && (!belongst((k - 1))))) { // F:2758 if( belongst(i,k) .and. ( .not. belongst(i,k-1) ) ) then
            kt = k; // F:2759 kt = k
        } else if (((!belongst(k)) && belongst((k - 1)))) { // F:2760 elseif( .not. belongst(i,k) .and. belongst(i,k-1) ) then
            kb = (k - 1); // F:2761 kb = k - 1
            lbulk = (z((kt - 1)) - z(kb)); // F:2762 lbulk = z(i,kt-1) - z(i,kb)
            for (ks = kt; ks <= kb; ks += 1) { // F:2763 do ks = kt, kb
                tunlramp = uw_r8(0x3fb5c28f5c28f5c3ul); // F:2765 tunlramp = tunl
                leng(ks) = uw_pow((uw_pow((uw_vk * zi(ks)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:2773 leng(i,ks) = ( (vk*zi(i,ks))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            }
        }
    }
    belongst((pver + 1)) = (!belongcv((pver + 1))); // F:2788 belongst(i,pver+1) = .not. belongcv(i,pver+1)
    if (belongst((pver + 1))) { // F:2790 if( belongst(i,pver+1) ) then
        turbtype((pver + 1)) = 1; // F:2792 turbtype(i,pver+1) = 1
        if (belongst(pver)) { // F:2794 if( belongst(i,pver) ) then
            lbulk = z((kt - 1)); // F:2796 lbulk = z(i,kt-1)
        } else {
            kt = (pver + 1); // F:2798 kt = pver+1
            lbulk = z((kt - 1)); // F:2799 lbulk = z(i,kt-1)
        }
        ktopbl = (kt - 1); // F:2807 ktopbl(i) = kt - 1
        pblh = z(ktopbl); // F:2808 pblh(i)   = z(i,ktopbl(i))
        pblhp = (uw_r8(0x3fe0000000000000ul) * (pi(ktopbl) + pi((ktopbl + 1)))); // F:2809 pblhp(i)  = 0.5_r8 * ( pi(i,ktopbl(i)) + pi(i,ktopbl(i)+1) )
        for (ks = kt; ks <= pver; ks += 1) { // F:2814 do ks = kt, pver
            tunlramp = uw_r8(0x3fb5c28f5c28f5c3ul); // F:2816 tunlramp = tunl
            leng(ks) = uw_pow((uw_pow((uw_vk * zi(ks)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:2824 leng(i,ks) = ( (vk*zi(i,ks))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
        }
        wpert = uw_r8(0x0000000000000000ul); // F:2834 wpert(i) = 0._r8
        tpert = uw_max(((((shflx * rrho) / uw_cpair) * uw_r8(0x4021000000000000ul)) / ustar), uw_r8(0x0000000000000000ul)); // F:2835 tpert(i) = max(shflx(i)*rrho(i)/cpair*fak/ustar(i),0._r8)
        qpert = uw_max((((qflx * rrho) * uw_r8(0x4021000000000000ul)) / ustar), uw_r8(0x0000000000000000ul)); // F:2836 qpert(i) = max(qflx(i)*rrho(i)*fak/ustar(i),0._r8)
        ipbl = uw_r8(0x0000000000000000ul); // F:2838 ipbl(i)  = 0._r8
        kpblh = ktopbl; // F:2839 kpblh(i) = ktopbl(i)
    }
    for (k = 2; k <= pver; k += 1) { // F:2850 do k = 2, pver
        if (belongst(k)) { // F:2852 if( belongst(i,k) ) then
            turbtype(k) = 1; // F:2854 turbtype(i,k) = 1
            trma = (((uw_r8(0xc0415694467381d8ul) * alph4exs) * ri(k)) + (uw_r8(0x4027333333333333ul) * (uw_r8(0xc01174bc6a7ef9dbul) - ((alph4exs * uw_r8(0x3fe65aee631f8a09ul)) * ri(k))))); // F:2855 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
            trmb = (((uw_r8(0xc0415694467381d8ul) + alph4exs) * ri(k)) + (uw_r8(0x4027333333333333ul) * ((-(uw_r8(0x3fe65aee631f8a09ul) * ri(k))) + uw_r8(0x3fe1cc63f141205cul)))); // F:2856 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri(k); // F:2857 trmc = ri(i,k)
            det = uw_max(((trmb * trmb) - ((uw_r8(0x4010000000000000ul) * trma) * trmc)), uw_r8(0x0000000000000000ul)); // F:2858 det = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            if ((det < uw_r8(0x0000000000000000ul))) { // F:2860 if( det .lt. 0._r8 ) then
                atomic_store_explicit(ws.err, 2865, memory_order_relaxed); return; // F:2865 stop
            }
            gh = (((-trmb) + uw_sqrt(det)) / (uw_r8(0x4000000000000000ul) * trma)); // F:2867 gh = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_min(uw_max(gh, ghmin), uw_r8(0x3f97dbf487fcb924ul)); // F:2870 gh = min(max(gh,ghmin),0.0233_r8)
            sh = uw_max(uw_r8(0x0000000000000000ul), (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh)))); // F:2871 sh = max(0._r8,alph5/(1._r8+alph3*gh))
            sm = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (alph4exs * gh)))); // F:2872 sm = max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
            tke(k) = ((uw_r8(0x4017333333333333ul) * uw_sq(leng(k))) * ((-(sh * n2(k))) + (sm * s2(k)))); // F:2874 tke(i,k)   = b1*(leng(i,k)**2)*(-sh*n2(i,k)+sm*s2(i,k))
            tke(k) = uw_min(tke(k), uw_r8(0x4034000000000000ul)); // F:2875 tke(i,k)   = min(tke(i,k),tkemax)
            wcap(k) = (tke(k) / uw_r8(0x4017333333333333ul)); // F:2876 wcap(i,k)  = tke(i,k)/b1
            kvh(k) = ((leng(k) * uw_sqrt(tke(k))) * sh); // F:2877 kvh(i,k)   = leng(i,k) * sqrt(tke(i,k)) * sh
            kvm(k) = ((leng(k) * uw_sqrt(tke(k))) * sm); // F:2878 kvm(i,k)   = leng(i,k) * sqrt(tke(i,k)) * sm
            bprod(k) = (-(kvh(k) * n2(k))); // F:2879 bprod(i,k) = -kvh(i,k) * n2(i,k)
            sprod(k) = (kvm(k) * s2(k)); // F:2880 sprod(i,k) =  kvm(i,k) * s2(i,k)
            sm_aw(k) = (sm / uw_r8(0x3fe1cc63f141205cul)); // F:2882 sm_aw(i,k) = sm/alph1
        }
    }
    for (k = 2; k <= pver; k += 1) { // F:2904 do k = 2, pver
        if ((((turbtype(k) == 3) || (turbtype(k) == 4)) || (turbtype(k) == 5))) { // F:2906 if( ( turbtype(i,k) .eq. 3 ) .or. ( turbtype(i,k) .eq. 4 ) .or. ( turbtype(i,k) .eq. 5 ) ) then
            trma = (((uw_r8(0xc0415694467381d8ul) * alph4exs) * ri(k)) + (uw_r8(0x4027333333333333ul) * (uw_r8(0xc01174bc6a7ef9dbul) - ((alph4exs * uw_r8(0x3fe65aee631f8a09ul)) * ri(k))))); // F:2909 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
            trmb = (((uw_r8(0xc0415694467381d8ul) + alph4exs) * ri(k)) + (uw_r8(0x4027333333333333ul) * ((-(uw_r8(0x3fe65aee631f8a09ul) * ri(k))) + uw_r8(0x3fe1cc63f141205cul)))); // F:2910 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri(k); // F:2911 trmc = ri(i,k)
            det = uw_max(((trmb * trmb) - ((uw_r8(0x4010000000000000ul) * trma) * trmc)), uw_r8(0x0000000000000000ul)); // F:2912 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = (((-trmb) + uw_sqrt(det)) / (uw_r8(0x4000000000000000ul) * trma)); // F:2913 gh   = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_min(uw_max(gh, ghmin), uw_r8(0x3f97dbf487fcb924ul)); // F:2916 gh   = min(max(gh,ghmin),0.0233_r8)
            sh = uw_max(uw_r8(0x0000000000000000ul), (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh)))); // F:2917 sh   = max(0._r8,alph5/(1._r8+alph3*gh))
            sm = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (alph4exs * gh)))); // F:2918 sm   = max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
            lbulk = (z((k - 1)) - z(k)); // F:2920 lbulk = z(i,k-1) - z(i,k)
            tunlramp = uw_r8(0x3fb5c28f5c28f5c3ul); // F:2923 tunlramp = tunl
            leng_imsi = uw_pow((uw_pow((uw_vk * zi(k)), uw_r8(0xc008000000000000ul)) + uw_pow((tunlramp * lbulk), uw_r8(0xc008000000000000ul))), uw_r8(0xbfd5555555555555ul)); // F:2931 leng_imsi = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            tke_imsi = ((uw_r8(0x4017333333333333ul) * uw_sq(leng_imsi)) * ((-(sh * n2(k))) + (sm * s2(k)))); // F:2937 tke_imsi = b1*(leng_imsi**2)*(-sh*n2(i,k)+sm*s2(i,k))
            tke_imsi = uw_min(uw_max(tke_imsi, uw_r8(0x0000000000000000ul)), uw_r8(0x4034000000000000ul)); // F:2938 tke_imsi = min(max(tke_imsi,0._r8),tkemax)
            kvh_imsi = ((leng_imsi * uw_sqrt(tke_imsi)) * sh); // F:2939 kvh_imsi = leng_imsi * sqrt(tke_imsi) * sh
            kvm_imsi = ((leng_imsi * uw_sqrt(tke_imsi)) * sm); // F:2940 kvm_imsi = leng_imsi * sqrt(tke_imsi) * sm
            if ((kvh(k) < kvh_imsi)) { // F:2942 if( kvh(i,k) .lt. kvh_imsi ) then
                kvh(k) = kvh_imsi; // F:2943 kvh(i,k)   =  kvh_imsi
                kvm(k) = kvm_imsi; // F:2944 kvm(i,k)   =  kvm_imsi
                leng(k) = leng_imsi; // F:2945 leng(i,k)  = leng_imsi
                tke(k) = tke_imsi; // F:2946 tke(i,k)   =  tke_imsi
                wcap(k) = (tke_imsi / uw_r8(0x4017333333333333ul)); // F:2947 wcap(i,k)  =  tke_imsi / b1
                bprod(k) = (-(kvh_imsi * n2(k))); // F:2948 bprod(i,k) = -kvh_imsi * n2(i,k)
                sprod(k) = (kvm_imsi * s2(k)); // F:2949 sprod(i,k) =  kvm_imsi * s2(i,k)
                sm_aw(k) = (sm / uw_r8(0x3fe1cc63f141205cul)); // F:2950 sm_aw(i,k) =  sm/alph1
                turbtype(k) = 1; // F:2951 turbtype(i,k) = 1
            }
        }
    }
    bprod((pver + 1)) = bflxs; // F:2987 bprod(i,pver+1) = bflxs(i)
    gg = ((((uw_r8(0x3fe0000000000000ul) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, uw_r8(0x3ff8000000000000ul))); // F:2989 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
    if ((uw_abs((uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))) <= uw_r8(0x3e7ad7f29abcaf48ul))) { // F:2990 if( abs(alph5-gg*alph3) .le. 1.e-7_r8 ) then
        if ((bprod((pver + 1)) > uw_r8(0x0000000000000000ul))) { // F:2992 if( bprod(i,pver+1) .gt. 0._r8 ) then
            gh = uw_r8(0xc00c4467381d7dbful); // F:2993 gh = -3.5334_r8
        } else {
            gh = ghmin; // F:2995 gh = ghmin
        }
    } else {
        gh = (gg / (uw_r8(0x3fe65aee631f8a09ul) - (gg * uw_r8(0xc0415694467381d8ul)))); // F:2998 gh = gg/(alph5-gg*alph3)
    }
    if ((bprod((pver + 1)) > uw_r8(0x0000000000000000ul))) { // F:3002 if( bprod(i,pver+1) .gt. 0._r8 ) then
        gh = uw_min(uw_max(gh, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3003 gh = min(max(gh,-3.5334_r8),0.0233_r8)
    } else {
        gh = uw_min(uw_max(gh, ghmin), uw_r8(0x3f97dbf487fcb924ul)); // F:3005 gh = min(max(gh,ghmin),0.0233_r8)
    }
    gh_a((pver + 1)) = gh; // F:3008 gh_a(i,pver+1) = gh
    sh_a((pver + 1)) = uw_max(uw_r8(0x0000000000000000ul), (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh)))); // F:3009 sh_a(i,pver+1) = max(0._r8,alph5/(1._r8+alph3*gh))
    if ((bprod((pver + 1)) > uw_r8(0x0000000000000000ul))) { // F:3010 if( bprod(i,pver+1) .gt. 0._r8 ) then
        sm_a((pver + 1)) = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh)))); // F:3011 sm_a(i,pver+1) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh))
    } else {
        sm_a((pver + 1)) = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (alph4exs * gh)))); // F:3013 sm_a(i,pver+1) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
    }
    sm_aw((pver + 1)) = (sm_a((pver + 1)) / uw_r8(0x3fe1cc63f141205cul)); // F:3015 sm_aw(i,pver+1) = sm_a(i,pver+1)/alph1
    ri_a((pver + 1)) = (-((sm_a((pver + 1)) / sh_a((pver + 1))) * (bprod((pver + 1)) / sprod((pver + 1))))); // F:3016 ri_a(i,pver+1)  = -(sm_a(i,pver+1)/sh_a(i,pver+1))*(bprod(i,pver+1)/sprod(i,pver+1))
    for (k = 1; k <= pver; k += 1) { // F:3018 do k = 1, pver
        if ((ri(k) < uw_r8(0x0000000000000000ul))) { // F:3019 if( ri(i,k) .lt. 0._r8 ) then
            trma = ((uw_r8(0x406a8f03ff93f46aul) * ri(k)) + (uw_r8(0x4027333333333333ul) * (uw_r8(0xc01174bc6a7ef9dbul) - (uw_r8(0xc0111f3168d8b188ul) * ri(k))))); // F:3020 trma = alph3*alph4*ri(i,k) + 2._r8*b1*(alph2-alph4*alph5*ri(i,k))
            trmb = ((uw_r8(0xc04466dc5d638866ul) * ri(k)) + (uw_r8(0x4027333333333333ul) * ((-(uw_r8(0x3fe65aee631f8a09ul) * ri(k))) + uw_r8(0x3fe1cc63f141205cul)))); // F:3021 trmb = (alph3+alph4)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri(k); // F:3022 trmc = ri(i,k)
            det = uw_max(((trmb * trmb) - ((uw_r8(0x4010000000000000ul) * trma) * trmc)), uw_r8(0x0000000000000000ul)); // F:3023 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = (((-trmb) + uw_sqrt(det)) / (uw_r8(0x4000000000000000ul) * trma)); // F:3024 gh   = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_min(uw_max(gh, uw_r8(0xc00c4467381d7dbful)), uw_r8(0x3f97dbf487fcb924ul)); // F:3025 gh   = min(max(gh,-3.5334_r8),0.0233_r8)
            gh_a(k) = gh; // F:3026 gh_a(i,k) = gh
            sh_a(k) = uw_max(uw_r8(0x0000000000000000ul), (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh)))); // F:3027 sh_a(i,k) = max(0._r8,alph5/(1._r8+alph3*gh))
            sm_a(k) = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0188240b780346eul) * gh)))); // F:3028 sm_a(i,k) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh))
            ri_a(k) = ri(k); // F:3029 ri_a(i,k) = ri(i,k)
        } else {
            if ((ri(k) > uw_r8(0x3fc851eb851eb852ul))) { // F:3031 if( ri(i,k) .gt. ricrit ) then
                gh_a(k) = ghmin; // F:3032 gh_a(i,k) = ghmin
                sh_a(k) = uw_r8(0x0000000000000000ul); // F:3033 sh_a(i,k) = 0._r8
                sm_a(k) = uw_r8(0x0000000000000000ul); // F:3034 sm_a(i,k) = 0._r8
                ri_a(k) = ri(k); // F:3035 ri_a(i,k) = ri(i,k)
            } else {
                trma = (((uw_r8(0xc0415694467381d8ul) * alph4exs) * ri(k)) + (uw_r8(0x4027333333333333ul) * (uw_r8(0xc01174bc6a7ef9dbul) - ((alph4exs * uw_r8(0x3fe65aee631f8a09ul)) * ri(k))))); // F:3037 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
                trmb = (((uw_r8(0xc0415694467381d8ul) + alph4exs) * ri(k)) + (uw_r8(0x4027333333333333ul) * ((-(uw_r8(0x3fe65aee631f8a09ul) * ri(k))) + uw_r8(0x3fe1cc63f141205cul)))); // F:3038 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
                trmc = ri(k); // F:3039 trmc = ri(i,k)
                det = uw_max(((trmb * trmb) - ((uw_r8(0x4010000000000000ul) * trma) * trmc)), uw_r8(0x0000000000000000ul)); // F:3040 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
                gh = (((-trmb) + uw_sqrt(det)) / (uw_r8(0x4000000000000000ul) * trma)); // F:3041 gh   = (-trmb + sqrt(det))/(2._r8*trma)
                gh = uw_min(uw_max(gh, ghmin), uw_r8(0x3f97dbf487fcb924ul)); // F:3042 gh   = min(max(gh,ghmin),0.0233_r8)
                gh_a(k) = gh; // F:3043 gh_a(i,k) = gh
                sh_a(k) = uw_max(uw_r8(0x0000000000000000ul), (uw_r8(0x3fe65aee631f8a09ul) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh)))); // F:3044 sh_a(i,k) = max(0._r8,alph5/(1._r8+alph3*gh))
                sm_a(k) = uw_max(uw_r8(0x0000000000000000ul), (((uw_r8(0x3fe1cc63f141205cul) + (uw_r8(0xc01174bc6a7ef9dbul) * gh)) / (uw_r8(0x3ff0000000000000ul) + (uw_r8(0xc0415694467381d8ul) * gh))) / (uw_r8(0x3ff0000000000000ul) + (alph4exs * gh)))); // F:3045 sm_a(i,k) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
                ri_a(k) = ri(k); // F:3046 ri_a(i,k) = ri(i,k)
            }
        }
    }
    for (k = 1; k <= (pver + 1); k += 1) { // F:3052 do k = 1, pver + 1
        turbtype_f(k) = uw_widen((float)(turbtype(k))); // F:3053 turbtype_f(i,k) = real(turbtype(i,k))
    }
    return; // F:3058 return
}
#endif

// WRF 4.7.1 eddy_diff, configured sftype=l, choice_evhc=choice_radf=maxi.
// Concatenate after common, saturation, vdiff, and caleddy. No includes.
// The column driver is in uwpbl_driver.cuh, concatenated after this header.
#ifndef UWPBL_EDDY_CUH
#define UWPBL_EDDY_CUH

static inline void uw_sfdiag(int pver, V qt, V ql, V sl, V pi, V pm, V zi,
    V cld, V sfi, V sfuh, V sflh, V slslope, V qtslope) {
    for (int k=1; k<=pver+1; ++k) sfi(k)=uw_r8(0x0000000000000000ul);
    for (int k=1; k<=pver; ++k) { sfuh(k)=uw_r8(0x0000000000000000ul); sflh(k)=uw_r8(0x0000000000000000ul); }
    // The maxi branch overwrites the preceding ql<qmin test unconditionally.
    for (int k=2; k<=pver; ++k) {
        sfuh(k)=cld(k); sflh(k)=cld(k);
        sfi(k)=uw_r8(0x3fe0000000000000ul)*(sflh(k-1)+uw_min(sfuh(k),sflh(k-1)));
    }
    sfi(pver+1)=sflh(pver);
}

static inline void uw_trbintd(int pver, V z, V u, V v, V t, V pmid,
    R8 taux, R8 tauy, thread R8& ustar, thread R8& rrho, V s2, V n2, V ri, V zi, V pi,
    V cld, V qt, V qv, V ql, V qi, V sfi, V sfuh, V sflh, V sl, V slv,
    V slslope, V qtslope, V chs, V chu, V cms, V cmu, thread R8& minpblh) {
    R8 latsub=R8(UW_LATVAP)+R8(UW_LATICE);
    rrho=R8(UW_RAIR)*t(pver)/pmid(pver);
    ustar=uw_max(uw_sqrt(uw_sqrt(uw_sq(taux)+uw_sq(tauy))*rrho),uw_r8(0x3f847ae147ae147bul));
    minpblh=uw_r8(0x4059000000000000ul)*ustar;
    for (int k=1; k<=pver; ++k) {
        R8 es,qs,gam;
        int status=uw_fqsatd(t(k),pmid(k),es,qs,gam);
        qt(k)=qv(k)+ql(k)+qi(k);
        sl(k)=R8(UW_CPAIR)*t(k)+R8(UW_GRAVIT)*z(k)-R8(UW_LATVAP)*ql(k)-latsub*qi(k);
        slv(k)=sl(k)*(uw_r8(0x3ff0000000000000ul)+R8(UW_ZVIR)*qt(k));
        R8 bfact=R8(UW_GRAVIT)/(t(k)*(uw_r8(0x3ff0000000000000ul)+R8(UW_ZVIR)*qv(k)-ql(k)-qi(k)));
        chu(k)=(uw_r8(0x3ff0000000000000ul)+R8(UW_ZVIR)*qt(k))*bfact/R8(UW_CPAIR);
        chs(k)=((uw_r8(0x3ff0000000000000ul)+(uw_r8(0x3ff0000000000000ul)+R8(UW_ZVIR))*gam*R8(UW_CPAIR)*t(k)/R8(UW_LATVAP))/(uw_r8(0x3ff0000000000000ul)+gam))*bfact/R8(UW_CPAIR);
        cmu(k)=R8(UW_ZVIR)*bfact*t(k);
        cms(k)=R8(UW_LATVAP)*chs(k)-bfact*t(k);
    }
    chu(pver+1)=chu(pver); chs(pver+1)=chs(pver);
    cmu(pver+1)=cmu(pver); cms(pver+1)=cms(pver);
    slslope(pver)=(sl(pver)-sl(pver-1))/(pmid(pver)-pmid(pver-1));
    qtslope(pver)=(qt(pver)-qt(pver-1))/(pmid(pver)-pmid(pver-1));
    slslope(1)=(sl(2)-sl(1))/(pmid(2)-pmid(1));
    qtslope(1)=(qt(2)-qt(1))/(pmid(2)-pmid(1));
    R8 dsldp_b=slslope(1),dqtdp_b=qtslope(1);
    for (int k=2; k<=pver-1; ++k) {
        R8 dsldp_a=dsldp_b,dqtdp_a=dqtdp_b;
        dsldp_b=(sl(k+1)-sl(k))/(pmid(k+1)-pmid(k));
        dqtdp_b=(qt(k+1)-qt(k))/(pmid(k+1)-pmid(k));
        R8 product=dsldp_a*dsldp_b;
        if (product<=uw_r8(0x0000000000000000ul)) slslope(k)=uw_r8(0x0000000000000000ul);
        else if (product>uw_r8(0x0000000000000000ul) && dsldp_a<uw_r8(0x0000000000000000ul)) slslope(k)=uw_max(dsldp_a,dsldp_b);
        else if (product>uw_r8(0x0000000000000000ul) && dsldp_a>uw_r8(0x0000000000000000ul)) slslope(k)=uw_min(dsldp_a,dsldp_b);
        product=dqtdp_a*dqtdp_b;
        if (product<=uw_r8(0x0000000000000000ul)) qtslope(k)=uw_r8(0x0000000000000000ul);
        else if (product>uw_r8(0x0000000000000000ul) && dqtdp_a<uw_r8(0x0000000000000000ul)) qtslope(k)=uw_max(dqtdp_a,dqtdp_b);
        else if (product>uw_r8(0x0000000000000000ul) && dqtdp_a>uw_r8(0x0000000000000000ul)) qtslope(k)=uw_min(dqtdp_a,dqtdp_b);
    }
    uw_sfdiag(pver,qt,ql,sl,pi,pmid,zi,cld,sfi,sfuh,sflh,slslope,qtslope);
    for (int k=pver; k>=2; --k) {
        int km1=k-1;
        R8 rdz=uw_r8(0x3ff0000000000000ul)/(z(km1)-z(k));
        R8 dsldz=(sl(km1)-sl(k))*rdz,dqtdz=(qt(km1)-qt(k))*rdz;
        chu(k)=(chu(km1)+chu(k))*uw_r8(0x3fe0000000000000ul);
        chs(k)=(chs(km1)+chs(k))*uw_r8(0x3fe0000000000000ul);
        cmu(k)=(cmu(km1)+cmu(k))*uw_r8(0x3fe0000000000000ul);
        cms(k)=(cms(km1)+cms(k))*uw_r8(0x3fe0000000000000ul);
        R8 ch=chu(k)*(uw_r8(0x3ff0000000000000ul)-sfi(k))+chs(k)*sfi(k);
        R8 cm=cmu(k)*(uw_r8(0x3ff0000000000000ul)-sfi(k))+cms(k)*sfi(k);
        n2(k)=ch*dsldz+cm*dqtdz;
        s2(k)=(uw_sq(u(km1)-u(k))+uw_sq(v(km1)-v(k)))*uw_sq(rdz);
        s2(k)=uw_max(uw_r8(0x3d719799812dea11ul),s2(k));
        ri(k)=n2(k)/s2(k);
    }
    n2(1)=n2(2); s2(1)=s2(2); ri(1)=ri(2);
}

static inline void uw_eddy_retrieve(int pver, V slfd, V qtfd, V qi, V z,
    V pmid, V tfd, V qvfd, V qlfd) {
    R8 latsub=R8(UW_LATVAP)+R8(UW_LATICE);
    for (int k=1; k<=pver; ++k) {
        R8 es,qs,gam;
        R8 templ=(slfd(k)-R8(UW_GRAVIT)*z(k))/R8(UW_CPAIR);
        int status=uw_fqsatd(templ,pmid(k),es,qs,gam);
        R8 temps=templ+(qtfd(k)-qs)/(R8(UW_CPAIR)/R8(UW_LATVAP)+R8(UW_LATVAP)*qs/(R8(UW_RAIR)*uw_sq(templ)));
        status=uw_fqsatd(temps,pmid(k),es,qs,gam);
        qlfd(k)=uw_max(qtfd(k)-qi(k)-qs,uw_r8(0x0000000000000000ul));
        qvfd(k)=uw_max(uw_r8(0x0000000000000000ul),qtfd(k)-qi(k)-qlfd(k));
        tfd(k)=(slfd(k)+R8(UW_LATVAP)*qlfd(k)+latsub*qi(k)-R8(UW_GRAVIT)*z(k))/R8(UW_CPAIR);
    }
}

static inline void uw_eddy_relax(int pver, V kvm_out, V kvh_out, V kvm, V kvh) {
    for (int k=1; k<=pver+1; ++k) {
        // 1-lambda is a constant subexpression folded to exactly 0.5.
        kvm_out(k)=uw_r8(0x3fe0000000000000ul)*kvm_out(k)+uw_r8(0x3fe0000000000000ul)*kvm(k);
        kvh_out(k)=uw_r8(0x3fe0000000000000ul)*kvh_out(k)+uw_r8(0x3fe0000000000000ul)*kvh(k);
    }
}
static inline R8 uw_eddy_error(int pver, V kvh, V kvh_out) {
    R8 error=uw_r8(0x0000000000000000ul);
    for (int k=1; k<=pver; ++k) error=error+uw_sq(kvh(k)-kvh_out(k));
    return uw_sqrt(error/uw_real(pver));
}

static inline void uw_compute_eddy_diff(int pver,
    V t, V qv, R8 ztodt, V ql, V qi, V s, V rpdel, V cldn, V qrl, V wsedl,
    V z, V zi, V pmid, V pi, V u, V v, R8 taux, R8 tauy, R8 shflx, R8 qflx,
    bool wstarent, int nturb, thread R8& ustar, thread R8& pblh, V kvm_in, V kvh_in,
    V kvm_out, V kvh_out, V kvq, V cgh, V cgs, thread R8& tpert, thread R8& qpert,
    thread R8& wpert, V tke, V bprod, V sprod, V sfi, bool kvinit,
    thread R8& tauresx, thread R8& tauresy, R8 ksrftms, thread R8& ipbl, thread R8& kpblh,
    thread R8& wstarPBL, V turbtype, V sm_aw, thread Ws& ws) {
    WsMark mark(ws);
    // ncvmax=pver. Diagnostic arrays have their declared Fortran lengths.
    V ufd=ws.r8(pver),vfd=ws.r8(pver),tfd=ws.r8(pver),qvfd=ws.r8(pver),qlfd=ws.r8(pver);
    V qt=ws.r8(pver),sl=ws.r8(pver),qtfd=ws.r8(pver),slfd=ws.r8(pver),slv=ws.r8(pver);
    V slslope=ws.r8(pver),qtslope=ws.r8(pver),s2=ws.r8(pver),n2=ws.r8(pver),ri=ws.r8(pver);
    V sfuh=ws.r8(pver),sflh=ws.r8(pver);
    V chs=ws.r8(pver+1),chu=ws.r8(pver+1),cms=ws.r8(pver+1),cmu=ws.r8(pver+1);
    V kvh=ws.r8(pver+1),kvm=ws.r8(pver+1),kvf=ws.r8(pver+1);
    V kbase_o=ws.r8(pver),ktop_o=ws.r8(pver),kbase_mg=ws.r8(pver),ktop_mg=ws.r8(pver);
    V kbase_f=ws.r8(pver),ktop_f=ws.r8(pver);
    V wet=ws.r8(pver),web=ws.r8(pver),jtbu=ws.r8(pver),jbbu=ws.r8(pver),evhc=ws.r8(pver);
    V jt2slv=ws.r8(pver),n2ht=ws.r8(pver),n2hb=ws.r8(pver),lwp=ws.r8(pver),opt_depth=ws.r8(pver);
    V radinvfrac=ws.r8(pver),radf=ws.r8(pver),wstar=ws.r8(pver),wstar3fact=ws.r8(pver);
    V ebrk=ws.r8(pver),wbrk=ws.r8(pver),lbrk=ws.r8(pver),ricl=ws.r8(pver);
    V ghcl=ws.r8(pver),shcl=ws.r8(pver),smcl=ws.r8(pver);
    V ghi=ws.r8(pver+1),shi=ws.r8(pver+1),smi=ws.r8(pver+1),rii=ws.r8(pver+1),lengi=ws.r8(pver+1),wcap=ws.r8(pver+1);
    V jnk2d=ws.r8(pver+1);
    R8 rrho,minpblh,tkes,pblhp,ncvfin_o,ncvfin_mg,ncvfin_f,jnk1d,errorPBL;
    R8 cflx[1]={qflx},qmincg[1]={uw_r8(0x0000000000000000ul)};
    UwVdiffFields fl{true,true,true,{true,false,false,false,false}};
    for (int k=1; k<=pver; ++k) {
        ufd(k)=u(k); vfd(k)=v(k); tfd(k)=t(k); qvfd(k)=qv(k); qlfd(k)=ql(k);
    }
    for (int iturb=1; iturb<=nturb; ++iturb) {
        R8 tautotx=taux-ksrftms*ufd(pver),tautoty=tauy-ksrftms*vfd(pver);
        uw_trbintd(pver,z,ufd,vfd,tfd,pmid,tautotx,tautoty,ustar,rrho,s2,n2,ri,zi,pi,cldn,qtfd,qvfd,qlfd,qi,sfi,sfuh,sflh,slfd,slv,slslope,qtslope,chs,chu,cms,cmu,minpblh);
        if (iturb==1) for (int k=1; k<=pver; ++k) { qt(k)=qtfd(k); sl(k)=slfd(k); }
        for (int k=1; k<=pver+1; ++k) {
            if (iturb==1) { kvh(k)=kvinit?uw_r8(0x0000000000000000ul):kvh_in(k); kvm(k)=kvinit?uw_r8(0x0000000000000000ul):kvm_in(k); }
            else { kvh(k)=kvh_out(k); kvm(k)=kvm_out(k); }
        }
        uw_caleddy(pver,slfd,qtfd,qlfd,slv,ufd,vfd,pi,z,zi,qflx,shflx,
            slslope,qtslope,chu,chs,cmu,cms,sfuh,sflh,n2,s2,ri,rrho,pblh,ustar,
            kvh,kvm,kvh_out,kvm_out,tpert,qpert,qrl,kvf,tke,wstarent,bprod,sprod,
            minpblh,wpert,tkes,turbtype,sm_aw,kbase_o,ktop_o,ncvfin_o,kbase_mg,
            ktop_mg,ncvfin_mg,kbase_f,ktop_f,ncvfin_f,wet,web,jtbu,jbbu,evhc,
            jt2slv,n2ht,n2hb,lwp,opt_depth,radinvfrac,radf,wstar,wstar3fact,
            ebrk,wbrk,lbrk,ricl,ghcl,shcl,smcl,ghi,shi,smi,rii,lengi,wcap,
            pblhp,cldn,ipbl,kpblh,wsedl,ws);
        if (iturb==nturb) errorPBL=uw_eddy_error(pver,kvh,kvh_out);
        if (iturb>1 && iturb<nturb) uw_eddy_relax(pver,kvm_out,kvh_out,kvm,kvh);
        for (int k=1; k<=pver+1; ++k) { cgh(k)=uw_r8(0x0000000000000000ul); cgs(k)=uw_r8(0x0000000000000000ul); }
        if (iturb<nturb) {
            for (int k=1; k<=pver; ++k) { slfd(k)=sl(k); qtfd(k)=qt(k); ufd(k)=u(k); vfd(k)=v(k); }
            V q[1]={qtfd}; int errflag;
            uw_compute_vdiff(pver,1,pmid,pi,rpdel,t,ztodt,taux,tauy,shflx,cflx,
                1,pver,kvh_out,kvm_out,kvh_out,cgs,cgh,zi,ksrftms,qmincg,fl,
                ufd,vfd,q,slfd,jnk1d,jnk1d,jnk2d,jnk1d,errflag,tauresx,tauresy,0,ws);
            uw_eddy_retrieve(pver,slfd,qtfd,qi,z,pmid,tfd,qvfd,qlfd);
        }
    }
    for (int k=1; k<=pver+1; ++k) kvq(k)=kvh_out(k);
    wstarPBL=ipbl==uw_r8(0x3ff0000000000000ul)?uw_max(uw_r8(0x0000000000000000ul),wstar(1)):uw_r8(0x0000000000000000ul);
}
#endif

// Column ABI: UwF32View.p points to this column's bottom element; s is the
// element stride. Views are 1-based WRF order (bottom first), nk mass levels,
// nk+1 interfaces. The enclosing kernel supplies UwColumnIn and UwColumnOut.
// Surface inputs are float values. Surface outputs are pointers to this
// column's single float. kvm3d/kvh3d and tauresx2d/tauresy2d are inout.
// Ws pools must include all nested eddy/caleddy/vdiff allocations. Overflow
// is signalled by ws.err. The lead chooses capacity for its integrated lanes.
// Concatenate after uwpbl_eddy.cuh, before the enclosing  kernel.
#ifndef UWPBL_DRIVER_CUH
#define UWPBL_DRIVER_CUH
struct UwF32View {
    device const float* p; int s;
    device const float& operator()(int k) const { return p[(long)(k-1)*s]; }
};
struct UwF32OutView {
    device float* p; int s;
    device float& operator()(int k) const { return p[(long)(k-1)*s]; }
};
struct UwColumnIn {
    UwF32View u,v,th,rho,qv,qc,qi,qnc,qni,p,z,t,cldfra,exner,rthratenlw,wsedl3d,p8w,z_at_w;
    float hfx,qfx,ust,ht,dt; int itimestep;
};
struct UwColumnOut {
    UwF32OutView kvm3d,kvh3d,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqniblten;
    UwF32OutView tke_pbl,turbtype3d,smaw3d;
    device float *tauresx2d,*tauresy2d,*tpert2d,*qpert2d,*wpert2d,*pblh2d;
    device int *kpbl2d; // INTEGER in WRF, module_bl_camuwpbl_driver.F:208
};
static inline void uw_camuwpbl_column(int nk, const thread UwColumnIn& in,
    const thread UwColumnOut& out, thread Ws& ws) {
    WsMark mark(ws);
    bool first=in.itimestep==1;
    R8 ztodt=uw_widen(in.dt),rztodt=uw_r8(0x3ff0000000000000ul)/ztodt;
    R8 phis=uw_widen(in.ht)*R8(UW_GRAVIT);
    V u=ws.r8(nk),v=ws.r8(nk),pmid=ws.r8(nk),pdel=ws.r8(nk),rpdel=ws.r8(nk);
    V z=ws.r8(nk),t=ws.r8(nk),s=ws.r8(nk),qrl=ws.r8(nk),wsedl=ws.r8(nk),exner=ws.r8(nk),cldn=ws.r8(nk);
    V pdeldry=ws.r8(nk),rpdeldry=ws.r8(nk),pmiddry=ws.r8(nk),pintdry=ws.r8(nk+1);
    V cloud[5],cloudtnd[5];
    for (int m=0; m<5; ++m) { cloud[m]=ws.r8(nk); cloudtnd[m]=ws.r8(nk); }
    V pi=ws.r8(nk+1),zi=ws.r8(nk+1),kvq=ws.r8(nk+1),cgh=ws.r8(nk+1),cgs=ws.r8(nk+1);
    V kvh=ws.r8(nk+1),kvm=ws.r8(nk+1),kvh_in=ws.r8(nk+1),kvm_in=ws.r8(nk+1);
    for (int k=1; k<=nk; ++k) {
        int kflip=nk-k+1;
        for (int m=0; m<5; ++m) { cloud[m](kflip)=uw_r8(0xc1cdcc8bc4800000ul); cloudtnd[m](kflip)=uw_r8(0xc1cdcc8bc4800000ul); }
        u(kflip)=uw_widen(in.u(k)); v(kflip)=uw_widen(in.v(k));
        pmid(kflip)=uw_widen(in.p(k));
        R8 dp=uw_widen((in.p8w(k)-in.p8w(k+1)));
        pdel(kflip)=dp; rpdel(kflip)=uw_r8(0x3ff0000000000000ul)/dp;
        z(kflip)=uw_widen((in.z(k)-in.ht));
        t(kflip)=uw_widen(in.t(k));
        s(kflip)=R8(UW_CPAIR)*t(kflip)+R8(UW_GRAVIT)*z(kflip)+phis;
        qrl(kflip)=uw_widen((in.rthratenlw(k)*in.exner(k)))*R8(UW_CPAIR)*dp;
        wsedl(kflip)=uw_widen(in.wsedl3d(k));
        R8 multFrc=uw_r8(0x3ff0000000000000ul)/(uw_r8(0x3ff0000000000000ul)+uw_widen(in.qv(k)));
        cloud[0](kflip)=uw_max(uw_widen(in.qv(k))*multFrc,uw_r8(0x39b4484bfeebc2a0ul));
        cloud[1](kflip)=uw_widen(in.qc(k))*multFrc;
        cloud[2](kflip)=uw_widen(in.qi(k))*multFrc;
        cloud[3](kflip)=uw_widen(in.qnc(k))*multFrc;
        cloud[4](kflip)=uw_widen(in.qni(k))*multFrc;
        exner(kflip)=uw_widen(in.exner(k)); cldn(kflip)=uw_widen(in.cldfra(k));
        pdeldry(kflip)=pdel(kflip)*(uw_r8(0x3ff0000000000000ul)-cloud[0](kflip));
        rpdeldry(kflip)=uw_r8(0x3ff0000000000000ul)/pdeldry(kflip);
    }
    for (int k=1; k<=nk+1; ++k) {
        int kflip=nk-k+2;
        pi(kflip)=uw_widen(in.p8w(k)); zi(kflip)=uw_widen((in.z_at_w(k)-in.ht));
        kvq(kflip)=uw_r8(0x0000000000000000ul); cgh(kflip)=uw_r8(0x0000000000000000ul); cgs(kflip)=uw_r8(0x0000000000000000ul);
        if (first) { out.kvm3d(k)=uw_narrow(uw_r8(0x0000000000000000ul)); out.kvh3d(k)=uw_narrow(uw_r8(0x0000000000000000ul)); }
        kvh(kflip)=uw_widen(out.kvh3d(k)); kvm(kflip)=uw_widen(out.kvm3d(k));
    }
    pintdry(1)=pi(1);
    for (int k=1; k<=nk; ++k) {
        pintdry(k+1)=pintdry(k)+pdeldry(k);
        pmiddry(k)=(pintdry(k+1)+pintdry(k))*uw_r8(0x3fe0000000000000ul);
    }
    R8 shflx=uw_widen(in.hfx);
    R8 sgh=uw_r8(0x0000000000000000ul),landfrac=uw_r8(0x0000000000000000ul);
    R8 uMean=uw_widen(sqrt(((in.u(1)*in.u(1))+(in.v(1)*in.v(1)))));
    // At exact calm, SFCLAY may retain its positive minimum UST while the
    // resolved lowest-level wind is (0,0). WRF's direction-normalized stress
    // expression then evaluates 0/0 followed by inf*0. The physical vector
    // limit at calm is zero stress; preserve the source operation order for
    // every nonzero wind and make that singular endpoint finite here.
    R8 taux=uw_r8(0x0000000000000000ul),tauy=uw_r8(0x0000000000000000ul);
    if (uMean != uw_r8(0x0000000000000000ul)) {
        R8 tauFac=uw_widen(((in.rho(1)*in.ust)*in.ust))/uMean;
        taux=-(tauFac*uw_widen(in.u(1)));
        tauy=-(tauFac*uw_widen(in.v(1)));
    }
    if (first) { *out.tauresx2d=uw_narrow(uw_r8(0x0000000000000000ul)); *out.tauresy2d=uw_narrow(uw_r8(0x0000000000000000ul)); }
    R8 tauresx=uw_widen(*out.tauresx2d),tauresy=uw_widen(*out.tauresy2d);
    R8 ksrftms=uw_r8(0x0000000000000000ul),tautotx=taux,tautoty=tauy;
    R8 cflx[5]={uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul)};
    cflx[0]=uw_widen(in.qfx);
    R8 ustar=uw_r8(0x0000000000000000ul),pblh=uw_r8(0x0000000000000000ul),ipbl=uw_r8(0x0000000000000000ul),kpblh=uw_r8(0x0000000000000000ul),wstarPBL=uw_r8(0x0000000000000000ul);
    R8 tpert,qpert,wpert;
    V tke=ws.r8(nk+1),bprod=ws.r8(nk+1),sprod=ws.r8(nk+1),sfi=ws.r8(nk+1),turbtype=ws.r8(nk+1),smaw=ws.r8(nk+1);
    for (int k=1; k<=nk+1; ++k) { kvm_in(k)=kvm(k); kvh_in(k)=kvh(k); }
    uw_compute_eddy_diff(nk,t,cloud[0],ztodt,cloud[1],cloud[2],s,rpdel,cldn,qrl,wsedl,
        z,zi,pmid,pi,u,v,taux,tauy,shflx,cflx[0],true,5,ustar,pblh,kvm_in,kvh_in,
        kvm,kvh,kvq,cgh,cgs,tpert,qpert,wpert,tke,bprod,sprod,sfi,first,tauresx,
        tauresy,ksrftms,ipbl,kpblh,wstarPBL,turbtype,smaw,ws);
    for (int k=1; k<=nk+1; ++k) {
        int kflip=nk-k+2;
        out.kvh3d(k)=uw_narrow(kvh(kflip)); out.kvm3d(k)=uw_narrow(kvm(kflip));
    }
    V stnd=ws.r8(nk),utnd=ws.r8(nk),vtnd=ws.r8(nk),sl_pre=ws.r8(nk),qt_pre=ws.r8(nk);
    V tem2=ws.r8(nk),ftem=ws.r8(nk),ftem_pre=ws.r8(nk),dtk=ws.r8(nk);
    for (int k=1; k<=nk; ++k) {
        for (int m=0; m<5; ++m) cloudtnd[m](k)=cloud[m](k);
        stnd(k)=s(k); utnd(k)=u(k); vtnd(k)=v(k);
        sl_pre(k)=stnd(k)-R8(UW_LATVAP)*cloudtnd[1](k)-(R8(UW_LATVAP)+R8(UW_LATICE))*cloudtnd[2](k);
        qt_pre(k)=cloudtnd[0](k)+cloudtnd[1](k)+cloudtnd[2](k);
    }
    uw_aqsat(t,pmid,tem2,ftem,1,nk);
    for (int k=1; k<=nk; ++k) ftem_pre(k)=cloud[0](k)/ftem(k)*uw_r8(0x4059000000000000ul);
    R8 qmincg[5]={R8(UW_QMIN_Q),uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul),uw_r8(0x0000000000000000ul)};
    UwVdiffFields fl{true,true,true,{true,true,true,true,true}};
    R8 tautmsx,tautmsy,topflx; int errflag;
    uw_compute_vdiff(nk,5,pmid,pi,rpdel,t,ztodt,taux,tauy,shflx,cflx,1,nk,
        kvh,kvm,kvq,cgs,cgh,zi,ksrftms,qmincg,fl,utnd,vtnd,cloudtnd,stnd,
        tautmsx,tautmsy,dtk,topflx,errflag,tauresx,tauresy,1,ws);
    if (errflag) { atomic_store_explicit(ws.err, 2, memory_order_relaxed); return; } // enclosing kernel reports WRF fatal
    *out.tauresx2d=uw_narrow(tauresx); *out.tauresy2d=uw_narrow(tauresy);
    V slten=ws.r8(nk),qtten=ws.r8(nk),taft=ws.r8(nk);
    for (int k=1; k<=nk; ++k) {
        R8 sl=stnd(k)-R8(UW_LATVAP)*cloudtnd[1](k)-(R8(UW_LATVAP)+R8(UW_LATICE))*cloudtnd[2](k);
        R8 qt=cloudtnd[0](k)+cloudtnd[1](k)+cloudtnd[2](k);
        slten(k)=(sl-sl_pre(k))*rztodt; qtten(k)=(qt-qt_pre(k))*rztodt;
        stnd(k)=(stnd(k)-s(k))*rztodt; utnd(k)=(utnd(k)-u(k))*rztodt; vtnd(k)=(vtnd(k)-v(k))*rztodt;
        for (int m=0; m<5; ++m) cloudtnd[m](k)=(cloudtnd[m](k)-cloud[m](k))*rztodt;
        R8 qv_aft=cloud[0](k)+cloudtnd[0](k)*ztodt;
        R8 ql_aft=cloud[1](k)+cloudtnd[1](k)*ztodt;
        R8 qi_aft=cloud[2](k)+cloudtnd[2](k)*ztodt;
        R8 s_aft=s(k)+stnd(k)*ztodt;
        taft(k)=(s_aft-R8(UW_GRAVIT)*z(k))/R8(UW_CPAIR);
        R8 u_aft=u(k)+utnd(k)*ztodt,v_aft=v(k)+vtnd(k)*ztodt;
    }
    uw_aqsat(taft,pmid,tem2,ftem,1,nk);
    for (int k=1; k<=nk; ++k) {
        R8 ftem_aft=(cloud[0](k)+cloudtnd[0](k)*ztodt)/ftem(k)*uw_r8(0x4059000000000000ul);
        R8 tten=(taft(k)-t(k))*rztodt,rhten=(ftem_aft-ftem_pre(k))*rztodt;
    }
    for (int k=1; k<=nk; ++k) {
        int kflip=nk-k+1;
        out.rublten(k)=uw_narrow(utnd(kflip)); out.rvblten(k)=uw_narrow(vtnd(kflip));
        out.rthblten(k)=uw_narrow(stnd(kflip)/R8(UW_CPAIR)/exner(kflip));
        R8 multFrc=uw_r8(0x3ff0000000000000ul)+uw_widen(in.qv(k));
        out.rqvblten(k)=uw_narrow(cloudtnd[0](kflip)*multFrc*multFrc);
        out.rqcblten(k)=uw_narrow(cloudtnd[1](kflip)*multFrc);
        out.rqiblten(k)=uw_narrow(cloudtnd[2](kflip)*multFrc);
        out.rqniblten(k)=uw_narrow(cloudtnd[4](kflip)*multFrc);
    }
    for (int k=1; k<=nk+1; ++k) {
        int kflip=nk-k+2;
        out.tke_pbl(k)=uw_narrow(tke(kflip)); out.turbtype3d(k)=uw_narrow(turbtype(kflip)); out.smaw3d(k)=uw_narrow(smaw(kflip));
    }
    *out.kpbl2d=nk-uw_int(kpblh)+1;
    *out.pblh2d=uw_narrow(pblh); *out.tpert2d=uw_narrow(tpert);
    *out.qpert2d=uw_narrow(qpert); *out.wpert2d=uw_narrow(wpert);
}
#endif

// uwpbl.cu -- the WRF v4.7.1 UW moist-turbulence PBL (bl_pbl_physics = 9),
// one thread per column.
//
// The loader (gpuwm/core/kernels/__init__.py _EXTRA_HEADERS["uwpbl"])
// prepends, in order: glibc_flt64.cuh, uwpbl_common.cuh, uwpbl_wvsat.cuh,
// uwpbl_vdiff.cuh, uwpbl_zisocl.cuh, uwpbl_caleddy.cuh, uwpbl_eddy.cuh,
// uwpbl_driver.cuh.
// uw_camuwpbl_column (uwpbl_driver.cuh) is module_bl_camuwpbl_driver.F's
// per-column body; this file only binds the engine's arrays to it and gives
// each thread its slice of the workspace pools.
//
// Layout.  Every 3-D field is WRF-ordered (k = 1 at the bottom) and stored
// level-major with the column index fastest: element k of column c is
// base[k*ncols + c], c = j*nx + i.  Surface fields are base[c].  The pools
// are slot-major with the chunk's thread index fastest, so a warp's reads of
// one slot are 32 consecutive words.

kernel void uwpbl_columns(constant UwpblColumnsResources &r [[buffer(0)]],
                          constant UwpblColumnsParams &params [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const int ncols=int(params.ncols), col0=int(params.col0), nchunk=int(params.nchunk);
    const int pool_stride=int(params.pool_stride), nk=int(params.nk);
    const float dt=params.dt; const int itimestep=int(params.itimestep);
    const int r8cap=int(params.r8cap), i4cap=int(params.i4cap);
    device const float *u=r.u,*v=r.v,*th=r.th,*rho=r.rho,*qv=r.qv,*qc=r.qc,*qi=r.qi,*qnc=r.qnc,*qni=r.qni;
    device const float *p=r.p,*z=r.z,*t=r.t,*cldfra=r.cldfra,*exner=r.exner,*rthratenlw=r.rthratenlw,*wsedl3d=r.wsedl3d;
    device const float *p8w=r.p8w,*z_at_w=r.z_at_w,*hfx=r.hfx,*qfx=r.qfx,*ust=r.ust,*ht=r.ht;
    device float *kvm3d=r.kvm3d,*kvh3d=r.kvh3d,*tauresx2d=r.tauresx2d,*tauresy2d=r.tauresy2d;
    device float *rublten=r.rublten,*rvblten=r.rvblten,*rthblten=r.rthblten,*rqvblten=r.rqvblten,*rqcblten=r.rqcblten,*rqiblten=r.rqiblten,*rqniblten=r.rqniblten;
    device float *tke_pbl=r.tke_pbl,*turbtype3d=r.turbtype3d,*smaw3d=r.smaw3d,*tpert2d=r.tpert2d,*qpert2d=r.qpert2d,*wpert2d=r.wpert2d,*pblh2d=r.pblh2d;
    device int *kpbl2d=r.kpbl2d;
    device R8 *r8pool=reinterpret_cast<device R8*>(r.r8pool); device int *i4pool=r.i4pool;
    device atomic_int *err=reinterpret_cast<device atomic_int*>(r.err);
    const int tid = int(flat);
    if (tid >= nchunk) return;
    const long c = col0 + tid;
    const int s = (int)ncols;
    Ws ws{r8pool + tid, i4pool + tid, pool_stride, 0, r8cap, 0, i4cap, err};
    // Views start at this column's bottom element; element k (1-based) is
    // base[(k-1)*ncols + c].  The column driver never writes an input view.
    #define UW_IN(a) UwF32View{(a) + c, s}
    #define UW_OUT(a) UwF32OutView{(a) + c, s}
    UwColumnIn in{UW_IN(u), UW_IN(v), UW_IN(th), UW_IN(rho), UW_IN(qv),
                  UW_IN(qc), UW_IN(qi), UW_IN(qnc), UW_IN(qni), UW_IN(p),
                  UW_IN(z), UW_IN(t), UW_IN(cldfra), UW_IN(exner),
                  UW_IN(rthratenlw), UW_IN(wsedl3d), UW_IN(p8w), UW_IN(z_at_w),
                  hfx[c], qfx[c], ust[c], ht[c], dt, itimestep};
    UwColumnOut out{UW_OUT(kvm3d), UW_OUT(kvh3d), UW_OUT(rublten),
                    UW_OUT(rvblten), UW_OUT(rthblten), UW_OUT(rqvblten),
                    UW_OUT(rqcblten), UW_OUT(rqiblten), UW_OUT(rqniblten),
                    UW_OUT(tke_pbl), UW_OUT(turbtype3d), UW_OUT(smaw3d),
                    tauresx2d + c, tauresy2d + c, tpert2d + c, qpert2d + c,
                    wpert2d + c, pblh2d + c, kpbl2d + c};
    #undef UW_IN
    #undef UW_OUT
    uw_camuwpbl_column(nk, in, out, ws);
}
