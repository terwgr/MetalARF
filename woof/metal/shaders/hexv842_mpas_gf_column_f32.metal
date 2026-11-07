#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_log.metalh"
struct GFReal { uint bits; GFReal()=default; GFReal(float f):bits(as_type<uint>(f)){} explicit operator int() const {return int(as_type<float>(bits));} explicit operator int() const device {return int(as_type<float>(bits));} static GFReal word(uint x){GFReal a;a.bits=x;return a;} };
static inline GFReal gf_nan(uint r,uint a,uint b){if(!soft32_is_nan(r))return GFReal::word(r);if(soft32_is_nan(a))return GFReal::word(soft32_quiet_nan(a));if(soft32_is_nan(b))return GFReal::word(soft32_quiet_nan(b));return GFReal::word(0xffc00000u);}
static inline GFReal operator+(GFReal a,GFReal b){return gf_nan(soft32_add(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator-(GFReal a,GFReal b){return gf_nan(soft32_sub(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator*(GFReal a,GFReal b){return gf_nan(soft32_mul(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator/(GFReal a,GFReal b){return gf_nan(soft32_div(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator-(GFReal a){return GFReal::word(a.bits^0x80000000u);}
static inline bool operator<(GFReal a,GFReal b){return soft32_less(a.bits,b.bits);}
static inline GFReal gf_log(GFReal a){return gf_nan(soft_gfk_log(a.bits),a.bits,0u);}
static inline GFReal gf_pow(GFReal a,GFReal b){return gf_nan(soft_gfk_pow_bits(a.bits,b.bits),a.bits,b.bits);}

static inline bool operator>(GFReal a,GFReal b){return soft32_greater(a.bits,b.bits);}
static inline bool operator<=(GFReal a,GFReal b){return soft32_less_equal(a.bits,b.bits);}
static inline bool operator>=(GFReal a,GFReal b){return soft32_greater_equal(a.bits,b.bits);}
static inline bool operator==(GFReal a,GFReal b){return soft32_equal(a.bits,b.bits);}
static inline bool operator!=(GFReal a,GFReal b){return !soft32_equal(a.bits,b.bits);}
static inline GFReal gf_add_rhs(GFReal a,GFReal b){return gf_nan(soft32_add(a.bits,b.bits),b.bits,a.bits);}
static inline GFReal gf_abs(GFReal a){return GFReal::word(a.bits&0x7fffffffu);}
static inline GFReal gf_sqrt(GFReal a){return gf_nan(soft32_sqrt(a.bits),a.bits,0u);}
static inline GFReal gf_powi(GFReal a,int p){return (a*a)*a;}
static inline GFReal gf_nan(){return GFReal::word(0x7fc00000u);}
static inline GFReal gf_infinity(){return GFReal::word(0x7f800000u);}
template<class A,class B> static inline auto gf_max(A a,B b)->decltype(a+b){return a>b?decltype(a+b)(a):decltype(a+b)(b);}
template<class A,class B> static inline auto gf_min(A a,B b)->decltype(a+b){return a<b?decltype(a+b)(a):decltype(a+b)(b);}
template<class D,class S> static inline void gf_memmove(D d,S s,ulong n){for(ulong i=0;i<n;i++)d[i]=s[i];}
template<class T> static inline void gf_memset(device T* d,int v,ulong n){device char* p=(device char*)d;for(ulong i=0;i<n;i++)p[i]=char(v);}
template<class T> static inline void gf_memset(thread T* d,int v,ulong n){thread char* p=(thread char*)d;for(ulong i=0;i<n;i++)p[i]=char(v);}
template<class A,class B> static inline int gf_compare_string(long na,A a,long nb,B b){for(long i=0;i<max(na,nb);i++){int x=i<na?(uchar)a[i]:32;int y=i<nb?(uchar)b[i]:32;if(x!=y)return x-y;}return 0;}
static inline void gf_unreachable_random(thread GFReal*){__builtin_trap();}
typedef char GFText[50];
struct GFWorkspace {device char* data;ulong capacity,position,highwater;uint diagnostics;
 device void* allocate(ulong n){n=(n+15)&~15ul;if(position+n>capacity){__builtin_trap();return nullptr;}device void* result=data+position;position+=n;highwater=max(highwater,position);return result;}};
struct GFFrame {thread GFWorkspace &ws;ulong saved;GFFrame(thread GFWorkspace &w):ws(w),saved(w.position){} ~GFFrame(){ws.position=saved;}};
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55,typename P56> static inline void cup_gf_sh(thread GFWorkspace &workspace, P0 xmb_out, P1 zo, P2 outqc, P3 j, P4 aaeq, P5 t, P6 q, P7 z1, P8 tn, P9 qo, P10 po, P11 pre, P12 p, P13 outt, P14 outq, P15 dtime, P16 ktau, P17 psur, P18 us, P19 vs, P20 tcrit, P21 ztexec, P22 zqexec, P23 ccn, P24 ccnclean, P25 rho, P26 dxcell, P27 areacell, P28 dhdt, P29 kpbl, P30 kbcon, P31 ktop, P32 cupclws, P33 k22, P34 xland, P35 gsw, P36 tscl_kf, P37 xl, P38 rv, P39 cp, P40 g, P41 ichoice, P42 ipr, P43 jpr, P44 ierr, P45 ierrc, P46 autoconv, P47 itf, P48 jtf, P49 ktf, P50 use_excess, P51 its, P52 ite, P53 jts, P54 jte, P55 kts, P56 kte, long _ierrc);
template<typename P0> static inline GFReal satvap(thread GFWorkspace &workspace, P0 temp2);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38> static inline void gdinit(thread GFWorkspace &workspace, P0 rthcuten, P1 rqvcuten, P2 rqccuten, P3 rqicuten, P4 mass_flux, P5 cp, P6 restart, P7 p_qc, P8 p_qi, P9 p_first_scalar, P10 rthften, P11 rqvften, P12 apr_gr, P13 apr_w, P14 apr_mc, P15 apr_st, P16 apr_as, P17 apr_capma, P18 apr_capme, P19 apr_capmi, P20 allowed_to_read, P21 ids, P22 ide, P23 jds, P24 jde, P25 kds, P26 kde, P27 ims, P28 ime, P29 jms, P30 jme, P31 kms, P32 kme, P33 its, P34 ite, P35 jts, P36 jte, P37 kts, P38 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44> static inline void cup_up_moisture(thread GFWorkspace &workspace, P0 name, P1 ierr, P2 z_cup, P3 qc, P4 qrc, P5 pw, P6 pwav, P7 ccnclean, P8 p_cup, P9 kbcon, P10 ktop, P11 cd, P12 dby, P13 clw_all, P14 t_cup, P15 q, P16 gamma_cup, P17 zu, P18 qes_cup, P19 k22, P20 qe_cup, P21 xl, P22 zqexec, P23 use_excess, P24 ccn, P25 rho, P26 up_massentr, P27 up_massdetr, P28 psum, P29 psumh, P30 autoconv, P31 aeroevap, P32 itest, P33 itf, P34 jtf, P35 ktf, P36 j, P37 ipr, P38 jpr, P39 its, P40 ite, P41 jts, P42 jte, P43 kts, P44 kte, long _name);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55> static inline void cup_output_ens_3d(thread GFWorkspace &workspace, P0 xf_ens, P1 ierr, P2 dellat, P3 dellaq, P4 dellaqc, P5 subt_ens, P6 subq_ens, P7 subt, P8 subq, P9 outtem, P10 outq, P11 outqc, P12 zu, P13 sub_mas, P14 pre, P15 pw, P16 xmb, P17 ktop, P18 j, P19 name, P20 nx, P21 nx2, P22 iens, P23 ierr2, P24 ierr3, P25 pr_ens, P26 maxens3, P27 ensdim, P28 sig, P29 apr_gr, P30 apr_w, P31 apr_mc, P32 apr_st, P33 apr_as, P34 apr_capma, P35 apr_capme, P36 apr_capmi, P37 closure_n, P38 xland1, P39 weight_gr, P40 weight_w, P41 weight_mc, P42 weight_st, P43 weight_as, P44 training, P45 ipr, P46 jpr, P47 itf, P48 jtf, P49 ktf, P50 its, P51 ite, P52 jts, P53 jte, P54 kts, P55 kte, long _name);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14> static inline void neg_check(thread GFWorkspace &workspace, P0 j, P1 subt, P2 subq, P3 dt, P4 q, P5 outq, P6 outt, P7 outqc, P8 pret, P9 its, P10 ite, P11 kts, P12 kte, P13 itf, P14 ktf);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43> static inline void g3init(thread GFWorkspace &workspace, P0 rthcuten, P1 rqvcuten, P2 rqccuten, P3 rqicuten, P4 mass_flux, P5 cp, P6 restart, P7 p_qc, P8 p_qi, P9 p_first_scalar, P10 rthften, P11 rqvften, P12 apr_gr, P13 apr_w, P14 apr_mc, P15 apr_st, P16 apr_as, P17 apr_capma, P18 apr_capme, P19 apr_capmi, P20 cugd_tten, P21 cugd_ttens, P22 cugd_qvten, P23 cugd_qvtens, P24 cugd_qcten, P25 allowed_to_read, P26 ids, P27 ide, P28 jds, P29 jde, P30 kds, P31 kde, P32 ims, P33 ime, P34 jms, P35 jme, P36 kms, P37 kme, P38 its, P39 ite, P40 jts, P41 jte, P42 kts, P43 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17> static inline void cup_up_aa0(thread GFWorkspace &workspace, P0 aa0, P1 z, P2 zu, P3 dby, P4 gamma_cup, P5 t_cup, P6 kbcon, P7 ktop, P8 ierr, P9 itf, P10 jtf, P11 ktf, P12 its, P13 ite, P14 jts, P15 jte, P16 kts, P17 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13> static inline void cup_minimi(thread GFWorkspace &workspace, P0 array, P1 ks, P2 kend, P3 kt, P4 ierr, P5 itf, P6 jtf, P7 ktf, P8 its, P9 ite, P10 jts, P11 jte, P12 kts, P13 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13> static inline void cup_maximi(thread GFWorkspace &workspace, P0 array, P1 ks, P2 ke, P3 maxx, P4 ierr, P5 itf, P6 jtf, P7 ktf, P8 its, P9 ite, P10 jts, P11 jte, P12 kts, P13 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14> static inline void cup_ktop(thread GFWorkspace &workspace, P0 ierrc, P1 ilo, P2 dby, P3 kbcon, P4 ktop, P5 ierr, P6 itf, P7 jtf, P8 ktf, P9 its, P10 ite, P11 jts, P12 jte, P13 kts, P14 kte, long _ierrc);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25> static inline void cup_kbcon(thread GFWorkspace &workspace, P0 ierrc, P1 cap_inc, P2 iloop, P3 k22, P4 kbcon, P5 he_cup, P6 hes_cup, P7 hkb, P8 ierr, P9 kbmax, P10 p_cup, P11 cap_max, P12 xl, P13 cp, P14 ztexec, P15 zqexec, P16 use_excess, P17 itf, P18 jtf, P19 ktf, P20 its, P21 ite, P22 jts, P23 jte, P24 kts, P25 kte, long _ierrc);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43> static inline void cup_forcing_ens_3d(thread GFWorkspace &workspace, P0 closure_n, P1 xland, P2 aa0, P3 aa1, P4 xaa0, P5 mbdt, P6 dtime, P7 ierr, P8 ierr2, P9 ierr3, P10 xf_ens, P11 j, P12 name, P13 axx, P14 maxens, P15 iens, P16 iedt, P17 maxens2, P18 maxens3, P19 mconv, P20 p_cup, P21 ktop, P22 omeg, P23 zd, P24 k22, P25 zu, P26 pr_ens, P27 edt, P28 kbcon, P29 ensdim, P30 icoic, P31 ipr, P32 jpr, P33 itf, P34 jtf, P35 ktf, P36 its, P37 ite, P38 jts, P39 jte, P40 kts, P41 kte, P42 ens4, P43 ktau, long _name);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29> static inline void cup_env_clev(thread GFWorkspace &workspace, P0 t, P1 qes, P2 q, P3 he, P4 hes, P5 z, P6 p, P7 qes_cup, P8 q_cup, P9 he_cup, P10 hes_cup, P11 z_cup, P12 p_cup, P13 gamma_cup, P14 t_cup, P15 psur, P16 ierr, P17 z1, P18 xl, P19 rv, P20 cp, P21 itf, P22 jtf, P23 ktf, P24 its, P25 ite, P26 jts, P27 jte, P28 kts, P29 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22> static inline void cup_env(thread GFWorkspace &workspace, P0 z, P1 qes, P2 he, P3 hes, P4 t, P5 q, P6 p, P7 z1, P8 psur, P9 ierr, P10 tcrit, P11 itest, P12 xl, P13 cp, P14 itf, P15 jtf, P16 ktf, P17 its, P18 ite, P19 jts, P20 jte, P21 kts, P22 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30> static inline void cup_dd_moisture_new(thread GFWorkspace &workspace, P0 ierrc, P1 zd, P2 hcd, P3 hes_cup, P4 qcd, P5 qes_cup, P6 pwd, P7 q_cup, P8 z_cup, P9 dd_massentr, P10 dd_massdetr, P11 jmin, P12 ierr, P13 gamma_cup, P14 pwev, P15 bu, P16 qrcd, P17 q, P18 he, P19 t_cup, P20 iloop, P21 xl, P22 itf, P23 jtf, P24 ktf, P25 its, P26 ite, P27 jts, P28 jte, P29 kts, P30 kte, long _ierrc);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32> static inline void cup_dd_edt(thread GFWorkspace &workspace, P0 ierr, P1 us, P2 vs, P3 z, P4 ktop, P5 kbcon, P6 edt, P7 p, P8 pwav, P9 pw, P10 ccn, P11 pwev, P12 edtmax, P13 edtmin, P14 maxens2, P15 edtc, P16 psum2, P17 psumh, P18 ccnclean, P19 rho, P20 aeroevap, P21 itf, P22 jtf, P23 ktf, P24 j, P25 ipr, P26 jpr, P27 its, P28 ite, P29 jts, P30 jte, P31 kts, P32 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55,typename P56,typename P57,typename P58,typename P59,typename P60,typename P61,typename P62,typename P63,typename P64,typename P65,typename P66,typename P67,typename P68,typename P69,typename P70,typename P71,typename P72,typename P73,typename P74,typename P75,typename P76> static inline void cup_gf(thread GFWorkspace &workspace, P0 zo, P1 outqc, P2 j, P3 aaeq, P4 t, P5 q, P6 z1, P7 sub_mas, P8 tn, P9 qo, P10 po, P11 pre, P12 p, P13 outt, P14 outq, P15 dtime, P16 ktau, P17 psur, P18 us, P19 vs, P20 tcrit, P21 iens, P22 ztexec, P23 zqexec, P24 ccn, P25 ccnclean, P26 rho, P27 dxcell, P28 areacell, P29 mconv, P30 omeg, P31 maxiens, P32 maxens, P33 maxens2, P34 maxens3, P35 ensdim, P36 apr_gr, P37 apr_w, P38 apr_mc, P39 apr_st, P40 apr_as, P41 apr_capma, P42 apr_capme, P43 apr_capmi, P44 kbcon, P45 ktop, P46 cupclw, P47 xf_ens, P48 pr_ens, P49 xland, P50 gsw, P51 subt, P52 subq, P53 xl, P54 rv, P55 cp, P56 g, P57 ichoice, P58 ipr, P59 jpr, P60 ierrc, P61 ens4, P62 beta, P63 autoconv, P64 aeroevap, P65 itf, P66 jtf, P67 ktf, P68 training, P69 xmb, P70 use_excess, P71 its, P72 ite, P73 jts, P74 jte, P75 kts, P76 kte, long _ierrc);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55,typename P56,typename P57,typename P58,typename P59,typename P60,typename P61,typename P62,typename P63,typename P64> static inline void cu_grell_freitas(thread GFWorkspace &workspace, P0 itimestep, P1 dt, P2 dxcell, P3 areacell, P4 u, P5 v, P6 t, P7 w, P8 q, P9 p, P10 pi, P11 rho, P12 dz8w, P13 p8w, P14 xland, P15 ht, P16 hfx, P17 qfx, P18 gsw, P19 rqvften, P20 rthften, P21 rthblten, P22 rqvblten, P23 rthraten, P24 kpbl, P25 xlv, P26 cp, P27 g, P28 r_v, P29 ichoice_deep, P30 ichoice_shallow, P31 ishallow_g3, P32 htop, P33 hbot, P34 k22_shallow, P35 kbcon_shallow, P36 ktop_shallow, P37 xmb_total, P38 xmb_shallow, P39 raincv, P40 pratec, P41 gdc, P42 gdc2, P43 rthcuten, P44 rqvcuten, P45 rqccuten, P46 rqicuten, P47 ims, P48 ime, P49 jms, P50 jme, P51 kms, P52 kme, P53 ids, P54 ide, P55 jds, P56 jde, P57 kds, P58 kde, P59 its, P60 ite, P61 jts, P62 jte, P63 kts, P64 kte);
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55,typename P56> static inline void cup_gf_sh(thread GFWorkspace &workspace, P0 xmb_out, P1 zo, P2 outqc, P3 j, P4 aaeq, P5 t, P6 q, P7 z1, P8 tn, P9 qo, P10 po, P11 pre, P12 p, P13 outt, P14 outq, P15 dtime, P16 ktau, P17 psur, P18 us, P19 vs, P20 tcrit, P21 ztexec, P22 zqexec, P23 ccn, P24 ccnclean, P25 rho, P26 dxcell, P27 areacell, P28 dhdt, P29 kpbl, P30 kbcon, P31 ktop, P32 cupclws, P33 k22, P34 xland, P35 gsw, P36 tscl_kf, P37 xl, P38 rv, P39 cp, P40 g, P41 ichoice, P42 ipr, P43 jpr, P44 ierr, P45 ierrc, P46 autoconv, P47 itf, P48 jtf, P49 ktf, P50 use_excess, P51 its, P52 ite, P53 jts, P54 jte, P55 kts, P56 kte, long _ierrc) {
  GFFrame frame(workspace);

  device void * D_6584;

  device void * D_6588;

  device void * D_6592;

  device void * D_6596;

  device void * D_6600;

  device void * D_6604;

  device void * D_6608;

  device void * D_6612;

  device void * D_6616;

  device void * D_6620;

  device void * D_6624;

  device void * D_6628;

  device void * D_6632;

  device void * D_6636;

  device void * D_6640;

  device void * D_6644;

  device void * D_6648;

  device void * D_6652;

  device void * D_6656;

  device void * D_6660;

  device void * D_6664;

  device void * D_6668;

  device void * D_6672;

  device void * D_6676;

  device void * D_6680;

  device void * D_6684;

  device void * D_6688;

  device void * D_6692;

  device void * D_6696;

  device void * D_6700;

  device void * D_6704;

  device void * D_6708;

  device void * D_6712;

  device void * D_6716;

  device void * D_6720;

  device void * D_6724;

  device void * D_6728;

  device void * D_6732;

  device void * D_6736;

  device void * D_6740;

  device void * D_6744;

  device void * D_6748;

  device void * D_6752;

  device void * D_6756;

  device void * D_6760;

  device void * D_6764;

  device void * D_6768;

  device void * D_6772;

  device void * D_6776;

  device void * D_6780;

  device void * D_6784;

  device void * D_6788;

  device void * D_6792;

  device void * D_6796;

  device void * D_6800;

  device void * D_6804;

  device void * D_6808;

  device void * D_6812;

  device void * D_6816;

  device void * D_6820;

  device void * D_6824;

  device void * D_6828;

  device void * D_6832;

  device void * D_6836;

  device void * D_6840;

  device void * D_6844;

  device void * D_6848;

  device void * D_6852;

  device void * D_6856;

  device void * D_6860;

  device void * D_6864;

  device void * D_6868;

  device void * D_6872;

  device void * D_6876;
  long lbound_0;
  long ubound_1;
  long offset_2;
  long size_3;

  device GFReal * aa0;
  long lbound_4;
  long ubound_5;
  long offset_6;
  long size_7;

  device GFReal * aa1;
  long lbound_8;
  long ubound_9;
  long offset_10;
  long size_11;
  GFReal blqe;
  long lbound_12;
  long ubound_13;
  long offset_14;
  long size_15;

  device GFReal * cap_max;
  long lbound_16;
  long ubound_17;
  long offset_18;
  long size_19;

  device GFReal * cap_max_increment;
  GFReal cap_maxs;
  long lbound_20;
  long ubound_21;
  long lbound_22;
  long ubound_23;
  long stride_24;
  long offset_25;
  long size_26;

  device GFReal * cd;
  long lbound_27;
  long ubound_28;
  long lbound_29;
  long ubound_30;
  long stride_31;
  long offset_32;
  long size_33;
  GFReal day;
  long lbound_34;
  long ubound_35;
  long lbound_36;
  long ubound_37;
  long stride_38;
  long offset_39;
  long size_40;

  device GFReal * dby;
  long lbound_41;
  long ubound_42;
  long lbound_43;
  long ubound_44;
  long stride_45;
  long offset_46;
  long size_47;

  device GFReal * dbyo;
  long lbound_48;
  long ubound_49;
  long lbound_50;
  long ubound_51;
  long stride_52;
  long offset_53;
  long size_54;

  device GFReal * dellah;
  long lbound_55;
  long ubound_56;
  long lbound_57;
  long ubound_58;
  long stride_59;
  long offset_60;
  long size_61;

  device GFReal * dellaq;
  long lbound_62;
  long ubound_63;
  long lbound_64;
  long ubound_65;
  long stride_66;
  long offset_67;
  long size_68;

  device GFReal * dellaqc;
  long lbound_69;
  long ubound_70;
  long lbound_71;
  long ubound_72;
  long stride_73;
  long offset_74;
  long size_75;

  device GFReal * dellat;
  GFReal depth_min;
  GFReal detup;
  GFReal detupk;
  GFReal dh;
  long lbound_76;
  long ubound_77;
  long lbound_78;
  long ubound_79;
  long stride_80;
  long offset_81;
  long size_82;
  GFReal dp;
  long lbound_83;
  long ubound_84;
  long lbound_85;
  long ubound_86;
  long stride_87;
  long offset_88;
  long size_89;

  device GFReal * dsubh;
  long lbound_90;
  long ubound_91;
  long lbound_92;
  long ubound_93;
  long stride_94;
  long offset_95;
  long size_96;

  device GFReal * dsubq;
  long lbound_97;
  long ubound_98;
  long lbound_99;
  long ubound_100;
  long stride_101;
  long offset_102;
  long size_103;

  device GFReal * dsubt;
  GFReal dz;
  GFReal entr_rate;
  long lbound_104;
  long ubound_105;
  long lbound_106;
  long ubound_107;
  long stride_108;
  long offset_109;
  long size_110;

  device GFReal * entr_rate_2d;
  GFReal entup;
  GFReal entupk;
  GFReal frh;
  GFReal fsum;
  long lbound_111;
  long ubound_112;
  long lbound_113;
  long ubound_114;
  long stride_115;
  long offset_116;
  long size_117;

  device GFReal * gamma_cup;
  long lbound_118;
  long ubound_119;
  long lbound_120;
  long ubound_121;
  long stride_122;
  long offset_123;
  long size_124;

  device GFReal * gammao_cup;
  long lbound_125;
  long ubound_126;
  long lbound_127;
  long ubound_128;
  long stride_129;
  long offset_130;
  long size_131;

  device GFReal * hc;
  long lbound_132;
  long ubound_133;
  long lbound_134;
  long ubound_135;
  long stride_136;
  long offset_137;
  long size_138;

  device GFReal * hco;
  long lbound_139;
  long ubound_140;
  long lbound_141;
  long ubound_142;
  long stride_143;
  long offset_144;
  long size_145;

  device GFReal * he;
  long lbound_146;
  long ubound_147;
  long lbound_148;
  long ubound_149;
  long stride_150;
  long offset_151;
  long size_152;

  device GFReal * he_cup;
  long lbound_153;
  long ubound_154;
  long lbound_155;
  long ubound_156;
  long stride_157;
  long offset_158;
  long size_159;

  device GFReal * heo;
  long lbound_160;
  long ubound_161;
  long lbound_162;
  long ubound_163;
  long stride_164;
  long offset_165;
  long size_166;

  device GFReal * heo_cup;
  long lbound_167;
  long ubound_168;
  long lbound_169;
  long ubound_170;
  long stride_171;
  long offset_172;
  long size_173;

  device GFReal * hes;
  long lbound_174;
  long ubound_175;
  long lbound_176;
  long ubound_177;
  long stride_178;
  long offset_179;
  long size_180;

  device GFReal * hes_cup;
  long lbound_181;
  long ubound_182;
  long lbound_183;
  long ubound_184;
  long stride_185;
  long offset_186;
  long size_187;

  device GFReal * heso;
  long lbound_188;
  long ubound_189;
  long lbound_190;
  long ubound_191;
  long stride_192;
  long offset_193;
  long size_194;

  device GFReal * heso_cup;
  long lbound_195;
  long ubound_196;
  long offset_197;
  long size_198;

  device GFReal * hkb;
  long lbound_199;
  long ubound_200;
  long offset_201;
  long size_202;

  device GFReal * hkbo;
  int i;
  long lbound_203;
  long ubound_204;
  long offset_205;
  long size_206;
  long lbound_207;
  long ubound_208;
  long offset_209;
  long size_210;
  int iresult;
  long lbound_211;
  long ubound_212;
  long offset_213;
  long size_214;

  device int * jmin;
  int k;
  int k1;
  int k2;
  long lbound_215;
  long ubound_216;
  long offset_217;
  long size_218;
  long lbound_219;
  long ubound_220;
  long offset_221;
  long size_222;
  long lbound_223;
  long ubound_224;
  long offset_225;
  long size_226;

  device int * kbmax;
  long lbound_227;
  long ubound_228;
  long offset_229;
  long size_230;
  int kstart;
  long lbound_231;
  long ubound_232;
  long offset_233;
  long size_234;
  int levadj;
  GFReal mbdt;
  long lbound_235;
  long ubound_236;
  long lbound_237;
  long ubound_238;
  long stride_239;
  long offset_240;
  long size_241;
  long lbound_242;
  long ubound_243;
  long lbound_244;
  long ubound_245;
  long stride_246;
  long offset_247;
  long size_248;
  long lbound_249;
  long ubound_250;
  long lbound_251;
  long ubound_252;
  long stride_253;
  long offset_254;
  long size_255;
  long lbound_256;
  long ubound_257;
  long lbound_258;
  long ubound_259;
  long stride_260;
  long offset_261;
  long size_262;
  long lbound_263;
  long ubound_264;
  long lbound_265;
  long ubound_266;
  long stride_267;
  long offset_268;
  long size_269;

  device GFReal * p_cup;
  long lbound_270;
  long ubound_271;
  long lbound_272;
  long ubound_273;
  long stride_274;
  long offset_275;
  long size_276;
  long lbound_277;
  long ubound_278;
  long lbound_279;
  long ubound_280;
  long stride_281;
  long offset_282;
  long size_283;

  device GFReal * po_cup;
  GFReal power_entr;
  long lbound_284;
  long ubound_285;
  long offset_286;
  long size_287;
  long lbound_288;
  long ubound_289;
  long lbound_290;
  long ubound_291;
  long stride_292;
  long offset_293;
  long size_294;
  long lbound_295;
  long ubound_296;
  long lbound_297;
  long ubound_298;
  long stride_299;
  long offset_300;
  long size_301;

  device GFReal * q_cup;
  long lbound_302;
  long ubound_303;
  long lbound_304;
  long ubound_305;
  long stride_306;
  long offset_307;
  long size_308;

  device GFReal * qco;
  long lbound_309;
  long ubound_310;
  long lbound_311;
  long ubound_312;
  long stride_313;
  long offset_314;
  long size_315;

  device GFReal * qes;
  long lbound_316;
  long ubound_317;
  long lbound_318;
  long ubound_319;
  long stride_320;
  long offset_321;
  long size_322;

  device GFReal * qes_cup;
  long lbound_323;
  long ubound_324;
  long lbound_325;
  long ubound_326;
  long stride_327;
  long offset_328;
  long size_329;

  device GFReal * qeso;
  long lbound_330;
  long ubound_331;
  long lbound_332;
  long ubound_333;
  long stride_334;
  long offset_335;
  long size_336;

  device GFReal * qeso_cup;
  long lbound_337;
  long ubound_338;
  long offset_339;
  long size_340;

  device GFReal * qkbo;
  long lbound_341;
  long ubound_342;
  long lbound_343;
  long ubound_344;
  long stride_345;
  long offset_346;
  long size_347;
  long lbound_348;
  long ubound_349;
  long lbound_350;
  long ubound_351;
  long stride_352;
  long offset_353;
  long size_354;

  device GFReal * qo_cup;
  long lbound_355;
  long ubound_356;
  long lbound_357;
  long ubound_358;
  long stride_359;
  long offset_360;
  long size_361;

  device GFReal * qrco;
  GFReal subdown;
  GFReal subin;
  long lbound_362;
  long ubound_363;
  long lbound_364;
  long ubound_365;
  long stride_366;
  long offset_367;
  long size_368;
  long lbound_369;
  long ubound_370;
  long lbound_371;
  long ubound_372;
  long stride_373;
  long offset_374;
  long size_375;

  device GFReal * t_cup;
  long lbound_376;
  long ubound_377;
  long lbound_378;
  long ubound_379;
  long stride_380;
  long offset_381;
  long size_382;
  long lbound_383;
  long ubound_384;
  long lbound_385;
  long ubound_386;
  long stride_387;
  long offset_388;
  long size_389;

  device GFReal * tn_cup;
  GFReal totmas;
  GFReal trash;
  long lbound_390;
  long ubound_391;
  long offset_392;
  long size_393;
  int tun_lim;
  long lbound_394;
  long ubound_395;
  long lbound_396;
  long ubound_397;
  long stride_398;
  long offset_399;
  long size_400;

  device GFReal * up_massdetr;
  long lbound_401;
  long ubound_402;
  long lbound_403;
  long ubound_404;
  long stride_405;
  long offset_406;
  long size_407;

  device GFReal * up_massdetro;
  long lbound_408;
  long ubound_409;
  long lbound_410;
  long ubound_411;
  long stride_412;
  long offset_413;
  long size_414;

  device GFReal * up_massentr;
  long lbound_415;
  long ubound_416;
  long lbound_417;
  long ubound_418;
  long stride_419;
  long offset_420;
  long size_421;

  device GFReal * up_massentro;
  long lbound_422;
  long ubound_423;
  long offset_424;
  long size_425;

  device GFReal * xaa0;
  long lbound_426;
  long ubound_427;
  long lbound_428;
  long ubound_429;
  long stride_430;
  long offset_431;
  long size_432;

  device GFReal * xdby;
  GFReal xff_max;
  GFReal xff_shal[9];
  long lbound_433;
  long ubound_434;
  long lbound_435;
  long ubound_436;
  long stride_437;
  long offset_438;
  long size_439;

  device GFReal * xhc;
  long lbound_440;
  long ubound_441;
  long lbound_442;
  long ubound_443;
  long stride_444;
  long offset_445;
  long size_446;

  device GFReal * xhe;
  long lbound_447;
  long ubound_448;
  long lbound_449;
  long ubound_450;
  long stride_451;
  long offset_452;
  long size_453;

  device GFReal * xhe_cup;
  long lbound_454;
  long ubound_455;
  long lbound_456;
  long ubound_457;
  long stride_458;
  long offset_459;
  long size_460;

  device GFReal * xhes;
  long lbound_461;
  long ubound_462;
  long lbound_463;
  long ubound_464;
  long stride_465;
  long offset_466;
  long size_467;

  device GFReal * xhes_cup;
  long lbound_468;
  long ubound_469;
  long offset_470;
  long size_471;

  device GFReal * xhkb;
  GFReal xkshal;
  long lbound_472;
  long ubound_473;
  long offset_474;
  long size_475;
  long lbound_476;
  long ubound_477;
  long offset_478;
  long size_479;

  device GFReal * xland1;
  long lbound_480;
  long ubound_481;
  long offset_482;
  long size_483;

  device GFReal * xmb;
  long lbound_484;
  long ubound_485;
  long offset_486;
  long size_487;
  long lbound_488;
  long ubound_489;
  long offset_490;
  long size_491;

  device GFReal * xmbmax;
  long lbound_492;
  long ubound_493;
  long lbound_494;
  long ubound_495;
  long stride_496;
  long offset_497;
  long size_498;

  device GFReal * xq;
  long lbound_499;
  long ubound_500;
  long lbound_501;
  long ubound_502;
  long stride_503;
  long offset_504;
  long size_505;

  device GFReal * xq_cup;
  long lbound_506;
  long ubound_507;
  long lbound_508;
  long ubound_509;
  long stride_510;
  long offset_511;
  long size_512;

  device GFReal * xqes;
  long lbound_513;
  long ubound_514;
  long lbound_515;
  long ubound_516;
  long stride_517;
  long offset_518;
  long size_519;

  device GFReal * xqes_cup;
  long lbound_520;
  long ubound_521;
  long lbound_522;
  long ubound_523;
  long stride_524;
  long offset_525;
  long size_526;

  device GFReal * xt;
  long lbound_527;
  long ubound_528;
  long lbound_529;
  long ubound_530;
  long stride_531;
  long offset_532;
  long size_533;

  device GFReal * xt_cup;
  long lbound_534;
  long ubound_535;
  long lbound_536;
  long ubound_537;
  long stride_538;
  long offset_539;
  long size_540;

  device GFReal * xz;
  long lbound_541;
  long ubound_542;
  long lbound_543;
  long ubound_544;
  long stride_545;
  long offset_546;
  long size_547;

  device GFReal * xz_cup;
  long lbound_548;
  long ubound_549;
  long lbound_550;
  long ubound_551;
  long stride_552;
  long offset_553;
  long size_554;

  device GFReal * xzu;
  long lbound_555;
  long ubound_556;
  long lbound_557;
  long ubound_558;
  long stride_559;
  long offset_560;
  long size_561;

  device GFReal * z;
  long lbound_562;
  long ubound_563;
  long offset_564;
  long size_565;
  long lbound_566;
  long ubound_567;
  long lbound_568;
  long ubound_569;
  long stride_570;
  long offset_571;
  long size_572;

  device GFReal * z_cup;
  GFReal zkbmax;
  long lbound_573;
  long ubound_574;
  long lbound_575;
  long ubound_576;
  long stride_577;
  long offset_578;
  long size_579;
  long lbound_580;
  long ubound_581;
  long lbound_582;
  long ubound_583;
  long stride_584;
  long offset_585;
  long size_586;

  device GFReal * zo_cup;
  long lbound_587;
  long ubound_588;
  long offset_589;
  long size_590;
  long lbound_591;
  long ubound_592;
  long offset_593;
  long size_594;
  long lbound_595;
  long ubound_596;
  long lbound_597;
  long ubound_598;
  long stride_599;
  long offset_600;
  long size_601;

  device GFReal * zu;
  GFReal zufinal;
  long lbound_602;
  long ubound_603;
  long offset_604;
  long size_605;

  device GFReal * zuhe;
  long lbound_606;
  long ubound_607;
  long lbound_608;
  long ubound_609;
  long stride_610;
  long offset_611;
  long size_612;

  device GFReal * zuo;
  GFReal zustart;
  long D_6581;
  ulong D_6582;
  ulong D_6583;
  long D_6585;
  ulong D_6586;
  ulong D_6587;
  long D_6589;
  ulong D_6590;
  ulong D_6591;
  long D_6593;
  ulong D_6594;
  ulong D_6595;
  long D_6597;
  ulong D_6598;
  ulong D_6599;
  long D_6601;
  ulong D_6602;
  ulong D_6603;
  long D_6605;
  ulong D_6606;
  ulong D_6607;
  long D_6609;
  ulong D_6610;
  ulong D_6611;
  long D_6613;
  ulong D_6614;
  ulong D_6615;
  long D_6617;
  ulong D_6618;
  ulong D_6619;
  long D_6621;
  ulong D_6622;
  ulong D_6623;
  long D_6625;
  ulong D_6626;
  ulong D_6627;
  long D_6629;
  ulong D_6630;
  ulong D_6631;
  long D_6633;
  ulong D_6634;
  ulong D_6635;
  long D_6637;
  ulong D_6638;
  ulong D_6639;
  long D_6641;
  ulong D_6642;
  ulong D_6643;
  long D_6645;
  ulong D_6646;
  ulong D_6647;
  long D_6649;
  ulong D_6650;
  ulong D_6651;
  long D_6653;
  ulong D_6654;
  ulong D_6655;
  long D_6657;
  ulong D_6658;
  ulong D_6659;
  long D_6661;
  ulong D_6662;
  ulong D_6663;
  long D_6665;
  ulong D_6666;
  ulong D_6667;
  long D_6669;
  ulong D_6670;
  ulong D_6671;
  long D_6673;
  ulong D_6674;
  ulong D_6675;
  long D_6677;
  ulong D_6678;
  ulong D_6679;
  long D_6681;
  ulong D_6682;
  ulong D_6683;
  long D_6685;
  ulong D_6686;
  ulong D_6687;
  long D_6689;
  ulong D_6690;
  ulong D_6691;
  long D_6693;
  ulong D_6694;
  ulong D_6695;
  long D_6697;
  ulong D_6698;
  ulong D_6699;
  long D_6701;
  ulong D_6702;
  ulong D_6703;
  long D_6705;
  ulong D_6706;
  ulong D_6707;
  long D_6709;
  ulong D_6710;
  ulong D_6711;
  long D_6713;
  ulong D_6714;
  ulong D_6715;
  long D_6717;
  ulong D_6718;
  ulong D_6719;
  long D_6721;
  ulong D_6722;
  ulong D_6723;
  long D_6725;
  ulong D_6726;
  ulong D_6727;
  long D_6729;
  ulong D_6730;
  ulong D_6731;
  long D_6733;
  ulong D_6734;
  ulong D_6735;
  long D_6737;
  ulong D_6738;
  ulong D_6739;
  long D_6741;
  ulong D_6742;
  ulong D_6743;
  long D_6745;
  ulong D_6746;
  ulong D_6747;
  long D_6749;
  ulong D_6750;
  ulong D_6751;
  long D_6753;
  ulong D_6754;
  ulong D_6755;
  long D_6757;
  ulong D_6758;
  ulong D_6759;
  long D_6761;
  ulong D_6762;
  ulong D_6763;
  long D_6765;
  ulong D_6766;
  ulong D_6767;
  long D_6769;
  ulong D_6770;
  ulong D_6771;
  long D_6773;
  ulong D_6774;
  ulong D_6775;
  long D_6777;
  ulong D_6778;
  ulong D_6779;
  long D_6781;
  ulong D_6782;
  ulong D_6783;
  long D_6785;
  ulong D_6786;
  ulong D_6787;
  long D_6789;
  ulong D_6790;
  ulong D_6791;
  long D_6793;
  ulong D_6794;
  ulong D_6795;
  long D_6797;
  ulong D_6798;
  ulong D_6799;
  long D_6801;
  ulong D_6802;
  ulong D_6803;
  long D_6805;
  ulong D_6806;
  ulong D_6807;
  long D_6809;
  ulong D_6810;
  ulong D_6811;
  long D_6813;
  ulong D_6814;
  ulong D_6815;
  long D_6817;
  ulong D_6818;
  ulong D_6819;
  long D_6821;
  ulong D_6822;
  ulong D_6823;
  long D_6825;
  ulong D_6826;
  ulong D_6827;
  long D_6829;
  ulong D_6830;
  ulong D_6831;
  long D_6833;
  ulong D_6834;
  ulong D_6835;
  long D_6837;
  ulong D_6838;
  ulong D_6839;
  long D_6841;
  ulong D_6842;
  ulong D_6843;
  long D_6845;
  ulong D_6846;
  ulong D_6847;
  long D_6849;
  ulong D_6850;
  ulong D_6851;
  long D_6853;
  ulong D_6854;
  ulong D_6855;
  long D_6857;
  ulong D_6858;
  ulong D_6859;
  long D_6861;
  ulong D_6862;
  ulong D_6863;
  long D_6865;
  ulong D_6866;
  ulong D_6867;
  long D_6869;
  ulong D_6870;
  ulong D_6871;
  long D_6873;
  ulong D_6874;
  ulong D_6875;
  long D_6877;
  ulong D_6878;
  ulong D_6879;
  long D_6880;
  ulong D_6881;
  ulong D_6882;
  long D_6883;
  ulong D_6884;
  ulong D_6885;
  long D_6886;
  ulong D_6887;
  ulong D_6888;
  long D_6889;
  ulong D_6890;
  ulong D_6891;
  long D_6892;
  ulong D_6893;
  ulong D_6894;
  long D_6895;
  ulong D_6896;
  ulong D_6897;
  long D_6898;
  ulong D_6899;
  ulong D_6900;
  long D_6901;
  ulong D_6902;
  ulong D_6903;
  long D_6904;
  ulong D_6905;
  ulong D_6906;
  long D_6907;
  ulong D_6908;
  ulong D_6909;
  long D_6910;
  ulong D_6911;
  ulong D_6912;
  long D_6913;
  ulong D_6914;
  ulong D_6915;
  long D_6916;
  ulong D_6917;
  ulong D_6918;
  long D_6919;
  ulong D_6920;
  ulong D_6921;
  long D_6922;
  ulong D_6923;
  ulong D_6924;
  long D_6925;
  ulong D_6926;
  ulong D_6927;
  long D_6928;
  ulong D_6929;
  ulong D_6930;
  long D_6931;
  ulong D_6932;
  ulong D_6933;
  long D_6934;
  ulong D_6935;
  ulong D_6936;
  long D_6937;
  ulong D_6938;
  ulong D_6939;
  long D_6940;
  ulong D_6941;
  ulong D_6942;
  long D_6943;
  ulong D_6944;
  ulong D_6945;
  long D_6946;
  ulong D_6947;
  ulong D_6948;
  long D_6949;
  ulong D_6950;
  ulong D_6951;
  long D_6952;
  ulong D_6953;
  ulong D_6954;

    {
      lbound_484 = (long) *its;
      ubound_485 = (long) *ite;
      size_487 = (1 - lbound_484) + ubound_485;
      size_487 = gf_max(size_487, 0);
      D_6952 = size_487 + -1;
      D_6953 = (ulong) (ulong) (size_487) * 32;
      D_6954 = (ulong) (size_487) * 4;
      offset_486 = -(lbound_484);
      lbound_472 = (long) *its;
      ubound_473 = (long) *ite;
      size_475 = (1 - lbound_472) + ubound_473;
      size_475 = gf_max(size_475, 0);
      D_6949 = size_475 + -1;
      D_6950 = (ulong) (ulong) (size_475) * 32;
      D_6951 = (ulong) (size_475) * 4;
      offset_474 = -(lbound_472);
      lbound_207 = (long) *its;
      ubound_208 = (long) *ite;
      size_210 = (1 - lbound_207) + ubound_208;
      size_210 = gf_max(size_210, 0);
      D_6946 = size_210 + -1;
      D_6947 = (ulong) (ulong) (size_210) * 400;
      D_6948 = (ulong) (size_210) * 50;
      offset_209 = -(lbound_207);
      lbound_573 = (long) *its;
      ubound_574 = (long) *ite;
      stride_577 = (1 - lbound_573) + ubound_574;
      stride_577 = gf_max(stride_577, 0);
      lbound_575 = (long) *kts;
      ubound_576 = (long) *kte;
      size_579 = ((1 - lbound_575) + ubound_576) * stride_577;
      size_579 = gf_max(size_579, 0);
      D_6943 = size_579 + -1;
      D_6944 = (ulong) (ulong) (size_579) * 32;
      D_6945 = (ulong) (size_579) * 4;
      offset_578 = -(lbound_573) - lbound_575 * stride_577;
      lbound_362 = (long) *its;
      ubound_363 = (long) *ite;
      stride_366 = (1 - lbound_362) + ubound_363;
      stride_366 = gf_max(stride_366, 0);
      lbound_364 = (long) *kts;
      ubound_365 = (long) *kte;
      size_368 = ((1 - lbound_364) + ubound_365) * stride_366;
      size_368 = gf_max(size_368, 0);
      D_6940 = size_368 + -1;
      D_6941 = (ulong) (ulong) (size_368) * 32;
      D_6942 = (ulong) (size_368) * 4;
      offset_367 = -(lbound_362) - lbound_364 * stride_366;
      lbound_288 = (long) *its;
      ubound_289 = (long) *ite;
      stride_292 = (1 - lbound_288) + ubound_289;
      stride_292 = gf_max(stride_292, 0);
      lbound_290 = (long) *kts;
      ubound_291 = (long) *kte;
      size_294 = ((1 - lbound_290) + ubound_291) * stride_292;
      size_294 = gf_max(size_294, 0);
      D_6937 = size_294 + -1;
      D_6938 = (ulong) (ulong) (size_294) * 32;
      D_6939 = (ulong) (size_294) * 4;
      offset_293 = -(lbound_288) - lbound_290 * stride_292;
      lbound_256 = (long) *its;
      ubound_257 = (long) *ite;
      stride_260 = (1 - lbound_256) + ubound_257;
      stride_260 = gf_max(stride_260, 0);
      lbound_258 = (long) *kts;
      ubound_259 = (long) *kte;
      size_262 = ((1 - lbound_258) + ubound_259) * stride_260;
      size_262 = gf_max(size_262, 0);
      D_6934 = size_262 + -1;
      D_6935 = (ulong) (ulong) (size_262) * 32;
      D_6936 = (ulong) (size_262) * 4;
      offset_261 = -(lbound_256) - lbound_258 * stride_260;
      lbound_562 = (long) *its;
      ubound_563 = (long) *ite;
      size_565 = (1 - lbound_562) + ubound_563;
      size_565 = gf_max(size_565, 0);
      D_6931 = size_565 + -1;
      D_6932 = (ulong) (ulong) (size_565) * 32;
      D_6933 = (ulong) (size_565) * 4;
      offset_564 = -(lbound_562);
      lbound_284 = (long) *its;
      ubound_285 = (long) *ite;
      size_287 = (1 - lbound_284) + ubound_285;
      size_287 = gf_max(size_287, 0);
      D_6928 = size_287 + -1;
      D_6929 = (ulong) (ulong) (size_287) * 32;
      D_6930 = (ulong) (size_287) * 4;
      offset_286 = -(lbound_284);
      lbound_203 = (long) *its;
      ubound_204 = (long) *ite;
      size_206 = (1 - lbound_203) + ubound_204;
      size_206 = gf_max(size_206, 0);
      D_6925 = size_206 + -1;
      D_6926 = (ulong) (ulong) (size_206) * 32;
      D_6927 = (ulong) (size_206) * 4;
      offset_205 = -(lbound_203);
      lbound_376 = (long) *its;
      ubound_377 = (long) *ite;
      stride_380 = (1 - lbound_376) + ubound_377;
      stride_380 = gf_max(stride_380, 0);
      lbound_378 = (long) *kts;
      ubound_379 = (long) *kte;
      size_382 = ((1 - lbound_378) + ubound_379) * stride_380;
      size_382 = gf_max(size_382, 0);
      D_6922 = size_382 + -1;
      D_6923 = (ulong) (ulong) (size_382) * 32;
      D_6924 = (ulong) (size_382) * 4;
      offset_381 = -(lbound_376) - lbound_378 * stride_380;
      lbound_341 = (long) *its;
      ubound_342 = (long) *ite;
      stride_345 = (1 - lbound_341) + ubound_342;
      stride_345 = gf_max(stride_345, 0);
      lbound_343 = (long) *kts;
      ubound_344 = (long) *kte;
      size_347 = ((1 - lbound_343) + ubound_344) * stride_345;
      size_347 = gf_max(size_347, 0);
      D_6919 = size_347 + -1;
      D_6920 = (ulong) (ulong) (size_347) * 32;
      D_6921 = (ulong) (size_347) * 4;
      offset_346 = -(lbound_341) - lbound_343 * stride_345;
      lbound_270 = (long) *its;
      ubound_271 = (long) *ite;
      stride_274 = (1 - lbound_270) + ubound_271;
      stride_274 = gf_max(stride_274, 0);
      lbound_272 = (long) *kts;
      ubound_273 = (long) *kte;
      size_276 = ((1 - lbound_272) + ubound_273) * stride_274;
      size_276 = gf_max(size_276, 0);
      D_6916 = size_276 + -1;
      D_6917 = (ulong) (ulong) (size_276) * 32;
      D_6918 = (ulong) (size_276) * 4;
      offset_275 = -(lbound_270) - lbound_272 * stride_274;
      lbound_8 = (long) *its;
      ubound_9 = (long) *ite;
      size_11 = (1 - lbound_8) + ubound_9;
      size_11 = gf_max(size_11, 0);
      D_6913 = size_11 + -1;
      D_6914 = (ulong) (ulong) (size_11) * 32;
      D_6915 = (ulong) (size_11) * 4;
      offset_10 = -(lbound_8);
      lbound_215 = (long) *its;
      ubound_216 = (long) *ite;
      size_218 = (1 - lbound_215) + ubound_216;
      size_218 = gf_max(size_218, 0);
      D_6910 = size_218 + -1;
      D_6911 = (ulong) (ulong) (size_218) * 32;
      D_6912 = (ulong) (size_218) * 4;
      offset_217 = -(lbound_215);
      lbound_227 = (long) *its;
      ubound_228 = (long) *ite;
      size_230 = (1 - lbound_227) + ubound_228;
      size_230 = gf_max(size_230, 0);
      D_6907 = size_230 + -1;
      D_6908 = (ulong) (ulong) (size_230) * 32;
      D_6909 = (ulong) (size_230) * 4;
      offset_229 = -(lbound_227);
      lbound_587 = (long) *its;
      ubound_588 = (long) *ite;
      size_590 = (1 - lbound_587) + ubound_588;
      size_590 = gf_max(size_590, 0);
      D_6904 = size_590 + -1;
      D_6905 = (ulong) (ulong) (size_590) * 32;
      D_6906 = (ulong) (size_590) * 4;
      offset_589 = -(lbound_587);
      lbound_591 = (long) *its;
      ubound_592 = (long) *ite;
      size_594 = (1 - lbound_591) + ubound_592;
      size_594 = gf_max(size_594, 0);
      D_6901 = size_594 + -1;
      D_6902 = (ulong) (ulong) (size_594) * 32;
      D_6903 = (ulong) (size_594) * 4;
      offset_593 = -(lbound_591);
      lbound_219 = (long) *its;
      ubound_220 = (long) *ite;
      size_222 = (1 - lbound_219) + ubound_220;
      size_222 = gf_max(size_222, 0);
      D_6898 = size_222 + -1;
      D_6899 = (ulong) (ulong) (size_222) * 32;
      D_6900 = (ulong) (size_222) * 4;
      offset_221 = -(lbound_219);
      lbound_231 = (long) *its;
      ubound_232 = (long) *ite;
      size_234 = (1 - lbound_231) + ubound_232;
      size_234 = gf_max(size_234, 0);
      D_6895 = size_234 + -1;
      D_6896 = (ulong) (ulong) (size_234) * 32;
      D_6897 = (ulong) (size_234) * 4;
      offset_233 = -(lbound_231);
      lbound_27 = (long) *its;
      ubound_28 = (long) *ite;
      stride_31 = (1 - lbound_27) + ubound_28;
      stride_31 = gf_max(stride_31, 0);
      lbound_29 = (long) *kts;
      ubound_30 = (long) *kte;
      size_33 = ((1 - lbound_29) + ubound_30) * stride_31;
      size_33 = gf_max(size_33, 0);
      D_6892 = size_33 + -1;
      D_6893 = (ulong) (ulong) (size_33) * 32;
      D_6894 = (ulong) (size_33) * 4;
      offset_32 = -(lbound_27) - lbound_29 * stride_31;
      lbound_76 = (long) *its;
      ubound_77 = (long) *ite;
      stride_80 = (1 - lbound_76) + ubound_77;
      stride_80 = gf_max(stride_80, 0);
      lbound_78 = (long) *kts;
      ubound_79 = (long) *kte;
      size_82 = ((1 - lbound_78) + ubound_79) * stride_80;
      size_82 = gf_max(size_82, 0);
      D_6889 = size_82 + -1;
      D_6890 = (ulong) (ulong) (size_82) * 32;
      D_6891 = (ulong) (size_82) * 4;
      offset_81 = -(lbound_76) - lbound_78 * stride_80;
      lbound_390 = (long) *its;
      ubound_391 = (long) *ite;
      size_393 = (1 - lbound_390) + ubound_391;
      size_393 = gf_max(size_393, 0);
      D_6886 = size_393 + -1;
      D_6887 = (ulong) (ulong) (size_393) * 32;
      D_6888 = (ulong) (size_393) * 4;
      offset_392 = -(lbound_390);
      lbound_249 = (long) *its;
      ubound_250 = (long) *ite;
      stride_253 = (1 - lbound_249) + ubound_250;
      stride_253 = gf_max(stride_253, 0);
      lbound_251 = (long) *kts;
      ubound_252 = (long) *kte;
      size_255 = ((1 - lbound_251) + ubound_252) * stride_253;
      size_255 = gf_max(size_255, 0);
      D_6883 = size_255 + -1;
      D_6884 = (ulong) (ulong) (size_255) * 32;
      D_6885 = (ulong) (size_255) * 4;
      offset_254 = -(lbound_249) - lbound_251 * stride_253;
      lbound_235 = (long) *its;
      ubound_236 = (long) *ite;
      stride_239 = (1 - lbound_235) + ubound_236;
      stride_239 = gf_max(stride_239, 0);
      lbound_237 = (long) *kts;
      ubound_238 = (long) *kte;
      size_241 = ((1 - lbound_237) + ubound_238) * stride_239;
      size_241 = gf_max(size_241, 0);
      D_6880 = size_241 + -1;
      D_6881 = (ulong) (ulong) (size_241) * 32;
      D_6882 = (ulong) (size_241) * 4;
      offset_240 = -(lbound_235) - lbound_237 * stride_239;
      lbound_242 = (long) *its;
      ubound_243 = (long) *ite;
      stride_246 = (1 - lbound_242) + ubound_243;
      stride_246 = gf_max(stride_246, 0);
      lbound_244 = (long) *kts;
      ubound_245 = (long) *kte;
      size_248 = ((1 - lbound_244) + ubound_245) * stride_246;
      size_248 = gf_max(size_248, 0);
      D_6877 = size_248 + -1;
      D_6878 = (ulong) (ulong) (size_248) * 32;
      D_6879 = (ulong) (size_248) * 4;
      offset_247 = -(lbound_242) - lbound_244 * stride_246;
      lbound_0 = (long) *its;
      ubound_1 = (long) *ite;
      size_3 = (1 - lbound_0) + ubound_1;
      size_3 = gf_max(size_3, 0);
      D_6873 = size_3 + -1;
      D_6874 = (ulong) (ulong) (size_3) * 32;
      D_6875 = (ulong) (size_3) * 4;
      D_6876 = (device void *) workspace.allocate (gf_max((ulong) (size_3 * 4), 1));
      aa0 = (device GFReal *) D_6876;
      offset_2 = -(lbound_0);
      lbound_4 = (long) *its;
      ubound_5 = (long) *ite;
      size_7 = (1 - lbound_4) + ubound_5;
      size_7 = gf_max(size_7, 0);
      D_6869 = size_7 + -1;
      D_6870 = (ulong) (ulong) (size_7) * 32;
      D_6871 = (ulong) (size_7) * 4;
      D_6872 = (device void *) workspace.allocate (gf_max((ulong) (size_7 * 4), 1));
      aa1 = (device GFReal *) D_6872;
      offset_6 = -(lbound_4);
      lbound_12 = (long) *its;
      ubound_13 = (long) *ite;
      size_15 = (1 - lbound_12) + ubound_13;
      size_15 = gf_max(size_15, 0);
      D_6865 = size_15 + -1;
      D_6866 = (ulong) (ulong) (size_15) * 32;
      D_6867 = (ulong) (size_15) * 4;
      D_6868 = (device void *) workspace.allocate (gf_max((ulong) (size_15 * 4), 1));
      cap_max = (device GFReal *) D_6868;
      offset_14 = -(lbound_12);
      lbound_16 = (long) *its;
      ubound_17 = (long) *ite;
      size_19 = (1 - lbound_16) + ubound_17;
      size_19 = gf_max(size_19, 0);
      D_6861 = size_19 + -1;
      D_6862 = (ulong) (ulong) (size_19) * 32;
      D_6863 = (ulong) (size_19) * 4;
      D_6864 = (device void *) workspace.allocate (gf_max((ulong) (size_19 * 4), 1));
      cap_max_increment = (device GFReal *) D_6864;
      offset_18 = -(lbound_16);
      lbound_20 = (long) *its;
      ubound_21 = (long) *ite;
      stride_24 = (1 - lbound_20) + ubound_21;
      stride_24 = gf_max(stride_24, 0);
      lbound_22 = (long) *kts;
      ubound_23 = (long) *kte;
      size_26 = ((1 - lbound_22) + ubound_23) * stride_24;
      size_26 = gf_max(size_26, 0);
      D_6857 = size_26 + -1;
      D_6858 = (ulong) (ulong) (size_26) * 32;
      D_6859 = (ulong) (size_26) * 4;
      D_6860 = (device void *) workspace.allocate (gf_max((ulong) (size_26 * 4), 1));
      cd = (device GFReal *) D_6860;
      offset_25 = -(lbound_20) - lbound_22 * stride_24;
      lbound_34 = (long) *its;
      ubound_35 = (long) *ite;
      stride_38 = (1 - lbound_34) + ubound_35;
      stride_38 = gf_max(stride_38, 0);
      lbound_36 = (long) *kts;
      ubound_37 = (long) *kte;
      size_40 = ((1 - lbound_36) + ubound_37) * stride_38;
      size_40 = gf_max(size_40, 0);
      D_6853 = size_40 + -1;
      D_6854 = (ulong) (ulong) (size_40) * 32;
      D_6855 = (ulong) (size_40) * 4;
      D_6856 = (device void *) workspace.allocate (gf_max((ulong) (size_40 * 4), 1));
      dby = (device GFReal *) D_6856;
      offset_39 = -(lbound_34) - lbound_36 * stride_38;
      lbound_41 = (long) *its;
      ubound_42 = (long) *ite;
      stride_45 = (1 - lbound_41) + ubound_42;
      stride_45 = gf_max(stride_45, 0);
      lbound_43 = (long) *kts;
      ubound_44 = (long) *kte;
      size_47 = ((1 - lbound_43) + ubound_44) * stride_45;
      size_47 = gf_max(size_47, 0);
      D_6849 = size_47 + -1;
      D_6850 = (ulong) (ulong) (size_47) * 32;
      D_6851 = (ulong) (size_47) * 4;
      D_6852 = (device void *) workspace.allocate (gf_max((ulong) (size_47 * 4), 1));
      dbyo = (device GFReal *) D_6852;
      offset_46 = -(lbound_41) - lbound_43 * stride_45;
      lbound_48 = (long) *its;
      ubound_49 = (long) *ite;
      stride_52 = (1 - lbound_48) + ubound_49;
      stride_52 = gf_max(stride_52, 0);
      lbound_50 = (long) *kts;
      ubound_51 = (long) *kte;
      size_54 = ((1 - lbound_50) + ubound_51) * stride_52;
      size_54 = gf_max(size_54, 0);
      D_6845 = size_54 + -1;
      D_6846 = (ulong) (ulong) (size_54) * 32;
      D_6847 = (ulong) (size_54) * 4;
      D_6848 = (device void *) workspace.allocate (gf_max((ulong) (size_54 * 4), 1));
      dellah = (device GFReal *) D_6848;
      offset_53 = -(lbound_48) - lbound_50 * stride_52;
      lbound_55 = (long) *its;
      ubound_56 = (long) *ite;
      stride_59 = (1 - lbound_55) + ubound_56;
      stride_59 = gf_max(stride_59, 0);
      lbound_57 = (long) *kts;
      ubound_58 = (long) *kte;
      size_61 = ((1 - lbound_57) + ubound_58) * stride_59;
      size_61 = gf_max(size_61, 0);
      D_6841 = size_61 + -1;
      D_6842 = (ulong) (ulong) (size_61) * 32;
      D_6843 = (ulong) (size_61) * 4;
      D_6844 = (device void *) workspace.allocate (gf_max((ulong) (size_61 * 4), 1));
      dellaq = (device GFReal *) D_6844;
      offset_60 = -(lbound_55) - lbound_57 * stride_59;
      lbound_62 = (long) *its;
      ubound_63 = (long) *ite;
      stride_66 = (1 - lbound_62) + ubound_63;
      stride_66 = gf_max(stride_66, 0);
      lbound_64 = (long) *kts;
      ubound_65 = (long) *kte;
      size_68 = ((1 - lbound_64) + ubound_65) * stride_66;
      size_68 = gf_max(size_68, 0);
      D_6837 = size_68 + -1;
      D_6838 = (ulong) (ulong) (size_68) * 32;
      D_6839 = (ulong) (size_68) * 4;
      D_6840 = (device void *) workspace.allocate (gf_max((ulong) (size_68 * 4), 1));
      dellaqc = (device GFReal *) D_6840;
      offset_67 = -(lbound_62) - lbound_64 * stride_66;
      lbound_69 = (long) *its;
      ubound_70 = (long) *ite;
      stride_73 = (1 - lbound_69) + ubound_70;
      stride_73 = gf_max(stride_73, 0);
      lbound_71 = (long) *kts;
      ubound_72 = (long) *kte;
      size_75 = ((1 - lbound_71) + ubound_72) * stride_73;
      size_75 = gf_max(size_75, 0);
      D_6833 = size_75 + -1;
      D_6834 = (ulong) (ulong) (size_75) * 32;
      D_6835 = (ulong) (size_75) * 4;
      D_6836 = (device void *) workspace.allocate (gf_max((ulong) (size_75 * 4), 1));
      dellat = (device GFReal *) D_6836;
      offset_74 = -(lbound_69) - lbound_71 * stride_73;
      lbound_83 = (long) *its;
      ubound_84 = (long) *ite;
      stride_87 = (1 - lbound_83) + ubound_84;
      stride_87 = gf_max(stride_87, 0);
      lbound_85 = (long) *kts;
      ubound_86 = (long) *kte;
      size_89 = ((1 - lbound_85) + ubound_86) * stride_87;
      size_89 = gf_max(size_89, 0);
      D_6829 = size_89 + -1;
      D_6830 = (ulong) (ulong) (size_89) * 32;
      D_6831 = (ulong) (size_89) * 4;
      D_6832 = (device void *) workspace.allocate (gf_max((ulong) (size_89 * 4), 1));
      dsubh = (device GFReal *) D_6832;
      offset_88 = -(lbound_83) - lbound_85 * stride_87;
      lbound_90 = (long) *its;
      ubound_91 = (long) *ite;
      stride_94 = (1 - lbound_90) + ubound_91;
      stride_94 = gf_max(stride_94, 0);
      lbound_92 = (long) *kts;
      ubound_93 = (long) *kte;
      size_96 = ((1 - lbound_92) + ubound_93) * stride_94;
      size_96 = gf_max(size_96, 0);
      D_6825 = size_96 + -1;
      D_6826 = (ulong) (ulong) (size_96) * 32;
      D_6827 = (ulong) (size_96) * 4;
      D_6828 = (device void *) workspace.allocate (gf_max((ulong) (size_96 * 4), 1));
      dsubq = (device GFReal *) D_6828;
      offset_95 = -(lbound_90) - lbound_92 * stride_94;
      lbound_97 = (long) *its;
      ubound_98 = (long) *ite;
      stride_101 = (1 - lbound_97) + ubound_98;
      stride_101 = gf_max(stride_101, 0);
      lbound_99 = (long) *kts;
      ubound_100 = (long) *kte;
      size_103 = ((1 - lbound_99) + ubound_100) * stride_101;
      size_103 = gf_max(size_103, 0);
      D_6821 = size_103 + -1;
      D_6822 = (ulong) (ulong) (size_103) * 32;
      D_6823 = (ulong) (size_103) * 4;
      D_6824 = (device void *) workspace.allocate (gf_max((ulong) (size_103 * 4), 1));
      dsubt = (device GFReal *) D_6824;
      offset_102 = -(lbound_97) - lbound_99 * stride_101;
      lbound_104 = (long) *its;
      ubound_105 = (long) *ite;
      stride_108 = (1 - lbound_104) + ubound_105;
      stride_108 = gf_max(stride_108, 0);
      lbound_106 = (long) *kts;
      ubound_107 = (long) *kte;
      size_110 = ((1 - lbound_106) + ubound_107) * stride_108;
      size_110 = gf_max(size_110, 0);
      D_6817 = size_110 + -1;
      D_6818 = (ulong) (ulong) (size_110) * 32;
      D_6819 = (ulong) (size_110) * 4;
      D_6820 = (device void *) workspace.allocate (gf_max((ulong) (size_110 * 4), 1));
      entr_rate_2d = (device GFReal *) D_6820;
      offset_109 = -(lbound_104) - lbound_106 * stride_108;
      lbound_111 = (long) *its;
      ubound_112 = (long) *ite;
      stride_115 = (1 - lbound_111) + ubound_112;
      stride_115 = gf_max(stride_115, 0);
      lbound_113 = (long) *kts;
      ubound_114 = (long) *kte;
      size_117 = ((1 - lbound_113) + ubound_114) * stride_115;
      size_117 = gf_max(size_117, 0);
      D_6813 = size_117 + -1;
      D_6814 = (ulong) (ulong) (size_117) * 32;
      D_6815 = (ulong) (size_117) * 4;
      D_6816 = (device void *) workspace.allocate (gf_max((ulong) (size_117 * 4), 1));
      gamma_cup = (device GFReal *) D_6816;
      offset_116 = -(lbound_111) - lbound_113 * stride_115;
      lbound_118 = (long) *its;
      ubound_119 = (long) *ite;
      stride_122 = (1 - lbound_118) + ubound_119;
      stride_122 = gf_max(stride_122, 0);
      lbound_120 = (long) *kts;
      ubound_121 = (long) *kte;
      size_124 = ((1 - lbound_120) + ubound_121) * stride_122;
      size_124 = gf_max(size_124, 0);
      D_6809 = size_124 + -1;
      D_6810 = (ulong) (ulong) (size_124) * 32;
      D_6811 = (ulong) (size_124) * 4;
      D_6812 = (device void *) workspace.allocate (gf_max((ulong) (size_124 * 4), 1));
      gammao_cup = (device GFReal *) D_6812;
      offset_123 = -(lbound_118) - lbound_120 * stride_122;
      lbound_125 = (long) *its;
      ubound_126 = (long) *ite;
      stride_129 = (1 - lbound_125) + ubound_126;
      stride_129 = gf_max(stride_129, 0);
      lbound_127 = (long) *kts;
      ubound_128 = (long) *kte;
      size_131 = ((1 - lbound_127) + ubound_128) * stride_129;
      size_131 = gf_max(size_131, 0);
      D_6805 = size_131 + -1;
      D_6806 = (ulong) (ulong) (size_131) * 32;
      D_6807 = (ulong) (size_131) * 4;
      D_6808 = (device void *) workspace.allocate (gf_max((ulong) (size_131 * 4), 1));
      hc = (device GFReal *) D_6808;
      offset_130 = -(lbound_125) - lbound_127 * stride_129;
      lbound_132 = (long) *its;
      ubound_133 = (long) *ite;
      stride_136 = (1 - lbound_132) + ubound_133;
      stride_136 = gf_max(stride_136, 0);
      lbound_134 = (long) *kts;
      ubound_135 = (long) *kte;
      size_138 = ((1 - lbound_134) + ubound_135) * stride_136;
      size_138 = gf_max(size_138, 0);
      D_6801 = size_138 + -1;
      D_6802 = (ulong) (ulong) (size_138) * 32;
      D_6803 = (ulong) (size_138) * 4;
      D_6804 = (device void *) workspace.allocate (gf_max((ulong) (size_138 * 4), 1));
      hco = (device GFReal *) D_6804;
      offset_137 = -(lbound_132) - lbound_134 * stride_136;
      lbound_139 = (long) *its;
      ubound_140 = (long) *ite;
      stride_143 = (1 - lbound_139) + ubound_140;
      stride_143 = gf_max(stride_143, 0);
      lbound_141 = (long) *kts;
      ubound_142 = (long) *kte;
      size_145 = ((1 - lbound_141) + ubound_142) * stride_143;
      size_145 = gf_max(size_145, 0);
      D_6797 = size_145 + -1;
      D_6798 = (ulong) (ulong) (size_145) * 32;
      D_6799 = (ulong) (size_145) * 4;
      D_6800 = (device void *) workspace.allocate (gf_max((ulong) (size_145 * 4), 1));
      he = (device GFReal *) D_6800;
      offset_144 = -(lbound_139) - lbound_141 * stride_143;
      lbound_146 = (long) *its;
      ubound_147 = (long) *ite;
      stride_150 = (1 - lbound_146) + ubound_147;
      stride_150 = gf_max(stride_150, 0);
      lbound_148 = (long) *kts;
      ubound_149 = (long) *kte;
      size_152 = ((1 - lbound_148) + ubound_149) * stride_150;
      size_152 = gf_max(size_152, 0);
      D_6793 = size_152 + -1;
      D_6794 = (ulong) (ulong) (size_152) * 32;
      D_6795 = (ulong) (size_152) * 4;
      D_6796 = (device void *) workspace.allocate (gf_max((ulong) (size_152 * 4), 1));
      he_cup = (device GFReal *) D_6796;
      offset_151 = -(lbound_146) - lbound_148 * stride_150;
      lbound_153 = (long) *its;
      ubound_154 = (long) *ite;
      stride_157 = (1 - lbound_153) + ubound_154;
      stride_157 = gf_max(stride_157, 0);
      lbound_155 = (long) *kts;
      ubound_156 = (long) *kte;
      size_159 = ((1 - lbound_155) + ubound_156) * stride_157;
      size_159 = gf_max(size_159, 0);
      D_6789 = size_159 + -1;
      D_6790 = (ulong) (ulong) (size_159) * 32;
      D_6791 = (ulong) (size_159) * 4;
      D_6792 = (device void *) workspace.allocate (gf_max((ulong) (size_159 * 4), 1));
      heo = (device GFReal *) D_6792;
      offset_158 = -(lbound_153) - lbound_155 * stride_157;
      lbound_160 = (long) *its;
      ubound_161 = (long) *ite;
      stride_164 = (1 - lbound_160) + ubound_161;
      stride_164 = gf_max(stride_164, 0);
      lbound_162 = (long) *kts;
      ubound_163 = (long) *kte;
      size_166 = ((1 - lbound_162) + ubound_163) * stride_164;
      size_166 = gf_max(size_166, 0);
      D_6785 = size_166 + -1;
      D_6786 = (ulong) (ulong) (size_166) * 32;
      D_6787 = (ulong) (size_166) * 4;
      D_6788 = (device void *) workspace.allocate (gf_max((ulong) (size_166 * 4), 1));
      heo_cup = (device GFReal *) D_6788;
      offset_165 = -(lbound_160) - lbound_162 * stride_164;
      lbound_167 = (long) *its;
      ubound_168 = (long) *ite;
      stride_171 = (1 - lbound_167) + ubound_168;
      stride_171 = gf_max(stride_171, 0);
      lbound_169 = (long) *kts;
      ubound_170 = (long) *kte;
      size_173 = ((1 - lbound_169) + ubound_170) * stride_171;
      size_173 = gf_max(size_173, 0);
      D_6781 = size_173 + -1;
      D_6782 = (ulong) (ulong) (size_173) * 32;
      D_6783 = (ulong) (size_173) * 4;
      D_6784 = (device void *) workspace.allocate (gf_max((ulong) (size_173 * 4), 1));
      hes = (device GFReal *) D_6784;
      offset_172 = -(lbound_167) - lbound_169 * stride_171;
      lbound_174 = (long) *its;
      ubound_175 = (long) *ite;
      stride_178 = (1 - lbound_174) + ubound_175;
      stride_178 = gf_max(stride_178, 0);
      lbound_176 = (long) *kts;
      ubound_177 = (long) *kte;
      size_180 = ((1 - lbound_176) + ubound_177) * stride_178;
      size_180 = gf_max(size_180, 0);
      D_6777 = size_180 + -1;
      D_6778 = (ulong) (ulong) (size_180) * 32;
      D_6779 = (ulong) (size_180) * 4;
      D_6780 = (device void *) workspace.allocate (gf_max((ulong) (size_180 * 4), 1));
      hes_cup = (device GFReal *) D_6780;
      offset_179 = -(lbound_174) - lbound_176 * stride_178;
      lbound_181 = (long) *its;
      ubound_182 = (long) *ite;
      stride_185 = (1 - lbound_181) + ubound_182;
      stride_185 = gf_max(stride_185, 0);
      lbound_183 = (long) *kts;
      ubound_184 = (long) *kte;
      size_187 = ((1 - lbound_183) + ubound_184) * stride_185;
      size_187 = gf_max(size_187, 0);
      D_6773 = size_187 + -1;
      D_6774 = (ulong) (ulong) (size_187) * 32;
      D_6775 = (ulong) (size_187) * 4;
      D_6776 = (device void *) workspace.allocate (gf_max((ulong) (size_187 * 4), 1));
      heso = (device GFReal *) D_6776;
      offset_186 = -(lbound_181) - lbound_183 * stride_185;
      lbound_188 = (long) *its;
      ubound_189 = (long) *ite;
      stride_192 = (1 - lbound_188) + ubound_189;
      stride_192 = gf_max(stride_192, 0);
      lbound_190 = (long) *kts;
      ubound_191 = (long) *kte;
      size_194 = ((1 - lbound_190) + ubound_191) * stride_192;
      size_194 = gf_max(size_194, 0);
      D_6769 = size_194 + -1;
      D_6770 = (ulong) (ulong) (size_194) * 32;
      D_6771 = (ulong) (size_194) * 4;
      D_6772 = (device void *) workspace.allocate (gf_max((ulong) (size_194 * 4), 1));
      heso_cup = (device GFReal *) D_6772;
      offset_193 = -(lbound_188) - lbound_190 * stride_192;
      lbound_195 = (long) *its;
      ubound_196 = (long) *ite;
      size_198 = (1 - lbound_195) + ubound_196;
      size_198 = gf_max(size_198, 0);
      D_6765 = size_198 + -1;
      D_6766 = (ulong) (ulong) (size_198) * 32;
      D_6767 = (ulong) (size_198) * 4;
      D_6768 = (device void *) workspace.allocate (gf_max((ulong) (size_198 * 4), 1));
      hkb = (device GFReal *) D_6768;
      offset_197 = -(lbound_195);
      lbound_199 = (long) *its;
      ubound_200 = (long) *ite;
      size_202 = (1 - lbound_199) + ubound_200;
      size_202 = gf_max(size_202, 0);
      D_6761 = size_202 + -1;
      D_6762 = (ulong) (ulong) (size_202) * 32;
      D_6763 = (ulong) (size_202) * 4;
      D_6764 = (device void *) workspace.allocate (gf_max((ulong) (size_202 * 4), 1));
      hkbo = (device GFReal *) D_6764;
      offset_201 = -(lbound_199);
      lbound_211 = (long) *its;
      ubound_212 = (long) *ite;
      size_214 = (1 - lbound_211) + ubound_212;
      size_214 = gf_max(size_214, 0);
      D_6757 = size_214 + -1;
      D_6758 = (ulong) (ulong) (size_214) * 32;
      D_6759 = (ulong) (size_214) * 4;
      D_6760 = (device void *) workspace.allocate (gf_max((ulong) (size_214 * 4), 1));
      jmin = (device int *) D_6760;
      offset_213 = -(lbound_211);
      lbound_223 = (long) *its;
      ubound_224 = (long) *ite;
      size_226 = (1 - lbound_223) + ubound_224;
      size_226 = gf_max(size_226, 0);
      D_6753 = size_226 + -1;
      D_6754 = (ulong) (ulong) (size_226) * 32;
      D_6755 = (ulong) (size_226) * 4;
      D_6756 = (device void *) workspace.allocate (gf_max((ulong) (size_226 * 4), 1));
      kbmax = (device int *) D_6756;
      offset_225 = -(lbound_223);
      lbound_263 = (long) *its;
      ubound_264 = (long) *ite;
      stride_267 = (1 - lbound_263) + ubound_264;
      stride_267 = gf_max(stride_267, 0);
      lbound_265 = (long) *kts;
      ubound_266 = (long) *kte;
      size_269 = ((1 - lbound_265) + ubound_266) * stride_267;
      size_269 = gf_max(size_269, 0);
      D_6749 = size_269 + -1;
      D_6750 = (ulong) (ulong) (size_269) * 32;
      D_6751 = (ulong) (size_269) * 4;
      D_6752 = (device void *) workspace.allocate (gf_max((ulong) (size_269 * 4), 1));
      p_cup = (device GFReal *) D_6752;
      offset_268 = -(lbound_263) - lbound_265 * stride_267;
      lbound_277 = (long) *its;
      ubound_278 = (long) *ite;
      stride_281 = (1 - lbound_277) + ubound_278;
      stride_281 = gf_max(stride_281, 0);
      lbound_279 = (long) *kts;
      ubound_280 = (long) *kte;
      size_283 = ((1 - lbound_279) + ubound_280) * stride_281;
      size_283 = gf_max(size_283, 0);
      D_6745 = size_283 + -1;
      D_6746 = (ulong) (ulong) (size_283) * 32;
      D_6747 = (ulong) (size_283) * 4;
      D_6748 = (device void *) workspace.allocate (gf_max((ulong) (size_283 * 4), 1));
      po_cup = (device GFReal *) D_6748;
      offset_282 = -(lbound_277) - lbound_279 * stride_281;
      lbound_295 = (long) *its;
      ubound_296 = (long) *ite;
      stride_299 = (1 - lbound_295) + ubound_296;
      stride_299 = gf_max(stride_299, 0);
      lbound_297 = (long) *kts;
      ubound_298 = (long) *kte;
      size_301 = ((1 - lbound_297) + ubound_298) * stride_299;
      size_301 = gf_max(size_301, 0);
      D_6741 = size_301 + -1;
      D_6742 = (ulong) (ulong) (size_301) * 32;
      D_6743 = (ulong) (size_301) * 4;
      D_6744 = (device void *) workspace.allocate (gf_max((ulong) (size_301 * 4), 1));
      q_cup = (device GFReal *) D_6744;
      offset_300 = -(lbound_295) - lbound_297 * stride_299;
      lbound_302 = (long) *its;
      ubound_303 = (long) *ite;
      stride_306 = (1 - lbound_302) + ubound_303;
      stride_306 = gf_max(stride_306, 0);
      lbound_304 = (long) *kts;
      ubound_305 = (long) *kte;
      size_308 = ((1 - lbound_304) + ubound_305) * stride_306;
      size_308 = gf_max(size_308, 0);
      D_6737 = size_308 + -1;
      D_6738 = (ulong) (ulong) (size_308) * 32;
      D_6739 = (ulong) (size_308) * 4;
      D_6740 = (device void *) workspace.allocate (gf_max((ulong) (size_308 * 4), 1));
      qco = (device GFReal *) D_6740;
      offset_307 = -(lbound_302) - lbound_304 * stride_306;
      // module_cu_gf.mpas.F:4784: at k=ktop the shallow dellaq loop reads qco(i,ktop+1)
      // (times detup=0), a word native never defines (qco is set only up to ktop,
      // F:4565-4637). Workspace slots are reused, so that word was whatever the slot
      // last held; a NaN there made rthcuten/rqvcuten NaN at ktop and the step was
      // refused. Zero it: native's finite
      // garbage times zero gives the same words.
      for (long e_qco = 0; e_qco < size_308; ++e_qco) qco[e_qco] = 0.0f;
      lbound_309 = (long) *its;
      ubound_310 = (long) *ite;
      stride_313 = (1 - lbound_309) + ubound_310;
      stride_313 = gf_max(stride_313, 0);
      lbound_311 = (long) *kts;
      ubound_312 = (long) *kte;
      size_315 = ((1 - lbound_311) + ubound_312) * stride_313;
      size_315 = gf_max(size_315, 0);
      D_6733 = size_315 + -1;
      D_6734 = (ulong) (ulong) (size_315) * 32;
      D_6735 = (ulong) (size_315) * 4;
      D_6736 = (device void *) workspace.allocate (gf_max((ulong) (size_315 * 4), 1));
      qes = (device GFReal *) D_6736;
      offset_314 = -(lbound_309) - lbound_311 * stride_313;
      lbound_316 = (long) *its;
      ubound_317 = (long) *ite;
      stride_320 = (1 - lbound_316) + ubound_317;
      stride_320 = gf_max(stride_320, 0);
      lbound_318 = (long) *kts;
      ubound_319 = (long) *kte;
      size_322 = ((1 - lbound_318) + ubound_319) * stride_320;
      size_322 = gf_max(size_322, 0);
      D_6729 = size_322 + -1;
      D_6730 = (ulong) (ulong) (size_322) * 32;
      D_6731 = (ulong) (size_322) * 4;
      D_6732 = (device void *) workspace.allocate (gf_max((ulong) (size_322 * 4), 1));
      qes_cup = (device GFReal *) D_6732;
      offset_321 = -(lbound_316) - lbound_318 * stride_320;
      lbound_323 = (long) *its;
      ubound_324 = (long) *ite;
      stride_327 = (1 - lbound_323) + ubound_324;
      stride_327 = gf_max(stride_327, 0);
      lbound_325 = (long) *kts;
      ubound_326 = (long) *kte;
      size_329 = ((1 - lbound_325) + ubound_326) * stride_327;
      size_329 = gf_max(size_329, 0);
      D_6725 = size_329 + -1;
      D_6726 = (ulong) (ulong) (size_329) * 32;
      D_6727 = (ulong) (size_329) * 4;
      D_6728 = (device void *) workspace.allocate (gf_max((ulong) (size_329 * 4), 1));
      qeso = (device GFReal *) D_6728;
      offset_328 = -(lbound_323) - lbound_325 * stride_327;
      lbound_330 = (long) *its;
      ubound_331 = (long) *ite;
      stride_334 = (1 - lbound_330) + ubound_331;
      stride_334 = gf_max(stride_334, 0);
      lbound_332 = (long) *kts;
      ubound_333 = (long) *kte;
      size_336 = ((1 - lbound_332) + ubound_333) * stride_334;
      size_336 = gf_max(size_336, 0);
      D_6721 = size_336 + -1;
      D_6722 = (ulong) (ulong) (size_336) * 32;
      D_6723 = (ulong) (size_336) * 4;
      D_6724 = (device void *) workspace.allocate (gf_max((ulong) (size_336 * 4), 1));
      qeso_cup = (device GFReal *) D_6724;
      offset_335 = -(lbound_330) - lbound_332 * stride_334;
      lbound_337 = (long) *its;
      ubound_338 = (long) *ite;
      size_340 = (1 - lbound_337) + ubound_338;
      size_340 = gf_max(size_340, 0);
      D_6717 = size_340 + -1;
      D_6718 = (ulong) (ulong) (size_340) * 32;
      D_6719 = (ulong) (size_340) * 4;
      D_6720 = (device void *) workspace.allocate (gf_max((ulong) (size_340 * 4), 1));
      qkbo = (device GFReal *) D_6720;
      offset_339 = -(lbound_337);
      lbound_348 = (long) *its;
      ubound_349 = (long) *ite;
      stride_352 = (1 - lbound_348) + ubound_349;
      stride_352 = gf_max(stride_352, 0);
      lbound_350 = (long) *kts;
      ubound_351 = (long) *kte;
      size_354 = ((1 - lbound_350) + ubound_351) * stride_352;
      size_354 = gf_max(size_354, 0);
      D_6713 = size_354 + -1;
      D_6714 = (ulong) (ulong) (size_354) * 32;
      D_6715 = (ulong) (size_354) * 4;
      D_6716 = (device void *) workspace.allocate (gf_max((ulong) (size_354 * 4), 1));
      qo_cup = (device GFReal *) D_6716;
      offset_353 = -(lbound_348) - lbound_350 * stride_352;
      lbound_355 = (long) *its;
      ubound_356 = (long) *ite;
      stride_359 = (1 - lbound_355) + ubound_356;
      stride_359 = gf_max(stride_359, 0);
      lbound_357 = (long) *kts;
      ubound_358 = (long) *kte;
      size_361 = ((1 - lbound_357) + ubound_358) * stride_359;
      size_361 = gf_max(size_361, 0);
      D_6709 = size_361 + -1;
      D_6710 = (ulong) (ulong) (size_361) * 32;
      D_6711 = (ulong) (size_361) * 4;
      D_6712 = (device void *) workspace.allocate (gf_max((ulong) (size_361 * 4), 1));
      qrco = (device GFReal *) D_6712;
      offset_360 = -(lbound_355) - lbound_357 * stride_359;
      lbound_369 = (long) *its;
      ubound_370 = (long) *ite;
      stride_373 = (1 - lbound_369) + ubound_370;
      stride_373 = gf_max(stride_373, 0);
      lbound_371 = (long) *kts;
      ubound_372 = (long) *kte;
      size_375 = ((1 - lbound_371) + ubound_372) * stride_373;
      size_375 = gf_max(size_375, 0);
      D_6705 = size_375 + -1;
      D_6706 = (ulong) (ulong) (size_375) * 32;
      D_6707 = (ulong) (size_375) * 4;
      D_6708 = (device void *) workspace.allocate (gf_max((ulong) (size_375 * 4), 1));
      t_cup = (device GFReal *) D_6708;
      offset_374 = -(lbound_369) - lbound_371 * stride_373;
      lbound_383 = (long) *its;
      ubound_384 = (long) *ite;
      stride_387 = (1 - lbound_383) + ubound_384;
      stride_387 = gf_max(stride_387, 0);
      lbound_385 = (long) *kts;
      ubound_386 = (long) *kte;
      size_389 = ((1 - lbound_385) + ubound_386) * stride_387;
      size_389 = gf_max(size_389, 0);
      D_6701 = size_389 + -1;
      D_6702 = (ulong) (ulong) (size_389) * 32;
      D_6703 = (ulong) (size_389) * 4;
      D_6704 = (device void *) workspace.allocate (gf_max((ulong) (size_389 * 4), 1));
      tn_cup = (device GFReal *) D_6704;
      offset_388 = -(lbound_383) - lbound_385 * stride_387;
      lbound_394 = (long) *its;
      ubound_395 = (long) *ite;
      stride_398 = (1 - lbound_394) + ubound_395;
      stride_398 = gf_max(stride_398, 0);
      lbound_396 = (long) *kts;
      ubound_397 = (long) *kte;
      size_400 = ((1 - lbound_396) + ubound_397) * stride_398;
      size_400 = gf_max(size_400, 0);
      D_6697 = size_400 + -1;
      D_6698 = (ulong) (ulong) (size_400) * 32;
      D_6699 = (ulong) (size_400) * 4;
      D_6700 = (device void *) workspace.allocate (gf_max((ulong) (size_400 * 4), 1));
      up_massdetr = (device GFReal *) D_6700;
      offset_399 = -(lbound_394) - lbound_396 * stride_398;
      lbound_401 = (long) *its;
      ubound_402 = (long) *ite;
      stride_405 = (1 - lbound_401) + ubound_402;
      stride_405 = gf_max(stride_405, 0);
      lbound_403 = (long) *kts;
      ubound_404 = (long) *kte;
      size_407 = ((1 - lbound_403) + ubound_404) * stride_405;
      size_407 = gf_max(size_407, 0);
      D_6693 = size_407 + -1;
      D_6694 = (ulong) (ulong) (size_407) * 32;
      D_6695 = (ulong) (size_407) * 4;
      D_6696 = (device void *) workspace.allocate (gf_max((ulong) (size_407 * 4), 1));
      up_massdetro = (device GFReal *) D_6696;
      offset_406 = -(lbound_401) - lbound_403 * stride_405;
      lbound_408 = (long) *its;
      ubound_409 = (long) *ite;
      stride_412 = (1 - lbound_408) + ubound_409;
      stride_412 = gf_max(stride_412, 0);
      lbound_410 = (long) *kts;
      ubound_411 = (long) *kte;
      size_414 = ((1 - lbound_410) + ubound_411) * stride_412;
      size_414 = gf_max(size_414, 0);
      D_6689 = size_414 + -1;
      D_6690 = (ulong) (ulong) (size_414) * 32;
      D_6691 = (ulong) (size_414) * 4;
      D_6692 = (device void *) workspace.allocate (gf_max((ulong) (size_414 * 4), 1));
      up_massentr = (device GFReal *) D_6692;
      offset_413 = -(lbound_408) - lbound_410 * stride_412;
      lbound_415 = (long) *its;
      ubound_416 = (long) *ite;
      stride_419 = (1 - lbound_415) + ubound_416;
      stride_419 = gf_max(stride_419, 0);
      lbound_417 = (long) *kts;
      ubound_418 = (long) *kte;
      size_421 = ((1 - lbound_417) + ubound_418) * stride_419;
      size_421 = gf_max(size_421, 0);
      D_6685 = size_421 + -1;
      D_6686 = (ulong) (ulong) (size_421) * 32;
      D_6687 = (ulong) (size_421) * 4;
      D_6688 = (device void *) workspace.allocate (gf_max((ulong) (size_421 * 4), 1));
      up_massentro = (device GFReal *) D_6688;
      offset_420 = -(lbound_415) - lbound_417 * stride_419;
      lbound_422 = (long) *its;
      ubound_423 = (long) *ite;
      size_425 = (1 - lbound_422) + ubound_423;
      size_425 = gf_max(size_425, 0);
      D_6681 = size_425 + -1;
      D_6682 = (ulong) (ulong) (size_425) * 32;
      D_6683 = (ulong) (size_425) * 4;
      D_6684 = (device void *) workspace.allocate (gf_max((ulong) (size_425 * 4), 1));
      xaa0 = (device GFReal *) D_6684;
      offset_424 = -(lbound_422);
      lbound_426 = (long) *its;
      ubound_427 = (long) *ite;
      stride_430 = (1 - lbound_426) + ubound_427;
      stride_430 = gf_max(stride_430, 0);
      lbound_428 = (long) *kts;
      ubound_429 = (long) *kte;
      size_432 = ((1 - lbound_428) + ubound_429) * stride_430;
      size_432 = gf_max(size_432, 0);
      D_6677 = size_432 + -1;
      D_6678 = (ulong) (ulong) (size_432) * 32;
      D_6679 = (ulong) (size_432) * 4;
      D_6680 = (device void *) workspace.allocate (gf_max((ulong) (size_432 * 4), 1));
      xdby = (device GFReal *) D_6680;
      offset_431 = -(lbound_426) - lbound_428 * stride_430;
      lbound_433 = (long) *its;
      ubound_434 = (long) *ite;
      stride_437 = (1 - lbound_433) + ubound_434;
      stride_437 = gf_max(stride_437, 0);
      lbound_435 = (long) *kts;
      ubound_436 = (long) *kte;
      size_439 = ((1 - lbound_435) + ubound_436) * stride_437;
      size_439 = gf_max(size_439, 0);
      D_6673 = size_439 + -1;
      D_6674 = (ulong) (ulong) (size_439) * 32;
      D_6675 = (ulong) (size_439) * 4;
      D_6676 = (device void *) workspace.allocate (gf_max((ulong) (size_439 * 4), 1));
      xhc = (device GFReal *) D_6676;
      offset_438 = -(lbound_433) - lbound_435 * stride_437;
      lbound_440 = (long) *its;
      ubound_441 = (long) *ite;
      stride_444 = (1 - lbound_440) + ubound_441;
      stride_444 = gf_max(stride_444, 0);
      lbound_442 = (long) *kts;
      ubound_443 = (long) *kte;
      size_446 = ((1 - lbound_442) + ubound_443) * stride_444;
      size_446 = gf_max(size_446, 0);
      D_6669 = size_446 + -1;
      D_6670 = (ulong) (ulong) (size_446) * 32;
      D_6671 = (ulong) (size_446) * 4;
      D_6672 = (device void *) workspace.allocate (gf_max((ulong) (size_446 * 4), 1));
      xhe = (device GFReal *) D_6672;
      offset_445 = -(lbound_440) - lbound_442 * stride_444;
      lbound_447 = (long) *its;
      ubound_448 = (long) *ite;
      stride_451 = (1 - lbound_447) + ubound_448;
      stride_451 = gf_max(stride_451, 0);
      lbound_449 = (long) *kts;
      ubound_450 = (long) *kte;
      size_453 = ((1 - lbound_449) + ubound_450) * stride_451;
      size_453 = gf_max(size_453, 0);
      D_6665 = size_453 + -1;
      D_6666 = (ulong) (ulong) (size_453) * 32;
      D_6667 = (ulong) (size_453) * 4;
      D_6668 = (device void *) workspace.allocate (gf_max((ulong) (size_453 * 4), 1));
      xhe_cup = (device GFReal *) D_6668;
      offset_452 = -(lbound_447) - lbound_449 * stride_451;
      lbound_454 = (long) *its;
      ubound_455 = (long) *ite;
      stride_458 = (1 - lbound_454) + ubound_455;
      stride_458 = gf_max(stride_458, 0);
      lbound_456 = (long) *kts;
      ubound_457 = (long) *kte;
      size_460 = ((1 - lbound_456) + ubound_457) * stride_458;
      size_460 = gf_max(size_460, 0);
      D_6661 = size_460 + -1;
      D_6662 = (ulong) (ulong) (size_460) * 32;
      D_6663 = (ulong) (size_460) * 4;
      D_6664 = (device void *) workspace.allocate (gf_max((ulong) (size_460 * 4), 1));
      xhes = (device GFReal *) D_6664;
      offset_459 = -(lbound_454) - lbound_456 * stride_458;
      lbound_461 = (long) *its;
      ubound_462 = (long) *ite;
      stride_465 = (1 - lbound_461) + ubound_462;
      stride_465 = gf_max(stride_465, 0);
      lbound_463 = (long) *kts;
      ubound_464 = (long) *kte;
      size_467 = ((1 - lbound_463) + ubound_464) * stride_465;
      size_467 = gf_max(size_467, 0);
      D_6657 = size_467 + -1;
      D_6658 = (ulong) (ulong) (size_467) * 32;
      D_6659 = (ulong) (size_467) * 4;
      D_6660 = (device void *) workspace.allocate (gf_max((ulong) (size_467 * 4), 1));
      xhes_cup = (device GFReal *) D_6660;
      offset_466 = -(lbound_461) - lbound_463 * stride_465;
      lbound_468 = (long) *its;
      ubound_469 = (long) *ite;
      size_471 = (1 - lbound_468) + ubound_469;
      size_471 = gf_max(size_471, 0);
      D_6653 = size_471 + -1;
      D_6654 = (ulong) (ulong) (size_471) * 32;
      D_6655 = (ulong) (size_471) * 4;
      D_6656 = (device void *) workspace.allocate (gf_max((ulong) (size_471 * 4), 1));
      xhkb = (device GFReal *) D_6656;
      offset_470 = -(lbound_468);
      lbound_476 = (long) *its;
      ubound_477 = (long) *ite;
      size_479 = (1 - lbound_476) + ubound_477;
      size_479 = gf_max(size_479, 0);
      D_6649 = size_479 + -1;
      D_6650 = (ulong) (ulong) (size_479) * 32;
      D_6651 = (ulong) (size_479) * 4;
      D_6652 = (device void *) workspace.allocate (gf_max((ulong) (size_479 * 4), 1));
      xland1 = (device GFReal *) D_6652;
      offset_478 = -(lbound_476);
      lbound_480 = (long) *its;
      ubound_481 = (long) *ite;
      size_483 = (1 - lbound_480) + ubound_481;
      size_483 = gf_max(size_483, 0);
      D_6645 = size_483 + -1;
      D_6646 = (ulong) (ulong) (size_483) * 32;
      D_6647 = (ulong) (size_483) * 4;
      D_6648 = (device void *) workspace.allocate (gf_max((ulong) (size_483 * 4), 1));
      xmb = (device GFReal *) D_6648;
      offset_482 = -(lbound_480);
      lbound_488 = (long) *its;
      ubound_489 = (long) *ite;
      size_491 = (1 - lbound_488) + ubound_489;
      size_491 = gf_max(size_491, 0);
      D_6641 = size_491 + -1;
      D_6642 = (ulong) (ulong) (size_491) * 32;
      D_6643 = (ulong) (size_491) * 4;
      D_6644 = (device void *) workspace.allocate (gf_max((ulong) (size_491 * 4), 1));
      xmbmax = (device GFReal *) D_6644;
      offset_490 = -(lbound_488);
      lbound_492 = (long) *its;
      ubound_493 = (long) *ite;
      stride_496 = (1 - lbound_492) + ubound_493;
      stride_496 = gf_max(stride_496, 0);
      lbound_494 = (long) *kts;
      ubound_495 = (long) *kte;
      size_498 = ((1 - lbound_494) + ubound_495) * stride_496;
      size_498 = gf_max(size_498, 0);
      D_6637 = size_498 + -1;
      D_6638 = (ulong) (ulong) (size_498) * 32;
      D_6639 = (ulong) (size_498) * 4;
      D_6640 = (device void *) workspace.allocate (gf_max((ulong) (size_498 * 4), 1));
      xq = (device GFReal *) D_6640;
      offset_497 = -(lbound_492) - lbound_494 * stride_496;
      lbound_499 = (long) *its;
      ubound_500 = (long) *ite;
      stride_503 = (1 - lbound_499) + ubound_500;
      stride_503 = gf_max(stride_503, 0);
      lbound_501 = (long) *kts;
      ubound_502 = (long) *kte;
      size_505 = ((1 - lbound_501) + ubound_502) * stride_503;
      size_505 = gf_max(size_505, 0);
      D_6633 = size_505 + -1;
      D_6634 = (ulong) (ulong) (size_505) * 32;
      D_6635 = (ulong) (size_505) * 4;
      D_6636 = (device void *) workspace.allocate (gf_max((ulong) (size_505 * 4), 1));
      xq_cup = (device GFReal *) D_6636;
      offset_504 = -(lbound_499) - lbound_501 * stride_503;
      lbound_506 = (long) *its;
      ubound_507 = (long) *ite;
      stride_510 = (1 - lbound_506) + ubound_507;
      stride_510 = gf_max(stride_510, 0);
      lbound_508 = (long) *kts;
      ubound_509 = (long) *kte;
      size_512 = ((1 - lbound_508) + ubound_509) * stride_510;
      size_512 = gf_max(size_512, 0);
      D_6629 = size_512 + -1;
      D_6630 = (ulong) (ulong) (size_512) * 32;
      D_6631 = (ulong) (size_512) * 4;
      D_6632 = (device void *) workspace.allocate (gf_max((ulong) (size_512 * 4), 1));
      xqes = (device GFReal *) D_6632;
      offset_511 = -(lbound_506) - lbound_508 * stride_510;
      lbound_513 = (long) *its;
      ubound_514 = (long) *ite;
      stride_517 = (1 - lbound_513) + ubound_514;
      stride_517 = gf_max(stride_517, 0);
      lbound_515 = (long) *kts;
      ubound_516 = (long) *kte;
      size_519 = ((1 - lbound_515) + ubound_516) * stride_517;
      size_519 = gf_max(size_519, 0);
      D_6625 = size_519 + -1;
      D_6626 = (ulong) (ulong) (size_519) * 32;
      D_6627 = (ulong) (size_519) * 4;
      D_6628 = (device void *) workspace.allocate (gf_max((ulong) (size_519 * 4), 1));
      xqes_cup = (device GFReal *) D_6628;
      offset_518 = -(lbound_513) - lbound_515 * stride_517;
      lbound_520 = (long) *its;
      ubound_521 = (long) *ite;
      stride_524 = (1 - lbound_520) + ubound_521;
      stride_524 = gf_max(stride_524, 0);
      lbound_522 = (long) *kts;
      ubound_523 = (long) *kte;
      size_526 = ((1 - lbound_522) + ubound_523) * stride_524;
      size_526 = gf_max(size_526, 0);
      D_6621 = size_526 + -1;
      D_6622 = (ulong) (ulong) (size_526) * 32;
      D_6623 = (ulong) (size_526) * 4;
      D_6624 = (device void *) workspace.allocate (gf_max((ulong) (size_526 * 4), 1));
      xt = (device GFReal *) D_6624;
      offset_525 = -(lbound_520) - lbound_522 * stride_524;
      lbound_527 = (long) *its;
      ubound_528 = (long) *ite;
      stride_531 = (1 - lbound_527) + ubound_528;
      stride_531 = gf_max(stride_531, 0);
      lbound_529 = (long) *kts;
      ubound_530 = (long) *kte;
      size_533 = ((1 - lbound_529) + ubound_530) * stride_531;
      size_533 = gf_max(size_533, 0);
      D_6617 = size_533 + -1;
      D_6618 = (ulong) (ulong) (size_533) * 32;
      D_6619 = (ulong) (size_533) * 4;
      D_6620 = (device void *) workspace.allocate (gf_max((ulong) (size_533 * 4), 1));
      xt_cup = (device GFReal *) D_6620;
      offset_532 = -(lbound_527) - lbound_529 * stride_531;
      lbound_534 = (long) *its;
      ubound_535 = (long) *ite;
      stride_538 = (1 - lbound_534) + ubound_535;
      stride_538 = gf_max(stride_538, 0);
      lbound_536 = (long) *kts;
      ubound_537 = (long) *kte;
      size_540 = ((1 - lbound_536) + ubound_537) * stride_538;
      size_540 = gf_max(size_540, 0);
      D_6613 = size_540 + -1;
      D_6614 = (ulong) (ulong) (size_540) * 32;
      D_6615 = (ulong) (size_540) * 4;
      D_6616 = (device void *) workspace.allocate (gf_max((ulong) (size_540 * 4), 1));
      xz = (device GFReal *) D_6616;
      offset_539 = -(lbound_534) - lbound_536 * stride_538;
      lbound_541 = (long) *its;
      ubound_542 = (long) *ite;
      stride_545 = (1 - lbound_541) + ubound_542;
      stride_545 = gf_max(stride_545, 0);
      lbound_543 = (long) *kts;
      ubound_544 = (long) *kte;
      size_547 = ((1 - lbound_543) + ubound_544) * stride_545;
      size_547 = gf_max(size_547, 0);
      D_6609 = size_547 + -1;
      D_6610 = (ulong) (ulong) (size_547) * 32;
      D_6611 = (ulong) (size_547) * 4;
      D_6612 = (device void *) workspace.allocate (gf_max((ulong) (size_547 * 4), 1));
      xz_cup = (device GFReal *) D_6612;
      offset_546 = -(lbound_541) - lbound_543 * stride_545;
      lbound_548 = (long) *its;
      ubound_549 = (long) *ite;
      stride_552 = (1 - lbound_548) + ubound_549;
      stride_552 = gf_max(stride_552, 0);
      lbound_550 = (long) *kts;
      ubound_551 = (long) *kte;
      size_554 = ((1 - lbound_550) + ubound_551) * stride_552;
      size_554 = gf_max(size_554, 0);
      D_6605 = size_554 + -1;
      D_6606 = (ulong) (ulong) (size_554) * 32;
      D_6607 = (ulong) (size_554) * 4;
      D_6608 = (device void *) workspace.allocate (gf_max((ulong) (size_554 * 4), 1));
      xzu = (device GFReal *) D_6608;
      offset_553 = -(lbound_548) - lbound_550 * stride_552;
      lbound_555 = (long) *its;
      ubound_556 = (long) *ite;
      stride_559 = (1 - lbound_555) + ubound_556;
      stride_559 = gf_max(stride_559, 0);
      lbound_557 = (long) *kts;
      ubound_558 = (long) *kte;
      size_561 = ((1 - lbound_557) + ubound_558) * stride_559;
      size_561 = gf_max(size_561, 0);
      D_6601 = size_561 + -1;
      D_6602 = (ulong) (ulong) (size_561) * 32;
      D_6603 = (ulong) (size_561) * 4;
      D_6604 = (device void *) workspace.allocate (gf_max((ulong) (size_561 * 4), 1));
      z = (device GFReal *) D_6604;
      offset_560 = -(lbound_555) - lbound_557 * stride_559;
      lbound_566 = (long) *its;
      ubound_567 = (long) *ite;
      stride_570 = (1 - lbound_566) + ubound_567;
      stride_570 = gf_max(stride_570, 0);
      lbound_568 = (long) *kts;
      ubound_569 = (long) *kte;
      size_572 = ((1 - lbound_568) + ubound_569) * stride_570;
      size_572 = gf_max(size_572, 0);
      D_6597 = size_572 + -1;
      D_6598 = (ulong) (ulong) (size_572) * 32;
      D_6599 = (ulong) (size_572) * 4;
      D_6600 = (device void *) workspace.allocate (gf_max((ulong) (size_572 * 4), 1));
      z_cup = (device GFReal *) D_6600;
      offset_571 = -(lbound_566) - lbound_568 * stride_570;
      lbound_580 = (long) *its;
      ubound_581 = (long) *ite;
      stride_584 = (1 - lbound_580) + ubound_581;
      stride_584 = gf_max(stride_584, 0);
      lbound_582 = (long) *kts;
      ubound_583 = (long) *kte;
      size_586 = ((1 - lbound_582) + ubound_583) * stride_584;
      size_586 = gf_max(size_586, 0);
      D_6593 = size_586 + -1;
      D_6594 = (ulong) (ulong) (size_586) * 32;
      D_6595 = (ulong) (size_586) * 4;
      D_6596 = (device void *) workspace.allocate (gf_max((ulong) (size_586 * 4), 1));
      zo_cup = (device GFReal *) D_6596;
      offset_585 = -(lbound_580) - lbound_582 * stride_584;
      lbound_595 = (long) *its;
      ubound_596 = (long) *ite;
      stride_599 = (1 - lbound_595) + ubound_596;
      stride_599 = gf_max(stride_599, 0);
      lbound_597 = (long) *kts;
      ubound_598 = (long) *kte;
      size_601 = ((1 - lbound_597) + ubound_598) * stride_599;
      size_601 = gf_max(size_601, 0);
      D_6589 = size_601 + -1;
      D_6590 = (ulong) (ulong) (size_601) * 32;
      D_6591 = (ulong) (size_601) * 4;
      D_6592 = (device void *) workspace.allocate (gf_max((ulong) (size_601 * 4), 1));
      zu = (device GFReal *) D_6592;
      offset_600 = -(lbound_595) - lbound_597 * stride_599;
      lbound_602 = (long) *its;
      ubound_603 = (long) *ite;
      size_605 = (1 - lbound_602) + ubound_603;
      size_605 = gf_max(size_605, 0);
      D_6585 = size_605 + -1;
      D_6586 = (ulong) (ulong) (size_605) * 32;
      D_6587 = (ulong) (size_605) * 4;
      D_6588 = (device void *) workspace.allocate (gf_max((ulong) (size_605 * 4), 1));
      zuhe = (device GFReal *) D_6588;
      offset_604 = -(lbound_602);
      lbound_606 = (long) *its;
      ubound_607 = (long) *ite;
      stride_610 = (1 - lbound_606) + ubound_607;
      stride_610 = gf_max(stride_610, 0);
      lbound_608 = (long) *kts;
      ubound_609 = (long) *kte;
      size_612 = ((1 - lbound_608) + ubound_609) * stride_610;
      size_612 = gf_max(size_612, 0);
      D_6581 = size_612 + -1;
      D_6582 = (ulong) (ulong) (size_612) * 32;
      D_6583 = (ulong) (size_612) * 4;
      D_6584 = (device void *) workspace.allocate (gf_max((ulong) (size_612 * 4), 1));
      zuo = (device GFReal *) D_6584;
      offset_611 = -(lbound_606) - lbound_608 * stride_610;
      zustart = 1.00000001490116119384765625e-1f;
      zufinal = 1.0e+0f;
      levadj = 4;
      power_entr = 2.0e+0f;
      day = 8.64e+4f;
      {
        int D_6162;
        int D_6163;

        D_6162 = *its;
        D_6163 = *itf;
        i = D_6162;
        while (1)
          {
            {
              bool D_6166;

              D_6166 = i > D_6163;
              if (D_6166) break;
              xmb_out[(long) i + offset_486] = 0.0f;
              xland1[(long) i + offset_478] = 1.0e+0f;
              if (xland[(long) i + offset_474] > 1.5e+0f)
                {
                  xland1[(long) i + offset_478] = 0.0f;
                }

              cap_max_increment[(long) i + offset_18] = 2.5e+1f;
              gf_memmove ((device char *)&ierrc[(long) i + offset_209], "", 0);
              gf_memset ((device char *)&ierrc[(long) i + offset_209], 32, 50);

              i = i + 1;
            }
          }

      }
      entr_rate = 1.000000047497451305389404296875e-3f;
      tun_lim = 7;
      {
        int D_6168;
        int D_6169;

        D_6168 = *kts;
        D_6169 = *ktf;
        k = D_6168;
        while (1)
          {
            {
              bool D_6172;

              D_6172 = k > D_6169;
              if (D_6172) break;
              {
                int D_6173;
                int D_6174;

                D_6173 = *its;
                D_6174 = *itf;
                i = D_6173;
                while (1)
                  {
                    {
                      bool D_6177;

                      D_6177 = i > D_6174;
                      if (D_6177) break;
                      up_massentro[((long) k * stride_419 + offset_420) + (long) i] = 0.0f;
                      up_massdetro[((long) k * stride_405 + offset_406) + (long) i] = 0.0f;
                      z[((long) k * stride_559 + offset_560) + (long) i] = zo[((long) k * stride_577 + offset_578) + (long) i];
                      xz[((long) k * stride_538 + offset_539) + (long) i] = zo[((long) k * stride_577 + offset_578) + (long) i];
                      qrco[((long) k * stride_359 + offset_360) + (long) i] = 0.0f;
                      cd[((long) k * stride_24 + offset_25) + (long) i] = (entr_rate);
                      dellaqc[((long) k * stride_66 + offset_67) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      depth_min = 5.0e+1f;
      cap_maxs = 2.5e+1f;
      {
        int D_6178;
        int D_6179;

        D_6178 = *its;
        D_6179 = *itf;
        i = D_6178;
        while (1)
          {
            {
              bool D_6182;

              D_6182 = i > D_6179;
              if (D_6182) break;
              kbmax[(long) i + offset_225] = 1;
              aa0[(long) i + offset_2] = 0.0f;
              aa1[(long) i + offset_6] = 0.0f;

              i = i + 1;
            }
          }

      }
      {
        int D_6183;
        int D_6184;

        D_6183 = *its;
        D_6184 = *itf;
        i = D_6183;
        while (1)
          {
            {
              bool D_6187;

              D_6187 = i > D_6184;
              if (D_6187) break;
              cap_max[(long) i + offset_14] = cap_maxs;
              iresult = 0;

              i = i + 1;
            }
          }

      }
      zkbmax = 4.0e+3f;
      {
        int C_6188 = -1;

        cup_env (workspace, z, qes, he, hes, t, q, p, z1, psur, ierr, tcrit, &C_6188, xl, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      {
        int C_6189 = -1;

        cup_env (workspace, zo, qeso, heo, heso, tn, qo, po, z1, psur, ierr, tcrit, &C_6189, xl, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      cup_env_clev (workspace, t, qes, q, he, hes, z, p, qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup, psur, ierr, z1, xl, rv, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      cup_env_clev (workspace, tn, qeso, qo, heo, heso, zo, po, qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup, psur, ierr, z1, xl, rv, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_6190;
        int D_6191;

        D_6190 = *its;
        D_6191 = *itf;
        i = D_6190;
        while (1)
          {
            {
              bool D_6194;

              D_6194 = i > D_6191;
              if (D_6194) break;
              if (ierr[(long) i + offset_205] == 0)
                {
                  if (aaeq[(long) i + offset_10] < -1.00000001490116119384765625e-1f)
                    {
                      ierr[(long) i + offset_205] = 20;
                    }

                  {
                    int D_6197;
                    int D_6198;

                    D_6197 = *kts;
                    D_6198 = *ktf;
                    k = D_6197;
                    while (1)
                      {
                        {
                          bool D_6201;

                          D_6201 = k > D_6198;
                          if (D_6201) break;
                          if (zo_cup[((long) k * stride_584 + offset_585) + (long) i] > z1[(long) i + offset_564] + zkbmax)
                            {
                              kbmax[(long) i + offset_225] = k;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }

                  {
                    int D_6204;
                    int M_613;
                    int D_6206;

                    D_6204 = kbmax[(long) i + offset_225];
                    M_613 = D_6204;
                    D_6206 = *ktf + -4;
                    M_613 = gf_min(D_6206, M_613);
                    kbmax[(long) i + offset_225] = M_613;
                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int C_6207 = 3;

        cup_maximi (workspace, heo_cup, &C_6207, kbmax, k22, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      {
        int D_6208;
        int D_6209;

        D_6208 = *its;
        D_6209 = *itf;
        i = D_6208;
        while (1)
          {
            {
              bool D_6212;

              D_6212 = i > D_6209;
              if (D_6212) break;
              if (kpbl[(long) i + offset_229] > 5)
                {
                  cap_max[(long) i + offset_14] = po_cup[((long) kpbl[(long) i + offset_229] * stride_281 + offset_282) + (long) i];
                }

              if ((GFReal) ierr[(long) i + offset_205] == 0.0f)
                {
                  if (k22[(long) i + offset_217] > kbmax[(long) i + offset_225])
                    {
                      ierr[(long) i + offset_205] = 2;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "could not find k22", 18);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 18, 32, 32);
                    }

                  if (kpbl[(long) i + offset_229] > 5)
                    {
                      k22[(long) i + offset_217] = kpbl[(long) i + offset_229];
                      ierr[(long) i + offset_205] = 0;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "reset to zero becausof kpbl", 27);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 27, 32, 23);
                    }

                }
              else
                {
                  gf_memmove ((device char *)&ierrc[(long) i + offset_209], "why here? ", 10);
                  gf_memset ((device char *)&ierrc[(long) i + offset_209] + 10, 32, 40);
                }



              i = i + 1;
            }
          }

      }
      {
        int D_6218;
        int D_6219;

        D_6218 = *its;
        D_6219 = *itf;
        i = D_6218;
        while (1)
          {
            {
              bool D_6222;

              D_6222 = i > D_6219;
              if (D_6222) break;
              if ((GFReal) ierr[(long) i + offset_205] == 0.0f)
                {
                  if (*use_excess == 2)
                    {
                      {
                        int M_614;
                        int D_6226;

                        M_614 = 1;
                        D_6226 = k22[(long) i + offset_217] + -1;
                        M_614 = gf_max(D_6226, M_614);
                        k1 = M_614;
                      }
                      k2 = k22[(long) i + offset_217] + 1;
                      {
                        GFReal val_615;
                        long D_6228;
                        long D_6229;
                        long D_6230;
                        long D_6231;

                        val_615 = 0.0f;
                        D_6228 = offset_151;
                        D_6229 = (long) k1;
                        D_6230 = (long) k2;
                        D_6231 = (long) i;
                        {
                          long S_616;
                          long D_6233;
                          long D_6234;

                          D_6233 = stride_150;
                          D_6234 = D_6228 + D_6231;
                          S_616 = D_6229;
                          while (1)
                            {
                              if (S_616 > D_6230) break;
                              val_615 = he_cup[S_616 * D_6233 + D_6234] + val_615;
                              S_616 = S_616 + 1;
                            }

                        }
                        hkb[(long) i + offset_197] = (val_615 / (GFReal) ((k2 - k1) + 1) + *xl * zqexec[(long) i + offset_589]) + *cp * ztexec[(long) i + offset_593];
                      }
                      {
                        GFReal val_617;
                        long D_6237;
                        long D_6238;
                        long D_6239;
                        long D_6240;

                        val_617 = 0.0f;
                        D_6237 = offset_165;
                        D_6238 = (long) k1;
                        D_6239 = (long) k2;
                        D_6240 = (long) i;
                        {
                          long S_618;
                          long D_6242;
                          long D_6243;

                          D_6242 = stride_164;
                          D_6243 = D_6237 + D_6240;
                          S_618 = D_6238;
                          while (1)
                            {
                              if (S_618 > D_6239) break;
                              val_617 = heo_cup[S_618 * D_6242 + D_6243] + val_617;
                              S_618 = S_618 + 1;
                            }

                        }
                        hkbo[(long) i + offset_201] = (val_617 / (GFReal) ((k2 - k1) + 1) + *xl * zqexec[(long) i + offset_589]) + *cp * ztexec[(long) i + offset_593];
                      }
                      {
                        GFReal val_619;
                        long D_6246;
                        long D_6247;
                        long D_6248;
                        long D_6249;

                        val_619 = 0.0f;
                        D_6246 = offset_353;
                        D_6247 = (long) k1;
                        D_6248 = (long) k2;
                        D_6249 = (long) i;
                        {
                          long S_620;
                          long D_6251;
                          long D_6252;

                          D_6251 = stride_352;
                          D_6252 = D_6246 + D_6249;
                          S_620 = D_6247;
                          while (1)
                            {
                              if (S_620 > D_6248) break;
                              val_619 = qo_cup[S_620 * D_6251 + D_6252] + val_619;
                              S_620 = S_620 + 1;
                            }

                        }
                        qkbo[(long) i + offset_339] = val_619 / (GFReal) ((k2 - k1) + 1) + *xl * zqexec[(long) i + offset_589];
                      }
                    }
                  else
                    {
                      if (*use_excess <= 1)
                        {
                          hkb[(long) i + offset_197] = he_cup[((long) k22[(long) i + offset_217] * stride_150 + offset_151) + (long) i] + (GFReal) *use_excess * ((*xl * zqexec[(long) i + offset_589] + *cp * ztexec[(long) i + offset_593]));
                          hkbo[(long) i + offset_201] = heo_cup[((long) k22[(long) i + offset_217] * stride_164 + offset_165) + (long) i] + (GFReal) *use_excess * ((*xl * zqexec[(long) i + offset_589] + *cp * ztexec[(long) i + offset_593]));
                          qkbo[(long) i + offset_339] = qo_cup[((long) k22[(long) i + offset_217] * stride_352 + offset_353) + (long) i] + (GFReal) *use_excess * ((*xl * zqexec[(long) i + offset_589]));
                        }

                    }

                  {
                    int D_6255;

                    D_6255 = k22[(long) i + offset_217];
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_6258;

                          D_6258 = k > D_6255;
                          if (D_6258) break;
                          {
                            GFReal D_6259;
                            GFReal M_621;
                            GFReal D_6261;

                            D_6259 = hkb[(long) i + offset_197];
                            M_621 = D_6259;
                            D_6261 = he_cup[((long) k * stride_150 + offset_151) + (long) i];
                            M_621 = gf_max(D_6261, M_621);
                            hkb[(long) i + offset_197] = M_621;
                          }
                          {
                            GFReal D_6262;
                            GFReal M_622;
                            GFReal D_6264;

                            D_6262 = hkbo[(long) i + offset_201];
                            M_622 = D_6262;
                            D_6264 = heo_cup[((long) k * stride_164 + offset_165) + (long) i];
                            M_622 = gf_max(D_6264, M_622);
                            hkbo[(long) i + offset_201] = M_622;
                          }
                          {
                            GFReal D_6265;
                            GFReal M_623;
                            GFReal D_6267;

                            D_6265 = qkbo[(long) i + offset_339];
                            M_623 = D_6265;
                            D_6267 = qo_cup[((long) k * stride_352 + offset_353) + (long) i];
                            M_623 = gf_max(D_6267, M_623);
                            qkbo[(long) i + offset_339] = M_623;
                          }

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int C_6268 = 5;

        cup_kbcon (workspace, ierrc, cap_max_increment, &C_6268, k22, kbcon, heo_cup, heso_cup, hkbo, ierr, kbmax, po_cup, cap_max, xl, cp, ztexec, zqexec, use_excess, itf, jtf, ktf, its, ite, jts, jte, kts, kte, 50);
      }
      {
        int D_6269;
        int D_6270;

        D_6269 = *its;
        D_6270 = *itf;
        i = D_6269;
        while (1)
          {
            {
              bool D_6273;

              D_6273 = i > D_6270;
              if (D_6273) break;
              do {
if ((GFReal) ierr[(long) i + offset_205] == 0.0f)
                {
                  if (*ktf + -3 <= kbcon[(long) i + offset_221])
                    {
                      ierr[(long) i + offset_205] = 231;
                      break;
                    }

                  {
                    int D_6277;
                    int D_6278;

                    D_6277 = *kts;
                    D_6278 = *ktf;
                    k = D_6277;
                    while (1)
                      {
                        {
                          bool D_6281;

                          D_6281 = k > D_6278;
                          if (D_6281) break;
                          {
                            GFReal D_6282;
                            GFReal M_624;

                            D_6282 = qo_cup[((long) k * stride_352 + offset_353) + (long) i] / qeso_cup[((long) k * stride_334 + offset_335) + (long) i];
                            M_624 = D_6282;
                            M_624 = gf_min(M_624, 1.0e+0f);
                            frh = M_624;
                          }
                          entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] = ((1.2999999523162841796875e+0f - frh)) * entr_rate;
                          cd[((long) k * stride_24 + offset_25) + (long) i] = entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  zuhe[(long) i + offset_604] = zustart;
                  kstart = 1;
                  {
                    GFReal D_6284;
                    GFReal D_6285;

                    D_6284 = (GFReal) kbcon[(long) i + offset_221];
                    D_6285 = (GFReal) kstart;
                    frh = ((zufinal - zustart)) / ((((gf_pow (D_6284, power_entr))) - ((gf_pow (D_6285, power_entr)))));
                  }
                  {
                    GFReal D_6286;

                    D_6286 = (GFReal) kstart;
                    dh = zuhe[(long) i + offset_604] - ((gf_pow (D_6286, power_entr))) * frh;
                  }
                  {
                    int D_6287;
                    int D_6288;

                    D_6287 = kstart;
                    D_6288 = kbcon[(long) i + offset_221] + -1;
                    k = D_6287;
                    while (1)
                      {
                        {
                          bool D_6291;

                          D_6291 = k > D_6288;
                          if (D_6291) break;
                          dz = z_cup[((long) (k + 1) * stride_570 + offset_571) + (long) i] - z_cup[((long) k * stride_570 + offset_571) + (long) i];
                          cd[((long) k * stride_24 + offset_25) + (long) i] = 0.0f;
                          {
                            GFReal D_6292;

                            D_6292 = (GFReal) (k + 1);
                            entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] = (((((((gf_pow (D_6292, power_entr))) * frh + dh)) / zuhe[(long) i + offset_604] - 1.0e+0f) + cd[((long) k * stride_24 + offset_25) + (long) i] * dz)) / dz;
                          }
                          zuhe[(long) i + offset_604] = (zuhe[(long) i + offset_604] + (entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] * dz) * zuhe[(long) i + offset_604]) - (cd[((long) k * stride_24 + offset_25) + (long) i] * dz) * zuhe[(long) i + offset_604];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    GFReal D_6293;
                    GFReal D_6294;

                    D_6293 = (GFReal) (kbcon[(long) i + offset_221] + tun_lim);
                    D_6294 = (GFReal) (kbcon[(long) i + offset_221] + -1);
                    frh = -(((1.00000001490116119384765625e-1f - zuhe[(long) i + offset_604])) / ((((gf_pow (D_6293, power_entr))) - ((gf_pow (D_6294, power_entr))))));
                  }
                  {
                    GFReal D_6295;

                    D_6295 = (GFReal) kbcon[(long) i + offset_221];
                    dh = zuhe[(long) i + offset_604] + ((gf_pow (D_6295, power_entr))) * frh;
                  }
                  {
                    int D_6296;
                    int D_6297;

                    D_6296 = kbcon[(long) i + offset_221];
                    D_6297 = kbcon[(long) i + offset_221] + tun_lim;
                    k = D_6296;
                    while (1)
                      {
                        {
                          bool D_6300;

                          D_6300 = k > D_6297;
                          if (D_6300) break;
                          dz = z_cup[((long) (k + 1) * stride_570 + offset_571) + (long) i] - z_cup[((long) k * stride_570 + offset_571) + (long) i];
                          {
                            GFReal D_6301;

                            D_6301 = (GFReal) (k + 1);
                            cd[((long) k * stride_24 + offset_25) + (long) i] = -((((((dh - ((gf_pow (D_6301, power_entr))) * frh)) / zuhe[(long) i + offset_604] - 1.0e+0f) - entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] * dz)) / dz);
                          }
                          zuhe[(long) i + offset_604] = (zuhe[(long) i + offset_604] + (entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] * dz) * zuhe[(long) i + offset_604]) - (cd[((long) k * stride_24 + offset_25) + (long) i] * dz) * zuhe[(long) i + offset_604];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_6302;
                    int D_6303;

                    D_6302 = (kbcon[(long) i + offset_221] + tun_lim) + 1;
                    D_6303 = *ktf;
                    k = D_6302;
                    while (1)
                      {
                        {
                          bool D_6306;

                          D_6306 = k > D_6303;
                          if (D_6306) break;
                          entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] = 0.0f;
                          cd[((long) k * stride_24 + offset_25) + (long) i] = 0.0f;

                          k = k + 1;
                        }
                      }

                  }
                }

              } while(false);

              i = i + 1;
            }
          }

      }
      {
        int D_6307;
        int D_6308;

        D_6307 = *kts;
        D_6308 = *ktf;
        k = D_6307;
        while (1)
          {
            {
              bool D_6311;

              D_6311 = k > D_6308;
              if (D_6311) break;
              {
                int D_6312;
                int D_6313;

                D_6312 = *its;
                D_6313 = *itf;
                i = D_6312;
                while (1)
                  {
                    {
                      bool D_6316;

                      D_6316 = i > D_6313;
                      if (D_6316) break;
                      hc[((long) k * stride_129 + offset_130) + (long) i] = 0.0f;
                      dby[((long) k * stride_38 + offset_39) + (long) i] = 0.0f;
                      hco[((long) k * stride_136 + offset_137) + (long) i] = 0.0f;
                      dbyo[((long) k * stride_45 + offset_46) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_6317;
        int D_6318;

        D_6317 = *its;
        D_6318 = *itf;
        i = D_6317;
        while (1)
          {
            {
              bool D_6321;

              D_6321 = i > D_6318;
              if (D_6321) break;
              if ((GFReal) ierr[(long) i + offset_205] == 0.0f)
                {
                  {
                    int D_6323;

                    D_6323 = kbcon[(long) i + offset_221] + -1;
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_6326;

                          D_6326 = k > D_6323;
                          if (D_6326) break;
                          hc[((long) k * stride_129 + offset_130) + (long) i] = hkb[(long) i + offset_197];
                          hco[((long) k * stride_136 + offset_137) + (long) i] = hkbo[(long) i + offset_201];
                          qco[((long) k * stride_306 + offset_307) + (long) i] = qkbo[(long) i + offset_339];

                          k = k + 1;
                        }
                      }

                  }
                  k = kbcon[(long) i + offset_221];
                  hc[((long) k * stride_129 + offset_130) + (long) i] = hkb[(long) i + offset_197];
                  qco[((long) k * stride_306 + offset_307) + (long) i] = qkbo[(long) i + offset_339];
                  dby[((long) kbcon[(long) i + offset_221] * stride_38 + offset_39) + (long) i] = hkb[(long) i + offset_197] - hes_cup[((long) k * stride_178 + offset_179) + (long) i];
                  hco[((long) k * stride_136 + offset_137) + (long) i] = hkbo[(long) i + offset_201];
                  dbyo[((long) kbcon[(long) i + offset_221] * stride_45 + offset_46) + (long) i] = hkbo[(long) i + offset_201] - heso_cup[((long) k * stride_192 + offset_193) + (long) i];
                  trash = qeso_cup[((long) k * stride_334 + offset_335) + (long) i] + (((1.0e+0f / *xl)) * ((gammao_cup[((long) k * stride_122 + offset_123) + (long) i] / ((gammao_cup[((long) k * stride_122 + offset_123) + (long) i] + 1.0e+0f))))) * dbyo[((long) k * stride_45 + offset_46) + (long) i];
                  {
                    GFReal M_625;
                    GFReal D_6328;

                    M_625 = 0.0f;
                    D_6328 = qco[((long) k * stride_306 + offset_307) + (long) i] - trash;
                    M_625 = gf_max(D_6328, M_625);
                    qrco[((long) k * stride_359 + offset_360) + (long) i] = M_625;
                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_6329;
        int D_6330;

        D_6329 = *its;
        D_6330 = *itf;
        i = D_6329;
        while (1)
          {
            {
              bool D_6333;

              D_6333 = i > D_6330;
              if (D_6333) break;
              do {
if (ierr[(long) i + offset_205] == 0)
                {
                  zu[(offset_600 + (stride_599)) + (long) i] = zustart;
                  zuo[(offset_611 + (stride_610)) + (long) i] = zustart;
                  {
                    int D_6335;

                    D_6335 = *ktf + -1;
                    k = 2;
                    while (1)
                      {
                        {
                          bool D_6338;

                          D_6338 = k > D_6335;
                          if (D_6338) break;
                          dz = zo_cup[((long) k * stride_584 + offset_585) + (long) i] - zo_cup[((long) (k + -1) * stride_584 + offset_585) + (long) i];
                          up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i] = (entr_rate_2d[((long) (k + -1) * stride_108 + offset_109) + (long) i] * dz) * zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i];
                          up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] = (cd[((long) (k + -1) * stride_24 + offset_25) + (long) i] * dz) * zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i];
                          zuo[((long) k * stride_610 + offset_611) + (long) i] = (zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i] + up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i]) - up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i];
                          if (zuo[((long) k * stride_610 + offset_611) + (long) i] < 5.00000007450580596923828125e-2f)
                            {
                              zuo[((long) k * stride_610 + offset_611) + (long) i] = 5.00000007450580596923828125e-2f;
                              up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] = (zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i] - 5.00000007450580596923828125e-2f) + up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i];
                              cd[((long) (k + -1) * stride_24 + offset_25) + (long) i] = (up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] / dz) / zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i];
                            }

                          zu[((long) k * stride_599 + offset_600) + (long) i] = zuo[((long) k * stride_610 + offset_611) + (long) i];
                          up_massentr[((long) (k + -1) * stride_412 + offset_413) + (long) i] = up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i];
                          up_massdetr[((long) (k + -1) * stride_398 + offset_399) + (long) i] = up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_6340;
                    int D_6341;

                    D_6340 = kbcon[(long) i + offset_221] + 1;
                    D_6341 = *ktf + -1;
                    k = D_6340;
                    while (1)
                      {
                        {
                          bool D_6344;

                          D_6344 = k > D_6341;
                          if (D_6344) break;
                          hc[((long) k * stride_129 + offset_130) + (long) i] = (((hc[((long) (k + -1) * stride_129 + offset_130) + (long) i] * zu[((long) (k + -1) * stride_599 + offset_600) + (long) i] - (up_massdetr[((long) (k + -1) * stride_398 + offset_399) + (long) i] * 5.0e-1f) * hc[((long) (k + -1) * stride_129 + offset_130) + (long) i]) + up_massentr[((long) (k + -1) * stride_412 + offset_413) + (long) i] * he[((long) (k + -1) * stride_143 + offset_144) + (long) i])) / (((zu[((long) (k + -1) * stride_599 + offset_600) + (long) i] - up_massdetr[((long) (k + -1) * stride_398 + offset_399) + (long) i] * 5.0e-1f) + up_massentr[((long) (k + -1) * stride_412 + offset_413) + (long) i]));
                          dby[((long) k * stride_38 + offset_39) + (long) i] = hc[((long) k * stride_129 + offset_130) + (long) i] - hes_cup[((long) k * stride_178 + offset_179) + (long) i];
                          hco[((long) k * stride_136 + offset_137) + (long) i] = (((hco[((long) (k + -1) * stride_136 + offset_137) + (long) i] * zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i] - (up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] * 5.0e-1f) * hco[((long) (k + -1) * stride_136 + offset_137) + (long) i]) + up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i] * heo[((long) (k + -1) * stride_157 + offset_158) + (long) i])) / (((zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i] - up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] * 5.0e-1f) + up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i]));
                          dbyo[((long) k * stride_45 + offset_46) + (long) i] = hco[((long) k * stride_136 + offset_137) + (long) i] - heso_cup[((long) k * stride_192 + offset_193) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_6345;
                    int D_6346;

                    D_6345 = kbcon[(long) i + offset_221] + 1;
                    D_6346 = *ktf;
                    k = D_6345;
                    while (1)
                      {
                        {
                          bool D_6349;

                          D_6349 = k > D_6346;
                          if (D_6349) break;
                          if (dbyo[((long) k * stride_45 + offset_46) + (long) i] < 0.0f)
                            {
                              ktop[(long) i + offset_233] = k + -1;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }

                  if ((kbcon[(long) i + offset_221]) >= ktop[(long) i + offset_233])
                    {
                      ierr[(long) i + offset_205] = 5;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "ktop is less than kbcon+1", 25);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 25, 32, 25);
                      break;
                    }

                  if (*ktf + -1 <= ktop[(long) i + offset_233])
                    {
                      ierr[(long) i + offset_205] = 5;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "ktop is larger than ktf-2", 25);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 25, 32, 25);
                      break;
                    }

                  {
                    int D_6355;
                    int D_6356;

                    D_6355 = kbcon[(long) i + offset_221] + 1;
                    D_6356 = ktop[(long) i + offset_233];
                    k = D_6355;
                    while (1)
                      {
                        {
                          bool D_6359;

                          D_6359 = k > D_6356;
                          if (D_6359) break;
                          trash = qeso_cup[((long) k * stride_334 + offset_335) + (long) i] + (((1.0e+0f / *xl)) * ((gammao_cup[((long) k * stride_122 + offset_123) + (long) i] / ((gammao_cup[((long) k * stride_122 + offset_123) + (long) i] + 1.0e+0f))))) * dbyo[((long) k * stride_45 + offset_46) + (long) i];
                          qco[((long) k * stride_306 + offset_307) + (long) i] = (((qco[((long) (k + -1) * stride_306 + offset_307) + (long) i] * zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i] - (up_massdetr[((long) (k + -1) * stride_398 + offset_399) + (long) i] * 5.0e-1f) * qco[((long) (k + -1) * stride_306 + offset_307) + (long) i]) + up_massentr[((long) (k + -1) * stride_412 + offset_413) + (long) i] * qo[((long) (k + -1) * stride_345 + offset_346) + (long) i])) / (((zuo[((long) (k + -1) * stride_610 + offset_611) + (long) i] - up_massdetr[((long) (k + -1) * stride_398 + offset_399) + (long) i] * 5.0e-1f) + up_massentr[((long) (k + -1) * stride_412 + offset_413) + (long) i]));
                          {
                            GFReal M_626;
                            GFReal D_6361;

                            M_626 = 0.0f;
                            D_6361 = qco[((long) k * stride_306 + offset_307) + (long) i] - trash;
                            M_626 = gf_max(D_6361, M_626);
                            qrco[((long) k * stride_359 + offset_360) + (long) i] = M_626;
                          }
                          cupclws[((long) k * stride_31 + offset_32) + (long) i] = qrco[((long) k * stride_359 + offset_360) + (long) i] * 5.0e-1f;

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_6362;
                    int D_6363;

                    D_6362 = ktop[(long) i + offset_233] + 1;
                    D_6363 = *ktf;
                    k = D_6362;
                    while (1)
                      {
                        {
                          bool D_6366;

                          D_6366 = k > D_6363;
                          if (D_6366) break;
                          hc[((long) k * stride_129 + offset_130) + (long) i] = hes_cup[((long) k * stride_178 + offset_179) + (long) i];
                          hco[((long) k * stride_136 + offset_137) + (long) i] = heso_cup[((long) k * stride_192 + offset_193) + (long) i];
                          dby[((long) k * stride_38 + offset_39) + (long) i] = 0.0f;
                          dbyo[((long) k * stride_45 + offset_46) + (long) i] = 0.0f;
                          zu[((long) k * stride_599 + offset_600) + (long) i] = 0.0f;
                          zuo[((long) k * stride_610 + offset_611) + (long) i] = 0.0f;
                          cd[((long) k * stride_24 + offset_25) + (long) i] = 0.0f;
                          entr_rate_2d[((long) k * stride_108 + offset_109) + (long) i] = 0.0f;
                          up_massentr[((long) k * stride_412 + offset_413) + (long) i] = 0.0f;
                          up_massdetr[((long) k * stride_398 + offset_399) + (long) i] = 0.0f;
                          up_massentro[((long) k * stride_419 + offset_420) + (long) i] = 0.0f;
                          up_massdetro[((long) k * stride_405 + offset_406) + (long) i] = 0.0f;

                          k = k + 1;
                        }
                      }

                  }
                }

              } while(false);

              i = i + 1;
            }
          }

      }
      cup_up_aa0 (workspace, aa0, z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      cup_up_aa0 (workspace, aa1, zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_6367;
        int D_6368;

        D_6367 = *its;
        D_6368 = *itf;
        i = D_6367;
        while (1)
          {
            {
              bool D_6371;

              D_6371 = i > D_6368;
              if (D_6371) break;
              if (ierr[(long) i + offset_205] == 0)
                {
                  if (aa1[(long) i + offset_6] == 0.0f)
                    {
                      ierr[(long) i + offset_205] = 17;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "cloud work function zero", 24);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 24, 32, 26);
                    }

                }


              i = i + 1;
            }
          }

      }
      {
        int D_6374;
        int D_6375;

        D_6374 = *kts;
        D_6375 = *ktf;
        k = D_6374;
        while (1)
          {
            {
              bool D_6378;

              D_6378 = k > D_6375;
              if (D_6378) break;
              {
                int D_6379;
                int D_6380;

                D_6379 = *its;
                D_6380 = *itf;
                i = D_6379;
                while (1)
                  {
                    {
                      bool D_6383;

                      D_6383 = i > D_6380;
                      if (D_6383) break;
                      dellah[((long) k * stride_52 + offset_53) + (long) i] = 0.0f;
                      dsubt[((long) k * stride_101 + offset_102) + (long) i] = 0.0f;
                      dsubh[((long) k * stride_87 + offset_88) + (long) i] = 0.0f;
                      dellaq[((long) k * stride_59 + offset_60) + (long) i] = 0.0f;
                      dsubq[((long) k * stride_94 + offset_95) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_6384;
        int D_6385;

        D_6384 = *its;
        D_6385 = *itf;
        i = D_6384;
        while (1)
          {
            {
              bool D_6388;

              D_6388 = i > D_6385;
              if (D_6388) break;
              if (ierr[(long) i + offset_205] == 0)
                {
                  dp = ((po_cup[(offset_282 + (stride_281)) + (long) i] - po_cup[(stride_281 * 2 + offset_282) + (long) i])) * 1.0e+2f;
                  dsubt[(offset_102 + (stride_101)) + (long) i] = 0.0f;
                  dsubq[(offset_95 + (stride_94)) + (long) i] = 0.0f;
                  {
                    int D_6390;
                    int D_6391;

                    D_6390 = *kts + 1;
                    D_6391 = ktop[(long) i + offset_233];
                    k = D_6390;
                    while (1)
                      {
                        {
                          bool D_6394;

                          D_6394 = k > D_6391;
                          if (D_6394) break;
                          subin = 0.0f;
                          subdown = 0.0f;
                          entupk = 0.0f;
                          detupk = 0.0f;
                          entup = up_massentro[((long) k * stride_419 + offset_420) + (long) i];
                          detup = up_massdetro[((long) k * stride_405 + offset_406) + (long) i];
                          if (ktop[(long) i + offset_233] == k)
                            {
                              detupk = zuo[((long) ktop[(long) i + offset_233] * stride_610 + offset_611) + (long) i];
                              subin = 0.0f;
                              subdown = 0.0f;
                              entup = 0.0f;
                              detup = 0.0f;
                            }

                          totmas = ((((((subin - subdown) + detup) - entup) - entupk) + detupk) + zuo[((long) (k + 1) * stride_610 + offset_611) + (long) i]) - zuo[((long) k * stride_610 + offset_611) + (long) i];
                          if (gf_abs(totmas) > 9.999999974752427078783512115478515625e-7f)
                            {
                              {
                                workspace.diagnostics |= 1;
                              }
                              {
                                workspace.diagnostics |= 1;
                              }
                              {
                                workspace.diagnostics |= 1;
                              }
                            }

                          dp = ((po_cup[((long) k * stride_281 + offset_282) + (long) i] - po_cup[((long) (k + 1) * stride_281 + offset_282) + (long) i])) * 1.0e+2f;
                          dellah[((long) k * stride_52 + offset_53) + (long) i] = ((((((((detup * 5.0e-1f) * ((hco[((long) (k + 1) * stride_136 + offset_137) + (long) i] + hco[((long) k * stride_136 + offset_137) + (long) i])) - heo[((long) k * stride_157 + offset_158) + (long) i] * entup) + heo_cup[((long) (k + 1) * stride_164 + offset_165) + (long) i] * subin) - heo_cup[((long) k * stride_164 + offset_165) + (long) i] * subdown) + ((hco[((long) ktop[(long) i + offset_233] * stride_136 + offset_137) + (long) i] - heo_cup[((long) ktop[(long) i + offset_233] * stride_164 + offset_165) + (long) i])) * detupk) - heo_cup[((long) k22[(long) i + offset_217] * stride_164 + offset_165) + (long) i] * entupk)) * *g) / dp;
                          dellaq[((long) k * stride_59 + offset_60) + (long) i] = ((((((((detup * 5.0e-1f) * ((((qco[((long) (k + 1) * stride_306 + offset_307) + (long) i] + qco[((long) k * stride_306 + offset_307) + (long) i]) - qrco[((long) (k + 1) * stride_359 + offset_360) + (long) i]) - qrco[((long) k * stride_359 + offset_360) + (long) i])) - qo[((long) k * stride_345 + offset_346) + (long) i] * entup) + qo_cup[((long) (k + 1) * stride_352 + offset_353) + (long) i] * subin) - qo_cup[((long) k * stride_352 + offset_353) + (long) i] * subdown) + (((qco[((long) ktop[(long) i + offset_233] * stride_306 + offset_307) + (long) i] - qrco[((long) ktop[(long) i + offset_233] * stride_359 + offset_360) + (long) i]) - qo_cup[((long) ktop[(long) i + offset_233] * stride_352 + offset_353) + (long) i])) * detupk) - qo_cup[((long) k22[(long) i + offset_217] * stride_352 + offset_353) + (long) i] * entupk)) * *g) / dp;
                          if (ktop[(long) i + offset_233] > k)
                            {
                              dsubt[((long) k * stride_101 + offset_102) + (long) i] = (((zuo[((long) (k + 1) * stride_610 + offset_611) + (long) i] * heo_cup[((long) (k + 1) * stride_164 + offset_165) + (long) i] - zuo[((long) k * stride_610 + offset_611) + (long) i] * heo_cup[((long) k * stride_164 + offset_165) + (long) i])) * *g) / dp;
                              dsubq[((long) k * stride_94 + offset_95) + (long) i] = (((zuo[((long) (k + 1) * stride_610 + offset_611) + (long) i] * qo_cup[((long) (k + 1) * stride_352 + offset_353) + (long) i] - zuo[((long) k * stride_610 + offset_611) + (long) i] * qo_cup[((long) k * stride_352 + offset_353) + (long) i])) * *g) / dp;
                            }


                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_6401;
        int D_6402;

        D_6401 = *kts;
        D_6402 = *ktf + -1;
        k = D_6401;
        while (1)
          {
            {
              bool D_6405;

              D_6405 = k > D_6402;
              if (D_6405) break;
              {
                int D_6406;
                int D_6407;

                D_6406 = *its;
                D_6407 = *itf;
                i = D_6406;
                while (1)
                  {
                    {
                      bool D_6410;

                      D_6410 = i > D_6407;
                      if (D_6410) break;
                      dellaqc[((long) k * stride_66 + offset_67) + (long) i] = 0.0f;
                      if (ierr[(long) i + offset_205] == 0)
                        {
                          if ((ktop[(long) i + offset_233]) == k)
                            {
                              dellaqc[((long) k * stride_66 + offset_67) + (long) i] = (((zuo[((long) ktop[(long) i + offset_233] * stride_610 + offset_611) + (long) i] * 9.99999977648258209228515625e-3f) * qrco[((long) ktop[(long) i + offset_233] * stride_359 + offset_360) + (long) i]) * 9.81000041961669921875e+0f) / ((po_cup[((long) k * stride_281 + offset_282) + (long) i] - po_cup[((long) (k + 1) * stride_281 + offset_282) + (long) i]));
                            }

                          if (ktop[(long) i + offset_233] > k && kbcon[(long) i + offset_221] < k)
                            {
                              dz = zo_cup[((long) (k + 1) * stride_584 + offset_585) + (long) i] - zo_cup[((long) k * stride_584 + offset_585) + (long) i];
                              dellaqc[((long) k * stride_66 + offset_67) + (long) i] = (((up_massdetro[((long) k * stride_405 + offset_406) + (long) i] * 9.8099999129772186279296875e-2f) * 5.0e-1f) * ((qrco[((long) k * stride_359 + offset_360) + (long) i] + qrco[((long) (k + 1) * stride_359 + offset_360) + (long) i]))) / ((po_cup[((long) k * stride_281 + offset_282) + (long) i] - po_cup[((long) (k + 1) * stride_281 + offset_282) + (long) i]));
                            }

                          if (dellaqc[((long) k * stride_66 + offset_67) + (long) i] < 0.0f)
                            {
                              {
                                workspace.diagnostics |= 1;
                              }
                            }

                          {
                            GFReal M_631;
                            GFReal D_6417;

                            M_631 = 0.0f;
                            D_6417 = dellaqc[((long) k * stride_66 + offset_67) + (long) i];
                            M_631 = gf_max(D_6417, M_631);
                            dellaqc[((long) k * stride_66 + offset_67) + (long) i] = M_631;
                          }
                        }


                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      mbdt = 3.000000142492353916168212890625e-4f;
      {
        int D_6418;
        int D_6419;

        D_6418 = *kts;
        D_6419 = *ktf;
        k = D_6418;
        while (1)
          {
            {
              bool D_6422;

              D_6422 = k > D_6419;
              if (D_6422) break;
              {
                int D_6423;
                int D_6424;

                D_6423 = *its;
                D_6424 = *itf;
                i = D_6423;
                while (1)
                  {
                    {
                      bool D_6427;

                      D_6427 = i > D_6424;
                      if (D_6427) break;
                      dellat[((long) k * stride_73 + offset_74) + (long) i] = 0.0f;
                      if (ierr[(long) i + offset_205] == 0)
                        {
                          dsubh[((long) k * stride_87 + offset_88) + (long) i] = dsubt[((long) k * stride_101 + offset_102) + (long) i];
                          dellaq[((long) k * stride_59 + offset_60) + (long) i] = dellaq[((long) k * stride_59 + offset_60) + (long) i] + dellaqc[((long) k * stride_66 + offset_67) + (long) i];
                          dellaqc[((long) k * stride_66 + offset_67) + (long) i] = 0.0f;
                          xhe[((long) k * stride_444 + offset_445) + (long) i] = ((dsubt[((long) k * stride_101 + offset_102) + (long) i] + dellah[((long) k * stride_52 + offset_53) + (long) i])) * mbdt + heo[((long) k * stride_157 + offset_158) + (long) i];
                          xq[((long) k * stride_496 + offset_497) + (long) i] = ((dsubq[((long) k * stride_94 + offset_95) + (long) i] + dellaq[((long) k * stride_59 + offset_60) + (long) i])) * mbdt + qo[((long) k * stride_345 + offset_346) + (long) i];
                          dellat[((long) k * stride_73 + offset_74) + (long) i] = ((1.0e+0f / *cp)) * ((dellah[((long) k * stride_52 + offset_53) + (long) i] - *xl * dellaq[((long) k * stride_59 + offset_60) + (long) i]));
                          dsubt[((long) k * stride_101 + offset_102) + (long) i] = ((1.0e+0f / *cp)) * ((dsubt[((long) k * stride_101 + offset_102) + (long) i] - *xl * dsubq[((long) k * stride_94 + offset_95) + (long) i]));
                          xt[((long) k * stride_524 + offset_525) + (long) i] = ((dellat[((long) k * stride_73 + offset_74) + (long) i] + dsubt[((long) k * stride_101 + offset_102) + (long) i])) * mbdt + tn[((long) k * stride_380 + offset_381) + (long) i];
                          if (xq[((long) k * stride_496 + offset_497) + (long) i] <= 0.0f)
                            {
                              xq[((long) k * stride_496 + offset_497) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                            }

                        }


                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_6430;
        int D_6431;

        D_6430 = *its;
        D_6431 = *itf;
        i = D_6430;
        while (1)
          {
            {
              bool D_6434;

              D_6434 = i > D_6431;
              if (D_6434) break;
              if (ierr[(long) i + offset_205] == 0)
                {
                  xhkb[(long) i + offset_470] = hkbo[(long) i + offset_201] + ((dsubh[((long) k22[(long) i + offset_217] * stride_87 + offset_88) + (long) i] + dellah[((long) k22[(long) i + offset_217] * stride_52 + offset_53) + (long) i])) * mbdt;
                  xhe[((long) *ktf * stride_444 + offset_445) + (long) i] = heo[((long) *ktf * stride_157 + offset_158) + (long) i];
                  xq[((long) *ktf * stride_496 + offset_497) + (long) i] = qo[((long) *ktf * stride_345 + offset_346) + (long) i];
                  xt[((long) *ktf * stride_524 + offset_525) + (long) i] = tn[((long) *ktf * stride_380 + offset_381) + (long) i];
                  if (xq[((long) *ktf * stride_496 + offset_497) + (long) i] <= 0.0f)
                    {
                      xq[((long) *ktf * stride_496 + offset_497) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                    }

                }


              i = i + 1;
            }
          }

      }
      {
        int C_6437 = -1;

        cup_env (workspace, xz, xqes, xhe, xhes, xt, xq, po, z1, psur, ierr, tcrit, &C_6437, xl, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      cup_env_clev (workspace, xt, xqes, xq, xhe, xhes, xz, po, xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup, xt_cup, psur, ierr, z1, xl, rv, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_6438;
        int D_6439;

        D_6438 = *kts;
        D_6439 = *ktf;
        k = D_6438;
        while (1)
          {
            {
              bool D_6442;

              D_6442 = k > D_6439;
              if (D_6442) break;
              {
                int D_6443;
                int D_6444;

                D_6443 = *its;
                D_6444 = *itf;
                i = D_6443;
                while (1)
                  {
                    {
                      bool D_6447;

                      D_6447 = i > D_6444;
                      if (D_6447) break;
                      xhc[((long) k * stride_437 + offset_438) + (long) i] = 0.0f;
                      xdby[((long) k * stride_430 + offset_431) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_6448;
        int D_6449;

        D_6448 = *its;
        D_6449 = *itf;
        i = D_6448;
        while (1)
          {
            {
              bool D_6452;

              D_6452 = i > D_6449;
              if (D_6452) break;
              if (ierr[(long) i + offset_205] == 0)
                {
                  {
                    int D_6454;

                    D_6454 = kbcon[(long) i + offset_221] + -1;
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_6457;

                          D_6457 = k > D_6454;
                          if (D_6457) break;
                          xhc[((long) k * stride_437 + offset_438) + (long) i] = xhkb[(long) i + offset_470];

                          k = k + 1;
                        }
                      }

                  }
                  k = kbcon[(long) i + offset_221];
                  xhc[((long) k * stride_437 + offset_438) + (long) i] = xhkb[(long) i + offset_470];
                  xdby[((long) kbcon[(long) i + offset_221] * stride_430 + offset_431) + (long) i] = xhkb[(long) i + offset_470] - xhes_cup[((long) k * stride_465 + offset_466) + (long) i];
                }


              i = i + 1;
            }
          }

      }
      {
        int D_6458;
        int D_6459;

        D_6458 = *its;
        D_6459 = *itf;
        i = D_6458;
        while (1)
          {
            {
              bool D_6462;

              D_6462 = i > D_6459;
              if (D_6462) break;
              if (ierr[(long) i + offset_205] == 0)
                {
                  {
                    long D_6464;
                    long D_6465;
                    long D_6466;
                    long D_6467;
                    long D_6468;
                    long D_6469;
                    long D_6470;
                    long D_6471;
                    long D_6472;

                    D_6464 = offset_611;
                    D_6465 = lbound_608;
                    D_6466 = ubound_609;
                    D_6467 = offset_553;
                    D_6468 = lbound_550;
                    D_6469 = ubound_551;
                    D_6470 = (long) i;
                    D_6471 = (long) i;
                    D_6472 = D_6468 - D_6465;
                    {
                      long S_632;
                      long D_6474;
                      long D_6475;
                      long D_6476;
                      long D_6477;

                      D_6474 = stride_610;
                      D_6475 = D_6464 + D_6470;
                      D_6476 = stride_552;
                      D_6477 = D_6467 + D_6471;
                      S_632 = D_6465;
                      while (1)
                        {
                          if (S_632 > D_6466) break;
                          xzu[(S_632 + D_6472) * D_6476 + D_6477] = zuo[S_632 * D_6474 + D_6475];
                          S_632 = S_632 + 1;
                        }

                    }
                  }
                  {
                    int D_6479;
                    int D_6480;

                    D_6479 = kbcon[(long) i + offset_221] + 1;
                    D_6480 = ktop[(long) i + offset_233];
                    k = D_6479;
                    while (1)
                      {
                        {
                          bool D_6483;

                          D_6483 = k > D_6480;
                          if (D_6483) break;
                          xhc[((long) k * stride_437 + offset_438) + (long) i] = (((xhc[((long) (k + -1) * stride_437 + offset_438) + (long) i] * xzu[((long) (k + -1) * stride_552 + offset_553) + (long) i] - (up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] * 5.0e-1f) * xhc[((long) (k + -1) * stride_437 + offset_438) + (long) i]) + up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i] * xhe[((long) (k + -1) * stride_444 + offset_445) + (long) i])) / (((xzu[((long) (k + -1) * stride_552 + offset_553) + (long) i] - up_massdetro[((long) (k + -1) * stride_405 + offset_406) + (long) i] * 5.0e-1f) + up_massentro[((long) (k + -1) * stride_419 + offset_420) + (long) i]));
                          xdby[((long) k * stride_430 + offset_431) + (long) i] = xhc[((long) k * stride_437 + offset_438) + (long) i] - xhes_cup[((long) k * stride_465 + offset_466) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_6484;
                    int D_6485;

                    D_6484 = ktop[(long) i + offset_233] + 1;
                    D_6485 = *ktf;
                    k = D_6484;
                    while (1)
                      {
                        {
                          bool D_6488;

                          D_6488 = k > D_6485;
                          if (D_6488) break;
                          xhc[((long) k * stride_437 + offset_438) + (long) i] = xhes_cup[((long) k * stride_465 + offset_466) + (long) i];
                          xdby[((long) k * stride_430 + offset_431) + (long) i] = 0.0f;
                          xzu[((long) k * stride_552 + offset_553) + (long) i] = 0.0f;

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      cup_up_aa0 (workspace, xaa0, xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_6489;
        int D_6490;

        D_6489 = *its;
        D_6490 = *itf;
        i = D_6489;
        while (1)
          {
            {
              bool D_6493;

              D_6493 = i > D_6490;
              if (D_6493) break;
              xmb[(long) i + offset_482] = 0.0f;
              gf_memset(xff_shal, 0, sizeof(xff_shal));
              if (ierr[(long) i + offset_205] == 0)
                {
                  xmbmax[(long) i + offset_490] = 1.00000001490116119384765625e-1f;
                  xkshal = ((xaa0[(long) i + offset_424] - aa1[(long) i + offset_6])) / mbdt;
                  if (xkshal >= 0.0f)
                    {
                      xkshal = 1.0e+6f;
                    }

                  if (xkshal > -9.99999974737875163555145263671875e-5f && xkshal < 0.0f)
                    {
                      xkshal = -9.99999974737875163555145263671875e-5f;
                    }

                  {
                    GFReal M_633;
                    GFReal D_6498;

                    M_633 = 0.0f;
                    D_6498 = -(((aa1[(long) i + offset_6] - aa0[(long) i + offset_2])) / ((*dtime * xkshal)));
                    M_633 = gf_max(D_6498, M_633);
                    xff_shal[0] = M_633;
                  }
                  {
                    GFReal D_6499;
                    GFReal M_634;
                    GFReal D_6501;

                    D_6499 = xmbmax[(long) i + offset_490];
                    M_634 = D_6499;
                    D_6501 = xff_shal[0];
                    M_634 = gf_min(D_6501, M_634);
                    xff_shal[0] = M_634;
                  }
                  {
                    GFReal M_635;
                    GFReal D_6503;

                    M_635 = 0.0f;
                    D_6503 = -(((aa1[(long) i + offset_6] - aa0[(long) i + offset_2])) / ((*dtime * xkshal)));
                    M_635 = gf_max(D_6503, M_635);
                    xff_shal[1] = M_635;
                  }
                  {
                    GFReal D_6504;
                    GFReal M_636;
                    GFReal D_6506;

                    D_6504 = xmbmax[(long) i + offset_490];
                    M_636 = D_6504;
                    D_6506 = xff_shal[1];
                    M_636 = gf_min(D_6506, M_636);
                    xff_shal[1] = M_636;
                  }
                  {
                    GFReal M_637;
                    GFReal D_6508;

                    M_637 = 0.0f;
                    D_6508 = -(((aa1[(long) i + offset_6] - aa0[(long) i + offset_2])) / ((*dtime * xkshal)));
                    M_637 = gf_max(D_6508, M_637);
                    xff_shal[2] = M_637;
                  }
                  {
                    GFReal D_6509;
                    GFReal M_638;
                    GFReal D_6511;

                    D_6509 = xmbmax[(long) i + offset_490];
                    M_638 = D_6509;
                    D_6511 = xff_shal[2];
                    M_638 = gf_min(D_6511, M_638);
                    xff_shal[2] = M_638;
                  }
                  if (aa1[(long) i + offset_6] <= 0.0f)
                    {
                      xff_shal[0] = 0.0f;
                      xff_shal[1] = 0.0f;
                      xff_shal[2] = 0.0f;
                    }

                  if (aa1[(long) i + offset_6] - aa0[(long) i + offset_2] <= 0.0f)
                    {
                      xff_shal[0] = 0.0f;
                      xff_shal[1] = 0.0f;
                      xff_shal[2] = 0.0f;
                    }

                  blqe = 0.0f;
                  trash = 0.0f;
                  if ((kpbl[(long) i + offset_229]) >= k22[(long) i + offset_217])
                    {
                      {
                        int D_6515;

                        D_6515 = kbcon[(long) i + offset_221] + -1;
                        k = 1;
                        while (1)
                          {
                            {
                              bool D_6518;

                              D_6518 = k > D_6515;
                              if (D_6518) break;
                              blqe = ((dhdt[((long) k * stride_80 + offset_81) + (long) i] * 1.0e+2f) * ((po_cup[((long) k * stride_281 + offset_282) + (long) i] - po_cup[((long) (k + 1) * stride_281 + offset_282) + (long) i]))) / *g + blqe;

                              k = k + 1;
                            }
                          }

                      }
                      {
                        GFReal D_6519;
                        GFReal M_639;

                        D_6519 = ((hc[((long) kbcon[(long) i + offset_221] * stride_129 + offset_130) + (long) i] - he_cup[((long) kbcon[(long) i + offset_221] * stride_150 + offset_151) + (long) i]));
                        M_639 = D_6519;
                        M_639 = gf_max(M_639, 1.0e+1f);
                        trash = M_639;
                      }
                      {
                        GFReal M_640;
                        GFReal D_6522;

                        M_640 = 0.0f;
                        D_6522 = blqe / trash;
                        M_640 = gf_max(D_6522, M_640);
                        xff_shal[6] = M_640;
                      }
                      {
                        GFReal D_6523;
                        GFReal M_641;
                        GFReal D_6525;

                        D_6523 = xmbmax[(long) i + offset_490];
                        M_641 = D_6523;
                        D_6525 = xff_shal[6];
                        M_641 = gf_min(D_6525, M_641);
                        xff_shal[6] = M_641;
                      }
                    }
                  else
                    {
                      xff_shal[6] = 0.0f;
                    }

                  xff_max = 0.0f;
                  if (xkshal < -1.100000008591450750827789306640625e-4f)
                    {
                      {
                        GFReal M_642;
                        GFReal D_6528;

                        M_642 = 0.0f;
                        D_6528 = -(aa0[(long) i + offset_2] / ((tscl_kf[(long) i + offset_392] * xkshal)));
                        M_642 = gf_max(D_6528, M_642);
                        xff_shal[3] = M_642;
                      }
                      {
                        GFReal D_6529;
                        GFReal M_643;
                        GFReal D_6531;

                        D_6529 = xmbmax[(long) i + offset_490];
                        M_643 = D_6529;
                        D_6531 = xff_shal[3];
                        M_643 = gf_min(D_6531, M_643);
                        xff_shal[3] = M_643;
                      }
                      xff_shal[4] = xff_shal[3];
                      xff_shal[5] = xff_shal[3];
                    }
                  else
                    {
                      xff_shal[3] = 0.0f;
                      xff_shal[4] = 0.0f;
                      xff_shal[5] = 0.0f;
                    }

                  xff_shal[7] = xff_shal[6];
                  xff_shal[8] = xff_shal[6];
                  fsum = 0.0f;
                  k = 1;
                  while (1)
                    {
                      {
                        bool D_6534;

                        D_6534 = k > 9;
                        if (D_6534) break;
                        if (*ichoice > 0)
                          {
                            xmb[(long) i + offset_482] = xmb[(long) i + offset_482] + xff_shal[(long) *ichoice + -1];
                          }
                        else
                          {
                            if (*ichoice == 0)
                              {
                                xmb[(long) i + offset_482] = xmb[(long) i + offset_482] + xff_shal[(long) k + -1];
                              }
                            else
                              {
                                if (*ichoice < 0)
                                  {
                                    {
                                      GFReal limit_644;

                                      limit_644 =  -gf_infinity();
                                      {
                                        long S_645;

                                        S_645 = 1;
 bool gf_found=false;
                                        while (1)
                                          {
                                            if (S_645 > 9) break;
                                            {
                                              GFReal D_6541;

                                              D_6541 = xff_shal[S_645 + -1];
                                              if (D_6541 >= limit_644)
                                                {
                                                  limit_644 = D_6541;
                                                  S_645 = S_645 + 1;
                                                  gf_found=true; break;
                                                }
                                            }
                                            S_645 = S_645 + 1;
                                          }

                                        if (!gf_found) limit_644 = gf_nan();
                                        while (1)
                                          {
                                            if (S_645 > 9) break;
                                            {
                                              GFReal D_6543;

                                              D_6543 = xff_shal[S_645 + -1];
                                              if (D_6543 > limit_644)
                                                {
                                                  limit_644 = D_6543;
                                                }
                                            }
                                            S_645 = S_645 + 1;
                                          }

                                      }
                                      xff_max = limit_644;
                                    }
                                    xmb[(long) i + offset_482] = xmb[(long) i + offset_482] + xff_max;
                                  }

                              }

                          }

                        fsum = fsum + 1.0e+0f;

                        k = k + 1;
                      }
                    }

                  {
                    GFReal D_6545;
                    GFReal M_646;
                    GFReal D_6547;

                    D_6545 = xmbmax[(long) i + offset_490];
                    M_646 = D_6545;
                    D_6547 = xmb[(long) i + offset_482] / fsum;
                    M_646 = gf_min(D_6547, M_646);
                    xmb[(long) i + offset_482] = M_646;
                  }
                  if (xmb[(long) i + offset_482] == 0.0f)
                    {
                      ierr[(long) i + offset_205] = 22;
                    }

                  if (xmb[(long) i + offset_482] == 0.0f)
                    {
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "22", 2);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 2, 32, 48);
                    }

                  if (xmb[(long) i + offset_482] < 0.0f)
                    {
                      ierr[(long) i + offset_205] = 21;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_209], "21", 2);
                      gf_memset ((device char *)&ierrc[(long) i + offset_209] + 2, 32, 48);
                    }

                }

              if (ierr[(long) i + offset_205] != 0)
                {
                  k22[(long) i + offset_217] = 0;
                  kbcon[(long) i + offset_221] = 0;
                  ktop[(long) i + offset_233] = 0;
                  xmb[(long) i + offset_482] = 0.0f;
                  {
                    int D_6552;
                    int D_6553;

                    D_6552 = *kts;
                    D_6553 = *ktf;
                    k = D_6552;
                    while (1)
                      {
                        {
                          bool D_6556;

                          D_6556 = k > D_6553;
                          if (D_6556) break;
                          outt[((long) k * stride_253 + offset_254) + (long) i] = 0.0f;
                          outq[((long) k * stride_239 + offset_240) + (long) i] = 0.0f;
                          outqc[((long) k * stride_246 + offset_247) + (long) i] = 0.0f;

                          k = k + 1;
                        }
                      }

                  }
                }
              else
                {
                  if (ierr[(long) i + offset_205] == 0)
                    {
                      trash = 0.0f;
                      {
                        int D_6558;

                        D_6558 = ktop[(long) i + offset_233];
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_6561;

                              D_6561 = k > D_6558;
                              if (D_6561) break;
                              {
                                GFReal M_647;
                                GFReal D_6563;

                                M_647 = trash;
                                D_6563 = (((dsubt[((long) k * stride_101 + offset_102) + (long) i] + dellat[((long) k * stride_73 + offset_74) + (long) i])) * 8.64e+4f) * xmb[(long) i + offset_482];
                                M_647 = gf_max(D_6563, M_647);
                                trash = M_647;
                              }

                              k = k + 1;
                            }
                          }

                      }
                      if (trash > 1.0e+2f)
                        {
                          xmb[(long) i + offset_482] = (xmb[(long) i + offset_482] * 1.0e+2f) / trash;
                        }

                      trash = 0.0f;
                      {
                        int D_6565;

                        D_6565 = ktop[(long) i + offset_233];
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_6568;

                              D_6568 = k > D_6565;
                              if (D_6568) break;
                              {
                                GFReal M_648;
                                GFReal D_6570;

                                M_648 = trash;
                                D_6570 = (((dsubt[((long) k * stride_101 + offset_102) + (long) i] + dellat[((long) k * stride_73 + offset_74) + (long) i])) * 8.64e+4f) * xmb[(long) i + offset_482];
                                M_648 = gf_min(D_6570, M_648);
                                trash = M_648;
                              }

                              k = k + 1;
                            }
                          }

                      }
                      if (trash < -1.0e+2f)
                        {
                          xmb[(long) i + offset_482] = -((xmb[(long) i + offset_482] * 1.0e+2f) / trash);
                        }

                      {
                        int D_6572;

                        D_6572 = ktop[(long) i + offset_233];
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_6575;

                              D_6575 = k > D_6572;
                              if (D_6575) break;
                              trash = q[((long) k * stride_292 + offset_293) + (long) i] + (((dsubq[((long) k * stride_94 + offset_95) + (long) i] + dellaq[((long) k * stride_59 + offset_60) + (long) i])) * xmb[(long) i + offset_482]) * *dtime;
                              if (trash < 9.999999960041972002500187954865396022796630859375e-13f)
                                {
                                  trash = ((9.999999960041972002500187954865396022796630859375e-13f - q[((long) k * stride_292 + offset_293) + (long) i])) / ((((dsubq[((long) k * stride_94 + offset_95) + (long) i] + dellaq[((long) k * stride_59 + offset_60) + (long) i])) * *dtime));
                                  xmb[(long) i + offset_482] = ((9.999999960041972002500187954865396022796630859375e-13f - q[((long) k * stride_292 + offset_293) + (long) i])) / ((((dsubq[((long) k * stride_94 + offset_95) + (long) i] + dellaq[((long) k * stride_59 + offset_60) + (long) i])) * *dtime));
                                }


                              k = k + 1;
                            }
                          }

                      }
                      xmb_out[(long) i + offset_486] = xmb[(long) i + offset_482];
                      {
                        int D_6577;

                        D_6577 = ktop[(long) i + offset_233];
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_6580;

                              D_6580 = k > D_6577;
                              if (D_6580) break;
                              outt[((long) k * stride_253 + offset_254) + (long) i] = ((dsubt[((long) k * stride_101 + offset_102) + (long) i] + dellat[((long) k * stride_73 + offset_74) + (long) i])) * xmb[(long) i + offset_482];
                              outq[((long) k * stride_239 + offset_240) + (long) i] = ((dsubq[((long) k * stride_94 + offset_95) + (long) i] + dellaq[((long) k * stride_59 + offset_60) + (long) i])) * xmb[(long) i + offset_482];

                              k = k + 1;
                            }
                          }

                      }
                    }

                }


              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}
template<typename P0> static inline GFReal satvap(thread GFWorkspace &workspace, P0 temp2) {
  GFFrame frame(workspace);
  GFReal eilog;
  GFReal ewlog;
  GFReal ewlog2;
  GFReal ewlog3;
  GFReal ewlog4;
  GFReal temp;
  GFReal toot;
  GFReal toto;
  GFReal tsot;
  GFReal __result_satvap;

  temp = *temp2 - 2.73154998779296875e+2f;
  if (temp < -2.0e+1f)
    {
      toot = 2.73160003662109375e+2f / *temp2;
      toto = 1.0e+0f / toot;
      eilog = ((-(((toot - 1.0e+0f)) * 9.09718036651611328125e+0f) - ((gf_log (toot) / 2.302585124969482421875e+0f)) * 3.5665400028228759765625e+0f) + ((1.0e+0f - toto)) * 8.7679302692413330078125e-1f) + 7.858350276947021484375e-1f;
      __result_satvap = gf_pow (1.0e+1f, eilog);
    }
  else
    {
      tsot = 3.73160003662109375e+2f / *temp2;
      ewlog = ((gf_log (tsot) / 2.302585124969482421875e+0f)) * 5.028079986572265625e+0f - ((tsot - 1.0e+0f)) * 7.90297985076904296875e+0f;
      ewlog2 = ewlog - ((gf_pow (1.0e+1f, ((((1.0e+0f - ((1.0e+0f / tsot)))) * 1.13439998626708984375e+1f))) - 1.0e+0f)) * 1.38160004325982299633324146270751953125e-7f;
      ewlog3 = ((gf_pow (1.0e+1f, ((-(((tsot - 1.0e+0f)) * 3.491489887237548828125e+0f)))) - 1.0e+0f)) * 8.13280045986175537109375e-3f + ewlog2;
      ewlog4 = ewlog3 + 3.005714893341064453125e+0f;
      __result_satvap = gf_pow (1.0e+1f, ewlog4);
    }

  return __result_satvap;
  return __result_satvap;
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38> static inline void gdinit(thread GFWorkspace &workspace, P0 rthcuten, P1 rqvcuten, P2 rqccuten, P3 rqicuten, P4 mass_flux, P5 cp, P6 restart, P7 p_qc, P8 p_qi, P9 p_first_scalar, P10 rthften, P11 rqvften, P12 apr_gr, P13 apr_w, P14 apr_mc, P15 apr_st, P16 apr_as, P17 apr_capma, P18 apr_capme, P19 apr_capmi, P20 allowed_to_read, P21 ids, P22 ide, P23 jds, P24 jde, P25 kds, P26 kde, P27 ims, P28 ime, P29 jms, P30 jme, P31 kms, P32 kme, P33 its, P34 ite, P35 jts, P36 jte, P37 kts, P38 kte) {
  GFFrame frame(workspace);
  long lbound_649;
  long ubound_650;
  long lbound_651;
  long ubound_652;
  long stride_653;
  long offset_654;
  long size_655;
  long lbound_656;
  long ubound_657;
  long lbound_658;
  long ubound_659;
  long stride_660;
  long offset_661;
  long size_662;
  long lbound_663;
  long ubound_664;
  long lbound_665;
  long ubound_666;
  long stride_667;
  long offset_668;
  long size_669;
  long lbound_670;
  long ubound_671;
  long lbound_672;
  long ubound_673;
  long stride_674;
  long offset_675;
  long size_676;
  long lbound_677;
  long ubound_678;
  long lbound_679;
  long ubound_680;
  long stride_681;
  long offset_682;
  long size_683;
  long lbound_684;
  long ubound_685;
  long lbound_686;
  long ubound_687;
  long stride_688;
  long offset_689;
  long size_690;
  long lbound_691;
  long ubound_692;
  long lbound_693;
  long ubound_694;
  long stride_695;
  long offset_696;
  long size_697;
  long lbound_698;
  long ubound_699;
  long lbound_700;
  long ubound_701;
  long stride_702;
  long offset_703;
  long size_704;
  long lbound_705;
  long ubound_706;
  long lbound_707;
  long ubound_708;
  long stride_709;
  long offset_710;
  long size_711;
  long lbound_712;
  long ubound_713;
  long lbound_714;
  long ubound_715;
  long stride_716;
  long lbound_717;
  long ubound_718;
  long stride_719;
  long offset_720;
  long size_721;
  long lbound_722;
  long ubound_723;
  long lbound_724;
  long ubound_725;
  long stride_726;
  long lbound_727;
  long ubound_728;
  long stride_729;
  long offset_730;
  long size_731;
  long lbound_732;
  long ubound_733;
  long lbound_734;
  long ubound_735;
  long stride_736;
  long lbound_737;
  long ubound_738;
  long stride_739;
  long offset_740;
  long size_741;
  long lbound_742;
  long ubound_743;
  long lbound_744;
  long ubound_745;
  long stride_746;
  long lbound_747;
  long ubound_748;
  long stride_749;
  long offset_750;
  long size_751;
  long lbound_752;
  long ubound_753;
  long lbound_754;
  long ubound_755;
  long stride_756;
  long lbound_757;
  long ubound_758;
  long stride_759;
  long offset_760;
  long size_761;
  long lbound_762;
  long ubound_763;
  long lbound_764;
  long ubound_765;
  long stride_766;
  long lbound_767;
  long ubound_768;
  long stride_769;
  long offset_770;
  long size_771;
  long D_7287;
  ulong D_7288;
  ulong D_7289;
  long D_7290;
  ulong D_7291;
  ulong D_7292;
  long D_7293;
  ulong D_7294;
  ulong D_7295;
  long D_7296;
  ulong D_7297;
  ulong D_7298;
  long D_7299;
  ulong D_7300;
  ulong D_7301;
  long D_7302;
  ulong D_7303;
  ulong D_7304;
  long D_7305;
  ulong D_7306;
  ulong D_7307;
  long D_7308;
  ulong D_7309;
  ulong D_7310;
  long D_7311;
  ulong D_7312;
  ulong D_7313;
  long D_7314;
  ulong D_7315;
  ulong D_7316;
  long D_7317;
  ulong D_7318;
  ulong D_7319;
  long D_7320;
  ulong D_7321;
  ulong D_7322;
  long D_7323;
  ulong D_7324;
  ulong D_7325;
  long D_7326;
  ulong D_7327;
  ulong D_7328;
  long D_7329;
  ulong D_7330;
  ulong D_7331;

  lbound_752 = (long) *ims;
  ubound_753 = (long) *ime;
  stride_756 = (1 - lbound_752) + ubound_753;
  stride_756 = gf_max(stride_756, 0);
  lbound_754 = (long) *kms;
  ubound_755 = (long) *kme;
  stride_759 = ((1 - lbound_754) + ubound_755) * stride_756;
  stride_759 = gf_max(stride_759, 0);
  lbound_757 = (long) *jms;
  ubound_758 = (long) *jme;
  size_761 = ((1 - lbound_757) + ubound_758) * stride_759;
  size_761 = gf_max(size_761, 0);
  D_7329 = size_761 + -1;
  D_7330 = (ulong) (ulong) (size_761) * 32;
  D_7331 = (ulong) (size_761) * 4;
  offset_760 = (-(lbound_752) - lbound_754 * stride_756) - lbound_757 * stride_759;
  lbound_732 = (long) *ims;
  ubound_733 = (long) *ime;
  stride_736 = (1 - lbound_732) + ubound_733;
  stride_736 = gf_max(stride_736, 0);
  lbound_734 = (long) *kms;
  ubound_735 = (long) *kme;
  stride_739 = ((1 - lbound_734) + ubound_735) * stride_736;
  stride_739 = gf_max(stride_739, 0);
  lbound_737 = (long) *jms;
  ubound_738 = (long) *jme;
  size_741 = ((1 - lbound_737) + ubound_738) * stride_739;
  size_741 = gf_max(size_741, 0);
  D_7326 = size_741 + -1;
  D_7327 = (ulong) (ulong) (size_741) * 32;
  D_7328 = (ulong) (size_741) * 4;
  offset_740 = (-(lbound_732) - lbound_734 * stride_736) - lbound_737 * stride_739;
  lbound_762 = (long) *ims;
  ubound_763 = (long) *ime;
  stride_766 = (1 - lbound_762) + ubound_763;
  stride_766 = gf_max(stride_766, 0);
  lbound_764 = (long) *kms;
  ubound_765 = (long) *kme;
  stride_769 = ((1 - lbound_764) + ubound_765) * stride_766;
  stride_769 = gf_max(stride_769, 0);
  lbound_767 = (long) *jms;
  ubound_768 = (long) *jme;
  size_771 = ((1 - lbound_767) + ubound_768) * stride_769;
  size_771 = gf_max(size_771, 0);
  D_7323 = size_771 + -1;
  D_7324 = (ulong) (ulong) (size_771) * 32;
  D_7325 = (ulong) (size_771) * 4;
  offset_770 = (-(lbound_762) - lbound_764 * stride_766) - lbound_767 * stride_769;
  lbound_742 = (long) *ims;
  ubound_743 = (long) *ime;
  stride_746 = (1 - lbound_742) + ubound_743;
  stride_746 = gf_max(stride_746, 0);
  lbound_744 = (long) *kms;
  ubound_745 = (long) *kme;
  stride_749 = ((1 - lbound_744) + ubound_745) * stride_746;
  stride_749 = gf_max(stride_749, 0);
  lbound_747 = (long) *jms;
  ubound_748 = (long) *jme;
  size_751 = ((1 - lbound_747) + ubound_748) * stride_749;
  size_751 = gf_max(size_751, 0);
  D_7320 = size_751 + -1;
  D_7321 = (ulong) (ulong) (size_751) * 32;
  D_7322 = (ulong) (size_751) * 4;
  offset_750 = (-(lbound_742) - lbound_744 * stride_746) - lbound_747 * stride_749;
  lbound_712 = (long) *ims;
  ubound_713 = (long) *ime;
  stride_716 = (1 - lbound_712) + ubound_713;
  stride_716 = gf_max(stride_716, 0);
  lbound_714 = (long) *kms;
  ubound_715 = (long) *kme;
  stride_719 = ((1 - lbound_714) + ubound_715) * stride_716;
  stride_719 = gf_max(stride_719, 0);
  lbound_717 = (long) *jms;
  ubound_718 = (long) *jme;
  size_721 = ((1 - lbound_717) + ubound_718) * stride_719;
  size_721 = gf_max(size_721, 0);
  D_7317 = size_721 + -1;
  D_7318 = (ulong) (ulong) (size_721) * 32;
  D_7319 = (ulong) (size_721) * 4;
  offset_720 = (-(lbound_712) - lbound_714 * stride_716) - lbound_717 * stride_719;
  lbound_722 = (long) *ims;
  ubound_723 = (long) *ime;
  stride_726 = (1 - lbound_722) + ubound_723;
  stride_726 = gf_max(stride_726, 0);
  lbound_724 = (long) *kms;
  ubound_725 = (long) *kme;
  stride_729 = ((1 - lbound_724) + ubound_725) * stride_726;
  stride_729 = gf_max(stride_729, 0);
  lbound_727 = (long) *jms;
  ubound_728 = (long) *jme;
  size_731 = ((1 - lbound_727) + ubound_728) * stride_729;
  size_731 = gf_max(size_731, 0);
  D_7314 = size_731 + -1;
  D_7315 = (ulong) (ulong) (size_731) * 32;
  D_7316 = (ulong) (size_731) * 4;
  offset_730 = (-(lbound_722) - lbound_724 * stride_726) - lbound_727 * stride_729;
  lbound_705 = (long) *ims;
  ubound_706 = (long) *ime;
  stride_709 = (1 - lbound_705) + ubound_706;
  stride_709 = gf_max(stride_709, 0);
  lbound_707 = (long) *jms;
  ubound_708 = (long) *jme;
  size_711 = ((1 - lbound_707) + ubound_708) * stride_709;
  size_711 = gf_max(size_711, 0);
  D_7311 = size_711 + -1;
  D_7312 = (ulong) (ulong) (size_711) * 32;
  D_7313 = (ulong) (size_711) * 4;
  offset_710 = -(lbound_705) - lbound_707 * stride_709;
  lbound_677 = (long) *ims;
  ubound_678 = (long) *ime;
  stride_681 = (1 - lbound_677) + ubound_678;
  stride_681 = gf_max(stride_681, 0);
  lbound_679 = (long) *jms;
  ubound_680 = (long) *jme;
  size_683 = ((1 - lbound_679) + ubound_680) * stride_681;
  size_683 = gf_max(size_683, 0);
  D_7308 = size_683 + -1;
  D_7309 = (ulong) (ulong) (size_683) * 32;
  D_7310 = (ulong) (size_683) * 4;
  offset_682 = -(lbound_677) - lbound_679 * stride_681;
  lbound_691 = (long) *ims;
  ubound_692 = (long) *ime;
  stride_695 = (1 - lbound_691) + ubound_692;
  stride_695 = gf_max(stride_695, 0);
  lbound_693 = (long) *jms;
  ubound_694 = (long) *jme;
  size_697 = ((1 - lbound_693) + ubound_694) * stride_695;
  size_697 = gf_max(size_697, 0);
  D_7305 = size_697 + -1;
  D_7306 = (ulong) (ulong) (size_697) * 32;
  D_7307 = (ulong) (size_697) * 4;
  offset_696 = -(lbound_691) - lbound_693 * stride_695;
  lbound_698 = (long) *ims;
  ubound_699 = (long) *ime;
  stride_702 = (1 - lbound_698) + ubound_699;
  stride_702 = gf_max(stride_702, 0);
  lbound_700 = (long) *jms;
  ubound_701 = (long) *jme;
  size_704 = ((1 - lbound_700) + ubound_701) * stride_702;
  size_704 = gf_max(size_704, 0);
  D_7302 = size_704 + -1;
  D_7303 = (ulong) (ulong) (size_704) * 32;
  D_7304 = (ulong) (size_704) * 4;
  offset_703 = -(lbound_698) - lbound_700 * stride_702;
  lbound_684 = (long) *ims;
  ubound_685 = (long) *ime;
  stride_688 = (1 - lbound_684) + ubound_685;
  stride_688 = gf_max(stride_688, 0);
  lbound_686 = (long) *jms;
  ubound_687 = (long) *jme;
  size_690 = ((1 - lbound_686) + ubound_687) * stride_688;
  size_690 = gf_max(size_690, 0);
  D_7299 = size_690 + -1;
  D_7300 = (ulong) (ulong) (size_690) * 32;
  D_7301 = (ulong) (size_690) * 4;
  offset_689 = -(lbound_684) - lbound_686 * stride_688;
  lbound_649 = (long) *ims;
  ubound_650 = (long) *ime;
  stride_653 = (1 - lbound_649) + ubound_650;
  stride_653 = gf_max(stride_653, 0);
  lbound_651 = (long) *jms;
  ubound_652 = (long) *jme;
  size_655 = ((1 - lbound_651) + ubound_652) * stride_653;
  size_655 = gf_max(size_655, 0);
  D_7296 = size_655 + -1;
  D_7297 = (ulong) (ulong) (size_655) * 32;
  D_7298 = (ulong) (size_655) * 4;
  offset_654 = -(lbound_649) - lbound_651 * stride_653;
  lbound_656 = (long) *ims;
  ubound_657 = (long) *ime;
  stride_660 = (1 - lbound_656) + ubound_657;
  stride_660 = gf_max(stride_660, 0);
  lbound_658 = (long) *jms;
  ubound_659 = (long) *jme;
  size_662 = ((1 - lbound_658) + ubound_659) * stride_660;
  size_662 = gf_max(size_662, 0);
  D_7293 = size_662 + -1;
  D_7294 = (ulong) (ulong) (size_662) * 32;
  D_7295 = (ulong) (size_662) * 4;
  offset_661 = -(lbound_656) - lbound_658 * stride_660;
  lbound_663 = (long) *ims;
  ubound_664 = (long) *ime;
  stride_667 = (1 - lbound_663) + ubound_664;
  stride_667 = gf_max(stride_667, 0);
  lbound_665 = (long) *jms;
  ubound_666 = (long) *jme;
  size_669 = ((1 - lbound_665) + ubound_666) * stride_667;
  size_669 = gf_max(size_669, 0);
  D_7290 = size_669 + -1;
  D_7291 = (ulong) (ulong) (size_669) * 32;
  D_7292 = (ulong) (size_669) * 4;
  offset_668 = -(lbound_663) - lbound_665 * stride_667;
  lbound_670 = (long) *ims;
  ubound_671 = (long) *ime;
  stride_674 = (1 - lbound_670) + ubound_671;
  stride_674 = gf_max(stride_674, 0);
  lbound_672 = (long) *jms;
  ubound_673 = (long) *jme;
  size_676 = ((1 - lbound_672) + ubound_673) * stride_674;
  size_676 = gf_max(size_676, 0);
  D_7287 = size_676 + -1;
  D_7288 = (ulong) (ulong) (size_676) * 32;
  D_7289 = (ulong) (size_676) * 4;
  offset_675 = -(lbound_670) - lbound_672 * stride_674;
  if (!*restart)
    {
      {
        long D_7105;
        long D_7106;
        long D_7107;
        long D_7108;
        long D_7109;
        long D_7110;
        long D_7111;

        D_7105 = offset_760;
        D_7106 = lbound_752;
        D_7107 = ubound_753;
        D_7108 = lbound_754;
        D_7109 = ubound_755;
        D_7110 = lbound_757;
        D_7111 = ubound_758;
        {
          long S_772;

          S_772 = D_7110;
          while (1)
            {
              if (S_772 > D_7111) break;
              {
                long S_773;
                long D_7114;

                D_7114 = S_772 * stride_759 + D_7105;
                S_773 = D_7108;
                while (1)
                  {
                    if (S_773 > D_7109) break;
                    {
                      long S_774;
                      long D_7116;

                      D_7116 = S_773 * stride_756 + D_7114;
                      S_774 = D_7106;
                      while (1)
                        {
                          if (S_774 > D_7107) break;
                          rthcuten[S_774 + D_7116] = 0.0f;
                          S_774 = S_774 + 1;
                        }

                    }
                    S_773 = S_773 + 1;
                  }

              }
              S_772 = S_772 + 1;
            }

        }
      }
      {
        long D_7120;
        long D_7121;
        long D_7122;
        long D_7123;
        long D_7124;
        long D_7125;
        long D_7126;

        D_7120 = offset_740;
        D_7121 = lbound_732;
        D_7122 = ubound_733;
        D_7123 = lbound_734;
        D_7124 = ubound_735;
        D_7125 = lbound_737;
        D_7126 = ubound_738;
        {
          long S_775;

          S_775 = D_7125;
          while (1)
            {
              if (S_775 > D_7126) break;
              {
                long S_776;
                long D_7129;

                D_7129 = S_775 * stride_739 + D_7120;
                S_776 = D_7123;
                while (1)
                  {
                    if (S_776 > D_7124) break;
                    {
                      long S_777;
                      long D_7131;

                      D_7131 = S_776 * stride_736 + D_7129;
                      S_777 = D_7121;
                      while (1)
                        {
                          if (S_777 > D_7122) break;
                          rqvcuten[S_777 + D_7131] = 0.0f;
                          S_777 = S_777 + 1;
                        }

                    }
                    S_776 = S_776 + 1;
                  }

              }
              S_775 = S_775 + 1;
            }

        }
      }
      {
        long D_7135;
        long D_7136;
        long D_7137;
        long D_7138;
        long D_7139;
        long D_7140;
        long D_7141;

        D_7135 = offset_770;
        D_7136 = lbound_762;
        D_7137 = ubound_763;
        D_7138 = lbound_764;
        D_7139 = ubound_765;
        D_7140 = lbound_767;
        D_7141 = ubound_768;
        {
          long S_778;

          S_778 = D_7140;
          while (1)
            {
              if (S_778 > D_7141) break;
              {
                long S_779;
                long D_7144;

                D_7144 = S_778 * stride_769 + D_7135;
                S_779 = D_7138;
                while (1)
                  {
                    if (S_779 > D_7139) break;
                    {
                      long S_780;
                      long D_7146;

                      D_7146 = S_779 * stride_766 + D_7144;
                      S_780 = D_7136;
                      while (1)
                        {
                          if (S_780 > D_7137) break;
                          rthften[S_780 + D_7146] = 0.0f;
                          S_780 = S_780 + 1;
                        }

                    }
                    S_779 = S_779 + 1;
                  }

              }
              S_778 = S_778 + 1;
            }

        }
      }
      {
        long D_7150;
        long D_7151;
        long D_7152;
        long D_7153;
        long D_7154;
        long D_7155;
        long D_7156;

        D_7150 = offset_750;
        D_7151 = lbound_742;
        D_7152 = ubound_743;
        D_7153 = lbound_744;
        D_7154 = ubound_745;
        D_7155 = lbound_747;
        D_7156 = ubound_748;
        {
          long S_781;

          S_781 = D_7155;
          while (1)
            {
              if (S_781 > D_7156) break;
              {
                long S_782;
                long D_7159;

                D_7159 = S_781 * stride_749 + D_7150;
                S_782 = D_7153;
                while (1)
                  {
                    if (S_782 > D_7154) break;
                    {
                      long S_783;
                      long D_7161;

                      D_7161 = S_782 * stride_746 + D_7159;
                      S_783 = D_7151;
                      while (1)
                        {
                          if (S_783 > D_7152) break;
                          rqvften[S_783 + D_7161] = 0.0f;
                          S_783 = S_783 + 1;
                        }

                    }
                    S_782 = S_782 + 1;
                  }

              }
              S_781 = S_781 + 1;
            }

        }
      }
      if (*p_qc >= *p_first_scalar)
        {
          {
            long D_7166;
            long D_7167;
            long D_7168;
            long D_7169;
            long D_7170;
            long D_7171;
            long D_7172;

            D_7166 = offset_720;
            D_7167 = lbound_712;
            D_7168 = ubound_713;
            D_7169 = lbound_714;
            D_7170 = ubound_715;
            D_7171 = lbound_717;
            D_7172 = ubound_718;
            {
              long S_784;

              S_784 = D_7171;
              while (1)
                {
                  if (S_784 > D_7172) break;
                  {
                    long S_785;
                    long D_7175;

                    D_7175 = S_784 * stride_719 + D_7166;
                    S_785 = D_7169;
                    while (1)
                      {
                        if (S_785 > D_7170) break;
                        {
                          long S_786;
                          long D_7177;

                          D_7177 = S_785 * stride_716 + D_7175;
                          S_786 = D_7167;
                          while (1)
                            {
                              if (S_786 > D_7168) break;
                              rqccuten[S_786 + D_7177] = 0.0f;
                              S_786 = S_786 + 1;
                            }

                        }
                        S_785 = S_785 + 1;
                      }

                  }
                  S_784 = S_784 + 1;
                }

            }
          }
        }

      if (*p_qi >= *p_first_scalar)
        {
          {
            long D_7182;
            long D_7183;
            long D_7184;
            long D_7185;
            long D_7186;
            long D_7187;
            long D_7188;

            D_7182 = offset_730;
            D_7183 = lbound_722;
            D_7184 = ubound_723;
            D_7185 = lbound_724;
            D_7186 = ubound_725;
            D_7187 = lbound_727;
            D_7188 = ubound_728;
            {
              long S_787;

              S_787 = D_7187;
              while (1)
                {
                  if (S_787 > D_7188) break;
                  {
                    long S_788;
                    long D_7191;

                    D_7191 = S_787 * stride_729 + D_7182;
                    S_788 = D_7185;
                    while (1)
                      {
                        if (S_788 > D_7186) break;
                        {
                          long S_789;
                          long D_7193;

                          D_7193 = S_788 * stride_726 + D_7191;
                          S_789 = D_7183;
                          while (1)
                            {
                              if (S_789 > D_7184) break;
                              rqicuten[S_789 + D_7193] = 0.0f;
                              S_789 = S_789 + 1;
                            }

                        }
                        S_788 = S_788 + 1;
                      }

                  }
                  S_787 = S_787 + 1;
                }

            }
          }
        }

      {
        long D_7197;
        long D_7198;
        long D_7199;
        long D_7200;
        long D_7201;

        D_7197 = offset_710;
        D_7198 = lbound_705;
        D_7199 = ubound_706;
        D_7200 = lbound_707;
        D_7201 = ubound_708;
        {
          long S_790;

          S_790 = D_7200;
          while (1)
            {
              if (S_790 > D_7201) break;
              {
                long S_791;
                long D_7204;

                D_7204 = S_790 * stride_709 + D_7197;
                S_791 = D_7198;
                while (1)
                  {
                    if (S_791 > D_7199) break;
                    mass_flux[S_791 + D_7204] = 0.0f;
                    S_791 = S_791 + 1;
                  }

              }
              S_790 = S_790 + 1;
            }

        }
      }
    }

  {
    long D_7207;
    long D_7208;
    long D_7209;
    long D_7210;
    long D_7211;

    D_7207 = offset_682;
    D_7208 = lbound_677;
    D_7209 = ubound_678;
    D_7210 = lbound_679;
    D_7211 = ubound_680;
    {
      long S_792;

      S_792 = D_7210;
      while (1)
        {
          if (S_792 > D_7211) break;
          {
            long S_793;
            long D_7214;

            D_7214 = S_792 * stride_681 + D_7207;
            S_793 = D_7208;
            while (1)
              {
                if (S_793 > D_7209) break;
                apr_gr[S_793 + D_7214] = 0.0f;
                S_793 = S_793 + 1;
              }

          }
          S_792 = S_792 + 1;
        }

    }
  }
  {
    long D_7217;
    long D_7218;
    long D_7219;
    long D_7220;
    long D_7221;

    D_7217 = offset_696;
    D_7218 = lbound_691;
    D_7219 = ubound_692;
    D_7220 = lbound_693;
    D_7221 = ubound_694;
    {
      long S_794;

      S_794 = D_7220;
      while (1)
        {
          if (S_794 > D_7221) break;
          {
            long S_795;
            long D_7224;

            D_7224 = S_794 * stride_695 + D_7217;
            S_795 = D_7218;
            while (1)
              {
                if (S_795 > D_7219) break;
                apr_st[S_795 + D_7224] = 0.0f;
                S_795 = S_795 + 1;
              }

          }
          S_794 = S_794 + 1;
        }

    }
  }
  {
    long D_7227;
    long D_7228;
    long D_7229;
    long D_7230;
    long D_7231;

    D_7227 = offset_703;
    D_7228 = lbound_698;
    D_7229 = ubound_699;
    D_7230 = lbound_700;
    D_7231 = ubound_701;
    {
      long S_796;

      S_796 = D_7230;
      while (1)
        {
          if (S_796 > D_7231) break;
          {
            long S_797;
            long D_7234;

            D_7234 = S_796 * stride_702 + D_7227;
            S_797 = D_7228;
            while (1)
              {
                if (S_797 > D_7229) break;
                apr_w[S_797 + D_7234] = 0.0f;
                S_797 = S_797 + 1;
              }

          }
          S_796 = S_796 + 1;
        }

    }
  }
  {
    long D_7237;
    long D_7238;
    long D_7239;
    long D_7240;
    long D_7241;

    D_7237 = offset_689;
    D_7238 = lbound_684;
    D_7239 = ubound_685;
    D_7240 = lbound_686;
    D_7241 = ubound_687;
    {
      long S_798;

      S_798 = D_7240;
      while (1)
        {
          if (S_798 > D_7241) break;
          {
            long S_799;
            long D_7244;

            D_7244 = S_798 * stride_688 + D_7237;
            S_799 = D_7238;
            while (1)
              {
                if (S_799 > D_7239) break;
                apr_mc[S_799 + D_7244] = 0.0f;
                S_799 = S_799 + 1;
              }

          }
          S_798 = S_798 + 1;
        }

    }
  }
  {
    long D_7247;
    long D_7248;
    long D_7249;
    long D_7250;
    long D_7251;

    D_7247 = offset_654;
    D_7248 = lbound_649;
    D_7249 = ubound_650;
    D_7250 = lbound_651;
    D_7251 = ubound_652;
    {
      long S_800;

      S_800 = D_7250;
      while (1)
        {
          if (S_800 > D_7251) break;
          {
            long S_801;
            long D_7254;

            D_7254 = S_800 * stride_653 + D_7247;
            S_801 = D_7248;
            while (1)
              {
                if (S_801 > D_7249) break;
                apr_as[S_801 + D_7254] = 0.0f;
                S_801 = S_801 + 1;
              }

          }
          S_800 = S_800 + 1;
        }

    }
  }
  {
    long D_7257;
    long D_7258;
    long D_7259;
    long D_7260;
    long D_7261;

    D_7257 = offset_661;
    D_7258 = lbound_656;
    D_7259 = ubound_657;
    D_7260 = lbound_658;
    D_7261 = ubound_659;
    {
      long S_802;

      S_802 = D_7260;
      while (1)
        {
          if (S_802 > D_7261) break;
          {
            long S_803;
            long D_7264;

            D_7264 = S_802 * stride_660 + D_7257;
            S_803 = D_7258;
            while (1)
              {
                if (S_803 > D_7259) break;
                apr_capma[S_803 + D_7264] = 0.0f;
                S_803 = S_803 + 1;
              }

          }
          S_802 = S_802 + 1;
        }

    }
  }
  {
    long D_7267;
    long D_7268;
    long D_7269;
    long D_7270;
    long D_7271;

    D_7267 = offset_668;
    D_7268 = lbound_663;
    D_7269 = ubound_664;
    D_7270 = lbound_665;
    D_7271 = ubound_666;
    {
      long S_804;

      S_804 = D_7270;
      while (1)
        {
          if (S_804 > D_7271) break;
          {
            long S_805;
            long D_7274;

            D_7274 = S_804 * stride_667 + D_7267;
            S_805 = D_7268;
            while (1)
              {
                if (S_805 > D_7269) break;
                apr_capme[S_805 + D_7274] = 0.0f;
                S_805 = S_805 + 1;
              }

          }
          S_804 = S_804 + 1;
        }

    }
  }
  {
    long D_7277;
    long D_7278;
    long D_7279;
    long D_7280;
    long D_7281;

    D_7277 = offset_675;
    D_7278 = lbound_670;
    D_7279 = ubound_671;
    D_7280 = lbound_672;
    D_7281 = ubound_673;
    {
      long S_806;

      S_806 = D_7280;
      while (1)
        {
          if (S_806 > D_7281) break;
          {
            long S_807;
            long D_7284;

            D_7284 = S_806 * stride_674 + D_7277;
            S_807 = D_7278;
            while (1)
              {
                if (S_807 > D_7279) break;
                apr_capmi[S_807 + D_7284] = 0.0f;
                S_807 = S_807 + 1;
              }

          }
          S_806 = S_806 + 1;
        }

    }
  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44> static inline void cup_up_moisture(thread GFWorkspace &workspace, P0 name, P1 ierr, P2 z_cup, P3 qc, P4 qrc, P5 pw, P6 pwav, P7 ccnclean, P8 p_cup, P9 kbcon, P10 ktop, P11 cd, P12 dby, P13 clw_all, P14 t_cup, P15 q, P16 gamma_cup, P17 zu, P18 qes_cup, P19 k22, P20 qe_cup, P21 xl, P22 zqexec, P23 use_excess, P24 ccn, P25 rho, P26 up_massentr, P27 up_massdetr, P28 psum, P29 psumh, P30 autoconv, P31 aeroevap, P32 itest, P33 itf, P34 jtf, P35 ktf, P36 j, P37 ipr, P38 jpr, P39 its, P40 ite, P41 jts, P42 jte, P43 kts, P44 kte, long _name) {
  GFFrame frame(workspace);

  device void * D_7662;

  device void * D_7666;

  device void * D_7670;

  device void * D_7674;

  device void * D_7678;

  device void * D_7682;
  GFReal bdsp;
  GFReal berryc;
  GFReal berryc0;
  GFReal c0;
  long lbound_808;
  long ubound_809;
  long offset_810;
  long size_811;
  long lbound_812;
  long ubound_813;
  long lbound_814;
  long ubound_815;
  long stride_816;
  long offset_817;
  long size_818;
  long lbound_819;
  long ubound_820;
  long lbound_821;
  long ubound_822;
  long stride_823;
  long offset_824;
  long size_825;

  device GFReal * clw_allh;
  long lbound_826;
  long ubound_827;
  long lbound_828;
  long ubound_829;
  long stride_830;
  long offset_831;
  long size_832;
  GFReal dp;
  GFReal dz;
  GFReal g;
  long lbound_833;
  long ubound_834;
  long lbound_835;
  long ubound_836;
  long stride_837;
  long offset_838;
  long size_839;
  int i;
  int iall;
  long lbound_840;
  long ubound_841;
  long offset_842;
  long size_843;
  int iprop;
  int k;
  int k1;
  int k2;
  long lbound_844;
  long ubound_845;
  long offset_846;
  long size_847;
  long lbound_848;
  long ubound_849;
  long offset_850;
  long size_851;
  long lbound_852;
  long ubound_853;
  long offset_854;
  long size_855;
  long lbound_856;
  long ubound_857;
  long lbound_858;
  long ubound_859;
  long stride_860;
  long offset_861;
  long size_862;
  GFReal prop_ave;
  long lbound_863;
  long ubound_864;
  long offset_865;
  long size_866;

  device GFReal * prop_b;
  long lbound_867;
  long ubound_868;
  long offset_869;
  long size_870;
  long lbound_871;
  long ubound_872;
  long offset_873;
  long size_874;
  long lbound_875;
  long ubound_876;
  long lbound_877;
  long ubound_878;
  long stride_879;
  long offset_880;
  long size_881;
  long lbound_882;
  long ubound_883;
  long offset_884;
  long size_885;
  long lbound_886;
  long ubound_887;
  long offset_888;
  long size_889;

  device GFReal * pwavh;
  long lbound_890;
  long ubound_891;
  long lbound_892;
  long ubound_893;
  long stride_894;
  long offset_895;
  long size_896;

  device GFReal * pwh;
  long lbound_897;
  long ubound_898;
  long lbound_899;
  long ubound_900;
  long stride_901;
  long offset_902;
  long size_903;
  GFReal q1;
  long lbound_904;
  long ubound_905;
  long lbound_906;
  long ubound_907;
  long stride_908;
  long offset_909;
  long size_910;
  long lbound_911;
  long ubound_912;
  long lbound_913;
  long ubound_914;
  long stride_915;
  long offset_916;
  long size_917;

  device GFReal * qch;
  long lbound_918;
  long ubound_919;
  long lbound_920;
  long ubound_921;
  long stride_922;
  long offset_923;
  long size_924;
  long lbound_925;
  long ubound_926;
  long lbound_927;
  long ubound_928;
  long stride_929;
  long offset_930;
  long size_931;
  long lbound_932;
  long ubound_933;
  long lbound_934;
  long ubound_935;
  long stride_936;
  long offset_937;
  long size_938;
  long lbound_939;
  long ubound_940;
  long lbound_941;
  long ubound_942;
  long stride_943;
  long offset_944;
  long size_945;

  device GFReal * qrcb;
  GFReal qrcb_h;
  GFReal qrch;
  long lbound_946;
  long ubound_947;
  long lbound_948;
  long ubound_949;
  long stride_950;
  long offset_951;
  long size_952;
  GFReal rhoc;
  long lbound_953;
  long ubound_954;
  long lbound_955;
  long ubound_956;
  long stride_957;
  long offset_958;
  long size_959;
  long lbound_960;
  long ubound_961;
  long lbound_962;
  long ubound_963;
  long stride_964;
  long offset_965;
  long size_966;
  long lbound_967;
  long ubound_968;
  long lbound_969;
  long ubound_970;
  long stride_971;
  long offset_972;
  long size_973;
  long lbound_974;
  long ubound_975;
  long offset_976;
  long size_977;
  long lbound_978;
  long ubound_979;
  long lbound_980;
  long ubound_981;
  long stride_982;
  long offset_983;
  long size_984;
  long D_7659;
  ulong D_7660;
  ulong D_7661;
  long D_7663;
  ulong D_7664;
  ulong D_7665;
  long D_7667;
  ulong D_7668;
  ulong D_7669;
  long D_7671;
  ulong D_7672;
  ulong D_7673;
  long D_7675;
  ulong D_7676;
  ulong D_7677;
  long D_7679;
  ulong D_7680;
  ulong D_7681;
  long D_7683;
  ulong D_7684;
  ulong D_7685;
  long D_7686;
  ulong D_7687;
  ulong D_7688;
  long D_7689;
  ulong D_7690;
  ulong D_7691;
  long D_7692;
  ulong D_7693;
  ulong D_7694;
  long D_7695;
  ulong D_7696;
  ulong D_7697;
  long D_7698;
  ulong D_7699;
  ulong D_7700;
  long D_7701;
  ulong D_7702;
  ulong D_7703;
  long D_7704;
  ulong D_7705;
  ulong D_7706;
  long D_7707;
  ulong D_7708;
  ulong D_7709;
  long D_7710;
  ulong D_7711;
  ulong D_7712;
  long D_7713;
  ulong D_7714;
  ulong D_7715;
  long D_7716;
  ulong D_7717;
  ulong D_7718;
  long D_7719;
  ulong D_7720;
  ulong D_7721;
  long D_7722;
  ulong D_7723;
  ulong D_7724;
  long D_7725;
  ulong D_7726;
  ulong D_7727;
  long D_7728;
  ulong D_7729;
  ulong D_7730;
  long D_7731;
  ulong D_7732;
  ulong D_7733;
  long D_7734;
  ulong D_7735;
  ulong D_7736;
  long D_7737;
  ulong D_7738;
  ulong D_7739;
  long D_7740;
  ulong D_7741;
  ulong D_7742;
  long D_7743;
  ulong D_7744;
  ulong D_7745;
  long D_7746;
  ulong D_7747;
  ulong D_7748;
  long D_7749;
  ulong D_7750;
  ulong D_7751;
  long D_7752;
  ulong D_7753;
  ulong D_7754;
  ulong D_7755;
  ulong D_7756;

    {
      D_7755 = (ulong) (ulong) (_name) * 8;
      D_7756 = (ulong) (_name);
      lbound_882 = (long) *its;
      ubound_883 = (long) *ite;
      size_885 = (1 - lbound_882) + ubound_883;
      size_885 = gf_max(size_885, 0);
      D_7752 = size_885 + -1;
      D_7753 = (ulong) (ulong) (size_885) * 32;
      D_7754 = (ulong) (size_885) * 4;
      offset_884 = -(lbound_882);
      lbound_867 = (long) *its;
      ubound_868 = (long) *ite;
      size_870 = (1 - lbound_867) + ubound_868;
      size_870 = gf_max(size_870, 0);
      D_7749 = size_870 + -1;
      D_7750 = (ulong) (ulong) (size_870) * 32;
      D_7751 = (ulong) (size_870) * 4;
      offset_869 = -(lbound_867);
      lbound_871 = (long) *its;
      ubound_872 = (long) *ite;
      size_874 = (1 - lbound_871) + ubound_872;
      size_874 = gf_max(size_874, 0);
      D_7746 = size_874 + -1;
      D_7747 = (ulong) (ulong) (size_874) * 32;
      D_7748 = (ulong) (size_874) * 4;
      offset_873 = -(lbound_871);
      lbound_875 = (long) *its;
      ubound_876 = (long) *ite;
      stride_879 = (1 - lbound_875) + ubound_876;
      stride_879 = gf_max(stride_879, 0);
      lbound_877 = (long) *kts;
      ubound_878 = (long) *kte;
      size_881 = ((1 - lbound_877) + ubound_878) * stride_879;
      size_881 = gf_max(size_881, 0);
      D_7743 = size_881 + -1;
      D_7744 = (ulong) (ulong) (size_881) * 32;
      D_7745 = (ulong) (size_881) * 4;
      offset_880 = -(lbound_875) - lbound_877 * stride_879;
      lbound_904 = (long) *its;
      ubound_905 = (long) *ite;
      stride_908 = (1 - lbound_904) + ubound_905;
      stride_908 = gf_max(stride_908, 0);
      lbound_906 = (long) *kts;
      ubound_907 = (long) *kte;
      size_910 = ((1 - lbound_906) + ubound_907) * stride_908;
      size_910 = gf_max(size_910, 0);
      D_7740 = size_910 + -1;
      D_7741 = (ulong) (ulong) (size_910) * 32;
      D_7742 = (ulong) (size_910) * 4;
      offset_909 = -(lbound_904) - lbound_906 * stride_908;
      lbound_840 = (long) *its;
      ubound_841 = (long) *ite;
      size_843 = (1 - lbound_840) + ubound_841;
      size_843 = gf_max(size_843, 0);
      D_7737 = size_843 + -1;
      D_7738 = (ulong) (ulong) (size_843) * 32;
      D_7739 = (ulong) (size_843) * 4;
      offset_842 = -(lbound_840);
      lbound_925 = (long) *its;
      ubound_926 = (long) *ite;
      stride_929 = (1 - lbound_925) + ubound_926;
      stride_929 = gf_max(stride_929, 0);
      lbound_927 = (long) *kts;
      ubound_928 = (long) *kte;
      size_931 = ((1 - lbound_927) + ubound_928) * stride_929;
      size_931 = gf_max(size_931, 0);
      D_7734 = size_931 + -1;
      D_7735 = (ulong) (ulong) (size_931) * 32;
      D_7736 = (ulong) (size_931) * 4;
      offset_930 = -(lbound_925) - lbound_927 * stride_929;
      lbound_812 = (long) *its;
      ubound_813 = (long) *ite;
      stride_816 = (1 - lbound_812) + ubound_813;
      stride_816 = gf_max(stride_816, 0);
      lbound_814 = (long) *kts;
      ubound_815 = (long) *kte;
      size_818 = ((1 - lbound_814) + ubound_815) * stride_816;
      size_818 = gf_max(size_818, 0);
      D_7731 = size_818 + -1;
      D_7732 = (ulong) (ulong) (size_818) * 32;
      D_7733 = (ulong) (size_818) * 4;
      offset_817 = -(lbound_812) - lbound_814 * stride_816;
      lbound_932 = (long) *its;
      ubound_933 = (long) *ite;
      stride_936 = (1 - lbound_932) + ubound_933;
      stride_936 = gf_max(stride_936, 0);
      lbound_934 = (long) *kts;
      ubound_935 = (long) *kte;
      size_938 = ((1 - lbound_934) + ubound_935) * stride_936;
      size_938 = gf_max(size_938, 0);
      D_7728 = size_938 + -1;
      D_7729 = (ulong) (ulong) (size_938) * 32;
      D_7730 = (ulong) (size_938) * 4;
      offset_937 = -(lbound_932) - lbound_934 * stride_936;
      lbound_848 = (long) *its;
      ubound_849 = (long) *ite;
      size_851 = (1 - lbound_848) + ubound_849;
      size_851 = gf_max(size_851, 0);
      D_7725 = size_851 + -1;
      D_7726 = (ulong) (ulong) (size_851) * 32;
      D_7727 = (ulong) (size_851) * 4;
      offset_850 = -(lbound_848);
      lbound_967 = (long) *its;
      ubound_968 = (long) *ite;
      stride_971 = (1 - lbound_967) + ubound_968;
      stride_971 = gf_max(stride_971, 0);
      lbound_969 = (long) *kts;
      ubound_970 = (long) *kte;
      size_973 = ((1 - lbound_969) + ubound_970) * stride_971;
      size_973 = gf_max(size_973, 0);
      D_7722 = size_973 + -1;
      D_7723 = (ulong) (ulong) (size_973) * 32;
      D_7724 = (ulong) (size_973) * 4;
      offset_972 = -(lbound_967) - lbound_969 * stride_971;
      lbound_918 = (long) *its;
      ubound_919 = (long) *ite;
      stride_922 = (1 - lbound_918) + ubound_919;
      stride_922 = gf_max(stride_922, 0);
      lbound_920 = (long) *kts;
      ubound_921 = (long) *kte;
      size_924 = ((1 - lbound_920) + ubound_921) * stride_922;
      size_924 = gf_max(size_924, 0);
      D_7719 = size_924 + -1;
      D_7720 = (ulong) (ulong) (size_924) * 32;
      D_7721 = (ulong) (size_924) * 4;
      offset_923 = -(lbound_918) - lbound_920 * stride_922;
      lbound_844 = (long) *its;
      ubound_845 = (long) *ite;
      size_847 = (1 - lbound_844) + ubound_845;
      size_847 = gf_max(size_847, 0);
      D_7716 = size_847 + -1;
      D_7717 = (ulong) (ulong) (size_847) * 32;
      D_7718 = (ulong) (size_847) * 4;
      offset_846 = -(lbound_844);
      lbound_974 = (long) *its;
      ubound_975 = (long) *ite;
      size_977 = (1 - lbound_974) + ubound_975;
      size_977 = gf_max(size_977, 0);
      D_7713 = size_977 + -1;
      D_7714 = (ulong) (ulong) (size_977) * 32;
      D_7715 = (ulong) (size_977) * 4;
      offset_976 = -(lbound_974);
      lbound_978 = (long) *its;
      ubound_979 = (long) *ite;
      stride_982 = (1 - lbound_978) + ubound_979;
      stride_982 = gf_max(stride_982, 0);
      lbound_980 = (long) *kts;
      ubound_981 = (long) *kte;
      size_984 = ((1 - lbound_980) + ubound_981) * stride_982;
      size_984 = gf_max(size_984, 0);
      D_7710 = size_984 + -1;
      D_7711 = (ulong) (ulong) (size_984) * 32;
      D_7712 = (ulong) (size_984) * 4;
      offset_983 = -(lbound_978) - lbound_980 * stride_982;
      lbound_852 = (long) *its;
      ubound_853 = (long) *ite;
      size_855 = (1 - lbound_852) + ubound_853;
      size_855 = gf_max(size_855, 0);
      D_7707 = size_855 + -1;
      D_7708 = (ulong) (ulong) (size_855) * 32;
      D_7709 = (ulong) (size_855) * 4;
      offset_854 = -(lbound_852);
      lbound_946 = (long) *its;
      ubound_947 = (long) *ite;
      stride_950 = (1 - lbound_946) + ubound_947;
      stride_950 = gf_max(stride_950, 0);
      lbound_948 = (long) *kts;
      ubound_949 = (long) *kte;
      size_952 = ((1 - lbound_948) + ubound_949) * stride_950;
      size_952 = gf_max(size_952, 0);
      D_7704 = size_952 + -1;
      D_7705 = (ulong) (ulong) (size_952) * 32;
      D_7706 = (ulong) (size_952) * 4;
      offset_951 = -(lbound_946) - lbound_948 * stride_950;
      lbound_856 = (long) *its;
      ubound_857 = (long) *ite;
      stride_860 = (1 - lbound_856) + ubound_857;
      stride_860 = gf_max(stride_860, 0);
      lbound_858 = (long) *kts;
      ubound_859 = (long) *kte;
      size_862 = ((1 - lbound_858) + ubound_859) * stride_860;
      size_862 = gf_max(size_862, 0);
      D_7701 = size_862 + -1;
      D_7702 = (ulong) (ulong) (size_862) * 32;
      D_7703 = (ulong) (size_862) * 4;
      offset_861 = -(lbound_856) - lbound_858 * stride_860;
      lbound_833 = (long) *its;
      ubound_834 = (long) *ite;
      stride_837 = (1 - lbound_833) + ubound_834;
      stride_837 = gf_max(stride_837, 0);
      lbound_835 = (long) *kts;
      ubound_836 = (long) *kte;
      size_839 = ((1 - lbound_835) + ubound_836) * stride_837;
      size_839 = gf_max(size_839, 0);
      D_7698 = size_839 + -1;
      D_7699 = (ulong) (ulong) (size_839) * 32;
      D_7700 = (ulong) (size_839) * 4;
      offset_838 = -(lbound_833) - lbound_835 * stride_837;
      lbound_826 = (long) *its;
      ubound_827 = (long) *ite;
      stride_830 = (1 - lbound_826) + ubound_827;
      stride_830 = gf_max(stride_830, 0);
      lbound_828 = (long) *kts;
      ubound_829 = (long) *kte;
      size_832 = ((1 - lbound_828) + ubound_829) * stride_830;
      size_832 = gf_max(size_832, 0);
      D_7695 = size_832 + -1;
      D_7696 = (ulong) (ulong) (size_832) * 32;
      D_7697 = (ulong) (size_832) * 4;
      offset_831 = -(lbound_826) - lbound_828 * stride_830;
      lbound_953 = (long) *its;
      ubound_954 = (long) *ite;
      stride_957 = (1 - lbound_953) + ubound_954;
      stride_957 = gf_max(stride_957, 0);
      lbound_955 = (long) *kts;
      ubound_956 = (long) *kte;
      size_959 = ((1 - lbound_955) + ubound_956) * stride_957;
      size_959 = gf_max(size_959, 0);
      D_7692 = size_959 + -1;
      D_7693 = (ulong) (ulong) (size_959) * 32;
      D_7694 = (ulong) (size_959) * 4;
      offset_958 = -(lbound_953) - lbound_955 * stride_957;
      lbound_960 = (long) *its;
      ubound_961 = (long) *ite;
      stride_964 = (1 - lbound_960) + ubound_961;
      stride_964 = gf_max(stride_964, 0);
      lbound_962 = (long) *kts;
      ubound_963 = (long) *kte;
      size_966 = ((1 - lbound_962) + ubound_963) * stride_964;
      size_966 = gf_max(size_966, 0);
      D_7689 = size_966 + -1;
      D_7690 = (ulong) (ulong) (size_966) * 32;
      D_7691 = (ulong) (size_966) * 4;
      offset_965 = -(lbound_960) - lbound_962 * stride_964;
      lbound_897 = (long) *its;
      ubound_898 = (long) *ite;
      stride_901 = (1 - lbound_897) + ubound_898;
      stride_901 = gf_max(stride_901, 0);
      lbound_899 = (long) *kts;
      ubound_900 = (long) *kte;
      size_903 = ((1 - lbound_899) + ubound_900) * stride_901;
      size_903 = gf_max(size_903, 0);
      D_7686 = size_903 + -1;
      D_7687 = (ulong) (ulong) (size_903) * 32;
      D_7688 = (ulong) (size_903) * 4;
      offset_902 = -(lbound_897) - lbound_899 * stride_901;
      lbound_808 = (long) *its;
      ubound_809 = (long) *ite;
      size_811 = (1 - lbound_808) + ubound_809;
      size_811 = gf_max(size_811, 0);
      D_7683 = size_811 + -1;
      D_7684 = (ulong) (ulong) (size_811) * 32;
      D_7685 = (ulong) (size_811) * 4;
      offset_810 = -(lbound_808);
      lbound_819 = (long) *its;
      ubound_820 = (long) *ite;
      stride_823 = (1 - lbound_819) + ubound_820;
      stride_823 = gf_max(stride_823, 0);
      lbound_821 = (long) *kts;
      ubound_822 = (long) *kte;
      size_825 = ((1 - lbound_821) + ubound_822) * stride_823;
      size_825 = gf_max(size_825, 0);
      D_7679 = size_825 + -1;
      D_7680 = (ulong) (ulong) (size_825) * 32;
      D_7681 = (ulong) (size_825) * 4;
      D_7682 = (device void *) workspace.allocate (gf_max((ulong) (size_825 * 4), 1));
      clw_allh = (device GFReal *) D_7682;
      offset_824 = -(lbound_819) - lbound_821 * stride_823;
      lbound_863 = (long) *kts;
      ubound_864 = (long) *kte;
      size_866 = (1 - lbound_863) + ubound_864;
      size_866 = gf_max(size_866, 0);
      D_7675 = size_866 + -1;
      D_7676 = (ulong) (ulong) (size_866) * 32;
      D_7677 = (ulong) (size_866) * 4;
      D_7678 = (device void *) workspace.allocate (gf_max((ulong) (size_866 * 4), 1));
      prop_b = (device GFReal *) D_7678;
      offset_865 = -(lbound_863);
      lbound_886 = (long) *its;
      ubound_887 = (long) *ite;
      size_889 = (1 - lbound_886) + ubound_887;
      size_889 = gf_max(size_889, 0);
      D_7671 = size_889 + -1;
      D_7672 = (ulong) (ulong) (size_889) * 32;
      D_7673 = (ulong) (size_889) * 4;
      D_7674 = (device void *) workspace.allocate (gf_max((ulong) (size_889 * 4), 1));
      pwavh = (device GFReal *) D_7674;
      offset_888 = -(lbound_886);
      lbound_890 = (long) *its;
      ubound_891 = (long) *ite;
      stride_894 = (1 - lbound_890) + ubound_891;
      stride_894 = gf_max(stride_894, 0);
      lbound_892 = (long) *kts;
      ubound_893 = (long) *kte;
      size_896 = ((1 - lbound_892) + ubound_893) * stride_894;
      size_896 = gf_max(size_896, 0);
      D_7667 = size_896 + -1;
      D_7668 = (ulong) (ulong) (size_896) * 32;
      D_7669 = (ulong) (size_896) * 4;
      D_7670 = (device void *) workspace.allocate (gf_max((ulong) (size_896 * 4), 1));
      pwh = (device GFReal *) D_7670;
      offset_895 = -(lbound_890) - lbound_892 * stride_894;
      lbound_911 = (long) *its;
      ubound_912 = (long) *ite;
      stride_915 = (1 - lbound_911) + ubound_912;
      stride_915 = gf_max(stride_915, 0);
      lbound_913 = (long) *kts;
      ubound_914 = (long) *kte;
      size_917 = ((1 - lbound_913) + ubound_914) * stride_915;
      size_917 = gf_max(size_917, 0);
      D_7663 = size_917 + -1;
      D_7664 = (ulong) (ulong) (size_917) * 32;
      D_7665 = (ulong) (size_917) * 4;
      D_7666 = (device void *) workspace.allocate (gf_max((ulong) (size_917 * 4), 1));
      qch = (device GFReal *) D_7666;
      offset_916 = -(lbound_911) - lbound_913 * stride_915;
      lbound_939 = (long) *its;
      ubound_940 = (long) *ite;
      stride_943 = (1 - lbound_939) + ubound_940;
      stride_943 = gf_max(stride_943, 0);
      lbound_941 = (long) *kts;
      ubound_942 = (long) *kte;
      size_945 = ((1 - lbound_941) + ubound_942) * stride_943;
      size_945 = gf_max(size_945, 0);
      D_7659 = size_945 + -1;
      D_7660 = (ulong) (ulong) (size_945) * 32;
      D_7661 = (ulong) (size_945) * 4;
      D_7662 = (device void *) workspace.allocate (gf_max((ulong) (size_945 * 4), 1));
      qrcb = (device GFReal *) D_7662;
      offset_944 = -(lbound_939) - lbound_941 * stride_943;
      {
        long D_7563;
        long D_7564;
        long D_7565;

        D_7563 = offset_865;
        D_7564 = (long) *kts;
        D_7565 = (long) *kte;
        {
          long S_985;

          S_985 = D_7564;
          while (1)
            {
              if (S_985 > D_7565) break;
              prop_b[S_985 + D_7563] = 0.0f;
              S_985 = S_985 + 1;
            }

        }
      }
      iall = 0;
      c0 = 2.00000009499490261077880859375e-3f;
      g = 9.81000041961669921875e+0f;
      bdsp = 3.65999996662139892578125e-1f;
      if (gf_compare_string (_name, name, 7, "shallow") == 0)
        {
          c0 = 0.0f;
        }

      {
        int D_7569;
        int D_7570;

        D_7569 = *its;
        D_7570 = *itf;
        i = D_7569;
        while (1)
          {
            {
              bool D_7573;

              D_7573 = i > D_7570;
              if (D_7573) break;
              pwav[(long) i + offset_884] = 0.0f;
              pwavh[(long) i + offset_888] = 0.0f;
              psum[(long) i + offset_869] = 0.0f;
              psumh[(long) i + offset_873] = 0.0f;

              i = i + 1;
            }
          }

      }
      {
        int D_7574;
        int D_7575;

        D_7574 = *kts;
        D_7575 = *ktf;
        k = D_7574;
        while (1)
          {
            {
              bool D_7578;

              D_7578 = k > D_7575;
              if (D_7578) break;
              {
                int D_7579;
                int D_7580;

                D_7579 = *its;
                D_7580 = *itf;
                i = D_7579;
                while (1)
                  {
                    {
                      bool D_7583;

                      D_7583 = i > D_7580;
                      if (D_7583) break;
                      pw[((long) k * stride_879 + offset_880) + (long) i] = 0.0f;
                      pwh[((long) k * stride_894 + offset_895) + (long) i] = 0.0f;
                      qc[((long) k * stride_908 + offset_909) + (long) i] = 0.0f;
                      if (ierr[(long) i + offset_842] == 0)
                        {
                          qc[((long) k * stride_908 + offset_909) + (long) i] = qes_cup[((long) k * stride_929 + offset_930) + (long) i];
                        }

                      if (ierr[(long) i + offset_842] == 0)
                        {
                          qch[((long) k * stride_915 + offset_916) + (long) i] = qes_cup[((long) k * stride_929 + offset_930) + (long) i];
                        }

                      clw_all[((long) k * stride_816 + offset_817) + (long) i] = 0.0f;
                      clw_allh[((long) k * stride_823 + offset_824) + (long) i] = 0.0f;
                      qrc[((long) k * stride_936 + offset_937) + (long) i] = 0.0f;
                      qrcb[((long) k * stride_943 + offset_944) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      if (*use_excess <= 1)
        {
          {
            int D_7587;
            int D_7588;

            D_7587 = *its;
            D_7588 = *itf;
            i = D_7587;
            while (1)
              {
                {
                  bool D_7591;

                  D_7591 = i > D_7588;
                  if (D_7591) break;
                  if ((GFReal) ierr[(long) i + offset_842] == 0.0f)
                    {
                      {
                        int D_7593;

                        D_7593 = kbcon[(long) i + offset_850] + -1;
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_7596;

                              D_7596 = k > D_7593;
                              if (D_7596) break;
                              dz = z_cup[((long) k * stride_971 + offset_972) + (long) i] - z_cup[((long) (k + -1) * stride_971 + offset_972) + (long) i];
                              qc[((long) k * stride_908 + offset_909) + (long) i] = qe_cup[((long) k22[(long) i + offset_846] * stride_922 + offset_923) + (long) i] + (GFReal) *use_excess * zqexec[(long) i + offset_976];
                              qch[((long) k * stride_915 + offset_916) + (long) i] = qe_cup[((long) k22[(long) i + offset_846] * stride_922 + offset_923) + (long) i] + (GFReal) *use_excess * zqexec[(long) i + offset_976];
                              if (qc[((long) k * stride_908 + offset_909) + (long) i] > qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i])
                                {
                                  pw[((long) k * stride_879 + offset_880) + (long) i] = zu[((long) k * stride_982 + offset_983) + (long) i] * ((qc[((long) k * stride_908 + offset_909) + (long) i] - qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i]));
                                  qc[((long) k * stride_908 + offset_909) + (long) i] = qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i];
                                  qch[((long) k * stride_915 + offset_916) + (long) i] = qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i];
                                  pwav[(long) i + offset_884] = pwav[(long) i + offset_884] + pw[((long) k * stride_879 + offset_880) + (long) i];
                                  psum[(long) i + offset_869] = psum[(long) i + offset_869] + pw[((long) k * stride_879 + offset_880) + (long) i] * dz;
                                }


                              k = k + 1;
                            }
                          }

                      }
                    }


                  i = i + 1;
                }
              }

          }
        }
      else
        {
          if (*use_excess == 2)
            {
              {
                int D_7599;
                int D_7600;

                D_7599 = *its;
                D_7600 = *itf;
                i = D_7599;
                while (1)
                  {
                    {
                      bool D_7603;

                      D_7603 = i > D_7600;
                      if (D_7603) break;
                      if ((GFReal) ierr[(long) i + offset_842] == 0.0f)
                        {
                          {
                            int M_986;
                            int D_7606;

                            M_986 = 1;
                            D_7606 = k22[(long) i + offset_846] + -1;
                            M_986 = gf_max(D_7606, M_986);
                            k1 = M_986;
                          }
                          k2 = k22[(long) i + offset_846] + 1;
                          {
                            int D_7607;

                            D_7607 = kbcon[(long) i + offset_850] + -1;
                            k = 2;
                            while (1)
                              {
                                {
                                  bool D_7610;

                                  D_7610 = k > D_7607;
                                  if (D_7610) break;
                                  dz = z_cup[((long) k * stride_971 + offset_972) + (long) i] - z_cup[((long) (k + -1) * stride_971 + offset_972) + (long) i];
                                  {
                                    GFReal val_987;
                                    long D_7612;
                                    long D_7613;
                                    long D_7614;
                                    long D_7615;

                                    val_987 = 0.0f;
                                    D_7612 = offset_923;
                                    D_7613 = (long) k1;
                                    D_7614 = (long) k2;
                                    D_7615 = (long) i;
                                    {
                                      long S_988;
                                      long D_7617;
                                      long D_7618;

                                      D_7617 = stride_922;
                                      D_7618 = D_7612 + D_7615;
                                      S_988 = D_7613;
                                      while (1)
                                        {
                                          if (S_988 > D_7614) break;
                                          val_987 = qe_cup[S_988 * D_7617 + D_7618] + val_987;
                                          S_988 = S_988 + 1;
                                        }

                                    }
                                    qc[((long) k * stride_908 + offset_909) + (long) i] = val_987 / (GFReal) ((k2 - k1) + 1) + zqexec[(long) i + offset_976];
                                  }
                                  {
                                    GFReal val_989;
                                    long D_7621;
                                    long D_7622;
                                    long D_7623;
                                    long D_7624;

                                    val_989 = 0.0f;
                                    D_7621 = offset_923;
                                    D_7622 = (long) k1;
                                    D_7623 = (long) k2;
                                    D_7624 = (long) i;
                                    {
                                      long S_990;
                                      long D_7626;
                                      long D_7627;

                                      D_7626 = stride_922;
                                      D_7627 = D_7621 + D_7624;
                                      S_990 = D_7622;
                                      while (1)
                                        {
                                          if (S_990 > D_7623) break;
                                          val_989 = qe_cup[S_990 * D_7626 + D_7627] + val_989;
                                          S_990 = S_990 + 1;
                                        }

                                    }
                                    qch[((long) k * stride_915 + offset_916) + (long) i] = val_989 / (GFReal) ((k2 - k1) + 1) + zqexec[(long) i + offset_976];
                                  }
                                  if (qc[((long) k * stride_908 + offset_909) + (long) i] > qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i])
                                    {
                                      pw[((long) k * stride_879 + offset_880) + (long) i] = zu[((long) k * stride_982 + offset_983) + (long) i] * ((qc[((long) k * stride_908 + offset_909) + (long) i] - qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i]));
                                      qc[((long) k * stride_908 + offset_909) + (long) i] = qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i];
                                      qch[((long) k * stride_915 + offset_916) + (long) i] = qes_cup[((long) (kbcon[(long) i + offset_850] + -1) * stride_929 + offset_930) + (long) i];
                                      pwav[(long) i + offset_884] = pwav[(long) i + offset_884] + pw[((long) k * stride_879 + offset_880) + (long) i];
                                      psum[(long) i + offset_869] = psum[(long) i + offset_869] + pw[((long) k * stride_879 + offset_880) + (long) i] * dz;
                                    }


                                  k = k + 1;
                                }
                              }

                          }
                        }


                      i = i + 1;
                    }
                  }

              }
            }

        }

      {
        int D_7630;
        int D_7631;

        D_7630 = *kts + 1;
        D_7631 = *ktf;
        k = D_7630;
        while (1)
          {
            {
              bool D_7634;

              D_7634 = k > D_7631;
              if (D_7634) break;
              {
                int D_7635;
                int D_7636;

                D_7635 = *its;
                D_7636 = *itf;
                i = D_7635;
                while (1)
                  {
                    {
                      bool D_7639;

                      D_7639 = i > D_7636;
                      if (D_7639) break;
                      do {
if (ierr[(long) i + offset_842] != 0) break;

                      if (kbcon[(long) i + offset_850] > k) break;

                      if (ktop[(long) i + offset_854] < k) break;

                      rhoc = ((rho[((long) k * stride_950 + offset_951) + (long) i] + rho[((long) (k + -1) * stride_950 + offset_951) + (long) i])) * 5.0e-1f;
                      dz = z_cup[((long) k * stride_971 + offset_972) + (long) i] - z_cup[((long) (k + -1) * stride_971 + offset_972) + (long) i];
                      dp = p_cup[((long) k * stride_860 + offset_861) + (long) i] - p_cup[((long) (k + -1) * stride_860 + offset_861) + (long) i];
                      qrch = qes_cup[((long) k * stride_929 + offset_930) + (long) i] + (((1.0e+0f / *xl)) * ((gamma_cup[((long) k * stride_837 + offset_838) + (long) i] / ((gamma_cup[((long) k * stride_837 + offset_838) + (long) i] + 1.0e+0f))))) * dby[((long) k * stride_830 + offset_831) + (long) i];
                      qc[((long) k * stride_908 + offset_909) + (long) i] = (((qc[((long) (k + -1) * stride_908 + offset_909) + (long) i] * zu[((long) (k + -1) * stride_982 + offset_983) + (long) i] - (up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f) * qc[((long) (k + -1) * stride_908 + offset_909) + (long) i]) + up_massentr[((long) (k + -1) * stride_964 + offset_965) + (long) i] * q[((long) (k + -1) * stride_901 + offset_902) + (long) i])) / (((zu[((long) (k + -1) * stride_982 + offset_983) + (long) i] - up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f) + up_massentr[((long) (k + -1) * stride_964 + offset_965) + (long) i]));
                      qch[((long) k * stride_915 + offset_916) + (long) i] = (((qch[((long) (k + -1) * stride_915 + offset_916) + (long) i] * zu[((long) (k + -1) * stride_982 + offset_983) + (long) i] - (up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f) * qch[((long) (k + -1) * stride_915 + offset_916) + (long) i]) + up_massentr[((long) (k + -1) * stride_964 + offset_965) + (long) i] * q[((long) (k + -1) * stride_901 + offset_902) + (long) i])) / (((zu[((long) (k + -1) * stride_982 + offset_983) + (long) i] - up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f) + up_massentr[((long) (k + -1) * stride_964 + offset_965) + (long) i]));
                      if (qc[((long) k * stride_908 + offset_909) + (long) i] <= qrch)
                        {
                          qc[((long) k * stride_908 + offset_909) + (long) i] = qrch;
                        }

                      if (qch[((long) k * stride_915 + offset_916) + (long) i] <= qrch)
                        {
                          qch[((long) k * stride_915 + offset_916) + (long) i] = qrch;
                        }

                      clw_all[((long) k * stride_816 + offset_817) + (long) i] = qc[((long) k * stride_908 + offset_909) + (long) i] - qrch;
                      qrc[((long) k * stride_936 + offset_937) + (long) i] = ((qc[((long) k * stride_908 + offset_909) + (long) i] - qrch));
                      clw_allh[((long) k * stride_823 + offset_824) + (long) i] = qch[((long) k * stride_915 + offset_916) + (long) i] - qrch;
                      qrcb[((long) k * stride_943 + offset_944) + (long) i] = ((qch[((long) k * stride_915 + offset_916) + (long) i] - qrch));
                      if (*autoconv == 2)
                        {
                          q1 = (rhoc * 1.0e+3f) * qrcb[((long) k * stride_943 + offset_944) + (long) i];
                          berryc0 = (q1 * q1) / (((((*ccnclean * 3.6600001156330108642578125e-2f) / ((q1 * bdsp)) + 5.0e+0f)) * 6.0e+1f));
                          qrcb_h = ((((qch[((long) k * stride_915 + offset_916) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i] - qrcb[((long) (k + -1) * stride_943 + offset_944) + (long) i] * ((up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f)))) / (((zu[((long) k * stride_982 + offset_983) + (long) i] + up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f) + (c0 * dz) * zu[((long) k * stride_982 + offset_983) + (long) i]));
                          prop_b[(long) k + offset_865] = ((c0 * qrcb_h) * zu[((long) k * stride_982 + offset_983) + (long) i]) / ((berryc0 * 1.000000047497451305389404296875e-3f));
                          pwh[((long) k * stride_894 + offset_895) + (long) i] = ((berryc0 * 1.000000047497451305389404296875e-3f) * dz) * prop_b[(long) k + offset_865];
                          berryc = qrcb[((long) k * stride_943 + offset_944) + (long) i];
                          qrcb[((long) k * stride_943 + offset_944) + (long) i] = (((((qch[((long) k * stride_915 + offset_916) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i] - pwh[((long) k * stride_894 + offset_895) + (long) i]) - qrcb[((long) (k + -1) * stride_943 + offset_944) + (long) i] * ((up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f)))) / ((zu[((long) k * stride_982 + offset_983) + (long) i] + up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f));
                          if (qrcb[((long) k * stride_943 + offset_944) + (long) i] < 0.0f)
                            {
                              berryc0 = (((((qrcb[((long) (k + -1) * stride_943 + offset_944) + (long) i] * ((up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f)) - ((qch[((long) k * stride_915 + offset_916) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i])) / zu[((long) k * stride_982 + offset_983) + (long) i]) * 1.000000047497451305389404296875e-3f) * dz) * prop_b[(long) k + offset_865];
                              pwh[((long) k * stride_894 + offset_895) + (long) i] = (((zu[((long) k * stride_982 + offset_983) + (long) i] * 1.000000047497451305389404296875e-3f) * berryc0) * dz) * prop_b[(long) k + offset_865];
                              qrcb[((long) k * stride_943 + offset_944) + (long) i] = 0.0f;
                            }

                          qch[((long) k * stride_915 + offset_916) + (long) i] = qrcb[((long) k * stride_943 + offset_944) + (long) i] + qrch;
                          pwavh[(long) i + offset_888] = pwavh[(long) i + offset_888] + pwh[((long) k * stride_894 + offset_895) + (long) i];
                          psumh[(long) i + offset_873] = psumh[(long) i + offset_873] + (clw_allh[((long) k * stride_823 + offset_824) + (long) i] * zu[((long) k * stride_982 + offset_983) + (long) i]) * dz;
                          q1 = (rhoc * 1.0e+3f) * qrc[((long) k * stride_936 + offset_937) + (long) i];
                          berryc0 = (q1 * q1) / (((((ccn[(long) i + offset_810] * 3.6600001156330108642578125e-2f) / ((q1 * bdsp)) + 5.0e+0f)) * 6.0e+1f));
                          berryc0 = ((berryc0 * 1.000000047497451305389404296875e-3f) * dz) * prop_b[(long) k + offset_865];
                          berryc = qrc[((long) k * stride_936 + offset_937) + (long) i];
                          qrc[((long) k * stride_936 + offset_937) + (long) i] = (((((qc[((long) k * stride_908 + offset_909) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i] - zu[((long) k * stride_982 + offset_983) + (long) i] * berryc0) - qrc[((long) (k + -1) * stride_936 + offset_937) + (long) i] * ((up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f)))) / ((zu[((long) k * stride_982 + offset_983) + (long) i] + up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f));
                          if (qrc[((long) k * stride_936 + offset_937) + (long) i] < 0.0f)
                            {
                              berryc0 = ((((qc[((long) k * stride_908 + offset_909) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i] - qrc[((long) (k + -1) * stride_936 + offset_937) + (long) i] * ((up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f)))) / zu[((long) k * stride_982 + offset_983) + (long) i];
                              qrc[((long) k * stride_936 + offset_937) + (long) i] = 0.0f;
                            }

                          pw[((long) k * stride_879 + offset_880) + (long) i] = zu[((long) k * stride_982 + offset_983) + (long) i] * berryc0;
                          qc[((long) k * stride_908 + offset_909) + (long) i] = qrc[((long) k * stride_936 + offset_937) + (long) i] + qrch;
                        }
                      else
                        {
                          qrc[((long) k * stride_936 + offset_937) + (long) i] = ((((qc[((long) k * stride_908 + offset_909) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i] - qrc[((long) (k + -1) * stride_936 + offset_937) + (long) i] * ((up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f)))) / (((zu[((long) k * stride_982 + offset_983) + (long) i] + up_massdetr[((long) (k + -1) * stride_957 + offset_958) + (long) i] * 5.0e-1f) + (c0 * dz) * zu[((long) k * stride_982 + offset_983) + (long) i]));
                          pw[((long) k * stride_879 + offset_880) + (long) i] = ((c0 * dz) * qrc[((long) k * stride_936 + offset_937) + (long) i]) * zu[((long) k * stride_982 + offset_983) + (long) i];
                          if (qrc[((long) k * stride_936 + offset_937) + (long) i] < 0.0f)
                            {
                              qrc[((long) k * stride_936 + offset_937) + (long) i] = 0.0f;
                              pw[((long) k * stride_879 + offset_880) + (long) i] = 0.0f;
                            }

                          if (iall == 1)
                            {
                              qrc[((long) k * stride_936 + offset_937) + (long) i] = 0.0f;
                              pw[((long) k * stride_879 + offset_880) + (long) i] = ((qc[((long) k * stride_908 + offset_909) + (long) i] - qrch)) * zu[((long) k * stride_982 + offset_983) + (long) i];
                              if (pw[((long) k * stride_879 + offset_880) + (long) i] < 0.0f)
                                {
                                  pw[((long) k * stride_879 + offset_880) + (long) i] = 0.0f;
                                }

                            }

                          qc[((long) k * stride_908 + offset_909) + (long) i] = qrc[((long) k * stride_936 + offset_937) + (long) i] + qrch;
                        }

                      pwav[(long) i + offset_884] = pwav[(long) i + offset_884] + pw[((long) k * stride_879 + offset_880) + (long) i];
                      psum[(long) i + offset_869] = psum[(long) i + offset_869] + (clw_all[((long) k * stride_816 + offset_817) + (long) i] * zu[((long) k * stride_982 + offset_983) + (long) i]) * dz;
                      } while(false);

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      prop_ave = 0.0f;
      iprop = 0;
      {
        int D_7652;
        int D_7653;

        D_7652 = *kts;
        D_7653 = *kte;
        k = D_7652;
        while (1)
          {
            {
              bool D_7656;

              D_7656 = k > D_7653;
              if (D_7656) break;
              prop_ave = prop_b[(long) k + offset_865] + prop_ave;
              if (prop_b[(long) k + offset_865] > 0.0f)
                {
                  iprop = iprop + 1;
                }


              k = k + 1;
            }
          }

      }
      {
        int M_991;

        M_991 = iprop;
        M_991 = gf_max(M_991, 1);
        iprop = M_991;
      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55> static inline void cup_output_ens_3d(thread GFWorkspace &workspace, P0 xf_ens, P1 ierr, P2 dellat, P3 dellaq, P4 dellaqc, P5 subt_ens, P6 subq_ens, P7 subt, P8 subq, P9 outtem, P10 outq, P11 outqc, P12 zu, P13 sub_mas, P14 pre, P15 pw, P16 xmb, P17 ktop, P18 j, P19 name, P20 nx, P21 nx2, P22 iens, P23 ierr2, P24 ierr3, P25 pr_ens, P26 maxens3, P27 ensdim, P28 sig, P29 apr_gr, P30 apr_w, P31 apr_mc, P32 apr_st, P33 apr_as, P34 apr_capma, P35 apr_capme, P36 apr_capmi, P37 closure_n, P38 xland1, P39 weight_gr, P40 weight_w, P41 weight_mc, P42 weight_st, P43 weight_as, P44 training, P45 ipr, P46 jpr, P47 itf, P48 jtf, P49 ktf, P50 its, P51 ite, P52 jts, P53 jte, P54 kts, P55 kte, long _name) {
  GFFrame frame(workspace);

  device void * D_8097;

  device void * D_8101;

  device void * D_8105;

  device void * D_8109;

  device void * D_8113;
  long lbound_992;
  long ubound_993;
  long lbound_994;
  long ubound_995;
  long stride_996;
  long offset_997;
  long size_998;
  long lbound_999;
  long ubound_1000;
  long lbound_1001;
  long ubound_1002;
  long stride_1003;
  long offset_1004;
  long size_1005;
  long lbound_1006;
  long ubound_1007;
  long lbound_1008;
  long ubound_1009;
  long stride_1010;
  long offset_1011;
  long size_1012;
  long lbound_1013;
  long ubound_1014;
  long lbound_1015;
  long ubound_1016;
  long stride_1017;
  long offset_1018;
  long size_1019;
  long lbound_1020;
  long ubound_1021;
  long lbound_1022;
  long ubound_1023;
  long stride_1024;
  long offset_1025;
  long size_1026;
  GFReal clos_wei;
  long lbound_1027;
  long ubound_1028;
  long offset_1029;
  long size_1030;
  GFReal ddtes;
  long lbound_1031;
  long ubound_1032;
  long lbound_1033;
  long ubound_1034;
  long stride_1035;
  long ubound_1036;
  long stride_1037;
  long offset_1038;
  long size_1039;
  long lbound_1040;
  long ubound_1041;
  long lbound_1042;
  long ubound_1043;
  long stride_1044;
  long ubound_1045;
  long stride_1046;
  long offset_1047;
  long size_1048;
  long lbound_1049;
  long ubound_1050;
  long lbound_1051;
  long ubound_1052;
  long stride_1053;
  long ubound_1054;
  long stride_1055;
  long offset_1056;
  long size_1057;
  GFReal dtpw;
  GFReal dtq;
  GFReal dtqc;
  GFReal dtqs;
  GFReal dtt;
  GFReal dtts;
  int i;
  long lbound_1058;
  long ubound_1059;
  long offset_1060;
  long size_1061;
  int k;
  long lbound_1062;
  long ubound_1063;
  long offset_1064;
  long size_1065;
  int n;
  long lbound_1066;
  long ubound_1067;
  long lbound_1068;
  long ubound_1069;
  long stride_1070;
  long offset_1071;
  long size_1072;
  long lbound_1073;
  long ubound_1074;
  long lbound_1075;
  long ubound_1076;
  long stride_1077;
  long offset_1078;
  long size_1079;
  long lbound_1080;
  long ubound_1081;
  long lbound_1082;
  long ubound_1083;
  long stride_1084;
  long offset_1085;
  long size_1086;
  long lbound_1087;
  long ubound_1088;
  long lbound_1089;
  long ubound_1090;
  long stride_1091;
  long ubound_1092;
  long stride_1093;
  long offset_1094;
  long size_1095;
  long lbound_1096;
  long ubound_1097;
  long offset_1098;
  long size_1099;
  long lbound_1100;
  long ubound_1101;
  long lbound_1102;
  long ubound_1103;
  long stride_1104;
  long ubound_1105;
  long stride_1106;
  long offset_1107;
  long size_1108;
  long lbound_1109;
  long ubound_1110;
  long offset_1111;
  long size_1112;
  long lbound_1113;
  long ubound_1114;
  long lbound_1115;
  long ubound_1116;
  long stride_1117;
  long offset_1118;
  long size_1119;
  long lbound_1120;
  long ubound_1121;
  long lbound_1122;
  long ubound_1123;
  long stride_1124;
  long offset_1125;
  long size_1126;
  long lbound_1127;
  long ubound_1128;
  long lbound_1129;
  long ubound_1130;
  long stride_1131;
  long ubound_1132;
  long stride_1133;
  long offset_1134;
  long size_1135;
  long lbound_1136;
  long ubound_1137;
  long lbound_1138;
  long ubound_1139;
  long stride_1140;
  long offset_1141;
  long size_1142;
  long lbound_1143;
  long ubound_1144;
  long lbound_1145;
  long ubound_1146;
  long stride_1147;
  long ubound_1148;
  long stride_1149;
  long offset_1150;
  long size_1151;
  GFReal weight[5];
  GFReal wm[5];
  long lbound_1152;
  long ubound_1153;
  long lbound_1154;
  long ubound_1155;
  long stride_1156;
  long ubound_1157;
  long stride_1158;
  long offset_1159;
  long size_1160;
  long lbound_1161;
  long ubound_1162;
  long offset_1163;
  long size_1164;

  device GFReal * xfac1;
  long lbound_1165;
  long ubound_1166;
  long offset_1167;
  long size_1168;

  device GFReal * xfac2;
  long lbound_1169;
  long ubound_1170;
  long offset_1171;
  long size_1172;
  long lbound_1173;
  long ubound_1174;
  long offset_1175;
  long size_1176;

  device GFReal * xmb_ave;
  long lbound_1177;
  long ubound_1178;
  long stride_1179;
  long offset_1180;
  long size_1181;

  device GFReal * xmb_w;
  long lbound_1182;
  long ubound_1183;
  long offset_1184;
  long size_1185;

  device GFReal * xmbweight;
  long lbound_1186;
  long ubound_1187;
  long lbound_1188;
  long ubound_1189;
  long stride_1190;
  long offset_1191;
  long size_1192;
  long D_8094;
  ulong D_8095;
  ulong D_8096;
  long D_8098;
  ulong D_8099;
  ulong D_8100;
  long D_8102;
  ulong D_8103;
  ulong D_8104;
  long D_8106;
  ulong D_8107;
  ulong D_8108;
  long D_8110;
  ulong D_8111;
  ulong D_8112;
  long D_8114;
  ulong D_8115;
  ulong D_8116;
  long D_8117;
  ulong D_8118;
  ulong D_8119;
  long D_8120;
  ulong D_8121;
  ulong D_8122;
  long D_8123;
  ulong D_8124;
  ulong D_8125;
  long D_8126;
  ulong D_8127;
  ulong D_8128;
  long D_8129;
  ulong D_8130;
  ulong D_8131;
  long D_8132;
  ulong D_8133;
  ulong D_8134;
  long D_8135;
  ulong D_8136;
  ulong D_8137;
  long D_8138;
  ulong D_8139;
  ulong D_8140;
  long D_8141;
  ulong D_8142;
  ulong D_8143;
  long D_8144;
  ulong D_8145;
  ulong D_8146;
  long D_8147;
  ulong D_8148;
  ulong D_8149;
  long D_8150;
  ulong D_8151;
  ulong D_8152;
  long D_8153;
  ulong D_8154;
  ulong D_8155;
  long D_8156;
  ulong D_8157;
  ulong D_8158;
  long D_8159;
  ulong D_8160;
  ulong D_8161;
  long D_8162;
  ulong D_8163;
  ulong D_8164;
  long D_8165;
  ulong D_8166;
  ulong D_8167;
  long D_8168;
  ulong D_8169;
  ulong D_8170;
  long D_8171;
  ulong D_8172;
  ulong D_8173;
  long D_8174;
  ulong D_8175;
  ulong D_8176;
  long D_8177;
  ulong D_8178;
  ulong D_8179;
  long D_8180;
  ulong D_8181;
  ulong D_8182;
  long D_8183;
  ulong D_8184;
  ulong D_8185;
  long D_8186;
  ulong D_8187;
  ulong D_8188;
  long D_8189;
  ulong D_8190;
  ulong D_8191;
  ulong D_8192;
  ulong D_8193;

    {
      D_8192 = (ulong) (ulong) (_name) * 8;
      D_8193 = (ulong) (_name);
      lbound_1080 = (long) *its;
      ubound_1081 = (long) *ite;
      stride_1084 = (1 - lbound_1080) + ubound_1081;
      stride_1084 = gf_max(stride_1084, 0);
      lbound_1082 = (long) *kts;
      ubound_1083 = (long) *kte;
      size_1086 = ((1 - lbound_1082) + ubound_1083) * stride_1084;
      size_1086 = gf_max(size_1086, 0);
      D_8189 = size_1086 + -1;
      D_8190 = (ulong) (ulong) (size_1086) * 32;
      D_8191 = (ulong) (size_1086) * 4;
      offset_1085 = -(lbound_1080) - lbound_1082 * stride_1084;
      lbound_1066 = (long) *its;
      ubound_1067 = (long) *ite;
      stride_1070 = (1 - lbound_1066) + ubound_1067;
      stride_1070 = gf_max(stride_1070, 0);
      lbound_1068 = (long) *kts;
      ubound_1069 = (long) *kte;
      size_1072 = ((1 - lbound_1068) + ubound_1069) * stride_1070;
      size_1072 = gf_max(size_1072, 0);
      D_8186 = size_1072 + -1;
      D_8187 = (ulong) (ulong) (size_1072) * 32;
      D_8188 = (ulong) (size_1072) * 4;
      offset_1071 = -(lbound_1066) - lbound_1068 * stride_1070;
      lbound_1073 = (long) *its;
      ubound_1074 = (long) *ite;
      stride_1077 = (1 - lbound_1073) + ubound_1074;
      stride_1077 = gf_max(stride_1077, 0);
      lbound_1075 = (long) *kts;
      ubound_1076 = (long) *kte;
      size_1079 = ((1 - lbound_1075) + ubound_1076) * stride_1077;
      size_1079 = gf_max(size_1079, 0);
      D_8183 = size_1079 + -1;
      D_8184 = (ulong) (ulong) (size_1079) * 32;
      D_8185 = (ulong) (size_1079) * 4;
      offset_1078 = -(lbound_1073) - lbound_1075 * stride_1077;
      lbound_1136 = (long) *its;
      ubound_1137 = (long) *ite;
      stride_1140 = (1 - lbound_1136) + ubound_1137;
      stride_1140 = gf_max(stride_1140, 0);
      lbound_1138 = (long) *kts;
      ubound_1139 = (long) *kte;
      size_1142 = ((1 - lbound_1138) + ubound_1139) * stride_1140;
      size_1142 = gf_max(size_1142, 0);
      D_8180 = size_1142 + -1;
      D_8181 = (ulong) (ulong) (size_1142) * 32;
      D_8182 = (ulong) (size_1142) * 4;
      offset_1141 = -(lbound_1136) - lbound_1138 * stride_1140;
      lbound_1120 = (long) *its;
      ubound_1121 = (long) *ite;
      stride_1124 = (1 - lbound_1120) + ubound_1121;
      stride_1124 = gf_max(stride_1124, 0);
      lbound_1122 = (long) *kts;
      ubound_1123 = (long) *kte;
      size_1126 = ((1 - lbound_1122) + ubound_1123) * stride_1124;
      size_1126 = gf_max(size_1126, 0);
      D_8177 = size_1126 + -1;
      D_8178 = (ulong) (ulong) (size_1126) * 32;
      D_8179 = (ulong) (size_1126) * 4;
      offset_1125 = -(lbound_1120) - lbound_1122 * stride_1124;
      lbound_1113 = (long) *its;
      ubound_1114 = (long) *ite;
      stride_1117 = (1 - lbound_1113) + ubound_1114;
      stride_1117 = gf_max(stride_1117, 0);
      lbound_1115 = (long) *kts;
      ubound_1116 = (long) *kte;
      size_1119 = ((1 - lbound_1115) + ubound_1116) * stride_1117;
      size_1119 = gf_max(size_1119, 0);
      D_8174 = size_1119 + -1;
      D_8175 = (ulong) (ulong) (size_1119) * 32;
      D_8176 = (ulong) (size_1119) * 4;
      offset_1118 = -(lbound_1113) - lbound_1115 * stride_1117;
      lbound_1096 = (long) *its;
      ubound_1097 = (long) *ite;
      size_1099 = (1 - lbound_1096) + ubound_1097;
      size_1099 = gf_max(size_1099, 0);
      D_8171 = size_1099 + -1;
      D_8172 = (ulong) (ulong) (size_1099) * 32;
      D_8173 = (ulong) (size_1099) * 4;
      offset_1098 = -(lbound_1096);
      lbound_1169 = (long) *its;
      ubound_1170 = (long) *ite;
      size_1172 = (1 - lbound_1169) + ubound_1170;
      size_1172 = gf_max(size_1172, 0);
      D_8168 = size_1172 + -1;
      D_8169 = (ulong) (ulong) (size_1172) * 32;
      D_8170 = (ulong) (size_1172) * 4;
      offset_1171 = -(lbound_1169);
      lbound_1058 = (long) *its;
      ubound_1059 = (long) *ite;
      size_1061 = (1 - lbound_1058) + ubound_1059;
      size_1061 = gf_max(size_1061, 0);
      D_8165 = size_1061 + -1;
      D_8166 = (ulong) (ulong) (size_1061) * 32;
      D_8167 = (ulong) (size_1061) * 4;
      offset_1060 = -(lbound_1058);
      lbound_1087 = (long) *its;
      ubound_1088 = (long) *ite;
      stride_1091 = (1 - lbound_1087) + ubound_1088;
      stride_1091 = gf_max(stride_1091, 0);
      lbound_1089 = (long) *jts;
      ubound_1090 = (long) *jte;
      stride_1093 = ((1 - lbound_1089) + ubound_1090) * stride_1091;
      stride_1093 = gf_max(stride_1093, 0);
      ubound_1092 = (long) *ensdim;
      size_1095 = stride_1093 * (ubound_1092);
      size_1095 = gf_max(size_1095, 0);
      D_8162 = size_1095 + -1;
      D_8163 = (ulong) (ulong) (size_1095) * 32;
      D_8164 = (ulong) (size_1095) * 4;
      offset_1094 = (-(lbound_1087) - lbound_1089 * stride_1091) - (stride_1093);
      lbound_1152 = (long) *its;
      ubound_1153 = (long) *ite;
      stride_1156 = (1 - lbound_1152) + ubound_1153;
      stride_1156 = gf_max(stride_1156, 0);
      lbound_1154 = (long) *jts;
      ubound_1155 = (long) *jte;
      stride_1158 = ((1 - lbound_1154) + ubound_1155) * stride_1156;
      stride_1158 = gf_max(stride_1158, 0);
      ubound_1157 = (long) *ensdim;
      size_1160 = stride_1158 * (ubound_1157);
      size_1160 = gf_max(size_1160, 0);
      D_8159 = size_1160 + -1;
      D_8160 = (ulong) (ulong) (size_1160) * 32;
      D_8161 = (ulong) (size_1160) * 4;
      offset_1159 = (-(lbound_1152) - lbound_1154 * stride_1156) - (stride_1158);
      lbound_1109 = (long) *its;
      ubound_1110 = (long) *ite;
      size_1112 = (1 - lbound_1109) + ubound_1110;
      size_1112 = gf_max(size_1112, 0);
      D_8156 = size_1112 + -1;
      D_8157 = (ulong) (ulong) (size_1112) * 32;
      D_8158 = (ulong) (size_1112) * 4;
      offset_1111 = -(lbound_1109);
      lbound_1027 = (long) *its;
      ubound_1028 = (long) *ite;
      size_1030 = (1 - lbound_1027) + ubound_1028;
      size_1030 = gf_max(size_1030, 0);
      D_8153 = size_1030 + -1;
      D_8154 = (ulong) (ulong) (size_1030) * 32;
      D_8155 = (ulong) (size_1030) * 4;
      offset_1029 = -(lbound_1027);
      lbound_1062 = (long) *its;
      ubound_1063 = (long) *ite;
      size_1065 = (1 - lbound_1062) + ubound_1063;
      size_1065 = gf_max(size_1065, 0);
      D_8150 = size_1065 + -1;
      D_8151 = (ulong) (ulong) (size_1065) * 32;
      D_8152 = (ulong) (size_1065) * 4;
      offset_1064 = -(lbound_1062);
      lbound_1049 = (long) *its;
      ubound_1050 = (long) *ite;
      stride_1053 = (1 - lbound_1049) + ubound_1050;
      stride_1053 = gf_max(stride_1053, 0);
      lbound_1051 = (long) *kts;
      ubound_1052 = (long) *kte;
      stride_1055 = ((1 - lbound_1051) + ubound_1052) * stride_1053;
      stride_1055 = gf_max(stride_1055, 0);
      ubound_1054 = (long) *nx;
      size_1057 = stride_1055 * (ubound_1054);
      size_1057 = gf_max(size_1057, 0);
      D_8147 = size_1057 + -1;
      D_8148 = (ulong) (ulong) (size_1057) * 32;
      D_8149 = (ulong) (size_1057) * 4;
      offset_1056 = (-(lbound_1049) - lbound_1051 * stride_1053) - (stride_1055);
      lbound_1143 = (long) *its;
      ubound_1144 = (long) *ite;
      stride_1147 = (1 - lbound_1143) + ubound_1144;
      stride_1147 = gf_max(stride_1147, 0);
      lbound_1145 = (long) *kts;
      ubound_1146 = (long) *kte;
      stride_1149 = ((1 - lbound_1145) + ubound_1146) * stride_1147;
      stride_1149 = gf_max(stride_1149, 0);
      ubound_1148 = (long) *nx;
      size_1151 = stride_1149 * (ubound_1148);
      size_1151 = gf_max(size_1151, 0);
      D_8144 = size_1151 + -1;
      D_8145 = (ulong) (ulong) (size_1151) * 32;
      D_8146 = (ulong) (size_1151) * 4;
      offset_1150 = (-(lbound_1143) - lbound_1145 * stride_1147) - (stride_1149);
      lbound_1031 = (long) *its;
      ubound_1032 = (long) *ite;
      stride_1035 = (1 - lbound_1031) + ubound_1032;
      stride_1035 = gf_max(stride_1035, 0);
      lbound_1033 = (long) *kts;
      ubound_1034 = (long) *kte;
      stride_1037 = ((1 - lbound_1033) + ubound_1034) * stride_1035;
      stride_1037 = gf_max(stride_1037, 0);
      ubound_1036 = (long) *nx;
      size_1039 = stride_1037 * (ubound_1036);
      size_1039 = gf_max(size_1039, 0);
      D_8141 = size_1039 + -1;
      D_8142 = (ulong) (ulong) (size_1039) * 32;
      D_8143 = (ulong) (size_1039) * 4;
      offset_1038 = (-(lbound_1031) - lbound_1033 * stride_1035) - (stride_1037);
      lbound_1127 = (long) *its;
      ubound_1128 = (long) *ite;
      stride_1131 = (1 - lbound_1127) + ubound_1128;
      stride_1131 = gf_max(stride_1131, 0);
      lbound_1129 = (long) *kts;
      ubound_1130 = (long) *kte;
      stride_1133 = ((1 - lbound_1129) + ubound_1130) * stride_1131;
      stride_1133 = gf_max(stride_1133, 0);
      ubound_1132 = (long) *nx;
      size_1135 = stride_1133 * (ubound_1132);
      size_1135 = gf_max(size_1135, 0);
      D_8138 = size_1135 + -1;
      D_8139 = (ulong) (ulong) (size_1135) * 32;
      D_8140 = (ulong) (size_1135) * 4;
      offset_1134 = (-(lbound_1127) - lbound_1129 * stride_1131) - (stride_1133);
      lbound_1040 = (long) *its;
      ubound_1041 = (long) *ite;
      stride_1044 = (1 - lbound_1040) + ubound_1041;
      stride_1044 = gf_max(stride_1044, 0);
      lbound_1042 = (long) *kts;
      ubound_1043 = (long) *kte;
      stride_1046 = ((1 - lbound_1042) + ubound_1043) * stride_1044;
      stride_1046 = gf_max(stride_1046, 0);
      ubound_1045 = (long) *nx;
      size_1048 = stride_1046 * (ubound_1045);
      size_1048 = gf_max(size_1048, 0);
      D_8135 = size_1048 + -1;
      D_8136 = (ulong) (ulong) (size_1048) * 32;
      D_8137 = (ulong) (size_1048) * 4;
      offset_1047 = (-(lbound_1040) - lbound_1042 * stride_1044) - (stride_1046);
      lbound_1100 = (long) *its;
      ubound_1101 = (long) *ite;
      stride_1104 = (1 - lbound_1100) + ubound_1101;
      stride_1104 = gf_max(stride_1104, 0);
      lbound_1102 = (long) *kts;
      ubound_1103 = (long) *kte;
      stride_1106 = ((1 - lbound_1102) + ubound_1103) * stride_1104;
      stride_1106 = gf_max(stride_1106, 0);
      ubound_1105 = (long) *nx;
      size_1108 = stride_1106 * (ubound_1105);
      size_1108 = gf_max(size_1108, 0);
      D_8132 = size_1108 + -1;
      D_8133 = (ulong) (ulong) (size_1108) * 32;
      D_8134 = (ulong) (size_1108) * 4;
      offset_1107 = (-(lbound_1100) - lbound_1102 * stride_1104) - (stride_1106);
      lbound_1186 = (long) *its;
      ubound_1187 = (long) *ite;
      stride_1190 = (1 - lbound_1186) + ubound_1187;
      stride_1190 = gf_max(stride_1190, 0);
      lbound_1188 = (long) *kts;
      ubound_1189 = (long) *kte;
      size_1192 = ((1 - lbound_1188) + ubound_1189) * stride_1190;
      size_1192 = gf_max(size_1192, 0);
      D_8129 = size_1192 + -1;
      D_8130 = (ulong) (ulong) (size_1192) * 32;
      D_8131 = (ulong) (size_1192) * 4;
      offset_1191 = -(lbound_1186) - lbound_1188 * stride_1190;
      lbound_1020 = (long) *its;
      ubound_1021 = (long) *ite;
      stride_1024 = (1 - lbound_1020) + ubound_1021;
      stride_1024 = gf_max(stride_1024, 0);
      lbound_1022 = (long) *jts;
      ubound_1023 = (long) *jte;
      size_1026 = ((1 - lbound_1022) + ubound_1023) * stride_1024;
      size_1026 = gf_max(size_1026, 0);
      D_8126 = size_1026 + -1;
      D_8127 = (ulong) (ulong) (size_1026) * 32;
      D_8128 = (ulong) (size_1026) * 4;
      offset_1025 = -(lbound_1020) - lbound_1022 * stride_1024;
      lbound_1013 = (long) *its;
      ubound_1014 = (long) *ite;
      stride_1017 = (1 - lbound_1013) + ubound_1014;
      stride_1017 = gf_max(stride_1017, 0);
      lbound_1015 = (long) *jts;
      ubound_1016 = (long) *jte;
      size_1019 = ((1 - lbound_1015) + ubound_1016) * stride_1017;
      size_1019 = gf_max(size_1019, 0);
      D_8123 = size_1019 + -1;
      D_8124 = (ulong) (ulong) (size_1019) * 32;
      D_8125 = (ulong) (size_1019) * 4;
      offset_1018 = -(lbound_1013) - lbound_1015 * stride_1017;
      lbound_999 = (long) *its;
      ubound_1000 = (long) *ite;
      stride_1003 = (1 - lbound_999) + ubound_1000;
      stride_1003 = gf_max(stride_1003, 0);
      lbound_1001 = (long) *jts;
      ubound_1002 = (long) *jte;
      size_1005 = ((1 - lbound_1001) + ubound_1002) * stride_1003;
      size_1005 = gf_max(size_1005, 0);
      D_8120 = size_1005 + -1;
      D_8121 = (ulong) (ulong) (size_1005) * 32;
      D_8122 = (ulong) (size_1005) * 4;
      offset_1004 = -(lbound_999) - lbound_1001 * stride_1003;
      lbound_1006 = (long) *its;
      ubound_1007 = (long) *ite;
      stride_1010 = (1 - lbound_1006) + ubound_1007;
      stride_1010 = gf_max(stride_1010, 0);
      lbound_1008 = (long) *jts;
      ubound_1009 = (long) *jte;
      size_1012 = ((1 - lbound_1008) + ubound_1009) * stride_1010;
      size_1012 = gf_max(size_1012, 0);
      D_8117 = size_1012 + -1;
      D_8118 = (ulong) (ulong) (size_1012) * 32;
      D_8119 = (ulong) (size_1012) * 4;
      offset_1011 = -(lbound_1006) - lbound_1008 * stride_1010;
      lbound_992 = (long) *its;
      ubound_993 = (long) *ite;
      stride_996 = (1 - lbound_992) + ubound_993;
      stride_996 = gf_max(stride_996, 0);
      lbound_994 = (long) *jts;
      ubound_995 = (long) *jte;
      size_998 = ((1 - lbound_994) + ubound_995) * stride_996;
      size_998 = gf_max(size_998, 0);
      D_8114 = size_998 + -1;
      D_8115 = (ulong) (ulong) (size_998) * 32;
      D_8116 = (ulong) (size_998) * 4;
      offset_997 = -(lbound_992) - lbound_994 * stride_996;
      lbound_1161 = (long) *its;
      ubound_1162 = (long) *ite;
      size_1164 = (1 - lbound_1161) + ubound_1162;
      size_1164 = gf_max(size_1164, 0);
      D_8110 = size_1164 + -1;
      D_8111 = (ulong) (ulong) (size_1164) * 32;
      D_8112 = (ulong) (size_1164) * 4;
      D_8113 = (device void *) workspace.allocate (gf_max((ulong) (size_1164 * 4), 1));
      xfac1 = (device GFReal *) D_8113;
      offset_1163 = -(lbound_1161);
      lbound_1165 = (long) *its;
      ubound_1166 = (long) *ite;
      size_1168 = (1 - lbound_1165) + ubound_1166;
      size_1168 = gf_max(size_1168, 0);
      D_8106 = size_1168 + -1;
      D_8107 = (ulong) (ulong) (size_1168) * 32;
      D_8108 = (ulong) (size_1168) * 4;
      D_8109 = (device void *) workspace.allocate (gf_max((ulong) (size_1168 * 4), 1));
      xfac2 = (device GFReal *) D_8109;
      offset_1167 = -(lbound_1165);
      lbound_1173 = (long) *its;
      ubound_1174 = (long) *ite;
      size_1176 = (1 - lbound_1173) + ubound_1174;
      size_1176 = gf_max(size_1176, 0);
      D_8102 = size_1176 + -1;
      D_8103 = (ulong) (ulong) (size_1176) * 32;
      D_8104 = (ulong) (size_1176) * 4;
      D_8105 = (device void *) workspace.allocate (gf_max((ulong) (size_1176 * 4), 1));
      xmb_ave = (device GFReal *) D_8105;
      offset_1175 = -(lbound_1173);
      lbound_1177 = (long) *its;
      ubound_1178 = (long) *ite;
      stride_1179 = (1 - lbound_1177) + ubound_1178;
      stride_1179 = gf_max(stride_1179, 0);
      size_1181 = stride_1179 * 5;
      size_1181 = gf_max(size_1181, 0);
      D_8098 = size_1181 + -1;
      D_8099 = (ulong) (ulong) (size_1181) * 32;
      D_8100 = (ulong) (size_1181) * 4;
      D_8101 = (device void *) workspace.allocate (gf_max((ulong) (size_1181 * 4), 1));
      xmb_w = (device GFReal *) D_8101;
      offset_1180 = -(lbound_1177) - (stride_1179);
      lbound_1182 = (long) *its;
      ubound_1183 = (long) *ite;
      size_1185 = (1 - lbound_1182) + ubound_1183;
      size_1185 = gf_max(size_1185, 0);
      D_8094 = size_1185 + -1;
      D_8095 = (ulong) (ulong) (size_1185) * 32;
      D_8096 = (ulong) (size_1185) * 4;
      D_8097 = (device void *) workspace.allocate (gf_max((ulong) (size_1185 * 4), 1));
      xmbweight = (device GFReal *) D_8097;
      offset_1184 = -(lbound_1182);
      weight[0] = -9.99e+2f;
      wm[0] = -9.99e+2f;
      {
        int D_8011;
        int D_8012;

        D_8011 = *kts;
        D_8012 = *ktf;
        k = D_8011;
        while (1)
          {
            {
              bool D_8015;

              D_8015 = k > D_8012;
              if (D_8015) break;
              {
                int D_8016;
                int D_8017;

                D_8016 = *its;
                D_8017 = *itf;
                i = D_8016;
                while (1)
                  {
                    {
                      bool D_8020;

                      D_8020 = i > D_8017;
                      if (D_8020) break;
                      outtem[((long) k * stride_1084 + offset_1085) + (long) i] = 0.0f;
                      outq[((long) k * stride_1070 + offset_1071) + (long) i] = 0.0f;
                      outqc[((long) k * stride_1077 + offset_1078) + (long) i] = 0.0f;
                      subt[((long) k * stride_1140 + offset_1141) + (long) i] = 0.0f;
                      subq[((long) k * stride_1124 + offset_1125) + (long) i] = 0.0f;
                      sub_mas[((long) k * stride_1117 + offset_1118) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_8021;
        int D_8022;

        D_8021 = *its;
        D_8022 = *itf;
        i = D_8021;
        while (1)
          {
            {
              bool D_8025;

              D_8025 = i > D_8022;
              if (D_8025) break;
              pre[(long) i + offset_1098] = 0.0f;
              xmb[(long) i + offset_1171] = 0.0f;
              xfac1[(long) i + offset_1163] = 0.0f;
              xfac2[(long) i + offset_1167] = 0.0f;
              xmbweight[(long) i + offset_1184] = 1.0e+0f;

              i = i + 1;
            }
          }

      }
      {
        int D_8026;
        int D_8027;

        D_8026 = *its;
        D_8027 = *itf;
        i = D_8026;
        while (1)
          {
            {
              bool D_8030;

              D_8030 = i > D_8027;
              if (D_8030) break;
              if (ierr[(long) i + offset_1060] == 0)
                {
                  {
                    int D_8032;
                    int D_8033;

                    D_8032 = (((*iens + -1) * *nx) * *nx2) * *maxens3 + 1;
                    D_8033 = ((*iens * *nx) * *nx2) * *maxens3;
                    n = D_8032;
                    while (1)
                      {
                        {
                          bool D_8036;

                          D_8036 = n > D_8033;
                          if (D_8036) break;
                          if (pr_ens[(((long) n * stride_1093 + offset_1094) + (long) *j * stride_1091) + (long) i] <= 0.0f)
                            {
                              xf_ens[(((long) n * stride_1158 + offset_1159) + (long) *j * stride_1156) + (long) i] = 0.0f;
                            }


                          n = n + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        long D_8038;
        long D_8039;
        long D_8040;

        D_8038 = offset_1180;
        D_8039 = lbound_1177;
        D_8040 = ubound_1178;
        {
          long S_1193;

          S_1193 = 1;
          while (1)
            {
              if (S_1193 > 5) break;
              {
                long S_1194;
                long D_8043;

                D_8043 = S_1193 * stride_1179 + D_8038;
                S_1194 = D_8039;
                while (1)
                  {
                    if (S_1194 > D_8040) break;
                    xmb_w[S_1194 + D_8043] = 0.0f;
                    S_1194 = S_1194 + 1;
                  }

              }
              S_1193 = S_1193 + 1;
            }

        }
      }
      ddtes = 1.0e+2f;
      {
        int D_8046;
        int D_8047;

        D_8046 = *its;
        D_8047 = *itf;
        i = D_8046;
        while (1)
          {
            {
              bool D_8050;

              D_8050 = i > D_8047;
              if (D_8050) break;
              if (ierr[(long) i + offset_1060] == 0)
                {
                  k = 0;
                  xmb_ave[(long) i + offset_1175] = 0.0f;
                  {
                    int D_8052;
                    int D_8053;

                    D_8052 = (((*iens + -1) * *nx) * *nx2) * *maxens3 + 1;
                    D_8053 = ((*iens * *nx) * *nx2) * *maxens3;
                    n = D_8052;
                    while (1)
                      {
                        {
                          bool D_8056;

                          D_8056 = n > D_8053;
                          if (D_8056) break;
                          k = k + 1;
                          xmb_ave[(long) i + offset_1175] = xmb_ave[(long) i + offset_1175] + xf_ens[(((long) n * stride_1158 + offset_1159) + (long) *j * stride_1156) + (long) i];

                          n = n + 1;
                        }
                      }

                  }
                  xmb_ave[(long) i + offset_1175] = xmb_ave[(long) i + offset_1175] / (GFReal) k;
                  if (xmb_ave[(long) i + offset_1175] <= 0.0f)
                    {
                      ierr[(long) i + offset_1060] = 13;
                      xmb_ave[(long) i + offset_1175] = 0.0f;
                    }

                  xmb[(long) i + offset_1171] = sig[(long) i + offset_1111] * xmb_ave[(long) i + offset_1175];
                  {
                    GFReal M_1195;
                    GFReal D_8059;

                    M_1195 = 1.0e+0f;
                    D_8059 = closure_n[(long) i + offset_1029];
                    M_1195 = gf_max(D_8059, M_1195);
                    clos_wei = 1.6e+1f / M_1195;
                  }
                  if (xmb[(long) i + offset_1171] == 0.0f)
                    {
                      ierr[(long) i + offset_1060] = 19;
                    }

                  if (xmb[(long) i + offset_1171] > 1.0e+2f)
                    {
                      ierr[(long) i + offset_1060] = 19;
                    }

                  xfac1[(long) i + offset_1163] = xmb[(long) i + offset_1171];
                  xfac2[(long) i + offset_1167] = xmb[(long) i + offset_1171];
                }


              i = i + 1;
            }
          }

      }
      {
        int D_8062;
        int D_8063;

        D_8062 = *kts;
        D_8063 = *ktf;
        k = D_8062;
        while (1)
          {
            {
              bool D_8066;

              D_8066 = k > D_8063;
              if (D_8066) break;
              {
                int D_8067;
                int D_8068;

                D_8067 = *its;
                D_8068 = *itf;
                i = D_8067;
                while (1)
                  {
                    {
                      bool D_8071;

                      D_8071 = i > D_8068;
                      if (D_8071) break;
                      dtt = 0.0f;
                      dtts = 0.0f;
                      dtq = 0.0f;
                      dtqs = 0.0f;
                      dtqc = 0.0f;
                      dtpw = 0.0f;
                      if (ierr[(long) i + offset_1060] == 0 && ktop[(long) i + offset_1064] >= k)
                        {
                          {
                            int D_8073;

                            D_8073 = *nx;
                            n = 1;
                            while (1)
                              {
                                {
                                  bool D_8076;

                                  D_8076 = n > D_8073;
                                  if (D_8076) break;
                                  dtt = dellat[(((long) n * stride_1055 + offset_1056) + (long) k * stride_1053) + (long) i] + dtt;
                                  dtts = subt_ens[(((long) n * stride_1149 + offset_1150) + (long) k * stride_1147) + (long) i] + dtts;
                                  dtq = dellaq[(((long) n * stride_1037 + offset_1038) + (long) k * stride_1035) + (long) i] + dtq;
                                  dtqs = subq_ens[(((long) n * stride_1133 + offset_1134) + (long) k * stride_1131) + (long) i] + dtqs;
                                  dtqc = dellaqc[(((long) n * stride_1046 + offset_1047) + (long) k * stride_1044) + (long) i] + dtqc;
                                  dtpw = pw[(((long) n * stride_1106 + offset_1107) + (long) k * stride_1104) + (long) i] + dtpw;

                                  n = n + 1;
                                }
                              }

                          }
                          outtem[((long) k * stride_1084 + offset_1085) + (long) i] = (xmb[(long) i + offset_1171] * dtt) / (GFReal) *nx;
                          subt[((long) k * stride_1140 + offset_1141) + (long) i] = (xmb[(long) i + offset_1171] * dtts) / (GFReal) *nx;
                          outq[((long) k * stride_1070 + offset_1071) + (long) i] = (xmb[(long) i + offset_1171] * dtq) / (GFReal) *nx;
                          subq[((long) k * stride_1124 + offset_1125) + (long) i] = (xmb[(long) i + offset_1171] * dtqs) / (GFReal) *nx;
                          outqc[((long) k * stride_1077 + offset_1078) + (long) i] = (xmb[(long) i + offset_1171] * dtqc) / (GFReal) *nx;
                          pre[(long) i + offset_1098] = pre[(long) i + offset_1098] + (xmb[(long) i + offset_1171] * dtpw) / (GFReal) *nx;
                          sub_mas[((long) k * stride_1117 + offset_1118) + (long) i] = zu[((long) k * stride_1190 + offset_1191) + (long) i] * xmb[(long) i + offset_1171];
                        }


                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_8077;
        int D_8078;

        D_8077 = *its;
        D_8078 = *itf;
        i = D_8077;
        while (1)
          {
            {
              bool D_8081;

              D_8081 = i > D_8078;
              if (D_8081) break;
              if (ierr[(long) i + offset_1060] == 0)
                {
                  {
                    int D_8083;
                    int D_8084;

                    D_8083 = (((*iens + -1) * *nx) * *nx2) * *maxens3 + 1;
                    D_8084 = ((*iens * *nx) * *nx2) * *maxens3;
                    k = D_8083;
                    while (1)
                      {
                        {
                          bool D_8087;

                          D_8087 = k > D_8084;
                          if (D_8087) break;
                          xf_ens[(((long) k * stride_1158 + offset_1159) + (long) *j * stride_1156) + (long) i] = (sig[(long) i + offset_1111] * xf_ens[(((long) k * stride_1158 + offset_1159) + (long) *j * stride_1156) + (long) i]) * xfac1[(long) i + offset_1163];

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_8088;
        int D_8089;

        D_8088 = *its;
        D_8089 = *itf;
        i = D_8088;
        while (1)
          {
            {
              bool D_8092;

              D_8092 = i > D_8089;
              if (D_8092) break;
              if (ierr[(long) i + offset_1060] != 0)
                {
                  apr_w[((long) *j * stride_1024 + offset_1025) + (long) i] = 0.0f;
                  apr_st[((long) *j * stride_1017 + offset_1018) + (long) i] = 0.0f;
                  apr_gr[((long) *j * stride_1003 + offset_1004) + (long) i] = 0.0f;
                  apr_mc[((long) *j * stride_1010 + offset_1011) + (long) i] = 0.0f;
                  apr_as[((long) *j * stride_996 + offset_997) + (long) i] = 0.0f;
                }


              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14> static inline void neg_check(thread GFWorkspace &workspace, P0 j, P1 subt, P2 subq, P3 dt, P4 q, P5 outq, P6 outt, P7 outqc, P8 pret, P9 its, P10 ite, P11 kts, P12 kte, P13 itf, P14 ktf) {
  GFFrame frame(workspace);
  int i;
  int k;
  long lbound_1196;
  long ubound_1197;
  long lbound_1198;
  long ubound_1199;
  long stride_1200;
  long offset_1201;
  long size_1202;
  long lbound_1203;
  long ubound_1204;
  long lbound_1205;
  long ubound_1206;
  long stride_1207;
  long offset_1208;
  long size_1209;
  long lbound_1210;
  long ubound_1211;
  long lbound_1212;
  long ubound_1213;
  long stride_1214;
  long offset_1215;
  long size_1216;
  long lbound_1217;
  long ubound_1218;
  long offset_1219;
  long size_1220;
  long lbound_1221;
  long ubound_1222;
  long lbound_1223;
  long ubound_1224;
  long stride_1225;
  long offset_1226;
  long size_1227;
  GFReal qmem;
  GFReal qmem1;
  GFReal qmem2;
  GFReal qmemf;
  GFReal qtest;
  long lbound_1228;
  long ubound_1229;
  long lbound_1230;
  long ubound_1231;
  long stride_1232;
  long offset_1233;
  long size_1234;
  long lbound_1235;
  long ubound_1236;
  long lbound_1237;
  long ubound_1238;
  long stride_1239;
  long offset_1240;
  long size_1241;
  GFReal thresh;
  long D_8294;
  ulong D_8295;
  ulong D_8296;
  long D_8297;
  ulong D_8298;
  ulong D_8299;
  long D_8300;
  ulong D_8301;
  ulong D_8302;
  long D_8303;
  ulong D_8304;
  ulong D_8305;
  long D_8306;
  ulong D_8307;
  ulong D_8308;
  long D_8309;
  ulong D_8310;
  ulong D_8311;
  long D_8312;
  ulong D_8313;
  ulong D_8314;

  lbound_1235 = (long) *its;
  ubound_1236 = (long) *ite;
  stride_1239 = (1 - lbound_1235) + ubound_1236;
  stride_1239 = gf_max(stride_1239, 0);
  lbound_1237 = (long) *kts;
  ubound_1238 = (long) *kte;
  size_1241 = ((1 - lbound_1237) + ubound_1238) * stride_1239;
  size_1241 = gf_max(size_1241, 0);
  D_8312 = size_1241 + -1;
  D_8313 = (ulong) (ulong) (size_1241) * 32;
  D_8314 = (ulong) (size_1241) * 4;
  offset_1240 = -(lbound_1235) - lbound_1237 * stride_1239;
  lbound_1210 = (long) *its;
  ubound_1211 = (long) *ite;
  stride_1214 = (1 - lbound_1210) + ubound_1211;
  stride_1214 = gf_max(stride_1214, 0);
  lbound_1212 = (long) *kts;
  ubound_1213 = (long) *kte;
  size_1216 = ((1 - lbound_1212) + ubound_1213) * stride_1214;
  size_1216 = gf_max(size_1216, 0);
  D_8309 = size_1216 + -1;
  D_8310 = (ulong) (ulong) (size_1216) * 32;
  D_8311 = (ulong) (size_1216) * 4;
  offset_1215 = -(lbound_1210) - lbound_1212 * stride_1214;
  lbound_1228 = (long) *its;
  ubound_1229 = (long) *ite;
  stride_1232 = (1 - lbound_1228) + ubound_1229;
  stride_1232 = gf_max(stride_1232, 0);
  lbound_1230 = (long) *kts;
  ubound_1231 = (long) *kte;
  size_1234 = ((1 - lbound_1230) + ubound_1231) * stride_1232;
  size_1234 = gf_max(size_1234, 0);
  D_8306 = size_1234 + -1;
  D_8307 = (ulong) (ulong) (size_1234) * 32;
  D_8308 = (ulong) (size_1234) * 4;
  offset_1233 = -(lbound_1228) - lbound_1230 * stride_1232;
  lbound_1196 = (long) *its;
  ubound_1197 = (long) *ite;
  stride_1200 = (1 - lbound_1196) + ubound_1197;
  stride_1200 = gf_max(stride_1200, 0);
  lbound_1198 = (long) *kts;
  ubound_1199 = (long) *kte;
  size_1202 = ((1 - lbound_1198) + ubound_1199) * stride_1200;
  size_1202 = gf_max(size_1202, 0);
  D_8303 = size_1202 + -1;
  D_8304 = (ulong) (ulong) (size_1202) * 32;
  D_8305 = (ulong) (size_1202) * 4;
  offset_1201 = -(lbound_1196) - lbound_1198 * stride_1200;
  lbound_1203 = (long) *its;
  ubound_1204 = (long) *ite;
  stride_1207 = (1 - lbound_1203) + ubound_1204;
  stride_1207 = gf_max(stride_1207, 0);
  lbound_1205 = (long) *kts;
  ubound_1206 = (long) *kte;
  size_1209 = ((1 - lbound_1205) + ubound_1206) * stride_1207;
  size_1209 = gf_max(size_1209, 0);
  D_8300 = size_1209 + -1;
  D_8301 = (ulong) (ulong) (size_1209) * 32;
  D_8302 = (ulong) (size_1209) * 4;
  offset_1208 = -(lbound_1203) - lbound_1205 * stride_1207;
  lbound_1217 = (long) *its;
  ubound_1218 = (long) *ite;
  size_1220 = (1 - lbound_1217) + ubound_1218;
  size_1220 = gf_max(size_1220, 0);
  D_8297 = size_1220 + -1;
  D_8298 = (ulong) (ulong) (size_1220) * 32;
  D_8299 = (ulong) (size_1220) * 4;
  offset_1219 = -(lbound_1217);
  lbound_1221 = (long) *its;
  ubound_1222 = (long) *ite;
  stride_1225 = (1 - lbound_1221) + ubound_1222;
  stride_1225 = gf_max(stride_1225, 0);
  lbound_1223 = (long) *kts;
  ubound_1224 = (long) *kte;
  size_1227 = ((1 - lbound_1223) + ubound_1224) * stride_1225;
  size_1227 = gf_max(size_1227, 0);
  D_8294 = size_1227 + -1;
  D_8295 = (ulong) (ulong) (size_1227) * 32;
  D_8296 = (ulong) (size_1227) * 4;
  offset_1226 = -(lbound_1221) - lbound_1223 * stride_1225;
  thresh = 3.00010009765625e+2f;
  {
    int D_8255;
    int D_8256;

    D_8255 = *its;
    D_8256 = *itf;
    i = D_8255;
    while (1)
      {
        {
          bool D_8259;

          D_8259 = i > D_8256;
          if (D_8259) break;
          qmemf = 1.0e+0f;
          qmem = 0.0f;
          {
            int D_8260;
            int D_8261;

            D_8260 = *kts;
            D_8261 = *ktf;
            k = D_8260;
            while (1)
              {
                {
                  bool D_8264;

                  D_8264 = k > D_8261;
                  if (D_8264) break;
                  qmem = ((subt[((long) k * stride_1239 + offset_1240) + (long) i] + outt[((long) k * stride_1214 + offset_1215) + (long) i])) * 8.64e+4f;
                  if (thresh * 2.0e+0f < qmem)
                    {
                      qmem2 = (thresh * 2.0e+0f) / qmem;
                      {
                        GFReal M_1242;

                        M_1242 = qmemf;
                        M_1242 = gf_min(qmem2, M_1242);
                        qmemf = M_1242;
                      }
                    }

                  if (-thresh > qmem)
                    {
                      qmem2 = -(thresh / qmem);
                      {
                        GFReal M_1243;

                        M_1243 = qmemf;
                        M_1243 = gf_min(qmem2, M_1243);
                        qmemf = M_1243;
                      }
                    }


                  k = k + 1;
                }
              }

          }
          {
            int D_8269;
            int D_8270;

            D_8269 = *kts;
            D_8270 = *ktf;
            k = D_8269;
            while (1)
              {
                {
                  bool D_8273;

                  D_8273 = k > D_8270;
                  if (D_8273) break;
                  subq[((long) k * stride_1232 + offset_1233) + (long) i] = subq[((long) k * stride_1232 + offset_1233) + (long) i] * qmemf;
                  subt[((long) k * stride_1239 + offset_1240) + (long) i] = subt[((long) k * stride_1239 + offset_1240) + (long) i] * qmemf;
                  outq[((long) k * stride_1200 + offset_1201) + (long) i] = outq[((long) k * stride_1200 + offset_1201) + (long) i] * qmemf;
                  outt[((long) k * stride_1214 + offset_1215) + (long) i] = outt[((long) k * stride_1214 + offset_1215) + (long) i] * qmemf;
                  outqc[((long) k * stride_1207 + offset_1208) + (long) i] = outqc[((long) k * stride_1207 + offset_1208) + (long) i] * qmemf;

                  k = k + 1;
                }
              }

          }
          pret[(long) i + offset_1219] = pret[(long) i + offset_1219] * qmemf;

          i = i + 1;
        }
      }

  }
  thresh = 1.00000001335143196001808973960578441619873046875e-10f;
  {
    int D_8274;
    int D_8275;

    D_8274 = *its;
    D_8275 = *itf;
    i = D_8274;
    while (1)
      {
        {
          bool D_8278;

          D_8278 = i > D_8275;
          if (D_8278) break;
          qmemf = 1.0e+0f;
          {
            int D_8279;
            int D_8280;

            D_8279 = *kts;
            D_8280 = *ktf + -1;
            k = D_8279;
            while (1)
              {
                {
                  bool D_8283;

                  D_8283 = k > D_8280;
                  if (D_8283) break;
                  qmem = subq[((long) k * stride_1232 + offset_1233) + (long) i] + outq[((long) k * stride_1200 + offset_1201) + (long) i];
                  if (gf_abs(qmem) > 0.0f)
                    {
                      qtest = q[((long) k * stride_1225 + offset_1226) + (long) i] + ((subq[((long) k * stride_1232 + offset_1233) + (long) i] + outq[((long) k * stride_1200 + offset_1201) + (long) i])) * *dt;
                      if (qtest < thresh)
                        {
                          qmem1 = outq[((long) k * stride_1200 + offset_1201) + (long) i] + subq[((long) k * stride_1232 + offset_1233) + (long) i];
                          qmem2 = ((thresh - q[((long) k * stride_1225 + offset_1226) + (long) i])) / *dt;
                          {
                            GFReal M_1244;
                            GFReal D_8287;

                            M_1244 = qmemf;
                            D_8287 = qmem2 / qmem1;
                            M_1244 = gf_min(D_8287, M_1244);
                            qmemf = M_1244;
                          }
                          {
                            GFReal M_1245;

                            M_1245 = 0.0f;
                            M_1245 = gf_max(qmemf, M_1245);
                            qmemf = M_1245;
                          }
                        }

                    }


                  k = k + 1;
                }
              }

          }
          {
            int D_8289;
            int D_8290;

            D_8289 = *kts;
            D_8290 = *ktf;
            k = D_8289;
            while (1)
              {
                {
                  bool D_8293;

                  D_8293 = k > D_8290;
                  if (D_8293) break;
                  subq[((long) k * stride_1232 + offset_1233) + (long) i] = subq[((long) k * stride_1232 + offset_1233) + (long) i] * qmemf;
                  subt[((long) k * stride_1239 + offset_1240) + (long) i] = subt[((long) k * stride_1239 + offset_1240) + (long) i] * qmemf;
                  outq[((long) k * stride_1200 + offset_1201) + (long) i] = outq[((long) k * stride_1200 + offset_1201) + (long) i] * qmemf;
                  outt[((long) k * stride_1214 + offset_1215) + (long) i] = outt[((long) k * stride_1214 + offset_1215) + (long) i] * qmemf;
                  outqc[((long) k * stride_1207 + offset_1208) + (long) i] = outqc[((long) k * stride_1207 + offset_1208) + (long) i] * qmemf;

                  k = k + 1;
                }
              }

          }
          pret[(long) i + offset_1219] = pret[(long) i + offset_1219] * qmemf;

          i = i + 1;
        }
      }

  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43> static inline void g3init(thread GFWorkspace &workspace, P0 rthcuten, P1 rqvcuten, P2 rqccuten, P3 rqicuten, P4 mass_flux, P5 cp, P6 restart, P7 p_qc, P8 p_qi, P9 p_first_scalar, P10 rthften, P11 rqvften, P12 apr_gr, P13 apr_w, P14 apr_mc, P15 apr_st, P16 apr_as, P17 apr_capma, P18 apr_capme, P19 apr_capmi, P20 cugd_tten, P21 cugd_ttens, P22 cugd_qvten, P23 cugd_qvtens, P24 cugd_qcten, P25 allowed_to_read, P26 ids, P27 ide, P28 jds, P29 jde, P30 kds, P31 kde, P32 ims, P33 ime, P34 jms, P35 jme, P36 kms, P37 kme, P38 its, P39 ite, P40 jts, P41 jte, P42 kts, P43 kte) {
  GFFrame frame(workspace);
  long lbound_1246;
  long ubound_1247;
  long lbound_1248;
  long ubound_1249;
  long stride_1250;
  long offset_1251;
  long size_1252;
  long lbound_1253;
  long ubound_1254;
  long lbound_1255;
  long ubound_1256;
  long stride_1257;
  long offset_1258;
  long size_1259;
  long lbound_1260;
  long ubound_1261;
  long lbound_1262;
  long ubound_1263;
  long stride_1264;
  long offset_1265;
  long size_1266;
  long lbound_1267;
  long ubound_1268;
  long lbound_1269;
  long ubound_1270;
  long stride_1271;
  long offset_1272;
  long size_1273;
  long lbound_1274;
  long ubound_1275;
  long lbound_1276;
  long ubound_1277;
  long stride_1278;
  long offset_1279;
  long size_1280;
  long lbound_1281;
  long ubound_1282;
  long lbound_1283;
  long ubound_1284;
  long stride_1285;
  long offset_1286;
  long size_1287;
  long lbound_1288;
  long ubound_1289;
  long lbound_1290;
  long ubound_1291;
  long stride_1292;
  long offset_1293;
  long size_1294;
  long lbound_1295;
  long ubound_1296;
  long lbound_1297;
  long ubound_1298;
  long stride_1299;
  long offset_1300;
  long size_1301;
  long lbound_1302;
  long ubound_1303;
  long lbound_1304;
  long ubound_1305;
  long stride_1306;
  long lbound_1307;
  long ubound_1308;
  long stride_1309;
  long offset_1310;
  long size_1311;
  long lbound_1312;
  long ubound_1313;
  long lbound_1314;
  long ubound_1315;
  long stride_1316;
  long lbound_1317;
  long ubound_1318;
  long stride_1319;
  long offset_1320;
  long size_1321;
  long lbound_1322;
  long ubound_1323;
  long lbound_1324;
  long ubound_1325;
  long stride_1326;
  long lbound_1327;
  long ubound_1328;
  long stride_1329;
  long offset_1330;
  long size_1331;
  long lbound_1332;
  long ubound_1333;
  long lbound_1334;
  long ubound_1335;
  long stride_1336;
  long lbound_1337;
  long ubound_1338;
  long stride_1339;
  long offset_1340;
  long size_1341;
  long lbound_1342;
  long ubound_1343;
  long lbound_1344;
  long ubound_1345;
  long stride_1346;
  long lbound_1347;
  long ubound_1348;
  long stride_1349;
  long offset_1350;
  long size_1351;
  int i;
  int itf;
  int j;
  int jtf;
  int k;
  int ktf;
  long lbound_1352;
  long ubound_1353;
  long lbound_1354;
  long ubound_1355;
  long stride_1356;
  long offset_1357;
  long size_1358;
  long lbound_1359;
  long ubound_1360;
  long lbound_1361;
  long ubound_1362;
  long stride_1363;
  long lbound_1364;
  long ubound_1365;
  long stride_1366;
  long offset_1367;
  long size_1368;
  long lbound_1369;
  long ubound_1370;
  long lbound_1371;
  long ubound_1372;
  long stride_1373;
  long lbound_1374;
  long ubound_1375;
  long stride_1376;
  long offset_1377;
  long size_1378;
  long lbound_1379;
  long ubound_1380;
  long lbound_1381;
  long ubound_1382;
  long stride_1383;
  long lbound_1384;
  long ubound_1385;
  long stride_1386;
  long offset_1387;
  long size_1388;
  long lbound_1389;
  long ubound_1390;
  long lbound_1391;
  long ubound_1392;
  long stride_1393;
  long lbound_1394;
  long ubound_1395;
  long stride_1396;
  long offset_1397;
  long size_1398;
  long lbound_1399;
  long ubound_1400;
  long lbound_1401;
  long ubound_1402;
  long stride_1403;
  long lbound_1404;
  long ubound_1405;
  long stride_1406;
  long offset_1407;
  long size_1408;
  long lbound_1409;
  long ubound_1410;
  long lbound_1411;
  long ubound_1412;
  long stride_1413;
  long lbound_1414;
  long ubound_1415;
  long stride_1416;
  long offset_1417;
  long size_1418;
  long D_8621;
  ulong D_8622;
  ulong D_8623;
  long D_8624;
  ulong D_8625;
  ulong D_8626;
  long D_8627;
  ulong D_8628;
  ulong D_8629;
  long D_8630;
  ulong D_8631;
  ulong D_8632;
  long D_8633;
  ulong D_8634;
  ulong D_8635;
  long D_8636;
  ulong D_8637;
  ulong D_8638;
  long D_8639;
  ulong D_8640;
  ulong D_8641;
  long D_8642;
  ulong D_8643;
  ulong D_8644;
  long D_8645;
  ulong D_8646;
  ulong D_8647;
  long D_8648;
  ulong D_8649;
  ulong D_8650;
  long D_8651;
  ulong D_8652;
  ulong D_8653;
  long D_8654;
  ulong D_8655;
  ulong D_8656;
  long D_8657;
  ulong D_8658;
  ulong D_8659;
  long D_8660;
  ulong D_8661;
  ulong D_8662;
  long D_8663;
  ulong D_8664;
  ulong D_8665;
  long D_8666;
  ulong D_8667;
  ulong D_8668;
  long D_8669;
  ulong D_8670;
  ulong D_8671;
  long D_8672;
  ulong D_8673;
  ulong D_8674;
  long D_8675;
  ulong D_8676;
  ulong D_8677;
  long D_8678;
  ulong D_8679;
  ulong D_8680;

  lbound_1399 = (long) *ims;
  ubound_1400 = (long) *ime;
  stride_1403 = (1 - lbound_1399) + ubound_1400;
  stride_1403 = gf_max(stride_1403, 0);
  lbound_1401 = (long) *kms;
  ubound_1402 = (long) *kme;
  stride_1406 = ((1 - lbound_1401) + ubound_1402) * stride_1403;
  stride_1406 = gf_max(stride_1406, 0);
  lbound_1404 = (long) *jms;
  ubound_1405 = (long) *jme;
  size_1408 = ((1 - lbound_1404) + ubound_1405) * stride_1406;
  size_1408 = gf_max(size_1408, 0);
  D_8678 = size_1408 + -1;
  D_8679 = (ulong) (ulong) (size_1408) * 32;
  D_8680 = (ulong) (size_1408) * 4;
  offset_1407 = (-(lbound_1399) - lbound_1401 * stride_1403) - lbound_1404 * stride_1406;
  lbound_1379 = (long) *ims;
  ubound_1380 = (long) *ime;
  stride_1383 = (1 - lbound_1379) + ubound_1380;
  stride_1383 = gf_max(stride_1383, 0);
  lbound_1381 = (long) *kms;
  ubound_1382 = (long) *kme;
  stride_1386 = ((1 - lbound_1381) + ubound_1382) * stride_1383;
  stride_1386 = gf_max(stride_1386, 0);
  lbound_1384 = (long) *jms;
  ubound_1385 = (long) *jme;
  size_1388 = ((1 - lbound_1384) + ubound_1385) * stride_1386;
  size_1388 = gf_max(size_1388, 0);
  D_8675 = size_1388 + -1;
  D_8676 = (ulong) (ulong) (size_1388) * 32;
  D_8677 = (ulong) (size_1388) * 4;
  offset_1387 = (-(lbound_1379) - lbound_1381 * stride_1383) - lbound_1384 * stride_1386;
  lbound_1332 = (long) *ims;
  ubound_1333 = (long) *ime;
  stride_1336 = (1 - lbound_1332) + ubound_1333;
  stride_1336 = gf_max(stride_1336, 0);
  lbound_1334 = (long) *kms;
  ubound_1335 = (long) *kme;
  stride_1339 = ((1 - lbound_1334) + ubound_1335) * stride_1336;
  stride_1339 = gf_max(stride_1339, 0);
  lbound_1337 = (long) *jms;
  ubound_1338 = (long) *jme;
  size_1341 = ((1 - lbound_1337) + ubound_1338) * stride_1339;
  size_1341 = gf_max(size_1341, 0);
  D_8672 = size_1341 + -1;
  D_8673 = (ulong) (ulong) (size_1341) * 32;
  D_8674 = (ulong) (size_1341) * 4;
  offset_1340 = (-(lbound_1332) - lbound_1334 * stride_1336) - lbound_1337 * stride_1339;
  lbound_1342 = (long) *ims;
  ubound_1343 = (long) *ime;
  stride_1346 = (1 - lbound_1342) + ubound_1343;
  stride_1346 = gf_max(stride_1346, 0);
  lbound_1344 = (long) *kms;
  ubound_1345 = (long) *kme;
  stride_1349 = ((1 - lbound_1344) + ubound_1345) * stride_1346;
  stride_1349 = gf_max(stride_1349, 0);
  lbound_1347 = (long) *jms;
  ubound_1348 = (long) *jme;
  size_1351 = ((1 - lbound_1347) + ubound_1348) * stride_1349;
  size_1351 = gf_max(size_1351, 0);
  D_8669 = size_1351 + -1;
  D_8670 = (ulong) (ulong) (size_1351) * 32;
  D_8671 = (ulong) (size_1351) * 4;
  offset_1350 = (-(lbound_1342) - lbound_1344 * stride_1346) - lbound_1347 * stride_1349;
  lbound_1312 = (long) *ims;
  ubound_1313 = (long) *ime;
  stride_1316 = (1 - lbound_1312) + ubound_1313;
  stride_1316 = gf_max(stride_1316, 0);
  lbound_1314 = (long) *kms;
  ubound_1315 = (long) *kme;
  stride_1319 = ((1 - lbound_1314) + ubound_1315) * stride_1316;
  stride_1319 = gf_max(stride_1319, 0);
  lbound_1317 = (long) *jms;
  ubound_1318 = (long) *jme;
  size_1321 = ((1 - lbound_1317) + ubound_1318) * stride_1319;
  size_1321 = gf_max(size_1321, 0);
  D_8666 = size_1321 + -1;
  D_8667 = (ulong) (ulong) (size_1321) * 32;
  D_8668 = (ulong) (size_1321) * 4;
  offset_1320 = (-(lbound_1312) - lbound_1314 * stride_1316) - lbound_1317 * stride_1319;
  lbound_1322 = (long) *ims;
  ubound_1323 = (long) *ime;
  stride_1326 = (1 - lbound_1322) + ubound_1323;
  stride_1326 = gf_max(stride_1326, 0);
  lbound_1324 = (long) *kms;
  ubound_1325 = (long) *kme;
  stride_1329 = ((1 - lbound_1324) + ubound_1325) * stride_1326;
  stride_1329 = gf_max(stride_1329, 0);
  lbound_1327 = (long) *jms;
  ubound_1328 = (long) *jme;
  size_1331 = ((1 - lbound_1327) + ubound_1328) * stride_1329;
  size_1331 = gf_max(size_1331, 0);
  D_8663 = size_1331 + -1;
  D_8664 = (ulong) (ulong) (size_1331) * 32;
  D_8665 = (ulong) (size_1331) * 4;
  offset_1330 = (-(lbound_1322) - lbound_1324 * stride_1326) - lbound_1327 * stride_1329;
  lbound_1409 = (long) *ims;
  ubound_1410 = (long) *ime;
  stride_1413 = (1 - lbound_1409) + ubound_1410;
  stride_1413 = gf_max(stride_1413, 0);
  lbound_1411 = (long) *kms;
  ubound_1412 = (long) *kme;
  stride_1416 = ((1 - lbound_1411) + ubound_1412) * stride_1413;
  stride_1416 = gf_max(stride_1416, 0);
  lbound_1414 = (long) *jms;
  ubound_1415 = (long) *jme;
  size_1418 = ((1 - lbound_1414) + ubound_1415) * stride_1416;
  size_1418 = gf_max(size_1418, 0);
  D_8660 = size_1418 + -1;
  D_8661 = (ulong) (ulong) (size_1418) * 32;
  D_8662 = (ulong) (size_1418) * 4;
  offset_1417 = (-(lbound_1409) - lbound_1411 * stride_1413) - lbound_1414 * stride_1416;
  lbound_1389 = (long) *ims;
  ubound_1390 = (long) *ime;
  stride_1393 = (1 - lbound_1389) + ubound_1390;
  stride_1393 = gf_max(stride_1393, 0);
  lbound_1391 = (long) *kms;
  ubound_1392 = (long) *kme;
  stride_1396 = ((1 - lbound_1391) + ubound_1392) * stride_1393;
  stride_1396 = gf_max(stride_1396, 0);
  lbound_1394 = (long) *jms;
  ubound_1395 = (long) *jme;
  size_1398 = ((1 - lbound_1394) + ubound_1395) * stride_1396;
  size_1398 = gf_max(size_1398, 0);
  D_8657 = size_1398 + -1;
  D_8658 = (ulong) (ulong) (size_1398) * 32;
  D_8659 = (ulong) (size_1398) * 4;
  offset_1397 = (-(lbound_1389) - lbound_1391 * stride_1393) - lbound_1394 * stride_1396;
  lbound_1359 = (long) *ims;
  ubound_1360 = (long) *ime;
  stride_1363 = (1 - lbound_1359) + ubound_1360;
  stride_1363 = gf_max(stride_1363, 0);
  lbound_1361 = (long) *kms;
  ubound_1362 = (long) *kme;
  stride_1366 = ((1 - lbound_1361) + ubound_1362) * stride_1363;
  stride_1366 = gf_max(stride_1366, 0);
  lbound_1364 = (long) *jms;
  ubound_1365 = (long) *jme;
  size_1368 = ((1 - lbound_1364) + ubound_1365) * stride_1366;
  size_1368 = gf_max(size_1368, 0);
  D_8654 = size_1368 + -1;
  D_8655 = (ulong) (ulong) (size_1368) * 32;
  D_8656 = (ulong) (size_1368) * 4;
  offset_1367 = (-(lbound_1359) - lbound_1361 * stride_1363) - lbound_1364 * stride_1366;
  lbound_1302 = (long) *ims;
  ubound_1303 = (long) *ime;
  stride_1306 = (1 - lbound_1302) + ubound_1303;
  stride_1306 = gf_max(stride_1306, 0);
  lbound_1304 = (long) *kms;
  ubound_1305 = (long) *kme;
  stride_1309 = ((1 - lbound_1304) + ubound_1305) * stride_1306;
  stride_1309 = gf_max(stride_1309, 0);
  lbound_1307 = (long) *jms;
  ubound_1308 = (long) *jme;
  size_1311 = ((1 - lbound_1307) + ubound_1308) * stride_1309;
  size_1311 = gf_max(size_1311, 0);
  D_8651 = size_1311 + -1;
  D_8652 = (ulong) (ulong) (size_1311) * 32;
  D_8653 = (ulong) (size_1311) * 4;
  offset_1310 = (-(lbound_1302) - lbound_1304 * stride_1306) - lbound_1307 * stride_1309;
  lbound_1369 = (long) *ims;
  ubound_1370 = (long) *ime;
  stride_1373 = (1 - lbound_1369) + ubound_1370;
  stride_1373 = gf_max(stride_1373, 0);
  lbound_1371 = (long) *kms;
  ubound_1372 = (long) *kme;
  stride_1376 = ((1 - lbound_1371) + ubound_1372) * stride_1373;
  stride_1376 = gf_max(stride_1376, 0);
  lbound_1374 = (long) *jms;
  ubound_1375 = (long) *jme;
  size_1378 = ((1 - lbound_1374) + ubound_1375) * stride_1376;
  size_1378 = gf_max(size_1378, 0);
  D_8648 = size_1378 + -1;
  D_8649 = (ulong) (ulong) (size_1378) * 32;
  D_8650 = (ulong) (size_1378) * 4;
  offset_1377 = (-(lbound_1369) - lbound_1371 * stride_1373) - lbound_1374 * stride_1376;
  lbound_1352 = (long) *ims;
  ubound_1353 = (long) *ime;
  stride_1356 = (1 - lbound_1352) + ubound_1353;
  stride_1356 = gf_max(stride_1356, 0);
  lbound_1354 = (long) *jms;
  ubound_1355 = (long) *jme;
  size_1358 = ((1 - lbound_1354) + ubound_1355) * stride_1356;
  size_1358 = gf_max(size_1358, 0);
  D_8645 = size_1358 + -1;
  D_8646 = (ulong) (ulong) (size_1358) * 32;
  D_8647 = (ulong) (size_1358) * 4;
  offset_1357 = -(lbound_1352) - lbound_1354 * stride_1356;
  lbound_1274 = (long) *ims;
  ubound_1275 = (long) *ime;
  stride_1278 = (1 - lbound_1274) + ubound_1275;
  stride_1278 = gf_max(stride_1278, 0);
  lbound_1276 = (long) *jms;
  ubound_1277 = (long) *jme;
  size_1280 = ((1 - lbound_1276) + ubound_1277) * stride_1278;
  size_1280 = gf_max(size_1280, 0);
  D_8642 = size_1280 + -1;
  D_8643 = (ulong) (ulong) (size_1280) * 32;
  D_8644 = (ulong) (size_1280) * 4;
  offset_1279 = -(lbound_1274) - lbound_1276 * stride_1278;
  lbound_1288 = (long) *ims;
  ubound_1289 = (long) *ime;
  stride_1292 = (1 - lbound_1288) + ubound_1289;
  stride_1292 = gf_max(stride_1292, 0);
  lbound_1290 = (long) *jms;
  ubound_1291 = (long) *jme;
  size_1294 = ((1 - lbound_1290) + ubound_1291) * stride_1292;
  size_1294 = gf_max(size_1294, 0);
  D_8639 = size_1294 + -1;
  D_8640 = (ulong) (ulong) (size_1294) * 32;
  D_8641 = (ulong) (size_1294) * 4;
  offset_1293 = -(lbound_1288) - lbound_1290 * stride_1292;
  lbound_1295 = (long) *ims;
  ubound_1296 = (long) *ime;
  stride_1299 = (1 - lbound_1295) + ubound_1296;
  stride_1299 = gf_max(stride_1299, 0);
  lbound_1297 = (long) *jms;
  ubound_1298 = (long) *jme;
  size_1301 = ((1 - lbound_1297) + ubound_1298) * stride_1299;
  size_1301 = gf_max(size_1301, 0);
  D_8636 = size_1301 + -1;
  D_8637 = (ulong) (ulong) (size_1301) * 32;
  D_8638 = (ulong) (size_1301) * 4;
  offset_1300 = -(lbound_1295) - lbound_1297 * stride_1299;
  lbound_1281 = (long) *ims;
  ubound_1282 = (long) *ime;
  stride_1285 = (1 - lbound_1281) + ubound_1282;
  stride_1285 = gf_max(stride_1285, 0);
  lbound_1283 = (long) *jms;
  ubound_1284 = (long) *jme;
  size_1287 = ((1 - lbound_1283) + ubound_1284) * stride_1285;
  size_1287 = gf_max(size_1287, 0);
  D_8633 = size_1287 + -1;
  D_8634 = (ulong) (ulong) (size_1287) * 32;
  D_8635 = (ulong) (size_1287) * 4;
  offset_1286 = -(lbound_1281) - lbound_1283 * stride_1285;
  lbound_1246 = (long) *ims;
  ubound_1247 = (long) *ime;
  stride_1250 = (1 - lbound_1246) + ubound_1247;
  stride_1250 = gf_max(stride_1250, 0);
  lbound_1248 = (long) *jms;
  ubound_1249 = (long) *jme;
  size_1252 = ((1 - lbound_1248) + ubound_1249) * stride_1250;
  size_1252 = gf_max(size_1252, 0);
  D_8630 = size_1252 + -1;
  D_8631 = (ulong) (ulong) (size_1252) * 32;
  D_8632 = (ulong) (size_1252) * 4;
  offset_1251 = -(lbound_1246) - lbound_1248 * stride_1250;
  lbound_1253 = (long) *ims;
  ubound_1254 = (long) *ime;
  stride_1257 = (1 - lbound_1253) + ubound_1254;
  stride_1257 = gf_max(stride_1257, 0);
  lbound_1255 = (long) *jms;
  ubound_1256 = (long) *jme;
  size_1259 = ((1 - lbound_1255) + ubound_1256) * stride_1257;
  size_1259 = gf_max(size_1259, 0);
  D_8627 = size_1259 + -1;
  D_8628 = (ulong) (ulong) (size_1259) * 32;
  D_8629 = (ulong) (size_1259) * 4;
  offset_1258 = -(lbound_1253) - lbound_1255 * stride_1257;
  lbound_1260 = (long) *ims;
  ubound_1261 = (long) *ime;
  stride_1264 = (1 - lbound_1260) + ubound_1261;
  stride_1264 = gf_max(stride_1264, 0);
  lbound_1262 = (long) *jms;
  ubound_1263 = (long) *jme;
  size_1266 = ((1 - lbound_1262) + ubound_1263) * stride_1264;
  size_1266 = gf_max(size_1266, 0);
  D_8624 = size_1266 + -1;
  D_8625 = (ulong) (ulong) (size_1266) * 32;
  D_8626 = (ulong) (size_1266) * 4;
  offset_1265 = -(lbound_1260) - lbound_1262 * stride_1264;
  lbound_1267 = (long) *ims;
  ubound_1268 = (long) *ime;
  stride_1271 = (1 - lbound_1267) + ubound_1268;
  stride_1271 = gf_max(stride_1271, 0);
  lbound_1269 = (long) *jms;
  ubound_1270 = (long) *jme;
  size_1273 = ((1 - lbound_1269) + ubound_1270) * stride_1271;
  size_1273 = gf_max(size_1273, 0);
  D_8621 = size_1273 + -1;
  D_8622 = (ulong) (ulong) (size_1273) * 32;
  D_8623 = (ulong) (size_1273) * 4;
  offset_1272 = -(lbound_1267) - lbound_1269 * stride_1271;
  {
    int D_8514;
    int M_1419;
    int D_8516;

    D_8514 = *jte;
    M_1419 = D_8514;
    D_8516 = *jde + -1;
    M_1419 = gf_min(D_8516, M_1419);
    jtf = M_1419;
  }
  {
    int D_8517;
    int M_1420;
    int D_8519;

    D_8517 = *kte;
    M_1420 = D_8517;
    D_8519 = *kde + -1;
    M_1420 = gf_min(D_8519, M_1420);
    ktf = M_1420;
  }
  {
    int D_8520;
    int M_1421;
    int D_8522;

    D_8520 = *ite;
    M_1421 = D_8520;
    D_8522 = *ide + -1;
    M_1421 = gf_min(D_8522, M_1421);
    itf = M_1421;
  }
  if (!*restart)
    {
      {
        int D_8524;
        int D_8525;

        D_8524 = *jts;
        D_8525 = *jte;
        j = D_8524;
        while (1)
          {
            {
              bool D_8528;

              D_8528 = j > D_8525;
              if (D_8528) break;
              {
                int D_8529;
                int D_8530;

                D_8529 = *kts;
                D_8530 = *kte;
                k = D_8529;
                while (1)
                  {
                    {
                      bool D_8533;

                      D_8533 = k > D_8530;
                      if (D_8533) break;
                      {
                        int D_8534;
                        int D_8535;

                        D_8534 = *its;
                        D_8535 = *ite;
                        i = D_8534;
                        while (1)
                          {
                            {
                              bool D_8538;

                              D_8538 = i > D_8535;
                              if (D_8538) break;
                              rthcuten[(((long) j * stride_1406 + offset_1407) + (long) k * stride_1403) + (long) i] = 0.0f;
                              rqvcuten[(((long) j * stride_1386 + offset_1387) + (long) k * stride_1383) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
      {
        int D_8539;
        int D_8540;

        D_8539 = *jts;
        D_8540 = *jte;
        j = D_8539;
        while (1)
          {
            {
              bool D_8543;

              D_8543 = j > D_8540;
              if (D_8543) break;
              {
                int D_8544;
                int D_8545;

                D_8544 = *kts;
                D_8545 = *kte;
                k = D_8544;
                while (1)
                  {
                    {
                      bool D_8548;

                      D_8548 = k > D_8545;
                      if (D_8548) break;
                      {
                        int D_8549;
                        int D_8550;

                        D_8549 = *its;
                        D_8550 = *ite;
                        i = D_8549;
                        while (1)
                          {
                            {
                              bool D_8553;

                              D_8553 = i > D_8550;
                              if (D_8553) break;
                              cugd_tten[(((long) j * stride_1339 + offset_1340) + (long) k * stride_1336) + (long) i] = 0.0f;
                              cugd_ttens[(((long) j * stride_1349 + offset_1350) + (long) k * stride_1346) + (long) i] = 0.0f;
                              cugd_qvten[(((long) j * stride_1319 + offset_1320) + (long) k * stride_1316) + (long) i] = 0.0f;
                              cugd_qvtens[(((long) j * stride_1329 + offset_1330) + (long) k * stride_1326) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
      {
        int D_8554;
        int D_8555;

        D_8554 = *jts;
        D_8555 = jtf;
        j = D_8554;
        while (1)
          {
            {
              bool D_8558;

              D_8558 = j > D_8555;
              if (D_8558) break;
              {
                int D_8559;
                int D_8560;

                D_8559 = *kts;
                D_8560 = ktf;
                k = D_8559;
                while (1)
                  {
                    {
                      bool D_8563;

                      D_8563 = k > D_8560;
                      if (D_8563) break;
                      {
                        int D_8564;
                        int D_8565;

                        D_8564 = *its;
                        D_8565 = itf;
                        i = D_8564;
                        while (1)
                          {
                            {
                              bool D_8568;

                              D_8568 = i > D_8565;
                              if (D_8568) break;
                              rthften[(((long) j * stride_1416 + offset_1417) + (long) k * stride_1413) + (long) i] = 0.0f;
                              rqvften[(((long) j * stride_1396 + offset_1397) + (long) k * stride_1393) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
      if (*p_qc >= *p_first_scalar)
        {
          {
            int D_8570;
            int D_8571;

            D_8570 = *jts;
            D_8571 = jtf;
            j = D_8570;
            while (1)
              {
                {
                  bool D_8574;

                  D_8574 = j > D_8571;
                  if (D_8574) break;
                  {
                    int D_8575;
                    int D_8576;

                    D_8575 = *kts;
                    D_8576 = ktf;
                    k = D_8575;
                    while (1)
                      {
                        {
                          bool D_8579;

                          D_8579 = k > D_8576;
                          if (D_8579) break;
                          {
                            int D_8580;
                            int D_8581;

                            D_8580 = *its;
                            D_8581 = itf;
                            i = D_8580;
                            while (1)
                              {
                                {
                                  bool D_8584;

                                  D_8584 = i > D_8581;
                                  if (D_8584) break;
                                  rqccuten[(((long) j * stride_1366 + offset_1367) + (long) k * stride_1363) + (long) i] = 0.0f;
                                  cugd_qcten[(((long) j * stride_1309 + offset_1310) + (long) k * stride_1306) + (long) i] = 0.0f;

                                  i = i + 1;
                                }
                              }

                          }

                          k = k + 1;
                        }
                      }

                  }

                  j = j + 1;
                }
              }

          }
        }

      if (*p_qi >= *p_first_scalar)
        {
          {
            int D_8586;
            int D_8587;

            D_8586 = *jts;
            D_8587 = jtf;
            j = D_8586;
            while (1)
              {
                {
                  bool D_8590;

                  D_8590 = j > D_8587;
                  if (D_8590) break;
                  {
                    int D_8591;
                    int D_8592;

                    D_8591 = *kts;
                    D_8592 = ktf;
                    k = D_8591;
                    while (1)
                      {
                        {
                          bool D_8595;

                          D_8595 = k > D_8592;
                          if (D_8595) break;
                          {
                            int D_8596;
                            int D_8597;

                            D_8596 = *its;
                            D_8597 = itf;
                            i = D_8596;
                            while (1)
                              {
                                {
                                  bool D_8600;

                                  D_8600 = i > D_8597;
                                  if (D_8600) break;
                                  rqicuten[(((long) j * stride_1376 + offset_1377) + (long) k * stride_1373) + (long) i] = 0.0f;

                                  i = i + 1;
                                }
                              }

                          }

                          k = k + 1;
                        }
                      }

                  }

                  j = j + 1;
                }
              }

          }
        }

      {
        int D_8601;
        int D_8602;

        D_8601 = *jts;
        D_8602 = jtf;
        j = D_8601;
        while (1)
          {
            {
              bool D_8605;

              D_8605 = j > D_8602;
              if (D_8605) break;
              {
                int D_8606;
                int D_8607;

                D_8606 = *its;
                D_8607 = itf;
                i = D_8606;
                while (1)
                  {
                    {
                      bool D_8610;

                      D_8610 = i > D_8607;
                      if (D_8610) break;
                      mass_flux[((long) j * stride_1356 + offset_1357) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
      {
        int D_8611;
        int D_8612;

        D_8611 = *jts;
        D_8612 = jtf;
        j = D_8611;
        while (1)
          {
            {
              bool D_8615;

              D_8615 = j > D_8612;
              if (D_8615) break;
              {
                int D_8616;
                int D_8617;

                D_8616 = *its;
                D_8617 = itf;
                i = D_8616;
                while (1)
                  {
                    {
                      bool D_8620;

                      D_8620 = i > D_8617;
                      if (D_8620) break;
                      apr_gr[((long) j * stride_1278 + offset_1279) + (long) i] = 0.0f;
                      apr_st[((long) j * stride_1292 + offset_1293) + (long) i] = 0.0f;
                      apr_w[((long) j * stride_1299 + offset_1300) + (long) i] = 0.0f;
                      apr_mc[((long) j * stride_1285 + offset_1286) + (long) i] = 0.0f;
                      apr_as[((long) j * stride_1250 + offset_1251) + (long) i] = 0.0f;
                      apr_capma[((long) j * stride_1257 + offset_1258) + (long) i] = 0.0f;
                      apr_capme[((long) j * stride_1264 + offset_1265) + (long) i] = 0.0f;
                      apr_capmi[((long) j * stride_1271 + offset_1272) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
    }

}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17> static inline void cup_up_aa0(thread GFWorkspace &workspace, P0 aa0, P1 z, P2 zu, P3 dby, P4 gamma_cup, P5 t_cup, P6 kbcon, P7 ktop, P8 ierr, P9 itf, P10 jtf, P11 ktf, P12 its, P13 ite, P14 jts, P15 jte, P16 kts, P17 kte) {
  GFFrame frame(workspace);
  long lbound_1422;
  long ubound_1423;
  long offset_1424;
  long size_1425;
  GFReal da;
  long lbound_1426;
  long ubound_1427;
  long lbound_1428;
  long ubound_1429;
  long stride_1430;
  long offset_1431;
  long size_1432;
  GFReal dz;
  long lbound_1433;
  long ubound_1434;
  long lbound_1435;
  long ubound_1436;
  long stride_1437;
  long offset_1438;
  long size_1439;
  int i;
  long lbound_1440;
  long ubound_1441;
  long offset_1442;
  long size_1443;
  int k;
  long lbound_1444;
  long ubound_1445;
  long offset_1446;
  long size_1447;
  long lbound_1448;
  long ubound_1449;
  long offset_1450;
  long size_1451;
  long lbound_1452;
  long ubound_1453;
  long lbound_1454;
  long ubound_1455;
  long stride_1456;
  long offset_1457;
  long size_1458;
  long lbound_1459;
  long ubound_1460;
  long lbound_1461;
  long ubound_1462;
  long stride_1463;
  long offset_1464;
  long size_1465;
  long lbound_1466;
  long ubound_1467;
  long lbound_1468;
  long ubound_1469;
  long stride_1470;
  long offset_1471;
  long size_1472;
  long D_8766;
  ulong D_8767;
  ulong D_8768;
  long D_8769;
  ulong D_8770;
  ulong D_8771;
  long D_8772;
  ulong D_8773;
  ulong D_8774;
  long D_8775;
  ulong D_8776;
  ulong D_8777;
  long D_8778;
  ulong D_8779;
  ulong D_8780;
  long D_8781;
  ulong D_8782;
  ulong D_8783;
  long D_8784;
  ulong D_8785;
  ulong D_8786;
  long D_8787;
  ulong D_8788;
  ulong D_8789;
  long D_8790;
  ulong D_8791;
  ulong D_8792;

  lbound_1422 = (long) *its;
  ubound_1423 = (long) *ite;
  size_1425 = (1 - lbound_1422) + ubound_1423;
  size_1425 = gf_max(size_1425, 0);
  D_8790 = size_1425 + -1;
  D_8791 = (ulong) (ulong) (size_1425) * 32;
  D_8792 = (ulong) (size_1425) * 4;
  offset_1424 = -(lbound_1422);
  lbound_1440 = (long) *its;
  ubound_1441 = (long) *ite;
  size_1443 = (1 - lbound_1440) + ubound_1441;
  size_1443 = gf_max(size_1443, 0);
  D_8787 = size_1443 + -1;
  D_8788 = (ulong) (ulong) (size_1443) * 32;
  D_8789 = (ulong) (size_1443) * 4;
  offset_1442 = -(lbound_1440);
  lbound_1444 = (long) *its;
  ubound_1445 = (long) *ite;
  size_1447 = (1 - lbound_1444) + ubound_1445;
  size_1447 = gf_max(size_1447, 0);
  D_8784 = size_1447 + -1;
  D_8785 = (ulong) (ulong) (size_1447) * 32;
  D_8786 = (ulong) (size_1447) * 4;
  offset_1446 = -(lbound_1444);
  lbound_1448 = (long) *its;
  ubound_1449 = (long) *ite;
  size_1451 = (1 - lbound_1448) + ubound_1449;
  size_1451 = gf_max(size_1451, 0);
  D_8781 = size_1451 + -1;
  D_8782 = (ulong) (ulong) (size_1451) * 32;
  D_8783 = (ulong) (size_1451) * 4;
  offset_1450 = -(lbound_1448);
  lbound_1459 = (long) *its;
  ubound_1460 = (long) *ite;
  stride_1463 = (1 - lbound_1459) + ubound_1460;
  stride_1463 = gf_max(stride_1463, 0);
  lbound_1461 = (long) *kts;
  ubound_1462 = (long) *kte;
  size_1465 = ((1 - lbound_1461) + ubound_1462) * stride_1463;
  size_1465 = gf_max(size_1465, 0);
  D_8778 = size_1465 + -1;
  D_8779 = (ulong) (ulong) (size_1465) * 32;
  D_8780 = (ulong) (size_1465) * 4;
  offset_1464 = -(lbound_1459) - lbound_1461 * stride_1463;
  lbound_1466 = (long) *its;
  ubound_1467 = (long) *ite;
  stride_1470 = (1 - lbound_1466) + ubound_1467;
  stride_1470 = gf_max(stride_1470, 0);
  lbound_1468 = (long) *kts;
  ubound_1469 = (long) *kte;
  size_1472 = ((1 - lbound_1468) + ubound_1469) * stride_1470;
  size_1472 = gf_max(size_1472, 0);
  D_8775 = size_1472 + -1;
  D_8776 = (ulong) (ulong) (size_1472) * 32;
  D_8777 = (ulong) (size_1472) * 4;
  offset_1471 = -(lbound_1466) - lbound_1468 * stride_1470;
  lbound_1452 = (long) *its;
  ubound_1453 = (long) *ite;
  stride_1456 = (1 - lbound_1452) + ubound_1453;
  stride_1456 = gf_max(stride_1456, 0);
  lbound_1454 = (long) *kts;
  ubound_1455 = (long) *kte;
  size_1458 = ((1 - lbound_1454) + ubound_1455) * stride_1456;
  size_1458 = gf_max(size_1458, 0);
  D_8772 = size_1458 + -1;
  D_8773 = (ulong) (ulong) (size_1458) * 32;
  D_8774 = (ulong) (size_1458) * 4;
  offset_1457 = -(lbound_1452) - lbound_1454 * stride_1456;
  lbound_1426 = (long) *its;
  ubound_1427 = (long) *ite;
  stride_1430 = (1 - lbound_1426) + ubound_1427;
  stride_1430 = gf_max(stride_1430, 0);
  lbound_1428 = (long) *kts;
  ubound_1429 = (long) *kte;
  size_1432 = ((1 - lbound_1428) + ubound_1429) * stride_1430;
  size_1432 = gf_max(size_1432, 0);
  D_8769 = size_1432 + -1;
  D_8770 = (ulong) (ulong) (size_1432) * 32;
  D_8771 = (ulong) (size_1432) * 4;
  offset_1431 = -(lbound_1426) - lbound_1428 * stride_1430;
  lbound_1433 = (long) *its;
  ubound_1434 = (long) *ite;
  stride_1437 = (1 - lbound_1433) + ubound_1434;
  stride_1437 = gf_max(stride_1437, 0);
  lbound_1435 = (long) *kts;
  ubound_1436 = (long) *kte;
  size_1439 = ((1 - lbound_1435) + ubound_1436) * stride_1437;
  size_1439 = gf_max(size_1439, 0);
  D_8766 = size_1439 + -1;
  D_8767 = (ulong) (ulong) (size_1439) * 32;
  D_8768 = (ulong) (size_1439) * 4;
  offset_1438 = -(lbound_1433) - lbound_1435 * stride_1437;
  {
    int D_8745;
    int D_8746;

    D_8745 = *its;
    D_8746 = *itf;
    i = D_8745;
    while (1)
      {
        {
          bool D_8749;

          D_8749 = i > D_8746;
          if (D_8749) break;
          aa0[(long) i + offset_1424] = 0.0f;

          i = i + 1;
        }
      }

  }
  {
    int D_8750;
    int D_8751;

    D_8750 = *kts + 1;
    D_8751 = *ktf;
    k = D_8750;
    while (1)
      {
        {
          bool D_8754;

          D_8754 = k > D_8751;
          if (D_8754) break;
          {
            int D_8755;
            int D_8756;

            D_8755 = *its;
            D_8756 = *itf;
            i = D_8755;
            while (1)
              {
                {
                  bool D_8759;

                  D_8759 = i > D_8756;
                  if (D_8759) break;
                  do {
if (ierr[(long) i + offset_1442] != 0) break;

                  if (kbcon[(long) i + offset_1446] >= k) break;

                  if (ktop[(long) i + offset_1450] < k) break;

                  dz = z[((long) k * stride_1463 + offset_1464) + (long) i] - z[((long) (k + -1) * stride_1463 + offset_1464) + (long) i];
                  da = (((zu[((long) k * stride_1470 + offset_1471) + (long) i] * dz) * ((9.81000041961669921875e+0f / ((((t_cup[((long) k * stride_1456 + offset_1457) + (long) i])) * 1.004e+3f))))) * dby[((long) (k + -1) * stride_1430 + offset_1431) + (long) i]) / ((gamma_cup[((long) k * stride_1437 + offset_1438) + (long) i] + 1.0e+0f));
                  if (ktop[(long) i + offset_1450] == k && da <= 0.0f) break;

                  aa0[(long) i + offset_1424] = aa0[(long) i + offset_1424] + da;
                  if (aa0[(long) i + offset_1424] < 0.0f)
                    {
                      aa0[(long) i + offset_1424] = 0.0f;
                    }

                  } while(false);

                  i = i + 1;
                }
              }

          }

          k = k + 1;
        }
      }

  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13> static inline void cup_minimi(thread GFWorkspace &workspace, P0 array, P1 ks, P2 kend, P3 kt, P4 ierr, P5 itf, P6 jtf, P7 ktf, P8 its, P9 ite, P10 jts, P11 jte, P12 kts, P13 kte) {
  GFFrame frame(workspace);

  device void * D_8850;
  long lbound_1473;
  long ubound_1474;
  long lbound_1475;
  long ubound_1476;
  long stride_1477;
  long offset_1478;
  long size_1479;
  int i;
  long lbound_1480;
  long ubound_1481;
  long offset_1482;
  long size_1483;
  int k;
  long lbound_1484;
  long ubound_1485;
  long offset_1486;
  long size_1487;
  long lbound_1488;
  long ubound_1489;
  long offset_1490;
  long size_1491;
  int kstop;
  long lbound_1492;
  long ubound_1493;
  long offset_1494;
  long size_1495;
  long lbound_1496;
  long ubound_1497;
  long offset_1498;
  long size_1499;

  device GFReal * x;
  long D_8847;
  ulong D_8848;
  ulong D_8849;
  long D_8851;
  ulong D_8852;
  ulong D_8853;
  long D_8854;
  ulong D_8855;
  ulong D_8856;
  long D_8857;
  ulong D_8858;
  ulong D_8859;
  long D_8860;
  ulong D_8861;
  ulong D_8862;
  long D_8863;
  ulong D_8864;
  ulong D_8865;

    {
      lbound_1492 = (long) *its;
      ubound_1493 = (long) *ite;
      size_1495 = (1 - lbound_1492) + ubound_1493;
      size_1495 = gf_max(size_1495, 0);
      D_8863 = size_1495 + -1;
      D_8864 = (ulong) (ulong) (size_1495) * 32;
      D_8865 = (ulong) (size_1495) * 4;
      offset_1494 = -(lbound_1492);
      lbound_1488 = (long) *its;
      ubound_1489 = (long) *ite;
      size_1491 = (1 - lbound_1488) + ubound_1489;
      size_1491 = gf_max(size_1491, 0);
      D_8860 = size_1491 + -1;
      D_8861 = (ulong) (ulong) (size_1491) * 32;
      D_8862 = (ulong) (size_1491) * 4;
      offset_1490 = -(lbound_1488);
      lbound_1480 = (long) *its;
      ubound_1481 = (long) *ite;
      size_1483 = (1 - lbound_1480) + ubound_1481;
      size_1483 = gf_max(size_1483, 0);
      D_8857 = size_1483 + -1;
      D_8858 = (ulong) (ulong) (size_1483) * 32;
      D_8859 = (ulong) (size_1483) * 4;
      offset_1482 = -(lbound_1480);
      lbound_1473 = (long) *its;
      ubound_1474 = (long) *ite;
      stride_1477 = (1 - lbound_1473) + ubound_1474;
      stride_1477 = gf_max(stride_1477, 0);
      lbound_1475 = (long) *kts;
      ubound_1476 = (long) *kte;
      size_1479 = ((1 - lbound_1475) + ubound_1476) * stride_1477;
      size_1479 = gf_max(size_1479, 0);
      D_8854 = size_1479 + -1;
      D_8855 = (ulong) (ulong) (size_1479) * 32;
      D_8856 = (ulong) (size_1479) * 4;
      offset_1478 = -(lbound_1473) - lbound_1475 * stride_1477;
      lbound_1484 = (long) *its;
      ubound_1485 = (long) *ite;
      size_1487 = (1 - lbound_1484) + ubound_1485;
      size_1487 = gf_max(size_1487, 0);
      D_8851 = size_1487 + -1;
      D_8852 = (ulong) (ulong) (size_1487) * 32;
      D_8853 = (ulong) (size_1487) * 4;
      offset_1486 = -(lbound_1484);
      lbound_1496 = (long) *its;
      ubound_1497 = (long) *ite;
      size_1499 = (1 - lbound_1496) + ubound_1497;
      size_1499 = gf_max(size_1499, 0);
      D_8847 = size_1499 + -1;
      D_8848 = (ulong) (ulong) (size_1499) * 32;
      D_8849 = (ulong) (size_1499) * 4;
      D_8850 = (device void *) workspace.allocate (gf_max((ulong) (size_1499 * 4), 1));
      x = (device GFReal *) D_8850;
      offset_1498 = -(lbound_1496);
      {
        int D_8830;
        int D_8831;

        D_8830 = *its;
        D_8831 = *itf;
        i = D_8830;
        while (1)
          {
            {
              bool D_8834;

              D_8834 = i > D_8831;
              if (D_8834) break;
              kt[(long) i + offset_1494] = ks[(long) i + offset_1490];
              if (ierr[(long) i + offset_1482] == 0)
                {
                  x[(long) i + offset_1498] = array[((long) ks[(long) i + offset_1490] * stride_1477 + offset_1478) + (long) i];
                  {
                    int D_8836;
                    int M_1500;
                    int D_8838;

                    D_8836 = ks[(long) i + offset_1490] + 1;
                    M_1500 = D_8836;
                    D_8838 = kend[(long) i + offset_1486];
                    M_1500 = gf_max(D_8838, M_1500);
                    kstop = M_1500;
                  }
                  {
                    int D_8839;
                    int D_8840;

                    D_8839 = ks[(long) i + offset_1490] + 1;
                    D_8840 = kstop;
                    k = D_8839;
                    while (1)
                      {
                        {
                          bool D_8843;

                          D_8843 = k > D_8840;
                          if (D_8843) break;
                          if (array[((long) k * stride_1477 + offset_1478) + (long) i] < x[(long) i + offset_1498])
                            {
                              x[(long) i + offset_1498] = array[((long) k * stride_1477 + offset_1478) + (long) i];
                              kt[(long) i + offset_1494] = k;
                            }



                          k = k + 1;
                        }
                      }

                  }
                }



              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13> static inline void cup_maximi(thread GFWorkspace &workspace, P0 array, P1 ks, P2 ke, P3 maxx, P4 ierr, P5 itf, P6 jtf, P7 ktf, P8 its, P9 ite, P10 jts, P11 jte, P12 kts, P13 kte) {
  GFFrame frame(workspace);

  device void * D_8915;
  long lbound_1501;
  long ubound_1502;
  long lbound_1503;
  long ubound_1504;
  long stride_1505;
  long offset_1506;
  long size_1507;
  int i;
  long lbound_1508;
  long ubound_1509;
  long offset_1510;
  long size_1511;
  int k;
  long lbound_1512;
  long ubound_1513;
  long offset_1514;
  long size_1515;
  long lbound_1516;
  long ubound_1517;
  long offset_1518;
  long size_1519;
  long lbound_1520;
  long ubound_1521;
  long offset_1522;
  long size_1523;

  device GFReal * x;
  GFReal xar;
  long D_8912;
  ulong D_8913;
  ulong D_8914;
  long D_8916;
  ulong D_8917;
  ulong D_8918;
  long D_8919;
  ulong D_8920;
  ulong D_8921;
  long D_8922;
  ulong D_8923;
  ulong D_8924;
  long D_8925;
  ulong D_8926;
  ulong D_8927;

    {
      lbound_1516 = (long) *its;
      ubound_1517 = (long) *ite;
      size_1519 = (1 - lbound_1516) + ubound_1517;
      size_1519 = gf_max(size_1519, 0);
      D_8925 = size_1519 + -1;
      D_8926 = (ulong) (ulong) (size_1519) * 32;
      D_8927 = (ulong) (size_1519) * 4;
      offset_1518 = -(lbound_1516);
      lbound_1508 = (long) *its;
      ubound_1509 = (long) *ite;
      size_1511 = (1 - lbound_1508) + ubound_1509;
      size_1511 = gf_max(size_1511, 0);
      D_8922 = size_1511 + -1;
      D_8923 = (ulong) (ulong) (size_1511) * 32;
      D_8924 = (ulong) (size_1511) * 4;
      offset_1510 = -(lbound_1508);
      lbound_1501 = (long) *its;
      ubound_1502 = (long) *ite;
      stride_1505 = (1 - lbound_1501) + ubound_1502;
      stride_1505 = gf_max(stride_1505, 0);
      lbound_1503 = (long) *kts;
      ubound_1504 = (long) *kte;
      size_1507 = ((1 - lbound_1503) + ubound_1504) * stride_1505;
      size_1507 = gf_max(size_1507, 0);
      D_8919 = size_1507 + -1;
      D_8920 = (ulong) (ulong) (size_1507) * 32;
      D_8921 = (ulong) (size_1507) * 4;
      offset_1506 = -(lbound_1501) - lbound_1503 * stride_1505;
      lbound_1512 = (long) *its;
      ubound_1513 = (long) *ite;
      size_1515 = (1 - lbound_1512) + ubound_1513;
      size_1515 = gf_max(size_1515, 0);
      D_8916 = size_1515 + -1;
      D_8917 = (ulong) (ulong) (size_1515) * 32;
      D_8918 = (ulong) (size_1515) * 4;
      offset_1514 = -(lbound_1512);
      lbound_1520 = (long) *its;
      ubound_1521 = (long) *ite;
      size_1523 = (1 - lbound_1520) + ubound_1521;
      size_1523 = gf_max(size_1523, 0);
      D_8912 = size_1523 + -1;
      D_8913 = (ulong) (ulong) (size_1523) * 32;
      D_8914 = (ulong) (size_1523) * 4;
      D_8915 = (device void *) workspace.allocate (gf_max((ulong) (size_1523 * 4), 1));
      x = (device GFReal *) D_8915;
      offset_1522 = -(lbound_1520);
      {
        int D_8898;
        int D_8899;

        D_8898 = *its;
        D_8899 = *itf;
        i = D_8898;
        while (1)
          {
            {
              bool D_8902;

              D_8902 = i > D_8899;
              if (D_8902) break;
              maxx[(long) i + offset_1518] = *ks;
              if (ierr[(long) i + offset_1510] == 0)
                {
                  x[(long) i + offset_1522] = array[((long) *ks * stride_1505 + offset_1506) + (long) i];
                  {
                    int D_8904;
                    int D_8905;

                    D_8904 = *ks;
                    D_8905 = ke[(long) i + offset_1514];
                    k = D_8904;
                    while (1)
                      {
                        {
                          bool D_8908;

                          D_8908 = k > D_8905;
                          if (D_8908) break;
                          xar = array[((long) k * stride_1505 + offset_1506) + (long) i];
                          if (x[(long) i + offset_1522] <= xar)
                            {
                              x[(long) i + offset_1522] = xar;
                              maxx[(long) i + offset_1518] = k;
                            }



                          k = k + 1;
                        }
                      }

                  }
                }



              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14> static inline void cup_ktop(thread GFWorkspace &workspace, P0 ierrc, P1 ilo, P2 dby, P3 kbcon, P4 ktop, P5 ierr, P6 itf, P7 jtf, P8 ktf, P9 its, P10 ite, P11 jts, P12 jte, P13 kts, P14 kte, long _ierrc) {
  GFFrame frame(workspace);
  long lbound_1524;
  long ubound_1525;
  long lbound_1526;
  long ubound_1527;
  long stride_1528;
  long offset_1529;
  long size_1530;
  int i;
  long lbound_1531;
  long ubound_1532;
  long offset_1533;
  long size_1534;
  long lbound_1535;
  long ubound_1536;
  long offset_1537;
  long size_1538;
  int k;
  long lbound_1539;
  long ubound_1540;
  long offset_1541;
  long size_1542;
  long lbound_1543;
  long ubound_1544;
  long offset_1545;
  long size_1546;
  long D_8981;
  ulong D_8982;
  ulong D_8983;
  long D_8984;
  ulong D_8985;
  ulong D_8986;
  long D_8987;
  ulong D_8988;
  ulong D_8989;
  long D_8990;
  ulong D_8991;
  ulong D_8992;
  long D_8993;
  ulong D_8994;
  ulong D_8995;

  lbound_1543 = (long) *its;
  ubound_1544 = (long) *ite;
  size_1546 = (1 - lbound_1543) + ubound_1544;
  size_1546 = gf_max(size_1546, 0);
  D_8993 = size_1546 + -1;
  D_8994 = (ulong) (ulong) (size_1546) * 32;
  D_8995 = (ulong) (size_1546) * 4;
  offset_1545 = -(lbound_1543);
  lbound_1531 = (long) *its;
  ubound_1532 = (long) *ite;
  size_1534 = (1 - lbound_1531) + ubound_1532;
  size_1534 = gf_max(size_1534, 0);
  D_8990 = size_1534 + -1;
  D_8991 = (ulong) (ulong) (size_1534) * 32;
  D_8992 = (ulong) (size_1534) * 4;
  offset_1533 = -(lbound_1531);
  lbound_1539 = (long) *its;
  ubound_1540 = (long) *ite;
  size_1542 = (1 - lbound_1539) + ubound_1540;
  size_1542 = gf_max(size_1542, 0);
  D_8987 = size_1542 + -1;
  D_8988 = (ulong) (ulong) (size_1542) * 32;
  D_8989 = (ulong) (size_1542) * 4;
  offset_1541 = -(lbound_1539);
  lbound_1524 = (long) *its;
  ubound_1525 = (long) *ite;
  stride_1528 = (1 - lbound_1524) + ubound_1525;
  stride_1528 = gf_max(stride_1528, 0);
  lbound_1526 = (long) *kts;
  ubound_1527 = (long) *kte;
  size_1530 = ((1 - lbound_1526) + ubound_1527) * stride_1528;
  size_1530 = gf_max(size_1530, 0);
  D_8984 = size_1530 + -1;
  D_8985 = (ulong) (ulong) (size_1530) * 32;
  D_8986 = (ulong) (size_1530) * 4;
  offset_1529 = -(lbound_1524) - lbound_1526 * stride_1528;
  lbound_1535 = (long) *its;
  ubound_1536 = (long) *ite;
  size_1538 = (1 - lbound_1535) + ubound_1536;
  size_1538 = gf_max(size_1538, 0);
  D_8981 = size_1538 + -1;
  D_8982 = (ulong) (ulong) (size_1538) * 400;
  D_8983 = (ulong) (size_1538) * 50;
  offset_1537 = -(lbound_1535);
  {
    int D_8958;
    int D_8959;

    D_8958 = *its;
    D_8959 = *itf;
    i = D_8958;
    while (1)
      {
        {
          bool D_8962;

          D_8962 = i > D_8959;
          if (D_8962) break;
          ktop[(long) i + offset_1545] = 1;
          if (ierr[(long) i + offset_1533] == 0)
            {
 bool gf_found=false;
              {
                int D_8964;
                int D_8965;

                D_8964 = kbcon[(long) i + offset_1541] + 1;
                D_8965 = *ktf + -1;
                k = D_8964;
                while (1)
                  {
                    {
                      bool D_8968;

                      D_8968 = k > D_8965;
                      if (D_8968) break;
                      if (dby[((long) k * stride_1528 + offset_1529) + (long) i] <= 0.0f)
                        {
                          ktop[(long) i + offset_1545] = k + -1;
                          gf_found=true; break;
                        }



                      k = k + 1;
                    }
                  }

              }
              if (!gf_found) {
if (*ilo == 1)
                {
                  ierr[(long) i + offset_1533] = 5;
                }

              if (*ilo == 1)
                {
                  gf_memmove ((device char *)&ierrc[(long) i + offset_1537], "problem with defining ktop", 26);
                  gf_memset ((device char *)&ierrc[(long) i + offset_1537] + 26, 32, 24);
                }

              } else {
              {
                int D_8975;
                int D_8976;

                D_8975 = ktop[(long) i + offset_1545] + 1;
                D_8976 = *ktf;
                k = D_8975;
                while (1)
                  {
                    {
                      bool D_8979;

                      D_8979 = k > D_8976;
                      if (D_8979) break;
                      dby[((long) k * stride_1528 + offset_1529) + (long) i] = 0.0f;

                      k = k + 1;
                    }
                  }

              }
              if (kbcon[(long) i + offset_1541] == ktop[(long) i + offset_1545])
                {
                  ierr[(long) i + offset_1533] = 55;
                  gf_memmove ((device char *)&ierrc[(long) i + offset_1537], "kbcon == ktop ", 14);
                  gf_memset ((device char *)&ierrc[(long) i + offset_1537] + 14, 32, 36);
                }

 }
            }



          i = i + 1;
        }
      }

  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25> static inline void cup_kbcon(thread GFWorkspace &workspace, P0 ierrc, P1 cap_inc, P2 iloop, P3 k22, P4 kbcon, P5 he_cup, P6 hes_cup, P7 hkb, P8 ierr, P9 kbmax, P10 p_cup, P11 cap_max, P12 xl, P13 cp, P14 ztexec, P15 zqexec, P16 use_excess, P17 itf, P18 jtf, P19 ktf, P20 its, P21 ite, P22 jts, P23 jte, P24 kts, P25 kte, long _ierrc) {
  GFFrame frame(workspace);
  long lbound_1547;
  long ubound_1548;
  long offset_1549;
  long size_1550;
  long lbound_1551;
  long ubound_1552;
  long offset_1553;
  long size_1554;
  long lbound_1555;
  long ubound_1556;
  long lbound_1557;
  long ubound_1558;
  long stride_1559;
  long offset_1560;
  long size_1561;
  long lbound_1562;
  long ubound_1563;
  long lbound_1564;
  long ubound_1565;
  long stride_1566;
  long offset_1567;
  long size_1568;
  GFReal hetest;
  long lbound_1569;
  long ubound_1570;
  long offset_1571;
  long size_1572;
  int i;
  long lbound_1573;
  long ubound_1574;
  long offset_1575;
  long size_1576;
  long lbound_1577;
  long ubound_1578;
  long offset_1579;
  long size_1580;
  int k1;
  int k2;
  long lbound_1581;
  long ubound_1582;
  long offset_1583;
  long size_1584;
  long lbound_1585;
  long ubound_1586;
  long offset_1587;
  long size_1588;
  long lbound_1589;
  long ubound_1590;
  long offset_1591;
  long size_1592;
  long lbound_1593;
  long ubound_1594;
  long lbound_1595;
  long ubound_1596;
  long stride_1597;
  long offset_1598;
  long size_1599;
  GFReal pbcdif;
  GFReal plus;
  long lbound_1600;
  long ubound_1601;
  long offset_1602;
  long size_1603;
  long lbound_1604;
  long ubound_1605;
  long offset_1606;
  long size_1607;
  long D_9120;
  ulong D_9121;
  ulong D_9122;
  long D_9123;
  ulong D_9124;
  ulong D_9125;
  long D_9126;
  ulong D_9127;
  ulong D_9128;
  long D_9129;
  ulong D_9130;
  ulong D_9131;
  long D_9132;
  ulong D_9133;
  ulong D_9134;
  long D_9135;
  ulong D_9136;
  ulong D_9137;
  long D_9138;
  ulong D_9139;
  ulong D_9140;
  long D_9141;
  ulong D_9142;
  ulong D_9143;
  long D_9144;
  ulong D_9145;
  ulong D_9146;
  long D_9147;
  ulong D_9148;
  ulong D_9149;
  long D_9150;
  ulong D_9151;
  ulong D_9152;
  long D_9153;
  ulong D_9154;
  ulong D_9155;
  long D_9156;
  ulong D_9157;
  ulong D_9158;

  lbound_1585 = (long) *its;
  ubound_1586 = (long) *ite;
  size_1588 = (1 - lbound_1585) + ubound_1586;
  size_1588 = gf_max(size_1588, 0);
  D_9156 = size_1588 + -1;
  D_9157 = (ulong) (ulong) (size_1588) * 32;
  D_9158 = (ulong) (size_1588) * 4;
  offset_1587 = -(lbound_1585);
  lbound_1573 = (long) *its;
  ubound_1574 = (long) *ite;
  size_1576 = (1 - lbound_1573) + ubound_1574;
  size_1576 = gf_max(size_1576, 0);
  D_9153 = size_1576 + -1;
  D_9154 = (ulong) (ulong) (size_1576) * 32;
  D_9155 = (ulong) (size_1576) * 4;
  offset_1575 = -(lbound_1573);
  lbound_1581 = (long) *its;
  ubound_1582 = (long) *ite;
  size_1584 = (1 - lbound_1581) + ubound_1582;
  size_1584 = gf_max(size_1584, 0);
  D_9150 = size_1584 + -1;
  D_9151 = (ulong) (ulong) (size_1584) * 32;
  D_9152 = (ulong) (size_1584) * 4;
  offset_1583 = -(lbound_1581);
  lbound_1589 = (long) *its;
  ubound_1590 = (long) *ite;
  size_1592 = (1 - lbound_1589) + ubound_1590;
  size_1592 = gf_max(size_1592, 0);
  D_9147 = size_1592 + -1;
  D_9148 = (ulong) (ulong) (size_1592) * 32;
  D_9149 = (ulong) (size_1592) * 4;
  offset_1591 = -(lbound_1589);
  lbound_1577 = (long) *its;
  ubound_1578 = (long) *ite;
  size_1580 = (1 - lbound_1577) + ubound_1578;
  size_1580 = gf_max(size_1580, 0);
  D_9144 = size_1580 + -1;
  D_9145 = (ulong) (ulong) (size_1580) * 400;
  D_9146 = (ulong) (size_1580) * 50;
  offset_1579 = -(lbound_1577);
  lbound_1569 = (long) *its;
  ubound_1570 = (long) *ite;
  size_1572 = (1 - lbound_1569) + ubound_1570;
  size_1572 = gf_max(size_1572, 0);
  D_9141 = size_1572 + -1;
  D_9142 = (ulong) (ulong) (size_1572) * 32;
  D_9143 = (ulong) (size_1572) * 4;
  offset_1571 = -(lbound_1569);
  lbound_1562 = (long) *its;
  ubound_1563 = (long) *ite;
  stride_1566 = (1 - lbound_1562) + ubound_1563;
  stride_1566 = gf_max(stride_1566, 0);
  lbound_1564 = (long) *kts;
  ubound_1565 = (long) *kte;
  size_1568 = ((1 - lbound_1564) + ubound_1565) * stride_1566;
  size_1568 = gf_max(size_1568, 0);
  D_9138 = size_1568 + -1;
  D_9139 = (ulong) (ulong) (size_1568) * 32;
  D_9140 = (ulong) (size_1568) * 4;
  offset_1567 = -(lbound_1562) - lbound_1564 * stride_1566;
  lbound_1593 = (long) *its;
  ubound_1594 = (long) *ite;
  stride_1597 = (1 - lbound_1593) + ubound_1594;
  stride_1597 = gf_max(stride_1597, 0);
  lbound_1595 = (long) *kts;
  ubound_1596 = (long) *kte;
  size_1599 = ((1 - lbound_1595) + ubound_1596) * stride_1597;
  size_1599 = gf_max(size_1599, 0);
  D_9135 = size_1599 + -1;
  D_9136 = (ulong) (ulong) (size_1599) * 32;
  D_9137 = (ulong) (size_1599) * 4;
  offset_1598 = -(lbound_1593) - lbound_1595 * stride_1597;
  lbound_1551 = (long) *its;
  ubound_1552 = (long) *ite;
  size_1554 = (1 - lbound_1551) + ubound_1552;
  size_1554 = gf_max(size_1554, 0);
  D_9132 = size_1554 + -1;
  D_9133 = (ulong) (ulong) (size_1554) * 32;
  D_9134 = (ulong) (size_1554) * 4;
  offset_1553 = -(lbound_1551);
  lbound_1547 = (long) *its;
  ubound_1548 = (long) *ite;
  size_1550 = (1 - lbound_1547) + ubound_1548;
  size_1550 = gf_max(size_1550, 0);
  D_9129 = size_1550 + -1;
  D_9130 = (ulong) (ulong) (size_1550) * 32;
  D_9131 = (ulong) (size_1550) * 4;
  offset_1549 = -(lbound_1547);
  lbound_1555 = (long) *its;
  ubound_1556 = (long) *ite;
  stride_1559 = (1 - lbound_1555) + ubound_1556;
  stride_1559 = gf_max(stride_1559, 0);
  lbound_1557 = (long) *kts;
  ubound_1558 = (long) *kte;
  size_1561 = ((1 - lbound_1557) + ubound_1558) * stride_1559;
  size_1561 = gf_max(size_1561, 0);
  D_9126 = size_1561 + -1;
  D_9127 = (ulong) (ulong) (size_1561) * 32;
  D_9128 = (ulong) (size_1561) * 4;
  offset_1560 = -(lbound_1555) - lbound_1557 * stride_1559;
  lbound_1600 = (long) *its;
  ubound_1601 = (long) *ite;
  size_1603 = (1 - lbound_1600) + ubound_1601;
  size_1603 = gf_max(size_1603, 0);
  D_9123 = size_1603 + -1;
  D_9124 = (ulong) (ulong) (size_1603) * 32;
  D_9125 = (ulong) (size_1603) * 4;
  offset_1602 = -(lbound_1600);
  lbound_1604 = (long) *its;
  ubound_1605 = (long) *ite;
  size_1607 = (1 - lbound_1604) + ubound_1605;
  size_1607 = gf_max(size_1607, 0);
  D_9120 = size_1607 + -1;
  D_9121 = (ulong) (ulong) (size_1607) * 32;
  D_9122 = (ulong) (size_1607) * 4;
  offset_1606 = -(lbound_1604);
  {
    int D_9076;
    int D_9077;

    D_9076 = *its;
    D_9077 = *itf;
    i = D_9076;
    while (1)
      {
        {
          bool D_9080;

          D_9080 = i > D_9077;
          if (D_9080) break;
          kbcon[(long) i + offset_1587] = 1;
          if (ierr[(long) i + offset_1575] == 0) {

          kbcon[(long) i + offset_1587] = k22[(long) i + offset_1583] + 1;
          if (*iloop == 5)
            {
              kbcon[(long) i + offset_1587] = k22[(long) i + offset_1583];
            }

          bool gf_first=true; while (true) { if (!gf_first) {
          kbcon[(long) i + offset_1587] = kbcon[(long) i + offset_1587] + 1;
          if (kbcon[(long) i + offset_1587] > kbmax[(long) i + offset_1591] + 2)
            {
              if (*iloop != 4)
                {
                  ierr[(long) i + offset_1575] = 3;
                  gf_memmove ((device char *)&ierrc[(long) i + offset_1579], "could not find reasonable kbcon in cup_kbcon", 44);
                  gf_memset ((device char *)&ierrc[(long) i + offset_1579] + 44, 32, 6);
                }

              break;
            }

          } gf_first=false;
          hetest = hkb[(long) i + offset_1571];
          if (*iloop == 5)
            {
              hetest = hkb[(long) i + offset_1571];
            }

          if (hes_cup[((long) kbcon[(long) i + offset_1587] * stride_1566 + offset_1567) + (long) i] > hetest) continue;

          if (kbcon[(long) i + offset_1587] - k22[(long) i + offset_1583] == 1) break;

          if (*iloop == 5 && kbcon[(long) i + offset_1587] == k22[(long) i + offset_1583]) break;

          pbcdif = p_cup[((long) k22[(long) i + offset_1583] * stride_1597 + offset_1598) + (long) i] - p_cup[((long) kbcon[(long) i + offset_1587] * stride_1597 + offset_1598) + (long) i];
          {
            GFReal M_1608;
            GFReal D_9093;

            M_1608 = 2.5e+1f;
            D_9093 = cap_max[(long) i + offset_1553] - (GFReal) (*iloop + -1) * cap_inc[(long) i + offset_1549];
            M_1608 = gf_max(D_9093, M_1608);
            plus = M_1608;
          }
          if (*iloop == 4)
            {
              plus = cap_max[(long) i + offset_1553];
            }

          if (*iloop == 5)
            {
              plus = 2.5e+1f;
            }

          if (*iloop == 5 && cap_max[(long) i + offset_1553] > 2.5e+1f)
            {
              pbcdif = cap_max[(long) i + offset_1553] - p_cup[((long) kbcon[(long) i + offset_1587] * stride_1597 + offset_1598) + (long) i];
            }

          if (pbcdif > plus)
            {
              k22[(long) i + offset_1583] = k22[(long) i + offset_1583] + 1;
              kbcon[(long) i + offset_1587] = k22[(long) i + offset_1583] + 1;
              if (*use_excess == 2)
                {
                  {
                    GFReal __var_1_fcn;

                    {
                      int M_1609;
                      int D_9100;

                      M_1609 = 1;
                      D_9100 = k22[(long) i + offset_1583] + -1;
                      M_1609 = gf_max(D_9100, M_1609);
                      k1 = M_1609;
                    }
                    {
                      int D_9101;
                      int M_1610;
                      int D_9103;
                      int M_1611;

                      D_9101 = kbcon[(long) i + offset_1587] + -1;
                      M_1610 = D_9101;
                      D_9103 = k22[(long) i + offset_1583] + 1;
                      M_1610 = gf_min(D_9103, M_1610);
                      M_1611 = 1;
                      M_1611 = gf_max(M_1610, M_1611);
                      k2 = M_1611;
                    }
                    k2 = k22[(long) i + offset_1583] + 1;
                    __var_1_fcn = (GFReal) ((k2 - k1) + 1);
                    {
                      GFReal val_1612;
                      long D_9108;
                      long D_9109;
                      long D_9110;
                      long D_9111;

                      val_1612 = 0.0f;
                      D_9108 = offset_1560;
                      D_9109 = (long) k1;
                      D_9110 = (long) k2;
                      D_9111 = (long) i;
                      {
                        long S_1613;
                        long D_9113;
                        long D_9114;

                        D_9113 = stride_1559;
                        D_9114 = D_9108 + D_9111;
                        S_1613 = D_9109;
                        while (1)
                          {
                            if (S_1613 > D_9110) break;
                            val_1612 = he_cup[S_1613 * D_9113 + D_9114] + val_1612;
                            S_1613 = S_1613 + 1;
                          }

                      }
                      hkb[(long) i + offset_1571] = val_1612 / __var_1_fcn + ((*xl * zqexec[(long) i + offset_1602] + *cp * ztexec[(long) i + offset_1606])) / __var_1_fcn;
                    }

                  }
                }
              else
                {
                  if (*use_excess <= 1)
                    {
                      hkb[(long) i + offset_1571] = he_cup[((long) k22[(long) i + offset_1583] * stride_1559 + offset_1560) + (long) i] + (GFReal) *use_excess * ((*xl * zqexec[(long) i + offset_1602] + *cp * ztexec[(long) i + offset_1606]));
                    }

                }

              if (*iloop == 5)
                {
                  kbcon[(long) i + offset_1587] = k22[(long) i + offset_1583];
                }

              if (kbcon[(long) i + offset_1587] > kbmax[(long) i + offset_1591] + 2)
                {
                  if (*iloop != 4)
                    {
                      ierr[(long) i + offset_1575] = 3;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_1579], "could not find reasonable kbcon in cup_kbcon", 44);
                      gf_memset ((device char *)&ierrc[(long) i + offset_1579] + 44, 32, 6);
                    }

                  break;
                }

              gf_first=true; continue;
            }

          break; } }

          i = i + 1;
        }
      }

  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43> static inline void cup_forcing_ens_3d(thread GFWorkspace &workspace, P0 closure_n, P1 xland, P2 aa0, P3 aa1, P4 xaa0, P5 mbdt, P6 dtime, P7 ierr, P8 ierr2, P9 ierr3, P10 xf_ens, P11 j, P12 name, P13 axx, P14 maxens, P15 iens, P16 iedt, P17 maxens2, P18 maxens3, P19 mconv, P20 p_cup, P21 ktop, P22 omeg, P23 zd, P24 k22, P25 zu, P26 pr_ens, P27 edt, P28 kbcon, P29 ensdim, P30 icoic, P31 ipr, P32 jpr, P33 itf, P34 jtf, P35 ktf, P36 its, P37 ite, P38 jts, P39 jte, P40 kts, P41 kte, P42 ens4, P43 ktau, long _name) {
  GFFrame frame(workspace);

  device void * D_9497;

  device void * D_9501;

  device void * D_9505;
  GFReal a1;
  GFReal a_ave;
  long lbound_1614;
  long ubound_1615;
  long offset_1616;
  long size_1617;
  long lbound_1618;
  long ubound_1619;
  long offset_1620;
  long size_1621;
  GFReal acrit[15] = {6.329999864101409912109375e-2f, 4.4500000774860382080078125e-2f, 5.53000010550022125244140625e-2f, 6.6399998962879180908203125e-2f, 7.500000298023223876953125e-2f, 1.0819999873638153076171875e-1f, 1.520999968051910400390625e-1f, 2.21599996089935302734375e-1f, 3.15100014209747314453125e-1f, 3.677000105381011962890625e-1f, 4.099999964237213134765625e-1f, 5.25499999523162841796875e-1f, 7.66300022602081298828125e-1f, 1.16859996318817138671875e+0f, 1.68509995937347412109375e+0f};
  GFReal acritt[15] = {2.0299999415874481201171875e-1f, 5.1499998569488525390625e-1f, 5.2100002765655517578125e-1f, 5.659999847412109375e-1f, 6.25e-1f, 6.65000021457672119140625e-1f, 6.58999979496002197265625e-1f, 6.88000023365020751953125e-1f, 7.42999970912933349609375e-1f, 8.13000023365020751953125e-1f, 8.85999977588653564453125e-1f, 9.47000026702880859375e-1f, 1.138000011444091796875e+0f, 1.37699997425079345703125e+0f, 1.89600002765655517578125e+0f};
  long lbound_1622;
  long ubound_1623;
  long ubound_1624;
  long stride_1625;
  long offset_1626;
  long size_1627;
  long lbound_1628;
  long ubound_1629;
  long offset_1630;
  long size_1631;
  long lbound_1632;
  long ubound_1633;
  long offset_1634;
  long size_1635;

  device GFReal * ens_adj;
  GFReal fens4;
  int i;
  long lbound_1636;
  long ubound_1637;
  long offset_1638;
  long size_1639;
  long lbound_1640;
  long ubound_1641;
  long offset_1642;
  long size_1643;
  long lbound_1644;
  long ubound_1645;
  long offset_1646;
  long size_1647;
  int irandom;
  int iresult;
  int iresultd;
  int iresulte;
  int ixxx;
  int k;
  long lbound_1648;
  long ubound_1649;
  long offset_1650;
  long size_1651;
  long lbound_1652;
  long ubound_1653;
  long offset_1654;
  long size_1655;
  GFReal massfld;
  long ubound_1656;
  long size_1657;
  long lbound_1658;
  long ubound_1659;
  long ubound_1660;
  long stride_1661;
  long offset_1662;
  long size_1663;
  int n;
  int nall;
  int nall2;
  int ne;
  int nens;
  long lbound_1664;
  long ubound_1665;
  long lbound_1666;
  long ubound_1667;
  long stride_1668;
  long ubound_1669;
  long stride_1670;
  long offset_1671;
  long size_1672;
  GFReal pcrit[15] = {8.5e+2f, 8.0e+2f, 7.5e+2f, 7.0e+2f, 6.5e+2f, 6.0e+2f, 5.5e+2f, 5.0e+2f, 4.5e+2f, 4.0e+2f, 3.5e+2f, 3.0e+2f, 2.5e+2f, 2.0e+2f, 1.5e+2f};
  long lbound_1673;
  long ubound_1674;
  long lbound_1675;
  long ubound_1676;
  long stride_1677;
  long ubound_1678;
  long stride_1679;
  long offset_1680;
  long size_1681;
  int seed[8];
  long lbound_1682;
  long ubound_1683;
  long ubound_1684;
  long stride_1685;
  long offset_1686;
  long size_1687;
  long lbound_1688;
  long ubound_1689;
  long lbound_1690;
  long ubound_1691;
  long stride_1692;
  long ubound_1693;
  long stride_1694;
  long offset_1695;
  long size_1696;
  GFReal xff0;
  long ubound_1697;
  long size_1698;

  device GFReal * xff_ens3;
  long ubound_1699;
  long size_1700;

  device GFReal * xk;
  long lbound_1701;
  long ubound_1702;
  long offset_1703;
  long size_1704;
  GFReal xomg;
  GFReal xxx;
  long D_9494;
  ulong D_9495;
  ulong D_9496;
  long D_9498;
  ulong D_9499;
  ulong D_9500;
  long D_9502;
  ulong D_9503;
  ulong D_9504;
  long D_9506;
  ulong D_9507;
  ulong D_9508;
  long D_9509;
  ulong D_9510;
  ulong D_9511;
  long D_9512;
  ulong D_9513;
  ulong D_9514;
  long D_9515;
  ulong D_9516;
  ulong D_9517;
  long D_9518;
  ulong D_9519;
  ulong D_9520;
  long D_9521;
  ulong D_9522;
  ulong D_9523;
  long D_9524;
  ulong D_9525;
  ulong D_9526;
  long D_9527;
  ulong D_9528;
  ulong D_9529;
  long D_9530;
  ulong D_9531;
  ulong D_9532;
  long D_9533;
  ulong D_9534;
  ulong D_9535;
  long D_9536;
  ulong D_9537;
  ulong D_9538;
  long D_9539;
  ulong D_9540;
  ulong D_9541;
  long D_9542;
  ulong D_9543;
  ulong D_9544;
  long D_9545;
  ulong D_9546;
  ulong D_9547;
  long D_9548;
  ulong D_9549;
  ulong D_9550;
  long D_9551;
  ulong D_9552;
  ulong D_9553;
  ulong D_9554;
  ulong D_9555;

    {
      D_9554 = (ulong) (ulong) (_name) * 8;
      D_9555 = (ulong) (_name);
      lbound_1636 = (long) *its;
      ubound_1637 = (long) *ite;
      size_1639 = (1 - lbound_1636) + ubound_1637;
      size_1639 = gf_max(size_1639, 0);
      D_9551 = size_1639 + -1;
      D_9552 = (ulong) (ulong) (size_1639) * 32;
      D_9553 = (ulong) (size_1639) * 4;
      offset_1638 = -(lbound_1636);
      lbound_1614 = (long) *its;
      ubound_1615 = (long) *ite;
      size_1617 = (1 - lbound_1614) + ubound_1615;
      size_1617 = gf_max(size_1617, 0);
      D_9548 = size_1617 + -1;
      D_9549 = (ulong) (ulong) (size_1617) * 32;
      D_9550 = (ulong) (size_1617) * 4;
      offset_1616 = -(lbound_1614);
      lbound_1618 = (long) *its;
      ubound_1619 = (long) *ite;
      size_1621 = (1 - lbound_1618) + ubound_1619;
      size_1621 = gf_max(size_1621, 0);
      D_9545 = size_1621 + -1;
      D_9546 = (ulong) (ulong) (size_1621) * 32;
      D_9547 = (ulong) (size_1621) * 4;
      offset_1620 = -(lbound_1618);
      lbound_1640 = (long) *its;
      ubound_1641 = (long) *ite;
      size_1643 = (1 - lbound_1640) + ubound_1641;
      size_1643 = gf_max(size_1643, 0);
      D_9542 = size_1643 + -1;
      D_9543 = (ulong) (ulong) (size_1643) * 32;
      D_9544 = (ulong) (size_1643) * 4;
      offset_1642 = -(lbound_1640);
      lbound_1644 = (long) *its;
      ubound_1645 = (long) *ite;
      size_1647 = (1 - lbound_1644) + ubound_1645;
      size_1647 = gf_max(size_1647, 0);
      D_9539 = size_1647 + -1;
      D_9540 = (ulong) (ulong) (size_1647) * 32;
      D_9541 = (ulong) (size_1647) * 4;
      offset_1646 = -(lbound_1644);
      lbound_1622 = (long) *its;
      ubound_1623 = (long) *ite;
      stride_1625 = (1 - lbound_1622) + ubound_1623;
      stride_1625 = gf_max(stride_1625, 0);
      ubound_1624 = (long) *ens4;
      size_1627 = stride_1625 * (ubound_1624);
      size_1627 = gf_max(size_1627, 0);
      D_9536 = size_1627 + -1;
      D_9537 = (ulong) (ulong) (size_1627) * 32;
      D_9538 = (ulong) (size_1627) * 4;
      offset_1626 = -(lbound_1622) - (stride_1625);
      lbound_1664 = (long) *its;
      ubound_1665 = (long) *ite;
      stride_1668 = (1 - lbound_1664) + ubound_1665;
      stride_1668 = gf_max(stride_1668, 0);
      lbound_1666 = (long) *kts;
      ubound_1667 = (long) *kte;
      stride_1670 = ((1 - lbound_1666) + ubound_1667) * stride_1668;
      stride_1670 = gf_max(stride_1670, 0);
      ubound_1669 = (long) *ens4;
      size_1672 = stride_1670 * (ubound_1669);
      size_1672 = gf_max(size_1672, 0);
      D_9533 = size_1672 + -1;
      D_9534 = (ulong) (ulong) (size_1672) * 32;
      D_9535 = (ulong) (size_1672) * 4;
      offset_1671 = (-(lbound_1664) - lbound_1666 * stride_1668) - (stride_1670);
      lbound_1648 = (long) *its;
      ubound_1649 = (long) *ite;
      size_1651 = (1 - lbound_1648) + ubound_1649;
      size_1651 = gf_max(size_1651, 0);
      D_9530 = size_1651 + -1;
      D_9531 = (ulong) (ulong) (size_1651) * 32;
      D_9532 = (ulong) (size_1651) * 4;
      offset_1650 = -(lbound_1648);
      lbound_1652 = (long) *its;
      ubound_1653 = (long) *ite;
      size_1655 = (1 - lbound_1652) + ubound_1653;
      size_1655 = gf_max(size_1655, 0);
      D_9527 = size_1655 + -1;
      D_9528 = (ulong) (ulong) (size_1655) * 32;
      D_9529 = (ulong) (size_1655) * 4;
      offset_1654 = -(lbound_1652);
      lbound_1658 = (long) *its;
      ubound_1659 = (long) *ite;
      stride_1661 = (1 - lbound_1658) + ubound_1659;
      stride_1661 = gf_max(stride_1661, 0);
      ubound_1660 = (long) *ens4;
      size_1663 = stride_1661 * (ubound_1660);
      size_1663 = gf_max(size_1663, 0);
      D_9524 = size_1663 + -1;
      D_9525 = (ulong) (ulong) (size_1663) * 32;
      D_9526 = (ulong) (size_1663) * 4;
      offset_1662 = -(lbound_1658) - (stride_1661);
      lbound_1701 = (long) *its;
      ubound_1702 = (long) *ite;
      size_1704 = (1 - lbound_1701) + ubound_1702;
      size_1704 = gf_max(size_1704, 0);
      D_9521 = size_1704 + -1;
      D_9522 = (ulong) (ulong) (size_1704) * 32;
      D_9523 = (ulong) (size_1704) * 4;
      offset_1703 = -(lbound_1701);
      lbound_1682 = (long) *its;
      ubound_1683 = (long) *ite;
      stride_1685 = (1 - lbound_1682) + ubound_1683;
      stride_1685 = gf_max(stride_1685, 0);
      ubound_1684 = (long) *maxens;
      size_1687 = stride_1685 * (ubound_1684);
      size_1687 = gf_max(size_1687, 0);
      D_9518 = size_1687 + -1;
      D_9519 = (ulong) (ulong) (size_1687) * 32;
      D_9520 = (ulong) (size_1687) * 4;
      offset_1686 = -(lbound_1682) - (stride_1685);
      ubound_1656 = (long) *maxens;
      size_1657 = (ubound_1656);
      size_1657 = gf_max(size_1657, 0);
      D_9515 = size_1657 + -1;
      D_9516 = (ulong) (ulong) (size_1657) * 32;
      D_9517 = (ulong) (size_1657) * 4;
      lbound_1688 = (long) *its;
      ubound_1689 = (long) *ite;
      stride_1692 = (1 - lbound_1688) + ubound_1689;
      stride_1692 = gf_max(stride_1692, 0);
      lbound_1690 = (long) *jts;
      ubound_1691 = (long) *jte;
      stride_1694 = ((1 - lbound_1690) + ubound_1691) * stride_1692;
      stride_1694 = gf_max(stride_1694, 0);
      ubound_1693 = (long) *ensdim;
      size_1696 = stride_1694 * (ubound_1693);
      size_1696 = gf_max(size_1696, 0);
      D_9512 = size_1696 + -1;
      D_9513 = (ulong) (ulong) (size_1696) * 32;
      D_9514 = (ulong) (size_1696) * 4;
      offset_1695 = (-(lbound_1688) - lbound_1690 * stride_1692) - (stride_1694);
      lbound_1673 = (long) *its;
      ubound_1674 = (long) *ite;
      stride_1677 = (1 - lbound_1673) + ubound_1674;
      stride_1677 = gf_max(stride_1677, 0);
      lbound_1675 = (long) *jts;
      ubound_1676 = (long) *jte;
      stride_1679 = ((1 - lbound_1675) + ubound_1676) * stride_1677;
      stride_1679 = gf_max(stride_1679, 0);
      ubound_1678 = (long) *ensdim;
      size_1681 = stride_1679 * (ubound_1678);
      size_1681 = gf_max(size_1681, 0);
      D_9509 = size_1681 + -1;
      D_9510 = (ulong) (ulong) (size_1681) * 32;
      D_9511 = (ulong) (size_1681) * 4;
      offset_1680 = (-(lbound_1673) - lbound_1675 * stride_1677) - (stride_1679);
      lbound_1628 = (long) *its;
      ubound_1629 = (long) *ite;
      size_1631 = (1 - lbound_1628) + ubound_1629;
      size_1631 = gf_max(size_1631, 0);
      D_9506 = size_1631 + -1;
      D_9507 = (ulong) (ulong) (size_1631) * 32;
      D_9508 = (ulong) (size_1631) * 4;
      offset_1630 = -(lbound_1628);
      lbound_1632 = (long) *its;
      ubound_1633 = (long) *ite;
      size_1635 = (1 - lbound_1632) + ubound_1633;
      size_1635 = gf_max(size_1635, 0);
      D_9502 = size_1635 + -1;
      D_9503 = (ulong) (ulong) (size_1635) * 32;
      D_9504 = (ulong) (size_1635) * 4;
      D_9505 = (device void *) workspace.allocate (gf_max((ulong) (size_1635 * 4), 1));
      ens_adj = (device GFReal *) D_9505;
      offset_1634 = -(lbound_1632);
      ubound_1697 = (long) *maxens3;
      size_1698 = (ubound_1697);
      size_1698 = gf_max(size_1698, 0);
      D_9498 = size_1698 + -1;
      D_9499 = (ulong) (ulong) (size_1698) * 32;
      D_9500 = (ulong) (size_1698) * 4;
      D_9501 = (device void *) workspace.allocate (gf_max((ulong) (size_1698 * 4), 1));
      xff_ens3 = (device GFReal *) D_9501;
      ubound_1699 = (long) *maxens;
      size_1700 = (ubound_1699);
      size_1700 = gf_max(size_1700, 0);
      D_9494 = size_1700 + -1;
      D_9495 = (ulong) (ulong) (size_1700) * 32;
      D_9496 = (ulong) (size_1700) * 4;
      D_9497 = (device void *) workspace.allocate (gf_max((ulong) (size_1700 * 4), 1));
      xk = (device GFReal *) D_9497;
      {
        long D_9301;
        long D_9302;
        long D_9303;

        D_9301 = offset_1634;
        D_9302 = lbound_1632;
        D_9303 = ubound_1633;
        {
          long S_1705;

          S_1705 = D_9302;
          while (1)
            {
              if (S_1705 > D_9303) break;
              ens_adj[S_1705 + D_9301] = 1.0e+0f;
              S_1705 = S_1705 + 1;
            }

        }
      }
      gf_memset(seed, 0, sizeof(seed));
      {
        int D_9306;
        int D_9307;

        D_9306 = *its;
        D_9307 = *itf;
        i = D_9306;
        while (1)
          {
            {
              bool D_9310;

              D_9310 = i > D_9307;
              if (D_9310) break;
              if (ierr[(long) i + offset_1638] == 0)
                {
                  seed[0] = (int) aa0[(long) i + offset_1616];
                  seed[1] = (int) aa1[(long) i + offset_1620];
                  break;
                }


              i = i + 1;
            }
          }

      }
      nens = 0;
      irandom = 0;
      fens4 = (GFReal) *ens4;
      {
        int D_9312;
        int D_9313;

        D_9312 = *its;
        D_9313 = *itf;
        i = D_9312;
        while (1)
          {
            {
              bool D_9316;

              D_9316 = i > D_9313;
              if (D_9316) break;
              if (gf_compare_string (_name, name, 5, "deeps") == 0 && ierr[(long) i + offset_1638] > 995)
                {
                  aa0[(long) i + offset_1616] = 0.0f;
                  ierr[(long) i + offset_1638] = 0;
                }

              if (ierr[(long) i + offset_1638] == 0)
                {
                  ens_adj[(long) i + offset_1634] = 1.0e+0f;
                  if (ierr2[(long) i + offset_1642] > 0 && ierr3[(long) i + offset_1646] == 0)
                    {
                      ens_adj[(long) i + offset_1634] = 0.0f;
                    }

                  if (ierr2[(long) i + offset_1642] > 0 && ierr3[(long) i + offset_1646] > 0)
                    {
                      ens_adj[(long) i + offset_1634] = 0.0f;
                    }

                  if (gf_compare_string (_name, name, 5, "deeps") == 0)
                    {
                      a_ave = 0.0f;
                      {
                        int D_9322;

                        D_9322 = *ens4;
                        ne = 1;
                        while (1)
                          {
                            {
                              bool D_9325;

                              D_9325 = ne > D_9322;
                              if (D_9325) break;
                              a_ave = axx[((long) ne * stride_1625 + offset_1626) + (long) i] + a_ave;

                              ne = ne + 1;
                            }
                          }

                      }
                      {
                        GFReal M_1706;
                        GFReal D_9327;

                        M_1706 = 0.0f;
                        D_9327 = a_ave / fens4;
                        M_1706 = gf_max(D_9327, M_1706);
                        a_ave = M_1706;
                      }
                      {
                        GFReal M_1707;
                        GFReal D_9329;

                        M_1707 = a_ave;
                        D_9329 = aa1[(long) i + offset_1620];
                        M_1707 = gf_min(D_9329, M_1707);
                        a_ave = M_1707;
                      }
                      {
                        GFReal M_1708;

                        M_1708 = 0.0f;
                        M_1708 = gf_max(a_ave, M_1708);
                        a_ave = M_1708;
                      }
                      ne = 1;
                      while (1)
                        {
                          {
                            bool D_9333;

                            D_9333 = ne > 16;
                            if (D_9333) break;
                            xff_ens3[(long) ne + -1] = 0.0f;

                            ne = ne + 1;
                          }
                        }

                      xff0 = ((aa1[(long) i + offset_1620] - aa0[(long) i + offset_1616])) / *dtime;
                      {
                        GFReal M_1709;
                        GFReal D_9335;

                        M_1709 = 0.0f;
                        D_9335 = ((aa1[(long) i + offset_1620] - aa0[(long) i + offset_1616])) / *dtime;
                        M_1709 = gf_max(D_9335, M_1709);
                        xff_ens3[0] = M_1709;
                      }
                      {
                        GFReal M_1710;
                        GFReal D_9337;

                        M_1710 = 0.0f;
                        D_9337 = ((a_ave - aa0[(long) i + offset_1616])) / *dtime;
                        M_1710 = gf_max(D_9337, M_1710);
                        xff_ens3[1] = M_1710;
                      }
                      if (irandom == 1)
                        {
                          gf_unreachable_random (&xxx);
                          {
                            int M_1711;
                            int D_9341;
                            int D_9342;
                            int M_1712;

                            M_1711 = 1;
                            D_9341 = (int) (fens4 * xxx + 9.99999993922529029077850282192230224609375e-9f);
                            M_1711 = gf_max(D_9341, M_1711);
                            D_9342 = *ens4;
                            M_1712 = D_9342;
                            M_1712 = gf_min(M_1711, M_1712);
                            ixxx = M_1712;
                          }
                          {
                            GFReal M_1713;
                            GFReal D_9345;

                            M_1713 = 0.0f;
                            D_9345 = ((axx[((long) ixxx * stride_1625 + offset_1626) + (long) i] - aa0[(long) i + offset_1616])) / *dtime;
                            M_1713 = gf_max(D_9345, M_1713);
                            xff_ens3[2] = M_1713;
                          }
                          gf_unreachable_random (&xxx);
                          {
                            int M_1714;
                            int D_9347;
                            int D_9348;
                            int M_1715;

                            M_1714 = 1;
                            D_9347 = (int) (fens4 * xxx + 9.99999993922529029077850282192230224609375e-9f);
                            M_1714 = gf_max(D_9347, M_1714);
                            D_9348 = *ens4;
                            M_1715 = D_9348;
                            M_1715 = gf_min(M_1714, M_1715);
                            ixxx = M_1715;
                          }
                          {
                            GFReal M_1716;
                            GFReal D_9351;

                            M_1716 = 0.0f;
                            D_9351 = ((axx[((long) ixxx * stride_1625 + offset_1626) + (long) i] - aa0[(long) i + offset_1616])) / *dtime;
                            M_1716 = gf_max(D_9351, M_1716);
                            xff_ens3[12] = M_1716;
                          }
                        }
                      else
                        {
                          {
                            GFReal M_1717;
                            GFReal D_9353;

                            M_1717 = 0.0f;
                            D_9353 = ((aa1[(long) i + offset_1620] - aa0[(long) i + offset_1616])) / *dtime;
                            M_1717 = gf_max(D_9353, M_1717);
                            xff_ens3[2] = M_1717;
                          }
                          {
                            GFReal M_1718;
                            GFReal D_9355;

                            M_1718 = 0.0f;
                            D_9355 = ((aa1[(long) i + offset_1620] - aa0[(long) i + offset_1616])) / *dtime;
                            M_1718 = gf_max(D_9355, M_1718);
                            xff_ens3[12] = M_1718;
                          }
                        }

                      xff_ens3[13] = 0.0f;
                      {
                        int D_9356;

                        D_9356 = *ens4;
                        ne = 1;
                        while (1)
                          {
                            {
                              bool D_9359;

                              D_9359 = ne > D_9356;
                              if (D_9359) break;
                              xff_ens3[13] = xff_ens3[13] - omeg[(((long) ne * stride_1670 + offset_1671) + (long) k22[(long) i + offset_1650] * stride_1668) + (long) i] / ((fens4 * 9.81000041961669921875e+0f));

                              ne = ne + 1;
                            }
                          }

                      }
                      if (xff_ens3[13] < 0.0f)
                        {
                          xff_ens3[13] = 0.0f;
                        }

                      xff_ens3[4] = 0.0f;
                      {
                        int D_9361;

                        D_9361 = *ens4;
                        ne = 1;
                        while (1)
                          {
                            {
                              bool D_9364;

                              D_9364 = ne > D_9361;
                              if (D_9364) break;
                              xff_ens3[4] = xff_ens3[4] - omeg[(((long) ne * stride_1670 + offset_1671) + (long) kbcon[(long) i + offset_1654] * stride_1668) + (long) i] / ((fens4 * 9.81000041961669921875e+0f));

                              ne = ne + 1;
                            }
                          }

                      }
                      if (xff_ens3[4] < 0.0f)
                        {
                          xff_ens3[4] = 0.0f;
                        }

                      xff_ens3[3] = -(omeg[((offset_1671 + (stride_1670)) + stride_1668 * 2) + (long) i] / 9.81000041961669921875e+0f);
                      {
                        int D_9366;

                        D_9366 = kbcon[(long) i + offset_1654] + -1;
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_9369;

                              D_9369 = k > D_9366;
                              if (D_9369) break;
                              {
                                int D_9370;

                                D_9370 = *ens4;
                                ne = 1;
                                while (1)
                                  {
                                    {
                                      bool D_9373;

                                      D_9373 = ne > D_9370;
                                      if (D_9373) break;
                                      xomg = -(omeg[(((long) ne * stride_1670 + offset_1671) + (long) k * stride_1668) + (long) i] / 9.81000041961669921875e+0f);
                                      if (xff_ens3[3] > xomg)
                                        {
                                          xff_ens3[3] = xomg;
                                        }


                                      ne = ne + 1;
                                    }
                                  }

                              }

                              k = k + 1;
                            }
                          }

                      }
                      if (xff_ens3[3] < 0.0f)
                        {
                          xff_ens3[3] = 0.0f;
                        }

                      xff_ens3[5] = -(omeg[((offset_1671 + (stride_1670)) + stride_1668 * 2) + (long) i] / 9.81000041961669921875e+0f);
                      {
                        int D_9376;

                        D_9376 = kbcon[(long) i + offset_1654] + -1;
                        k = 2;
                        while (1)
                          {
                            {
                              bool D_9379;

                              D_9379 = k > D_9376;
                              if (D_9379) break;
                              {
                                int D_9380;

                                D_9380 = *ens4;
                                ne = 1;
                                while (1)
                                  {
                                    {
                                      bool D_9383;

                                      D_9383 = ne > D_9380;
                                      if (D_9383) break;
                                      xomg = -(omeg[(((long) ne * stride_1670 + offset_1671) + (long) k * stride_1668) + (long) i] / 9.81000041961669921875e+0f);
                                      if (xff_ens3[5] < xomg)
                                        {
                                          xff_ens3[5] = xomg;
                                        }


                                      ne = ne + 1;
                                    }
                                  }

                              }

                              k = k + 1;
                            }
                          }

                      }
                      if (xff_ens3[5] < 0.0f)
                        {
                          xff_ens3[5] = 0.0f;
                        }

                      xff_ens3[4] = xff_ens3[5];
                      xff_ens3[3] = xff_ens3[5];
                      xff_ens3[6] = mconv[(offset_1662 + (stride_1661)) + (long) i];
                      xff_ens3[7] = mconv[(offset_1662 + (stride_1661)) + (long) i];
                      xff_ens3[8] = mconv[(offset_1662 + (stride_1661)) + (long) i];
                      if (*ens4 > 1)
                        {
                          {
                            int D_9387;

                            D_9387 = *ens4;
                            ne = 2;
                            while (1)
                              {
                                {
                                  bool D_9390;

                                  D_9390 = ne > D_9387;
                                  if (D_9390) break;
                                  if (mconv[((long) ne * stride_1661 + offset_1662) + (long) i] > xff_ens3[6])
                                    {
                                      xff_ens3[6] = mconv[((long) ne * stride_1661 + offset_1662) + (long) i];
                                    }


                                  ne = ne + 1;
                                }
                              }

                          }
                          {
                            int D_9392;

                            D_9392 = *ens4;
                            ne = 2;
                            while (1)
                              {
                                {
                                  bool D_9395;

                                  D_9395 = ne > D_9392;
                                  if (D_9395) break;
                                  if (mconv[((long) ne * stride_1661 + offset_1662) + (long) i] < xff_ens3[7])
                                    {
                                      xff_ens3[7] = mconv[((long) ne * stride_1661 + offset_1662) + (long) i];
                                    }


                                  ne = ne + 1;
                                }
                              }

                          }
                          {
                            int D_9397;

                            D_9397 = *ens4;
                            ne = 2;
                            while (1)
                              {
                                {
                                  bool D_9400;

                                  D_9400 = ne > D_9397;
                                  if (D_9400) break;
                                  xff_ens3[8] = xff_ens3[8] + mconv[((long) ne * stride_1661 + offset_1662) + (long) i];

                                  ne = ne + 1;
                                }
                              }

                          }
                          xff_ens3[8] = xff_ens3[8] / fens4;
                        }

                      if (irandom == 1)
                        {
                          gf_unreachable_random (&xxx);
                          {
                            int M_1719;
                            int D_9403;
                            int D_9404;
                            int M_1720;

                            M_1719 = 1;
                            D_9403 = (int) (fens4 * xxx + 9.99999993922529029077850282192230224609375e-9f);
                            M_1719 = gf_max(D_9403, M_1719);
                            D_9404 = *ens4;
                            M_1720 = D_9404;
                            M_1720 = gf_min(M_1719, M_1720);
                            ixxx = M_1720;
                          }
                          xff_ens3[14] = mconv[((long) ixxx * stride_1661 + offset_1662) + (long) i];
                        }
                      else
                        {
                          xff_ens3[14] = mconv[(offset_1662 + (stride_1661)) + (long) i];
                        }

                      xff_ens3[9] = aa0[(long) i + offset_1616] / 1.2e+3f;
                      xff_ens3[10] = aa0[(long) i + offset_1616] / 1.2e+3f;
                      xff_ens3[15] = aa0[(long) i + offset_1616] / 1.2e+3f;
                      if (irandom == 1)
                        {
                          gf_unreachable_random (&xxx);
                          {
                            int M_1721;
                            int D_9408;
                            int D_9409;
                            int M_1722;

                            M_1721 = 1;
                            D_9408 = (int) (fens4 * xxx + 9.99999993922529029077850282192230224609375e-9f);
                            M_1721 = gf_max(D_9408, M_1721);
                            D_9409 = *ens4;
                            M_1722 = D_9409;
                            M_1722 = gf_min(M_1721, M_1722);
                            ixxx = M_1722;
                          }
                          xff_ens3[11] = aa0[(long) i + offset_1616] / 1.2e+3f;
                        }
                      else
                        {
                          xff_ens3[11] = aa0[(long) i + offset_1616] / 1.2e+3f;
                        }

                      if (*icoic == 0)
                        {
                          if (xff0 < 0.0f)
                            {
                              xff_ens3[0] = 0.0f;
                              xff_ens3[1] = 0.0f;
                              xff_ens3[2] = 0.0f;
                              xff_ens3[12] = 0.0f;
                              xff_ens3[9] = 0.0f;
                              xff_ens3[10] = 0.0f;
                              xff_ens3[11] = 0.0f;
                            }

                          if (xff0 < 0.0f && xland[(long) i + offset_1703] < 1.00000001490116119384765625e-1f)
                            {
                              {
                                long D_9414;

                                D_9414 = ubound_1697;
                                {
                                  long S_1723;

                                  S_1723 = 1;
                                  while (1)
                                    {
                                      if (S_1723 > D_9414) break;
                                      xff_ens3[S_1723 + -1] = 0.0f;
                                      S_1723 = S_1723 + 1;
                                    }

                                }
                              }
                            }

                        }

                      {
                        int D_9417;

                        D_9417 = *maxens;
                        nens = 1;
                        while (1)
                          {
                            {
                              bool D_9420;

                              D_9420 = nens > D_9417;
                              if (D_9420) break;
                              xk[(long) nens + -1] = ((xaa0[((long) nens * stride_1685 + offset_1686) + (long) i] - aa1[(long) i + offset_1620])) / mbdt[0];
                              if (xk[(long) nens + -1] <= 0.0f && xk[(long) nens + -1] > -9.99999977648258209228515625e-3f)
                                {
                                  xk[(long) nens + -1] = -9.99999977648258209228515625e-3f;
                                }

                              if (xk[(long) nens + -1] > 0.0f && xk[(long) nens + -1] < 9.99999977648258209228515625e-3f)
                                {
                                  xk[(long) nens + -1] = 9.99999977648258209228515625e-3f;
                                }


                              nens = nens + 1;
                            }
                          }

                      }
                      {
                        int D_9423;

                        D_9423 = *maxens;
                        ne = 1;
                        while (1)
                          {
                            {
                              bool D_9426;

                              D_9426 = ne > D_9423;
                              if (D_9426) break;
                              iresult = 0;
                              iresultd = 0;
                              iresulte = 0;
                              nall = ((((*iens + -1) * *maxens3) * *maxens) * *maxens2 + ((*iedt + -1) * *maxens) * *maxens3) + (ne + -1) * *maxens3;
                              if (*maxens > 0 && xland[(long) i + offset_1703] < 1.00000001490116119384765625e-1f)
                                {
                                  if (ierr2[(long) i + offset_1642] > 0 || ierr3[(long) i + offset_1646] > 0)
                                    {
                                      xff_ens3[0] = ens_adj[(long) i + offset_1634] * xff_ens3[0];
                                      xff_ens3[1] = ens_adj[(long) i + offset_1634] * xff_ens3[1];
                                      xff_ens3[2] = ens_adj[(long) i + offset_1634] * xff_ens3[2];
                                      xff_ens3[12] = ens_adj[(long) i + offset_1634] * xff_ens3[12];
                                      xff_ens3[9] = ens_adj[(long) i + offset_1634] * xff_ens3[9];
                                      xff_ens3[10] = ens_adj[(long) i + offset_1634] * xff_ens3[10];
                                      xff_ens3[11] = ens_adj[(long) i + offset_1634] * xff_ens3[11];
                                      xff_ens3[15] = ens_adj[(long) i + offset_1634] * xff_ens3[15];
                                      xff_ens3[6] = ens_adj[(long) i + offset_1634] * xff_ens3[6];
                                      xff_ens3[7] = ens_adj[(long) i + offset_1634] * xff_ens3[7];
                                      xff_ens3[8] = ens_adj[(long) i + offset_1634] * xff_ens3[8];
                                      xff_ens3[14] = ens_adj[(long) i + offset_1634] * xff_ens3[14];
                                    }

                                }

                              massfld = 0.0f;
                              if (xk[(long) ne + -1] < 0.0f && xff0 > 0.0f)
                                {
                                  iresultd = 1;
                                }

                              {
                                int M_1724;

                                M_1724 = iresult;
                                M_1724 = gf_max(iresultd, M_1724);
                                iresulte = M_1724;
                              }
                              iresulte = 1;
                              if (iresulte == 1)
                                {
                                  if (xff0 >= 0.0f)
                                    {
                                      if (xff_ens3[0] > 0.0f)
                                        {
                                          {
                                            GFReal M_1725;
                                            GFReal D_9435;

                                            M_1725 = 0.0f;
                                            D_9435 = -(xff_ens3[0] / xk[(long) ne + -1]);
                                            M_1725 = gf_max(D_9435, M_1725);
                                            xf_ens[(((long) (nall + 1) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1725;
                                          }
                                        }

                                      if (xff_ens3[1] > 0.0f)
                                        {
                                          {
                                            GFReal M_1726;
                                            GFReal D_9438;

                                            M_1726 = 0.0f;
                                            D_9438 = -(xff_ens3[1] / xk[(long) ne + -1]);
                                            M_1726 = gf_max(D_9438, M_1726);
                                            xf_ens[(((long) (nall + 2) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1726;
                                          }
                                        }

                                      if (xff_ens3[2] > 0.0f)
                                        {
                                          {
                                            GFReal M_1727;
                                            GFReal D_9441;

                                            M_1727 = 0.0f;
                                            D_9441 = -(xff_ens3[2] / xk[(long) ne + -1]);
                                            M_1727 = gf_max(D_9441, M_1727);
                                            xf_ens[(((long) (nall + 3) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1727;
                                          }
                                        }

                                      if (xff_ens3[12] > 0.0f)
                                        {
                                          {
                                            GFReal M_1728;
                                            GFReal D_9444;

                                            M_1728 = 0.0f;
                                            D_9444 = -(xff_ens3[12] / xk[(long) ne + -1]);
                                            M_1728 = gf_max(D_9444, M_1728);
                                            xf_ens[(((long) (nall + 13) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1728;
                                          }
                                        }

                                    }

                                  {
                                    GFReal M_1729;
                                    GFReal D_9446;

                                    M_1729 = 0.0f;
                                    D_9446 = xff_ens3[3];
                                    M_1729 = gf_max(D_9446, M_1729);
                                    xf_ens[(((long) (nall + 4) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1729;
                                  }
                                  {
                                    GFReal M_1730;
                                    GFReal D_9448;

                                    M_1730 = 0.0f;
                                    D_9448 = xff_ens3[4];
                                    M_1730 = gf_max(D_9448, M_1730);
                                    xf_ens[(((long) (nall + 5) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1730;
                                  }
                                  {
                                    GFReal M_1731;
                                    GFReal D_9450;

                                    M_1731 = 0.0f;
                                    D_9450 = xff_ens3[5];
                                    M_1731 = gf_max(D_9450, M_1731);
                                    xf_ens[(((long) (nall + 6) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1731;
                                  }
                                  {
                                    GFReal M_1732;
                                    GFReal D_9452;

                                    M_1732 = 0.0f;
                                    D_9452 = xff_ens3[13];
                                    M_1732 = gf_max(D_9452, M_1732);
                                    xf_ens[(((long) (nall + 14) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1732;
                                  }
                                  {
                                    GFReal M_1733;
                                    GFReal D_9454;

                                    M_1733 = 1.000000047497451305389404296875e-3f;
                                    D_9454 = pr_ens[(((long) (nall + 7) * stride_1679 + offset_1680) + (long) *j * stride_1677) + (long) i];
                                    M_1733 = gf_max(D_9454, M_1733);
                                    a1 = M_1733;
                                  }
                                  {
                                    GFReal M_1734;
                                    GFReal D_9456;

                                    M_1734 = 0.0f;
                                    D_9456 = xff_ens3[6] / a1;
                                    M_1734 = gf_max(D_9456, M_1734);
                                    xf_ens[(((long) (nall + 7) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1734;
                                  }
                                  {
                                    GFReal M_1735;
                                    GFReal D_9458;

                                    M_1735 = 1.000000047497451305389404296875e-3f;
                                    D_9458 = pr_ens[(((long) (nall + 8) * stride_1679 + offset_1680) + (long) *j * stride_1677) + (long) i];
                                    M_1735 = gf_max(D_9458, M_1735);
                                    a1 = M_1735;
                                  }
                                  {
                                    GFReal M_1736;
                                    GFReal D_9460;

                                    M_1736 = 0.0f;
                                    D_9460 = xff_ens3[7] / a1;
                                    M_1736 = gf_max(D_9460, M_1736);
                                    xf_ens[(((long) (nall + 8) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1736;
                                  }
                                  {
                                    GFReal M_1737;
                                    GFReal D_9462;

                                    M_1737 = 1.000000047497451305389404296875e-3f;
                                    D_9462 = pr_ens[(((long) (nall + 9) * stride_1679 + offset_1680) + (long) *j * stride_1677) + (long) i];
                                    M_1737 = gf_max(D_9462, M_1737);
                                    a1 = M_1737;
                                  }
                                  {
                                    GFReal M_1738;
                                    GFReal D_9464;

                                    M_1738 = 0.0f;
                                    D_9464 = xff_ens3[8] / a1;
                                    M_1738 = gf_max(D_9464, M_1738);
                                    xf_ens[(((long) (nall + 9) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1738;
                                  }
                                  {
                                    GFReal M_1739;
                                    GFReal D_9466;

                                    M_1739 = 1.000000047497451305389404296875e-3f;
                                    D_9466 = pr_ens[(((long) (nall + 15) * stride_1679 + offset_1680) + (long) *j * stride_1677) + (long) i];
                                    M_1739 = gf_max(D_9466, M_1739);
                                    a1 = M_1739;
                                  }
                                  {
                                    GFReal M_1740;
                                    GFReal D_9468;

                                    M_1740 = 0.0f;
                                    D_9468 = xff_ens3[14] / a1;
                                    M_1740 = gf_max(D_9468, M_1740);
                                    xf_ens[(((long) (nall + 15) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1740;
                                  }
                                  if (xk[(long) ne + -1] < 0.0f)
                                    {
                                      {
                                        GFReal M_1741;
                                        GFReal D_9471;

                                        M_1741 = 0.0f;
                                        D_9471 = -(xff_ens3[9] / xk[(long) ne + -1]);
                                        M_1741 = gf_max(D_9471, M_1741);
                                        xf_ens[(((long) (nall + 10) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1741;
                                      }
                                      {
                                        GFReal M_1742;
                                        GFReal D_9473;

                                        M_1742 = 0.0f;
                                        D_9473 = -(xff_ens3[10] / xk[(long) ne + -1]);
                                        M_1742 = gf_max(D_9473, M_1742);
                                        xf_ens[(((long) (nall + 11) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1742;
                                      }
                                      {
                                        GFReal M_1743;
                                        GFReal D_9475;

                                        M_1743 = 0.0f;
                                        D_9475 = -(xff_ens3[11] / xk[(long) ne + -1]);
                                        M_1743 = gf_max(D_9475, M_1743);
                                        xf_ens[(((long) (nall + 12) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1743;
                                      }
                                      {
                                        GFReal M_1744;
                                        GFReal D_9477;

                                        M_1744 = 0.0f;
                                        D_9477 = -(xff_ens3[15] / xk[(long) ne + -1]);
                                        M_1744 = gf_max(D_9477, M_1744);
                                        xf_ens[(((long) (nall + 16) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = M_1744;
                                      }
                                    }

                                  if (*icoic > 0)
                                    {
                                      closure_n[(long) i + offset_1630] = 0.0f;
                                      xf_ens[(((long) (nall + 1) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 2) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 3) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 4) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 5) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 6) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 7) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 8) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 9) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 10) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 11) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 12) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 13) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 14) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 15) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                      xf_ens[(((long) (nall + 16) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (*icoic + nall) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                    }

                                  if (irandom == 1)
                                    {
                                      gf_unreachable_random (&xxx);
                                      {
                                        int M_1745;
                                        int D_9481;
                                        int M_1746;

                                        M_1745 = 1;
                                        D_9481 = (int) (xxx * 1.5e+1f + 9.99999993922529029077850282192230224609375e-9f);
                                        M_1745 = gf_max(D_9481, M_1745);
                                        M_1746 = 15;
                                        M_1746 = gf_min(M_1745, M_1746);
                                        ixxx = M_1746;
                                      }
                                      xf_ens[(((long) (nall + 16) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall + ixxx) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                                    }

                                  if (*maxens > 1)
                                    {
                                      if (ne == 2 && ierr2[(long) i + offset_1642] > 0)
                                        {
                                          xf_ens[(((long) (nall + 1) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 2) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 3) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 4) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 5) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 6) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 7) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 8) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 9) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 10) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 11) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 12) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 13) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 14) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 15) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 16) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                        }

                                      if (ne == 3 && ierr3[(long) i + offset_1646] > 0)
                                        {
                                          xf_ens[(((long) (nall + 1) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 2) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 3) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 4) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 5) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 6) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 7) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 8) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 9) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 10) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 11) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 12) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 13) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 14) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 15) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                          xf_ens[(((long) (nall + 16) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;
                                        }

                                    }

                                }



                              ne = ne + 1;
                            }
                          }

                      }
                      if (*maxens > 1)
                        {
                          nall = (((*iens + -1) * *maxens3) * *maxens) * *maxens2 + ((*iedt + -1) * *maxens) * *maxens3;
                          nall2 = ((((*iens + -1) * *maxens3) * *maxens) * *maxens2 + ((*iedt + -1) * *maxens) * *maxens3) + (*maxens3);
                          xf_ens[(((long) (nall + 4) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 4) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 5) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 5) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 6) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 6) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 14) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 14) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 7) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 7) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 8) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 8) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 9) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 9) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 15) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 15) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 10) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 10) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 11) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 11) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                          xf_ens[(((long) (nall + 12) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = xf_ens[(((long) (nall2 + 12) * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i];
                        }


                    }

                }
              else
                {
                  if ((ierr[(long) i + offset_1638]) != 20 && (ierr[(long) i + offset_1638]) != 0)
                    {
                      {
                        int D_9490;

                        D_9490 = *ensdim;
                        n = 1;
                        while (1)
                          {
                            {
                              bool D_9493;

                              D_9493 = n > D_9490;
                              if (D_9493) break;
                              xf_ens[(((long) n * stride_1694 + offset_1695) + (long) *j * stride_1692) + (long) i] = 0.0f;

                              n = n + 1;
                            }
                          }

                      }
                    }

                }



              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29> static inline void cup_env_clev(thread GFWorkspace &workspace, P0 t, P1 qes, P2 q, P3 he, P4 hes, P5 z, P6 p, P7 qes_cup, P8 q_cup, P9 he_cup, P10 hes_cup, P11 z_cup, P12 p_cup, P13 gamma_cup, P14 t_cup, P15 psur, P16 ierr, P17 z1, P18 xl, P19 rv, P20 cp, P21 itf, P22 jtf, P23 ktf, P24 its, P25 ite, P26 jts, P27 jte, P28 kts, P29 kte) {
  GFFrame frame(workspace);
  long lbound_1747;
  long ubound_1748;
  long lbound_1749;
  long ubound_1750;
  long stride_1751;
  long offset_1752;
  long size_1753;
  long lbound_1754;
  long ubound_1755;
  long lbound_1756;
  long ubound_1757;
  long stride_1758;
  long offset_1759;
  long size_1760;
  long lbound_1761;
  long ubound_1762;
  long lbound_1763;
  long ubound_1764;
  long stride_1765;
  long offset_1766;
  long size_1767;
  long lbound_1768;
  long ubound_1769;
  long lbound_1770;
  long ubound_1771;
  long stride_1772;
  long offset_1773;
  long size_1774;
  long lbound_1775;
  long ubound_1776;
  long lbound_1777;
  long ubound_1778;
  long stride_1779;
  long offset_1780;
  long size_1781;
  int i;
  long lbound_1782;
  long ubound_1783;
  long offset_1784;
  long size_1785;
  int k;
  long lbound_1786;
  long ubound_1787;
  long lbound_1788;
  long ubound_1789;
  long stride_1790;
  long offset_1791;
  long size_1792;
  long lbound_1793;
  long ubound_1794;
  long lbound_1795;
  long ubound_1796;
  long stride_1797;
  long offset_1798;
  long size_1799;
  long lbound_1800;
  long ubound_1801;
  long offset_1802;
  long size_1803;
  long lbound_1804;
  long ubound_1805;
  long lbound_1806;
  long ubound_1807;
  long stride_1808;
  long offset_1809;
  long size_1810;
  long lbound_1811;
  long ubound_1812;
  long lbound_1813;
  long ubound_1814;
  long stride_1815;
  long offset_1816;
  long size_1817;
  long lbound_1818;
  long ubound_1819;
  long lbound_1820;
  long ubound_1821;
  long stride_1822;
  long offset_1823;
  long size_1824;
  long lbound_1825;
  long ubound_1826;
  long lbound_1827;
  long ubound_1828;
  long stride_1829;
  long offset_1830;
  long size_1831;
  long lbound_1832;
  long ubound_1833;
  long lbound_1834;
  long ubound_1835;
  long stride_1836;
  long offset_1837;
  long size_1838;
  long lbound_1839;
  long ubound_1840;
  long lbound_1841;
  long ubound_1842;
  long stride_1843;
  long offset_1844;
  long size_1845;
  long lbound_1846;
  long ubound_1847;
  long lbound_1848;
  long ubound_1849;
  long stride_1850;
  long offset_1851;
  long size_1852;
  long lbound_1853;
  long ubound_1854;
  long offset_1855;
  long size_1856;
  long lbound_1857;
  long ubound_1858;
  long lbound_1859;
  long ubound_1860;
  long stride_1861;
  long offset_1862;
  long size_1863;
  long D_9722;
  ulong D_9723;
  ulong D_9724;
  long D_9725;
  ulong D_9726;
  ulong D_9727;
  long D_9728;
  ulong D_9729;
  ulong D_9730;
  long D_9731;
  ulong D_9732;
  ulong D_9733;
  long D_9734;
  ulong D_9735;
  ulong D_9736;
  long D_9737;
  ulong D_9738;
  ulong D_9739;
  long D_9740;
  ulong D_9741;
  ulong D_9742;
  long D_9743;
  ulong D_9744;
  ulong D_9745;
  long D_9746;
  ulong D_9747;
  ulong D_9748;
  long D_9749;
  ulong D_9750;
  ulong D_9751;
  long D_9752;
  ulong D_9753;
  ulong D_9754;
  long D_9755;
  ulong D_9756;
  ulong D_9757;
  long D_9758;
  ulong D_9759;
  ulong D_9760;
  long D_9761;
  ulong D_9762;
  ulong D_9763;
  long D_9764;
  ulong D_9765;
  ulong D_9766;
  long D_9767;
  ulong D_9768;
  ulong D_9769;
  long D_9770;
  ulong D_9771;
  ulong D_9772;
  long D_9773;
  ulong D_9774;
  ulong D_9775;

  lbound_1825 = (long) *its;
  ubound_1826 = (long) *ite;
  stride_1829 = (1 - lbound_1825) + ubound_1826;
  stride_1829 = gf_max(stride_1829, 0);
  lbound_1827 = (long) *kts;
  ubound_1828 = (long) *kte;
  size_1831 = ((1 - lbound_1827) + ubound_1828) * stride_1829;
  size_1831 = gf_max(size_1831, 0);
  D_9773 = size_1831 + -1;
  D_9774 = (ulong) (ulong) (size_1831) * 32;
  D_9775 = (ulong) (size_1831) * 4;
  offset_1830 = -(lbound_1825) - lbound_1827 * stride_1829;
  lbound_1811 = (long) *its;
  ubound_1812 = (long) *ite;
  stride_1815 = (1 - lbound_1811) + ubound_1812;
  stride_1815 = gf_max(stride_1815, 0);
  lbound_1813 = (long) *kts;
  ubound_1814 = (long) *kte;
  size_1817 = ((1 - lbound_1813) + ubound_1814) * stride_1815;
  size_1817 = gf_max(size_1817, 0);
  D_9770 = size_1817 + -1;
  D_9771 = (ulong) (ulong) (size_1817) * 32;
  D_9772 = (ulong) (size_1817) * 4;
  offset_1816 = -(lbound_1811) - lbound_1813 * stride_1815;
  lbound_1775 = (long) *its;
  ubound_1776 = (long) *ite;
  stride_1779 = (1 - lbound_1775) + ubound_1776;
  stride_1779 = gf_max(stride_1779, 0);
  lbound_1777 = (long) *kts;
  ubound_1778 = (long) *kte;
  size_1781 = ((1 - lbound_1777) + ubound_1778) * stride_1779;
  size_1781 = gf_max(size_1781, 0);
  D_9767 = size_1781 + -1;
  D_9768 = (ulong) (ulong) (size_1781) * 32;
  D_9769 = (ulong) (size_1781) * 4;
  offset_1780 = -(lbound_1775) - lbound_1777 * stride_1779;
  lbound_1761 = (long) *its;
  ubound_1762 = (long) *ite;
  stride_1765 = (1 - lbound_1761) + ubound_1762;
  stride_1765 = gf_max(stride_1765, 0);
  lbound_1763 = (long) *kts;
  ubound_1764 = (long) *kte;
  size_1767 = ((1 - lbound_1763) + ubound_1764) * stride_1765;
  size_1767 = gf_max(size_1767, 0);
  D_9764 = size_1767 + -1;
  D_9765 = (ulong) (ulong) (size_1767) * 32;
  D_9766 = (ulong) (size_1767) * 4;
  offset_1766 = -(lbound_1761) - lbound_1763 * stride_1765;
  lbound_1857 = (long) *its;
  ubound_1858 = (long) *ite;
  stride_1861 = (1 - lbound_1857) + ubound_1858;
  stride_1861 = gf_max(stride_1861, 0);
  lbound_1859 = (long) *kts;
  ubound_1860 = (long) *kte;
  size_1863 = ((1 - lbound_1859) + ubound_1860) * stride_1861;
  size_1863 = gf_max(size_1863, 0);
  D_9761 = size_1863 + -1;
  D_9762 = (ulong) (ulong) (size_1863) * 32;
  D_9763 = (ulong) (size_1863) * 4;
  offset_1862 = -(lbound_1857) - lbound_1859 * stride_1861;
  lbound_1793 = (long) *its;
  ubound_1794 = (long) *ite;
  stride_1797 = (1 - lbound_1793) + ubound_1794;
  stride_1797 = gf_max(stride_1797, 0);
  lbound_1795 = (long) *kts;
  ubound_1796 = (long) *kte;
  size_1799 = ((1 - lbound_1795) + ubound_1796) * stride_1797;
  size_1799 = gf_max(size_1799, 0);
  D_9758 = size_1799 + -1;
  D_9759 = (ulong) (ulong) (size_1799) * 32;
  D_9760 = (ulong) (size_1799) * 4;
  offset_1798 = -(lbound_1793) - lbound_1795 * stride_1797;
  lbound_1839 = (long) *its;
  ubound_1840 = (long) *ite;
  stride_1843 = (1 - lbound_1839) + ubound_1840;
  stride_1843 = gf_max(stride_1843, 0);
  lbound_1841 = (long) *kts;
  ubound_1842 = (long) *kte;
  size_1845 = ((1 - lbound_1841) + ubound_1842) * stride_1843;
  size_1845 = gf_max(size_1845, 0);
  D_9755 = size_1845 + -1;
  D_9756 = (ulong) (ulong) (size_1845) * 32;
  D_9757 = (ulong) (size_1845) * 4;
  offset_1844 = -(lbound_1839) - lbound_1841 * stride_1843;
  lbound_1747 = (long) *its;
  ubound_1748 = (long) *ite;
  stride_1751 = (1 - lbound_1747) + ubound_1748;
  stride_1751 = gf_max(stride_1751, 0);
  lbound_1749 = (long) *kts;
  ubound_1750 = (long) *kte;
  size_1753 = ((1 - lbound_1749) + ubound_1750) * stride_1751;
  size_1753 = gf_max(size_1753, 0);
  D_9752 = size_1753 + -1;
  D_9753 = (ulong) (ulong) (size_1753) * 32;
  D_9754 = (ulong) (size_1753) * 4;
  offset_1752 = -(lbound_1747) - lbound_1749 * stride_1751;
  lbound_1782 = (long) *its;
  ubound_1783 = (long) *ite;
  size_1785 = (1 - lbound_1782) + ubound_1783;
  size_1785 = gf_max(size_1785, 0);
  D_9749 = size_1785 + -1;
  D_9750 = (ulong) (ulong) (size_1785) * 32;
  D_9751 = (ulong) (size_1785) * 4;
  offset_1784 = -(lbound_1782);
  lbound_1818 = (long) *its;
  ubound_1819 = (long) *ite;
  stride_1822 = (1 - lbound_1818) + ubound_1819;
  stride_1822 = gf_max(stride_1822, 0);
  lbound_1820 = (long) *kts;
  ubound_1821 = (long) *kte;
  size_1824 = ((1 - lbound_1820) + ubound_1821) * stride_1822;
  size_1824 = gf_max(size_1824, 0);
  D_9746 = size_1824 + -1;
  D_9747 = (ulong) (ulong) (size_1824) * 32;
  D_9748 = (ulong) (size_1824) * 4;
  offset_1823 = -(lbound_1818) - lbound_1820 * stride_1822;
  lbound_1804 = (long) *its;
  ubound_1805 = (long) *ite;
  stride_1808 = (1 - lbound_1804) + ubound_1805;
  stride_1808 = gf_max(stride_1808, 0);
  lbound_1806 = (long) *kts;
  ubound_1807 = (long) *kte;
  size_1810 = ((1 - lbound_1806) + ubound_1807) * stride_1808;
  size_1810 = gf_max(size_1810, 0);
  D_9743 = size_1810 + -1;
  D_9744 = (ulong) (ulong) (size_1810) * 32;
  D_9745 = (ulong) (size_1810) * 4;
  offset_1809 = -(lbound_1804) - lbound_1806 * stride_1808;
  lbound_1768 = (long) *its;
  ubound_1769 = (long) *ite;
  stride_1772 = (1 - lbound_1768) + ubound_1769;
  stride_1772 = gf_max(stride_1772, 0);
  lbound_1770 = (long) *kts;
  ubound_1771 = (long) *kte;
  size_1774 = ((1 - lbound_1770) + ubound_1771) * stride_1772;
  size_1774 = gf_max(size_1774, 0);
  D_9740 = size_1774 + -1;
  D_9741 = (ulong) (ulong) (size_1774) * 32;
  D_9742 = (ulong) (size_1774) * 4;
  offset_1773 = -(lbound_1768) - lbound_1770 * stride_1772;
  lbound_1754 = (long) *its;
  ubound_1755 = (long) *ite;
  stride_1758 = (1 - lbound_1754) + ubound_1755;
  stride_1758 = gf_max(stride_1758, 0);
  lbound_1756 = (long) *kts;
  ubound_1757 = (long) *kte;
  size_1760 = ((1 - lbound_1756) + ubound_1757) * stride_1758;
  size_1760 = gf_max(size_1760, 0);
  D_9737 = size_1760 + -1;
  D_9738 = (ulong) (ulong) (size_1760) * 32;
  D_9739 = (ulong) (size_1760) * 4;
  offset_1759 = -(lbound_1754) - lbound_1756 * stride_1758;
  lbound_1846 = (long) *its;
  ubound_1847 = (long) *ite;
  stride_1850 = (1 - lbound_1846) + ubound_1847;
  stride_1850 = gf_max(stride_1850, 0);
  lbound_1848 = (long) *kts;
  ubound_1849 = (long) *kte;
  size_1852 = ((1 - lbound_1848) + ubound_1849) * stride_1850;
  size_1852 = gf_max(size_1852, 0);
  D_9734 = size_1852 + -1;
  D_9735 = (ulong) (ulong) (size_1852) * 32;
  D_9736 = (ulong) (size_1852) * 4;
  offset_1851 = -(lbound_1846) - lbound_1848 * stride_1850;
  lbound_1786 = (long) *its;
  ubound_1787 = (long) *ite;
  stride_1790 = (1 - lbound_1786) + ubound_1787;
  stride_1790 = gf_max(stride_1790, 0);
  lbound_1788 = (long) *kts;
  ubound_1789 = (long) *kte;
  size_1792 = ((1 - lbound_1788) + ubound_1789) * stride_1790;
  size_1792 = gf_max(size_1792, 0);
  D_9731 = size_1792 + -1;
  D_9732 = (ulong) (ulong) (size_1792) * 32;
  D_9733 = (ulong) (size_1792) * 4;
  offset_1791 = -(lbound_1786) - lbound_1788 * stride_1790;
  lbound_1832 = (long) *its;
  ubound_1833 = (long) *ite;
  stride_1836 = (1 - lbound_1832) + ubound_1833;
  stride_1836 = gf_max(stride_1836, 0);
  lbound_1834 = (long) *kts;
  ubound_1835 = (long) *kte;
  size_1838 = ((1 - lbound_1834) + ubound_1835) * stride_1836;
  size_1838 = gf_max(size_1838, 0);
  D_9728 = size_1838 + -1;
  D_9729 = (ulong) (ulong) (size_1838) * 32;
  D_9730 = (ulong) (size_1838) * 4;
  offset_1837 = -(lbound_1832) - lbound_1834 * stride_1836;
  lbound_1853 = (long) *its;
  ubound_1854 = (long) *ite;
  size_1856 = (1 - lbound_1853) + ubound_1854;
  size_1856 = gf_max(size_1856, 0);
  D_9725 = size_1856 + -1;
  D_9726 = (ulong) (ulong) (size_1856) * 32;
  D_9727 = (ulong) (size_1856) * 4;
  offset_1855 = -(lbound_1853);
  lbound_1800 = (long) *its;
  ubound_1801 = (long) *ite;
  size_1803 = (1 - lbound_1800) + ubound_1801;
  size_1803 = gf_max(size_1803, 0);
  D_9722 = size_1803 + -1;
  D_9723 = (ulong) (ulong) (size_1803) * 32;
  D_9724 = (ulong) (size_1803) * 4;
  offset_1802 = -(lbound_1800);
  {
    int D_9694;
    int D_9695;

    D_9694 = *kts;
    D_9695 = *ktf;
    k = D_9694;
    while (1)
      {
        {
          bool D_9698;

          D_9698 = k > D_9695;
          if (D_9698) break;
          {
            int D_9699;
            int D_9700;

            D_9699 = *its;
            D_9700 = *itf;
            i = D_9699;
            while (1)
              {
                {
                  bool D_9703;

                  D_9703 = i > D_9700;
                  if (D_9703) break;
                  qes_cup[((long) k * stride_1829 + offset_1830) + (long) i] = 0.0f;
                  q_cup[((long) k * stride_1815 + offset_1816) + (long) i] = 0.0f;
                  hes_cup[((long) k * stride_1779 + offset_1780) + (long) i] = 0.0f;
                  he_cup[((long) k * stride_1765 + offset_1766) + (long) i] = 0.0f;
                  z_cup[((long) k * stride_1861 + offset_1862) + (long) i] = 0.0f;
                  p_cup[((long) k * stride_1797 + offset_1798) + (long) i] = 0.0f;
                  t_cup[((long) k * stride_1843 + offset_1844) + (long) i] = 0.0f;
                  gamma_cup[((long) k * stride_1751 + offset_1752) + (long) i] = 0.0f;

                  i = i + 1;
                }
              }

          }

          k = k + 1;
        }
      }

  }
  {
    int D_9704;
    int D_9705;

    D_9704 = *kts + 1;
    D_9705 = *ktf;
    k = D_9704;
    while (1)
      {
        {
          bool D_9708;

          D_9708 = k > D_9705;
          if (D_9708) break;
          {
            int D_9709;
            int D_9710;

            D_9709 = *its;
            D_9710 = *itf;
            i = D_9709;
            while (1)
              {
                {
                  bool D_9713;

                  D_9713 = i > D_9710;
                  if (D_9713) break;
                  if (ierr[(long) i + offset_1784] == 0)
                    {
                      qes_cup[((long) k * stride_1829 + offset_1830) + (long) i] = ((qes[((long) (k + -1) * stride_1822 + offset_1823) + (long) i] + qes[((long) k * stride_1822 + offset_1823) + (long) i])) * 5.0e-1f;
                      q_cup[((long) k * stride_1815 + offset_1816) + (long) i] = ((q[((long) (k + -1) * stride_1808 + offset_1809) + (long) i] + q[((long) k * stride_1808 + offset_1809) + (long) i])) * 5.0e-1f;
                      hes_cup[((long) k * stride_1779 + offset_1780) + (long) i] = ((hes[((long) (k + -1) * stride_1772 + offset_1773) + (long) i] + hes[((long) k * stride_1772 + offset_1773) + (long) i])) * 5.0e-1f;
                      he_cup[((long) k * stride_1765 + offset_1766) + (long) i] = ((he[((long) (k + -1) * stride_1758 + offset_1759) + (long) i] + he[((long) k * stride_1758 + offset_1759) + (long) i])) * 5.0e-1f;
                      if (he_cup[((long) k * stride_1765 + offset_1766) + (long) i] > hes_cup[((long) k * stride_1779 + offset_1780) + (long) i])
                        {
                          he_cup[((long) k * stride_1765 + offset_1766) + (long) i] = hes_cup[((long) k * stride_1779 + offset_1780) + (long) i];
                        }

                      z_cup[((long) k * stride_1861 + offset_1862) + (long) i] = ((z[((long) (k + -1) * stride_1850 + offset_1851) + (long) i] + z[((long) k * stride_1850 + offset_1851) + (long) i])) * 5.0e-1f;
                      p_cup[((long) k * stride_1797 + offset_1798) + (long) i] = ((p[((long) (k + -1) * stride_1790 + offset_1791) + (long) i] + p[((long) k * stride_1790 + offset_1791) + (long) i])) * 5.0e-1f;
                      t_cup[((long) k * stride_1843 + offset_1844) + (long) i] = ((t[((long) (k + -1) * stride_1836 + offset_1837) + (long) i] + t[((long) k * stride_1836 + offset_1837) + (long) i])) * 5.0e-1f;
                      gamma_cup[((long) k * stride_1751 + offset_1752) + (long) i] = (((*xl / *cp)) * ((*xl / (((*rv * t_cup[((long) k * stride_1843 + offset_1844) + (long) i]) * t_cup[((long) k * stride_1843 + offset_1844) + (long) i]))))) * qes_cup[((long) k * stride_1829 + offset_1830) + (long) i];
                    }


                  i = i + 1;
                }
              }

          }

          k = k + 1;
        }
      }

  }
  {
    int D_9716;
    int D_9717;

    D_9716 = *its;
    D_9717 = *itf;
    i = D_9716;
    while (1)
      {
        {
          bool D_9720;

          D_9720 = i > D_9717;
          if (D_9720) break;
          if (ierr[(long) i + offset_1784] == 0)
            {
              qes_cup[(offset_1830 + (stride_1829)) + (long) i] = qes[(offset_1823 + (stride_1822)) + (long) i];
              q_cup[(offset_1816 + (stride_1815)) + (long) i] = q[(offset_1809 + (stride_1808)) + (long) i];
              hes_cup[(offset_1780 + (stride_1779)) + (long) i] = (z1[(long) i + offset_1855] * 9.81000041961669921875e+0f + t[(offset_1837 + (stride_1836)) + (long) i] * 1.004e+3f) + qes[(offset_1823 + (stride_1822)) + (long) i] * 2.5e+6f;
              he_cup[(offset_1766 + (stride_1765)) + (long) i] = (z1[(long) i + offset_1855] * 9.81000041961669921875e+0f + t[(offset_1837 + (stride_1836)) + (long) i] * 1.004e+3f) + q[(offset_1809 + (stride_1808)) + (long) i] * 2.5e+6f;
              z_cup[(offset_1862 + (stride_1861)) + (long) i] = ((z[(offset_1851 + (stride_1850)) + (long) i] + z1[(long) i + offset_1855])) * 5.0e-1f;
              p_cup[(offset_1798 + (stride_1797)) + (long) i] = ((p[(offset_1791 + (stride_1790)) + (long) i] + psur[(long) i + offset_1802])) * 5.0e-1f;
              z_cup[(offset_1862 + (stride_1861)) + (long) i] = z1[(long) i + offset_1855];
              p_cup[(offset_1798 + (stride_1797)) + (long) i] = psur[(long) i + offset_1802];
              t_cup[(offset_1844 + (stride_1843)) + (long) i] = t[(offset_1837 + (stride_1836)) + (long) i];
              gamma_cup[(offset_1752 + (stride_1751)) + (long) i] = ((*xl / *cp) * ((*xl / (((*rv * t_cup[(offset_1844 + (stride_1843)) + (long) i]) * t_cup[(offset_1844 + (stride_1843)) + (long) i]))))) * qes_cup[(offset_1830 + (stride_1829)) + (long) i];
            }


          i = i + 1;
        }
      }

  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22> static inline void cup_env(thread GFWorkspace &workspace, P0 z, P1 qes, P2 he, P3 hes, P4 t, P5 q, P6 p, P7 z1, P8 psur, P9 ierr, P10 tcrit, P11 itest, P12 xl, P13 cp, P14 itf, P15 jtf, P16 ktf, P17 its, P18 ite, P19 jts, P20 jte, P21 kts, P22 kte) {
  GFFrame frame(workspace);

  device void * D_9934;
  GFReal ae[2];
  GFReal be[2];
  GFReal e;
  long lbound_1864;
  long ubound_1865;
  long lbound_1866;
  long ubound_1867;
  long stride_1868;
  long offset_1869;
  long size_1870;
  long lbound_1871;
  long ubound_1872;
  long lbound_1873;
  long ubound_1874;
  long stride_1875;
  long offset_1876;
  long size_1877;
  GFReal ht[2];
  int i;
  long lbound_1878;
  long ubound_1879;
  long offset_1880;
  long size_1881;
  int iph;
  int k;
  long lbound_1882;
  long ubound_1883;
  long lbound_1884;
  long ubound_1885;
  long stride_1886;
  long offset_1887;
  long size_1888;
  long lbound_1889;
  long ubound_1890;
  long offset_1891;
  long size_1892;
  long lbound_1893;
  long ubound_1894;
  long lbound_1895;
  long ubound_1896;
  long stride_1897;
  long offset_1898;
  long size_1899;
  long lbound_1900;
  long ubound_1901;
  long lbound_1902;
  long ubound_1903;
  long stride_1904;
  long offset_1905;
  long size_1906;
  long lbound_1907;
  long ubound_1908;
  long lbound_1909;
  long ubound_1910;
  long stride_1911;
  long offset_1912;
  long size_1913;
  long lbound_1914;
  long ubound_1915;
  long lbound_1916;
  long ubound_1917;
  long stride_1918;
  long offset_1919;
  long size_1920;

  device GFReal * tv;
  GFReal tvbar;
  long lbound_1921;
  long ubound_1922;
  long lbound_1923;
  long ubound_1924;
  long stride_1925;
  long offset_1926;
  long size_1927;
  long lbound_1928;
  long ubound_1929;
  long offset_1930;
  long size_1931;
  long D_9931;
  ulong D_9932;
  ulong D_9933;
  long D_9935;
  ulong D_9936;
  ulong D_9937;
  long D_9938;
  ulong D_9939;
  ulong D_9940;
  long D_9941;
  ulong D_9942;
  ulong D_9943;
  long D_9944;
  ulong D_9945;
  ulong D_9946;
  long D_9947;
  ulong D_9948;
  ulong D_9949;
  long D_9950;
  ulong D_9951;
  ulong D_9952;
  long D_9953;
  ulong D_9954;
  ulong D_9955;
  long D_9956;
  ulong D_9957;
  ulong D_9958;
  long D_9959;
  ulong D_9960;
  ulong D_9961;
  long D_9962;
  ulong D_9963;
  ulong D_9964;

    {
      lbound_1878 = (long) *its;
      ubound_1879 = (long) *ite;
      size_1881 = (1 - lbound_1878) + ubound_1879;
      size_1881 = gf_max(size_1881, 0);
      D_9962 = size_1881 + -1;
      D_9963 = (ulong) (ulong) (size_1881) * 32;
      D_9964 = (ulong) (size_1881) * 4;
      offset_1880 = -(lbound_1878);
      lbound_1907 = (long) *its;
      ubound_1908 = (long) *ite;
      stride_1911 = (1 - lbound_1907) + ubound_1908;
      stride_1911 = gf_max(stride_1911, 0);
      lbound_1909 = (long) *kts;
      ubound_1910 = (long) *kte;
      size_1913 = ((1 - lbound_1909) + ubound_1910) * stride_1911;
      size_1913 = gf_max(size_1913, 0);
      D_9959 = size_1913 + -1;
      D_9960 = (ulong) (ulong) (size_1913) * 32;
      D_9961 = (ulong) (size_1913) * 4;
      offset_1912 = -(lbound_1907) - lbound_1909 * stride_1911;
      lbound_1900 = (long) *its;
      ubound_1901 = (long) *ite;
      stride_1904 = (1 - lbound_1900) + ubound_1901;
      stride_1904 = gf_max(stride_1904, 0);
      lbound_1902 = (long) *kts;
      ubound_1903 = (long) *kte;
      size_1906 = ((1 - lbound_1902) + ubound_1903) * stride_1904;
      size_1906 = gf_max(size_1906, 0);
      D_9956 = size_1906 + -1;
      D_9957 = (ulong) (ulong) (size_1906) * 32;
      D_9958 = (ulong) (size_1906) * 4;
      offset_1905 = -(lbound_1900) - lbound_1902 * stride_1904;
      lbound_1882 = (long) *its;
      ubound_1883 = (long) *ite;
      stride_1886 = (1 - lbound_1882) + ubound_1883;
      stride_1886 = gf_max(stride_1886, 0);
      lbound_1884 = (long) *kts;
      ubound_1885 = (long) *kte;
      size_1888 = ((1 - lbound_1884) + ubound_1885) * stride_1886;
      size_1888 = gf_max(size_1888, 0);
      D_9953 = size_1888 + -1;
      D_9954 = (ulong) (ulong) (size_1888) * 32;
      D_9955 = (ulong) (size_1888) * 4;
      offset_1887 = -(lbound_1882) - lbound_1884 * stride_1886;
      lbound_1893 = (long) *its;
      ubound_1894 = (long) *ite;
      stride_1897 = (1 - lbound_1893) + ubound_1894;
      stride_1897 = gf_max(stride_1897, 0);
      lbound_1895 = (long) *kts;
      ubound_1896 = (long) *kte;
      size_1899 = ((1 - lbound_1895) + ubound_1896) * stride_1897;
      size_1899 = gf_max(size_1899, 0);
      D_9950 = size_1899 + -1;
      D_9951 = (ulong) (ulong) (size_1899) * 32;
      D_9952 = (ulong) (size_1899) * 4;
      offset_1898 = -(lbound_1893) - lbound_1895 * stride_1897;
      lbound_1921 = (long) *its;
      ubound_1922 = (long) *ite;
      stride_1925 = (1 - lbound_1921) + ubound_1922;
      stride_1925 = gf_max(stride_1925, 0);
      lbound_1923 = (long) *kts;
      ubound_1924 = (long) *kte;
      size_1927 = ((1 - lbound_1923) + ubound_1924) * stride_1925;
      size_1927 = gf_max(size_1927, 0);
      D_9947 = size_1927 + -1;
      D_9948 = (ulong) (ulong) (size_1927) * 32;
      D_9949 = (ulong) (size_1927) * 4;
      offset_1926 = -(lbound_1921) - lbound_1923 * stride_1925;
      lbound_1928 = (long) *its;
      ubound_1929 = (long) *ite;
      size_1931 = (1 - lbound_1928) + ubound_1929;
      size_1931 = gf_max(size_1931, 0);
      D_9944 = size_1931 + -1;
      D_9945 = (ulong) (ulong) (size_1931) * 32;
      D_9946 = (ulong) (size_1931) * 4;
      offset_1930 = -(lbound_1928);
      lbound_1889 = (long) *its;
      ubound_1890 = (long) *ite;
      size_1892 = (1 - lbound_1889) + ubound_1890;
      size_1892 = gf_max(size_1892, 0);
      D_9941 = size_1892 + -1;
      D_9942 = (ulong) (ulong) (size_1892) * 32;
      D_9943 = (ulong) (size_1892) * 4;
      offset_1891 = -(lbound_1889);
      lbound_1864 = (long) *its;
      ubound_1865 = (long) *ite;
      stride_1868 = (1 - lbound_1864) + ubound_1865;
      stride_1868 = gf_max(stride_1868, 0);
      lbound_1866 = (long) *kts;
      ubound_1867 = (long) *kte;
      size_1870 = ((1 - lbound_1866) + ubound_1867) * stride_1868;
      size_1870 = gf_max(size_1870, 0);
      D_9938 = size_1870 + -1;
      D_9939 = (ulong) (ulong) (size_1870) * 32;
      D_9940 = (ulong) (size_1870) * 4;
      offset_1869 = -(lbound_1864) - lbound_1866 * stride_1868;
      lbound_1871 = (long) *its;
      ubound_1872 = (long) *ite;
      stride_1875 = (1 - lbound_1871) + ubound_1872;
      stride_1875 = gf_max(stride_1875, 0);
      lbound_1873 = (long) *kts;
      ubound_1874 = (long) *kte;
      size_1877 = ((1 - lbound_1873) + ubound_1874) * stride_1875;
      size_1877 = gf_max(size_1877, 0);
      D_9935 = size_1877 + -1;
      D_9936 = (ulong) (ulong) (size_1877) * 32;
      D_9937 = (ulong) (size_1877) * 4;
      offset_1876 = -(lbound_1871) - lbound_1873 * stride_1875;
      lbound_1914 = (long) *its;
      ubound_1915 = (long) *ite;
      stride_1918 = (1 - lbound_1914) + ubound_1915;
      stride_1918 = gf_max(stride_1918, 0);
      lbound_1916 = (long) *kts;
      ubound_1917 = (long) *kte;
      size_1920 = ((1 - lbound_1916) + ubound_1917) * stride_1918;
      size_1920 = gf_max(size_1920, 0);
      D_9931 = size_1920 + -1;
      D_9932 = (ulong) (ulong) (size_1920) * 32;
      D_9933 = (ulong) (size_1920) * 4;
      D_9934 = (device void *) workspace.allocate (gf_max((ulong) (size_1920 * 4), 1));
      tv = (device GFReal *) D_9934;
      offset_1919 = -(lbound_1914) - lbound_1916 * stride_1918;
      ht[0] = *xl / *cp;
      ht[1] = 2.834e+6f / *cp;
      be[0] = (ht[0] * 6.219999790191650390625e-1f) / 2.860000133514404296875e-1f;
      ae[0] = be[0] / 2.73e+2f + 6.41462230682373046875e+0f;
      be[1] = (ht[1] * 6.219999790191650390625e-1f) / 2.860000133514404296875e-1f;
      ae[1] = be[1] / 2.73e+2f + 6.41462230682373046875e+0f;
      {
        int D_9867;
        int D_9868;

        D_9867 = *kts;
        D_9868 = *ktf;
        k = D_9867;
        while (1)
          {
            {
              bool D_9871;

              D_9871 = k > D_9868;
              if (D_9871) break;
              {
                int D_9872;
                int D_9873;

                D_9872 = *its;
                D_9873 = *itf;
                i = D_9872;
                while (1)
                  {
                    {
                      bool D_9876;

                      D_9876 = i > D_9873;
                      if (D_9876) break;
                      if (ierr[(long) i + offset_1880] == 0)
                        {
                          iph = 1;
                          if (t[((long) k * stride_1911 + offset_1912) + (long) i] <= *tcrit)
                            {
                              iph = 2;
                            }

                          e = satvap (workspace, &t[((long) k * stride_1911 + offset_1912) + (long) i]);
                          {
                            GFReal M_1932;
                            GFReal D_9880;

                            M_1932 = 9.99999993922529029077850282192230224609375e-9f;
                            D_9880 = ((p[((long) k * stride_1886 + offset_1887) + (long) i] - e));
                            M_1932 = gf_max(D_9880, M_1932);
                            qes[((long) k * stride_1904 + offset_1905) + (long) i] = (e * 6.219999790191650390625e-1f) / M_1932;
                          }
                          if (qes[((long) k * stride_1904 + offset_1905) + (long) i] <= 9.99999993922529029077850282192230224609375e-9f)
                            {
                              qes[((long) k * stride_1904 + offset_1905) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                            }

                          if (qes[((long) k * stride_1904 + offset_1905) + (long) i] < q[((long) k * stride_1897 + offset_1898) + (long) i])
                            {
                              qes[((long) k * stride_1904 + offset_1905) + (long) i] = q[((long) k * stride_1897 + offset_1898) + (long) i];
                            }

                          tv[((long) k * stride_1918 + offset_1919) + (long) i] = t[((long) k * stride_1911 + offset_1912) + (long) i] + (q[((long) k * stride_1897 + offset_1898) + (long) i] * 6.07999980449676513671875e-1f) * t[((long) k * stride_1911 + offset_1912) + (long) i];
                        }


                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      if ((unsigned int) *itest <= 1)
        {
          {
            int D_9884;
            int D_9885;

            D_9884 = *its;
            D_9885 = *itf;
            i = D_9884;
            while (1)
              {
                {
                  bool D_9888;

                  D_9888 = i > D_9885;
                  if (D_9888) break;
                  if (ierr[(long) i + offset_1880] == 0)
                    {
                      {
                        GFReal M_1933;
                        GFReal D_9891;

                        M_1933 = 0.0f;
                        D_9891 = z1[(long) i + offset_1930];
                        M_1933 = gf_max(D_9891, M_1933);
                        z[(offset_1926 + (stride_1925)) + (long) i] = M_1933 - ((((gf_log (p[(offset_1887 + (stride_1886)) + (long) i]) - gf_log (psur[(long) i + offset_1891]))) * 2.87e+2f) * tv[(offset_1919 + (stride_1918)) + (long) i]) / 9.81000041961669921875e+0f;
                      }
                    }


                  i = i + 1;
                }
              }

          }
          {
            int D_9892;
            int D_9893;

            D_9892 = *kts + 1;
            D_9893 = *ktf;
            k = D_9892;
            while (1)
              {
                {
                  bool D_9896;

                  D_9896 = k > D_9893;
                  if (D_9896) break;
                  {
                    int D_9897;
                    int D_9898;

                    D_9897 = *its;
                    D_9898 = *itf;
                    i = D_9897;
                    while (1)
                      {
                        {
                          bool D_9901;

                          D_9901 = i > D_9898;
                          if (D_9901) break;
                          if (ierr[(long) i + offset_1880] == 0)
                            {
                              tvbar = tv[((long) k * stride_1918 + offset_1919) + (long) i] * 5.0e-1f + tv[((long) (k + -1) * stride_1918 + offset_1919) + (long) i] * 5.0e-1f;
                              z[((long) k * stride_1925 + offset_1926) + (long) i] = z[((long) (k + -1) * stride_1925 + offset_1926) + (long) i] - ((((gf_log (p[((long) k * stride_1886 + offset_1887) + (long) i]) - gf_log (p[((long) (k + -1) * stride_1886 + offset_1887) + (long) i]))) * 2.87e+2f) * tvbar) / 9.81000041961669921875e+0f;
                            }


                          i = i + 1;
                        }
                      }

                  }

                  k = k + 1;
                }
              }

          }
        }
      else
        {
          if (*itest == 2)
            {
              {
                int D_9904;
                int D_9905;

                D_9904 = *kts;
                D_9905 = *ktf;
                k = D_9904;
                while (1)
                  {
                    {
                      bool D_9908;

                      D_9908 = k > D_9905;
                      if (D_9908) break;
                      {
                        int D_9909;
                        int D_9910;

                        D_9909 = *its;
                        D_9910 = *itf;
                        i = D_9909;
                        while (1)
                          {
                            {
                              bool D_9913;

                              D_9913 = i > D_9910;
                              if (D_9913) break;
                              if (ierr[(long) i + offset_1880] == 0)
                                {
                                  z[((long) k * stride_1925 + offset_1926) + (long) i] = (((he[((long) k * stride_1868 + offset_1869) + (long) i] - t[((long) k * stride_1911 + offset_1912) + (long) i] * 1.004e+3f) - q[((long) k * stride_1897 + offset_1898) + (long) i] * 2.5e+6f)) / 9.81000041961669921875e+0f;
                                  {
                                    GFReal M_1934;
                                    GFReal D_9916;

                                    M_1934 = 1.000000047497451305389404296875e-3f;
                                    D_9916 = z[((long) k * stride_1925 + offset_1926) + (long) i];
                                    M_1934 = gf_max(D_9916, M_1934);
                                    z[((long) k * stride_1925 + offset_1926) + (long) i] = M_1934;
                                  }
                                }


                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
            }
          else
            {

            }

        }

      {
        int D_9918;
        int D_9919;

        D_9918 = *kts;
        D_9919 = *ktf;
        k = D_9918;
        while (1)
          {
            {
              bool D_9922;

              D_9922 = k > D_9919;
              if (D_9922) break;
              {
                int D_9923;
                int D_9924;

                D_9923 = *its;
                D_9924 = *itf;
                i = D_9923;
                while (1)
                  {
                    {
                      bool D_9927;

                      D_9927 = i > D_9924;
                      if (D_9927) break;
                      if (ierr[(long) i + offset_1880] == 0)
                        {
                          if (*itest <= 0)
                            {
                              he[((long) k * stride_1868 + offset_1869) + (long) i] = (z[((long) k * stride_1925 + offset_1926) + (long) i] * 9.81000041961669921875e+0f + t[((long) k * stride_1911 + offset_1912) + (long) i] * 1.004e+3f) + q[((long) k * stride_1897 + offset_1898) + (long) i] * 2.5e+6f;
                            }

                          hes[((long) k * stride_1875 + offset_1876) + (long) i] = (z[((long) k * stride_1925 + offset_1926) + (long) i] * 9.81000041961669921875e+0f + t[((long) k * stride_1911 + offset_1912) + (long) i] * 1.004e+3f) + qes[((long) k * stride_1904 + offset_1905) + (long) i] * 2.5e+6f;
                          if (he[((long) k * stride_1868 + offset_1869) + (long) i] >= hes[((long) k * stride_1875 + offset_1876) + (long) i])
                            {
                              he[((long) k * stride_1868 + offset_1869) + (long) i] = hes[((long) k * stride_1875 + offset_1876) + (long) i];
                            }

                        }


                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30> static inline void cup_dd_moisture_new(thread GFWorkspace &workspace, P0 ierrc, P1 zd, P2 hcd, P3 hes_cup, P4 qcd, P5 qes_cup, P6 pwd, P7 q_cup, P8 z_cup, P9 dd_massentr, P10 dd_massdetr, P11 jmin, P12 ierr, P13 gamma_cup, P14 pwev, P15 bu, P16 qrcd, P17 q, P18 he, P19 t_cup, P20 iloop, P21 xl, P22 itf, P23 jtf, P24 ktf, P25 its, P26 ite, P27 jts, P28 jte, P29 kts, P30 kte, long _ierrc) {
  GFFrame frame(workspace);
  long lbound_1935;
  long ubound_1936;
  long offset_1937;
  long size_1938;
  long lbound_1939;
  long ubound_1940;
  long lbound_1941;
  long ubound_1942;
  long stride_1943;
  long offset_1944;
  long size_1945;
  long lbound_1946;
  long ubound_1947;
  long lbound_1948;
  long ubound_1949;
  long stride_1950;
  long offset_1951;
  long size_1952;
  GFReal dh;
  GFReal dqeva;
  GFReal dz;
  long lbound_1953;
  long ubound_1954;
  long lbound_1955;
  long ubound_1956;
  long stride_1957;
  long offset_1958;
  long size_1959;
  long lbound_1960;
  long ubound_1961;
  long lbound_1962;
  long ubound_1963;
  long stride_1964;
  long offset_1965;
  long size_1966;
  long lbound_1967;
  long ubound_1968;
  long lbound_1969;
  long ubound_1970;
  long stride_1971;
  long offset_1972;
  long size_1973;
  int i;
  long lbound_1974;
  long ubound_1975;
  long offset_1976;
  long size_1977;
  long lbound_1978;
  long ubound_1979;
  long offset_1980;
  long size_1981;
  long lbound_1982;
  long ubound_1983;
  long offset_1984;
  long size_1985;
  int k;
  int ki;
  long lbound_1986;
  long ubound_1987;
  long lbound_1988;
  long ubound_1989;
  long stride_1990;
  long offset_1991;
  long size_1992;
  long lbound_1993;
  long ubound_1994;
  long offset_1995;
  long size_1996;
  long lbound_1997;
  long ubound_1998;
  long lbound_1999;
  long ubound_2000;
  long stride_2001;
  long offset_2002;
  long size_2003;
  long lbound_2004;
  long ubound_2005;
  long lbound_2006;
  long ubound_2007;
  long stride_2008;
  long offset_2009;
  long size_2010;
  long lbound_2011;
  long ubound_2012;
  long lbound_2013;
  long ubound_2014;
  long stride_2015;
  long offset_2016;
  long size_2017;
  long lbound_2018;
  long ubound_2019;
  long lbound_2020;
  long ubound_2021;
  long stride_2022;
  long offset_2023;
  long size_2024;
  long lbound_2025;
  long ubound_2026;
  long lbound_2027;
  long ubound_2028;
  long stride_2029;
  long offset_2030;
  long size_2031;
  long lbound_2032;
  long ubound_2033;
  long lbound_2034;
  long ubound_2035;
  long stride_2036;
  long offset_2037;
  long size_2038;
  long lbound_2039;
  long ubound_2040;
  long lbound_2041;
  long ubound_2042;
  long stride_2043;
  long offset_2044;
  long size_2045;
  long D_10132;
  ulong D_10133;
  ulong D_10134;
  long D_10135;
  ulong D_10136;
  ulong D_10137;
  long D_10138;
  ulong D_10139;
  ulong D_10140;
  long D_10141;
  ulong D_10142;
  ulong D_10143;
  long D_10144;
  ulong D_10145;
  ulong D_10146;
  long D_10147;
  ulong D_10148;
  ulong D_10149;
  long D_10150;
  ulong D_10151;
  ulong D_10152;
  long D_10153;
  ulong D_10154;
  ulong D_10155;
  long D_10156;
  ulong D_10157;
  ulong D_10158;
  long D_10159;
  ulong D_10160;
  ulong D_10161;
  long D_10162;
  ulong D_10163;
  ulong D_10164;
  long D_10165;
  ulong D_10166;
  ulong D_10167;
  long D_10168;
  ulong D_10169;
  ulong D_10170;
  long D_10171;
  ulong D_10172;
  ulong D_10173;
  long D_10174;
  ulong D_10175;
  ulong D_10176;
  long D_10177;
  ulong D_10178;
  ulong D_10179;
  long D_10180;
  ulong D_10181;
  ulong D_10182;
  long D_10183;
  ulong D_10184;
  ulong D_10185;

  lbound_1935 = (long) *its;
  ubound_1936 = (long) *ite;
  size_1938 = (1 - lbound_1935) + ubound_1936;
  size_1938 = gf_max(size_1938, 0);
  D_10183 = size_1938 + -1;
  D_10184 = (ulong) (ulong) (size_1938) * 32;
  D_10185 = (ulong) (size_1938) * 4;
  offset_1937 = -(lbound_1935);
  lbound_1993 = (long) *its;
  ubound_1994 = (long) *ite;
  size_1996 = (1 - lbound_1993) + ubound_1994;
  size_1996 = gf_max(size_1996, 0);
  D_10180 = size_1996 + -1;
  D_10181 = (ulong) (ulong) (size_1996) * 32;
  D_10182 = (ulong) (size_1996) * 4;
  offset_1995 = -(lbound_1993);
  lbound_2011 = (long) *its;
  ubound_2012 = (long) *ite;
  stride_2015 = (1 - lbound_2011) + ubound_2012;
  stride_2015 = gf_max(stride_2015, 0);
  lbound_2013 = (long) *kts;
  ubound_2014 = (long) *kte;
  size_2017 = ((1 - lbound_2013) + ubound_2014) * stride_2015;
  size_2017 = gf_max(size_2017, 0);
  D_10177 = size_2017 + -1;
  D_10178 = (ulong) (ulong) (size_2017) * 32;
  D_10179 = (ulong) (size_2017) * 4;
  offset_2016 = -(lbound_2011) - lbound_2013 * stride_2015;
  lbound_2025 = (long) *its;
  ubound_2026 = (long) *ite;
  stride_2029 = (1 - lbound_2025) + ubound_2026;
  stride_2029 = gf_max(stride_2029, 0);
  lbound_2027 = (long) *kts;
  ubound_2028 = (long) *kte;
  size_2031 = ((1 - lbound_2027) + ubound_2028) * stride_2029;
  size_2031 = gf_max(size_2031, 0);
  D_10174 = size_2031 + -1;
  D_10175 = (ulong) (ulong) (size_2031) * 32;
  D_10176 = (ulong) (size_2031) * 4;
  offset_2030 = -(lbound_2025) - lbound_2027 * stride_2029;
  lbound_1986 = (long) *its;
  ubound_1987 = (long) *ite;
  stride_1990 = (1 - lbound_1986) + ubound_1987;
  stride_1990 = gf_max(stride_1990, 0);
  lbound_1988 = (long) *kts;
  ubound_1989 = (long) *kte;
  size_1992 = ((1 - lbound_1988) + ubound_1989) * stride_1990;
  size_1992 = gf_max(size_1992, 0);
  D_10171 = size_1992 + -1;
  D_10172 = (ulong) (ulong) (size_1992) * 32;
  D_10173 = (ulong) (size_1992) * 4;
  offset_1991 = -(lbound_1986) - lbound_1988 * stride_1990;
  lbound_1974 = (long) *its;
  ubound_1975 = (long) *ite;
  size_1977 = (1 - lbound_1974) + ubound_1975;
  size_1977 = gf_max(size_1977, 0);
  D_10168 = size_1977 + -1;
  D_10169 = (ulong) (ulong) (size_1977) * 32;
  D_10170 = (ulong) (size_1977) * 4;
  offset_1976 = -(lbound_1974);
  lbound_1982 = (long) *its;
  ubound_1983 = (long) *ite;
  size_1985 = (1 - lbound_1982) + ubound_1983;
  size_1985 = gf_max(size_1985, 0);
  D_10165 = size_1985 + -1;
  D_10166 = (ulong) (ulong) (size_1985) * 32;
  D_10167 = (ulong) (size_1985) * 4;
  offset_1984 = -(lbound_1982);
  lbound_2032 = (long) *its;
  ubound_2033 = (long) *ite;
  stride_2036 = (1 - lbound_2032) + ubound_2033;
  stride_2036 = gf_max(stride_2036, 0);
  lbound_2034 = (long) *kts;
  ubound_2035 = (long) *kte;
  size_2038 = ((1 - lbound_2034) + ubound_2035) * stride_2036;
  size_2038 = gf_max(size_2038, 0);
  D_10162 = size_2038 + -1;
  D_10163 = (ulong) (ulong) (size_2038) * 32;
  D_10164 = (ulong) (size_2038) * 4;
  offset_2037 = -(lbound_2032) - lbound_2034 * stride_2036;
  lbound_2004 = (long) *its;
  ubound_2005 = (long) *ite;
  stride_2008 = (1 - lbound_2004) + ubound_2005;
  stride_2008 = gf_max(stride_2008, 0);
  lbound_2006 = (long) *kts;
  ubound_2007 = (long) *kte;
  size_2010 = ((1 - lbound_2006) + ubound_2007) * stride_2008;
  size_2010 = gf_max(size_2010, 0);
  D_10159 = size_2010 + -1;
  D_10160 = (ulong) (ulong) (size_2010) * 32;
  D_10161 = (ulong) (size_2010) * 4;
  offset_2009 = -(lbound_2004) - lbound_2006 * stride_2008;
  lbound_1960 = (long) *its;
  ubound_1961 = (long) *ite;
  stride_1964 = (1 - lbound_1960) + ubound_1961;
  stride_1964 = gf_max(stride_1964, 0);
  lbound_1962 = (long) *kts;
  ubound_1963 = (long) *kte;
  size_1966 = ((1 - lbound_1962) + ubound_1963) * stride_1964;
  size_1966 = gf_max(size_1966, 0);
  D_10156 = size_1966 + -1;
  D_10157 = (ulong) (ulong) (size_1966) * 32;
  D_10158 = (ulong) (size_1966) * 4;
  offset_1965 = -(lbound_1960) - lbound_1962 * stride_1964;
  lbound_1967 = (long) *its;
  ubound_1968 = (long) *ite;
  stride_1971 = (1 - lbound_1967) + ubound_1968;
  stride_1971 = gf_max(stride_1971, 0);
  lbound_1969 = (long) *kts;
  ubound_1970 = (long) *kte;
  size_1973 = ((1 - lbound_1969) + ubound_1970) * stride_1971;
  size_1973 = gf_max(size_1973, 0);
  D_10153 = size_1973 + -1;
  D_10154 = (ulong) (ulong) (size_1973) * 32;
  D_10155 = (ulong) (size_1973) * 4;
  offset_1972 = -(lbound_1967) - lbound_1969 * stride_1971;
  lbound_2018 = (long) *its;
  ubound_2019 = (long) *ite;
  stride_2022 = (1 - lbound_2018) + ubound_2019;
  stride_2022 = gf_max(stride_2022, 0);
  lbound_2020 = (long) *kts;
  ubound_2021 = (long) *kte;
  size_2024 = ((1 - lbound_2020) + ubound_2021) * stride_2022;
  size_2024 = gf_max(size_2024, 0);
  D_10150 = size_2024 + -1;
  D_10151 = (ulong) (ulong) (size_2024) * 32;
  D_10152 = (ulong) (size_2024) * 4;
  offset_2023 = -(lbound_2018) - lbound_2020 * stride_2022;
  lbound_1953 = (long) *its;
  ubound_1954 = (long) *ite;
  stride_1957 = (1 - lbound_1953) + ubound_1954;
  stride_1957 = gf_max(stride_1957, 0);
  lbound_1955 = (long) *kts;
  ubound_1956 = (long) *kte;
  size_1959 = ((1 - lbound_1955) + ubound_1956) * stride_1957;
  size_1959 = gf_max(size_1959, 0);
  D_10147 = size_1959 + -1;
  D_10148 = (ulong) (ulong) (size_1959) * 32;
  D_10149 = (ulong) (size_1959) * 4;
  offset_1958 = -(lbound_1953) - lbound_1955 * stride_1957;
  lbound_2039 = (long) *its;
  ubound_2040 = (long) *ite;
  stride_2043 = (1 - lbound_2039) + ubound_2040;
  stride_2043 = gf_max(stride_2043, 0);
  lbound_2041 = (long) *kts;
  ubound_2042 = (long) *kte;
  size_2045 = ((1 - lbound_2041) + ubound_2042) * stride_2043;
  size_2045 = gf_max(size_2045, 0);
  D_10144 = size_2045 + -1;
  D_10145 = (ulong) (ulong) (size_2045) * 32;
  D_10146 = (ulong) (size_2045) * 4;
  offset_2044 = -(lbound_2039) - lbound_2041 * stride_2043;
  lbound_1939 = (long) *its;
  ubound_1940 = (long) *ite;
  stride_1943 = (1 - lbound_1939) + ubound_1940;
  stride_1943 = gf_max(stride_1943, 0);
  lbound_1941 = (long) *kts;
  ubound_1942 = (long) *kte;
  size_1945 = ((1 - lbound_1941) + ubound_1942) * stride_1943;
  size_1945 = gf_max(size_1945, 0);
  D_10141 = size_1945 + -1;
  D_10142 = (ulong) (ulong) (size_1945) * 32;
  D_10143 = (ulong) (size_1945) * 4;
  offset_1944 = -(lbound_1939) - lbound_1941 * stride_1943;
  lbound_1946 = (long) *its;
  ubound_1947 = (long) *ite;
  stride_1950 = (1 - lbound_1946) + ubound_1947;
  stride_1950 = gf_max(stride_1950, 0);
  lbound_1948 = (long) *kts;
  ubound_1949 = (long) *kte;
  size_1952 = ((1 - lbound_1948) + ubound_1949) * stride_1950;
  size_1952 = gf_max(size_1952, 0);
  D_10138 = size_1952 + -1;
  D_10139 = (ulong) (ulong) (size_1952) * 32;
  D_10140 = (ulong) (size_1952) * 4;
  offset_1951 = -(lbound_1946) - lbound_1948 * stride_1950;
  lbound_1997 = (long) *its;
  ubound_1998 = (long) *ite;
  stride_2001 = (1 - lbound_1997) + ubound_1998;
  stride_2001 = gf_max(stride_2001, 0);
  lbound_1999 = (long) *kts;
  ubound_2000 = (long) *kte;
  size_2003 = ((1 - lbound_1999) + ubound_2000) * stride_2001;
  size_2003 = gf_max(size_2003, 0);
  D_10135 = size_2003 + -1;
  D_10136 = (ulong) (ulong) (size_2003) * 32;
  D_10137 = (ulong) (size_2003) * 4;
  offset_2002 = -(lbound_1997) - lbound_1999 * stride_2001;
  lbound_1978 = (long) *its;
  ubound_1979 = (long) *ite;
  size_1981 = (1 - lbound_1978) + ubound_1979;
  size_1981 = gf_max(size_1981, 0);
  D_10132 = size_1981 + -1;
  D_10133 = (ulong) (ulong) (size_1981) * 400;
  D_10134 = (ulong) (size_1981) * 50;
  offset_1980 = -(lbound_1978);
  {
    int D_10100;
    int D_10101;

    D_10100 = *its;
    D_10101 = *itf;
    i = D_10100;
    while (1)
      {
        {
          bool D_10104;

          D_10104 = i > D_10101;
          if (D_10104) break;
          bu[(long) i + offset_1937] = 0.0f;
          pwev[(long) i + offset_1995] = 0.0f;

          i = i + 1;
        }
      }

  }
  {
    int D_10105;
    int D_10106;

    D_10105 = *kts;
    D_10106 = *ktf;
    k = D_10105;
    while (1)
      {
        {
          bool D_10109;

          D_10109 = k > D_10106;
          if (D_10109) break;
          {
            int D_10110;
            int D_10111;

            D_10110 = *its;
            D_10111 = *itf;
            i = D_10110;
            while (1)
              {
                {
                  bool D_10114;

                  D_10114 = i > D_10111;
                  if (D_10114) break;
                  qcd[((long) k * stride_2015 + offset_2016) + (long) i] = 0.0f;
                  qrcd[((long) k * stride_2029 + offset_2030) + (long) i] = 0.0f;
                  pwd[((long) k * stride_1990 + offset_1991) + (long) i] = 0.0f;

                  i = i + 1;
                }
              }

          }

          k = k + 1;
        }
      }

  }
  {
    int D_10115;
    int D_10116;

    D_10115 = *its;
    D_10116 = *itf;
    i = D_10115;
    while (1)
      {
        {
          bool D_10119;

          D_10119 = i > D_10116;
          if (D_10119) break;
          if (ierr[(long) i + offset_1976] == 0)
            {
              k = jmin[(long) i + offset_1984];
              dz = z_cup[((long) (k + 1) * stride_2036 + offset_2037) + (long) i] - z_cup[((long) k * stride_2036 + offset_2037) + (long) i];
              qcd[((long) k * stride_2015 + offset_2016) + (long) i] = q_cup[((long) k * stride_2008 + offset_2009) + (long) i];
              dh = hcd[((long) k * stride_1964 + offset_1965) + (long) i] - hes_cup[((long) k * stride_1971 + offset_1972) + (long) i];
              if (dh < 0.0f)
                {
                  qrcd[((long) k * stride_2029 + offset_2030) + (long) i] = ((qes_cup[((long) k * stride_2022 + offset_2023) + (long) i] + (((1.0e+0f / *xl)) * ((gamma_cup[((long) k * stride_1957 + offset_1958) + (long) i] / ((gamma_cup[((long) k * stride_1957 + offset_1958) + (long) i] + 1.0e+0f))))) * dh));
                }
              else
                {
                  qrcd[((long) k * stride_2029 + offset_2030) + (long) i] = qes_cup[((long) k * stride_2022 + offset_2023) + (long) i];
                }

              {
                GFReal M_2046;
                GFReal D_10123;

                M_2046 = 0.0f;
                D_10123 = qcd[((long) k * stride_2015 + offset_2016) + (long) i] - qrcd[((long) k * stride_2029 + offset_2030) + (long) i];
                M_2046 = gf_min(D_10123, M_2046);
                pwd[((long) jmin[(long) i + offset_1984] * stride_1990 + offset_1991) + (long) i] = zd[((long) jmin[(long) i + offset_1984] * stride_2043 + offset_2044) + (long) i] * M_2046;
              }
              qcd[((long) k * stride_2015 + offset_2016) + (long) i] = qrcd[((long) k * stride_2029 + offset_2030) + (long) i];
              pwev[(long) i + offset_1995] = pwev[(long) i + offset_1995] + pwd[((long) jmin[(long) i + offset_1984] * stride_1990 + offset_1991) + (long) i];
              bu[(long) i + offset_1937] = dz * dh;
              {
                int D_10124;

                D_10124 = jmin[(long) i + offset_1984] + -1;
                ki = D_10124;
                while (1)
                  {
                    {
                      bool D_10127;

                      D_10127 = ki <= 0;
                      if (D_10127) break;
                      dz = z_cup[((long) (ki + 1) * stride_2036 + offset_2037) + (long) i] - z_cup[((long) ki * stride_2036 + offset_2037) + (long) i];
                      qcd[((long) ki * stride_2015 + offset_2016) + (long) i] = (((qcd[((long) (ki + 1) * stride_2015 + offset_2016) + (long) i] * zd[((long) (ki + 1) * stride_2043 + offset_2044) + (long) i] - (dd_massdetr[((long) ki * stride_1943 + offset_1944) + (long) i] * 5.0e-1f) * qcd[((long) (ki + 1) * stride_2015 + offset_2016) + (long) i]) + dd_massentr[((long) ki * stride_1950 + offset_1951) + (long) i] * q[((long) ki * stride_2001 + offset_2002) + (long) i])) / (((zd[((long) (ki + 1) * stride_2043 + offset_2044) + (long) i] - dd_massdetr[((long) ki * stride_1943 + offset_1944) + (long) i] * 5.0e-1f) + dd_massentr[((long) ki * stride_1950 + offset_1951) + (long) i]));
                      dh = hcd[((long) ki * stride_1964 + offset_1965) + (long) i] - hes_cup[((long) ki * stride_1971 + offset_1972) + (long) i];
                      bu[(long) i + offset_1937] = bu[(long) i + offset_1937] + dz * dh;
                      qrcd[((long) ki * stride_2029 + offset_2030) + (long) i] = qes_cup[((long) ki * stride_2022 + offset_2023) + (long) i] + (((1.0e+0f / *xl)) * ((gamma_cup[((long) ki * stride_1957 + offset_1958) + (long) i] / ((gamma_cup[((long) ki * stride_1957 + offset_1958) + (long) i] + 1.0e+0f))))) * dh;
                      dqeva = qcd[((long) ki * stride_2015 + offset_2016) + (long) i] - qrcd[((long) ki * stride_2029 + offset_2030) + (long) i];
                      if (dqeva > 0.0f)
                        {
                          dqeva = 0.0f;
                          qrcd[((long) ki * stride_2029 + offset_2030) + (long) i] = qcd[((long) ki * stride_2015 + offset_2016) + (long) i];
                        }

                      pwd[((long) ki * stride_1990 + offset_1991) + (long) i] = zd[((long) ki * stride_2043 + offset_2044) + (long) i] * dqeva;
                      qcd[((long) ki * stride_2015 + offset_2016) + (long) i] = qrcd[((long) ki * stride_2029 + offset_2030) + (long) i];
                      pwev[(long) i + offset_1995] = pwev[(long) i + offset_1995] + pwd[((long) ki * stride_1990 + offset_1991) + (long) i];

                      ki = ki + -1;
                    }
                  }

              }
              if (pwev[(long) i + offset_1995] == 0.0f && *iloop == 1)
                {
                  ierr[(long) i + offset_1976] = 7;
                  gf_memmove ((device char *)&ierrc[(long) i + offset_1980], "problem with buoy in cup_dd_moisture", 36);
                  gf_memset ((device char *)&ierrc[(long) i + offset_1980] + 36, 32, 14);
                }

              if (bu[(long) i + offset_1937] >= 0.0f && *iloop == 1)
                {
                  ierr[(long) i + offset_1976] = 7;
                  gf_memmove ((device char *)&ierrc[(long) i + offset_1980], "problem2 with buoy in cup_dd_moisture", 37);
                  gf_memset ((device char *)&ierrc[(long) i + offset_1980] + 37, 32, 13);
                }

            }



          i = i + 1;
        }
      }

  }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32> static inline void cup_dd_edt(thread GFWorkspace &workspace, P0 ierr, P1 us, P2 vs, P3 z, P4 ktop, P5 kbcon, P6 edt, P7 p, P8 pwav, P9 pw, P10 ccn, P11 pwev, P12 edtmax, P13 edtmin, P14 maxens2, P15 edtc, P16 psum2, P17 psumh, P18 ccnclean, P19 rho, P20 aeroevap, P21 itf, P22 jtf, P23 ktf, P24 j, P25 ipr, P26 jpr, P27 its, P28 ite, P29 jts, P30 jte, P31 kts, P32 kte) {
  GFFrame frame(workspace);

  device void * D_10374;

  device void * D_10378;

  device void * D_10382;
  GFReal aeroadd;
  GFReal alpha3;
  GFReal beta3;
  long lbound_2047;
  long ubound_2048;
  long offset_2049;
  long size_2050;
  long lbound_2051;
  long ubound_2052;
  long offset_2053;
  long size_2054;
  long lbound_2055;
  long ubound_2056;
  long ubound_2057;
  long stride_2058;
  long offset_2059;
  long size_2060;
  GFReal einc;
  int i;
  long lbound_2061;
  long ubound_2062;
  long offset_2063;
  long size_2064;
  int k;
  long lbound_2065;
  long ubound_2066;
  long offset_2067;
  long size_2068;
  int kk;
  long lbound_2069;
  long ubound_2070;
  long offset_2071;
  long size_2072;
  long lbound_2073;
  long ubound_2074;
  long lbound_2075;
  long ubound_2076;
  long stride_2077;
  long offset_2078;
  long size_2079;
  GFReal pef;
  GFReal pefb;
  GFReal pefc;
  GFReal prezk;
  GFReal prop_c;
  long lbound_2080;
  long ubound_2081;
  long offset_2082;
  long size_2083;
  long lbound_2084;
  long ubound_2085;
  long offset_2086;
  long size_2087;
  long lbound_2088;
  long ubound_2089;
  long offset_2090;
  long size_2091;
  long lbound_2092;
  long ubound_2093;
  long offset_2094;
  long size_2095;
  long lbound_2096;
  long ubound_2097;
  long offset_2098;
  long size_2099;

  device GFReal * sdp;
  long lbound_2100;
  long ubound_2101;
  long lbound_2102;
  long ubound_2103;
  long stride_2104;
  long offset_2105;
  long size_2106;
  long lbound_2107;
  long ubound_2108;
  long lbound_2109;
  long ubound_2110;
  long stride_2111;
  long offset_2112;
  long size_2113;
  long lbound_2114;
  long ubound_2115;
  long offset_2116;
  long size_2117;

  device GFReal * vshear;
  long lbound_2118;
  long ubound_2119;
  long offset_2120;
  long size_2121;

  device GFReal * vws;
  long lbound_2122;
  long ubound_2123;
  long lbound_2124;
  long ubound_2125;
  long stride_2126;
  long offset_2127;
  long size_2128;
  GFReal zkbc;
  long D_10371;
  ulong D_10372;
  ulong D_10373;
  long D_10375;
  ulong D_10376;
  ulong D_10377;
  long D_10379;
  ulong D_10380;
  ulong D_10381;
  long D_10383;
  ulong D_10384;
  ulong D_10385;
  long D_10386;
  ulong D_10387;
  ulong D_10388;
  long D_10389;
  ulong D_10390;
  ulong D_10391;
  long D_10392;
  ulong D_10393;
  ulong D_10394;
  long D_10395;
  ulong D_10396;
  ulong D_10397;
  long D_10398;
  ulong D_10399;
  ulong D_10400;
  long D_10401;
  ulong D_10402;
  ulong D_10403;
  long D_10404;
  ulong D_10405;
  ulong D_10406;
  long D_10407;
  ulong D_10408;
  ulong D_10409;
  long D_10410;
  ulong D_10411;
  ulong D_10412;
  long D_10413;
  ulong D_10414;
  ulong D_10415;
  long D_10416;
  ulong D_10417;
  ulong D_10418;
  long D_10419;
  ulong D_10420;
  ulong D_10421;
  long D_10422;
  ulong D_10423;
  ulong D_10424;

    {
      lbound_2051 = (long) *its;
      ubound_2052 = (long) *ite;
      size_2054 = (1 - lbound_2051) + ubound_2052;
      size_2054 = gf_max(size_2054, 0);
      D_10422 = size_2054 + -1;
      D_10423 = (ulong) (ulong) (size_2054) * 32;
      D_10424 = (ulong) (size_2054) * 4;
      offset_2053 = -(lbound_2051);
      lbound_2055 = (long) *its;
      ubound_2056 = (long) *ite;
      stride_2058 = (1 - lbound_2055) + ubound_2056;
      stride_2058 = gf_max(stride_2058, 0);
      ubound_2057 = (long) *maxens2;
      size_2060 = stride_2058 * (ubound_2057);
      size_2060 = gf_max(size_2060, 0);
      D_10419 = size_2060 + -1;
      D_10420 = (ulong) (ulong) (size_2060) * 32;
      D_10421 = (ulong) (size_2060) * 4;
      offset_2059 = -(lbound_2055) - (stride_2058);
      lbound_2061 = (long) *its;
      ubound_2062 = (long) *ite;
      size_2064 = (1 - lbound_2061) + ubound_2062;
      size_2064 = gf_max(size_2064, 0);
      D_10416 = size_2064 + -1;
      D_10417 = (ulong) (ulong) (size_2064) * 32;
      D_10418 = (ulong) (size_2064) * 4;
      offset_2063 = -(lbound_2061);
      lbound_2069 = (long) *its;
      ubound_2070 = (long) *ite;
      size_2072 = (1 - lbound_2069) + ubound_2070;
      size_2072 = gf_max(size_2072, 0);
      D_10413 = size_2072 + -1;
      D_10414 = (ulong) (ulong) (size_2072) * 32;
      D_10415 = (ulong) (size_2072) * 4;
      offset_2071 = -(lbound_2069);
      lbound_2065 = (long) *its;
      ubound_2066 = (long) *ite;
      size_2068 = (1 - lbound_2065) + ubound_2066;
      size_2068 = gf_max(size_2068, 0);
      D_10410 = size_2068 + -1;
      D_10411 = (ulong) (ulong) (size_2068) * 32;
      D_10412 = (ulong) (size_2068) * 4;
      offset_2067 = -(lbound_2065);
      lbound_2100 = (long) *its;
      ubound_2101 = (long) *ite;
      stride_2104 = (1 - lbound_2100) + ubound_2101;
      stride_2104 = gf_max(stride_2104, 0);
      lbound_2102 = (long) *kts;
      ubound_2103 = (long) *kte;
      size_2106 = ((1 - lbound_2102) + ubound_2103) * stride_2104;
      size_2106 = gf_max(size_2106, 0);
      D_10407 = size_2106 + -1;
      D_10408 = (ulong) (ulong) (size_2106) * 32;
      D_10409 = (ulong) (size_2106) * 4;
      offset_2105 = -(lbound_2100) - lbound_2102 * stride_2104;
      lbound_2122 = (long) *its;
      ubound_2123 = (long) *ite;
      stride_2126 = (1 - lbound_2122) + ubound_2123;
      stride_2126 = gf_max(stride_2126, 0);
      lbound_2124 = (long) *kts;
      ubound_2125 = (long) *kte;
      size_2128 = ((1 - lbound_2124) + ubound_2125) * stride_2126;
      size_2128 = gf_max(size_2128, 0);
      D_10404 = size_2128 + -1;
      D_10405 = (ulong) (ulong) (size_2128) * 32;
      D_10406 = (ulong) (size_2128) * 4;
      offset_2127 = -(lbound_2122) - lbound_2124 * stride_2126;
      lbound_2107 = (long) *its;
      ubound_2108 = (long) *ite;
      stride_2111 = (1 - lbound_2107) + ubound_2108;
      stride_2111 = gf_max(stride_2111, 0);
      lbound_2109 = (long) *kts;
      ubound_2110 = (long) *kte;
      size_2113 = ((1 - lbound_2109) + ubound_2110) * stride_2111;
      size_2113 = gf_max(size_2113, 0);
      D_10401 = size_2113 + -1;
      D_10402 = (ulong) (ulong) (size_2113) * 32;
      D_10403 = (ulong) (size_2113) * 4;
      offset_2112 = -(lbound_2107) - lbound_2109 * stride_2111;
      lbound_2073 = (long) *its;
      ubound_2074 = (long) *ite;
      stride_2077 = (1 - lbound_2073) + ubound_2074;
      stride_2077 = gf_max(stride_2077, 0);
      lbound_2075 = (long) *kts;
      ubound_2076 = (long) *kte;
      size_2079 = ((1 - lbound_2075) + ubound_2076) * stride_2077;
      size_2079 = gf_max(size_2079, 0);
      D_10398 = size_2079 + -1;
      D_10399 = (ulong) (ulong) (size_2079) * 32;
      D_10400 = (ulong) (size_2079) * 4;
      offset_2078 = -(lbound_2073) - lbound_2075 * stride_2077;
      lbound_2084 = (long) *its;
      ubound_2085 = (long) *ite;
      size_2087 = (1 - lbound_2084) + ubound_2085;
      size_2087 = gf_max(size_2087, 0);
      D_10395 = size_2087 + -1;
      D_10396 = (ulong) (ulong) (size_2087) * 32;
      D_10397 = (ulong) (size_2087) * 4;
      offset_2086 = -(lbound_2084);
      lbound_2047 = (long) *its;
      ubound_2048 = (long) *ite;
      size_2050 = (1 - lbound_2047) + ubound_2048;
      size_2050 = gf_max(size_2050, 0);
      D_10392 = size_2050 + -1;
      D_10393 = (ulong) (ulong) (size_2050) * 32;
      D_10394 = (ulong) (size_2050) * 4;
      offset_2049 = -(lbound_2047);
      lbound_2080 = (long) *its;
      ubound_2081 = (long) *ite;
      size_2083 = (1 - lbound_2080) + ubound_2081;
      size_2083 = gf_max(size_2083, 0);
      D_10389 = size_2083 + -1;
      D_10390 = (ulong) (ulong) (size_2083) * 32;
      D_10391 = (ulong) (size_2083) * 4;
      offset_2082 = -(lbound_2080);
      lbound_2088 = (long) *its;
      ubound_2089 = (long) *ite;
      size_2091 = (1 - lbound_2088) + ubound_2089;
      size_2091 = gf_max(size_2091, 0);
      D_10386 = size_2091 + -1;
      D_10387 = (ulong) (ulong) (size_2091) * 32;
      D_10388 = (ulong) (size_2091) * 4;
      offset_2090 = -(lbound_2088);
      lbound_2092 = (long) *its;
      ubound_2093 = (long) *ite;
      size_2095 = (1 - lbound_2092) + ubound_2093;
      size_2095 = gf_max(size_2095, 0);
      D_10383 = size_2095 + -1;
      D_10384 = (ulong) (ulong) (size_2095) * 32;
      D_10385 = (ulong) (size_2095) * 4;
      offset_2094 = -(lbound_2092);
      lbound_2096 = (long) *its;
      ubound_2097 = (long) *ite;
      size_2099 = (1 - lbound_2096) + ubound_2097;
      size_2099 = gf_max(size_2099, 0);
      D_10379 = size_2099 + -1;
      D_10380 = (ulong) (ulong) (size_2099) * 32;
      D_10381 = (ulong) (size_2099) * 4;
      D_10382 = (device void *) workspace.allocate (gf_max((ulong) (size_2099 * 4), 1));
      sdp = (device GFReal *) D_10382;
      offset_2098 = -(lbound_2096);
      lbound_2114 = (long) *its;
      ubound_2115 = (long) *ite;
      size_2117 = (1 - lbound_2114) + ubound_2115;
      size_2117 = gf_max(size_2117, 0);
      D_10375 = size_2117 + -1;
      D_10376 = (ulong) (ulong) (size_2117) * 32;
      D_10377 = (ulong) (size_2117) * 4;
      D_10378 = (device void *) workspace.allocate (gf_max((ulong) (size_2117 * 4), 1));
      vshear = (device GFReal *) D_10378;
      offset_2116 = -(lbound_2114);
      lbound_2118 = (long) *its;
      ubound_2119 = (long) *ite;
      size_2121 = (1 - lbound_2118) + ubound_2119;
      size_2121 = gf_max(size_2121, 0);
      D_10371 = size_2121 + -1;
      D_10372 = (ulong) (ulong) (size_2121) * 32;
      D_10373 = (ulong) (size_2121) * 4;
      D_10374 = (device void *) workspace.allocate (gf_max((ulong) (size_2121 * 4), 1));
      vws = (device GFReal *) D_10374;
      offset_2120 = -(lbound_2118);
      prop_c = 8.0e+0f;
      alpha3 = 1.89999997615814208984375e+0f;
      beta3 = -1.12999999523162841796875e+0f;
      pefc = 0.0f;
      {
        int D_10301;
        int D_10302;

        D_10301 = *its;
        D_10302 = *itf;
        i = D_10301;
        while (1)
          {
            {
              bool D_10305;

              D_10305 = i > D_10302;
              if (D_10305) break;
              edt[(long) i + offset_2053] = 0.0f;
              vws[(long) i + offset_2120] = 0.0f;
              sdp[(long) i + offset_2098] = 0.0f;
              vshear[(long) i + offset_2116] = 0.0f;

              i = i + 1;
            }
          }

      }
      {
        int D_10306;

        D_10306 = *maxens2;
        k = 1;
        while (1)
          {
            {
              bool D_10309;

              D_10309 = k > D_10306;
              if (D_10309) break;
              {
                int D_10310;
                int D_10311;

                D_10310 = *its;
                D_10311 = *itf;
                i = D_10310;
                while (1)
                  {
                    {
                      bool D_10314;

                      D_10314 = i > D_10311;
                      if (D_10314) break;
                      edtc[((long) k * stride_2058 + offset_2059) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_10315;
        int D_10316;

        D_10315 = *kts;
        D_10316 = *ktf + -1;
        kk = D_10315;
        while (1)
          {
            {
              bool D_10319;

              D_10319 = kk > D_10316;
              if (D_10319) break;
              {
                int D_10320;
                int D_10321;

                D_10320 = *its;
                D_10321 = *itf;
                i = D_10320;
                while (1)
                  {
                    {
                      bool D_10324;

                      D_10324 = i > D_10321;
                      if (D_10324) break;
                      do {
if (ierr[(long) i + offset_2063] != 0) break;

                      {
                        int D_10328;
                        int M_2129;
                        int D_10330;

                        D_10328 = ktop[(long) i + offset_2071];
                        M_2129 = D_10328;
                        D_10330 = *ktf;
                        M_2129 = gf_min(D_10330, M_2129);
                        if (kk <= M_2129 && kbcon[(long) i + offset_2067] <= kk)
                          {
                            vws[(long) i + offset_2120] = vws[(long) i + offset_2120] + ((gf_abs(((us[((long) (kk + 1) * stride_2104 + offset_2105) + (long) i] - us[((long) kk * stride_2104 + offset_2105) + (long) i])) / ((z[((long) (kk + 1) * stride_2126 + offset_2127) + (long) i] - z[((long) kk * stride_2126 + offset_2127) + (long) i]))) + gf_abs(((vs[((long) (kk + 1) * stride_2111 + offset_2112) + (long) i] - vs[((long) kk * stride_2111 + offset_2112) + (long) i])) / ((z[((long) (kk + 1) * stride_2126 + offset_2127) + (long) i] - z[((long) kk * stride_2126 + offset_2127) + (long) i]))))) * ((p[((long) kk * stride_2077 + offset_2078) + (long) i] - p[((long) (kk + 1) * stride_2077 + offset_2078) + (long) i]));
                            sdp[(long) i + offset_2098] = (sdp[(long) i + offset_2098] + p[((long) kk * stride_2077 + offset_2078) + (long) i]) - p[((long) (kk + 1) * stride_2077 + offset_2078) + (long) i];
                          }
                      }

                      if (*ktf + -1 == kk)
                        {
                          vshear[(long) i + offset_2116] = (vws[(long) i + offset_2120] * 1.0e+3f) / sdp[(long) i + offset_2098];
                        }

                      } while(false);

                      i = i + 1;
                    }
                  }

              }

              kk = kk + 1;
            }
          }

      }
      {
        int D_10332;
        int D_10333;

        D_10332 = *its;
        D_10333 = *itf;
        i = D_10332;
        while (1)
          {
            {
              bool D_10336;

              D_10336 = i > D_10333;
              if (D_10336) break;
              if (ierr[(long) i + offset_2063] == 0)
                {
                  {
                    GFReal D_10338;
                    GFReal D_10339;
                    GFReal D_10340;

                    D_10338 = vshear[(long) i + offset_2116];
                    D_10339 = D_10338 * D_10338;
                    D_10340 = vshear[(long) i + offset_2116];
                    pef = ((((1.59099996089935302734375e+0f - vshear[(long) i + offset_2116] * 6.38999998569488525390625e-1f) + ((D_10339)) * 9.5299996435642242431640625e-2f) - ((gf_powi (D_10340, 3))) * 4.9600000493228435516357421875e-3f));
                  }
                  if (pef > 8.9999997615814208984375e-1f)
                    {
                      pef = 8.9999997615814208984375e-1f;
                    }

                  if (pef < 1.00000001490116119384765625e-1f)
                    {
                      pef = 1.00000001490116119384765625e-1f;
                    }

                  zkbc = z[((long) kbcon[(long) i + offset_2067] * stride_2126 + offset_2127) + (long) i] * 3.281000070273876190185546875e-3f;
                  prezk = 1.99999995529651641845703125e-2f;
                  if (zkbc > 3.0e+0f)
                    {
                      prezk = ((((((((4.2771999142132699489593505859375e-4f - zkbc * 5.44000022273394279181957244873046875e-6f)) * zkbc - 1.256979815661907196044921875e-2f)) * zkbc + 1.6217990219593048095703125e-1f)) * zkbc - 7.00341641902923583984375e-1f)) * zkbc + 9.672935009002685546875e-1f;
                    }

                  if (zkbc > 2.5e+1f)
                    {
                      prezk = 2.400000095367431640625e+0f;
                    }

                  pefb = 1.0e+0f / ((prezk + 1.0e+0f));
                  if (pefb > 8.9999997615814208984375e-1f)
                    {
                      pefb = 8.9999997615814208984375e-1f;
                    }

                  if (pefb < 1.00000001490116119384765625e-1f)
                    {
                      pefb = 1.00000001490116119384765625e-1f;
                    }

                  edt[(long) i + offset_2053] = 1.0e+0f - ((pefb + pef)) * 5.0e-1f;
                  if (*aeroevap > 1)
                    {
                      {
                        GFReal D_10348;
                        GFReal D_10349;

                        D_10348 = *ccnclean;
                        D_10349 = ((psumh[(long) i + offset_2086]));
                        aeroadd = ((gf_pow (D_10348, beta3))) * ((gf_pow (D_10349, ((alpha3 - 1.0e+0f)))));
                      }
                      prop_c = (((pefb + pef)) * 5.0e-1f) / aeroadd;
                      {
                        GFReal D_10350;
                        GFReal D_10351;

                        D_10350 = ccn[(long) i + offset_2049];
                        D_10351 = ((psum2[(long) i + offset_2082]));
                        aeroadd = ((gf_pow (D_10350, beta3))) * ((gf_pow (D_10351, ((alpha3 - 1.0e+0f)))));
                      }
                      aeroadd = prop_c * aeroadd;
                      pefc = aeroadd;
                      if (pefc > 8.9999997615814208984375e-1f)
                        {
                          pefc = 8.9999997615814208984375e-1f;
                        }

                      if (pefc < 1.00000001490116119384765625e-1f)
                        {
                          pefc = 1.00000001490116119384765625e-1f;
                        }

                      edt[(long) i + offset_2053] = 1.0e+0f - pefc;
                      if (*aeroevap == 2)
                        {
                          edt[(long) i + offset_2053] = 1.0e+0f - (((pefb + pef) + pefc * 2.0e+0f)) * 2.5e-1f;
                        }

                    }

                  einc = edt[(long) i + offset_2053] * 2.0000000298023223876953125e-1f;
                  {
                    int D_10355;

                    D_10355 = *maxens2;
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_10358;

                          D_10358 = k > D_10355;
                          if (D_10358) break;
                          edtc[((long) k * stride_2058 + offset_2059) + (long) i] = edt[(long) i + offset_2053] + (GFReal) (k + -2) * einc;

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_10359;
        int D_10360;

        D_10359 = *its;
        D_10360 = *itf;
        i = D_10359;
        while (1)
          {
            {
              bool D_10363;

              D_10363 = i > D_10360;
              if (D_10363) break;
              if (ierr[(long) i + offset_2063] == 0)
                {
                  {
                    int D_10365;

                    D_10365 = *maxens2;
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_10368;

                          D_10368 = k > D_10365;
                          if (D_10368) break;
                          edtc[((long) k * stride_2058 + offset_2059) + (long) i] = -((edtc[((long) k * stride_2058 + offset_2059) + (long) i] * pwav[(long) i + offset_2090]) / pwev[(long) i + offset_2094]);
                          if (edtc[((long) k * stride_2058 + offset_2059) + (long) i] > *edtmax)
                            {
                              edtc[((long) k * stride_2058 + offset_2059) + (long) i] = *edtmax;
                            }

                          if (edtc[((long) k * stride_2058 + offset_2059) + (long) i] < *edtmin)
                            {
                              edtc[((long) k * stride_2058 + offset_2059) + (long) i] = *edtmin;
                            }


                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55,typename P56,typename P57,typename P58,typename P59,typename P60,typename P61,typename P62,typename P63,typename P64,typename P65,typename P66,typename P67,typename P68,typename P69,typename P70,typename P71,typename P72,typename P73,typename P74,typename P75,typename P76> static inline void cup_gf(thread GFWorkspace &workspace, P0 zo, P1 outqc, P2 j, P3 aaeq, P4 t, P5 q, P6 z1, P7 sub_mas, P8 tn, P9 qo, P10 po, P11 pre, P12 p, P13 outt, P14 outq, P15 dtime, P16 ktau, P17 psur, P18 us, P19 vs, P20 tcrit, P21 iens, P22 ztexec, P23 zqexec, P24 ccn, P25 ccnclean, P26 rho, P27 dxcell, P28 areacell, P29 mconv, P30 omeg, P31 maxiens, P32 maxens, P33 maxens2, P34 maxens3, P35 ensdim, P36 apr_gr, P37 apr_w, P38 apr_mc, P39 apr_st, P40 apr_as, P41 apr_capma, P42 apr_capme, P43 apr_capmi, P44 kbcon, P45 ktop, P46 cupclw, P47 xf_ens, P48 pr_ens, P49 xland, P50 gsw, P51 subt, P52 subq, P53 xl, P54 rv, P55 cp, P56 g, P57 ichoice, P58 ipr, P59 jpr, P60 ierrc, P61 ens4, P62 beta, P63 autoconv, P64 aeroevap, P65 itf, P66 jtf, P67 ktf, P68 training, P69 xmb, P70 use_excess, P71 its, P72 ite, P73 jts, P74 jte, P75 kts, P76 kte, long _ierrc) {
  GFFrame frame(workspace);

  device void * D_12415;

  device void * D_12419;

  device void * D_12423;

  device void * D_12427;

  device void * D_12431;

  device void * D_12435;

  device void * D_12439;

  device void * D_12443;

  device void * D_12447;

  device void * D_12451;

  device void * D_12455;

  device void * D_12459;

  device void * D_12463;

  device void * D_12467;

  device void * D_12471;

  device void * D_12475;

  device void * D_12479;

  device void * D_12483;

  device void * D_12487;

  device void * D_12491;

  device void * D_12495;

  device void * D_12499;

  device void * D_12503;

  device void * D_12507;

  device void * D_12511;

  device void * D_12515;

  device void * D_12519;

  device void * D_12523;

  device void * D_12527;

  device void * D_12531;

  device void * D_12535;

  device void * D_12539;

  device void * D_12543;

  device void * D_12547;

  device void * D_12551;

  device void * D_12555;

  device void * D_12559;

  device void * D_12563;

  device void * D_12567;

  device void * D_12571;

  device void * D_12575;

  device void * D_12579;

  device void * D_12583;

  device void * D_12587;

  device void * D_12591;

  device void * D_12595;

  device void * D_12599;

  device void * D_12603;

  device void * D_12607;

  device void * D_12611;

  device void * D_12615;

  device void * D_12619;

  device void * D_12623;

  device void * D_12627;

  device void * D_12631;

  device void * D_12635;

  device void * D_12639;

  device void * D_12643;

  device void * D_12647;

  device void * D_12651;

  device void * D_12655;

  device void * D_12659;

  device void * D_12663;

  device void * D_12667;

  device void * D_12671;

  device void * D_12675;

  device void * D_12679;

  device void * D_12683;

  device void * D_12687;

  device void * D_12691;

  device void * D_12695;

  device void * D_12699;

  device void * D_12703;

  device void * D_12707;

  device void * D_12711;

  device void * D_12715;

  device void * D_12719;

  device void * D_12723;

  device void * D_12727;

  device void * D_12731;

  device void * D_12735;

  device void * D_12739;

  device void * D_12743;

  device void * D_12747;

  device void * D_12751;

  device void * D_12755;

  device void * D_12759;

  device void * D_12763;

  device void * D_12767;

  device void * D_12771;

  device void * D_12775;

  device void * D_12779;

  device void * D_12783;

  device void * D_12787;

  device void * D_12791;

  device void * D_12795;

  device void * D_12799;

  device void * D_12803;

  device void * D_12807;

  device void * D_12811;

  device void * D_12815;

  device void * D_12819;

  device void * D_12823;

  device void * D_12827;

  device void * D_12831;

  device void * D_12835;

  device void * D_12839;

  device void * D_12843;

  device void * D_12847;

  device void * D_12851;

  device void * D_12855;

  device void * D_12859;

  device void * D_12863;

  device void * D_12867;

  device void * D_12871;

  device void * D_12875;

  device void * D_12879;

  device void * D_12883;

  device void * D_12887;

  device void * D_12891;

  device void * D_12895;

  device void * D_12899;

  device void * D_12903;

  device void * D_12907;

  device void * D_12911;

  device void * D_12915;

  device void * D_12919;

  device void * D_12923;

  device void * D_12927;
  long lbound_2130;
  long ubound_2131;
  long offset_2132;
  long size_2133;

  device GFReal * aa0;
  long lbound_2134;
  long ubound_2135;
  long offset_2136;
  long size_2137;

  device GFReal * aa1;
  long lbound_2138;
  long ubound_2139;
  long offset_2140;
  long size_2141;
  long lbound_2142;
  long ubound_2143;
  long lbound_2144;
  long ubound_2145;
  long stride_2146;
  long offset_2147;
  long size_2148;
  long lbound_2149;
  long ubound_2150;
  long lbound_2151;
  long ubound_2152;
  long stride_2153;
  long offset_2154;
  long size_2155;
  long lbound_2156;
  long ubound_2157;
  long lbound_2158;
  long ubound_2159;
  long stride_2160;
  long offset_2161;
  long size_2162;
  long lbound_2163;
  long ubound_2164;
  long lbound_2165;
  long ubound_2166;
  long stride_2167;
  long offset_2168;
  long size_2169;
  long lbound_2170;
  long ubound_2171;
  long lbound_2172;
  long ubound_2173;
  long stride_2174;
  long offset_2175;
  long size_2176;
  long lbound_2177;
  long ubound_2178;
  long lbound_2179;
  long ubound_2180;
  long stride_2181;
  long offset_2182;
  long size_2183;
  long lbound_2184;
  long ubound_2185;
  long lbound_2186;
  long ubound_2187;
  long stride_2188;
  long offset_2189;
  long size_2190;
  long lbound_2191;
  long ubound_2192;
  long lbound_2193;
  long ubound_2194;
  long stride_2195;
  long offset_2196;
  long size_2197;
  long lbound_2198;
  long ubound_2199;
  long ubound_2200;
  long stride_2201;
  long offset_2202;
  long size_2203;

  device GFReal * axx;
  long lbound_2204;
  long ubound_2205;
  long offset_2206;
  long size_2207;

  device GFReal * bu;
  long lbound_2208;
  long ubound_2209;
  long offset_2210;
  long size_2211;

  device GFReal * bud;
  long lbound_2212;
  long ubound_2213;
  long offset_2214;
  long size_2215;

  device GFReal * cap_max;
  long lbound_2216;
  long ubound_2217;
  long offset_2218;
  long size_2219;

  device GFReal * cap_max_increment;
  GFReal cap_maxs;
  long lbound_2220;
  long ubound_2221;
  long offset_2222;
  long size_2223;
  long lbound_2224;
  long ubound_2225;
  long lbound_2226;
  long ubound_2227;
  long stride_2228;
  long offset_2229;
  long size_2230;

  device GFReal * cd;
  long lbound_2231;
  long ubound_2232;
  long lbound_2233;
  long ubound_2234;
  long stride_2235;
  long offset_2236;
  long size_2237;

  device GFReal * cdd;
  long lbound_2238;
  long ubound_2239;
  long offset_2240;
  long size_2241;

  device GFReal * closure_n;
  long lbound_2242;
  long ubound_2243;
  long lbound_2244;
  long ubound_2245;
  long stride_2246;
  long offset_2247;
  long size_2248;

  device GFReal * clw_all;
  long lbound_2249;
  long ubound_2250;
  long lbound_2251;
  long ubound_2252;
  long stride_2253;
  long offset_2254;
  long size_2255;
  GFReal day;
  long lbound_2256;
  long ubound_2257;
  long lbound_2258;
  long ubound_2259;
  long stride_2260;
  long offset_2261;
  long size_2262;

  device GFReal * dby;
  long lbound_2263;
  long ubound_2264;
  long lbound_2265;
  long ubound_2266;
  long stride_2267;
  long offset_2268;
  long size_2269;

  device GFReal * dbydo;
  long lbound_2270;
  long ubound_2271;
  long lbound_2272;
  long ubound_2273;
  long stride_2274;
  long offset_2275;
  long size_2276;

  device GFReal * dbyo;
  long lbound_2277;
  long ubound_2278;
  long lbound_2279;
  long ubound_2280;
  long stride_2281;
  long offset_2282;
  long size_2283;

  device GFReal * dd_massdetr;
  long lbound_2284;
  long ubound_2285;
  long lbound_2286;
  long ubound_2287;
  long stride_2288;
  long offset_2289;
  long size_2290;

  device GFReal * dd_massdetro;
  long lbound_2291;
  long ubound_2292;
  long lbound_2293;
  long ubound_2294;
  long stride_2295;
  long offset_2296;
  long size_2297;

  device GFReal * dd_massentr;
  long lbound_2298;
  long ubound_2299;
  long lbound_2300;
  long ubound_2301;
  long stride_2302;
  long offset_2303;
  long size_2304;

  device GFReal * dd_massentro;
  long lbound_2305;
  long ubound_2306;
  long lbound_2307;
  long ubound_2308;
  long stride_2309;
  long offset_2310;
  long size_2311;

  device GFReal * dellah;
  long lbound_2312;
  long ubound_2313;
  long lbound_2314;
  long ubound_2315;
  long stride_2316;
  long offset_2317;
  long size_2318;

  device GFReal * dellaq;
  long lbound_2319;
  long ubound_2320;
  long lbound_2321;
  long ubound_2322;
  long stride_2323;
  long ubound_2324;
  long stride_2325;
  long offset_2326;
  long size_2327;

  device GFReal * dellaq_ens;
  long lbound_2328;
  long ubound_2329;
  long lbound_2330;
  long ubound_2331;
  long stride_2332;
  long offset_2333;
  long size_2334;

  device GFReal * dellaqc;
  long lbound_2335;
  long ubound_2336;
  long lbound_2337;
  long ubound_2338;
  long stride_2339;
  long ubound_2340;
  long stride_2341;
  long offset_2342;
  long size_2343;

  device GFReal * dellaqc_ens;
  long lbound_2344;
  long ubound_2345;
  long lbound_2346;
  long ubound_2347;
  long stride_2348;
  long offset_2349;
  long size_2350;

  device GFReal * dellat;
  long lbound_2351;
  long ubound_2352;
  long lbound_2353;
  long ubound_2354;
  long stride_2355;
  long ubound_2356;
  long stride_2357;
  long offset_2358;
  long size_2359;

  device GFReal * dellat_ens;
  GFReal depth_min;
  GFReal detdo;
  GFReal detup;
  GFReal detupk;
  GFReal dh;
  GFReal dp;
  long lbound_2360;
  long ubound_2361;
  long lbound_2362;
  long ubound_2363;
  long stride_2364;
  long offset_2365;
  long size_2366;

  device GFReal * dsubh;
  long lbound_2367;
  long ubound_2368;
  long lbound_2369;
  long ubound_2370;
  long stride_2371;
  long offset_2372;
  long size_2373;

  device GFReal * dsubq;
  long lbound_2374;
  long ubound_2375;
  long lbound_2376;
  long ubound_2377;
  long stride_2378;
  long offset_2379;
  long size_2380;

  device GFReal * dsubt;
  long lbound_2381;
  long ubound_2382;
  long offset_2383;
  long size_2384;
  GFReal dz;
  GFReal dzm1;
  GFReal dzo;
  GFReal dzp1;
  long lbound_2385;
  long ubound_2386;
  long offset_2387;
  long size_2388;

  device GFReal * edt;
  long ubound_2389;
  long size_2390;

  device GFReal * edt_ens;
  long lbound_2391;
  long ubound_2392;
  long ubound_2393;
  long stride_2394;
  long offset_2395;
  long size_2396;

  device GFReal * edtc;
  GFReal edtmax;
  GFReal edtmin;
  long lbound_2397;
  long ubound_2398;
  long offset_2399;
  long size_2400;

  device GFReal * edto;
  long lbound_2401;
  long ubound_2402;
  long offset_2403;
  long size_2404;

  device GFReal * edtx;
  GFReal entdo;
  GFReal entdoj;
  long lbound_2405;
  long ubound_2406;
  long offset_2407;
  long size_2408;

  device GFReal * entr_rate;
  long lbound_2409;
  long ubound_2410;
  long lbound_2411;
  long ubound_2412;
  long stride_2413;
  long offset_2414;
  long size_2415;

  device GFReal * entr_rate_2d;
  GFReal entup;
  GFReal entupk;
  long lbound_2416;
  long ubound_2417;
  long offset_2418;
  long size_2419;

  device GFReal * frh;
  long lbound_2420;
  long ubound_2421;
  long lbound_2422;
  long ubound_2423;
  long stride_2424;
  long offset_2425;
  long size_2426;

  device GFReal * gamma_cup;
  long lbound_2427;
  long ubound_2428;
  long lbound_2429;
  long ubound_2430;
  long stride_2431;
  long offset_2432;
  long size_2433;

  device GFReal * gammao_cup;
  long lbound_2434;
  long ubound_2435;
  long lbound_2436;
  long ubound_2437;
  long stride_2438;
  long offset_2439;
  long size_2440;

  device GFReal * hc;
  long lbound_2441;
  long ubound_2442;
  long lbound_2443;
  long ubound_2444;
  long stride_2445;
  long offset_2446;
  long size_2447;

  device GFReal * hcdo;
  long lbound_2448;
  long ubound_2449;
  long lbound_2450;
  long ubound_2451;
  long stride_2452;
  long offset_2453;
  long size_2454;

  device GFReal * hco;
  long lbound_2455;
  long ubound_2456;
  long lbound_2457;
  long ubound_2458;
  long stride_2459;
  long offset_2460;
  long size_2461;

  device GFReal * he;
  long lbound_2462;
  long ubound_2463;
  long lbound_2464;
  long ubound_2465;
  long stride_2466;
  long offset_2467;
  long size_2468;

  device GFReal * he_cup;
  long lbound_2469;
  long ubound_2470;
  long lbound_2471;
  long ubound_2472;
  long stride_2473;
  long offset_2474;
  long size_2475;

  device GFReal * heo;
  long lbound_2476;
  long ubound_2477;
  long lbound_2478;
  long ubound_2479;
  long stride_2480;
  long offset_2481;
  long size_2482;

  device GFReal * heo_cup;
  long lbound_2483;
  long ubound_2484;
  long lbound_2485;
  long ubound_2486;
  long stride_2487;
  long offset_2488;
  long size_2489;

  device GFReal * hes;
  long lbound_2490;
  long ubound_2491;
  long lbound_2492;
  long ubound_2493;
  long stride_2494;
  long offset_2495;
  long size_2496;

  device GFReal * hes_cup;
  long lbound_2497;
  long ubound_2498;
  long lbound_2499;
  long ubound_2500;
  long stride_2501;
  long offset_2502;
  long size_2503;

  device GFReal * heso;
  long lbound_2504;
  long ubound_2505;
  long lbound_2506;
  long ubound_2507;
  long stride_2508;
  long offset_2509;
  long size_2510;

  device GFReal * heso_cup;
  long lbound_2511;
  long ubound_2512;
  long offset_2513;
  long size_2514;

  device GFReal * hkb;
  long lbound_2515;
  long ubound_2516;
  long offset_2517;
  long size_2518;

  device GFReal * hkbo;
  int i;
  int iedt;
  long lbound_2519;
  long ubound_2520;
  long offset_2521;
  long size_2522;

  device int * ierr;
  long lbound_2523;
  long ubound_2524;
  long offset_2525;
  long size_2526;

  device int * ierr2;
  long lbound_2527;
  long ubound_2528;
  long offset_2529;
  long size_2530;

  device int * ierr3;
  long lbound_2531;
  long ubound_2532;
  long offset_2533;
  long size_2534;
  int iresult;
  long lbound_2535;
  long ubound_2536;
  long offset_2537;
  long size_2538;

  device int * jmin;
  int jmini;
  int k;
  int k1;
  int k2;
  long lbound_2539;
  long ubound_2540;
  long offset_2541;
  long size_2542;

  device int * k22;
  long lbound_2543;
  long ubound_2544;
  long offset_2545;
  long size_2546;

  device int * k22x;
  long lbound_2547;
  long ubound_2548;
  long offset_2549;
  long size_2550;
  long lbound_2551;
  long ubound_2552;
  long offset_2553;
  long size_2554;

  device int * kbconx;
  int kbegzu;
  long lbound_2555;
  long ubound_2556;
  long offset_2557;
  long size_2558;

  device int * kbmax;
  long lbound_2559;
  long ubound_2560;
  long offset_2561;
  long size_2562;

  device int * kdet;
  bool keep_going;
  int kfinalzu;
  int ki;
  long lbound_2563;
  long ubound_2564;
  long offset_2565;
  long size_2566;

  device int * kstabi;
  long lbound_2567;
  long ubound_2568;
  long offset_2569;
  long size_2570;

  device int * kstabm;
  int kstart;
  long lbound_2571;
  long ubound_2572;
  long offset_2573;
  long size_2574;
  long lbound_2575;
  long ubound_2576;
  long offset_2577;
  long size_2578;

  device int * kzdown;
  int levadj;
  GFReal mbdt;
  long ubound_2579;
  long size_2580;

  device GFReal * mbdt_ens;
  long lbound_2581;
  long ubound_2582;
  long ubound_2583;
  long stride_2584;
  long offset_2585;
  long size_2586;
  long lbound_2587;
  long ubound_2588;
  long offset_2589;
  long size_2590;

  device GFReal * mentrd_rate;
  long lbound_2591;
  long ubound_2592;
  long lbound_2593;
  long ubound_2594;
  long stride_2595;
  long offset_2596;
  long size_2597;

  device GFReal * mentrd_rate_2d;
  int nall;
  int nens;
  int nens3;
  long lbound_2598;
  long ubound_2599;
  long lbound_2600;
  long ubound_2601;
  long stride_2602;
  long ubound_2603;
  long stride_2604;
  long offset_2605;
  long size_2606;
  long lbound_2607;
  long ubound_2608;
  long lbound_2609;
  long ubound_2610;
  long stride_2611;
  long offset_2612;
  long size_2613;
  long lbound_2614;
  long ubound_2615;
  long lbound_2616;
  long ubound_2617;
  long stride_2618;
  long offset_2619;
  long size_2620;
  long lbound_2621;
  long ubound_2622;
  long lbound_2623;
  long ubound_2624;
  long stride_2625;
  long offset_2626;
  long size_2627;
  long lbound_2628;
  long ubound_2629;
  long lbound_2630;
  long ubound_2631;
  long stride_2632;
  long offset_2633;
  long size_2634;
  long lbound_2635;
  long ubound_2636;
  long lbound_2637;
  long ubound_2638;
  long stride_2639;
  long offset_2640;
  long size_2641;

  device GFReal * p_cup;
  long lbound_2642;
  long ubound_2643;
  long lbound_2644;
  long ubound_2645;
  long stride_2646;
  long offset_2647;
  long size_2648;
  long lbound_2649;
  long ubound_2650;
  long lbound_2651;
  long ubound_2652;
  long stride_2653;
  long offset_2654;
  long size_2655;

  device GFReal * po_cup;
  GFReal power_entr;
  long lbound_2656;
  long ubound_2657;
  long lbound_2658;
  long ubound_2659;
  long stride_2660;
  long ubound_2661;
  long stride_2662;
  long offset_2663;
  long size_2664;
  long lbound_2665;
  long ubound_2666;
  long offset_2667;
  long size_2668;
  long lbound_2669;
  long ubound_2670;
  long offset_2671;
  long size_2672;

  device GFReal * psum;
  long lbound_2673;
  long ubound_2674;
  long offset_2675;
  long size_2676;

  device GFReal * psumh;
  long lbound_2677;
  long ubound_2678;
  long offset_2679;
  long size_2680;
  long lbound_2681;
  long ubound_2682;
  long offset_2683;
  long size_2684;

  device GFReal * pwavo;
  long lbound_2685;
  long ubound_2686;
  long lbound_2687;
  long ubound_2688;
  long stride_2689;
  long offset_2690;
  long size_2691;

  device GFReal * pwdo;
  long lbound_2692;
  long ubound_2693;
  long offset_2694;
  long size_2695;

  device GFReal * pwevo;
  long lbound_2696;
  long ubound_2697;
  long lbound_2698;
  long ubound_2699;
  long stride_2700;
  long offset_2701;
  long size_2702;

  device GFReal * pwo;
  long lbound_2703;
  long ubound_2704;
  long lbound_2705;
  long ubound_2706;
  long stride_2707;
  long ubound_2708;
  long stride_2709;
  long offset_2710;
  long size_2711;

  device GFReal * pwo_ens;
  long lbound_2712;
  long ubound_2713;
  long lbound_2714;
  long ubound_2715;
  long stride_2716;
  long offset_2717;
  long size_2718;
  long lbound_2719;
  long ubound_2720;
  long lbound_2721;
  long ubound_2722;
  long stride_2723;
  long offset_2724;
  long size_2725;

  device GFReal * q_cup;
  long lbound_2726;
  long ubound_2727;
  long lbound_2728;
  long ubound_2729;
  long stride_2730;
  long offset_2731;
  long size_2732;

  device GFReal * qcdo;
  long lbound_2733;
  long ubound_2734;
  long lbound_2735;
  long ubound_2736;
  long stride_2737;
  long offset_2738;
  long size_2739;

  device GFReal * qco;
  long lbound_2740;
  long ubound_2741;
  long lbound_2742;
  long ubound_2743;
  long stride_2744;
  long offset_2745;
  long size_2746;

  device GFReal * qes;
  long lbound_2747;
  long ubound_2748;
  long lbound_2749;
  long ubound_2750;
  long stride_2751;
  long offset_2752;
  long size_2753;

  device GFReal * qes_cup;
  long lbound_2754;
  long ubound_2755;
  long lbound_2756;
  long ubound_2757;
  long stride_2758;
  long offset_2759;
  long size_2760;

  device GFReal * qeso;
  long lbound_2761;
  long ubound_2762;
  long lbound_2763;
  long ubound_2764;
  long stride_2765;
  long offset_2766;
  long size_2767;

  device GFReal * qeso_cup;
  long lbound_2768;
  long ubound_2769;
  long lbound_2770;
  long ubound_2771;
  long stride_2772;
  long offset_2773;
  long size_2774;
  long lbound_2775;
  long ubound_2776;
  long lbound_2777;
  long ubound_2778;
  long stride_2779;
  long offset_2780;
  long size_2781;

  device GFReal * qo_cup;
  long lbound_2782;
  long ubound_2783;
  long lbound_2784;
  long ubound_2785;
  long stride_2786;
  long offset_2787;
  long size_2788;

  device GFReal * qrcdo;
  long lbound_2789;
  long ubound_2790;
  long lbound_2791;
  long ubound_2792;
  long stride_2793;
  long offset_2794;
  long size_2795;

  device GFReal * qrco;
  long lbound_2796;
  long ubound_2797;
  long offset_2798;
  long size_2799;

  device GFReal * radius;
  long lbound_2800;
  long ubound_2801;
  long lbound_2802;
  long ubound_2803;
  long stride_2804;
  long offset_2805;
  long size_2806;
  long lbound_2807;
  long ubound_2808;
  long offset_2809;
  long size_2810;

  device GFReal * sig;
  long lbound_2811;
  long ubound_2812;
  long offset_2813;
  long size_2814;

  device GFReal * smth;
  long lbound_2815;
  long ubound_2816;
  long lbound_2817;
  long ubound_2818;
  long stride_2819;
  long offset_2820;
  long size_2821;
  GFReal subdown;
  GFReal subin;
  long lbound_2822;
  long ubound_2823;
  long lbound_2824;
  long ubound_2825;
  long stride_2826;
  long offset_2827;
  long size_2828;
  long lbound_2829;
  long ubound_2830;
  long lbound_2831;
  long ubound_2832;
  long stride_2833;
  long ubound_2834;
  long stride_2835;
  long offset_2836;
  long size_2837;

  device GFReal * subq_ens;
  long lbound_2838;
  long ubound_2839;
  long lbound_2840;
  long ubound_2841;
  long stride_2842;
  long offset_2843;
  long size_2844;
  long lbound_2845;
  long ubound_2846;
  long lbound_2847;
  long ubound_2848;
  long stride_2849;
  long ubound_2850;
  long stride_2851;
  long offset_2852;
  long size_2853;

  device GFReal * subt_ens;
  long lbound_2854;
  long ubound_2855;
  long lbound_2856;
  long ubound_2857;
  long stride_2858;
  long offset_2859;
  long size_2860;
  long lbound_2861;
  long ubound_2862;
  long lbound_2863;
  long ubound_2864;
  long stride_2865;
  long offset_2866;
  long size_2867;

  device GFReal * t_cup;
  long lbound_2868;
  long ubound_2869;
  long lbound_2870;
  long ubound_2871;
  long stride_2872;
  long offset_2873;
  long size_2874;
  long lbound_2875;
  long ubound_2876;
  long lbound_2877;
  long ubound_2878;
  long stride_2879;
  long offset_2880;
  long size_2881;

  device GFReal * tn_cup;
  GFReal totmas;
  long lbound_2882;
  long ubound_2883;
  long lbound_2884;
  long ubound_2885;
  long stride_2886;
  long offset_2887;
  long size_2888;

  device GFReal * up_massdetr;
  long lbound_2889;
  long ubound_2890;
  long lbound_2891;
  long ubound_2892;
  long stride_2893;
  long offset_2894;
  long size_2895;

  device GFReal * up_massdetro;
  long lbound_2896;
  long ubound_2897;
  long lbound_2898;
  long ubound_2899;
  long stride_2900;
  long offset_2901;
  long size_2902;

  device GFReal * up_massentr;
  long lbound_2903;
  long ubound_2904;
  long lbound_2905;
  long ubound_2906;
  long stride_2907;
  long offset_2908;
  long size_2909;

  device GFReal * up_massentro;
  long lbound_2910;
  long ubound_2911;
  long lbound_2912;
  long ubound_2913;
  long stride_2914;
  long offset_2915;
  long size_2916;
  long lbound_2917;
  long ubound_2918;
  long lbound_2919;
  long ubound_2920;
  long stride_2921;
  long offset_2922;
  long size_2923;
  long lbound_2924;
  long ubound_2925;
  long lbound_2926;
  long ubound_2927;
  long stride_2928;
  long offset_2929;
  long size_2930;

  device GFReal * weight_as;
  long lbound_2931;
  long ubound_2932;
  long lbound_2933;
  long ubound_2934;
  long stride_2935;
  long offset_2936;
  long size_2937;

  device GFReal * weight_gr;
  long lbound_2938;
  long ubound_2939;
  long lbound_2940;
  long ubound_2941;
  long stride_2942;
  long offset_2943;
  long size_2944;

  device GFReal * weight_mc;
  long lbound_2945;
  long ubound_2946;
  long lbound_2947;
  long ubound_2948;
  long stride_2949;
  long offset_2950;
  long size_2951;

  device GFReal * weight_st;
  long lbound_2952;
  long ubound_2953;
  long lbound_2954;
  long ubound_2955;
  long stride_2956;
  long offset_2957;
  long size_2958;

  device GFReal * weight_w;
  long lbound_2959;
  long ubound_2960;
  long offset_2961;
  long size_2962;

  device GFReal * xaa0;
  long lbound_2963;
  long ubound_2964;
  long ubound_2965;
  long stride_2966;
  long offset_2967;
  long size_2968;

  device GFReal * xaa0_ens;
  long lbound_2969;
  long ubound_2970;
  long lbound_2971;
  long ubound_2972;
  long stride_2973;
  long offset_2974;
  long size_2975;

  device GFReal * xdby;
  long lbound_2976;
  long ubound_2977;
  long lbound_2978;
  long ubound_2979;
  long stride_2980;
  long ubound_2981;
  long stride_2982;
  long offset_2983;
  long size_2984;
  long lbound_2985;
  long ubound_2986;
  long lbound_2987;
  long ubound_2988;
  long stride_2989;
  long offset_2990;
  long size_2991;

  device GFReal * xhc;
  long lbound_2992;
  long ubound_2993;
  long lbound_2994;
  long ubound_2995;
  long stride_2996;
  long offset_2997;
  long size_2998;

  device GFReal * xhe;
  long lbound_2999;
  long ubound_3000;
  long lbound_3001;
  long ubound_3002;
  long stride_3003;
  long offset_3004;
  long size_3005;

  device GFReal * xhe_cup;
  long lbound_3006;
  long ubound_3007;
  long lbound_3008;
  long ubound_3009;
  long stride_3010;
  long offset_3011;
  long size_3012;

  device GFReal * xhes;
  long lbound_3013;
  long ubound_3014;
  long lbound_3015;
  long ubound_3016;
  long stride_3017;
  long offset_3018;
  long size_3019;

  device GFReal * xhes_cup;
  long lbound_3020;
  long ubound_3021;
  long offset_3022;
  long size_3023;

  device GFReal * xhkb;
  long lbound_3024;
  long ubound_3025;
  long offset_3026;
  long size_3027;

  device GFReal * xlamdd;
  long lbound_3028;
  long ubound_3029;
  long offset_3030;
  long size_3031;
  long lbound_3032;
  long ubound_3033;
  long offset_3034;
  long size_3035;

  device GFReal * xland1;
  long lbound_3036;
  long ubound_3037;
  long offset_3038;
  long size_3039;
  long lbound_3040;
  long ubound_3041;
  long lbound_3042;
  long ubound_3043;
  long stride_3044;
  long offset_3045;
  long size_3046;

  device GFReal * xq;
  long lbound_3047;
  long ubound_3048;
  long lbound_3049;
  long ubound_3050;
  long stride_3051;
  long offset_3052;
  long size_3053;

  device GFReal * xq_cup;
  long lbound_3054;
  long ubound_3055;
  long lbound_3056;
  long ubound_3057;
  long stride_3058;
  long offset_3059;
  long size_3060;

  device GFReal * xqes;
  long lbound_3061;
  long ubound_3062;
  long lbound_3063;
  long ubound_3064;
  long stride_3065;
  long offset_3066;
  long size_3067;

  device GFReal * xqes_cup;
  long lbound_3068;
  long ubound_3069;
  long lbound_3070;
  long ubound_3071;
  long stride_3072;
  long offset_3073;
  long size_3074;

  device GFReal * xt;
  long lbound_3075;
  long ubound_3076;
  long lbound_3077;
  long ubound_3078;
  long stride_3079;
  long offset_3080;
  long size_3081;

  device GFReal * xt_cup;
  long lbound_3082;
  long ubound_3083;
  long lbound_3084;
  long ubound_3085;
  long stride_3086;
  long offset_3087;
  long size_3088;

  device GFReal * xz;
  long lbound_3089;
  long ubound_3090;
  long lbound_3091;
  long ubound_3092;
  long stride_3093;
  long offset_3094;
  long size_3095;

  device GFReal * xz_cup;
  long lbound_3096;
  long ubound_3097;
  long lbound_3098;
  long ubound_3099;
  long stride_3100;
  long offset_3101;
  long size_3102;

  device GFReal * xzu;
  long lbound_3103;
  long ubound_3104;
  long lbound_3105;
  long ubound_3106;
  long stride_3107;
  long offset_3108;
  long size_3109;

  device GFReal * z;
  long lbound_3110;
  long ubound_3111;
  long offset_3112;
  long size_3113;
  long lbound_3114;
  long ubound_3115;
  long lbound_3116;
  long ubound_3117;
  long stride_3118;
  long offset_3119;
  long size_3120;

  device GFReal * z_cup;
  GFReal z_detr;
  GFReal zcutdown;
  long lbound_3121;
  long ubound_3122;
  long lbound_3123;
  long ubound_3124;
  long stride_3125;
  long offset_3126;
  long size_3127;

  device GFReal * zd;
  long lbound_3128;
  long ubound_3129;
  long lbound_3130;
  long ubound_3131;
  long stride_3132;
  long offset_3133;
  long size_3134;

  device GFReal * zdo;
  GFReal zkbmax;
  GFReal zktop;
  long lbound_3135;
  long ubound_3136;
  long lbound_3137;
  long ubound_3138;
  long stride_3139;
  long offset_3140;
  long size_3141;
  long lbound_3142;
  long ubound_3143;
  long lbound_3144;
  long ubound_3145;
  long stride_3146;
  long offset_3147;
  long size_3148;

  device GFReal * zo_cup;
  long lbound_3149;
  long ubound_3150;
  long offset_3151;
  long size_3152;
  long lbound_3153;
  long ubound_3154;
  long offset_3155;
  long size_3156;
  long lbound_3157;
  long ubound_3158;
  long lbound_3159;
  long ubound_3160;
  long stride_3161;
  long offset_3162;
  long size_3163;

  device GFReal * zu;
  GFReal zufinal;
  long lbound_3164;
  long ubound_3165;
  long offset_3166;
  long size_3167;

  device GFReal * zuhe;
  long lbound_3168;
  long ubound_3169;
  long lbound_3170;
  long ubound_3171;
  long stride_3172;
  long offset_3173;
  long size_3174;

  device GFReal * zuo;
  GFReal zustart;
  long D_12412;
  ulong D_12413;
  ulong D_12414;
  long D_12416;
  ulong D_12417;
  ulong D_12418;
  long D_12420;
  ulong D_12421;
  ulong D_12422;
  long D_12424;
  ulong D_12425;
  ulong D_12426;
  long D_12428;
  ulong D_12429;
  ulong D_12430;
  long D_12432;
  ulong D_12433;
  ulong D_12434;
  long D_12436;
  ulong D_12437;
  ulong D_12438;
  long D_12440;
  ulong D_12441;
  ulong D_12442;
  long D_12444;
  ulong D_12445;
  ulong D_12446;
  long D_12448;
  ulong D_12449;
  ulong D_12450;
  long D_12452;
  ulong D_12453;
  ulong D_12454;
  long D_12456;
  ulong D_12457;
  ulong D_12458;
  long D_12460;
  ulong D_12461;
  ulong D_12462;
  long D_12464;
  ulong D_12465;
  ulong D_12466;
  long D_12468;
  ulong D_12469;
  ulong D_12470;
  long D_12472;
  ulong D_12473;
  ulong D_12474;
  long D_12476;
  ulong D_12477;
  ulong D_12478;
  long D_12480;
  ulong D_12481;
  ulong D_12482;
  long D_12484;
  ulong D_12485;
  ulong D_12486;
  long D_12488;
  ulong D_12489;
  ulong D_12490;
  long D_12492;
  ulong D_12493;
  ulong D_12494;
  long D_12496;
  ulong D_12497;
  ulong D_12498;
  long D_12500;
  ulong D_12501;
  ulong D_12502;
  long D_12504;
  ulong D_12505;
  ulong D_12506;
  long D_12508;
  ulong D_12509;
  ulong D_12510;
  long D_12512;
  ulong D_12513;
  ulong D_12514;
  long D_12516;
  ulong D_12517;
  ulong D_12518;
  long D_12520;
  ulong D_12521;
  ulong D_12522;
  long D_12524;
  ulong D_12525;
  ulong D_12526;
  long D_12528;
  ulong D_12529;
  ulong D_12530;
  long D_12532;
  ulong D_12533;
  ulong D_12534;
  long D_12536;
  ulong D_12537;
  ulong D_12538;
  long D_12540;
  ulong D_12541;
  ulong D_12542;
  long D_12544;
  ulong D_12545;
  ulong D_12546;
  long D_12548;
  ulong D_12549;
  ulong D_12550;
  long D_12552;
  ulong D_12553;
  ulong D_12554;
  long D_12556;
  ulong D_12557;
  ulong D_12558;
  long D_12560;
  ulong D_12561;
  ulong D_12562;
  long D_12564;
  ulong D_12565;
  ulong D_12566;
  long D_12568;
  ulong D_12569;
  ulong D_12570;
  long D_12572;
  ulong D_12573;
  ulong D_12574;
  long D_12576;
  ulong D_12577;
  ulong D_12578;
  long D_12580;
  ulong D_12581;
  ulong D_12582;
  long D_12584;
  ulong D_12585;
  ulong D_12586;
  long D_12588;
  ulong D_12589;
  ulong D_12590;
  long D_12592;
  ulong D_12593;
  ulong D_12594;
  long D_12596;
  ulong D_12597;
  ulong D_12598;
  long D_12600;
  ulong D_12601;
  ulong D_12602;
  long D_12604;
  ulong D_12605;
  ulong D_12606;
  long D_12608;
  ulong D_12609;
  ulong D_12610;
  long D_12612;
  ulong D_12613;
  ulong D_12614;
  long D_12616;
  ulong D_12617;
  ulong D_12618;
  long D_12620;
  ulong D_12621;
  ulong D_12622;
  long D_12624;
  ulong D_12625;
  ulong D_12626;
  long D_12628;
  ulong D_12629;
  ulong D_12630;
  long D_12632;
  ulong D_12633;
  ulong D_12634;
  long D_12636;
  ulong D_12637;
  ulong D_12638;
  long D_12640;
  ulong D_12641;
  ulong D_12642;
  long D_12644;
  ulong D_12645;
  ulong D_12646;
  long D_12648;
  ulong D_12649;
  ulong D_12650;
  long D_12652;
  ulong D_12653;
  ulong D_12654;
  long D_12656;
  ulong D_12657;
  ulong D_12658;
  long D_12660;
  ulong D_12661;
  ulong D_12662;
  long D_12664;
  ulong D_12665;
  ulong D_12666;
  long D_12668;
  ulong D_12669;
  ulong D_12670;
  long D_12672;
  ulong D_12673;
  ulong D_12674;
  long D_12676;
  ulong D_12677;
  ulong D_12678;
  long D_12680;
  ulong D_12681;
  ulong D_12682;
  long D_12684;
  ulong D_12685;
  ulong D_12686;
  long D_12688;
  ulong D_12689;
  ulong D_12690;
  long D_12692;
  ulong D_12693;
  ulong D_12694;
  long D_12696;
  ulong D_12697;
  ulong D_12698;
  long D_12700;
  ulong D_12701;
  ulong D_12702;
  long D_12704;
  ulong D_12705;
  ulong D_12706;
  long D_12708;
  ulong D_12709;
  ulong D_12710;
  long D_12712;
  ulong D_12713;
  ulong D_12714;
  long D_12716;
  ulong D_12717;
  ulong D_12718;
  long D_12720;
  ulong D_12721;
  ulong D_12722;
  long D_12724;
  ulong D_12725;
  ulong D_12726;
  long D_12728;
  ulong D_12729;
  ulong D_12730;
  long D_12732;
  ulong D_12733;
  ulong D_12734;
  long D_12736;
  ulong D_12737;
  ulong D_12738;
  long D_12740;
  ulong D_12741;
  ulong D_12742;
  long D_12744;
  ulong D_12745;
  ulong D_12746;
  long D_12748;
  ulong D_12749;
  ulong D_12750;
  long D_12752;
  ulong D_12753;
  ulong D_12754;
  long D_12756;
  ulong D_12757;
  ulong D_12758;
  long D_12760;
  ulong D_12761;
  ulong D_12762;
  long D_12764;
  ulong D_12765;
  ulong D_12766;
  long D_12768;
  ulong D_12769;
  ulong D_12770;
  long D_12772;
  ulong D_12773;
  ulong D_12774;
  long D_12776;
  ulong D_12777;
  ulong D_12778;
  long D_12780;
  ulong D_12781;
  ulong D_12782;
  long D_12784;
  ulong D_12785;
  ulong D_12786;
  long D_12788;
  ulong D_12789;
  ulong D_12790;
  long D_12792;
  ulong D_12793;
  ulong D_12794;
  long D_12796;
  ulong D_12797;
  ulong D_12798;
  long D_12800;
  ulong D_12801;
  ulong D_12802;
  long D_12804;
  ulong D_12805;
  ulong D_12806;
  long D_12808;
  ulong D_12809;
  ulong D_12810;
  long D_12812;
  ulong D_12813;
  ulong D_12814;
  long D_12816;
  ulong D_12817;
  ulong D_12818;
  long D_12820;
  ulong D_12821;
  ulong D_12822;
  long D_12824;
  ulong D_12825;
  ulong D_12826;
  long D_12828;
  ulong D_12829;
  ulong D_12830;
  long D_12832;
  ulong D_12833;
  ulong D_12834;
  long D_12836;
  ulong D_12837;
  ulong D_12838;
  long D_12840;
  ulong D_12841;
  ulong D_12842;
  long D_12844;
  ulong D_12845;
  ulong D_12846;
  long D_12848;
  ulong D_12849;
  ulong D_12850;
  long D_12852;
  ulong D_12853;
  ulong D_12854;
  long D_12856;
  ulong D_12857;
  ulong D_12858;
  long D_12860;
  ulong D_12861;
  ulong D_12862;
  long D_12864;
  ulong D_12865;
  ulong D_12866;
  long D_12868;
  ulong D_12869;
  ulong D_12870;
  long D_12872;
  ulong D_12873;
  ulong D_12874;
  long D_12876;
  ulong D_12877;
  ulong D_12878;
  long D_12880;
  ulong D_12881;
  ulong D_12882;
  long D_12884;
  ulong D_12885;
  ulong D_12886;
  long D_12888;
  ulong D_12889;
  ulong D_12890;
  long D_12892;
  ulong D_12893;
  ulong D_12894;
  long D_12896;
  ulong D_12897;
  ulong D_12898;
  long D_12900;
  ulong D_12901;
  ulong D_12902;
  long D_12904;
  ulong D_12905;
  ulong D_12906;
  long D_12908;
  ulong D_12909;
  ulong D_12910;
  long D_12912;
  ulong D_12913;
  ulong D_12914;
  long D_12916;
  ulong D_12917;
  ulong D_12918;
  long D_12920;
  ulong D_12921;
  ulong D_12922;
  long D_12924;
  ulong D_12925;
  ulong D_12926;
  long D_12928;
  ulong D_12929;
  ulong D_12930;
  long D_12931;
  ulong D_12932;
  ulong D_12933;
  long D_12934;
  ulong D_12935;
  ulong D_12936;
  long D_12937;
  ulong D_12938;
  ulong D_12939;
  long D_12940;
  ulong D_12941;
  ulong D_12942;
  long D_12943;
  ulong D_12944;
  ulong D_12945;
  long D_12946;
  ulong D_12947;
  ulong D_12948;
  long D_12949;
  ulong D_12950;
  ulong D_12951;
  long D_12952;
  ulong D_12953;
  ulong D_12954;
  long D_12955;
  ulong D_12956;
  ulong D_12957;
  long D_12958;
  ulong D_12959;
  ulong D_12960;
  long D_12961;
  ulong D_12962;
  ulong D_12963;
  long D_12964;
  ulong D_12965;
  ulong D_12966;
  long D_12967;
  ulong D_12968;
  ulong D_12969;
  long D_12970;
  ulong D_12971;
  ulong D_12972;
  long D_12973;
  ulong D_12974;
  ulong D_12975;
  long D_12976;
  ulong D_12977;
  ulong D_12978;
  long D_12979;
  ulong D_12980;
  ulong D_12981;
  long D_12982;
  ulong D_12983;
  ulong D_12984;
  long D_12985;
  ulong D_12986;
  ulong D_12987;
  long D_12988;
  ulong D_12989;
  ulong D_12990;
  long D_12991;
  ulong D_12992;
  ulong D_12993;
  long D_12994;
  ulong D_12995;
  ulong D_12996;
  long D_12997;
  ulong D_12998;
  ulong D_12999;
  long D_13000;
  ulong D_13001;
  ulong D_13002;
  long D_13003;
  ulong D_13004;
  ulong D_13005;
  long D_13006;
  ulong D_13007;
  ulong D_13008;
  long D_13009;
  ulong D_13010;
  ulong D_13011;
  long D_13012;
  ulong D_13013;
  ulong D_13014;
  long D_13015;
  ulong D_13016;
  ulong D_13017;
  long D_13018;
  ulong D_13019;
  ulong D_13020;
  long D_13021;
  ulong D_13022;
  ulong D_13023;
  long D_13024;
  ulong D_13025;
  ulong D_13026;
  long D_13027;
  ulong D_13028;
  ulong D_13029;
  long D_13030;
  ulong D_13031;
  ulong D_13032;
  long D_13033;
  ulong D_13034;
  ulong D_13035;
  long D_13036;
  ulong D_13037;
  ulong D_13038;
  long D_13039;
  ulong D_13040;
  ulong D_13041;
  long D_13042;
  ulong D_13043;
  ulong D_13044;
  long D_13045;
  ulong D_13046;
  ulong D_13047;
  long D_13048;
  ulong D_13049;
  ulong D_13050;
  long D_13051;
  ulong D_13052;
  ulong D_13053;

    {
      lbound_3028 = (long) *its;
      ubound_3029 = (long) *ite;
      size_3031 = (1 - lbound_3028) + ubound_3029;
      size_3031 = gf_max(size_3031, 0);
      D_13051 = size_3031 + -1;
      D_13052 = (ulong) (ulong) (size_3031) * 32;
      D_13053 = (ulong) (size_3031) * 4;
      offset_3030 = -(lbound_3028);
      lbound_2531 = (long) *its;
      ubound_2532 = (long) *ite;
      size_2534 = (1 - lbound_2531) + ubound_2532;
      size_2534 = gf_max(size_2534, 0);
      D_13048 = size_2534 + -1;
      D_13049 = (ulong) (ulong) (size_2534) * 400;
      D_13050 = (ulong) (size_2534) * 50;
      offset_2533 = -(lbound_2531);
      lbound_2381 = (long) *its;
      ubound_2382 = (long) *ite;
      size_2384 = (1 - lbound_2381) + ubound_2382;
      size_2384 = gf_max(size_2384, 0);
      D_13045 = size_2384 + -1;
      D_13046 = (ulong) (ulong) (size_2384) * 32;
      D_13047 = (ulong) (size_2384) * 4;
      offset_2383 = -(lbound_2381);
      lbound_3135 = (long) *its;
      ubound_3136 = (long) *ite;
      stride_3139 = (1 - lbound_3135) + ubound_3136;
      stride_3139 = gf_max(stride_3139, 0);
      lbound_3137 = (long) *kts;
      ubound_3138 = (long) *kte;
      size_3141 = ((1 - lbound_3137) + ubound_3138) * stride_3139;
      size_3141 = gf_max(size_3141, 0);
      D_13042 = size_3141 + -1;
      D_13043 = (ulong) (ulong) (size_3141) * 32;
      D_13044 = (ulong) (size_3141) * 4;
      offset_3140 = -(lbound_3135) - lbound_3137 * stride_3139;
      lbound_2249 = (long) *its;
      ubound_2250 = (long) *ite;
      stride_2253 = (1 - lbound_2249) + ubound_2250;
      stride_2253 = gf_max(stride_2253, 0);
      lbound_2251 = (long) *kts;
      ubound_2252 = (long) *kte;
      size_2255 = ((1 - lbound_2251) + ubound_2252) * stride_2253;
      size_2255 = gf_max(size_2255, 0);
      D_13039 = size_2255 + -1;
      D_13040 = (ulong) (ulong) (size_2255) * 32;
      D_13041 = (ulong) (size_2255) * 4;
      offset_2254 = -(lbound_2249) - lbound_2251 * stride_2253;
      lbound_2976 = (long) *its;
      ubound_2977 = (long) *ite;
      stride_2980 = (1 - lbound_2976) + ubound_2977;
      stride_2980 = gf_max(stride_2980, 0);
      lbound_2978 = (long) *jts;
      ubound_2979 = (long) *jte;
      stride_2982 = ((1 - lbound_2978) + ubound_2979) * stride_2980;
      stride_2982 = gf_max(stride_2982, 0);
      ubound_2981 = (long) *ensdim;
      size_2984 = stride_2982 * (ubound_2981);
      size_2984 = gf_max(size_2984, 0);
      D_13036 = size_2984 + -1;
      D_13037 = (ulong) (ulong) (size_2984) * 32;
      D_13038 = (ulong) (size_2984) * 4;
      offset_2983 = (-(lbound_2976) - lbound_2978 * stride_2980) - (stride_2982);
      lbound_2656 = (long) *its;
      ubound_2657 = (long) *ite;
      stride_2660 = (1 - lbound_2656) + ubound_2657;
      stride_2660 = gf_max(stride_2660, 0);
      lbound_2658 = (long) *jts;
      ubound_2659 = (long) *jte;
      stride_2662 = ((1 - lbound_2658) + ubound_2659) * stride_2660;
      stride_2662 = gf_max(stride_2662, 0);
      ubound_2661 = (long) *ensdim;
      size_2664 = stride_2662 * (ubound_2661);
      size_2664 = gf_max(size_2664, 0);
      D_13033 = size_2664 + -1;
      D_13034 = (ulong) (ulong) (size_2664) * 32;
      D_13035 = (ulong) (size_2664) * 4;
      offset_2663 = (-(lbound_2656) - lbound_2658 * stride_2660) - (stride_2662);
      lbound_2854 = (long) *its;
      ubound_2855 = (long) *ite;
      stride_2858 = (1 - lbound_2854) + ubound_2855;
      stride_2858 = gf_max(stride_2858, 0);
      lbound_2856 = (long) *kts;
      ubound_2857 = (long) *kte;
      size_2860 = ((1 - lbound_2856) + ubound_2857) * stride_2858;
      size_2860 = gf_max(size_2860, 0);
      D_13030 = size_2860 + -1;
      D_13031 = (ulong) (ulong) (size_2860) * 32;
      D_13032 = (ulong) (size_2860) * 4;
      offset_2859 = -(lbound_2854) - lbound_2856 * stride_2858;
      lbound_2712 = (long) *its;
      ubound_2713 = (long) *ite;
      stride_2716 = (1 - lbound_2712) + ubound_2713;
      stride_2716 = gf_max(stride_2716, 0);
      lbound_2714 = (long) *kts;
      ubound_2715 = (long) *kte;
      size_2718 = ((1 - lbound_2714) + ubound_2715) * stride_2716;
      size_2718 = gf_max(size_2718, 0);
      D_13027 = size_2718 + -1;
      D_13028 = (ulong) (ulong) (size_2718) * 32;
      D_13029 = (ulong) (size_2718) * 4;
      offset_2717 = -(lbound_2712) - lbound_2714 * stride_2716;
      lbound_2628 = (long) *its;
      ubound_2629 = (long) *ite;
      stride_2632 = (1 - lbound_2628) + ubound_2629;
      stride_2632 = gf_max(stride_2632, 0);
      lbound_2630 = (long) *kts;
      ubound_2631 = (long) *kte;
      size_2634 = ((1 - lbound_2630) + ubound_2631) * stride_2632;
      size_2634 = gf_max(size_2634, 0);
      D_13024 = size_2634 + -1;
      D_13025 = (ulong) (ulong) (size_2634) * 32;
      D_13026 = (ulong) (size_2634) * 4;
      offset_2633 = -(lbound_2628) - lbound_2630 * stride_2632;
      lbound_3110 = (long) *its;
      ubound_3111 = (long) *ite;
      size_3113 = (1 - lbound_3110) + ubound_3111;
      size_3113 = gf_max(size_3113, 0);
      D_13021 = size_3113 + -1;
      D_13022 = (ulong) (ulong) (size_3113) * 32;
      D_13023 = (ulong) (size_3113) * 4;
      offset_3112 = -(lbound_3110);
      lbound_2677 = (long) *its;
      ubound_2678 = (long) *ite;
      size_2680 = (1 - lbound_2677) + ubound_2678;
      size_2680 = gf_max(size_2680, 0);
      D_13018 = size_2680 + -1;
      D_13019 = (ulong) (ulong) (size_2680) * 32;
      D_13020 = (ulong) (size_2680) * 4;
      offset_2679 = -(lbound_2677);
      lbound_2868 = (long) *its;
      ubound_2869 = (long) *ite;
      stride_2872 = (1 - lbound_2868) + ubound_2869;
      stride_2872 = gf_max(stride_2872, 0);
      lbound_2870 = (long) *kts;
      ubound_2871 = (long) *kte;
      size_2874 = ((1 - lbound_2870) + ubound_2871) * stride_2872;
      size_2874 = gf_max(size_2874, 0);
      D_13015 = size_2874 + -1;
      D_13016 = (ulong) (ulong) (size_2874) * 32;
      D_13017 = (ulong) (size_2874) * 4;
      offset_2873 = -(lbound_2868) - lbound_2870 * stride_2872;
      lbound_2768 = (long) *its;
      ubound_2769 = (long) *ite;
      stride_2772 = (1 - lbound_2768) + ubound_2769;
      stride_2772 = gf_max(stride_2772, 0);
      lbound_2770 = (long) *kts;
      ubound_2771 = (long) *kte;
      size_2774 = ((1 - lbound_2770) + ubound_2771) * stride_2772;
      size_2774 = gf_max(size_2774, 0);
      D_13012 = size_2774 + -1;
      D_13013 = (ulong) (ulong) (size_2774) * 32;
      D_13014 = (ulong) (size_2774) * 4;
      offset_2773 = -(lbound_2768) - lbound_2770 * stride_2772;
      lbound_2642 = (long) *its;
      ubound_2643 = (long) *ite;
      stride_2646 = (1 - lbound_2642) + ubound_2643;
      stride_2646 = gf_max(stride_2646, 0);
      lbound_2644 = (long) *kts;
      ubound_2645 = (long) *kte;
      size_2648 = ((1 - lbound_2644) + ubound_2645) * stride_2646;
      size_2648 = gf_max(size_2648, 0);
      D_13009 = size_2648 + -1;
      D_13010 = (ulong) (ulong) (size_2648) * 32;
      D_13011 = (ulong) (size_2648) * 4;
      offset_2647 = -(lbound_2642) - lbound_2644 * stride_2646;
      lbound_2138 = (long) *its;
      ubound_2139 = (long) *ite;
      size_2141 = (1 - lbound_2138) + ubound_2139;
      size_2141 = gf_max(size_2141, 0);
      D_13006 = size_2141 + -1;
      D_13007 = (ulong) (ulong) (size_2141) * 32;
      D_13008 = (ulong) (size_2141) * 4;
      offset_2140 = -(lbound_2138);
      lbound_2598 = (long) *its;
      ubound_2599 = (long) *ite;
      stride_2602 = (1 - lbound_2598) + ubound_2599;
      stride_2602 = gf_max(stride_2602, 0);
      lbound_2600 = (long) *kts;
      ubound_2601 = (long) *kte;
      stride_2604 = ((1 - lbound_2600) + ubound_2601) * stride_2602;
      stride_2604 = gf_max(stride_2604, 0);
      ubound_2603 = (long) *ens4;
      size_2606 = stride_2604 * (ubound_2603);
      size_2606 = gf_max(size_2606, 0);
      D_13003 = size_2606 + -1;
      D_13004 = (ulong) (ulong) (size_2606) * 32;
      D_13005 = (ulong) (size_2606) * 4;
      offset_2605 = (-(lbound_2598) - lbound_2600 * stride_2602) - (stride_2604);
      lbound_3149 = (long) *its;
      ubound_3150 = (long) *ite;
      size_3152 = (1 - lbound_3149) + ubound_3150;
      size_3152 = gf_max(size_3152, 0);
      D_13000 = size_3152 + -1;
      D_13001 = (ulong) (ulong) (size_3152) * 32;
      D_13002 = (ulong) (size_3152) * 4;
      offset_3151 = -(lbound_3149);
      lbound_3153 = (long) *its;
      ubound_3154 = (long) *ite;
      size_3156 = (1 - lbound_3153) + ubound_3154;
      size_3156 = gf_max(size_3156, 0);
      D_12997 = size_3156 + -1;
      D_12998 = (ulong) (ulong) (size_3156) * 32;
      D_12999 = (ulong) (size_3156) * 4;
      offset_3155 = -(lbound_3153);
      lbound_2547 = (long) *its;
      ubound_2548 = (long) *ite;
      size_2550 = (1 - lbound_2547) + ubound_2548;
      size_2550 = gf_max(size_2550, 0);
      D_12994 = size_2550 + -1;
      D_12995 = (ulong) (ulong) (size_2550) * 32;
      D_12996 = (ulong) (size_2550) * 4;
      offset_2549 = -(lbound_2547);
      lbound_2571 = (long) *its;
      ubound_2572 = (long) *ite;
      size_2574 = (1 - lbound_2571) + ubound_2572;
      size_2574 = gf_max(size_2574, 0);
      D_12991 = size_2574 + -1;
      D_12992 = (ulong) (ulong) (size_2574) * 32;
      D_12993 = (ulong) (size_2574) * 4;
      offset_2573 = -(lbound_2571);
      lbound_2220 = (long) *its;
      ubound_2221 = (long) *ite;
      size_2223 = (1 - lbound_2220) + ubound_2221;
      size_2223 = gf_max(size_2223, 0);
      D_12988 = size_2223 + -1;
      D_12989 = (ulong) (ulong) (size_2223) * 32;
      D_12990 = (ulong) (size_2223) * 4;
      offset_2222 = -(lbound_2220);
      lbound_2800 = (long) *its;
      ubound_2801 = (long) *ite;
      stride_2804 = (1 - lbound_2800) + ubound_2801;
      stride_2804 = gf_max(stride_2804, 0);
      lbound_2802 = (long) *kts;
      ubound_2803 = (long) *kte;
      size_2806 = ((1 - lbound_2802) + ubound_2803) * stride_2804;
      size_2806 = gf_max(size_2806, 0);
      D_12985 = size_2806 + -1;
      D_12986 = (ulong) (ulong) (size_2806) * 32;
      D_12987 = (ulong) (size_2806) * 4;
      offset_2805 = -(lbound_2800) - lbound_2802 * stride_2804;
      lbound_2910 = (long) *its;
      ubound_2911 = (long) *ite;
      stride_2914 = (1 - lbound_2910) + ubound_2911;
      stride_2914 = gf_max(stride_2914, 0);
      lbound_2912 = (long) *kts;
      ubound_2913 = (long) *kte;
      size_2916 = ((1 - lbound_2912) + ubound_2913) * stride_2914;
      size_2916 = gf_max(size_2916, 0);
      D_12982 = size_2916 + -1;
      D_12983 = (ulong) (ulong) (size_2916) * 32;
      D_12984 = (ulong) (size_2916) * 4;
      offset_2915 = -(lbound_2910) - lbound_2912 * stride_2914;
      lbound_2917 = (long) *its;
      ubound_2918 = (long) *ite;
      stride_2921 = (1 - lbound_2917) + ubound_2918;
      stride_2921 = gf_max(stride_2921, 0);
      lbound_2919 = (long) *kts;
      ubound_2920 = (long) *kte;
      size_2923 = ((1 - lbound_2919) + ubound_2920) * stride_2921;
      size_2923 = gf_max(size_2923, 0);
      D_12979 = size_2923 + -1;
      D_12980 = (ulong) (ulong) (size_2923) * 32;
      D_12981 = (ulong) (size_2923) * 4;
      offset_2922 = -(lbound_2917) - lbound_2919 * stride_2921;
      lbound_2581 = (long) *its;
      ubound_2582 = (long) *ite;
      stride_2584 = (1 - lbound_2581) + ubound_2582;
      stride_2584 = gf_max(stride_2584, 0);
      ubound_2583 = (long) *ens4;
      size_2586 = stride_2584 * (ubound_2583);
      size_2586 = gf_max(size_2586, 0);
      D_12976 = size_2586 + -1;
      D_12977 = (ulong) (ulong) (size_2586) * 32;
      D_12978 = (ulong) (size_2586) * 4;
      offset_2585 = -(lbound_2581) - (stride_2584);
      lbound_2838 = (long) *its;
      ubound_2839 = (long) *ite;
      stride_2842 = (1 - lbound_2838) + ubound_2839;
      stride_2842 = gf_max(stride_2842, 0);
      lbound_2840 = (long) *kts;
      ubound_2841 = (long) *kte;
      size_2844 = ((1 - lbound_2840) + ubound_2841) * stride_2842;
      size_2844 = gf_max(size_2844, 0);
      D_12973 = size_2844 + -1;
      D_12974 = (ulong) (ulong) (size_2844) * 32;
      D_12975 = (ulong) (size_2844) * 4;
      offset_2843 = -(lbound_2838) - lbound_2840 * stride_2842;
      lbound_2822 = (long) *its;
      ubound_2823 = (long) *ite;
      stride_2826 = (1 - lbound_2822) + ubound_2823;
      stride_2826 = gf_max(stride_2826, 0);
      lbound_2824 = (long) *kts;
      ubound_2825 = (long) *kte;
      size_2828 = ((1 - lbound_2824) + ubound_2825) * stride_2826;
      size_2828 = gf_max(size_2828, 0);
      D_12970 = size_2828 + -1;
      D_12971 = (ulong) (ulong) (size_2828) * 32;
      D_12972 = (ulong) (size_2828) * 4;
      offset_2827 = -(lbound_2822) - lbound_2824 * stride_2826;
      lbound_2621 = (long) *its;
      ubound_2622 = (long) *ite;
      stride_2625 = (1 - lbound_2621) + ubound_2622;
      stride_2625 = gf_max(stride_2625, 0);
      lbound_2623 = (long) *kts;
      ubound_2624 = (long) *kte;
      size_2627 = ((1 - lbound_2623) + ubound_2624) * stride_2625;
      size_2627 = gf_max(size_2627, 0);
      D_12967 = size_2627 + -1;
      D_12968 = (ulong) (ulong) (size_2627) * 32;
      D_12969 = (ulong) (size_2627) * 4;
      offset_2626 = -(lbound_2621) - lbound_2623 * stride_2625;
      lbound_2607 = (long) *its;
      ubound_2608 = (long) *ite;
      stride_2611 = (1 - lbound_2607) + ubound_2608;
      stride_2611 = gf_max(stride_2611, 0);
      lbound_2609 = (long) *kts;
      ubound_2610 = (long) *kte;
      size_2613 = ((1 - lbound_2609) + ubound_2610) * stride_2611;
      size_2613 = gf_max(size_2613, 0);
      D_12964 = size_2613 + -1;
      D_12965 = (ulong) (ulong) (size_2613) * 32;
      D_12966 = (ulong) (size_2613) * 4;
      offset_2612 = -(lbound_2607) - lbound_2609 * stride_2611;
      lbound_2614 = (long) *its;
      ubound_2615 = (long) *ite;
      stride_2618 = (1 - lbound_2614) + ubound_2615;
      stride_2618 = gf_max(stride_2618, 0);
      lbound_2616 = (long) *kts;
      ubound_2617 = (long) *kte;
      size_2620 = ((1 - lbound_2616) + ubound_2617) * stride_2618;
      size_2620 = gf_max(size_2620, 0);
      D_12961 = size_2620 + -1;
      D_12962 = (ulong) (ulong) (size_2620) * 32;
      D_12963 = (ulong) (size_2620) * 4;
      offset_2619 = -(lbound_2614) - lbound_2616 * stride_2618;
      lbound_2815 = (long) *its;
      ubound_2816 = (long) *ite;
      stride_2819 = (1 - lbound_2815) + ubound_2816;
      stride_2819 = gf_max(stride_2819, 0);
      lbound_2817 = (long) *kts;
      ubound_2818 = (long) *kte;
      size_2821 = ((1 - lbound_2817) + ubound_2818) * stride_2819;
      size_2821 = gf_max(size_2821, 0);
      D_12958 = size_2821 + -1;
      D_12959 = (ulong) (ulong) (size_2821) * 32;
      D_12960 = (ulong) (size_2821) * 4;
      offset_2820 = -(lbound_2815) - lbound_2817 * stride_2819;
      lbound_2665 = (long) *its;
      ubound_2666 = (long) *ite;
      size_2668 = (1 - lbound_2665) + ubound_2666;
      size_2668 = gf_max(size_2668, 0);
      D_12955 = size_2668 + -1;
      D_12956 = (ulong) (ulong) (size_2668) * 32;
      D_12957 = (ulong) (size_2668) * 4;
      offset_2667 = -(lbound_2665);
      lbound_3036 = (long) *its;
      ubound_3037 = (long) *ite;
      size_3039 = (1 - lbound_3036) + ubound_3037;
      size_3039 = gf_max(size_3039, 0);
      D_12952 = size_3039 + -1;
      D_12953 = (ulong) (ulong) (size_3039) * 32;
      D_12954 = (ulong) (size_3039) * 4;
      offset_3038 = -(lbound_3036);
      lbound_2170 = (long) *its;
      ubound_2171 = (long) *ite;
      stride_2174 = (1 - lbound_2170) + ubound_2171;
      stride_2174 = gf_max(stride_2174, 0);
      lbound_2172 = (long) *jts;
      ubound_2173 = (long) *jte;
      size_2176 = ((1 - lbound_2172) + ubound_2173) * stride_2174;
      size_2176 = gf_max(size_2176, 0);
      D_12949 = size_2176 + -1;
      D_12950 = (ulong) (ulong) (size_2176) * 32;
      D_12951 = (ulong) (size_2176) * 4;
      offset_2175 = -(lbound_2170) - lbound_2172 * stride_2174;
      lbound_2191 = (long) *its;
      ubound_2192 = (long) *ite;
      stride_2195 = (1 - lbound_2191) + ubound_2192;
      stride_2195 = gf_max(stride_2195, 0);
      lbound_2193 = (long) *jts;
      ubound_2194 = (long) *jte;
      size_2197 = ((1 - lbound_2193) + ubound_2194) * stride_2195;
      size_2197 = gf_max(size_2197, 0);
      D_12946 = size_2197 + -1;
      D_12947 = (ulong) (ulong) (size_2197) * 32;
      D_12948 = (ulong) (size_2197) * 4;
      offset_2196 = -(lbound_2191) - lbound_2193 * stride_2195;
      lbound_2177 = (long) *its;
      ubound_2178 = (long) *ite;
      stride_2181 = (1 - lbound_2177) + ubound_2178;
      stride_2181 = gf_max(stride_2181, 0);
      lbound_2179 = (long) *jts;
      ubound_2180 = (long) *jte;
      size_2183 = ((1 - lbound_2179) + ubound_2180) * stride_2181;
      size_2183 = gf_max(size_2183, 0);
      D_12943 = size_2183 + -1;
      D_12944 = (ulong) (ulong) (size_2183) * 32;
      D_12945 = (ulong) (size_2183) * 4;
      offset_2182 = -(lbound_2177) - lbound_2179 * stride_2181;
      lbound_2184 = (long) *its;
      ubound_2185 = (long) *ite;
      stride_2188 = (1 - lbound_2184) + ubound_2185;
      stride_2188 = gf_max(stride_2188, 0);
      lbound_2186 = (long) *jts;
      ubound_2187 = (long) *jte;
      size_2190 = ((1 - lbound_2186) + ubound_2187) * stride_2188;
      size_2190 = gf_max(size_2190, 0);
      D_12940 = size_2190 + -1;
      D_12941 = (ulong) (ulong) (size_2190) * 32;
      D_12942 = (ulong) (size_2190) * 4;
      offset_2189 = -(lbound_2184) - lbound_2186 * stride_2188;
      lbound_2142 = (long) *its;
      ubound_2143 = (long) *ite;
      stride_2146 = (1 - lbound_2142) + ubound_2143;
      stride_2146 = gf_max(stride_2146, 0);
      lbound_2144 = (long) *jts;
      ubound_2145 = (long) *jte;
      size_2148 = ((1 - lbound_2144) + ubound_2145) * stride_2146;
      size_2148 = gf_max(size_2148, 0);
      D_12937 = size_2148 + -1;
      D_12938 = (ulong) (ulong) (size_2148) * 32;
      D_12939 = (ulong) (size_2148) * 4;
      offset_2147 = -(lbound_2142) - lbound_2144 * stride_2146;
      lbound_2149 = (long) *its;
      ubound_2150 = (long) *ite;
      stride_2153 = (1 - lbound_2149) + ubound_2150;
      stride_2153 = gf_max(stride_2153, 0);
      lbound_2151 = (long) *jts;
      ubound_2152 = (long) *jte;
      size_2155 = ((1 - lbound_2151) + ubound_2152) * stride_2153;
      size_2155 = gf_max(size_2155, 0);
      D_12934 = size_2155 + -1;
      D_12935 = (ulong) (ulong) (size_2155) * 32;
      D_12936 = (ulong) (size_2155) * 4;
      offset_2154 = -(lbound_2149) - lbound_2151 * stride_2153;
      lbound_2156 = (long) *its;
      ubound_2157 = (long) *ite;
      stride_2160 = (1 - lbound_2156) + ubound_2157;
      stride_2160 = gf_max(stride_2160, 0);
      lbound_2158 = (long) *jts;
      ubound_2159 = (long) *jte;
      size_2162 = ((1 - lbound_2158) + ubound_2159) * stride_2160;
      size_2162 = gf_max(size_2162, 0);
      D_12931 = size_2162 + -1;
      D_12932 = (ulong) (ulong) (size_2162) * 32;
      D_12933 = (ulong) (size_2162) * 4;
      offset_2161 = -(lbound_2156) - lbound_2158 * stride_2160;
      lbound_2163 = (long) *its;
      ubound_2164 = (long) *ite;
      stride_2167 = (1 - lbound_2163) + ubound_2164;
      stride_2167 = gf_max(stride_2167, 0);
      lbound_2165 = (long) *jts;
      ubound_2166 = (long) *jte;
      size_2169 = ((1 - lbound_2165) + ubound_2166) * stride_2167;
      size_2169 = gf_max(size_2169, 0);
      D_12928 = size_2169 + -1;
      D_12929 = (ulong) (ulong) (size_2169) * 32;
      D_12930 = (ulong) (size_2169) * 4;
      offset_2168 = -(lbound_2163) - lbound_2165 * stride_2167;
      lbound_2130 = (long) *its;
      ubound_2131 = (long) *ite;
      size_2133 = (1 - lbound_2130) + ubound_2131;
      size_2133 = gf_max(size_2133, 0);
      D_12924 = size_2133 + -1;
      D_12925 = (ulong) (ulong) (size_2133) * 32;
      D_12926 = (ulong) (size_2133) * 4;
      D_12927 = (device void *) workspace.allocate (gf_max((ulong) (size_2133 * 4), 1));
      aa0 = (device GFReal *) D_12927;
      offset_2132 = -(lbound_2130);
      lbound_2134 = (long) *its;
      ubound_2135 = (long) *ite;
      size_2137 = (1 - lbound_2134) + ubound_2135;
      size_2137 = gf_max(size_2137, 0);
      D_12920 = size_2137 + -1;
      D_12921 = (ulong) (ulong) (size_2137) * 32;
      D_12922 = (ulong) (size_2137) * 4;
      D_12923 = (device void *) workspace.allocate (gf_max((ulong) (size_2137 * 4), 1));
      aa1 = (device GFReal *) D_12923;
      offset_2136 = -(lbound_2134);
      lbound_2198 = (long) *its;
      ubound_2199 = (long) *ite;
      stride_2201 = (1 - lbound_2198) + ubound_2199;
      stride_2201 = gf_max(stride_2201, 0);
      ubound_2200 = (long) *ens4;
      size_2203 = stride_2201 * (ubound_2200);
      size_2203 = gf_max(size_2203, 0);
      D_12916 = size_2203 + -1;
      D_12917 = (ulong) (ulong) (size_2203) * 32;
      D_12918 = (ulong) (size_2203) * 4;
      D_12919 = (device void *) workspace.allocate (gf_max((ulong) (size_2203 * 4), 1));
      axx = (device GFReal *) D_12919;
      offset_2202 = -(lbound_2198) - (stride_2201);
      lbound_2204 = (long) *its;
      ubound_2205 = (long) *ite;
      size_2207 = (1 - lbound_2204) + ubound_2205;
      size_2207 = gf_max(size_2207, 0);
      D_12912 = size_2207 + -1;
      D_12913 = (ulong) (ulong) (size_2207) * 32;
      D_12914 = (ulong) (size_2207) * 4;
      D_12915 = (device void *) workspace.allocate (gf_max((ulong) (size_2207 * 4), 1));
      bu = (device GFReal *) D_12915;
      offset_2206 = -(lbound_2204);
      lbound_2208 = (long) *its;
      ubound_2209 = (long) *ite;
      size_2211 = (1 - lbound_2208) + ubound_2209;
      size_2211 = gf_max(size_2211, 0);
      D_12908 = size_2211 + -1;
      D_12909 = (ulong) (ulong) (size_2211) * 32;
      D_12910 = (ulong) (size_2211) * 4;
      D_12911 = (device void *) workspace.allocate (gf_max((ulong) (size_2211 * 4), 1));
      bud = (device GFReal *) D_12911;
      offset_2210 = -(lbound_2208);
      lbound_2212 = (long) *its;
      ubound_2213 = (long) *ite;
      size_2215 = (1 - lbound_2212) + ubound_2213;
      size_2215 = gf_max(size_2215, 0);
      D_12904 = size_2215 + -1;
      D_12905 = (ulong) (ulong) (size_2215) * 32;
      D_12906 = (ulong) (size_2215) * 4;
      D_12907 = (device void *) workspace.allocate (gf_max((ulong) (size_2215 * 4), 1));
      cap_max = (device GFReal *) D_12907;
      offset_2214 = -(lbound_2212);
      lbound_2216 = (long) *its;
      ubound_2217 = (long) *ite;
      size_2219 = (1 - lbound_2216) + ubound_2217;
      size_2219 = gf_max(size_2219, 0);
      D_12900 = size_2219 + -1;
      D_12901 = (ulong) (ulong) (size_2219) * 32;
      D_12902 = (ulong) (size_2219) * 4;
      D_12903 = (device void *) workspace.allocate (gf_max((ulong) (size_2219 * 4), 1));
      cap_max_increment = (device GFReal *) D_12903;
      offset_2218 = -(lbound_2216);
      lbound_2224 = (long) *its;
      ubound_2225 = (long) *ite;
      stride_2228 = (1 - lbound_2224) + ubound_2225;
      stride_2228 = gf_max(stride_2228, 0);
      lbound_2226 = (long) *kts;
      ubound_2227 = (long) *kte;
      size_2230 = ((1 - lbound_2226) + ubound_2227) * stride_2228;
      size_2230 = gf_max(size_2230, 0);
      D_12896 = size_2230 + -1;
      D_12897 = (ulong) (ulong) (size_2230) * 32;
      D_12898 = (ulong) (size_2230) * 4;
      D_12899 = (device void *) workspace.allocate (gf_max((ulong) (size_2230 * 4), 1));
      cd = (device GFReal *) D_12899;
      offset_2229 = -(lbound_2224) - lbound_2226 * stride_2228;
      lbound_2231 = (long) *its;
      ubound_2232 = (long) *ite;
      stride_2235 = (1 - lbound_2231) + ubound_2232;
      stride_2235 = gf_max(stride_2235, 0);
      lbound_2233 = (long) *kts;
      ubound_2234 = (long) *kte;
      size_2237 = ((1 - lbound_2233) + ubound_2234) * stride_2235;
      size_2237 = gf_max(size_2237, 0);
      D_12892 = size_2237 + -1;
      D_12893 = (ulong) (ulong) (size_2237) * 32;
      D_12894 = (ulong) (size_2237) * 4;
      D_12895 = (device void *) workspace.allocate (gf_max((ulong) (size_2237 * 4), 1));
      cdd = (device GFReal *) D_12895;
      offset_2236 = -(lbound_2231) - lbound_2233 * stride_2235;
      lbound_2238 = (long) *its;
      ubound_2239 = (long) *ite;
      size_2241 = (1 - lbound_2238) + ubound_2239;
      size_2241 = gf_max(size_2241, 0);
      D_12888 = size_2241 + -1;
      D_12889 = (ulong) (ulong) (size_2241) * 32;
      D_12890 = (ulong) (size_2241) * 4;
      D_12891 = (device void *) workspace.allocate (gf_max((ulong) (size_2241 * 4), 1));
      closure_n = (device GFReal *) D_12891;
      offset_2240 = -(lbound_2238);
      lbound_2242 = (long) *its;
      ubound_2243 = (long) *ite;
      stride_2246 = (1 - lbound_2242) + ubound_2243;
      stride_2246 = gf_max(stride_2246, 0);
      lbound_2244 = (long) *kts;
      ubound_2245 = (long) *kte;
      size_2248 = ((1 - lbound_2244) + ubound_2245) * stride_2246;
      size_2248 = gf_max(size_2248, 0);
      D_12884 = size_2248 + -1;
      D_12885 = (ulong) (ulong) (size_2248) * 32;
      D_12886 = (ulong) (size_2248) * 4;
      D_12887 = (device void *) workspace.allocate (gf_max((ulong) (size_2248 * 4), 1));
      clw_all = (device GFReal *) D_12887;
      offset_2247 = -(lbound_2242) - lbound_2244 * stride_2246;
      lbound_2256 = (long) *its;
      ubound_2257 = (long) *ite;
      stride_2260 = (1 - lbound_2256) + ubound_2257;
      stride_2260 = gf_max(stride_2260, 0);
      lbound_2258 = (long) *kts;
      ubound_2259 = (long) *kte;
      size_2262 = ((1 - lbound_2258) + ubound_2259) * stride_2260;
      size_2262 = gf_max(size_2262, 0);
      D_12880 = size_2262 + -1;
      D_12881 = (ulong) (ulong) (size_2262) * 32;
      D_12882 = (ulong) (size_2262) * 4;
      D_12883 = (device void *) workspace.allocate (gf_max((ulong) (size_2262 * 4), 1));
      dby = (device GFReal *) D_12883;
      offset_2261 = -(lbound_2256) - lbound_2258 * stride_2260;
      lbound_2263 = (long) *its;
      ubound_2264 = (long) *ite;
      stride_2267 = (1 - lbound_2263) + ubound_2264;
      stride_2267 = gf_max(stride_2267, 0);
      lbound_2265 = (long) *kts;
      ubound_2266 = (long) *kte;
      size_2269 = ((1 - lbound_2265) + ubound_2266) * stride_2267;
      size_2269 = gf_max(size_2269, 0);
      D_12876 = size_2269 + -1;
      D_12877 = (ulong) (ulong) (size_2269) * 32;
      D_12878 = (ulong) (size_2269) * 4;
      D_12879 = (device void *) workspace.allocate (gf_max((ulong) (size_2269 * 4), 1));
      dbydo = (device GFReal *) D_12879;
      offset_2268 = -(lbound_2263) - lbound_2265 * stride_2267;
      lbound_2270 = (long) *its;
      ubound_2271 = (long) *ite;
      stride_2274 = (1 - lbound_2270) + ubound_2271;
      stride_2274 = gf_max(stride_2274, 0);
      lbound_2272 = (long) *kts;
      ubound_2273 = (long) *kte;
      size_2276 = ((1 - lbound_2272) + ubound_2273) * stride_2274;
      size_2276 = gf_max(size_2276, 0);
      D_12872 = size_2276 + -1;
      D_12873 = (ulong) (ulong) (size_2276) * 32;
      D_12874 = (ulong) (size_2276) * 4;
      D_12875 = (device void *) workspace.allocate (gf_max((ulong) (size_2276 * 4), 1));
      dbyo = (device GFReal *) D_12875;
      offset_2275 = -(lbound_2270) - lbound_2272 * stride_2274;
      lbound_2277 = (long) *its;
      ubound_2278 = (long) *ite;
      stride_2281 = (1 - lbound_2277) + ubound_2278;
      stride_2281 = gf_max(stride_2281, 0);
      lbound_2279 = (long) *kts;
      ubound_2280 = (long) *kte;
      size_2283 = ((1 - lbound_2279) + ubound_2280) * stride_2281;
      size_2283 = gf_max(size_2283, 0);
      D_12868 = size_2283 + -1;
      D_12869 = (ulong) (ulong) (size_2283) * 32;
      D_12870 = (ulong) (size_2283) * 4;
      D_12871 = (device void *) workspace.allocate (gf_max((ulong) (size_2283 * 4), 1));
      dd_massdetr = (device GFReal *) D_12871;
      offset_2282 = -(lbound_2277) - lbound_2279 * stride_2281;
      lbound_2284 = (long) *its;
      ubound_2285 = (long) *ite;
      stride_2288 = (1 - lbound_2284) + ubound_2285;
      stride_2288 = gf_max(stride_2288, 0);
      lbound_2286 = (long) *kts;
      ubound_2287 = (long) *kte;
      size_2290 = ((1 - lbound_2286) + ubound_2287) * stride_2288;
      size_2290 = gf_max(size_2290, 0);
      D_12864 = size_2290 + -1;
      D_12865 = (ulong) (ulong) (size_2290) * 32;
      D_12866 = (ulong) (size_2290) * 4;
      D_12867 = (device void *) workspace.allocate (gf_max((ulong) (size_2290 * 4), 1));
      dd_massdetro = (device GFReal *) D_12867;
      offset_2289 = -(lbound_2284) - lbound_2286 * stride_2288;
      lbound_2291 = (long) *its;
      ubound_2292 = (long) *ite;
      stride_2295 = (1 - lbound_2291) + ubound_2292;
      stride_2295 = gf_max(stride_2295, 0);
      lbound_2293 = (long) *kts;
      ubound_2294 = (long) *kte;
      size_2297 = ((1 - lbound_2293) + ubound_2294) * stride_2295;
      size_2297 = gf_max(size_2297, 0);
      D_12860 = size_2297 + -1;
      D_12861 = (ulong) (ulong) (size_2297) * 32;
      D_12862 = (ulong) (size_2297) * 4;
      D_12863 = (device void *) workspace.allocate (gf_max((ulong) (size_2297 * 4), 1));
      dd_massentr = (device GFReal *) D_12863;
      offset_2296 = -(lbound_2291) - lbound_2293 * stride_2295;
      lbound_2298 = (long) *its;
      ubound_2299 = (long) *ite;
      stride_2302 = (1 - lbound_2298) + ubound_2299;
      stride_2302 = gf_max(stride_2302, 0);
      lbound_2300 = (long) *kts;
      ubound_2301 = (long) *kte;
      size_2304 = ((1 - lbound_2300) + ubound_2301) * stride_2302;
      size_2304 = gf_max(size_2304, 0);
      D_12856 = size_2304 + -1;
      D_12857 = (ulong) (ulong) (size_2304) * 32;
      D_12858 = (ulong) (size_2304) * 4;
      D_12859 = (device void *) workspace.allocate (gf_max((ulong) (size_2304 * 4), 1));
      dd_massentro = (device GFReal *) D_12859;
      offset_2303 = -(lbound_2298) - lbound_2300 * stride_2302;
      lbound_2305 = (long) *its;
      ubound_2306 = (long) *ite;
      stride_2309 = (1 - lbound_2305) + ubound_2306;
      stride_2309 = gf_max(stride_2309, 0);
      lbound_2307 = (long) *kts;
      ubound_2308 = (long) *kte;
      size_2311 = ((1 - lbound_2307) + ubound_2308) * stride_2309;
      size_2311 = gf_max(size_2311, 0);
      D_12852 = size_2311 + -1;
      D_12853 = (ulong) (ulong) (size_2311) * 32;
      D_12854 = (ulong) (size_2311) * 4;
      D_12855 = (device void *) workspace.allocate (gf_max((ulong) (size_2311 * 4), 1));
      dellah = (device GFReal *) D_12855;
      offset_2310 = -(lbound_2305) - lbound_2307 * stride_2309;
      lbound_2312 = (long) *its;
      ubound_2313 = (long) *ite;
      stride_2316 = (1 - lbound_2312) + ubound_2313;
      stride_2316 = gf_max(stride_2316, 0);
      lbound_2314 = (long) *kts;
      ubound_2315 = (long) *kte;
      size_2318 = ((1 - lbound_2314) + ubound_2315) * stride_2316;
      size_2318 = gf_max(size_2318, 0);
      D_12848 = size_2318 + -1;
      D_12849 = (ulong) (ulong) (size_2318) * 32;
      D_12850 = (ulong) (size_2318) * 4;
      D_12851 = (device void *) workspace.allocate (gf_max((ulong) (size_2318 * 4), 1));
      dellaq = (device GFReal *) D_12851;
      offset_2317 = -(lbound_2312) - lbound_2314 * stride_2316;
      lbound_2319 = (long) *its;
      ubound_2320 = (long) *ite;
      stride_2323 = (1 - lbound_2319) + ubound_2320;
      stride_2323 = gf_max(stride_2323, 0);
      lbound_2321 = (long) *kts;
      ubound_2322 = (long) *kte;
      stride_2325 = ((1 - lbound_2321) + ubound_2322) * stride_2323;
      stride_2325 = gf_max(stride_2325, 0);
      ubound_2324 = (long) *maxens2;
      size_2327 = stride_2325 * (ubound_2324);
      size_2327 = gf_max(size_2327, 0);
      D_12844 = size_2327 + -1;
      D_12845 = (ulong) (ulong) (size_2327) * 32;
      D_12846 = (ulong) (size_2327) * 4;
      D_12847 = (device void *) workspace.allocate (gf_max((ulong) (size_2327 * 4), 1));
      dellaq_ens = (device GFReal *) D_12847;
      offset_2326 = (-(lbound_2319) - lbound_2321 * stride_2323) - (stride_2325);
      lbound_2328 = (long) *its;
      ubound_2329 = (long) *ite;
      stride_2332 = (1 - lbound_2328) + ubound_2329;
      stride_2332 = gf_max(stride_2332, 0);
      lbound_2330 = (long) *kts;
      ubound_2331 = (long) *kte;
      size_2334 = ((1 - lbound_2330) + ubound_2331) * stride_2332;
      size_2334 = gf_max(size_2334, 0);
      D_12840 = size_2334 + -1;
      D_12841 = (ulong) (ulong) (size_2334) * 32;
      D_12842 = (ulong) (size_2334) * 4;
      D_12843 = (device void *) workspace.allocate (gf_max((ulong) (size_2334 * 4), 1));
      dellaqc = (device GFReal *) D_12843;
      offset_2333 = -(lbound_2328) - lbound_2330 * stride_2332;
      lbound_2335 = (long) *its;
      ubound_2336 = (long) *ite;
      stride_2339 = (1 - lbound_2335) + ubound_2336;
      stride_2339 = gf_max(stride_2339, 0);
      lbound_2337 = (long) *kts;
      ubound_2338 = (long) *kte;
      stride_2341 = ((1 - lbound_2337) + ubound_2338) * stride_2339;
      stride_2341 = gf_max(stride_2341, 0);
      ubound_2340 = (long) *maxens2;
      size_2343 = stride_2341 * (ubound_2340);
      size_2343 = gf_max(size_2343, 0);
      D_12836 = size_2343 + -1;
      D_12837 = (ulong) (ulong) (size_2343) * 32;
      D_12838 = (ulong) (size_2343) * 4;
      D_12839 = (device void *) workspace.allocate (gf_max((ulong) (size_2343 * 4), 1));
      dellaqc_ens = (device GFReal *) D_12839;
      offset_2342 = (-(lbound_2335) - lbound_2337 * stride_2339) - (stride_2341);
      lbound_2344 = (long) *its;
      ubound_2345 = (long) *ite;
      stride_2348 = (1 - lbound_2344) + ubound_2345;
      stride_2348 = gf_max(stride_2348, 0);
      lbound_2346 = (long) *kts;
      ubound_2347 = (long) *kte;
      size_2350 = ((1 - lbound_2346) + ubound_2347) * stride_2348;
      size_2350 = gf_max(size_2350, 0);
      D_12832 = size_2350 + -1;
      D_12833 = (ulong) (ulong) (size_2350) * 32;
      D_12834 = (ulong) (size_2350) * 4;
      D_12835 = (device void *) workspace.allocate (gf_max((ulong) (size_2350 * 4), 1));
      dellat = (device GFReal *) D_12835;
      offset_2349 = -(lbound_2344) - lbound_2346 * stride_2348;
      lbound_2351 = (long) *its;
      ubound_2352 = (long) *ite;
      stride_2355 = (1 - lbound_2351) + ubound_2352;
      stride_2355 = gf_max(stride_2355, 0);
      lbound_2353 = (long) *kts;
      ubound_2354 = (long) *kte;
      stride_2357 = ((1 - lbound_2353) + ubound_2354) * stride_2355;
      stride_2357 = gf_max(stride_2357, 0);
      ubound_2356 = (long) *maxens2;
      size_2359 = stride_2357 * (ubound_2356);
      size_2359 = gf_max(size_2359, 0);
      D_12828 = size_2359 + -1;
      D_12829 = (ulong) (ulong) (size_2359) * 32;
      D_12830 = (ulong) (size_2359) * 4;
      D_12831 = (device void *) workspace.allocate (gf_max((ulong) (size_2359 * 4), 1));
      dellat_ens = (device GFReal *) D_12831;
      offset_2358 = (-(lbound_2351) - lbound_2353 * stride_2355) - (stride_2357);
      lbound_2360 = (long) *its;
      ubound_2361 = (long) *ite;
      stride_2364 = (1 - lbound_2360) + ubound_2361;
      stride_2364 = gf_max(stride_2364, 0);
      lbound_2362 = (long) *kts;
      ubound_2363 = (long) *kte;
      size_2366 = ((1 - lbound_2362) + ubound_2363) * stride_2364;
      size_2366 = gf_max(size_2366, 0);
      D_12824 = size_2366 + -1;
      D_12825 = (ulong) (ulong) (size_2366) * 32;
      D_12826 = (ulong) (size_2366) * 4;
      D_12827 = (device void *) workspace.allocate (gf_max((ulong) (size_2366 * 4), 1));
      dsubh = (device GFReal *) D_12827;
      offset_2365 = -(lbound_2360) - lbound_2362 * stride_2364;
      lbound_2367 = (long) *its;
      ubound_2368 = (long) *ite;
      stride_2371 = (1 - lbound_2367) + ubound_2368;
      stride_2371 = gf_max(stride_2371, 0);
      lbound_2369 = (long) *kts;
      ubound_2370 = (long) *kte;
      size_2373 = ((1 - lbound_2369) + ubound_2370) * stride_2371;
      size_2373 = gf_max(size_2373, 0);
      D_12820 = size_2373 + -1;
      D_12821 = (ulong) (ulong) (size_2373) * 32;
      D_12822 = (ulong) (size_2373) * 4;
      D_12823 = (device void *) workspace.allocate (gf_max((ulong) (size_2373 * 4), 1));
      dsubq = (device GFReal *) D_12823;
      offset_2372 = -(lbound_2367) - lbound_2369 * stride_2371;
      lbound_2374 = (long) *its;
      ubound_2375 = (long) *ite;
      stride_2378 = (1 - lbound_2374) + ubound_2375;
      stride_2378 = gf_max(stride_2378, 0);
      lbound_2376 = (long) *kts;
      ubound_2377 = (long) *kte;
      size_2380 = ((1 - lbound_2376) + ubound_2377) * stride_2378;
      size_2380 = gf_max(size_2380, 0);
      D_12816 = size_2380 + -1;
      D_12817 = (ulong) (ulong) (size_2380) * 32;
      D_12818 = (ulong) (size_2380) * 4;
      D_12819 = (device void *) workspace.allocate (gf_max((ulong) (size_2380 * 4), 1));
      dsubt = (device GFReal *) D_12819;
      offset_2379 = -(lbound_2374) - lbound_2376 * stride_2378;
      lbound_2385 = (long) *its;
      ubound_2386 = (long) *ite;
      size_2388 = (1 - lbound_2385) + ubound_2386;
      size_2388 = gf_max(size_2388, 0);
      D_12812 = size_2388 + -1;
      D_12813 = (ulong) (ulong) (size_2388) * 32;
      D_12814 = (ulong) (size_2388) * 4;
      D_12815 = (device void *) workspace.allocate (gf_max((ulong) (size_2388 * 4), 1));
      edt = (device GFReal *) D_12815;
      offset_2387 = -(lbound_2385);
      ubound_2389 = (long) *maxens2;
      size_2390 = (ubound_2389);
      size_2390 = gf_max(size_2390, 0);
      D_12808 = size_2390 + -1;
      D_12809 = (ulong) (ulong) (size_2390) * 32;
      D_12810 = (ulong) (size_2390) * 4;
      D_12811 = (device void *) workspace.allocate (gf_max((ulong) (size_2390 * 4), 1));
      edt_ens = (device GFReal *) D_12811;
      lbound_2391 = (long) *its;
      ubound_2392 = (long) *ite;
      stride_2394 = (1 - lbound_2391) + ubound_2392;
      stride_2394 = gf_max(stride_2394, 0);
      ubound_2393 = (long) *maxens2;
      size_2396 = stride_2394 * (ubound_2393);
      size_2396 = gf_max(size_2396, 0);
      D_12804 = size_2396 + -1;
      D_12805 = (ulong) (ulong) (size_2396) * 32;
      D_12806 = (ulong) (size_2396) * 4;
      D_12807 = (device void *) workspace.allocate (gf_max((ulong) (size_2396 * 4), 1));
      edtc = (device GFReal *) D_12807;
      offset_2395 = -(lbound_2391) - (stride_2394);
      lbound_2397 = (long) *its;
      ubound_2398 = (long) *ite;
      size_2400 = (1 - lbound_2397) + ubound_2398;
      size_2400 = gf_max(size_2400, 0);
      D_12800 = size_2400 + -1;
      D_12801 = (ulong) (ulong) (size_2400) * 32;
      D_12802 = (ulong) (size_2400) * 4;
      D_12803 = (device void *) workspace.allocate (gf_max((ulong) (size_2400 * 4), 1));
      edto = (device GFReal *) D_12803;
      offset_2399 = -(lbound_2397);
      lbound_2401 = (long) *its;
      ubound_2402 = (long) *ite;
      size_2404 = (1 - lbound_2401) + ubound_2402;
      size_2404 = gf_max(size_2404, 0);
      D_12796 = size_2404 + -1;
      D_12797 = (ulong) (ulong) (size_2404) * 32;
      D_12798 = (ulong) (size_2404) * 4;
      D_12799 = (device void *) workspace.allocate (gf_max((ulong) (size_2404 * 4), 1));
      edtx = (device GFReal *) D_12799;
      offset_2403 = -(lbound_2401);
      lbound_2405 = (long) *its;
      ubound_2406 = (long) *ite;
      size_2408 = (1 - lbound_2405) + ubound_2406;
      size_2408 = gf_max(size_2408, 0);
      D_12792 = size_2408 + -1;
      D_12793 = (ulong) (ulong) (size_2408) * 32;
      D_12794 = (ulong) (size_2408) * 4;
      D_12795 = (device void *) workspace.allocate (gf_max((ulong) (size_2408 * 4), 1));
      entr_rate = (device GFReal *) D_12795;
      offset_2407 = -(lbound_2405);
      lbound_2409 = (long) *its;
      ubound_2410 = (long) *ite;
      stride_2413 = (1 - lbound_2409) + ubound_2410;
      stride_2413 = gf_max(stride_2413, 0);
      lbound_2411 = (long) *kts;
      ubound_2412 = (long) *kte;
      size_2415 = ((1 - lbound_2411) + ubound_2412) * stride_2413;
      size_2415 = gf_max(size_2415, 0);
      D_12788 = size_2415 + -1;
      D_12789 = (ulong) (ulong) (size_2415) * 32;
      D_12790 = (ulong) (size_2415) * 4;
      D_12791 = (device void *) workspace.allocate (gf_max((ulong) (size_2415 * 4), 1));
      entr_rate_2d = (device GFReal *) D_12791;
      offset_2414 = -(lbound_2409) - lbound_2411 * stride_2413;
      lbound_2416 = (long) *its;
      ubound_2417 = (long) *ite;
      size_2419 = (1 - lbound_2416) + ubound_2417;
      size_2419 = gf_max(size_2419, 0);
      D_12784 = size_2419 + -1;
      D_12785 = (ulong) (ulong) (size_2419) * 32;
      D_12786 = (ulong) (size_2419) * 4;
      D_12787 = (device void *) workspace.allocate (gf_max((ulong) (size_2419 * 4), 1));
      frh = (device GFReal *) D_12787;
      offset_2418 = -(lbound_2416);
      lbound_2420 = (long) *its;
      ubound_2421 = (long) *ite;
      stride_2424 = (1 - lbound_2420) + ubound_2421;
      stride_2424 = gf_max(stride_2424, 0);
      lbound_2422 = (long) *kts;
      ubound_2423 = (long) *kte;
      size_2426 = ((1 - lbound_2422) + ubound_2423) * stride_2424;
      size_2426 = gf_max(size_2426, 0);
      D_12780 = size_2426 + -1;
      D_12781 = (ulong) (ulong) (size_2426) * 32;
      D_12782 = (ulong) (size_2426) * 4;
      D_12783 = (device void *) workspace.allocate (gf_max((ulong) (size_2426 * 4), 1));
      gamma_cup = (device GFReal *) D_12783;
      offset_2425 = -(lbound_2420) - lbound_2422 * stride_2424;
      lbound_2427 = (long) *its;
      ubound_2428 = (long) *ite;
      stride_2431 = (1 - lbound_2427) + ubound_2428;
      stride_2431 = gf_max(stride_2431, 0);
      lbound_2429 = (long) *kts;
      ubound_2430 = (long) *kte;
      size_2433 = ((1 - lbound_2429) + ubound_2430) * stride_2431;
      size_2433 = gf_max(size_2433, 0);
      D_12776 = size_2433 + -1;
      D_12777 = (ulong) (ulong) (size_2433) * 32;
      D_12778 = (ulong) (size_2433) * 4;
      D_12779 = (device void *) workspace.allocate (gf_max((ulong) (size_2433 * 4), 1));
      gammao_cup = (device GFReal *) D_12779;
      offset_2432 = -(lbound_2427) - lbound_2429 * stride_2431;
      lbound_2434 = (long) *its;
      ubound_2435 = (long) *ite;
      stride_2438 = (1 - lbound_2434) + ubound_2435;
      stride_2438 = gf_max(stride_2438, 0);
      lbound_2436 = (long) *kts;
      ubound_2437 = (long) *kte;
      size_2440 = ((1 - lbound_2436) + ubound_2437) * stride_2438;
      size_2440 = gf_max(size_2440, 0);
      D_12772 = size_2440 + -1;
      D_12773 = (ulong) (ulong) (size_2440) * 32;
      D_12774 = (ulong) (size_2440) * 4;
      D_12775 = (device void *) workspace.allocate (gf_max((ulong) (size_2440 * 4), 1));
      hc = (device GFReal *) D_12775;
      offset_2439 = -(lbound_2434) - lbound_2436 * stride_2438;
      lbound_2441 = (long) *its;
      ubound_2442 = (long) *ite;
      stride_2445 = (1 - lbound_2441) + ubound_2442;
      stride_2445 = gf_max(stride_2445, 0);
      lbound_2443 = (long) *kts;
      ubound_2444 = (long) *kte;
      size_2447 = ((1 - lbound_2443) + ubound_2444) * stride_2445;
      size_2447 = gf_max(size_2447, 0);
      D_12768 = size_2447 + -1;
      D_12769 = (ulong) (ulong) (size_2447) * 32;
      D_12770 = (ulong) (size_2447) * 4;
      D_12771 = (device void *) workspace.allocate (gf_max((ulong) (size_2447 * 4), 1));
      hcdo = (device GFReal *) D_12771;
      offset_2446 = -(lbound_2441) - lbound_2443 * stride_2445;
      lbound_2448 = (long) *its;
      ubound_2449 = (long) *ite;
      stride_2452 = (1 - lbound_2448) + ubound_2449;
      stride_2452 = gf_max(stride_2452, 0);
      lbound_2450 = (long) *kts;
      ubound_2451 = (long) *kte;
      size_2454 = ((1 - lbound_2450) + ubound_2451) * stride_2452;
      size_2454 = gf_max(size_2454, 0);
      D_12764 = size_2454 + -1;
      D_12765 = (ulong) (ulong) (size_2454) * 32;
      D_12766 = (ulong) (size_2454) * 4;
      D_12767 = (device void *) workspace.allocate (gf_max((ulong) (size_2454 * 4), 1));
      hco = (device GFReal *) D_12767;
      offset_2453 = -(lbound_2448) - lbound_2450 * stride_2452;
      lbound_2455 = (long) *its;
      ubound_2456 = (long) *ite;
      stride_2459 = (1 - lbound_2455) + ubound_2456;
      stride_2459 = gf_max(stride_2459, 0);
      lbound_2457 = (long) *kts;
      ubound_2458 = (long) *kte;
      size_2461 = ((1 - lbound_2457) + ubound_2458) * stride_2459;
      size_2461 = gf_max(size_2461, 0);
      D_12760 = size_2461 + -1;
      D_12761 = (ulong) (ulong) (size_2461) * 32;
      D_12762 = (ulong) (size_2461) * 4;
      D_12763 = (device void *) workspace.allocate (gf_max((ulong) (size_2461 * 4), 1));
      he = (device GFReal *) D_12763;
      offset_2460 = -(lbound_2455) - lbound_2457 * stride_2459;
      lbound_2462 = (long) *its;
      ubound_2463 = (long) *ite;
      stride_2466 = (1 - lbound_2462) + ubound_2463;
      stride_2466 = gf_max(stride_2466, 0);
      lbound_2464 = (long) *kts;
      ubound_2465 = (long) *kte;
      size_2468 = ((1 - lbound_2464) + ubound_2465) * stride_2466;
      size_2468 = gf_max(size_2468, 0);
      D_12756 = size_2468 + -1;
      D_12757 = (ulong) (ulong) (size_2468) * 32;
      D_12758 = (ulong) (size_2468) * 4;
      D_12759 = (device void *) workspace.allocate (gf_max((ulong) (size_2468 * 4), 1));
      he_cup = (device GFReal *) D_12759;
      offset_2467 = -(lbound_2462) - lbound_2464 * stride_2466;
      lbound_2469 = (long) *its;
      ubound_2470 = (long) *ite;
      stride_2473 = (1 - lbound_2469) + ubound_2470;
      stride_2473 = gf_max(stride_2473, 0);
      lbound_2471 = (long) *kts;
      ubound_2472 = (long) *kte;
      size_2475 = ((1 - lbound_2471) + ubound_2472) * stride_2473;
      size_2475 = gf_max(size_2475, 0);
      D_12752 = size_2475 + -1;
      D_12753 = (ulong) (ulong) (size_2475) * 32;
      D_12754 = (ulong) (size_2475) * 4;
      D_12755 = (device void *) workspace.allocate (gf_max((ulong) (size_2475 * 4), 1));
      heo = (device GFReal *) D_12755;
      offset_2474 = -(lbound_2469) - lbound_2471 * stride_2473;
      lbound_2476 = (long) *its;
      ubound_2477 = (long) *ite;
      stride_2480 = (1 - lbound_2476) + ubound_2477;
      stride_2480 = gf_max(stride_2480, 0);
      lbound_2478 = (long) *kts;
      ubound_2479 = (long) *kte;
      size_2482 = ((1 - lbound_2478) + ubound_2479) * stride_2480;
      size_2482 = gf_max(size_2482, 0);
      D_12748 = size_2482 + -1;
      D_12749 = (ulong) (ulong) (size_2482) * 32;
      D_12750 = (ulong) (size_2482) * 4;
      D_12751 = (device void *) workspace.allocate (gf_max((ulong) (size_2482 * 4), 1));
      heo_cup = (device GFReal *) D_12751;
      offset_2481 = -(lbound_2476) - lbound_2478 * stride_2480;
      lbound_2483 = (long) *its;
      ubound_2484 = (long) *ite;
      stride_2487 = (1 - lbound_2483) + ubound_2484;
      stride_2487 = gf_max(stride_2487, 0);
      lbound_2485 = (long) *kts;
      ubound_2486 = (long) *kte;
      size_2489 = ((1 - lbound_2485) + ubound_2486) * stride_2487;
      size_2489 = gf_max(size_2489, 0);
      D_12744 = size_2489 + -1;
      D_12745 = (ulong) (ulong) (size_2489) * 32;
      D_12746 = (ulong) (size_2489) * 4;
      D_12747 = (device void *) workspace.allocate (gf_max((ulong) (size_2489 * 4), 1));
      hes = (device GFReal *) D_12747;
      offset_2488 = -(lbound_2483) - lbound_2485 * stride_2487;
      lbound_2490 = (long) *its;
      ubound_2491 = (long) *ite;
      stride_2494 = (1 - lbound_2490) + ubound_2491;
      stride_2494 = gf_max(stride_2494, 0);
      lbound_2492 = (long) *kts;
      ubound_2493 = (long) *kte;
      size_2496 = ((1 - lbound_2492) + ubound_2493) * stride_2494;
      size_2496 = gf_max(size_2496, 0);
      D_12740 = size_2496 + -1;
      D_12741 = (ulong) (ulong) (size_2496) * 32;
      D_12742 = (ulong) (size_2496) * 4;
      D_12743 = (device void *) workspace.allocate (gf_max((ulong) (size_2496 * 4), 1));
      hes_cup = (device GFReal *) D_12743;
      offset_2495 = -(lbound_2490) - lbound_2492 * stride_2494;
      lbound_2497 = (long) *its;
      ubound_2498 = (long) *ite;
      stride_2501 = (1 - lbound_2497) + ubound_2498;
      stride_2501 = gf_max(stride_2501, 0);
      lbound_2499 = (long) *kts;
      ubound_2500 = (long) *kte;
      size_2503 = ((1 - lbound_2499) + ubound_2500) * stride_2501;
      size_2503 = gf_max(size_2503, 0);
      D_12736 = size_2503 + -1;
      D_12737 = (ulong) (ulong) (size_2503) * 32;
      D_12738 = (ulong) (size_2503) * 4;
      D_12739 = (device void *) workspace.allocate (gf_max((ulong) (size_2503 * 4), 1));
      heso = (device GFReal *) D_12739;
      offset_2502 = -(lbound_2497) - lbound_2499 * stride_2501;
      lbound_2504 = (long) *its;
      ubound_2505 = (long) *ite;
      stride_2508 = (1 - lbound_2504) + ubound_2505;
      stride_2508 = gf_max(stride_2508, 0);
      lbound_2506 = (long) *kts;
      ubound_2507 = (long) *kte;
      size_2510 = ((1 - lbound_2506) + ubound_2507) * stride_2508;
      size_2510 = gf_max(size_2510, 0);
      D_12732 = size_2510 + -1;
      D_12733 = (ulong) (ulong) (size_2510) * 32;
      D_12734 = (ulong) (size_2510) * 4;
      D_12735 = (device void *) workspace.allocate (gf_max((ulong) (size_2510 * 4), 1));
      heso_cup = (device GFReal *) D_12735;
      offset_2509 = -(lbound_2504) - lbound_2506 * stride_2508;
      lbound_2511 = (long) *its;
      ubound_2512 = (long) *ite;
      size_2514 = (1 - lbound_2511) + ubound_2512;
      size_2514 = gf_max(size_2514, 0);
      D_12728 = size_2514 + -1;
      D_12729 = (ulong) (ulong) (size_2514) * 32;
      D_12730 = (ulong) (size_2514) * 4;
      D_12731 = (device void *) workspace.allocate (gf_max((ulong) (size_2514 * 4), 1));
      hkb = (device GFReal *) D_12731;
      offset_2513 = -(lbound_2511);
      lbound_2515 = (long) *its;
      ubound_2516 = (long) *ite;
      size_2518 = (1 - lbound_2515) + ubound_2516;
      size_2518 = gf_max(size_2518, 0);
      D_12724 = size_2518 + -1;
      D_12725 = (ulong) (ulong) (size_2518) * 32;
      D_12726 = (ulong) (size_2518) * 4;
      D_12727 = (device void *) workspace.allocate (gf_max((ulong) (size_2518 * 4), 1));
      hkbo = (device GFReal *) D_12727;
      offset_2517 = -(lbound_2515);
      lbound_2519 = (long) *its;
      ubound_2520 = (long) *ite;
      size_2522 = (1 - lbound_2519) + ubound_2520;
      size_2522 = gf_max(size_2522, 0);
      D_12720 = size_2522 + -1;
      D_12721 = (ulong) (ulong) (size_2522) * 32;
      D_12722 = (ulong) (size_2522) * 4;
      D_12723 = (device void *) workspace.allocate (gf_max((ulong) (size_2522 * 4), 1));
      ierr = (device int *) D_12723;
      offset_2521 = -(lbound_2519);
      lbound_2523 = (long) *its;
      ubound_2524 = (long) *ite;
      size_2526 = (1 - lbound_2523) + ubound_2524;
      size_2526 = gf_max(size_2526, 0);
      D_12716 = size_2526 + -1;
      D_12717 = (ulong) (ulong) (size_2526) * 32;
      D_12718 = (ulong) (size_2526) * 4;
      D_12719 = (device void *) workspace.allocate (gf_max((ulong) (size_2526 * 4), 1));
      ierr2 = (device int *) D_12719;
      offset_2525 = -(lbound_2523);
      lbound_2527 = (long) *its;
      ubound_2528 = (long) *ite;
      size_2530 = (1 - lbound_2527) + ubound_2528;
      size_2530 = gf_max(size_2530, 0);
      D_12712 = size_2530 + -1;
      D_12713 = (ulong) (ulong) (size_2530) * 32;
      D_12714 = (ulong) (size_2530) * 4;
      D_12715 = (device void *) workspace.allocate (gf_max((ulong) (size_2530 * 4), 1));
      ierr3 = (device int *) D_12715;
      offset_2529 = -(lbound_2527);
      lbound_2535 = (long) *its;
      ubound_2536 = (long) *ite;
      size_2538 = (1 - lbound_2535) + ubound_2536;
      size_2538 = gf_max(size_2538, 0);
      D_12708 = size_2538 + -1;
      D_12709 = (ulong) (ulong) (size_2538) * 32;
      D_12710 = (ulong) (size_2538) * 4;
      D_12711 = (device void *) workspace.allocate (gf_max((ulong) (size_2538 * 4), 1));
      jmin = (device int *) D_12711;
      offset_2537 = -(lbound_2535);
      lbound_2539 = (long) *its;
      ubound_2540 = (long) *ite;
      size_2542 = (1 - lbound_2539) + ubound_2540;
      size_2542 = gf_max(size_2542, 0);
      D_12704 = size_2542 + -1;
      D_12705 = (ulong) (ulong) (size_2542) * 32;
      D_12706 = (ulong) (size_2542) * 4;
      D_12707 = (device void *) workspace.allocate (gf_max((ulong) (size_2542 * 4), 1));
      k22 = (device int *) D_12707;
      offset_2541 = -(lbound_2539);
      lbound_2543 = (long) *its;
      ubound_2544 = (long) *ite;
      size_2546 = (1 - lbound_2543) + ubound_2544;
      size_2546 = gf_max(size_2546, 0);
      D_12700 = size_2546 + -1;
      D_12701 = (ulong) (ulong) (size_2546) * 32;
      D_12702 = (ulong) (size_2546) * 4;
      D_12703 = (device void *) workspace.allocate (gf_max((ulong) (size_2546 * 4), 1));
      k22x = (device int *) D_12703;
      offset_2545 = -(lbound_2543);
      lbound_2551 = (long) *its;
      ubound_2552 = (long) *ite;
      size_2554 = (1 - lbound_2551) + ubound_2552;
      size_2554 = gf_max(size_2554, 0);
      D_12696 = size_2554 + -1;
      D_12697 = (ulong) (ulong) (size_2554) * 32;
      D_12698 = (ulong) (size_2554) * 4;
      D_12699 = (device void *) workspace.allocate (gf_max((ulong) (size_2554 * 4), 1));
      kbconx = (device int *) D_12699;
      offset_2553 = -(lbound_2551);
      lbound_2555 = (long) *its;
      ubound_2556 = (long) *ite;
      size_2558 = (1 - lbound_2555) + ubound_2556;
      size_2558 = gf_max(size_2558, 0);
      D_12692 = size_2558 + -1;
      D_12693 = (ulong) (ulong) (size_2558) * 32;
      D_12694 = (ulong) (size_2558) * 4;
      D_12695 = (device void *) workspace.allocate (gf_max((ulong) (size_2558 * 4), 1));
      kbmax = (device int *) D_12695;
      offset_2557 = -(lbound_2555);
      lbound_2559 = (long) *its;
      ubound_2560 = (long) *ite;
      size_2562 = (1 - lbound_2559) + ubound_2560;
      size_2562 = gf_max(size_2562, 0);
      D_12688 = size_2562 + -1;
      D_12689 = (ulong) (ulong) (size_2562) * 32;
      D_12690 = (ulong) (size_2562) * 4;
      D_12691 = (device void *) workspace.allocate (gf_max((ulong) (size_2562 * 4), 1));
      kdet = (device int *) D_12691;
      offset_2561 = -(lbound_2559);
      lbound_2563 = (long) *its;
      ubound_2564 = (long) *ite;
      size_2566 = (1 - lbound_2563) + ubound_2564;
      size_2566 = gf_max(size_2566, 0);
      D_12684 = size_2566 + -1;
      D_12685 = (ulong) (ulong) (size_2566) * 32;
      D_12686 = (ulong) (size_2566) * 4;
      D_12687 = (device void *) workspace.allocate (gf_max((ulong) (size_2566 * 4), 1));
      kstabi = (device int *) D_12687;
      offset_2565 = -(lbound_2563);
      lbound_2567 = (long) *its;
      ubound_2568 = (long) *ite;
      size_2570 = (1 - lbound_2567) + ubound_2568;
      size_2570 = gf_max(size_2570, 0);
      D_12680 = size_2570 + -1;
      D_12681 = (ulong) (ulong) (size_2570) * 32;
      D_12682 = (ulong) (size_2570) * 4;
      D_12683 = (device void *) workspace.allocate (gf_max((ulong) (size_2570 * 4), 1));
      kstabm = (device int *) D_12683;
      offset_2569 = -(lbound_2567);
      lbound_2575 = (long) *its;
      ubound_2576 = (long) *ite;
      size_2578 = (1 - lbound_2575) + ubound_2576;
      size_2578 = gf_max(size_2578, 0);
      D_12676 = size_2578 + -1;
      D_12677 = (ulong) (ulong) (size_2578) * 32;
      D_12678 = (ulong) (size_2578) * 4;
      D_12679 = (device void *) workspace.allocate (gf_max((ulong) (size_2578 * 4), 1));
      kzdown = (device int *) D_12679;
      offset_2577 = -(lbound_2575);
      ubound_2579 = (long) *maxens;
      size_2580 = (ubound_2579);
      size_2580 = gf_max(size_2580, 0);
      D_12672 = size_2580 + -1;
      D_12673 = (ulong) (ulong) (size_2580) * 32;
      D_12674 = (ulong) (size_2580) * 4;
      D_12675 = (device void *) workspace.allocate (gf_max((ulong) (size_2580 * 4), 1));
      mbdt_ens = (device GFReal *) D_12675;
      lbound_2587 = (long) *its;
      ubound_2588 = (long) *ite;
      size_2590 = (1 - lbound_2587) + ubound_2588;
      size_2590 = gf_max(size_2590, 0);
      D_12668 = size_2590 + -1;
      D_12669 = (ulong) (ulong) (size_2590) * 32;
      D_12670 = (ulong) (size_2590) * 4;
      D_12671 = (device void *) workspace.allocate (gf_max((ulong) (size_2590 * 4), 1));
      mentrd_rate = (device GFReal *) D_12671;
      offset_2589 = -(lbound_2587);
      lbound_2591 = (long) *its;
      ubound_2592 = (long) *ite;
      stride_2595 = (1 - lbound_2591) + ubound_2592;
      stride_2595 = gf_max(stride_2595, 0);
      lbound_2593 = (long) *kts;
      ubound_2594 = (long) *kte;
      size_2597 = ((1 - lbound_2593) + ubound_2594) * stride_2595;
      size_2597 = gf_max(size_2597, 0);
      D_12664 = size_2597 + -1;
      D_12665 = (ulong) (ulong) (size_2597) * 32;
      D_12666 = (ulong) (size_2597) * 4;
      D_12667 = (device void *) workspace.allocate (gf_max((ulong) (size_2597 * 4), 1));
      mentrd_rate_2d = (device GFReal *) D_12667;
      offset_2596 = -(lbound_2591) - lbound_2593 * stride_2595;
      lbound_2635 = (long) *its;
      ubound_2636 = (long) *ite;
      stride_2639 = (1 - lbound_2635) + ubound_2636;
      stride_2639 = gf_max(stride_2639, 0);
      lbound_2637 = (long) *kts;
      ubound_2638 = (long) *kte;
      size_2641 = ((1 - lbound_2637) + ubound_2638) * stride_2639;
      size_2641 = gf_max(size_2641, 0);
      D_12660 = size_2641 + -1;
      D_12661 = (ulong) (ulong) (size_2641) * 32;
      D_12662 = (ulong) (size_2641) * 4;
      D_12663 = (device void *) workspace.allocate (gf_max((ulong) (size_2641 * 4), 1));
      p_cup = (device GFReal *) D_12663;
      offset_2640 = -(lbound_2635) - lbound_2637 * stride_2639;
      lbound_2649 = (long) *its;
      ubound_2650 = (long) *ite;
      stride_2653 = (1 - lbound_2649) + ubound_2650;
      stride_2653 = gf_max(stride_2653, 0);
      lbound_2651 = (long) *kts;
      ubound_2652 = (long) *kte;
      size_2655 = ((1 - lbound_2651) + ubound_2652) * stride_2653;
      size_2655 = gf_max(size_2655, 0);
      D_12656 = size_2655 + -1;
      D_12657 = (ulong) (ulong) (size_2655) * 32;
      D_12658 = (ulong) (size_2655) * 4;
      D_12659 = (device void *) workspace.allocate (gf_max((ulong) (size_2655 * 4), 1));
      po_cup = (device GFReal *) D_12659;
      offset_2654 = -(lbound_2649) - lbound_2651 * stride_2653;
      lbound_2669 = (long) *its;
      ubound_2670 = (long) *ite;
      size_2672 = (1 - lbound_2669) + ubound_2670;
      size_2672 = gf_max(size_2672, 0);
      D_12652 = size_2672 + -1;
      D_12653 = (ulong) (ulong) (size_2672) * 32;
      D_12654 = (ulong) (size_2672) * 4;
      D_12655 = (device void *) workspace.allocate (gf_max((ulong) (size_2672 * 4), 1));
      psum = (device GFReal *) D_12655;
      offset_2671 = -(lbound_2669);
      lbound_2673 = (long) *its;
      ubound_2674 = (long) *ite;
      size_2676 = (1 - lbound_2673) + ubound_2674;
      size_2676 = gf_max(size_2676, 0);
      D_12648 = size_2676 + -1;
      D_12649 = (ulong) (ulong) (size_2676) * 32;
      D_12650 = (ulong) (size_2676) * 4;
      D_12651 = (device void *) workspace.allocate (gf_max((ulong) (size_2676 * 4), 1));
      psumh = (device GFReal *) D_12651;
      offset_2675 = -(lbound_2673);
      lbound_2681 = (long) *its;
      ubound_2682 = (long) *ite;
      size_2684 = (1 - lbound_2681) + ubound_2682;
      size_2684 = gf_max(size_2684, 0);
      D_12644 = size_2684 + -1;
      D_12645 = (ulong) (ulong) (size_2684) * 32;
      D_12646 = (ulong) (size_2684) * 4;
      D_12647 = (device void *) workspace.allocate (gf_max((ulong) (size_2684 * 4), 1));
      pwavo = (device GFReal *) D_12647;
      offset_2683 = -(lbound_2681);
      lbound_2685 = (long) *its;
      ubound_2686 = (long) *ite;
      stride_2689 = (1 - lbound_2685) + ubound_2686;
      stride_2689 = gf_max(stride_2689, 0);
      lbound_2687 = (long) *kts;
      ubound_2688 = (long) *kte;
      size_2691 = ((1 - lbound_2687) + ubound_2688) * stride_2689;
      size_2691 = gf_max(size_2691, 0);
      D_12640 = size_2691 + -1;
      D_12641 = (ulong) (ulong) (size_2691) * 32;
      D_12642 = (ulong) (size_2691) * 4;
      D_12643 = (device void *) workspace.allocate (gf_max((ulong) (size_2691 * 4), 1));
      pwdo = (device GFReal *) D_12643;
      offset_2690 = -(lbound_2685) - lbound_2687 * stride_2689;
      lbound_2692 = (long) *its;
      ubound_2693 = (long) *ite;
      size_2695 = (1 - lbound_2692) + ubound_2693;
      size_2695 = gf_max(size_2695, 0);
      D_12636 = size_2695 + -1;
      D_12637 = (ulong) (ulong) (size_2695) * 32;
      D_12638 = (ulong) (size_2695) * 4;
      D_12639 = (device void *) workspace.allocate (gf_max((ulong) (size_2695 * 4), 1));
      pwevo = (device GFReal *) D_12639;
      offset_2694 = -(lbound_2692);
      lbound_2696 = (long) *its;
      ubound_2697 = (long) *ite;
      stride_2700 = (1 - lbound_2696) + ubound_2697;
      stride_2700 = gf_max(stride_2700, 0);
      lbound_2698 = (long) *kts;
      ubound_2699 = (long) *kte;
      size_2702 = ((1 - lbound_2698) + ubound_2699) * stride_2700;
      size_2702 = gf_max(size_2702, 0);
      D_12632 = size_2702 + -1;
      D_12633 = (ulong) (ulong) (size_2702) * 32;
      D_12634 = (ulong) (size_2702) * 4;
      D_12635 = (device void *) workspace.allocate (gf_max((ulong) (size_2702 * 4), 1));
      pwo = (device GFReal *) D_12635;
      offset_2701 = -(lbound_2696) - lbound_2698 * stride_2700;
      lbound_2703 = (long) *its;
      ubound_2704 = (long) *ite;
      stride_2707 = (1 - lbound_2703) + ubound_2704;
      stride_2707 = gf_max(stride_2707, 0);
      lbound_2705 = (long) *kts;
      ubound_2706 = (long) *kte;
      stride_2709 = ((1 - lbound_2705) + ubound_2706) * stride_2707;
      stride_2709 = gf_max(stride_2709, 0);
      ubound_2708 = (long) *maxens2;
      size_2711 = stride_2709 * (ubound_2708);
      size_2711 = gf_max(size_2711, 0);
      D_12628 = size_2711 + -1;
      D_12629 = (ulong) (ulong) (size_2711) * 32;
      D_12630 = (ulong) (size_2711) * 4;
      D_12631 = (device void *) workspace.allocate (gf_max((ulong) (size_2711 * 4), 1));
      pwo_ens = (device GFReal *) D_12631;
      offset_2710 = (-(lbound_2703) - lbound_2705 * stride_2707) - (stride_2709);
      lbound_2719 = (long) *its;
      ubound_2720 = (long) *ite;
      stride_2723 = (1 - lbound_2719) + ubound_2720;
      stride_2723 = gf_max(stride_2723, 0);
      lbound_2721 = (long) *kts;
      ubound_2722 = (long) *kte;
      size_2725 = ((1 - lbound_2721) + ubound_2722) * stride_2723;
      size_2725 = gf_max(size_2725, 0);
      D_12624 = size_2725 + -1;
      D_12625 = (ulong) (ulong) (size_2725) * 32;
      D_12626 = (ulong) (size_2725) * 4;
      D_12627 = (device void *) workspace.allocate (gf_max((ulong) (size_2725 * 4), 1));
      q_cup = (device GFReal *) D_12627;
      offset_2724 = -(lbound_2719) - lbound_2721 * stride_2723;
      lbound_2726 = (long) *its;
      ubound_2727 = (long) *ite;
      stride_2730 = (1 - lbound_2726) + ubound_2727;
      stride_2730 = gf_max(stride_2730, 0);
      lbound_2728 = (long) *kts;
      ubound_2729 = (long) *kte;
      size_2732 = ((1 - lbound_2728) + ubound_2729) * stride_2730;
      size_2732 = gf_max(size_2732, 0);
      D_12620 = size_2732 + -1;
      D_12621 = (ulong) (ulong) (size_2732) * 32;
      D_12622 = (ulong) (size_2732) * 4;
      D_12623 = (device void *) workspace.allocate (gf_max((ulong) (size_2732 * 4), 1));
      qcdo = (device GFReal *) D_12623;
      offset_2731 = -(lbound_2726) - lbound_2728 * stride_2730;
      lbound_2733 = (long) *its;
      ubound_2734 = (long) *ite;
      stride_2737 = (1 - lbound_2733) + ubound_2734;
      stride_2737 = gf_max(stride_2737, 0);
      lbound_2735 = (long) *kts;
      ubound_2736 = (long) *kte;
      size_2739 = ((1 - lbound_2735) + ubound_2736) * stride_2737;
      size_2739 = gf_max(size_2739, 0);
      D_12616 = size_2739 + -1;
      D_12617 = (ulong) (ulong) (size_2739) * 32;
      D_12618 = (ulong) (size_2739) * 4;
      D_12619 = (device void *) workspace.allocate (gf_max((ulong) (size_2739 * 4), 1));
      qco = (device GFReal *) D_12619;
      offset_2738 = -(lbound_2733) - lbound_2735 * stride_2737;
      lbound_2740 = (long) *its;
      ubound_2741 = (long) *ite;
      stride_2744 = (1 - lbound_2740) + ubound_2741;
      stride_2744 = gf_max(stride_2744, 0);
      lbound_2742 = (long) *kts;
      ubound_2743 = (long) *kte;
      size_2746 = ((1 - lbound_2742) + ubound_2743) * stride_2744;
      size_2746 = gf_max(size_2746, 0);
      D_12612 = size_2746 + -1;
      D_12613 = (ulong) (ulong) (size_2746) * 32;
      D_12614 = (ulong) (size_2746) * 4;
      D_12615 = (device void *) workspace.allocate (gf_max((ulong) (size_2746 * 4), 1));
      qes = (device GFReal *) D_12615;
      offset_2745 = -(lbound_2740) - lbound_2742 * stride_2744;
      lbound_2747 = (long) *its;
      ubound_2748 = (long) *ite;
      stride_2751 = (1 - lbound_2747) + ubound_2748;
      stride_2751 = gf_max(stride_2751, 0);
      lbound_2749 = (long) *kts;
      ubound_2750 = (long) *kte;
      size_2753 = ((1 - lbound_2749) + ubound_2750) * stride_2751;
      size_2753 = gf_max(size_2753, 0);
      D_12608 = size_2753 + -1;
      D_12609 = (ulong) (ulong) (size_2753) * 32;
      D_12610 = (ulong) (size_2753) * 4;
      D_12611 = (device void *) workspace.allocate (gf_max((ulong) (size_2753 * 4), 1));
      qes_cup = (device GFReal *) D_12611;
      offset_2752 = -(lbound_2747) - lbound_2749 * stride_2751;
      lbound_2754 = (long) *its;
      ubound_2755 = (long) *ite;
      stride_2758 = (1 - lbound_2754) + ubound_2755;
      stride_2758 = gf_max(stride_2758, 0);
      lbound_2756 = (long) *kts;
      ubound_2757 = (long) *kte;
      size_2760 = ((1 - lbound_2756) + ubound_2757) * stride_2758;
      size_2760 = gf_max(size_2760, 0);
      D_12604 = size_2760 + -1;
      D_12605 = (ulong) (ulong) (size_2760) * 32;
      D_12606 = (ulong) (size_2760) * 4;
      D_12607 = (device void *) workspace.allocate (gf_max((ulong) (size_2760 * 4), 1));
      qeso = (device GFReal *) D_12607;
      offset_2759 = -(lbound_2754) - lbound_2756 * stride_2758;
      lbound_2761 = (long) *its;
      ubound_2762 = (long) *ite;
      stride_2765 = (1 - lbound_2761) + ubound_2762;
      stride_2765 = gf_max(stride_2765, 0);
      lbound_2763 = (long) *kts;
      ubound_2764 = (long) *kte;
      size_2767 = ((1 - lbound_2763) + ubound_2764) * stride_2765;
      size_2767 = gf_max(size_2767, 0);
      D_12600 = size_2767 + -1;
      D_12601 = (ulong) (ulong) (size_2767) * 32;
      D_12602 = (ulong) (size_2767) * 4;
      D_12603 = (device void *) workspace.allocate (gf_max((ulong) (size_2767 * 4), 1));
      qeso_cup = (device GFReal *) D_12603;
      offset_2766 = -(lbound_2761) - lbound_2763 * stride_2765;
      lbound_2775 = (long) *its;
      ubound_2776 = (long) *ite;
      stride_2779 = (1 - lbound_2775) + ubound_2776;
      stride_2779 = gf_max(stride_2779, 0);
      lbound_2777 = (long) *kts;
      ubound_2778 = (long) *kte;
      size_2781 = ((1 - lbound_2777) + ubound_2778) * stride_2779;
      size_2781 = gf_max(size_2781, 0);
      D_12596 = size_2781 + -1;
      D_12597 = (ulong) (ulong) (size_2781) * 32;
      D_12598 = (ulong) (size_2781) * 4;
      D_12599 = (device void *) workspace.allocate (gf_max((ulong) (size_2781 * 4), 1));
      qo_cup = (device GFReal *) D_12599;
      offset_2780 = -(lbound_2775) - lbound_2777 * stride_2779;
      lbound_2782 = (long) *its;
      ubound_2783 = (long) *ite;
      stride_2786 = (1 - lbound_2782) + ubound_2783;
      stride_2786 = gf_max(stride_2786, 0);
      lbound_2784 = (long) *kts;
      ubound_2785 = (long) *kte;
      size_2788 = ((1 - lbound_2784) + ubound_2785) * stride_2786;
      size_2788 = gf_max(size_2788, 0);
      D_12592 = size_2788 + -1;
      D_12593 = (ulong) (ulong) (size_2788) * 32;
      D_12594 = (ulong) (size_2788) * 4;
      D_12595 = (device void *) workspace.allocate (gf_max((ulong) (size_2788 * 4), 1));
      qrcdo = (device GFReal *) D_12595;
      offset_2787 = -(lbound_2782) - lbound_2784 * stride_2786;
      lbound_2789 = (long) *its;
      ubound_2790 = (long) *ite;
      stride_2793 = (1 - lbound_2789) + ubound_2790;
      stride_2793 = gf_max(stride_2793, 0);
      lbound_2791 = (long) *kts;
      ubound_2792 = (long) *kte;
      size_2795 = ((1 - lbound_2791) + ubound_2792) * stride_2793;
      size_2795 = gf_max(size_2795, 0);
      D_12588 = size_2795 + -1;
      D_12589 = (ulong) (ulong) (size_2795) * 32;
      D_12590 = (ulong) (size_2795) * 4;
      D_12591 = (device void *) workspace.allocate (gf_max((ulong) (size_2795 * 4), 1));
      qrco = (device GFReal *) D_12591;
      offset_2794 = -(lbound_2789) - lbound_2791 * stride_2793;
      lbound_2796 = (long) *its;
      ubound_2797 = (long) *ite;
      size_2799 = (1 - lbound_2796) + ubound_2797;
      size_2799 = gf_max(size_2799, 0);
      D_12584 = size_2799 + -1;
      D_12585 = (ulong) (ulong) (size_2799) * 32;
      D_12586 = (ulong) (size_2799) * 4;
      D_12587 = (device void *) workspace.allocate (gf_max((ulong) (size_2799 * 4), 1));
      radius = (device GFReal *) D_12587;
      offset_2798 = -(lbound_2796);
      lbound_2807 = (long) *its;
      ubound_2808 = (long) *ite;
      size_2810 = (1 - lbound_2807) + ubound_2808;
      size_2810 = gf_max(size_2810, 0);
      D_12580 = size_2810 + -1;
      D_12581 = (ulong) (ulong) (size_2810) * 32;
      D_12582 = (ulong) (size_2810) * 4;
      D_12583 = (device void *) workspace.allocate (gf_max((ulong) (size_2810 * 4), 1));
      sig = (device GFReal *) D_12583;
      offset_2809 = -(lbound_2807);
      lbound_2811 = (long) *kts;
      ubound_2812 = (long) *kte;
      size_2814 = (1 - lbound_2811) + ubound_2812;
      size_2814 = gf_max(size_2814, 0);
      D_12576 = size_2814 + -1;
      D_12577 = (ulong) (ulong) (size_2814) * 32;
      D_12578 = (ulong) (size_2814) * 4;
      D_12579 = (device void *) workspace.allocate (gf_max((ulong) (size_2814 * 4), 1));
      smth = (device GFReal *) D_12579;
      offset_2813 = -(lbound_2811);
      lbound_2829 = (long) *its;
      ubound_2830 = (long) *ite;
      stride_2833 = (1 - lbound_2829) + ubound_2830;
      stride_2833 = gf_max(stride_2833, 0);
      lbound_2831 = (long) *kts;
      ubound_2832 = (long) *kte;
      stride_2835 = ((1 - lbound_2831) + ubound_2832) * stride_2833;
      stride_2835 = gf_max(stride_2835, 0);
      ubound_2834 = (long) *maxens2;
      size_2837 = stride_2835 * (ubound_2834);
      size_2837 = gf_max(size_2837, 0);
      D_12572 = size_2837 + -1;
      D_12573 = (ulong) (ulong) (size_2837) * 32;
      D_12574 = (ulong) (size_2837) * 4;
      D_12575 = (device void *) workspace.allocate (gf_max((ulong) (size_2837 * 4), 1));
      subq_ens = (device GFReal *) D_12575;
      offset_2836 = (-(lbound_2829) - lbound_2831 * stride_2833) - (stride_2835);
      lbound_2845 = (long) *its;
      ubound_2846 = (long) *ite;
      stride_2849 = (1 - lbound_2845) + ubound_2846;
      stride_2849 = gf_max(stride_2849, 0);
      lbound_2847 = (long) *kts;
      ubound_2848 = (long) *kte;
      stride_2851 = ((1 - lbound_2847) + ubound_2848) * stride_2849;
      stride_2851 = gf_max(stride_2851, 0);
      ubound_2850 = (long) *maxens2;
      size_2853 = stride_2851 * (ubound_2850);
      size_2853 = gf_max(size_2853, 0);
      D_12568 = size_2853 + -1;
      D_12569 = (ulong) (ulong) (size_2853) * 32;
      D_12570 = (ulong) (size_2853) * 4;
      D_12571 = (device void *) workspace.allocate (gf_max((ulong) (size_2853 * 4), 1));
      subt_ens = (device GFReal *) D_12571;
      offset_2852 = (-(lbound_2845) - lbound_2847 * stride_2849) - (stride_2851);
      lbound_2861 = (long) *its;
      ubound_2862 = (long) *ite;
      stride_2865 = (1 - lbound_2861) + ubound_2862;
      stride_2865 = gf_max(stride_2865, 0);
      lbound_2863 = (long) *kts;
      ubound_2864 = (long) *kte;
      size_2867 = ((1 - lbound_2863) + ubound_2864) * stride_2865;
      size_2867 = gf_max(size_2867, 0);
      D_12564 = size_2867 + -1;
      D_12565 = (ulong) (ulong) (size_2867) * 32;
      D_12566 = (ulong) (size_2867) * 4;
      D_12567 = (device void *) workspace.allocate (gf_max((ulong) (size_2867 * 4), 1));
      t_cup = (device GFReal *) D_12567;
      offset_2866 = -(lbound_2861) - lbound_2863 * stride_2865;
      lbound_2875 = (long) *its;
      ubound_2876 = (long) *ite;
      stride_2879 = (1 - lbound_2875) + ubound_2876;
      stride_2879 = gf_max(stride_2879, 0);
      lbound_2877 = (long) *kts;
      ubound_2878 = (long) *kte;
      size_2881 = ((1 - lbound_2877) + ubound_2878) * stride_2879;
      size_2881 = gf_max(size_2881, 0);
      D_12560 = size_2881 + -1;
      D_12561 = (ulong) (ulong) (size_2881) * 32;
      D_12562 = (ulong) (size_2881) * 4;
      D_12563 = (device void *) workspace.allocate (gf_max((ulong) (size_2881 * 4), 1));
      tn_cup = (device GFReal *) D_12563;
      offset_2880 = -(lbound_2875) - lbound_2877 * stride_2879;
      lbound_2882 = (long) *its;
      ubound_2883 = (long) *ite;
      stride_2886 = (1 - lbound_2882) + ubound_2883;
      stride_2886 = gf_max(stride_2886, 0);
      lbound_2884 = (long) *kts;
      ubound_2885 = (long) *kte;
      size_2888 = ((1 - lbound_2884) + ubound_2885) * stride_2886;
      size_2888 = gf_max(size_2888, 0);
      D_12556 = size_2888 + -1;
      D_12557 = (ulong) (ulong) (size_2888) * 32;
      D_12558 = (ulong) (size_2888) * 4;
      D_12559 = (device void *) workspace.allocate (gf_max((ulong) (size_2888 * 4), 1));
      up_massdetr = (device GFReal *) D_12559;
      offset_2887 = -(lbound_2882) - lbound_2884 * stride_2886;
      lbound_2889 = (long) *its;
      ubound_2890 = (long) *ite;
      stride_2893 = (1 - lbound_2889) + ubound_2890;
      stride_2893 = gf_max(stride_2893, 0);
      lbound_2891 = (long) *kts;
      ubound_2892 = (long) *kte;
      size_2895 = ((1 - lbound_2891) + ubound_2892) * stride_2893;
      size_2895 = gf_max(size_2895, 0);
      D_12552 = size_2895 + -1;
      D_12553 = (ulong) (ulong) (size_2895) * 32;
      D_12554 = (ulong) (size_2895) * 4;
      D_12555 = (device void *) workspace.allocate (gf_max((ulong) (size_2895 * 4), 1));
      up_massdetro = (device GFReal *) D_12555;
      offset_2894 = -(lbound_2889) - lbound_2891 * stride_2893;
      lbound_2896 = (long) *its;
      ubound_2897 = (long) *ite;
      stride_2900 = (1 - lbound_2896) + ubound_2897;
      stride_2900 = gf_max(stride_2900, 0);
      lbound_2898 = (long) *kts;
      ubound_2899 = (long) *kte;
      size_2902 = ((1 - lbound_2898) + ubound_2899) * stride_2900;
      size_2902 = gf_max(size_2902, 0);
      D_12548 = size_2902 + -1;
      D_12549 = (ulong) (ulong) (size_2902) * 32;
      D_12550 = (ulong) (size_2902) * 4;
      D_12551 = (device void *) workspace.allocate (gf_max((ulong) (size_2902 * 4), 1));
      up_massentr = (device GFReal *) D_12551;
      offset_2901 = -(lbound_2896) - lbound_2898 * stride_2900;
      lbound_2903 = (long) *its;
      ubound_2904 = (long) *ite;
      stride_2907 = (1 - lbound_2903) + ubound_2904;
      stride_2907 = gf_max(stride_2907, 0);
      lbound_2905 = (long) *kts;
      ubound_2906 = (long) *kte;
      size_2909 = ((1 - lbound_2905) + ubound_2906) * stride_2907;
      size_2909 = gf_max(size_2909, 0);
      D_12544 = size_2909 + -1;
      D_12545 = (ulong) (ulong) (size_2909) * 32;
      D_12546 = (ulong) (size_2909) * 4;
      D_12547 = (device void *) workspace.allocate (gf_max((ulong) (size_2909 * 4), 1));
      up_massentro = (device GFReal *) D_12547;
      offset_2908 = -(lbound_2903) - lbound_2905 * stride_2907;
      lbound_2924 = (long) *its;
      ubound_2925 = (long) *ite;
      stride_2928 = (1 - lbound_2924) + ubound_2925;
      stride_2928 = gf_max(stride_2928, 0);
      lbound_2926 = (long) *jts;
      ubound_2927 = (long) *jte;
      size_2930 = ((1 - lbound_2926) + ubound_2927) * stride_2928;
      size_2930 = gf_max(size_2930, 0);
      D_12540 = size_2930 + -1;
      D_12541 = (ulong) (ulong) (size_2930) * 32;
      D_12542 = (ulong) (size_2930) * 4;
      D_12543 = (device void *) workspace.allocate (gf_max((ulong) (size_2930 * 4), 1));
      weight_as = (device GFReal *) D_12543;
      offset_2929 = -(lbound_2924) - lbound_2926 * stride_2928;
      lbound_2931 = (long) *its;
      ubound_2932 = (long) *ite;
      stride_2935 = (1 - lbound_2931) + ubound_2932;
      stride_2935 = gf_max(stride_2935, 0);
      lbound_2933 = (long) *jts;
      ubound_2934 = (long) *jte;
      size_2937 = ((1 - lbound_2933) + ubound_2934) * stride_2935;
      size_2937 = gf_max(size_2937, 0);
      D_12536 = size_2937 + -1;
      D_12537 = (ulong) (ulong) (size_2937) * 32;
      D_12538 = (ulong) (size_2937) * 4;
      D_12539 = (device void *) workspace.allocate (gf_max((ulong) (size_2937 * 4), 1));
      weight_gr = (device GFReal *) D_12539;
      offset_2936 = -(lbound_2931) - lbound_2933 * stride_2935;
      lbound_2938 = (long) *its;
      ubound_2939 = (long) *ite;
      stride_2942 = (1 - lbound_2938) + ubound_2939;
      stride_2942 = gf_max(stride_2942, 0);
      lbound_2940 = (long) *jts;
      ubound_2941 = (long) *jte;
      size_2944 = ((1 - lbound_2940) + ubound_2941) * stride_2942;
      size_2944 = gf_max(size_2944, 0);
      D_12532 = size_2944 + -1;
      D_12533 = (ulong) (ulong) (size_2944) * 32;
      D_12534 = (ulong) (size_2944) * 4;
      D_12535 = (device void *) workspace.allocate (gf_max((ulong) (size_2944 * 4), 1));
      weight_mc = (device GFReal *) D_12535;
      offset_2943 = -(lbound_2938) - lbound_2940 * stride_2942;
      lbound_2945 = (long) *its;
      ubound_2946 = (long) *ite;
      stride_2949 = (1 - lbound_2945) + ubound_2946;
      stride_2949 = gf_max(stride_2949, 0);
      lbound_2947 = (long) *jts;
      ubound_2948 = (long) *jte;
      size_2951 = ((1 - lbound_2947) + ubound_2948) * stride_2949;
      size_2951 = gf_max(size_2951, 0);
      D_12528 = size_2951 + -1;
      D_12529 = (ulong) (ulong) (size_2951) * 32;
      D_12530 = (ulong) (size_2951) * 4;
      D_12531 = (device void *) workspace.allocate (gf_max((ulong) (size_2951 * 4), 1));
      weight_st = (device GFReal *) D_12531;
      offset_2950 = -(lbound_2945) - lbound_2947 * stride_2949;
      lbound_2952 = (long) *its;
      ubound_2953 = (long) *ite;
      stride_2956 = (1 - lbound_2952) + ubound_2953;
      stride_2956 = gf_max(stride_2956, 0);
      lbound_2954 = (long) *jts;
      ubound_2955 = (long) *jte;
      size_2958 = ((1 - lbound_2954) + ubound_2955) * stride_2956;
      size_2958 = gf_max(size_2958, 0);
      D_12524 = size_2958 + -1;
      D_12525 = (ulong) (ulong) (size_2958) * 32;
      D_12526 = (ulong) (size_2958) * 4;
      D_12527 = (device void *) workspace.allocate (gf_max((ulong) (size_2958 * 4), 1));
      weight_w = (device GFReal *) D_12527;
      offset_2957 = -(lbound_2952) - lbound_2954 * stride_2956;
      lbound_2959 = (long) *its;
      ubound_2960 = (long) *ite;
      size_2962 = (1 - lbound_2959) + ubound_2960;
      size_2962 = gf_max(size_2962, 0);
      D_12520 = size_2962 + -1;
      D_12521 = (ulong) (ulong) (size_2962) * 32;
      D_12522 = (ulong) (size_2962) * 4;
      D_12523 = (device void *) workspace.allocate (gf_max((ulong) (size_2962 * 4), 1));
      xaa0 = (device GFReal *) D_12523;
      offset_2961 = -(lbound_2959);
      lbound_2963 = (long) *its;
      ubound_2964 = (long) *ite;
      stride_2966 = (1 - lbound_2963) + ubound_2964;
      stride_2966 = gf_max(stride_2966, 0);
      ubound_2965 = (long) *maxens;
      size_2968 = stride_2966 * (ubound_2965);
      size_2968 = gf_max(size_2968, 0);
      D_12516 = size_2968 + -1;
      D_12517 = (ulong) (ulong) (size_2968) * 32;
      D_12518 = (ulong) (size_2968) * 4;
      D_12519 = (device void *) workspace.allocate (gf_max((ulong) (size_2968 * 4), 1));
      xaa0_ens = (device GFReal *) D_12519;
      offset_2967 = -(lbound_2963) - (stride_2966);
      lbound_2969 = (long) *its;
      ubound_2970 = (long) *ite;
      stride_2973 = (1 - lbound_2969) + ubound_2970;
      stride_2973 = gf_max(stride_2973, 0);
      lbound_2971 = (long) *kts;
      ubound_2972 = (long) *kte;
      size_2975 = ((1 - lbound_2971) + ubound_2972) * stride_2973;
      size_2975 = gf_max(size_2975, 0);
      D_12512 = size_2975 + -1;
      D_12513 = (ulong) (ulong) (size_2975) * 32;
      D_12514 = (ulong) (size_2975) * 4;
      D_12515 = (device void *) workspace.allocate (gf_max((ulong) (size_2975 * 4), 1));
      xdby = (device GFReal *) D_12515;
      offset_2974 = -(lbound_2969) - lbound_2971 * stride_2973;
      lbound_2985 = (long) *its;
      ubound_2986 = (long) *ite;
      stride_2989 = (1 - lbound_2985) + ubound_2986;
      stride_2989 = gf_max(stride_2989, 0);
      lbound_2987 = (long) *kts;
      ubound_2988 = (long) *kte;
      size_2991 = ((1 - lbound_2987) + ubound_2988) * stride_2989;
      size_2991 = gf_max(size_2991, 0);
      D_12508 = size_2991 + -1;
      D_12509 = (ulong) (ulong) (size_2991) * 32;
      D_12510 = (ulong) (size_2991) * 4;
      D_12511 = (device void *) workspace.allocate (gf_max((ulong) (size_2991 * 4), 1));
      xhc = (device GFReal *) D_12511;
      offset_2990 = -(lbound_2985) - lbound_2987 * stride_2989;
      lbound_2992 = (long) *its;
      ubound_2993 = (long) *ite;
      stride_2996 = (1 - lbound_2992) + ubound_2993;
      stride_2996 = gf_max(stride_2996, 0);
      lbound_2994 = (long) *kts;
      ubound_2995 = (long) *kte;
      size_2998 = ((1 - lbound_2994) + ubound_2995) * stride_2996;
      size_2998 = gf_max(size_2998, 0);
      D_12504 = size_2998 + -1;
      D_12505 = (ulong) (ulong) (size_2998) * 32;
      D_12506 = (ulong) (size_2998) * 4;
      D_12507 = (device void *) workspace.allocate (gf_max((ulong) (size_2998 * 4), 1));
      xhe = (device GFReal *) D_12507;
      offset_2997 = -(lbound_2992) - lbound_2994 * stride_2996;
      lbound_2999 = (long) *its;
      ubound_3000 = (long) *ite;
      stride_3003 = (1 - lbound_2999) + ubound_3000;
      stride_3003 = gf_max(stride_3003, 0);
      lbound_3001 = (long) *kts;
      ubound_3002 = (long) *kte;
      size_3005 = ((1 - lbound_3001) + ubound_3002) * stride_3003;
      size_3005 = gf_max(size_3005, 0);
      D_12500 = size_3005 + -1;
      D_12501 = (ulong) (ulong) (size_3005) * 32;
      D_12502 = (ulong) (size_3005) * 4;
      D_12503 = (device void *) workspace.allocate (gf_max((ulong) (size_3005 * 4), 1));
      xhe_cup = (device GFReal *) D_12503;
      offset_3004 = -(lbound_2999) - lbound_3001 * stride_3003;
      lbound_3006 = (long) *its;
      ubound_3007 = (long) *ite;
      stride_3010 = (1 - lbound_3006) + ubound_3007;
      stride_3010 = gf_max(stride_3010, 0);
      lbound_3008 = (long) *kts;
      ubound_3009 = (long) *kte;
      size_3012 = ((1 - lbound_3008) + ubound_3009) * stride_3010;
      size_3012 = gf_max(size_3012, 0);
      D_12496 = size_3012 + -1;
      D_12497 = (ulong) (ulong) (size_3012) * 32;
      D_12498 = (ulong) (size_3012) * 4;
      D_12499 = (device void *) workspace.allocate (gf_max((ulong) (size_3012 * 4), 1));
      xhes = (device GFReal *) D_12499;
      offset_3011 = -(lbound_3006) - lbound_3008 * stride_3010;
      lbound_3013 = (long) *its;
      ubound_3014 = (long) *ite;
      stride_3017 = (1 - lbound_3013) + ubound_3014;
      stride_3017 = gf_max(stride_3017, 0);
      lbound_3015 = (long) *kts;
      ubound_3016 = (long) *kte;
      size_3019 = ((1 - lbound_3015) + ubound_3016) * stride_3017;
      size_3019 = gf_max(size_3019, 0);
      D_12492 = size_3019 + -1;
      D_12493 = (ulong) (ulong) (size_3019) * 32;
      D_12494 = (ulong) (size_3019) * 4;
      D_12495 = (device void *) workspace.allocate (gf_max((ulong) (size_3019 * 4), 1));
      xhes_cup = (device GFReal *) D_12495;
      offset_3018 = -(lbound_3013) - lbound_3015 * stride_3017;
      lbound_3020 = (long) *its;
      ubound_3021 = (long) *ite;
      size_3023 = (1 - lbound_3020) + ubound_3021;
      size_3023 = gf_max(size_3023, 0);
      D_12488 = size_3023 + -1;
      D_12489 = (ulong) (ulong) (size_3023) * 32;
      D_12490 = (ulong) (size_3023) * 4;
      D_12491 = (device void *) workspace.allocate (gf_max((ulong) (size_3023 * 4), 1));
      xhkb = (device GFReal *) D_12491;
      offset_3022 = -(lbound_3020);
      lbound_3024 = (long) *its;
      ubound_3025 = (long) *ite;
      size_3027 = (1 - lbound_3024) + ubound_3025;
      size_3027 = gf_max(size_3027, 0);
      D_12484 = size_3027 + -1;
      D_12485 = (ulong) (ulong) (size_3027) * 32;
      D_12486 = (ulong) (size_3027) * 4;
      D_12487 = (device void *) workspace.allocate (gf_max((ulong) (size_3027 * 4), 1));
      xlamdd = (device GFReal *) D_12487;
      offset_3026 = -(lbound_3024);
      lbound_3032 = (long) *its;
      ubound_3033 = (long) *ite;
      size_3035 = (1 - lbound_3032) + ubound_3033;
      size_3035 = gf_max(size_3035, 0);
      D_12480 = size_3035 + -1;
      D_12481 = (ulong) (ulong) (size_3035) * 32;
      D_12482 = (ulong) (size_3035) * 4;
      D_12483 = (device void *) workspace.allocate (gf_max((ulong) (size_3035 * 4), 1));
      xland1 = (device GFReal *) D_12483;
      offset_3034 = -(lbound_3032);
      lbound_3040 = (long) *its;
      ubound_3041 = (long) *ite;
      stride_3044 = (1 - lbound_3040) + ubound_3041;
      stride_3044 = gf_max(stride_3044, 0);
      lbound_3042 = (long) *kts;
      ubound_3043 = (long) *kte;
      size_3046 = ((1 - lbound_3042) + ubound_3043) * stride_3044;
      size_3046 = gf_max(size_3046, 0);
      D_12476 = size_3046 + -1;
      D_12477 = (ulong) (ulong) (size_3046) * 32;
      D_12478 = (ulong) (size_3046) * 4;
      D_12479 = (device void *) workspace.allocate (gf_max((ulong) (size_3046 * 4), 1));
      xq = (device GFReal *) D_12479;
      offset_3045 = -(lbound_3040) - lbound_3042 * stride_3044;
      lbound_3047 = (long) *its;
      ubound_3048 = (long) *ite;
      stride_3051 = (1 - lbound_3047) + ubound_3048;
      stride_3051 = gf_max(stride_3051, 0);
      lbound_3049 = (long) *kts;
      ubound_3050 = (long) *kte;
      size_3053 = ((1 - lbound_3049) + ubound_3050) * stride_3051;
      size_3053 = gf_max(size_3053, 0);
      D_12472 = size_3053 + -1;
      D_12473 = (ulong) (ulong) (size_3053) * 32;
      D_12474 = (ulong) (size_3053) * 4;
      D_12475 = (device void *) workspace.allocate (gf_max((ulong) (size_3053 * 4), 1));
      xq_cup = (device GFReal *) D_12475;
      offset_3052 = -(lbound_3047) - lbound_3049 * stride_3051;
      lbound_3054 = (long) *its;
      ubound_3055 = (long) *ite;
      stride_3058 = (1 - lbound_3054) + ubound_3055;
      stride_3058 = gf_max(stride_3058, 0);
      lbound_3056 = (long) *kts;
      ubound_3057 = (long) *kte;
      size_3060 = ((1 - lbound_3056) + ubound_3057) * stride_3058;
      size_3060 = gf_max(size_3060, 0);
      D_12468 = size_3060 + -1;
      D_12469 = (ulong) (ulong) (size_3060) * 32;
      D_12470 = (ulong) (size_3060) * 4;
      D_12471 = (device void *) workspace.allocate (gf_max((ulong) (size_3060 * 4), 1));
      xqes = (device GFReal *) D_12471;
      offset_3059 = -(lbound_3054) - lbound_3056 * stride_3058;
      lbound_3061 = (long) *its;
      ubound_3062 = (long) *ite;
      stride_3065 = (1 - lbound_3061) + ubound_3062;
      stride_3065 = gf_max(stride_3065, 0);
      lbound_3063 = (long) *kts;
      ubound_3064 = (long) *kte;
      size_3067 = ((1 - lbound_3063) + ubound_3064) * stride_3065;
      size_3067 = gf_max(size_3067, 0);
      D_12464 = size_3067 + -1;
      D_12465 = (ulong) (ulong) (size_3067) * 32;
      D_12466 = (ulong) (size_3067) * 4;
      D_12467 = (device void *) workspace.allocate (gf_max((ulong) (size_3067 * 4), 1));
      xqes_cup = (device GFReal *) D_12467;
      offset_3066 = -(lbound_3061) - lbound_3063 * stride_3065;
      lbound_3068 = (long) *its;
      ubound_3069 = (long) *ite;
      stride_3072 = (1 - lbound_3068) + ubound_3069;
      stride_3072 = gf_max(stride_3072, 0);
      lbound_3070 = (long) *kts;
      ubound_3071 = (long) *kte;
      size_3074 = ((1 - lbound_3070) + ubound_3071) * stride_3072;
      size_3074 = gf_max(size_3074, 0);
      D_12460 = size_3074 + -1;
      D_12461 = (ulong) (ulong) (size_3074) * 32;
      D_12462 = (ulong) (size_3074) * 4;
      D_12463 = (device void *) workspace.allocate (gf_max((ulong) (size_3074 * 4), 1));
      xt = (device GFReal *) D_12463;
      offset_3073 = -(lbound_3068) - lbound_3070 * stride_3072;
      lbound_3075 = (long) *its;
      ubound_3076 = (long) *ite;
      stride_3079 = (1 - lbound_3075) + ubound_3076;
      stride_3079 = gf_max(stride_3079, 0);
      lbound_3077 = (long) *kts;
      ubound_3078 = (long) *kte;
      size_3081 = ((1 - lbound_3077) + ubound_3078) * stride_3079;
      size_3081 = gf_max(size_3081, 0);
      D_12456 = size_3081 + -1;
      D_12457 = (ulong) (ulong) (size_3081) * 32;
      D_12458 = (ulong) (size_3081) * 4;
      D_12459 = (device void *) workspace.allocate (gf_max((ulong) (size_3081 * 4), 1));
      xt_cup = (device GFReal *) D_12459;
      offset_3080 = -(lbound_3075) - lbound_3077 * stride_3079;
      lbound_3082 = (long) *its;
      ubound_3083 = (long) *ite;
      stride_3086 = (1 - lbound_3082) + ubound_3083;
      stride_3086 = gf_max(stride_3086, 0);
      lbound_3084 = (long) *kts;
      ubound_3085 = (long) *kte;
      size_3088 = ((1 - lbound_3084) + ubound_3085) * stride_3086;
      size_3088 = gf_max(size_3088, 0);
      D_12452 = size_3088 + -1;
      D_12453 = (ulong) (ulong) (size_3088) * 32;
      D_12454 = (ulong) (size_3088) * 4;
      D_12455 = (device void *) workspace.allocate (gf_max((ulong) (size_3088 * 4), 1));
      xz = (device GFReal *) D_12455;
      offset_3087 = -(lbound_3082) - lbound_3084 * stride_3086;
      lbound_3089 = (long) *its;
      ubound_3090 = (long) *ite;
      stride_3093 = (1 - lbound_3089) + ubound_3090;
      stride_3093 = gf_max(stride_3093, 0);
      lbound_3091 = (long) *kts;
      ubound_3092 = (long) *kte;
      size_3095 = ((1 - lbound_3091) + ubound_3092) * stride_3093;
      size_3095 = gf_max(size_3095, 0);
      D_12448 = size_3095 + -1;
      D_12449 = (ulong) (ulong) (size_3095) * 32;
      D_12450 = (ulong) (size_3095) * 4;
      D_12451 = (device void *) workspace.allocate (gf_max((ulong) (size_3095 * 4), 1));
      xz_cup = (device GFReal *) D_12451;
      offset_3094 = -(lbound_3089) - lbound_3091 * stride_3093;
      lbound_3096 = (long) *its;
      ubound_3097 = (long) *ite;
      stride_3100 = (1 - lbound_3096) + ubound_3097;
      stride_3100 = gf_max(stride_3100, 0);
      lbound_3098 = (long) *kts;
      ubound_3099 = (long) *kte;
      size_3102 = ((1 - lbound_3098) + ubound_3099) * stride_3100;
      size_3102 = gf_max(size_3102, 0);
      D_12444 = size_3102 + -1;
      D_12445 = (ulong) (ulong) (size_3102) * 32;
      D_12446 = (ulong) (size_3102) * 4;
      D_12447 = (device void *) workspace.allocate (gf_max((ulong) (size_3102 * 4), 1));
      xzu = (device GFReal *) D_12447;
      offset_3101 = -(lbound_3096) - lbound_3098 * stride_3100;
      lbound_3103 = (long) *its;
      ubound_3104 = (long) *ite;
      stride_3107 = (1 - lbound_3103) + ubound_3104;
      stride_3107 = gf_max(stride_3107, 0);
      lbound_3105 = (long) *kts;
      ubound_3106 = (long) *kte;
      size_3109 = ((1 - lbound_3105) + ubound_3106) * stride_3107;
      size_3109 = gf_max(size_3109, 0);
      D_12440 = size_3109 + -1;
      D_12441 = (ulong) (ulong) (size_3109) * 32;
      D_12442 = (ulong) (size_3109) * 4;
      D_12443 = (device void *) workspace.allocate (gf_max((ulong) (size_3109 * 4), 1));
      z = (device GFReal *) D_12443;
      offset_3108 = -(lbound_3103) - lbound_3105 * stride_3107;
      lbound_3114 = (long) *its;
      ubound_3115 = (long) *ite;
      stride_3118 = (1 - lbound_3114) + ubound_3115;
      stride_3118 = gf_max(stride_3118, 0);
      lbound_3116 = (long) *kts;
      ubound_3117 = (long) *kte;
      size_3120 = ((1 - lbound_3116) + ubound_3117) * stride_3118;
      size_3120 = gf_max(size_3120, 0);
      D_12436 = size_3120 + -1;
      D_12437 = (ulong) (ulong) (size_3120) * 32;
      D_12438 = (ulong) (size_3120) * 4;
      D_12439 = (device void *) workspace.allocate (gf_max((ulong) (size_3120 * 4), 1));
      z_cup = (device GFReal *) D_12439;
      offset_3119 = -(lbound_3114) - lbound_3116 * stride_3118;
      lbound_3121 = (long) *its;
      ubound_3122 = (long) *ite;
      stride_3125 = (1 - lbound_3121) + ubound_3122;
      stride_3125 = gf_max(stride_3125, 0);
      lbound_3123 = (long) *kts;
      ubound_3124 = (long) *kte;
      size_3127 = ((1 - lbound_3123) + ubound_3124) * stride_3125;
      size_3127 = gf_max(size_3127, 0);
      D_12432 = size_3127 + -1;
      D_12433 = (ulong) (ulong) (size_3127) * 32;
      D_12434 = (ulong) (size_3127) * 4;
      D_12435 = (device void *) workspace.allocate (gf_max((ulong) (size_3127 * 4), 1));
      zd = (device GFReal *) D_12435;
      offset_3126 = -(lbound_3121) - lbound_3123 * stride_3125;
      lbound_3128 = (long) *its;
      ubound_3129 = (long) *ite;
      stride_3132 = (1 - lbound_3128) + ubound_3129;
      stride_3132 = gf_max(stride_3132, 0);
      lbound_3130 = (long) *kts;
      ubound_3131 = (long) *kte;
      size_3134 = ((1 - lbound_3130) + ubound_3131) * stride_3132;
      size_3134 = gf_max(size_3134, 0);
      D_12428 = size_3134 + -1;
      D_12429 = (ulong) (ulong) (size_3134) * 32;
      D_12430 = (ulong) (size_3134) * 4;
      D_12431 = (device void *) workspace.allocate (gf_max((ulong) (size_3134 * 4), 1));
      zdo = (device GFReal *) D_12431;
      offset_3133 = -(lbound_3128) - lbound_3130 * stride_3132;
      lbound_3142 = (long) *its;
      ubound_3143 = (long) *ite;
      stride_3146 = (1 - lbound_3142) + ubound_3143;
      stride_3146 = gf_max(stride_3146, 0);
      lbound_3144 = (long) *kts;
      ubound_3145 = (long) *kte;
      size_3148 = ((1 - lbound_3144) + ubound_3145) * stride_3146;
      size_3148 = gf_max(size_3148, 0);
      D_12424 = size_3148 + -1;
      D_12425 = (ulong) (ulong) (size_3148) * 32;
      D_12426 = (ulong) (size_3148) * 4;
      D_12427 = (device void *) workspace.allocate (gf_max((ulong) (size_3148 * 4), 1));
      zo_cup = (device GFReal *) D_12427;
      offset_3147 = -(lbound_3142) - lbound_3144 * stride_3146;
      lbound_3157 = (long) *its;
      ubound_3158 = (long) *ite;
      stride_3161 = (1 - lbound_3157) + ubound_3158;
      stride_3161 = gf_max(stride_3161, 0);
      lbound_3159 = (long) *kts;
      ubound_3160 = (long) *kte;
      size_3163 = ((1 - lbound_3159) + ubound_3160) * stride_3161;
      size_3163 = gf_max(size_3163, 0);
      D_12420 = size_3163 + -1;
      D_12421 = (ulong) (ulong) (size_3163) * 32;
      D_12422 = (ulong) (size_3163) * 4;
      D_12423 = (device void *) workspace.allocate (gf_max((ulong) (size_3163 * 4), 1));
      zu = (device GFReal *) D_12423;
      offset_3162 = -(lbound_3157) - lbound_3159 * stride_3161;
      lbound_3164 = (long) *its;
      ubound_3165 = (long) *ite;
      size_3167 = (1 - lbound_3164) + ubound_3165;
      size_3167 = gf_max(size_3167, 0);
      D_12416 = size_3167 + -1;
      D_12417 = (ulong) (ulong) (size_3167) * 32;
      D_12418 = (ulong) (size_3167) * 4;
      D_12419 = (device void *) workspace.allocate (gf_max((ulong) (size_3167 * 4), 1));
      zuhe = (device GFReal *) D_12419;
      offset_3166 = -(lbound_3164);
      lbound_3168 = (long) *its;
      ubound_3169 = (long) *ite;
      stride_3172 = (1 - lbound_3168) + ubound_3169;
      stride_3172 = gf_max(stride_3172, 0);
      lbound_3170 = (long) *kts;
      ubound_3171 = (long) *kte;
      size_3174 = ((1 - lbound_3170) + ubound_3171) * stride_3172;
      size_3174 = gf_max(size_3174, 0);
      D_12412 = size_3174 + -1;
      D_12413 = (ulong) (ulong) (size_3174) * 32;
      D_12414 = (ulong) (size_3174) * 4;
      D_12415 = (device void *) workspace.allocate (gf_max((ulong) (size_3174 * 4), 1));
      zuo = (device GFReal *) D_12415;
      offset_3173 = -(lbound_3168) - lbound_3170 * stride_3172;
      levadj = 5;
      power_entr = 2.0e+0f;
      zustart = 1.00000001490116119384765625e-1f;
      zufinal = 1.0e+0f;
      day = 8.64e+4f;
      {
        int D_11817;
        int D_11818;

        D_11817 = *its;
        D_11818 = *itf;
        i = D_11817;
        while (1)
          {
            {
              bool D_11821;

              D_11821 = i > D_11818;
              if (D_11821) break;
              closure_n[(long) i + offset_2240] = 1.6e+1f;
              xland1[(long) i + offset_3034] = 1.0e+0f;
              if (xland[(long) i + offset_3030] > 1.5e+0f)
                {
                  xland1[(long) i + offset_3034] = 0.0f;
                }

              cap_max_increment[(long) i + offset_2218] = 2.5e+1f;
              gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "", 0);
              gf_memset ((device char *)&ierrc[(long) i + offset_2533], 32, 50);

              i = i + 1;
            }
          }

      }
      {
        int D_11823;
        int D_11824;

        D_11823 = *its;
        D_11824 = *ite;
        i = D_11823;
        while (1)
          {
            {
              bool D_11827;

              D_11827 = i > D_11824;
              if (D_11827) break;
              entr_rate[(long) i + offset_2407] = 7.00000018696300685405731201171875e-5f;
              radius[(long) i + offset_2798] = 2.0000000298023223876953125e-1f / entr_rate[(long) i + offset_2407];
              frh[(long) i + offset_2418] = ((((radius[(long) i + offset_2798] * radius[(long) i + offset_2798])) * 3.1400001049041748046875e+0f) / dxcell[(long) i + offset_2383]) / dxcell[(long) i + offset_2383];
              if (frh[(long) i + offset_2418] > 6.99999988079071044921875e-1f)
                {
                  frh[(long) i + offset_2418] = 6.99999988079071044921875e-1f;
                  radius[(long) i + offset_2798] = gf_sqrt (((frh[(long) i + offset_2418] * dxcell[(long) i + offset_2383]) * dxcell[(long) i + offset_2383]) / 3.1400001049041748046875e+0f);
                  entr_rate[(long) i + offset_2407] = 2.0000000298023223876953125e-1f / radius[(long) i + offset_2798];
                }

              {
                GFReal D_11829;
                GFReal D_11830;

                D_11829 = ((1.0e+0f - frh[(long) i + offset_2418]));
                D_11830 = D_11829 * D_11829;
                sig[(long) i + offset_2809] = D_11830;
              }
              mentrd_rate[(long) i + offset_2589] = entr_rate[(long) i + offset_2407];
              xlamdd[(long) i + offset_3026] = mentrd_rate[(long) i + offset_2589];

              i = i + 1;
            }
          }

      }
      {
        int D_11831;
        int D_11832;

        D_11831 = *kts;
        D_11832 = *ktf;
        k = D_11831;
        while (1)
          {
            {
              bool D_11835;

              D_11835 = k > D_11832;
              if (D_11835) break;
              {
                int D_11836;
                int D_11837;

                D_11836 = *its;
                D_11837 = *itf;
                i = D_11836;
                while (1)
                  {
                    {
                      bool D_11840;

                      D_11840 = i > D_11837;
                      if (D_11840) break;
                      z[((long) k * stride_3107 + offset_3108) + (long) i] = zo[((long) k * stride_3139 + offset_3140) + (long) i];
                      xz[((long) k * stride_3086 + offset_3087) + (long) i] = zo[((long) k * stride_3139 + offset_3140) + (long) i];
                      cupclw[((long) k * stride_2253 + offset_2254) + (long) i] = 0.0f;
                      cd[((long) k * stride_2228 + offset_2229) + (long) i] = (entr_rate[(long) i + offset_2407]);
                      cdd[((long) k * stride_2235 + offset_2236) + (long) i] = xlamdd[(long) i + offset_3026];
                      hcdo[((long) k * stride_2445 + offset_2446) + (long) i] = 0.0f;
                      qrcdo[((long) k * stride_2786 + offset_2787) + (long) i] = 0.0f;
                      dellaqc[((long) k * stride_2332 + offset_2333) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      edtmax = 1.0e+0f;
      edtmin = 1.00000001490116119384765625e-1f;
      depth_min = 1.0e+3f;
      cap_maxs = 7.5e+1f;
      {
        int D_11841;
        int D_11842;

        D_11841 = *its;
        D_11842 = *itf;
        i = D_11841;
        while (1)
          {
            {
              bool D_11845;

              D_11845 = i > D_11842;
              if (D_11845) break;
              kbmax[(long) i + offset_2557] = 1;
              aa0[(long) i + offset_2132] = 0.0f;
              aa1[(long) i + offset_2136] = 0.0f;
              edt[(long) i + offset_2387] = 0.0f;
              kstabm[(long) i + offset_2569] = *ktf + -1;
              ierr[(long) i + offset_2521] = 0;
              ierr2[(long) i + offset_2525] = 0;
              ierr3[(long) i + offset_2529] = 0;

              i = i + 1;
            }
          }

      }
      {
        int D_11846;
        int D_11847;

        D_11846 = *its;
        D_11847 = *itf;
        i = D_11846;
        while (1)
          {
            {
              bool D_11850;

              D_11850 = i > D_11847;
              if (D_11850) break;
              cap_max[(long) i + offset_2214] = cap_maxs;
              iresult = 0;

              i = i + 1;
            }
          }

      }
      zkbmax = 4.0e+3f;
      zcutdown = 3.0e+3f;
      z_detr = 1.25e+3f;
      {
        int D_11851;

        D_11851 = *maxens;
        nens = 1;
        while (1)
          {
            {
              bool D_11854;

              D_11854 = nens > D_11851;
              if (D_11854) break;
              mbdt_ens[(long) nens + -1] = ((((GFReal) nens - 3.0e+0f)) * *dtime) * 1.000000047497451305389404296875e-3f + *dtime * 4.999999888241291046142578125e-3f;

              nens = nens + 1;
            }
          }

      }
      {
        int D_11855;

        D_11855 = *maxens2;
        nens = 1;
        while (1)
          {
            {
              bool D_11858;

              D_11858 = nens > D_11855;
              if (D_11858) break;
              edt_ens[(long) nens + -1] = 9.49999988079071044921875e-1f - (GFReal) nens * 9.99999977648258209228515625e-3f;

              nens = nens + 1;
            }
          }

      }
      {
        int D_11859;
        int D_11860;

        D_11859 = *its;
        D_11860 = *itf;
        i = D_11859;
        while (1)
          {
            {
              bool D_11863;

              D_11863 = i > D_11860;
              if (D_11863) break;
              if (ierr[(long) i + offset_2521] != 20)
                {
                  {
                    int D_11865;

                    D_11865 = (*maxens * *maxens2) * *maxens3;
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_11868;

                          D_11868 = k > D_11865;
                          if (D_11868) break;
                          xf_ens[(((long) ((((*iens + -1) * *maxens) * *maxens2) * *maxens3 + k) * stride_2982 + offset_2983) + (long) *j * stride_2980) + (long) i] = 0.0f;
                          pr_ens[(((long) ((((*iens + -1) * *maxens) * *maxens2) * *maxens3 + k) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = 0.0f;

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int C_11869 = -1;

        cup_env (workspace, z, qes, he, hes, t, q, p, z1, psur, ierr, tcrit, &C_11869, xl, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      {
        int C_11870 = -1;

        cup_env (workspace, zo, qeso, heo, heso, tn, qo, po, z1, psur, ierr, tcrit, &C_11870, xl, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      cup_env_clev (workspace, t, qes, q, he, hes, z, p, qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup, psur, ierr, z1, xl, rv, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      cup_env_clev (workspace, tn, qeso, qo, heo, heso, zo, po, qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup, psur, ierr, z1, xl, rv, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_11871;
        int D_11872;

        D_11871 = *its;
        D_11872 = *itf;
        i = D_11871;
        while (1)
          {
            {
              bool D_11875;

              D_11875 = i > D_11872;
              if (D_11875) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  if (aaeq[(long) i + offset_2140] < -1.00000001490116119384765625e-1f)
                    {
                      ierr[(long) i + offset_2521] = 20;
                    }

                  {
                    int D_11878;
                    int D_11879;

                    D_11878 = *kts;
                    D_11879 = *ktf;
                    k = D_11878;
                    while (1)
                      {
                        {
                          bool D_11882;

                          D_11882 = k > D_11879;
                          if (D_11882) break;
                          if (zo_cup[((long) k * stride_3146 + offset_3147) + (long) i] > z1[(long) i + offset_3112] + zkbmax)
                            {
                              kbmax[(long) i + offset_2557] = k;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }

                  {
                    int D_11885;
                    int D_11886;

                    D_11885 = *kts;
                    D_11886 = *ktf;
                    k = D_11885;
                    while (1)
                      {
                        {
                          bool D_11889;

                          D_11889 = k > D_11886;
                          if (D_11889) break;
                          if (zo_cup[((long) k * stride_3146 + offset_3147) + (long) i] > z1[(long) i + offset_3112] + z_detr)
                            {
                              kdet[(long) i + offset_2561] = k;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }

                }


              i = i + 1;
            }
          }

      }
      {
        int C_11892 = 3;

        cup_maximi (workspace, heo_cup, &C_11892, kbmax, k22, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      }
      {
        int D_11893;
        int D_11894;

        D_11893 = *its;
        D_11894 = *itf;
        i = D_11893;
        while (1)
          {
            {
              bool D_11897;

              D_11897 = i > D_11894;
              if (D_11897) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  frh[(long) i + offset_2418] = q_cup[((long) k22[(long) i + offset_2541] * stride_2723 + offset_2724) + (long) i] / qes_cup[((long) k22[(long) i + offset_2541] * stride_2751 + offset_2752) + (long) i];
                  if ((omeg[((offset_2605 + (stride_2604)) + (long) k22[(long) i + offset_2541] * stride_2602) + (long) i] < 0.0f && frh[(long) i + offset_2418] >= 9.900000095367431640625e-1f) && sig[(long) i + offset_2809] < 9.0999998152256011962890625e-2f)
                    {
                      ierr[(long) i + offset_2521] = 1200;
                    }

                  if (k22[(long) i + offset_2541] >= kbmax[(long) i + offset_2557])
                    {
                      ierr[(long) i + offset_2521] = 2;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "could not find k22", 18);
                      gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 18, 32, 32);
                    }

                }



              i = i + 1;
            }
          }

      }
      {
        int D_11902;
        int D_11903;

        D_11902 = *its;
        D_11903 = *itf;
        i = D_11902;
        while (1)
          {
            {
              bool D_11906;

              D_11906 = i > D_11903;
              if (D_11906) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  if (*use_excess == 2)
                    {
                      {
                        int M_3175;
                        int D_11910;

                        M_3175 = 1;
                        D_11910 = k22[(long) i + offset_2541] + -1;
                        M_3175 = gf_max(D_11910, M_3175);
                        k1 = M_3175;
                      }
                      k2 = k22[(long) i + offset_2541] + 1;
                      hkb[(long) i + offset_2513] = he_cup[((long) k22[(long) i + offset_2541] * stride_2466 + offset_2467) + (long) i];
                      {
                        GFReal val_3176;
                        long D_11912;
                        long D_11913;
                        long D_11914;
                        long D_11915;

                        val_3176 = 0.0f;
                        D_11912 = offset_2481;
                        D_11913 = (long) k1;
                        D_11914 = (long) k2;
                        D_11915 = (long) i;
                        {
                          long S_3177;
                          long D_11917;
                          long D_11918;

                          D_11917 = stride_2480;
                          D_11918 = D_11912 + D_11915;
                          S_3177 = D_11913;
                          while (1)
                            {
                              if (S_3177 > D_11914) break;
                              val_3176 = heo_cup[S_3177 * D_11917 + D_11918] + val_3176;
                              S_3177 = S_3177 + 1;
                            }

                        }
                        hkbo[(long) i + offset_2517] = (val_3176 / (GFReal) ((k2 - k1) + 1) + *xl * zqexec[(long) i + offset_3151]) + *cp * ztexec[(long) i + offset_3155];
                      }
                    }
                  else
                    {
                      if (*use_excess <= 1)
                        {
                          hkb[(long) i + offset_2513] = he_cup[((long) k22[(long) i + offset_2541] * stride_2466 + offset_2467) + (long) i];
                          hkbo[(long) i + offset_2517] = heo_cup[((long) k22[(long) i + offset_2541] * stride_2480 + offset_2481) + (long) i] + (GFReal) *use_excess * ((*xl * zqexec[(long) i + offset_3151] + *cp * ztexec[(long) i + offset_3155]));
                        }

                    }

                }


              i = i + 1;
            }
          }

      }
      {
        int C_11921 = 1;

        cup_kbcon (workspace, ierrc, cap_max_increment, &C_11921, k22, kbcon, heo_cup, heso_cup, hkbo, ierr, kbmax, po_cup, cap_max, xl, cp, ztexec, zqexec, use_excess, itf, jtf, ktf, its, ite, jts, jte, kts, kte, 50);
      }
      cup_minimi (workspace, heso_cup, kbcon, kstabm, kstabi, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_11922;
        int D_11923;

        D_11922 = *its;
        D_11923 = *itf;
        i = D_11922;
        while (1)
          {
            {
              bool D_11926;

              D_11926 = i > D_11923;
              if (D_11926) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  {
                    int D_11928;
                    int D_11929;

                    D_11928 = k22[(long) i + offset_2541];
                    D_11929 = kbcon[(long) i + offset_2549];
                    k = D_11928;
                    while (1)
                      {
                        {
                          bool D_11932;

                          D_11932 = k > D_11929;
                          if (D_11932) break;
                          frh[(long) i + offset_2418] = q_cup[((long) k * stride_2723 + offset_2724) + (long) i] / qes_cup[((long) k * stride_2751 + offset_2752) + (long) i];
                          if ((omeg[((offset_2605 + (stride_2604)) + (long) k * stride_2602) + (long) i] < -9.999999974752427078783512115478515625e-7f && frh[(long) i + offset_2418] >= 9.900000095367431640625e-1f) && sig[(long) i + offset_2809] < 9.0999998152256011962890625e-2f)
                            {
                              ierr[(long) i + offset_2521] = 1200;
                            }


                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_11934;
        int D_11935;

        D_11934 = *its;
        D_11935 = *itf;
        i = D_11934;
        while (1)
          {
            {
              bool D_11938;

              D_11938 = i > D_11935;
              if (D_11938) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  {
                    int D_11940;
                    int D_11941;

                    D_11940 = *kts;
                    D_11941 = *ktf;
                    k = D_11940;
                    while (1)
                      {
                        {
                          bool D_11944;

                          D_11944 = k > D_11941;
                          if (D_11944) break;
                          {
                            GFReal D_11945;
                            GFReal M_3178;

                            D_11945 = qo_cup[((long) k * stride_2779 + offset_2780) + (long) i] / qeso_cup[((long) k * stride_2765 + offset_2766) + (long) i];
                            M_3178 = D_11945;
                            M_3178 = gf_min(M_3178, 1.0e+0f);
                            frh[(long) i + offset_2418] = M_3178;
                          }
                          entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] = entr_rate[(long) i + offset_2407] * ((1.2999999523162841796875e+0f - frh[(long) i + offset_2418]));

                          k = k + 1;
                        }
                      }

                  }
                  zuhe[(long) i + offset_3166] = zustart;
                  kstart = 1;
                  frh[(long) i + offset_2418] = ((zufinal - zustart)) / (((((GFReal) (kbcon[(long) i + offset_2549] * kbcon[(long) i + offset_2549]))) - (((GFReal) (kstart * kstart)))));
                  dh = zuhe[(long) i + offset_3166] - frh[(long) i + offset_2418] * (((GFReal) (kstart * kstart)));
                  {
                    int D_11947;
                    int D_11948;

                    D_11947 = kstart;
                    D_11948 = kbcon[(long) i + offset_2549] + -1;
                    k = D_11947;
                    while (1)
                      {
                        {
                          bool D_11951;

                          D_11951 = k > D_11948;
                          if (D_11951) break;
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          if (p_cup[((long) k * stride_2639 + offset_2640) + (long) i] > p_cup[((long) kstabi[(long) i + offset_2565] * stride_2639 + offset_2640) + (long) i])
                            {
                              cd[((long) k * stride_2228 + offset_2229) + (long) i] = 9.999999974752427078783512115478515625e-7f;
                            }

                          entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] = (((((frh[(long) i + offset_2418] * (((GFReal) ((k + 1) * (k + 1)))) + dh)) / zuhe[(long) i + offset_3166] - 1.0e+0f) + cd[((long) k * stride_2228 + offset_2229) + (long) i] * dz)) / dz;
                          zuhe[(long) i + offset_3166] = (zuhe[(long) i + offset_3166] + (entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] * dz) * zuhe[(long) i + offset_3166]) - (cd[((long) k * stride_2228 + offset_2229) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                          k = k + 1;
                        }
                      }

                  }
                  kbegzu = kstabi[(long) i + offset_2565] + 4;
                  {
                    int M_3179;
                    int D_11954;

                    M_3179 = kbegzu;
                    D_11954 = *ktf + -1;
                    M_3179 = gf_min(D_11954, M_3179);
                    kbegzu = M_3179;
                  }
                  kfinalzu = kbegzu + 1;
                  {
                    int D_11955;
                    int D_11956;

                    D_11955 = *kts;
                    D_11956 = *ktf;
                    k = D_11955;
                    while (1)
                      {
                        {
                          bool D_11959;

                          D_11959 = k > D_11956;
                          if (D_11959) break;
                          cd[((long) k * stride_2228 + offset_2229) + (long) i] = entr_rate_2d[((long) kbcon[(long) i + offset_2549] * stride_2413 + offset_2414) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_11960;
                    int D_11961;

                    D_11960 = kbcon[(long) i + offset_2549];
                    D_11961 = kbegzu;
                    k = D_11960;
                    while (1)
                      {
                        {
                          bool D_11964;

                          D_11964 = k > D_11961;
                          if (D_11964) break;
                          cd[((long) k * stride_2228 + offset_2229) + (long) i] = entr_rate_2d[((long) kbcon[(long) i + offset_2549] * stride_2413 + offset_2414) + (long) i];
                          if (p_cup[((long) k * stride_2639 + offset_2640) + (long) i] > p_cup[((long) kstabi[(long) i + offset_2565] * stride_2639 + offset_2640) + (long) i])
                            {
                              cd[((long) k * stride_2228 + offset_2229) + (long) i] = 9.999999974752427078783512115478515625e-7f;
                            }

                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          zuhe[(long) i + offset_3166] = (zuhe[(long) i + offset_3166] + (entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] * dz) * zuhe[(long) i + offset_3166]) - (cd[((long) k * stride_2228 + offset_2229) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_11966;
                    int D_11967;

                    D_11966 = kstabi[(long) i + offset_2565];
                    D_11967 = *ktf + -2;
                    k = D_11966;
                    while (1)
                      {
                        {
                          bool D_11970;

                          D_11970 = k > D_11967;
                          if (D_11970) break;
                          if (((hkb[(long) i + offset_2513] - hes_cup[((long) k * stride_2494 + offset_2495) + (long) i])) < 0.0f)
                            {
                              kfinalzu = k + -3;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }

                  {
                    int M_3180;
                    int D_11974;

                    M_3180 = kfinalzu;
                    D_11974 = kbegzu + 1;
                    M_3180 = gf_max(D_11974, M_3180);
                    kfinalzu = M_3180;
                  }
                  {
                    int M_3181;
                    int D_11976;

                    M_3181 = kfinalzu;
                    D_11976 = *ktf + -1;
                    M_3181 = gf_min(D_11976, M_3181);
                    kfinalzu = M_3181;
                  }
                  frh[(long) i + offset_2418] = -(((2.0000000298023223876953125e-1f - zuhe[(long) i + offset_3166])) / (((((GFReal) (kfinalzu * kfinalzu))) - (((GFReal) (kbegzu * kbegzu))))));
                  dh = zuhe[(long) i + offset_3166] + frh[(long) i + offset_2418] * (((GFReal) (kbegzu * kbegzu)));
                  {
                    int D_11977;
                    int D_11978;

                    D_11977 = kbegzu + 1;
                    D_11978 = kfinalzu;
                    k = D_11977;
                    while (1)
                      {
                        {
                          bool D_11981;

                          D_11981 = k > D_11978;
                          if (D_11981) break;
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          cd[((long) k * stride_2228 + offset_2229) + (long) i] = -((((((dh - frh[(long) i + offset_2418] * (((GFReal) ((k + 1) * (k + 1)))))) / zuhe[(long) i + offset_3166] - 1.0e+0f) - entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] * dz)) / dz);
                          zuhe[(long) i + offset_3166] = (zuhe[(long) i + offset_3166] + (entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] * dz) * zuhe[(long) i + offset_3166]) - (cd[((long) k * stride_2228 + offset_2229) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_11982;
                    int D_11983;

                    D_11982 = kfinalzu + 1;
                    D_11983 = *ktf;
                    k = D_11982;
                    while (1)
                      {
                        {
                          bool D_11986;

                          D_11986 = k > D_11983;
                          if (D_11986) break;
                          cd[((long) k * stride_2228 + offset_2229) + (long) i] = entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_11987;
                    int D_11988;

                    D_11987 = *kts + 1;
                    D_11988 = *ktf + -2;
                    k = D_11987;
                    while (1)
                      {
                        {
                          bool D_11991;

                          D_11991 = k > D_11988;
                          if (D_11991) break;
                          dzm1 = z_cup[((long) k * stride_3118 + offset_3119) + (long) i] - z_cup[((long) (k + -1) * stride_3118 + offset_3119) + (long) i];
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          dzp1 = z_cup[((long) (k + 2) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i];
                          smth[(long) k + offset_2813] = (((cd[((long) (k + -1) * stride_2228 + offset_2229) + (long) i] * dzm1 + (dz * 2.0e+0f) * cd[((long) k * stride_2228 + offset_2229) + (long) i]) + cd[((long) (k + 1) * stride_2228 + offset_2229) + (long) i] * dzp1)) * 2.5e-1f;

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_11992;
                    int D_11993;

                    D_11992 = *kts + 1;
                    D_11993 = *ktf + -2;
                    k = D_11992;
                    while (1)
                      {
                        {
                          bool D_11996;

                          D_11996 = k > D_11993;
                          if (D_11996) break;
                          dzm1 = z_cup[((long) k * stride_3118 + offset_3119) + (long) i] - z_cup[((long) (k + -1) * stride_3118 + offset_3119) + (long) i];
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          dzp1 = z_cup[((long) (k + 2) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i];
                          cd[((long) k * stride_2228 + offset_2229) + (long) i] = smth[(long) k + offset_2813] / dz;

                          k = k + 1;
                        }
                      }

                  }
                  {
                    long D_11997;
                    long D_11998;
                    long D_11999;

                    D_11997 = offset_2813;
                    D_11998 = lbound_2811;
                    D_11999 = ubound_2812;
                    {
                      long S_3182;

                      S_3182 = D_11998;
                      while (1)
                        {
                          if (S_3182 > D_11999) break;
                          smth[S_3182 + D_11997] = 0.0f;
                          S_3182 = S_3182 + 1;
                        }

                    }
                  }
                  {
                    int D_12002;

                    D_12002 = *ktf + -2;
                    k = 2;
                    while (1)
                      {
                        {
                          bool D_12005;

                          D_12005 = k > D_12002;
                          if (D_12005) break;
                          dzm1 = z_cup[((long) k * stride_3118 + offset_3119) + (long) i] - z_cup[((long) (k + -1) * stride_3118 + offset_3119) + (long) i];
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          dzp1 = z_cup[((long) (k + 2) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i];
                          smth[(long) k + offset_2813] = (((entr_rate_2d[((long) (k + -1) * stride_2413 + offset_2414) + (long) i] * dzm1 + (dz * 2.0e+0f) * entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i]) + entr_rate_2d[((long) (k + 1) * stride_2413 + offset_2414) + (long) i] * dzp1)) * 2.5e-1f;

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_12006;

                    D_12006 = *ktf + -2;
                    k = 2;
                    while (1)
                      {
                        {
                          bool D_12009;

                          D_12009 = k > D_12006;
                          if (D_12009) break;
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] = smth[(long) k + offset_2813] / dz;

                          k = k + 1;
                        }
                      }

                  }
                  zuhe[(long) i + offset_3166] = zustart;
                  {
                    int D_12010;

                    D_12010 = kbegzu;
                    k = 2;
                    while (1)
                      {
                        {
                          bool D_12013;

                          D_12013 = k > D_12010;
                          if (D_12013) break;
                          dz = z_cup[((long) (k + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) k * stride_3118 + offset_3119) + (long) i];
                          frh[(long) i + offset_2418] = zuhe[(long) i + offset_3166];
                          zuhe[(long) i + offset_3166] = (zuhe[(long) i + offset_3166] + (entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] * dz) * zuhe[(long) i + offset_3166]) - (cd[((long) k * stride_2228 + offset_2229) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_12014;
        int D_12015;

        D_12014 = *kts;
        D_12015 = *ktf;
        k = D_12014;
        while (1)
          {
            {
              bool D_12018;

              D_12018 = k > D_12015;
              if (D_12018) break;
              {
                int D_12019;
                int D_12020;

                D_12019 = *its;
                D_12020 = *itf;
                i = D_12019;
                while (1)
                  {
                    {
                      bool D_12023;

                      D_12023 = i > D_12020;
                      if (D_12023) break;
                      hc[((long) k * stride_2438 + offset_2439) + (long) i] = 0.0f;
                      dby[((long) k * stride_2260 + offset_2261) + (long) i] = 0.0f;
                      hco[((long) k * stride_2452 + offset_2453) + (long) i] = 0.0f;
                      dbyo[((long) k * stride_2274 + offset_2275) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_12024;
        int D_12025;

        D_12024 = *its;
        D_12025 = *itf;
        i = D_12024;
        while (1)
          {
            {
              bool D_12028;

              D_12028 = i > D_12025;
              if (D_12028) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  {
                    int D_12030;

                    D_12030 = kbcon[(long) i + offset_2549] + -1;
                    k = 1;
                    while (1)
                      {
                        {
                          bool D_12033;

                          D_12033 = k > D_12030;
                          if (D_12033) break;
                          hc[((long) k * stride_2438 + offset_2439) + (long) i] = hkb[(long) i + offset_2513];
                          hco[((long) k * stride_2452 + offset_2453) + (long) i] = hkbo[(long) i + offset_2517];

                          k = k + 1;
                        }
                      }

                  }
                  k = kbcon[(long) i + offset_2549];
                  hc[((long) k * stride_2438 + offset_2439) + (long) i] = hkb[(long) i + offset_2513];
                  dby[((long) kbcon[(long) i + offset_2549] * stride_2260 + offset_2261) + (long) i] = hkb[(long) i + offset_2513] - hes_cup[((long) k * stride_2494 + offset_2495) + (long) i];
                  hco[((long) k * stride_2452 + offset_2453) + (long) i] = hkbo[(long) i + offset_2517];
                  dbyo[((long) kbcon[(long) i + offset_2549] * stride_2274 + offset_2275) + (long) i] = hkbo[(long) i + offset_2517] - heso_cup[((long) k * stride_2508 + offset_2509) + (long) i];
                }


              i = i + 1;
            }
          }

      }
      {
        int D_12034;
        int D_12035;

        D_12034 = *its;
        D_12035 = *itf;
        i = D_12034;
        while (1)
          {
            {
              bool D_12038;

              D_12038 = i > D_12035;
              if (D_12038) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  zu[(offset_3162 + (stride_3161)) + (long) i] = zustart;
                  zuo[(offset_3173 + (stride_3172)) + (long) i] = zustart;
                  {
                    int D_12040;

                    D_12040 = *ktf + -1;
                    k = 2;
                    while (1)
                      {
                        {
                          bool D_12043;

                          D_12043 = k > D_12040;
                          if (D_12043) break;
                          dz = zo_cup[((long) k * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) (k + -1) * stride_3146 + offset_3147) + (long) i];
                          up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i] = (entr_rate_2d[((long) (k + -1) * stride_2413 + offset_2414) + (long) i] * dz) * zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i];
                          up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] = (cd[((long) (k + -1) * stride_2228 + offset_2229) + (long) i] * dz) * zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i];
                          zuo[((long) k * stride_3172 + offset_3173) + (long) i] = (zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i] + up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i]) - up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i];
                          if (zuo[((long) k * stride_3172 + offset_3173) + (long) i] < 5.00000007450580596923828125e-2f)
                            {
                              zuo[((long) k * stride_3172 + offset_3173) + (long) i] = 5.00000007450580596923828125e-2f;
                              up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] = (zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i] - 5.00000007450580596923828125e-2f) + up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i];
                              cd[((long) (k + -1) * stride_2228 + offset_2229) + (long) i] = (up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] / dz) / zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i];
                            }

                          zu[((long) k * stride_3161 + offset_3162) + (long) i] = zuo[((long) k * stride_3172 + offset_3173) + (long) i];
                          up_massentr[((long) (k + -1) * stride_2900 + offset_2901) + (long) i] = up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i];
                          up_massdetr[((long) (k + -1) * stride_2886 + offset_2887) + (long) i] = up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_12045;
                    int D_12046;

                    D_12045 = kbcon[(long) i + offset_2549] + 1;
                    D_12046 = *ktf + -1;
                    k = D_12045;
                    while (1)
                      {
                        {
                          bool D_12049;

                          D_12049 = k > D_12046;
                          if (D_12049) break;
                          hc[((long) k * stride_2438 + offset_2439) + (long) i] = (((hc[((long) (k + -1) * stride_2438 + offset_2439) + (long) i] * zu[((long) (k + -1) * stride_3161 + offset_3162) + (long) i] - (up_massdetr[((long) (k + -1) * stride_2886 + offset_2887) + (long) i] * 5.0e-1f) * hc[((long) (k + -1) * stride_2438 + offset_2439) + (long) i]) + up_massentr[((long) (k + -1) * stride_2900 + offset_2901) + (long) i] * he[((long) (k + -1) * stride_2459 + offset_2460) + (long) i])) / (((zu[((long) (k + -1) * stride_3161 + offset_3162) + (long) i] - up_massdetr[((long) (k + -1) * stride_2886 + offset_2887) + (long) i] * 5.0e-1f) + up_massentr[((long) (k + -1) * stride_2900 + offset_2901) + (long) i]));
                          dby[((long) k * stride_2260 + offset_2261) + (long) i] = hc[((long) k * stride_2438 + offset_2439) + (long) i] - hes_cup[((long) k * stride_2494 + offset_2495) + (long) i];
                          hco[((long) k * stride_2452 + offset_2453) + (long) i] = (((hco[((long) (k + -1) * stride_2452 + offset_2453) + (long) i] * zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i] - (up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] * 5.0e-1f) * hco[((long) (k + -1) * stride_2452 + offset_2453) + (long) i]) + up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i] * heo[((long) (k + -1) * stride_2473 + offset_2474) + (long) i])) / (((zuo[((long) (k + -1) * stride_3172 + offset_3173) + (long) i] - up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] * 5.0e-1f) + up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i]));
                          dbyo[((long) k * stride_2274 + offset_2275) + (long) i] = hco[((long) k * stride_2452 + offset_2453) + (long) i] - heso_cup[((long) k * stride_2508 + offset_2509) + (long) i];

                          k = k + 1;
                        }
                      }

                  }
                  {
                    int D_12050;
                    int D_12051;

                    D_12050 = kbcon[(long) i + offset_2549] + 1;
                    D_12051 = *ktf;
                    k = D_12050;
                    while (1)
                      {
                        {
                          bool D_12054;

                          D_12054 = k > D_12051;
                          if (D_12054) break;
                          if (dbyo[((long) k * stride_2274 + offset_2275) + (long) i] < 0.0f)
                            {
                              ktop[(long) i + offset_2573] = k + -1;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }

                  if (kbcon[(long) i + offset_2549] + 1 >= ktop[(long) i + offset_2573])
                    {
                      ierr[(long) i + offset_2521] = 5;
                    }

                  {
                    int D_12058;
                    int D_12059;

                    D_12058 = ktop[(long) i + offset_2573] + 1;
                    D_12059 = *ktf;
                    k = D_12058;
                    while (1)
                      {
                        {
                          bool D_12062;

                          D_12062 = k > D_12059;
                          if (D_12062) break;
                          hc[((long) k * stride_2438 + offset_2439) + (long) i] = hes_cup[((long) k * stride_2494 + offset_2495) + (long) i];
                          hco[((long) k * stride_2452 + offset_2453) + (long) i] = heso_cup[((long) k * stride_2508 + offset_2509) + (long) i];
                          dby[((long) k * stride_2260 + offset_2261) + (long) i] = 0.0f;
                          dbyo[((long) k * stride_2274 + offset_2275) + (long) i] = 0.0f;
                          zu[((long) k * stride_3161 + offset_3162) + (long) i] = 0.0f;
                          zuo[((long) k * stride_3172 + offset_3173) + (long) i] = 0.0f;
                          cd[((long) k * stride_2228 + offset_2229) + (long) i] = 0.0f;
                          entr_rate_2d[((long) k * stride_2413 + offset_2414) + (long) i] = 0.0f;
                          up_massentr[((long) k * stride_2900 + offset_2901) + (long) i] = 0.0f;
                          up_massdetr[((long) k * stride_2886 + offset_2887) + (long) i] = 0.0f;
                          up_massentro[((long) k * stride_2907 + offset_2908) + (long) i] = 0.0f;
                          up_massdetro[((long) k * stride_2893 + offset_2894) + (long) i] = 0.0f;

                          k = k + 1;
                        }
                      }

                  }
                }


              i = i + 1;
            }
          }

      }
      {
        int D_12063;
        int D_12064;

        D_12063 = *its;
        D_12064 = *itf;
        i = D_12063;
        while (1)
          {
            {
              bool D_12067;

              D_12067 = i > D_12064;
              if (D_12067) break;
              kzdown[(long) i + offset_2577] = 0;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  zktop = ((zo_cup[((long) ktop[(long) i + offset_2573] * stride_3146 + offset_3147) + (long) i] - z1[(long) i + offset_3112])) * 6.0000002384185791015625e-1f;
                  {
                    GFReal D_12069;
                    GFReal M_3183;
                    GFReal D_12071;

                    D_12069 = z1[(long) i + offset_3112] + zktop;
                    M_3183 = D_12069;
                    D_12071 = z1[(long) i + offset_3112] + zcutdown;
                    M_3183 = gf_min(D_12071, M_3183);
                    zktop = M_3183;
                  }
                  {
                    int D_12072;
                    int D_12073;

                    D_12072 = *kts;
                    D_12073 = *ktf;
                    k = D_12072;
                    while (1)
                      {
                        {
                          bool D_12076;

                          D_12076 = k > D_12073;
                          if (D_12076) break;
                          if (zo_cup[((long) k * stride_3146 + offset_3147) + (long) i] > zktop)
                            {
                              kzdown[(long) i + offset_2577] = k;
                              break;
                            }


                          k = k + 1;
                        }
                      }

                  }
                }



              i = i + 1;
            }
          }

      }
      cup_minimi (workspace, heso_cup, k22, kzdown, jmin, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_12079;
        int D_12080;

        D_12079 = *its;
        D_12080 = *itf;
        i = D_12079;
        while (1)
          {
            {
              bool D_12083;

              D_12083 = i > D_12080;
              if (D_12083) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  jmini = jmin[(long) i + offset_2537];
                  keep_going = 1;
                  while (1)
                    {
                      if (!keep_going) break;

                      keep_going = 0;
                      if (kdet[(long) i + offset_2561] >= (jmini))
                        {
                          kdet[(long) i + offset_2561] = jmini + -1;
                        }

                      if (ktop[(long) i + offset_2573] + -1 <= jmini)
                        {
                          jmini = ktop[(long) i + offset_2573] + -2;
                        }

                      ki = jmini;
                      hcdo[((long) ki * stride_2445 + offset_2446) + (long) i] = heso_cup[((long) ki * stride_2508 + offset_2509) + (long) i];
                      dz = zo_cup[((long) (ki + 1) * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) ki * stride_3146 + offset_3147) + (long) i];
                      dh = 0.0f;
                      {
                        int D_12090;

                        D_12090 = ki + -1;
                        k = D_12090;
                        while (1)
                          {
                            {
                              bool D_12093;

                              D_12093 = k <= 0;
                              if (D_12093) break;
                              hcdo[((long) k * stride_2445 + offset_2446) + (long) i] = heso_cup[((long) jmini * stride_2508 + offset_2509) + (long) i];
                              dz = zo_cup[((long) (k + 1) * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) k * stride_3146 + offset_3147) + (long) i];
                              dh = ((hcdo[((long) k * stride_2445 + offset_2446) + (long) i] - heso_cup[((long) k * stride_2508 + offset_2509) + (long) i])) * dz + dh;
                              if (dh > 0.0f)
                                {
                                  jmini = jmini + -1;
                                  if (jmini > 5)
                                    {
                                      keep_going = 1;
                                    }
                                  else
                                    {
                                      ierr[(long) i + offset_2521] = 9;
                                      gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "could not find jmini9", 21);
                                      gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 21, 32, 29);
                                      break;
                                    }

                                }


                              k = k + -1;
                            }
                          }

                      }

                    }

                  jmin[(long) i + offset_2537] = jmini;
                  if (jmini <= 5)
                    {
                      ierr[(long) i + offset_2521] = 4;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "could not find jmini4", 21);
                      gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 21, 32, 29);
                    }

                }



              i = i + 1;
            }
          }

      }
      {
        int D_12098;
        int D_12099;

        D_12098 = *its;
        D_12099 = *itf;
        i = D_12098;
        while (1)
          {
            {
              bool D_12102;

              D_12102 = i > D_12099;
              if (D_12102) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  if (zo_cup[((long) ktop[(long) i + offset_2573] * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) kbcon[(long) i + offset_2549] * stride_3146 + offset_3147) + (long) i] < depth_min)
                    {
                      ierr[(long) i + offset_2521] = 6;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "cloud depth very shallow", 24);
                      gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 24, 32, 26);
                    }

                }


              i = i + 1;
            }
          }

      }
      {
        int D_12105;
        int D_12106;

        D_12105 = *kts;
        D_12106 = *ktf;
        k = D_12105;
        while (1)
          {
            {
              bool D_12109;

              D_12109 = k > D_12106;
              if (D_12109) break;
              {
                int D_12110;
                int D_12111;

                D_12110 = *its;
                D_12111 = *itf;
                i = D_12110;
                while (1)
                  {
                    {
                      bool D_12114;

                      D_12114 = i > D_12111;
                      if (D_12114) break;
                      zd[((long) k * stride_3125 + offset_3126) + (long) i] = 0.0f;
                      zdo[((long) k * stride_3132 + offset_3133) + (long) i] = 0.0f;
                      cdd[((long) k * stride_2235 + offset_2236) + (long) i] = 0.0f;
                      dd_massentr[((long) k * stride_2295 + offset_2296) + (long) i] = 0.0f;
                      dd_massdetr[((long) k * stride_2281 + offset_2282) + (long) i] = 0.0f;
                      dd_massentro[((long) k * stride_2302 + offset_2303) + (long) i] = 0.0f;
                      dd_massdetro[((long) k * stride_2288 + offset_2289) + (long) i] = 0.0f;
                      hcdo[((long) k * stride_2445 + offset_2446) + (long) i] = heso_cup[((long) k * stride_2508 + offset_2509) + (long) i];
                      dbydo[((long) k * stride_2267 + offset_2268) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      {
        int D_12115;
        int D_12116;

        D_12115 = *its;
        D_12116 = *itf;
        i = D_12115;
        while (1)
          {
            {
              bool D_12119;

              D_12119 = i > D_12116;
              if (D_12119) break;
              bud[(long) i + offset_2210] = 0.0f;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  {
                    long D_12121;
                    long D_12122;
                    long D_12123;
                    GFReal D_12124;
                    long D_12125;

                    D_12121 = offset_2596;
                    D_12122 = lbound_2593;
                    D_12123 = ubound_2594;
                    D_12124 = mentrd_rate[(long) i + offset_2589];
                    D_12125 = (long) i;
                    {
                      long S_3184;
                      long D_12127;
                      long D_12128;

                      D_12127 = stride_2595;
                      D_12128 = D_12121 + D_12125;
                      S_3184 = D_12122;
                      while (1)
                        {
                          if (S_3184 > D_12123) break;
                          mentrd_rate_2d[S_3184 * D_12127 + D_12128] = D_12124;
                          S_3184 = S_3184 + 1;
                        }

                    }
                  }
                  {
                    long D_12130;
                    long D_12131;
                    GFReal D_12132;
                    long D_12133;

                    D_12130 = offset_2236;
                    D_12131 = (long) jmin[(long) i + offset_2537];
                    D_12132 = xlamdd[(long) i + offset_3026];
                    D_12133 = (long) i;
                    {
                      long S_3185;
                      long D_12135;
                      long D_12136;

                      D_12135 = stride_2235;
                      D_12136 = D_12130 + D_12133;
                      S_3185 = 1;
                      while (1)
                        {
                          if (S_3185 > D_12131) break;
                          cdd[S_3185 * D_12135 + D_12136] = D_12132;
                          S_3185 = S_3185 + 1;
                        }

                    }
                  }
                  cdd[((long) jmin[(long) i + offset_2537] * stride_2235 + offset_2236) + (long) i] = 0.0f;
                  zd[((long) jmin[(long) i + offset_2537] * stride_3125 + offset_3126) + (long) i] = 2.0000000298023223876953125e-1f;
                  zdo[((long) jmin[(long) i + offset_2537] * stride_3132 + offset_3133) + (long) i] = 2.0000000298023223876953125e-1f;
                  frh[(long) i + offset_2418] = ((zdo[((long) jmin[(long) i + offset_2537] * stride_3132 + offset_3133) + (long) i] - 1.0e+0f)) / (((GFReal) (jmin[(long) i + offset_2537] * jmin[(long) i + offset_2537]) - (GFReal) ((jmin[(long) i + offset_2537] - levadj) * (jmin[(long) i + offset_2537] - levadj))));
                  dh = zdo[((long) jmin[(long) i + offset_2537] * stride_3132 + offset_3133) + (long) i] - frh[(long) i + offset_2418] * (GFReal) (jmin[(long) i + offset_2537] * jmin[(long) i + offset_2537]);
                  zuhe[(long) i + offset_3166] = zdo[((long) jmin[(long) i + offset_2537] * stride_3132 + offset_3133) + (long) i];
                  {
                    int D_12138;
                    int D_12139;

                    D_12138 = jmin[(long) i + offset_2537] + -1;
                    D_12139 = jmin[(long) i + offset_2537] - levadj;
                    ki = D_12138;
                    while (1)
                      {
                        {
                          bool D_12142;

                          D_12142 = ki < D_12139;
                          if (D_12142) break;
                          cdd[((long) ki * stride_2235 + offset_2236) + (long) i] = 0.0f;
                          dz = z_cup[((long) (ki + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) ki * stride_3118 + offset_3119) + (long) i];
                          mentrd_rate_2d[((long) ki * stride_2595 + offset_2596) + (long) i] = ((((frh[(long) i + offset_2418] * (GFReal) (ki * ki) + dh)) / zuhe[(long) i + offset_3166] - 1.0e+0f)) / dz;
                          zuhe[(long) i + offset_3166] = zuhe[(long) i + offset_3166] + (mentrd_rate_2d[((long) ki * stride_2595 + offset_2596) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                          ki = ki + -1;
                        }
                      }

                  }
                  {
                    int D_12143;
                    int M_3186;
                    int D_12145;

                    D_12143 = kbcon[(long) i + offset_2549];
                    M_3186 = D_12143;
                    D_12145 = kdet[(long) i + offset_2561];
                    M_3186 = gf_max(D_12145, M_3186);
                    kstart = M_3186 + -1;
                  }
                  {
                    int D_12146;
                    int M_3187;

                    D_12146 = jmin[(long) i + offset_2537] - levadj;
                    M_3187 = D_12146;
                    M_3187 = gf_min(kstart, M_3187);
                    kstart = M_3187;
                  }
                  {
                    int M_3188;

                    M_3188 = 2;
                    M_3188 = gf_max(kstart, M_3188);
                    kstart = M_3188;
                  }
                  if ((jmin[(long) i + offset_2537] - levadj) + -1 > kstart)
                    {
                      {
                        int D_12150;
                        int D_12151;

                        D_12150 = (jmin[(long) i + offset_2537] - levadj) + -1;
                        D_12151 = kstart;
                        ki = D_12150;
                        while (1)
                          {
                            {
                              bool D_12154;

                              D_12154 = ki < D_12151;
                              if (D_12154) break;
                              dz = z_cup[((long) (ki + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) ki * stride_3118 + offset_3119) + (long) i];
                              mentrd_rate_2d[((long) ki * stride_2595 + offset_2596) + (long) i] = mentrd_rate[(long) i + offset_2589];
                              cdd[((long) ki * stride_2235 + offset_2236) + (long) i] = xlamdd[(long) i + offset_3026];
                              zuhe[(long) i + offset_3166] = (zuhe[(long) i + offset_3166] - (cdd[((long) ki * stride_2235 + offset_2236) + (long) i] * dz) * zuhe[(long) i + offset_3166]) + (mentrd_rate_2d[((long) ki * stride_2595 + offset_2596) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                              ki = ki + -1;
                            }
                          }

                      }
                    }

                  frh[(long) i + offset_2418] = ((zuhe[(long) i + offset_3166] - *beta)) / (((GFReal) (kstart * kstart) - 1.0e+0f));
                  dh = *beta - frh[(long) i + offset_2418];
                  mentrd_rate_2d[((long) kstart * stride_2595 + offset_2596) + (long) i] = 0.0f;
                  {
                    int D_12155;

                    D_12155 = kstart + 1;
                    ki = D_12155;
                    while (1)
                      {
                        {
                          bool D_12158;

                          D_12158 = ki <= 0;
                          if (D_12158) break;
                          mentrd_rate_2d[((long) ki * stride_2595 + offset_2596) + (long) i] = 0.0f;
                          dz = z_cup[((long) (ki + 1) * stride_3118 + offset_3119) + (long) i] - z_cup[((long) ki * stride_3118 + offset_3119) + (long) i];
                          {
                            GFReal M_3189;
                            GFReal D_12160;

                            M_3189 = 0.0f;
                            D_12160 = ((1.0e+0f - ((frh[(long) i + offset_2418] * (GFReal) (ki * ki) + dh)) / zuhe[(long) i + offset_3166])) / dz;
                            M_3189 = gf_max(D_12160, M_3189);
                            cdd[((long) ki * stride_2235 + offset_2236) + (long) i] = M_3189;
                          }
                          zuhe[(long) i + offset_3166] = zuhe[(long) i + offset_3166] - (cdd[((long) ki * stride_2235 + offset_2236) + (long) i] * dz) * zuhe[(long) i + offset_3166];

                          ki = ki + -1;
                        }
                      }

                  }
                  {
                    int D_12161;

                    D_12161 = jmin[(long) i + offset_2537] + -1;
                    ki = D_12161;
                    while (1)
                      {
                        {
                          bool D_12164;

                          D_12164 = ki <= 0;
                          if (D_12164) break;
                          mentrd_rate[(long) i + offset_2589] = mentrd_rate_2d[((long) ki * stride_2595 + offset_2596) + (long) i];
                          dzo = zo_cup[((long) (ki + 1) * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) ki * stride_3146 + offset_3147) + (long) i];
                          dd_massentro[((long) ki * stride_2302 + offset_2303) + (long) i] = (mentrd_rate[(long) i + offset_2589] * dzo) * zdo[((long) (ki + 1) * stride_3132 + offset_3133) + (long) i];
                          dd_massdetro[((long) ki * stride_2288 + offset_2289) + (long) i] = (cdd[((long) ki * stride_2235 + offset_2236) + (long) i] * dzo) * zdo[((long) (ki + 1) * stride_3132 + offset_3133) + (long) i];
                          zdo[((long) ki * stride_3132 + offset_3133) + (long) i] = (zdo[((long) (ki + 1) * stride_3132 + offset_3133) + (long) i] + dd_massentro[((long) ki * stride_2302 + offset_2303) + (long) i]) - dd_massdetro[((long) ki * stride_2288 + offset_2289) + (long) i];

                          ki = ki + -1;
                        }
                      }

                  }
                  dbydo[((long) jmin[(long) i + offset_2537] * stride_2267 + offset_2268) + (long) i] = hcdo[((long) jmin[(long) i + offset_2537] * stride_2445 + offset_2446) + (long) i] - heso_cup[((long) jmin[(long) i + offset_2537] * stride_2508 + offset_2509) + (long) i];
                  bud[(long) i + offset_2210] = dbydo[((long) jmin[(long) i + offset_2537] * stride_2267 + offset_2268) + (long) i] * ((zo_cup[((long) (jmin[(long) i + offset_2537] + 1) * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) jmin[(long) i + offset_2537] * stride_3146 + offset_3147) + (long) i]));
                  {
                    int D_12165;

                    D_12165 = jmin[(long) i + offset_2537] + -1;
                    ki = D_12165;
                    while (1)
                      {
                        {
                          bool D_12168;

                          D_12168 = ki <= 0;
                          if (D_12168) break;
                          dzo = zo_cup[((long) (ki + 1) * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) ki * stride_3146 + offset_3147) + (long) i];
                          hcdo[((long) ki * stride_2445 + offset_2446) + (long) i] = (((hcdo[((long) (ki + 1) * stride_2445 + offset_2446) + (long) i] * zdo[((long) (ki + 1) * stride_3132 + offset_3133) + (long) i] - (dd_massdetro[((long) ki * stride_2288 + offset_2289) + (long) i] * 5.0e-1f) * hcdo[((long) (ki + 1) * stride_2445 + offset_2446) + (long) i]) + dd_massentro[((long) ki * stride_2302 + offset_2303) + (long) i] * heo[((long) ki * stride_2473 + offset_2474) + (long) i])) / (((zdo[((long) (ki + 1) * stride_3132 + offset_3133) + (long) i] - dd_massdetro[((long) ki * stride_2288 + offset_2289) + (long) i] * 5.0e-1f) + dd_massentro[((long) ki * stride_2302 + offset_2303) + (long) i]));
                          dbydo[((long) ki * stride_2267 + offset_2268) + (long) i] = hcdo[((long) ki * stride_2445 + offset_2446) + (long) i] - heso_cup[((long) ki * stride_2508 + offset_2509) + (long) i];
                          bud[(long) i + offset_2210] = bud[(long) i + offset_2210] + dbydo[((long) ki * stride_2267 + offset_2268) + (long) i] * dzo;

                          ki = ki + -1;
                        }
                      }

                  }
                }

              if (bud[(long) i + offset_2210] > 0.0f)
                {
                  ierr[(long) i + offset_2521] = 7;
                  gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "downdraft is not negatively buoyant ", 36);
                  gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 36, 32, 14);
                }


              i = i + 1;
            }
          }

      }
      {
        int C_12170 = 1;

        cup_dd_moisture_new (workspace, ierrc, zdo, hcdo, heso_cup, qcdo, qeso_cup, pwdo, qo_cup, zo_cup, dd_massentro, dd_massdetro, jmin, ierr, gammao_cup, pwevo, bu, qrcdo, qo, heo, tn_cup, &C_12170, xl, itf, jtf, ktf, its, ite, jts, jte, kts, kte, 50);
      }
      {
        int C_12171 = 1;

        cup_up_moisture (workspace, "deep", ierr, zo_cup, qco, qrco, pwo, pwavo, ccnclean, p_cup, kbcon, ktop, cd, dbyo, clw_all, t_cup, qo, gammao_cup, zuo, qeso_cup, k22, qo_cup, xl, zqexec, use_excess, ccn, rho, up_massentr, up_massdetr, psum, psumh, autoconv, aeroevap, &C_12171, itf, jtf, ktf, j, ipr, jpr, its, ite, jts, jte, kts, kte, 4);
      }
      {
        int D_12172;
        int D_12173;

        D_12172 = *kts;
        D_12173 = *ktf;
        k = D_12172;
        while (1)
          {
            {
              bool D_12176;

              D_12176 = k > D_12173;
              if (D_12176) break;
              {
                int D_12177;
                int D_12178;

                D_12177 = *its;
                D_12178 = *itf;
                i = D_12177;
                while (1)
                  {
                    {
                      bool D_12181;

                      D_12181 = i > D_12178;
                      if (D_12181) break;
                      cupclw[((long) k * stride_2253 + offset_2254) + (long) i] = qrco[((long) k * stride_2793 + offset_2794) + (long) i];

                      i = i + 1;
                    }
                  }

              }

              k = k + 1;
            }
          }

      }
      cup_up_aa0 (workspace, aa0, z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      cup_up_aa0 (workspace, aa1, zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
      {
        int D_12182;
        int D_12183;

        D_12182 = *its;
        D_12183 = *itf;
        i = D_12182;
        while (1)
          {
            {
              bool D_12186;

              D_12186 = i > D_12183;
              if (D_12186) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  if (aa1[(long) i + offset_2136] == 0.0f)
                    {
                      ierr[(long) i + offset_2521] = 17;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "cloud work function zero", 24);
                      gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 24, 32, 26);
                    }

                }


              i = i + 1;
            }
          }

      }
      {
        int D_12189;

        D_12189 = *ens4;
        i = 1;
        while (1)
          {
            {
              bool D_12192;

              D_12192 = i > D_12189;
              if (D_12192) break;
              {
                long D_12193;
                long D_12194;
                long D_12195;
                long D_12196;
                long D_12197;
                long D_12198;
                long D_12199;
                long D_12200;

                D_12193 = offset_2136;
                D_12194 = lbound_2134;
                D_12195 = ubound_2135;
                D_12196 = offset_2202;
                D_12197 = lbound_2198;
                D_12198 = ubound_2199;
                D_12199 = (long) i;
                D_12200 = D_12197 - D_12194;
                {
                  long S_3190;
                  long D_12202;

                  D_12202 = D_12199 * stride_2201 + D_12196;
                  S_3190 = D_12194;
                  while (1)
                    {
                      if (S_3190 > D_12195) break;
                      axx[(S_3190 + D_12200) + D_12202] = aa1[S_3190 + D_12193];
                      S_3190 = S_3190 + 1;
                    }

                }
              }

              i = i + 1;
            }
          }

      }
      cup_dd_edt (workspace, ierr, us, vs, zo, ktop, kbcon, edt, po, pwavo, pwo, ccn, pwevo, &edtmax, &edtmin, maxens2, edtc, psum, psumh, ccnclean, rho, aeroevap, itf, jtf, ktf, j, ipr, jpr, its, ite, jts, jte, kts, kte);
      {
        int D_12204;

        D_12204 = *maxens2;
        iedt = 1;
        while (1)
          {
            {
              bool D_12207;

              D_12207 = iedt > D_12204;
              if (D_12207) break;
              {
                int D_12208;
                int D_12209;

                D_12208 = *its;
                D_12209 = *itf;
                i = D_12208;
                while (1)
                  {
                    {
                      bool D_12212;

                      D_12212 = i > D_12209;
                      if (D_12212) break;
                      if (ierr[(long) i + offset_2521] == 0)
                        {
                          edt[(long) i + offset_2387] = edtc[((long) iedt * stride_2394 + offset_2395) + (long) i];
                          edto[(long) i + offset_2399] = edtc[((long) iedt * stride_2394 + offset_2395) + (long) i];
                          edtx[(long) i + offset_2403] = edtc[((long) iedt * stride_2394 + offset_2395) + (long) i];
                          if (*maxens2 == 3)
                            {
                              edt[(long) i + offset_2387] = edtc[(stride_2394 * 3 + offset_2395) + (long) i];
                              edto[(long) i + offset_2399] = edtc[(stride_2394 * 3 + offset_2395) + (long) i];
                              edtx[(long) i + offset_2403] = edtc[(stride_2394 * 3 + offset_2395) + (long) i];
                            }

                        }


                      i = i + 1;
                    }
                  }

              }
              {
                int D_12215;
                int D_12216;

                D_12215 = *kts;
                D_12216 = *ktf;
                k = D_12215;
                while (1)
                  {
                    {
                      bool D_12219;

                      D_12219 = k > D_12216;
                      if (D_12219) break;
                      {
                        int D_12220;
                        int D_12221;

                        D_12220 = *its;
                        D_12221 = *itf;
                        i = D_12220;
                        while (1)
                          {
                            {
                              bool D_12224;

                              D_12224 = i > D_12221;
                              if (D_12224) break;
                              subt_ens[(((long) iedt * stride_2851 + offset_2852) + (long) k * stride_2849) + (long) i] = 0.0f;
                              subq_ens[(((long) iedt * stride_2835 + offset_2836) + (long) k * stride_2833) + (long) i] = 0.0f;
                              dellat_ens[(((long) iedt * stride_2357 + offset_2358) + (long) k * stride_2355) + (long) i] = 0.0f;
                              dellaq_ens[(((long) iedt * stride_2325 + offset_2326) + (long) k * stride_2323) + (long) i] = 0.0f;
                              dellaqc_ens[(((long) iedt * stride_2341 + offset_2342) + (long) k * stride_2339) + (long) i] = 0.0f;
                              pwo_ens[(((long) iedt * stride_2709 + offset_2710) + (long) k * stride_2707) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              {
                int D_12225;
                int D_12226;

                D_12225 = *kts;
                D_12226 = *ktf;
                k = D_12225;
                while (1)
                  {
                    {
                      bool D_12229;

                      D_12229 = k > D_12226;
                      if (D_12229) break;
                      {
                        int D_12230;
                        int D_12231;

                        D_12230 = *its;
                        D_12231 = *itf;
                        i = D_12230;
                        while (1)
                          {
                            {
                              bool D_12234;

                              D_12234 = i > D_12231;
                              if (D_12234) break;
                              dellah[((long) k * stride_2309 + offset_2310) + (long) i] = 0.0f;
                              dsubt[((long) k * stride_2378 + offset_2379) + (long) i] = 0.0f;
                              dsubh[((long) k * stride_2364 + offset_2365) + (long) i] = 0.0f;
                              dellaq[((long) k * stride_2316 + offset_2317) + (long) i] = 0.0f;
                              dsubq[((long) k * stride_2371 + offset_2372) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              {
                int D_12235;
                int D_12236;

                D_12235 = *its;
                D_12236 = *itf;
                i = D_12235;
                while (1)
                  {
                    {
                      bool D_12239;

                      D_12239 = i > D_12236;
                      if (D_12239) break;
                      if (ierr[(long) i + offset_2521] == 0)
                        {
                          dp = ((po_cup[(offset_2654 + (stride_2653)) + (long) i] - po_cup[(stride_2653 * 2 + offset_2654) + (long) i])) * 1.0e+2f;
                          dellah[(offset_2310 + (stride_2309)) + (long) i] = ((((edto[(long) i + offset_2399] * zdo[(stride_3132 * 2 + offset_3133) + (long) i]) * hcdo[(stride_2445 * 2 + offset_2446) + (long) i] - (edto[(long) i + offset_2399] * zdo[(stride_3132 * 2 + offset_3133) + (long) i]) * heo_cup[(stride_2480 * 2 + offset_2481) + (long) i])) * *g) / dp;
                          dellaq[(offset_2317 + (stride_2316)) + (long) i] = ((((edto[(long) i + offset_2399] * zdo[(stride_3132 * 2 + offset_3133) + (long) i]) * qrcdo[(stride_2786 * 2 + offset_2787) + (long) i] - (edto[(long) i + offset_2399] * zdo[(stride_3132 * 2 + offset_3133) + (long) i]) * qo_cup[(stride_2779 * 2 + offset_2780) + (long) i])) * *g) / dp;
                          dsubt[(offset_2379 + (stride_2378)) + (long) i] = 0.0f;
                          dsubq[(offset_2372 + (stride_2371)) + (long) i] = 0.0f;
                          {
                            int D_12241;
                            int D_12242;

                            D_12241 = *kts + 1;
                            D_12242 = ktop[(long) i + offset_2573];
                            k = D_12241;
                            while (1)
                              {
                                {
                                  bool D_12245;

                                  D_12245 = k > D_12242;
                                  if (D_12245) break;
                                  entupk = 0.0f;
                                  detupk = 0.0f;
                                  entdoj = 0.0f;
                                  detdo = edto[(long) i + offset_2399] * dd_massdetro[((long) k * stride_2288 + offset_2289) + (long) i];
                                  entdo = edto[(long) i + offset_2399] * dd_massentro[((long) k * stride_2302 + offset_2303) + (long) i];
                                  entup = up_massentro[((long) k * stride_2907 + offset_2908) + (long) i];
                                  detup = up_massdetro[((long) k * stride_2893 + offset_2894) + (long) i];
                                  subin = -(zdo[((long) (k + 1) * stride_3132 + offset_3133) + (long) i] * edto[(long) i + offset_2399]);
                                  subdown = -(zdo[((long) k * stride_3132 + offset_3133) + (long) i] * edto[(long) i + offset_2399]);
                                  if (jmin[(long) i + offset_2537] == k)
                                    {
                                      entdoj = edto[(long) i + offset_2399] * zdo[((long) k * stride_3132 + offset_3133) + (long) i];
                                    }

                                  if (ktop[(long) i + offset_2573] == k)
                                    {
                                      detupk = zuo[((long) ktop[(long) i + offset_2573] * stride_3172 + offset_3173) + (long) i];
                                      subin = 0.0f;
                                      subdown = 0.0f;
                                      detdo = 0.0f;
                                      entdo = 0.0f;
                                      entup = 0.0f;
                                      detup = 0.0f;
                                    }

                                  totmas = (((((((((subin - subdown) + detup) - entup) - entdo) + detdo) - entupk) - entdoj) + detupk) + zuo[((long) (k + 1) * stride_3172 + offset_3173) + (long) i]) - zuo[((long) k * stride_3172 + offset_3173) + (long) i];
                                  if (gf_abs(totmas) > 9.999999974752427078783512115478515625e-7f)
                                    {
                                      {
                                        workspace.diagnostics |= 1;
                                      }
                                      {
                                        workspace.diagnostics |= 1;
                                      }
                                    }

                                  dp = ((po_cup[((long) k * stride_2653 + offset_2654) + (long) i] - po_cup[((long) (k + 1) * stride_2653 + offset_2654) + (long) i])) * 1.0e+2f;
                                  dellah[((long) k * stride_2309 + offset_2310) + (long) i] = ((((((gf_add_rhs(((((detup * 5.0e-1f) * ((hco[((long) (k + 1) * stride_2452 + offset_2453) + (long) i] + hco[((long) k * stride_2452 + offset_2453) + (long) i])) + (detdo * 5.0e-1f) * ((hcdo[((long) (k + 1) * stride_2445 + offset_2446) + (long) i] + hcdo[((long) k * stride_2445 + offset_2446) + (long) i]))) - heo[((long) k * stride_2473 + offset_2474) + (long) i] * entup) - heo[((long) k * stride_2473 + offset_2474) + (long) i] * entdo),heo_cup[((long) (k + 1) * stride_2480 + offset_2481) + (long) i] * subin) - heo_cup[((long) k * stride_2480 + offset_2481) + (long) i] * subdown) + ((hco[((long) ktop[(long) i + offset_2573] * stride_2452 + offset_2453) + (long) i] - heo_cup[((long) ktop[(long) i + offset_2573] * stride_2480 + offset_2481) + (long) i])) * detupk) - heo_cup[((long) k22[(long) i + offset_2541] * stride_2480 + offset_2481) + (long) i] * entupk) - heo_cup[((long) jmin[(long) i + offset_2537] * stride_2480 + offset_2481) + (long) i] * entdoj)) * *g) / dp;
                                  dellaq[((long) k * stride_2316 + offset_2317) + (long) i] = (((((((((((detup * 5.0e-1f) * ((((qco[((long) (k + 1) * stride_2737 + offset_2738) + (long) i] + qco[((long) k * stride_2737 + offset_2738) + (long) i]) - qrco[((long) (k + 1) * stride_2793 + offset_2794) + (long) i]) - qrco[((long) k * stride_2793 + offset_2794) + (long) i])) + (detdo * 5.0e-1f) * ((qrcdo[((long) (k + 1) * stride_2786 + offset_2787) + (long) i] + qrcdo[((long) k * stride_2786 + offset_2787) + (long) i]))) - qo[((long) k * stride_2772 + offset_2773) + (long) i] * entup) - qo[((long) k * stride_2772 + offset_2773) + (long) i] * entdo) + qo_cup[((long) (k + 1) * stride_2779 + offset_2780) + (long) i] * subin) - qo_cup[((long) k * stride_2779 + offset_2780) + (long) i] * subdown) + (((qco[((long) ktop[(long) i + offset_2573] * stride_2737 + offset_2738) + (long) i] - qrco[((long) ktop[(long) i + offset_2573] * stride_2793 + offset_2794) + (long) i]) - qo_cup[((long) ktop[(long) i + offset_2573] * stride_2779 + offset_2780) + (long) i])) * detupk) - qo_cup[((long) k22[(long) i + offset_2541] * stride_2779 + offset_2780) + (long) i] * entupk) - qo_cup[((long) jmin[(long) i + offset_2537] * stride_2779 + offset_2780) + (long) i] * entdoj)) * *g) / dp;
                                  if (ktop[(long) i + offset_2573] > k)
                                    {
                                      dsubt[((long) k * stride_2378 + offset_2379) + (long) i] = (((zuo[((long) (k + 1) * stride_3172 + offset_3173) + (long) i] * heo_cup[((long) (k + 1) * stride_2480 + offset_2481) + (long) i] - zuo[((long) k * stride_3172 + offset_3173) + (long) i] * heo_cup[((long) k * stride_2480 + offset_2481) + (long) i])) * *g) / dp;
                                      dsubq[((long) k * stride_2371 + offset_2372) + (long) i] = (((zuo[((long) (k + 1) * stride_3172 + offset_3173) + (long) i] * qo_cup[((long) (k + 1) * stride_2779 + offset_2780) + (long) i] - zuo[((long) k * stride_3172 + offset_3173) + (long) i] * qo_cup[((long) k * stride_2779 + offset_2780) + (long) i])) * *g) / dp;
                                    }


                                  k = k + 1;
                                }
                              }

                          }
                        }


                      i = i + 1;
                    }
                  }

              }
              {
                int D_12252;
                int D_12253;

                D_12252 = *kts;
                D_12253 = *ktf + -1;
                k = D_12252;
                while (1)
                  {
                    {
                      bool D_12256;

                      D_12256 = k > D_12253;
                      if (D_12256) break;
                      {
                        int D_12257;
                        int D_12258;

                        D_12257 = *its;
                        D_12258 = *itf;
                        i = D_12257;
                        while (1)
                          {
                            {
                              bool D_12261;

                              D_12261 = i > D_12258;
                              if (D_12261) break;
                              dellaqc[((long) k * stride_2332 + offset_2333) + (long) i] = 0.0f;
                              if (ierr[(long) i + offset_2521] == 0)
                                {
                                  if ((ktop[(long) i + offset_2573]) == k)
                                    {
                                      dellaqc[((long) k * stride_2332 + offset_2333) + (long) i] = (((zuo[((long) ktop[(long) i + offset_2573] * stride_3172 + offset_3173) + (long) i] * 9.99999977648258209228515625e-3f) * qrco[((long) ktop[(long) i + offset_2573] * stride_2793 + offset_2794) + (long) i]) * 9.81000041961669921875e+0f) / ((po_cup[((long) k * stride_2653 + offset_2654) + (long) i] - po_cup[((long) (k + 1) * stride_2653 + offset_2654) + (long) i]));
                                    }

                                  if (ktop[(long) i + offset_2573] > k && kbcon[(long) i + offset_2549] < k)
                                    {
                                      dz = zo_cup[((long) (k + 1) * stride_3146 + offset_3147) + (long) i] - zo_cup[((long) k * stride_3146 + offset_3147) + (long) i];
                                      dellaqc[((long) k * stride_2332 + offset_2333) + (long) i] = (((up_massdetro[((long) k * stride_2893 + offset_2894) + (long) i] * 9.8099999129772186279296875e-2f) * 5.0e-1f) * ((qrco[((long) k * stride_2793 + offset_2794) + (long) i] + qrco[((long) (k + 1) * stride_2793 + offset_2794) + (long) i]))) / ((po_cup[((long) k * stride_2653 + offset_2654) + (long) i] - po_cup[((long) (k + 1) * stride_2653 + offset_2654) + (long) i]));
                                    }

                                  {
                                    GFReal M_3193;
                                    GFReal D_12266;

                                    M_3193 = 0.0f;
                                    D_12266 = dellaqc[((long) k * stride_2332 + offset_2333) + (long) i];
                                    M_3193 = gf_max(D_12266, M_3193);
                                    dellaqc[((long) k * stride_2332 + offset_2333) + (long) i] = M_3193;
                                  }
                                }


                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              mbdt = mbdt_ens[0];
              {
                int D_12267;
                int D_12268;

                D_12267 = *its;
                D_12268 = *itf;
                i = D_12267;
                while (1)
                  {
                    {
                      bool D_12271;

                      D_12271 = i > D_12268;
                      if (D_12271) break;
                      {
                        long D_12272;
                        long D_12273;
                        long D_12274;

                        D_12272 = offset_2967;
                        D_12273 = ubound_2965;
                        D_12274 = (long) i;
                        {
                          long S_3194;
                          long D_12276;
                          long D_12277;

                          D_12276 = stride_2966;
                          D_12277 = D_12272 + D_12274;
                          S_3194 = 1;
                          while (1)
                            {
                              if (S_3194 > D_12273) break;
                              xaa0_ens[S_3194 * D_12276 + D_12277] = 0.0f;
                              S_3194 = S_3194 + 1;
                            }

                        }
                      }

                      i = i + 1;
                    }
                  }

              }
              {
                int D_12279;
                int D_12280;

                D_12279 = *kts;
                D_12280 = *ktf;
                k = D_12279;
                while (1)
                  {
                    {
                      bool D_12283;

                      D_12283 = k > D_12280;
                      if (D_12283) break;
                      {
                        int D_12284;
                        int D_12285;

                        D_12284 = *its;
                        D_12285 = *itf;
                        i = D_12284;
                        while (1)
                          {
                            {
                              bool D_12288;

                              D_12288 = i > D_12285;
                              if (D_12288) break;
                              dellat[((long) k * stride_2348 + offset_2349) + (long) i] = 0.0f;
                              if (ierr[(long) i + offset_2521] == 0)
                                {
                                  dsubh[((long) k * stride_2364 + offset_2365) + (long) i] = dsubt[((long) k * stride_2378 + offset_2379) + (long) i];
                                  xhe[((long) k * stride_2996 + offset_2997) + (long) i] = ((dsubt[((long) k * stride_2378 + offset_2379) + (long) i] + dellah[((long) k * stride_2309 + offset_2310) + (long) i])) * mbdt + heo[((long) k * stride_2473 + offset_2474) + (long) i];
                                  xq[((long) k * stride_3044 + offset_3045) + (long) i] = (((dsubq[((long) k * stride_2371 + offset_2372) + (long) i] + dellaq[((long) k * stride_2316 + offset_2317) + (long) i]) + dellaqc[((long) k * stride_2332 + offset_2333) + (long) i])) * mbdt + qo[((long) k * stride_2772 + offset_2773) + (long) i];
                                  dellat[((long) k * stride_2348 + offset_2349) + (long) i] = ((1.0e+0f / *cp)) * ((dellah[((long) k * stride_2309 + offset_2310) + (long) i] - *xl * dellaq[((long) k * stride_2316 + offset_2317) + (long) i]));
                                  dsubt[((long) k * stride_2378 + offset_2379) + (long) i] = ((1.0e+0f / *cp)) * ((dsubt[((long) k * stride_2378 + offset_2379) + (long) i] - *xl * dsubq[((long) k * stride_2371 + offset_2372) + (long) i]));
                                  xt[((long) k * stride_3072 + offset_3073) + (long) i] = (((dellat[((long) k * stride_2348 + offset_2349) + (long) i] + dsubt[((long) k * stride_2378 + offset_2379) + (long) i]) - (dellaqc[((long) k * stride_2332 + offset_2333) + (long) i] * *xl) / *cp)) * mbdt + tn[((long) k * stride_2872 + offset_2873) + (long) i];
                                  if (xq[((long) k * stride_3044 + offset_3045) + (long) i] <= 0.0f)
                                    {
                                      xq[((long) k * stride_3044 + offset_3045) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                                    }

                                }


                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              {
                int D_12291;
                int D_12292;

                D_12291 = *its;
                D_12292 = *itf;
                i = D_12291;
                while (1)
                  {
                    {
                      bool D_12295;

                      D_12295 = i > D_12292;
                      if (D_12295) break;
                      if (ierr[(long) i + offset_2521] == 0)
                        {
                          xhkb[(long) i + offset_3022] = hkbo[(long) i + offset_2517] + ((dsubh[((long) k22[(long) i + offset_2541] * stride_2364 + offset_2365) + (long) i] + dellah[((long) k22[(long) i + offset_2541] * stride_2309 + offset_2310) + (long) i])) * mbdt;
                          xhe[((long) *ktf * stride_2996 + offset_2997) + (long) i] = heo[((long) *ktf * stride_2473 + offset_2474) + (long) i];
                          xq[((long) *ktf * stride_3044 + offset_3045) + (long) i] = qo[((long) *ktf * stride_2772 + offset_2773) + (long) i];
                          xt[((long) *ktf * stride_3072 + offset_3073) + (long) i] = tn[((long) *ktf * stride_2872 + offset_2873) + (long) i];
                          if (xq[((long) *ktf * stride_3044 + offset_3045) + (long) i] <= 0.0f)
                            {
                              xq[((long) *ktf * stride_3044 + offset_3045) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                            }

                        }


                      i = i + 1;
                    }
                  }

              }
              {
                int C_12298 = -1;

                cup_env (workspace, xz, xqes, xhe, xhes, xt, xq, po, z1, psur, ierr, tcrit, &C_12298, xl, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
              }
              cup_env_clev (workspace, xt, xqes, xq, xhe, xhes, xz, po, xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup, xt_cup, psur, ierr, z1, xl, rv, cp, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
              {
                int D_12299;
                int D_12300;

                D_12299 = *kts;
                D_12300 = *ktf;
                k = D_12299;
                while (1)
                  {
                    {
                      bool D_12303;

                      D_12303 = k > D_12300;
                      if (D_12303) break;
                      {
                        int D_12304;
                        int D_12305;

                        D_12304 = *its;
                        D_12305 = *itf;
                        i = D_12304;
                        while (1)
                          {
                            {
                              bool D_12308;

                              D_12308 = i > D_12305;
                              if (D_12308) break;
                              xhc[((long) k * stride_2989 + offset_2990) + (long) i] = 0.0f;
                              xdby[((long) k * stride_2973 + offset_2974) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              {
                int D_12309;
                int D_12310;

                D_12309 = *its;
                D_12310 = *itf;
                i = D_12309;
                while (1)
                  {
                    {
                      bool D_12313;

                      D_12313 = i > D_12310;
                      if (D_12313) break;
                      if (ierr[(long) i + offset_2521] == 0)
                        {
                          {
                            int D_12315;

                            D_12315 = kbcon[(long) i + offset_2549] + -1;
                            k = 1;
                            while (1)
                              {
                                {
                                  bool D_12318;

                                  D_12318 = k > D_12315;
                                  if (D_12318) break;
                                  xhc[((long) k * stride_2989 + offset_2990) + (long) i] = xhkb[(long) i + offset_3022];

                                  k = k + 1;
                                }
                              }

                          }
                          k = kbcon[(long) i + offset_2549];
                          xhc[((long) k * stride_2989 + offset_2990) + (long) i] = xhkb[(long) i + offset_3022];
                          xdby[((long) kbcon[(long) i + offset_2549] * stride_2973 + offset_2974) + (long) i] = xhkb[(long) i + offset_3022] - xhes_cup[((long) k * stride_3017 + offset_3018) + (long) i];
                        }


                      i = i + 1;
                    }
                  }

              }
              {
                int D_12319;
                int D_12320;

                D_12319 = *its;
                D_12320 = *itf;
                i = D_12319;
                while (1)
                  {
                    {
                      bool D_12323;

                      D_12323 = i > D_12320;
                      if (D_12323) break;
                      if (ierr[(long) i + offset_2521] == 0)
                        {
                          {
                            long D_12325;
                            long D_12326;
                            long D_12327;
                            long D_12328;
                            long D_12329;
                            long D_12330;
                            long D_12331;
                            long D_12332;
                            long D_12333;

                            D_12325 = offset_3173;
                            D_12326 = lbound_3170;
                            D_12327 = ubound_3171;
                            D_12328 = offset_3101;
                            D_12329 = lbound_3098;
                            D_12330 = ubound_3099;
                            D_12331 = (long) i;
                            D_12332 = (long) i;
                            D_12333 = D_12329 - D_12326;
                            {
                              long S_3195;
                              long D_12335;
                              long D_12336;
                              long D_12337;
                              long D_12338;

                              D_12335 = stride_3172;
                              D_12336 = D_12325 + D_12331;
                              D_12337 = stride_3100;
                              D_12338 = D_12328 + D_12332;
                              S_3195 = D_12326;
                              while (1)
                                {
                                  if (S_3195 > D_12327) break;
                                  xzu[(S_3195 + D_12333) * D_12337 + D_12338] = zuo[S_3195 * D_12335 + D_12336];
                                  S_3195 = S_3195 + 1;
                                }

                            }
                          }
                          {
                            int D_12340;
                            int D_12341;

                            D_12340 = kbcon[(long) i + offset_2549] + 1;
                            D_12341 = ktop[(long) i + offset_2573];
                            k = D_12340;
                            while (1)
                              {
                                {
                                  bool D_12344;

                                  D_12344 = k > D_12341;
                                  if (D_12344) break;
                                  xhc[((long) k * stride_2989 + offset_2990) + (long) i] = (((xhc[((long) (k + -1) * stride_2989 + offset_2990) + (long) i] * xzu[((long) (k + -1) * stride_3100 + offset_3101) + (long) i] - (up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] * 5.0e-1f) * xhc[((long) (k + -1) * stride_2989 + offset_2990) + (long) i]) + up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i] * xhe[((long) (k + -1) * stride_2996 + offset_2997) + (long) i])) / (((xzu[((long) (k + -1) * stride_3100 + offset_3101) + (long) i] - up_massdetro[((long) (k + -1) * stride_2893 + offset_2894) + (long) i] * 5.0e-1f) + up_massentro[((long) (k + -1) * stride_2907 + offset_2908) + (long) i]));
                                  xdby[((long) k * stride_2973 + offset_2974) + (long) i] = xhc[((long) k * stride_2989 + offset_2990) + (long) i] - xhes_cup[((long) k * stride_3017 + offset_3018) + (long) i];

                                  k = k + 1;
                                }
                              }

                          }
                          {
                            int D_12345;
                            int D_12346;

                            D_12345 = ktop[(long) i + offset_2573] + 1;
                            D_12346 = *ktf;
                            k = D_12345;
                            while (1)
                              {
                                {
                                  bool D_12349;

                                  D_12349 = k > D_12346;
                                  if (D_12349) break;
                                  xhc[((long) k * stride_2989 + offset_2990) + (long) i] = xhes_cup[((long) k * stride_3017 + offset_3018) + (long) i];
                                  xdby[((long) k * stride_2973 + offset_2974) + (long) i] = 0.0f;
                                  xzu[((long) k * stride_3100 + offset_3101) + (long) i] = 0.0f;

                                  k = k + 1;
                                }
                              }

                          }
                        }


                      i = i + 1;
                    }
                  }

              }
              cup_up_aa0 (workspace, xaa0, xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop, ierr, itf, jtf, ktf, its, ite, jts, jte, kts, kte);
              {
                int D_12350;

                D_12350 = *maxens;
                nens = 1;
                while (1)
                  {
                    {
                      bool D_12353;

                      D_12353 = nens > D_12350;
                      if (D_12353) break;
                      {
                        int D_12354;
                        int D_12355;

                        D_12354 = *its;
                        D_12355 = *itf;
                        i = D_12354;
                        while (1)
                          {
                            {
                              bool D_12358;

                              D_12358 = i > D_12355;
                              if (D_12358) break;
                              if (ierr[(long) i + offset_2521] == 0)
                                {
                                  xaa0_ens[((long) nens * stride_2966 + offset_2967) + (long) i] = xaa0[(long) i + offset_2961];
                                  nall = ((((*iens + -1) * *maxens3) * *maxens) * *maxens2 + ((iedt + -1) * *maxens) * *maxens3) + (nens + -1) * *maxens3;
                                  {
                                    int D_12360;
                                    int D_12361;

                                    D_12360 = *kts;
                                    D_12361 = *ktf;
                                    k = D_12360;
                                    while (1)
                                      {
                                        {
                                          bool D_12364;

                                          D_12364 = k > D_12361;
                                          if (D_12364) break;
                                          if (ktop[(long) i + offset_2573] >= k)
                                            {
                                              {
                                                int D_12366;

                                                D_12366 = *maxens3;
                                                nens3 = 1;
                                                while (1)
                                                  {
                                                    {
                                                      bool D_12369;

                                                      D_12369 = nens3 > D_12366;
                                                      if (D_12369) break;
                                                      if (nens3 == 7)
                                                        {
                                                          pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] + pwo[((long) k * stride_2700 + offset_2701) + (long) i];
                                                        }
                                                      else
                                                        {
                                                          if (nens3 == 8)
                                                            {
                                                              pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] + pwo[((long) k * stride_2700 + offset_2701) + (long) i];
                                                            }
                                                          else
                                                            {
                                                              if (nens3 == 9)
                                                                {
                                                                  pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] + pwo[((long) k * stride_2700 + offset_2701) + (long) i];
                                                                }
                                                              else
                                                                {
                                                                  pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] + pwo[((long) k * stride_2700 + offset_2701) + (long) i];
                                                                }

                                                            }

                                                        }


                                                      nens3 = nens3 + 1;
                                                    }
                                                  }

                                              }
                                            }


                                          k = k + 1;
                                        }
                                      }

                                  }
                                  if (pr_ens[(((long) (nall + 7) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] < 9.999999974752427078783512115478515625e-7f)
                                    {
                                      ierr[(long) i + offset_2521] = 18;
                                      gf_memmove ((device char *)&ierrc[(long) i + offset_2533], "total normalized condensate too small", 37);
                                      gf_memset ((device char *)&ierrc[(long) i + offset_2533] + 37, 32, 13);
                                      {
                                        int D_12374;

                                        D_12374 = *maxens3;
                                        nens3 = 1;
                                        while (1)
                                          {
                                            {
                                              bool D_12377;

                                              D_12377 = nens3 > D_12374;
                                              if (D_12377) break;
                                              pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = 0.0f;

                                              nens3 = nens3 + 1;
                                            }
                                          }

                                      }
                                    }

                                  {
                                    int D_12378;

                                    D_12378 = *maxens3;
                                    nens3 = 1;
                                    while (1)
                                      {
                                        {
                                          bool D_12381;

                                          D_12381 = nens3 > D_12378;
                                          if (D_12381) break;
                                          if (pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] < 9.99999974737875163555145263671875e-5f)
                                            {
                                              pr_ens[(((long) (nall + nens3) * stride_2662 + offset_2663) + (long) *j * stride_2660) + (long) i] = 0.0f;
                                            }


                                          nens3 = nens3 + 1;
                                        }
                                      }

                                  }
                                }


                              i = i + 1;
                            }
                          }

                      }


                      nens = nens + 1;
                    }
                  }

              }
              {
                int D_12384;
                int D_12385;

                D_12384 = *its;
                D_12385 = *itf;
                i = D_12384;
                while (1)
                  {
                    {
                      bool D_12388;

                      D_12388 = i > D_12385;
                      if (D_12388) break;
                      ierr2[(long) i + offset_2525] = ierr[(long) i + offset_2521];
                      ierr3[(long) i + offset_2529] = ierr[(long) i + offset_2521];
                      k22x[(long) i + offset_2545] = k22[(long) i + offset_2541];

                      i = i + 1;
                    }
                  }

              }
              if (*maxens > 0)
                {
                  {
                    int C_12390 = 2;

                    cup_kbcon (workspace, ierrc, cap_max_increment, &C_12390, k22x, kbconx, heo_cup, heso_cup, hkbo, ierr2, kbmax, po_cup, cap_max, xl, cp, ztexec, zqexec, use_excess, itf, jtf, ktf, its, ite, jts, jte, kts, kte, 50);
                  }
                  {
                    int C_12391 = 3;

                    cup_kbcon (workspace, ierrc, cap_max_increment, &C_12391, k22x, kbconx, heo_cup, heso_cup, hkbo, ierr3, kbmax, po_cup, cap_max, xl, cp, ztexec, zqexec, use_excess, itf, jtf, ktf, its, ite, jts, jte, kts, kte, 50);
                  }
                }

              cup_forcing_ens_3d (workspace, closure_n, xland1, aa0, aa1, xaa0_ens, mbdt_ens, dtime, ierr, ierr2, ierr3, xf_ens, j, "deeps", axx, maxens, iens, &iedt, maxens2, maxens3, mconv, po_cup, ktop, omeg, zdo, k22, zuo, pr_ens, edto, kbcon, ensdim, ichoice, ipr, jpr, itf, jtf, ktf, its, ite, jts, jte, kts, kte, ens4, ktau, 5);
              {
                int D_12392;
                int D_12393;

                D_12392 = *kts;
                D_12393 = *ktf;
                k = D_12392;
                while (1)
                  {
                    {
                      bool D_12396;

                      D_12396 = k > D_12393;
                      if (D_12396) break;
                      {
                        int D_12397;
                        int D_12398;

                        D_12397 = *its;
                        D_12398 = *itf;
                        i = D_12397;
                        while (1)
                          {
                            {
                              bool D_12401;

                              D_12401 = i > D_12398;
                              if (D_12401) break;
                              if (ierr[(long) i + offset_2521] == 0)
                                {
                                  subt_ens[(((long) iedt * stride_2851 + offset_2852) + (long) k * stride_2849) + (long) i] = dsubt[((long) k * stride_2378 + offset_2379) + (long) i];
                                  subq_ens[(((long) iedt * stride_2835 + offset_2836) + (long) k * stride_2833) + (long) i] = dsubq[((long) k * stride_2371 + offset_2372) + (long) i];
                                  dellat_ens[(((long) iedt * stride_2357 + offset_2358) + (long) k * stride_2355) + (long) i] = dellat[((long) k * stride_2348 + offset_2349) + (long) i];
                                  dellaq_ens[(((long) iedt * stride_2325 + offset_2326) + (long) k * stride_2323) + (long) i] = dellaq[((long) k * stride_2316 + offset_2317) + (long) i];
                                  dellaqc_ens[(((long) iedt * stride_2341 + offset_2342) + (long) k * stride_2339) + (long) i] = dellaqc[((long) k * stride_2332 + offset_2333) + (long) i];
                                  pwo_ens[(((long) iedt * stride_2709 + offset_2710) + (long) k * stride_2707) + (long) i] = pwo[((long) k * stride_2700 + offset_2701) + (long) i] + edt[(long) i + offset_2387] * pwdo[((long) k * stride_2689 + offset_2690) + (long) i];
                                }
                              else
                                {
                                  subt_ens[(((long) iedt * stride_2851 + offset_2852) + (long) k * stride_2849) + (long) i] = 0.0f;
                                  subq_ens[(((long) iedt * stride_2835 + offset_2836) + (long) k * stride_2833) + (long) i] = 0.0f;
                                  dellat_ens[(((long) iedt * stride_2357 + offset_2358) + (long) k * stride_2355) + (long) i] = 0.0f;
                                  dellaq_ens[(((long) iedt * stride_2325 + offset_2326) + (long) k * stride_2323) + (long) i] = 0.0f;
                                  dellaqc_ens[(((long) iedt * stride_2341 + offset_2342) + (long) k * stride_2339) + (long) i] = 0.0f;
                                  pwo_ens[(((long) iedt * stride_2709 + offset_2710) + (long) k * stride_2707) + (long) i] = 0.0f;
                                }


                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }


              iedt = iedt + 1;
            }
          }

      }
      cup_output_ens_3d (workspace, xf_ens, ierr, dellat_ens, dellaq_ens, dellaqc_ens, subt_ens, subq_ens, subt, subq, outt, outq, outqc, zuo, sub_mas, pre, pwo_ens, xmb, ktop, j, "deep", maxens2, maxens, iens, ierr2, ierr3, pr_ens, maxens3, ensdim, sig, apr_gr, apr_w, apr_mc, apr_st, apr_as, apr_capma, apr_capme, apr_capmi, closure_n, xland1, weight_gr, weight_w, weight_mc, weight_st, weight_as, training, ipr, jpr, itf, jtf, ktf, its, ite, jts, jte, kts, kte, 4);
      k = 1;
      {
        int D_12404;
        int D_12405;

        D_12404 = *its;
        D_12405 = *itf;
        i = D_12404;
        while (1)
          {
            {
              bool D_12408;

              D_12408 = i > D_12405;
              if (D_12408) break;
              if (ierr[(long) i + offset_2521] == 0)
                {
                  {
                    GFReal D_12410;
                    GFReal M_3196;

                    D_12410 = pre[(long) i + offset_2667];
                    M_3196 = D_12410;
                    M_3196 = gf_max(M_3196, 0.0f);
                    pre[(long) i + offset_2667] = M_3196;
                  }
                }


              i = i + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}
template<typename P0,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename P8,typename P9,typename P10,typename P11,typename P12,typename P13,typename P14,typename P15,typename P16,typename P17,typename P18,typename P19,typename P20,typename P21,typename P22,typename P23,typename P24,typename P25,typename P26,typename P27,typename P28,typename P29,typename P30,typename P31,typename P32,typename P33,typename P34,typename P35,typename P36,typename P37,typename P38,typename P39,typename P40,typename P41,typename P42,typename P43,typename P44,typename P45,typename P46,typename P47,typename P48,typename P49,typename P50,typename P51,typename P52,typename P53,typename P54,typename P55,typename P56,typename P57,typename P58,typename P59,typename P60,typename P61,typename P62,typename P63,typename P64> static inline void cu_grell_freitas(thread GFWorkspace &workspace, P0 itimestep, P1 dt, P2 dxcell, P3 areacell, P4 u, P5 v, P6 t, P7 w, P8 q, P9 p, P10 pi, P11 rho, P12 dz8w, P13 p8w, P14 xland, P15 ht, P16 hfx, P17 qfx, P18 gsw, P19 rqvften, P20 rthften, P21 rthblten, P22 rqvblten, P23 rthraten, P24 kpbl, P25 xlv, P26 cp, P27 g, P28 r_v, P29 ichoice_deep, P30 ichoice_shallow, P31 ishallow_g3, P32 htop, P33 hbot, P34 k22_shallow, P35 kbcon_shallow, P36 ktop_shallow, P37 xmb_total, P38 xmb_shallow, P39 raincv, P40 pratec, P41 gdc, P42 gdc2, P43 rthcuten, P44 rqvcuten, P45 rqccuten, P46 rqicuten, P47 ims, P48 ime, P49 jms, P50 jme, P51 kms, P52 kme, P53 ids, P54 ide, P55 jds, P56 jde, P57 kds, P58 kde, P59 its, P60 ite, P61 jts, P62 jte, P63 kts, P64 kte) {
  GFFrame frame(workspace);

  device void * D_14173;

  device void * D_14177;

  device void * D_14181;

  device void * D_14185;

  device void * D_14189;

  device void * D_14193;

  device void * D_14197;

  device void * D_14201;

  device void * D_14205;

  device void * D_14209;

  device void * D_14213;

  device void * D_14217;

  device void * D_14221;

  device void * D_14225;

  device void * D_14229;

  device void * D_14233;

  device void * D_14237;

  device void * D_14241;

  device void * D_14245;

  device void * D_14249;

  device void * D_14253;

  device void * D_14257;

  device void * D_14261;

  device void * D_14265;

  device void * D_14269;

  device void * D_14273;

  device void * D_14277;

  device void * D_14281;

  device void * D_14285;

  device void * D_14289;

  device void * D_14293;

  device void * D_14297;

  device void * D_14301;

  device void * D_14305;

  device void * D_14309;

  device void * D_14313;

  device void * D_14317;

  device void * D_14321;

  device void * D_14325;

  device void * D_14329;

  device void * D_14333;

  device void * D_14337;

  device void * D_14341;

  device void * D_14345;

  device void * D_14349;

  device void * D_14353;

  device void * D_14357;

  device void * D_14361;

  device void * D_14365;

  device void * D_14369;

  device void * D_14373;

  device void * D_14377;

  device void * D_14381;

  device void * D_14385;

  device void * D_14389;

  device void * D_14393;

  device void * D_14397;

  device void * D_14401;

  device void * D_14405;

  device void * D_14409;

  device void * D_14413;

  device void * D_14417;

  device void * D_14421;

  device void * D_14425;

  device void * D_14429;

  device void * D_14433;
  long lbound_3197;
  long ubound_3198;
  long offset_3199;
  long size_3200;

  device GFReal * aaeq;
  long lbound_3201;
  long ubound_3202;
  long lbound_3203;
  long ubound_3204;
  long stride_3205;
  long offset_3206;
  long size_3207;

  device GFReal * apri_as;
  long lbound_3208;
  long ubound_3209;
  long lbound_3210;
  long ubound_3211;
  long stride_3212;
  long offset_3213;
  long size_3214;

  device GFReal * apri_capma;
  long lbound_3215;
  long ubound_3216;
  long lbound_3217;
  long ubound_3218;
  long stride_3219;
  long offset_3220;
  long size_3221;

  device GFReal * apri_capme;
  long lbound_3222;
  long ubound_3223;
  long lbound_3224;
  long ubound_3225;
  long stride_3226;
  long offset_3227;
  long size_3228;

  device GFReal * apri_capmi;
  long lbound_3229;
  long ubound_3230;
  long lbound_3231;
  long ubound_3232;
  long stride_3233;
  long offset_3234;
  long size_3235;

  device GFReal * apri_gr;
  long lbound_3236;
  long ubound_3237;
  long lbound_3238;
  long ubound_3239;
  long stride_3240;
  long offset_3241;
  long size_3242;

  device GFReal * apri_mc;
  long lbound_3243;
  long ubound_3244;
  long lbound_3245;
  long ubound_3246;
  long stride_3247;
  long offset_3248;
  long size_3249;

  device GFReal * apri_st;
  long lbound_3250;
  long ubound_3251;
  long lbound_3252;
  long ubound_3253;
  long stride_3254;
  long offset_3255;
  long size_3256;

  device GFReal * apri_w;
  long lbound_3257;
  long ubound_3258;
  long offset_3259;
  long size_3260;

  device GFReal * area_loc;
  long lbound_3261;
  long ubound_3262;
  long lbound_3263;
  long ubound_3264;
  long stride_3265;
  long offset_3266;
  long size_3267;
  long lbound_3268;
  long ubound_3269;
  long offset_3270;
  long size_3271;

  device GFReal * ccn;
  long lbound_3272;
  long ubound_3273;
  long lbound_3274;
  long ubound_3275;
  long stride_3276;
  long offset_3277;
  long size_3278;

  device GFReal * cupclw;
  long lbound_3279;
  long ubound_3280;
  long lbound_3281;
  long ubound_3282;
  long stride_3283;
  long offset_3284;
  long size_3285;

  device GFReal * cupclws;
  long lbound_3286;
  long ubound_3287;
  long offset_3288;
  long size_3289;

  device GFReal * cuten;
  long lbound_3290;
  long ubound_3291;
  long lbound_3292;
  long ubound_3293;
  long stride_3294;
  long offset_3295;
  long size_3296;

  device GFReal * dhdt;
  GFReal dp;
  GFReal dq;
  long lbound_3297;
  long ubound_3298;
  long offset_3299;
  long size_3300;

  device GFReal * dx_loc;
  long lbound_3301;
  long ubound_3302;
  long lbound_3303;
  long ubound_3304;
  long stride_3305;
  long offset_3306;
  long size_3307;
  long lbound_3308;
  long ubound_3309;
  long lbound_3310;
  long ubound_3311;
  long stride_3312;
  long lbound_3313;
  long ubound_3314;
  long stride_3315;
  long offset_3316;
  long size_3317;
  long lbound_3318;
  long ubound_3319;
  long lbound_3320;
  long ubound_3321;
  long stride_3322;
  long offset_3323;
  long size_3324;

  device GFReal * edti_out;
  long lbound_3325;
  long ubound_3326;
  long lbound_3327;
  long ubound_3328;
  long stride_3329;
  long lbound_3330;
  long ubound_3331;
  long stride_3332;
  long offset_3333;
  long size_3334;
  long lbound_3335;
  long ubound_3336;
  long lbound_3337;
  long ubound_3338;
  long stride_3339;
  long lbound_3340;
  long ubound_3341;
  long stride_3342;
  long offset_3343;
  long size_3344;
  long lbound_3345;
  long ubound_3346;
  long lbound_3347;
  long ubound_3348;
  long stride_3349;
  long offset_3350;
  long size_3351;
  long lbound_3352;
  long ubound_3353;
  long lbound_3354;
  long ubound_3355;
  long stride_3356;
  long offset_3357;
  long size_3358;

  device GFReal * gswi;
  long lbound_3359;
  long ubound_3360;
  long lbound_3361;
  long ubound_3362;
  long stride_3363;
  long offset_3364;
  long size_3365;
  long lbound_3366;
  long ubound_3367;
  long lbound_3368;
  long ubound_3369;
  long stride_3370;
  long offset_3371;
  long size_3372;
  long lbound_3373;
  long ubound_3374;
  long lbound_3375;
  long ubound_3376;
  long stride_3377;
  long offset_3378;
  long size_3379;
  long lbound_3380;
  long ubound_3381;
  long lbound_3382;
  long ubound_3383;
  long stride_3384;
  long offset_3385;
  long size_3386;
  int i;
  long lbound_3387;
  long ubound_3388;
  long offset_3389;
  long size_3390;

  device int * ierr;
  long lbound_3391;
  long ubound_3392;
  long offset_3393;
  long size_3394;

  device GFText * ierrc;
  long lbound_3395;
  long ubound_3396;
  long offset_3397;
  long size_3398;

  device GFText * ierrcs;
  long lbound_3399;
  long ubound_3400;
  long offset_3401;
  long size_3402;

  device int * ierrs;
  int ipr;
  int itf;
  int j;
  int jpr;
  int jtf;
  int k;
  long lbound_3403;
  long ubound_3404;
  long lbound_3405;
  long ubound_3406;
  long stride_3407;
  long offset_3408;
  long size_3409;
  long lbound_3410;
  long ubound_3411;
  long offset_3412;
  long size_3413;

  device int * k22s;
  long lbound_3414;
  long ubound_3415;
  long offset_3416;
  long size_3417;

  device int * kbcon;
  long lbound_3418;
  long ubound_3419;
  long lbound_3420;
  long ubound_3421;
  long stride_3422;
  long offset_3423;
  long size_3424;
  long lbound_3425;
  long ubound_3426;
  long offset_3427;
  long size_3428;

  device int * kbcons;
  long lbound_3429;
  long ubound_3430;
  long lbound_3431;
  long ubound_3432;
  long stride_3433;
  long offset_3434;
  long size_3435;
  long lbound_3436;
  long ubound_3437;
  long offset_3438;
  long size_3439;

  device int * kpbli;
  int ktf;
  long lbound_3440;
  long ubound_3441;
  long offset_3442;
  long size_3443;

  device int * ktop;
  long lbound_3444;
  long ubound_3445;
  long lbound_3446;
  long ubound_3447;
  long stride_3448;
  long offset_3449;
  long size_3450;
  long lbound_3451;
  long ubound_3452;
  long offset_3453;
  long size_3454;

  device int * ktops;
  long lbound_3455;
  long ubound_3456;
  long stride_3457;
  long offset_3458;
  long size_3459;

  device GFReal * mconv;
  int n;
  long lbound_3460;
  long ubound_3461;
  long lbound_3462;
  long ubound_3463;
  long stride_3464;
  long stride_3465;
  long offset_3466;
  long size_3467;

  device GFReal * omeg;
  long lbound_3468;
  long ubound_3469;
  long lbound_3470;
  long ubound_3471;
  long stride_3472;
  long offset_3473;
  long size_3474;

  device GFReal * outq;
  long lbound_3475;
  long ubound_3476;
  long lbound_3477;
  long ubound_3478;
  long stride_3479;
  long offset_3480;
  long size_3481;

  device GFReal * outqc;
  long lbound_3482;
  long ubound_3483;
  long lbound_3484;
  long ubound_3485;
  long stride_3486;
  long offset_3487;
  long size_3488;

  device GFReal * outqcs;
  long lbound_3489;
  long ubound_3490;
  long lbound_3491;
  long ubound_3492;
  long stride_3493;
  long offset_3494;
  long size_3495;

  device GFReal * outqs;
  long lbound_3496;
  long ubound_3497;
  long lbound_3498;
  long ubound_3499;
  long stride_3500;
  long offset_3501;
  long size_3502;

  device GFReal * outt;
  long lbound_3503;
  long ubound_3504;
  long lbound_3505;
  long ubound_3506;
  long stride_3507;
  long offset_3508;
  long size_3509;

  device GFReal * outts;
  long lbound_3510;
  long ubound_3511;
  long lbound_3512;
  long ubound_3513;
  long stride_3514;
  long lbound_3515;
  long ubound_3516;
  long stride_3517;
  long offset_3518;
  long size_3519;
  long lbound_3520;
  long ubound_3521;
  long lbound_3522;
  long ubound_3523;
  long stride_3524;
  long offset_3525;
  long size_3526;

  device GFReal * p2d;
  long lbound_3527;
  long ubound_3528;
  long lbound_3529;
  long ubound_3530;
  long stride_3531;
  long lbound_3532;
  long ubound_3533;
  long stride_3534;
  long offset_3535;
  long size_3536;
  GFReal pahfs;
  GFReal pgeoh;
  long lbound_3537;
  long ubound_3538;
  long lbound_3539;
  long ubound_3540;
  long stride_3541;
  long offset_3542;
  long size_3543;

  device GFReal * phh;
  long lbound_3544;
  long ubound_3545;
  long lbound_3546;
  long ubound_3547;
  long stride_3548;
  long lbound_3549;
  long ubound_3550;
  long stride_3551;
  long offset_3552;
  long size_3553;
  long lbound_3554;
  long ubound_3555;
  long offset_3556;
  long size_3557;

  device GFReal * pmean;
  long lbound_3558;
  long ubound_3559;
  long lbound_3560;
  long ubound_3561;
  long stride_3562;
  long offset_3563;
  long size_3564;

  device GFReal * po;
  GFReal pqhfl;
  long lbound_3565;
  long ubound_3566;
  long lbound_3567;
  long ubound_3568;
  long stride_3569;
  long offset_3570;
  long size_3571;
  long lbound_3572;
  long ubound_3573;
  long offset_3574;
  long size_3575;

  device GFReal * pret;
  long lbound_3576;
  long ubound_3577;
  long lbound_3578;
  long ubound_3579;
  long stride_3580;
  long stride_3581;
  long offset_3582;
  long size_3583;

  device GFReal * pri_ens;
  long lbound_3584;
  long ubound_3585;
  long offset_3586;
  long size_3587;

  device GFReal * psur;
  long lbound_3588;
  long ubound_3589;
  long lbound_3590;
  long ubound_3591;
  long stride_3592;
  long lbound_3593;
  long ubound_3594;
  long stride_3595;
  long offset_3596;
  long size_3597;
  long lbound_3598;
  long ubound_3599;
  long lbound_3600;
  long ubound_3601;
  long stride_3602;
  long offset_3603;
  long size_3604;

  device GFReal * q2d;
  long lbound_3605;
  long ubound_3606;
  long lbound_3607;
  long ubound_3608;
  long stride_3609;
  long offset_3610;
  long size_3611;
  long lbound_3612;
  long ubound_3613;
  long lbound_3614;
  long ubound_3615;
  long stride_3616;
  long offset_3617;
  long size_3618;

  device GFReal * qo;
  long lbound_3619;
  long ubound_3620;
  long lbound_3621;
  long ubound_3622;
  long stride_3623;
  long offset_3624;
  long size_3625;

  device GFReal * qshall;
  long lbound_3626;
  long ubound_3627;
  long lbound_3628;
  long ubound_3629;
  long stride_3630;
  long offset_3631;
  long size_3632;
  long lbound_3633;
  long ubound_3634;
  long lbound_3635;
  long ubound_3636;
  long stride_3637;
  long lbound_3638;
  long ubound_3639;
  long stride_3640;
  long offset_3641;
  long size_3642;
  long lbound_3643;
  long ubound_3644;
  long lbound_3645;
  long ubound_3646;
  long stride_3647;
  long offset_3648;
  long size_3649;

  device GFReal * rhoi;
  long lbound_3650;
  long ubound_3651;
  long lbound_3652;
  long ubound_3653;
  long stride_3654;
  long lbound_3655;
  long ubound_3656;
  long stride_3657;
  long offset_3658;
  long size_3659;
  long lbound_3660;
  long ubound_3661;
  long lbound_3662;
  long ubound_3663;
  long stride_3664;
  long lbound_3665;
  long ubound_3666;
  long stride_3667;
  long offset_3668;
  long size_3669;
  long lbound_3670;
  long ubound_3671;
  long lbound_3672;
  long ubound_3673;
  long stride_3674;
  long lbound_3675;
  long ubound_3676;
  long stride_3677;
  long offset_3678;
  long size_3679;
  long lbound_3680;
  long ubound_3681;
  long lbound_3682;
  long ubound_3683;
  long stride_3684;
  long lbound_3685;
  long ubound_3686;
  long stride_3687;
  long offset_3688;
  long size_3689;
  long lbound_3690;
  long ubound_3691;
  long lbound_3692;
  long ubound_3693;
  long stride_3694;
  long lbound_3695;
  long ubound_3696;
  long stride_3697;
  long offset_3698;
  long size_3699;
  long lbound_3700;
  long ubound_3701;
  long lbound_3702;
  long ubound_3703;
  long stride_3704;
  long lbound_3705;
  long ubound_3706;
  long stride_3707;
  long offset_3708;
  long size_3709;
  long lbound_3710;
  long ubound_3711;
  long lbound_3712;
  long ubound_3713;
  long stride_3714;
  long lbound_3715;
  long ubound_3716;
  long stride_3717;
  long offset_3718;
  long size_3719;
  long lbound_3720;
  long ubound_3721;
  long lbound_3722;
  long ubound_3723;
  long stride_3724;
  long lbound_3725;
  long ubound_3726;
  long stride_3727;
  long offset_3728;
  long size_3729;
  long lbound_3730;
  long ubound_3731;
  long lbound_3732;
  long ubound_3733;
  long stride_3734;
  long lbound_3735;
  long ubound_3736;
  long stride_3737;
  long offset_3738;
  long size_3739;
  long lbound_3740;
  long ubound_3741;
  long lbound_3742;
  long ubound_3743;
  long stride_3744;
  long offset_3745;
  long size_3746;

  device GFReal * subm;
  long lbound_3747;
  long ubound_3748;
  long lbound_3749;
  long ubound_3750;
  long stride_3751;
  long offset_3752;
  long size_3753;

  device GFReal * subq;
  long lbound_3754;
  long ubound_3755;
  long lbound_3756;
  long ubound_3757;
  long stride_3758;
  long offset_3759;
  long size_3760;

  device GFReal * subt;
  long lbound_3761;
  long ubound_3762;
  long lbound_3763;
  long ubound_3764;
  long stride_3765;
  long lbound_3766;
  long ubound_3767;
  long stride_3768;
  long offset_3769;
  long size_3770;
  long lbound_3771;
  long ubound_3772;
  long lbound_3773;
  long ubound_3774;
  long stride_3775;
  long offset_3776;
  long size_3777;

  device GFReal * t2d;
  long lbound_3778;
  long ubound_3779;
  long offset_3780;
  long size_3781;

  device GFReal * ter11;
  long lbound_3782;
  long ubound_3783;
  long lbound_3784;
  long ubound_3785;
  long stride_3786;
  long offset_3787;
  long size_3788;

  device GFReal * tn;
  long lbound_3789;
  long ubound_3790;
  long offset_3791;
  long size_3792;

  device GFReal * tscl_kf;
  long lbound_3793;
  long ubound_3794;
  long lbound_3795;
  long ubound_3796;
  long stride_3797;
  long offset_3798;
  long size_3799;

  device GFReal * tshall;
  long lbound_3800;
  long ubound_3801;
  long lbound_3802;
  long ubound_3803;
  long stride_3804;
  long lbound_3805;
  long ubound_3806;
  long stride_3807;
  long offset_3808;
  long size_3809;
  long lbound_3810;
  long ubound_3811;
  long offset_3812;
  long size_3813;

  device GFReal * umean;
  long lbound_3814;
  long ubound_3815;
  long lbound_3816;
  long ubound_3817;
  long stride_3818;
  long offset_3819;
  long size_3820;

  device GFReal * us;
  long lbound_3821;
  long ubound_3822;
  long lbound_3823;
  long ubound_3824;
  long stride_3825;
  long lbound_3826;
  long ubound_3827;
  long stride_3828;
  long offset_3829;
  long size_3830;
  long lbound_3831;
  long ubound_3832;
  long offset_3833;
  long size_3834;

  device GFReal * vmean;
  long lbound_3835;
  long ubound_3836;
  long lbound_3837;
  long ubound_3838;
  long stride_3839;
  long offset_3840;
  long size_3841;

  device GFReal * vs;
  long lbound_3842;
  long ubound_3843;
  long lbound_3844;
  long ubound_3845;
  long stride_3846;
  long lbound_3847;
  long ubound_3848;
  long stride_3849;
  long offset_3850;
  long size_3851;
  long lbound_3852;
  long ubound_3853;
  long lbound_3854;
  long ubound_3855;
  long stride_3856;
  long stride_3857;
  long offset_3858;
  long size_3859;

  device GFReal * xfi_ens;
  long lbound_3860;
  long ubound_3861;
  long lbound_3862;
  long ubound_3863;
  long stride_3864;
  long offset_3865;
  long size_3866;
  long lbound_3867;
  long ubound_3868;
  long offset_3869;
  long size_3870;

  device GFReal * xlandi;
  long lbound_3871;
  long ubound_3872;
  long offset_3873;
  long size_3874;

  device GFReal * xmb;
  long lbound_3875;
  long ubound_3876;
  long lbound_3877;
  long ubound_3878;
  long stride_3879;
  long offset_3880;
  long size_3881;
  long lbound_3882;
  long ubound_3883;
  long lbound_3884;
  long ubound_3885;
  long stride_3886;
  long offset_3887;
  long size_3888;
  long lbound_3889;
  long ubound_3890;
  long offset_3891;
  long size_3892;

  device GFReal * xmbs;
  GFReal zkhvfl;
  long lbound_3893;
  long ubound_3894;
  long lbound_3895;
  long ubound_3896;
  long stride_3897;
  long offset_3898;
  long size_3899;

  device GFReal * zo;
  long lbound_3900;
  long ubound_3901;
  long offset_3902;
  long size_3903;

  device GFReal * zqexec;
  GFReal zrho;
  long lbound_3904;
  long ubound_3905;
  long offset_3906;
  long size_3907;

  device GFReal * ztexec;
  GFReal zws;
  long D_14170;
  ulong D_14171;
  ulong D_14172;
  long D_14174;
  ulong D_14175;
  ulong D_14176;
  long D_14178;
  ulong D_14179;
  ulong D_14180;
  long D_14182;
  ulong D_14183;
  ulong D_14184;
  long D_14186;
  ulong D_14187;
  ulong D_14188;
  long D_14190;
  ulong D_14191;
  ulong D_14192;
  long D_14194;
  ulong D_14195;
  ulong D_14196;
  long D_14198;
  ulong D_14199;
  ulong D_14200;
  long D_14202;
  ulong D_14203;
  ulong D_14204;
  long D_14206;
  ulong D_14207;
  ulong D_14208;
  long D_14210;
  ulong D_14211;
  ulong D_14212;
  long D_14214;
  ulong D_14215;
  ulong D_14216;
  long D_14218;
  ulong D_14219;
  ulong D_14220;
  long D_14222;
  ulong D_14223;
  ulong D_14224;
  long D_14226;
  ulong D_14227;
  ulong D_14228;
  long D_14230;
  ulong D_14231;
  ulong D_14232;
  long D_14234;
  ulong D_14235;
  ulong D_14236;
  long D_14238;
  ulong D_14239;
  ulong D_14240;
  long D_14242;
  ulong D_14243;
  ulong D_14244;
  long D_14246;
  ulong D_14247;
  ulong D_14248;
  long D_14250;
  ulong D_14251;
  ulong D_14252;
  long D_14254;
  ulong D_14255;
  ulong D_14256;
  long D_14258;
  ulong D_14259;
  ulong D_14260;
  long D_14262;
  ulong D_14263;
  ulong D_14264;
  long D_14266;
  ulong D_14267;
  ulong D_14268;
  long D_14270;
  ulong D_14271;
  ulong D_14272;
  long D_14274;
  ulong D_14275;
  ulong D_14276;
  long D_14278;
  ulong D_14279;
  ulong D_14280;
  long D_14282;
  ulong D_14283;
  ulong D_14284;
  long D_14286;
  ulong D_14287;
  ulong D_14288;
  long D_14290;
  ulong D_14291;
  ulong D_14292;
  long D_14294;
  ulong D_14295;
  ulong D_14296;
  long D_14298;
  ulong D_14299;
  ulong D_14300;
  long D_14302;
  ulong D_14303;
  ulong D_14304;
  long D_14306;
  ulong D_14307;
  ulong D_14308;
  long D_14310;
  ulong D_14311;
  ulong D_14312;
  long D_14314;
  ulong D_14315;
  ulong D_14316;
  long D_14318;
  ulong D_14319;
  ulong D_14320;
  long D_14322;
  ulong D_14323;
  ulong D_14324;
  long D_14326;
  ulong D_14327;
  ulong D_14328;
  long D_14330;
  ulong D_14331;
  ulong D_14332;
  long D_14334;
  ulong D_14335;
  ulong D_14336;
  long D_14338;
  ulong D_14339;
  ulong D_14340;
  long D_14342;
  ulong D_14343;
  ulong D_14344;
  long D_14346;
  ulong D_14347;
  ulong D_14348;
  long D_14350;
  ulong D_14351;
  ulong D_14352;
  long D_14354;
  ulong D_14355;
  ulong D_14356;
  long D_14358;
  ulong D_14359;
  ulong D_14360;
  long D_14362;
  ulong D_14363;
  ulong D_14364;
  long D_14366;
  ulong D_14367;
  ulong D_14368;
  long D_14370;
  ulong D_14371;
  ulong D_14372;
  long D_14374;
  ulong D_14375;
  ulong D_14376;
  long D_14378;
  ulong D_14379;
  ulong D_14380;
  long D_14382;
  ulong D_14383;
  ulong D_14384;
  long D_14386;
  ulong D_14387;
  ulong D_14388;
  long D_14390;
  ulong D_14391;
  ulong D_14392;
  long D_14394;
  ulong D_14395;
  ulong D_14396;
  long D_14398;
  ulong D_14399;
  ulong D_14400;
  long D_14402;
  ulong D_14403;
  ulong D_14404;
  long D_14406;
  ulong D_14407;
  ulong D_14408;
  long D_14410;
  ulong D_14411;
  ulong D_14412;
  long D_14414;
  ulong D_14415;
  ulong D_14416;
  long D_14418;
  ulong D_14419;
  ulong D_14420;
  long D_14422;
  ulong D_14423;
  ulong D_14424;
  long D_14426;
  ulong D_14427;
  ulong D_14428;
  long D_14430;
  ulong D_14431;
  ulong D_14432;
  long D_14434;
  ulong D_14435;
  ulong D_14436;
  long D_14437;
  ulong D_14438;
  ulong D_14439;
  long D_14440;
  ulong D_14441;
  ulong D_14442;
  long D_14443;
  ulong D_14444;
  ulong D_14445;
  long D_14446;
  ulong D_14447;
  ulong D_14448;
  long D_14449;
  ulong D_14450;
  ulong D_14451;
  long D_14452;
  ulong D_14453;
  ulong D_14454;
  long D_14455;
  ulong D_14456;
  ulong D_14457;
  long D_14458;
  ulong D_14459;
  ulong D_14460;
  long D_14461;
  ulong D_14462;
  ulong D_14463;
  long D_14464;
  ulong D_14465;
  ulong D_14466;
  long D_14467;
  ulong D_14468;
  ulong D_14469;
  long D_14470;
  ulong D_14471;
  ulong D_14472;
  long D_14473;
  ulong D_14474;
  ulong D_14475;
  long D_14476;
  ulong D_14477;
  ulong D_14478;
  long D_14479;
  ulong D_14480;
  ulong D_14481;
  long D_14482;
  ulong D_14483;
  ulong D_14484;
  long D_14485;
  ulong D_14486;
  ulong D_14487;
  long D_14488;
  ulong D_14489;
  ulong D_14490;
  long D_14491;
  ulong D_14492;
  ulong D_14493;
  long D_14494;
  ulong D_14495;
  ulong D_14496;
  long D_14497;
  ulong D_14498;
  ulong D_14499;
  long D_14500;
  ulong D_14501;
  ulong D_14502;
  long D_14503;
  ulong D_14504;
  ulong D_14505;
  long D_14506;
  ulong D_14507;
  ulong D_14508;
  long D_14509;
  ulong D_14510;
  ulong D_14511;
  long D_14512;
  ulong D_14513;
  ulong D_14514;
  long D_14515;
  ulong D_14516;
  ulong D_14517;
  long D_14518;
  ulong D_14519;
  ulong D_14520;
  long D_14521;
  ulong D_14522;
  ulong D_14523;
  long D_14524;
  ulong D_14525;
  ulong D_14526;
  long D_14527;
  ulong D_14528;
  ulong D_14529;
  long D_14530;
  ulong D_14531;
  ulong D_14532;
  long D_14533;
  ulong D_14534;
  ulong D_14535;
  long D_14536;
  ulong D_14537;
  ulong D_14538;
  long D_14539;
  ulong D_14540;
  ulong D_14541;
  long D_14542;
  ulong D_14543;
  ulong D_14544;
  long D_14545;
  ulong D_14546;
  ulong D_14547;

    {
      lbound_3359 = (long) *ims;
      ubound_3360 = (long) *ime;
      stride_3363 = (1 - lbound_3359) + ubound_3360;
      stride_3363 = gf_max(stride_3363, 0);
      lbound_3361 = (long) *jms;
      ubound_3362 = (long) *jme;
      size_3365 = ((1 - lbound_3361) + ubound_3362) * stride_3363;
      size_3365 = gf_max(size_3365, 0);
      D_14545 = size_3365 + -1;
      D_14546 = (ulong) (ulong) (size_3365) * 32;
      D_14547 = (ulong) (size_3365) * 4;
      offset_3364 = -(lbound_3359) - lbound_3361 * stride_3363;
      lbound_3380 = (long) *ims;
      ubound_3381 = (long) *ime;
      stride_3384 = (1 - lbound_3380) + ubound_3381;
      stride_3384 = gf_max(stride_3384, 0);
      lbound_3382 = (long) *jms;
      ubound_3383 = (long) *jme;
      size_3386 = ((1 - lbound_3382) + ubound_3383) * stride_3384;
      size_3386 = gf_max(size_3386, 0);
      D_14542 = size_3386 + -1;
      D_14543 = (ulong) (ulong) (size_3386) * 32;
      D_14544 = (ulong) (size_3386) * 4;
      offset_3385 = -(lbound_3380) - lbound_3382 * stride_3384;
      lbound_3882 = (long) *ims;
      ubound_3883 = (long) *ime;
      stride_3886 = (1 - lbound_3882) + ubound_3883;
      stride_3886 = gf_max(stride_3886, 0);
      lbound_3884 = (long) *jms;
      ubound_3885 = (long) *jme;
      size_3888 = ((1 - lbound_3884) + ubound_3885) * stride_3886;
      size_3888 = gf_max(size_3888, 0);
      D_14539 = size_3888 + -1;
      D_14540 = (ulong) (ulong) (size_3888) * 32;
      D_14541 = (ulong) (size_3888) * 4;
      offset_3887 = -(lbound_3882) - lbound_3884 * stride_3886;
      lbound_3626 = (long) *ims;
      ubound_3627 = (long) *ime;
      stride_3630 = (1 - lbound_3626) + ubound_3627;
      stride_3630 = gf_max(stride_3630, 0);
      lbound_3628 = (long) *jms;
      ubound_3629 = (long) *jme;
      size_3632 = ((1 - lbound_3628) + ubound_3629) * stride_3630;
      size_3632 = gf_max(size_3632, 0);
      D_14536 = size_3632 + -1;
      D_14537 = (ulong) (ulong) (size_3632) * 32;
      D_14538 = (ulong) (size_3632) * 4;
      offset_3631 = -(lbound_3626) - lbound_3628 * stride_3630;
      lbound_3565 = (long) *ims;
      ubound_3566 = (long) *ime;
      stride_3569 = (1 - lbound_3565) + ubound_3566;
      stride_3569 = gf_max(stride_3569, 0);
      lbound_3567 = (long) *jms;
      ubound_3568 = (long) *jme;
      size_3571 = ((1 - lbound_3567) + ubound_3568) * stride_3569;
      size_3571 = gf_max(size_3571, 0);
      D_14533 = size_3571 + -1;
      D_14534 = (ulong) (ulong) (size_3571) * 32;
      D_14535 = (ulong) (size_3571) * 4;
      offset_3570 = -(lbound_3565) - lbound_3567 * stride_3569;
      lbound_3403 = (long) *ims;
      ubound_3404 = (long) *ime;
      stride_3407 = (1 - lbound_3403) + ubound_3404;
      stride_3407 = gf_max(stride_3407, 0);
      lbound_3405 = (long) *jms;
      ubound_3406 = (long) *jme;
      size_3409 = ((1 - lbound_3405) + ubound_3406) * stride_3407;
      size_3409 = gf_max(size_3409, 0);
      D_14530 = size_3409 + -1;
      D_14531 = (ulong) (ulong) (size_3409) * 32;
      D_14532 = (ulong) (size_3409) * 4;
      offset_3408 = -(lbound_3403) - lbound_3405 * stride_3407;
      lbound_3418 = (long) *ims;
      ubound_3419 = (long) *ime;
      stride_3422 = (1 - lbound_3418) + ubound_3419;
      stride_3422 = gf_max(stride_3422, 0);
      lbound_3420 = (long) *jms;
      ubound_3421 = (long) *jme;
      size_3424 = ((1 - lbound_3420) + ubound_3421) * stride_3422;
      size_3424 = gf_max(size_3424, 0);
      D_14527 = size_3424 + -1;
      D_14528 = (ulong) (ulong) (size_3424) * 32;
      D_14529 = (ulong) (size_3424) * 4;
      offset_3423 = -(lbound_3418) - lbound_3420 * stride_3422;
      lbound_3444 = (long) *ims;
      ubound_3445 = (long) *ime;
      stride_3448 = (1 - lbound_3444) + ubound_3445;
      stride_3448 = gf_max(stride_3448, 0);
      lbound_3446 = (long) *jms;
      ubound_3447 = (long) *jme;
      size_3450 = ((1 - lbound_3446) + ubound_3447) * stride_3448;
      size_3450 = gf_max(size_3450, 0);
      D_14524 = size_3450 + -1;
      D_14525 = (ulong) (ulong) (size_3450) * 32;
      D_14526 = (ulong) (size_3450) * 4;
      offset_3449 = -(lbound_3444) - lbound_3446 * stride_3448;
      lbound_3875 = (long) *ims;
      ubound_3876 = (long) *ime;
      stride_3879 = (1 - lbound_3875) + ubound_3876;
      stride_3879 = gf_max(stride_3879, 0);
      lbound_3877 = (long) *jms;
      ubound_3878 = (long) *jme;
      size_3881 = ((1 - lbound_3877) + ubound_3878) * stride_3879;
      size_3881 = gf_max(size_3881, 0);
      D_14521 = size_3881 + -1;
      D_14522 = (ulong) (ulong) (size_3881) * 32;
      D_14523 = (ulong) (size_3881) * 4;
      offset_3880 = -(lbound_3875) - lbound_3877 * stride_3879;
      if (gdc != nullptr)
        {
          lbound_3325 = (long) *ims;
          ubound_3326 = (long) *ime;
          stride_3329 = (1 - lbound_3325) + ubound_3326;
          stride_3329 = gf_max(stride_3329, 0);
          lbound_3327 = (long) *kms;
          ubound_3328 = (long) *kme;
          stride_3332 = ((1 - lbound_3327) + ubound_3328) * stride_3329;
          stride_3332 = gf_max(stride_3332, 0);
          lbound_3330 = (long) *jms;
          ubound_3331 = (long) *jme;
          size_3334 = ((1 - lbound_3330) + ubound_3331) * stride_3332;
          size_3334 = gf_max(size_3334, 0);
          D_14518 = size_3334 + -1;
          D_14519 = (ulong) (ulong) (size_3334) * 32;
          D_14520 = (ulong) (size_3334) * 4;
          offset_3333 = (-(lbound_3325) - lbound_3327 * stride_3329) - lbound_3330 * stride_3332;
        }
      if (gdc2 != nullptr)
        {
          lbound_3335 = (long) *ims;
          ubound_3336 = (long) *ime;
          stride_3339 = (1 - lbound_3335) + ubound_3336;
          stride_3339 = gf_max(stride_3339, 0);
          lbound_3337 = (long) *kms;
          ubound_3338 = (long) *kme;
          stride_3342 = ((1 - lbound_3337) + ubound_3338) * stride_3339;
          stride_3342 = gf_max(stride_3342, 0);
          lbound_3340 = (long) *jms;
          ubound_3341 = (long) *jme;
          size_3344 = ((1 - lbound_3340) + ubound_3341) * stride_3342;
          size_3344 = gf_max(size_3344, 0);
          D_14515 = size_3344 + -1;
          D_14516 = (ulong) (ulong) (size_3344) * 32;
          D_14517 = (ulong) (size_3344) * 4;
          offset_3343 = (-(lbound_3335) - lbound_3337 * stride_3339) - lbound_3340 * stride_3342;
        }
      lbound_3345 = (long) *ims;
      ubound_3346 = (long) *ime;
      stride_3349 = (1 - lbound_3345) + ubound_3346;
      stride_3349 = gf_max(stride_3349, 0);
      lbound_3347 = (long) *jms;
      ubound_3348 = (long) *jme;
      size_3351 = ((1 - lbound_3347) + ubound_3348) * stride_3349;
      size_3351 = gf_max(size_3351, 0);
      D_14512 = size_3351 + -1;
      D_14513 = (ulong) (ulong) (size_3351) * 32;
      D_14514 = (ulong) (size_3351) * 4;
      offset_3350 = -(lbound_3345) - lbound_3347 * stride_3349;
      lbound_3301 = (long) *ims;
      ubound_3302 = (long) *ime;
      stride_3305 = (1 - lbound_3301) + ubound_3302;
      stride_3305 = gf_max(stride_3305, 0);
      lbound_3303 = (long) *jms;
      ubound_3304 = (long) *jme;
      size_3307 = ((1 - lbound_3303) + ubound_3304) * stride_3305;
      size_3307 = gf_max(size_3307, 0);
      D_14509 = size_3307 + -1;
      D_14510 = (ulong) (ulong) (size_3307) * 32;
      D_14511 = (ulong) (size_3307) * 4;
      offset_3306 = -(lbound_3301) - lbound_3303 * stride_3305;
      lbound_3261 = (long) *ims;
      ubound_3262 = (long) *ime;
      stride_3265 = (1 - lbound_3261) + ubound_3262;
      stride_3265 = gf_max(stride_3265, 0);
      lbound_3263 = (long) *jms;
      ubound_3264 = (long) *jme;
      size_3267 = ((1 - lbound_3263) + ubound_3264) * stride_3265;
      size_3267 = gf_max(size_3267, 0);
      D_14506 = size_3267 + -1;
      D_14507 = (ulong) (ulong) (size_3267) * 32;
      D_14508 = (ulong) (size_3267) * 4;
      offset_3266 = -(lbound_3261) - lbound_3263 * stride_3265;
      lbound_3373 = (long) *ims;
      ubound_3374 = (long) *ime;
      stride_3377 = (1 - lbound_3373) + ubound_3374;
      stride_3377 = gf_max(stride_3377, 0);
      lbound_3375 = (long) *jms;
      ubound_3376 = (long) *jme;
      size_3379 = ((1 - lbound_3375) + ubound_3376) * stride_3377;
      size_3379 = gf_max(size_3379, 0);
      D_14503 = size_3379 + -1;
      D_14504 = (ulong) (ulong) (size_3379) * 32;
      D_14505 = (ulong) (size_3379) * 4;
      offset_3378 = -(lbound_3373) - lbound_3375 * stride_3377;
      lbound_3308 = (long) *ims;
      ubound_3309 = (long) *ime;
      stride_3312 = (1 - lbound_3308) + ubound_3309;
      stride_3312 = gf_max(stride_3312, 0);
      lbound_3310 = (long) *kms;
      ubound_3311 = (long) *kme;
      stride_3315 = ((1 - lbound_3310) + ubound_3311) * stride_3312;
      stride_3315 = gf_max(stride_3315, 0);
      lbound_3313 = (long) *jms;
      ubound_3314 = (long) *jme;
      size_3317 = ((1 - lbound_3313) + ubound_3314) * stride_3315;
      size_3317 = gf_max(size_3317, 0);
      D_14500 = size_3317 + -1;
      D_14501 = (ulong) (ulong) (size_3317) * 32;
      D_14502 = (ulong) (size_3317) * 4;
      offset_3316 = (-(lbound_3308) - lbound_3310 * stride_3312) - lbound_3313 * stride_3315;
      lbound_3527 = (long) *ims;
      ubound_3528 = (long) *ime;
      stride_3531 = (1 - lbound_3527) + ubound_3528;
      stride_3531 = gf_max(stride_3531, 0);
      lbound_3529 = (long) *kms;
      ubound_3530 = (long) *kme;
      stride_3534 = ((1 - lbound_3529) + ubound_3530) * stride_3531;
      stride_3534 = gf_max(stride_3534, 0);
      lbound_3532 = (long) *jms;
      ubound_3533 = (long) *jme;
      size_3536 = ((1 - lbound_3532) + ubound_3533) * stride_3534;
      size_3536 = gf_max(size_3536, 0);
      D_14497 = size_3536 + -1;
      D_14498 = (ulong) (ulong) (size_3536) * 32;
      D_14499 = (ulong) (size_3536) * 4;
      offset_3535 = (-(lbound_3527) - lbound_3529 * stride_3531) - lbound_3532 * stride_3534;
      lbound_3429 = (long) *ims;
      ubound_3430 = (long) *ime;
      stride_3433 = (1 - lbound_3429) + ubound_3430;
      stride_3433 = gf_max(stride_3433, 0);
      lbound_3431 = (long) *jms;
      ubound_3432 = (long) *jme;
      size_3435 = ((1 - lbound_3431) + ubound_3432) * stride_3433;
      size_3435 = gf_max(size_3435, 0);
      D_14494 = size_3435 + -1;
      D_14495 = (ulong) (ulong) (size_3435) * 32;
      D_14496 = (ulong) (size_3435) * 4;
      offset_3434 = -(lbound_3429) - lbound_3431 * stride_3433;
      lbound_3860 = (long) *ims;
      ubound_3861 = (long) *ime;
      stride_3864 = (1 - lbound_3860) + ubound_3861;
      stride_3864 = gf_max(stride_3864, 0);
      lbound_3862 = (long) *jms;
      ubound_3863 = (long) *jme;
      size_3866 = ((1 - lbound_3862) + ubound_3863) * stride_3864;
      size_3866 = gf_max(size_3866, 0);
      D_14491 = size_3866 + -1;
      D_14492 = (ulong) (ulong) (size_3866) * 32;
      D_14493 = (ulong) (size_3866) * 4;
      offset_3865 = -(lbound_3860) - lbound_3862 * stride_3864;
      lbound_3800 = (long) *ims;
      ubound_3801 = (long) *ime;
      stride_3804 = (1 - lbound_3800) + ubound_3801;
      stride_3804 = gf_max(stride_3804, 0);
      lbound_3802 = (long) *kms;
      ubound_3803 = (long) *kme;
      stride_3807 = ((1 - lbound_3802) + ubound_3803) * stride_3804;
      stride_3807 = gf_max(stride_3807, 0);
      lbound_3805 = (long) *jms;
      ubound_3806 = (long) *jme;
      size_3809 = ((1 - lbound_3805) + ubound_3806) * stride_3807;
      size_3809 = gf_max(size_3809, 0);
      D_14488 = size_3809 + -1;
      D_14489 = (ulong) (ulong) (size_3809) * 32;
      D_14490 = (ulong) (size_3809) * 4;
      offset_3808 = (-(lbound_3800) - lbound_3802 * stride_3804) - lbound_3805 * stride_3807;
      lbound_3821 = (long) *ims;
      ubound_3822 = (long) *ime;
      stride_3825 = (1 - lbound_3821) + ubound_3822;
      stride_3825 = gf_max(stride_3825, 0);
      lbound_3823 = (long) *kms;
      ubound_3824 = (long) *kme;
      stride_3828 = ((1 - lbound_3823) + ubound_3824) * stride_3825;
      stride_3828 = gf_max(stride_3828, 0);
      lbound_3826 = (long) *jms;
      ubound_3827 = (long) *jme;
      size_3830 = ((1 - lbound_3826) + ubound_3827) * stride_3828;
      size_3830 = gf_max(size_3830, 0);
      D_14485 = size_3830 + -1;
      D_14486 = (ulong) (ulong) (size_3830) * 32;
      D_14487 = (ulong) (size_3830) * 4;
      offset_3829 = (-(lbound_3821) - lbound_3823 * stride_3825) - lbound_3826 * stride_3828;
      lbound_3633 = (long) *ims;
      ubound_3634 = (long) *ime;
      stride_3637 = (1 - lbound_3633) + ubound_3634;
      stride_3637 = gf_max(stride_3637, 0);
      lbound_3635 = (long) *kms;
      ubound_3636 = (long) *kme;
      stride_3640 = ((1 - lbound_3635) + ubound_3636) * stride_3637;
      stride_3640 = gf_max(stride_3640, 0);
      lbound_3638 = (long) *jms;
      ubound_3639 = (long) *jme;
      size_3642 = ((1 - lbound_3638) + ubound_3639) * stride_3640;
      size_3642 = gf_max(size_3642, 0);
      D_14482 = size_3642 + -1;
      D_14483 = (ulong) (ulong) (size_3642) * 32;
      D_14484 = (ulong) (size_3642) * 4;
      offset_3641 = (-(lbound_3633) - lbound_3635 * stride_3637) - lbound_3638 * stride_3640;
      lbound_3761 = (long) *ims;
      ubound_3762 = (long) *ime;
      stride_3765 = (1 - lbound_3761) + ubound_3762;
      stride_3765 = gf_max(stride_3765, 0);
      lbound_3763 = (long) *kms;
      ubound_3764 = (long) *kme;
      stride_3768 = ((1 - lbound_3763) + ubound_3764) * stride_3765;
      stride_3768 = gf_max(stride_3768, 0);
      lbound_3766 = (long) *jms;
      ubound_3767 = (long) *jme;
      size_3770 = ((1 - lbound_3766) + ubound_3767) * stride_3768;
      size_3770 = gf_max(size_3770, 0);
      D_14479 = size_3770 + -1;
      D_14480 = (ulong) (ulong) (size_3770) * 32;
      D_14481 = (ulong) (size_3770) * 4;
      offset_3769 = (-(lbound_3761) - lbound_3763 * stride_3765) - lbound_3766 * stride_3768;
      lbound_3588 = (long) *ims;
      ubound_3589 = (long) *ime;
      stride_3592 = (1 - lbound_3588) + ubound_3589;
      stride_3592 = gf_max(stride_3592, 0);
      lbound_3590 = (long) *kms;
      ubound_3591 = (long) *kme;
      stride_3595 = ((1 - lbound_3590) + ubound_3591) * stride_3592;
      stride_3595 = gf_max(stride_3595, 0);
      lbound_3593 = (long) *jms;
      ubound_3594 = (long) *jme;
      size_3597 = ((1 - lbound_3593) + ubound_3594) * stride_3595;
      size_3597 = gf_max(size_3597, 0);
      D_14476 = size_3597 + -1;
      D_14477 = (ulong) (ulong) (size_3597) * 32;
      D_14478 = (ulong) (size_3597) * 4;
      offset_3596 = (-(lbound_3588) - lbound_3590 * stride_3592) - lbound_3593 * stride_3595;
      if (rthften != nullptr)
        {
          lbound_3720 = (long) *ims;
          ubound_3721 = (long) *ime;
          stride_3724 = (1 - lbound_3720) + ubound_3721;
          stride_3724 = gf_max(stride_3724, 0);
          lbound_3722 = (long) *kms;
          ubound_3723 = (long) *kme;
          stride_3727 = ((1 - lbound_3722) + ubound_3723) * stride_3724;
          stride_3727 = gf_max(stride_3727, 0);
          lbound_3725 = (long) *jms;
          ubound_3726 = (long) *jme;
          size_3729 = ((1 - lbound_3725) + ubound_3726) * stride_3727;
          size_3729 = gf_max(size_3729, 0);
          D_14473 = size_3729 + -1;
          D_14474 = (ulong) (ulong) (size_3729) * 32;
          D_14475 = (ulong) (size_3729) * 4;
          offset_3728 = (-(lbound_3720) - lbound_3722 * stride_3724) - lbound_3725 * stride_3727;
        }
      lbound_3730 = (long) *ims;
      ubound_3731 = (long) *ime;
      stride_3734 = (1 - lbound_3730) + ubound_3731;
      stride_3734 = gf_max(stride_3734, 0);
      lbound_3732 = (long) *kms;
      ubound_3733 = (long) *kme;
      stride_3737 = ((1 - lbound_3732) + ubound_3733) * stride_3734;
      stride_3737 = gf_max(stride_3737, 0);
      lbound_3735 = (long) *jms;
      ubound_3736 = (long) *jme;
      size_3739 = ((1 - lbound_3735) + ubound_3736) * stride_3737;
      size_3739 = gf_max(size_3739, 0);
      D_14470 = size_3739 + -1;
      D_14471 = (ulong) (ulong) (size_3739) * 32;
      D_14472 = (ulong) (size_3739) * 4;
      offset_3738 = (-(lbound_3730) - lbound_3732 * stride_3734) - lbound_3735 * stride_3737;
      lbound_3700 = (long) *ims;
      ubound_3701 = (long) *ime;
      stride_3704 = (1 - lbound_3700) + ubound_3701;
      stride_3704 = gf_max(stride_3704, 0);
      lbound_3702 = (long) *kms;
      ubound_3703 = (long) *kme;
      stride_3707 = ((1 - lbound_3702) + ubound_3703) * stride_3704;
      stride_3707 = gf_max(stride_3707, 0);
      lbound_3705 = (long) *jms;
      ubound_3706 = (long) *jme;
      size_3709 = ((1 - lbound_3705) + ubound_3706) * stride_3707;
      size_3709 = gf_max(size_3709, 0);
      D_14467 = size_3709 + -1;
      D_14468 = (ulong) (ulong) (size_3709) * 32;
      D_14469 = (ulong) (size_3709) * 4;
      offset_3708 = (-(lbound_3700) - lbound_3702 * stride_3704) - lbound_3705 * stride_3707;
      lbound_3544 = (long) *ims;
      ubound_3545 = (long) *ime;
      stride_3548 = (1 - lbound_3544) + ubound_3545;
      stride_3548 = gf_max(stride_3548, 0);
      lbound_3546 = (long) *kms;
      ubound_3547 = (long) *kme;
      stride_3551 = ((1 - lbound_3546) + ubound_3547) * stride_3548;
      stride_3551 = gf_max(stride_3551, 0);
      lbound_3549 = (long) *jms;
      ubound_3550 = (long) *jme;
      size_3553 = ((1 - lbound_3549) + ubound_3550) * stride_3551;
      size_3553 = gf_max(size_3553, 0);
      D_14464 = size_3553 + -1;
      D_14465 = (ulong) (ulong) (size_3553) * 32;
      D_14466 = (ulong) (size_3553) * 4;
      offset_3552 = (-(lbound_3544) - lbound_3546 * stride_3548) - lbound_3549 * stride_3551;
      if (rqvften != nullptr)
        {
          lbound_3690 = (long) *ims;
          ubound_3691 = (long) *ime;
          stride_3694 = (1 - lbound_3690) + ubound_3691;
          stride_3694 = gf_max(stride_3694, 0);
          lbound_3692 = (long) *kms;
          ubound_3693 = (long) *kme;
          stride_3697 = ((1 - lbound_3692) + ubound_3693) * stride_3694;
          stride_3697 = gf_max(stride_3697, 0);
          lbound_3695 = (long) *jms;
          ubound_3696 = (long) *jme;
          size_3699 = ((1 - lbound_3695) + ubound_3696) * stride_3697;
          size_3699 = gf_max(size_3699, 0);
          D_14461 = size_3699 + -1;
          D_14462 = (ulong) (ulong) (size_3699) * 32;
          D_14463 = (ulong) (size_3699) * 4;
          offset_3698 = (-(lbound_3690) - lbound_3692 * stride_3694) - lbound_3695 * stride_3697;
        }
      lbound_3670 = (long) *ims;
      ubound_3671 = (long) *ime;
      stride_3674 = (1 - lbound_3670) + ubound_3671;
      stride_3674 = gf_max(stride_3674, 0);
      lbound_3672 = (long) *kms;
      ubound_3673 = (long) *kme;
      stride_3677 = ((1 - lbound_3672) + ubound_3673) * stride_3674;
      stride_3677 = gf_max(stride_3677, 0);
      lbound_3675 = (long) *jms;
      ubound_3676 = (long) *jme;
      size_3679 = ((1 - lbound_3675) + ubound_3676) * stride_3677;
      size_3679 = gf_max(size_3679, 0);
      D_14458 = size_3679 + -1;
      D_14459 = (ulong) (ulong) (size_3679) * 32;
      D_14460 = (ulong) (size_3679) * 4;
      offset_3678 = (-(lbound_3670) - lbound_3672 * stride_3674) - lbound_3675 * stride_3677;
      lbound_3510 = (long) *ims;
      ubound_3511 = (long) *ime;
      stride_3514 = (1 - lbound_3510) + ubound_3511;
      stride_3514 = gf_max(stride_3514, 0);
      lbound_3512 = (long) *kms;
      ubound_3513 = (long) *kme;
      stride_3517 = ((1 - lbound_3512) + ubound_3513) * stride_3514;
      stride_3517 = gf_max(stride_3517, 0);
      lbound_3515 = (long) *jms;
      ubound_3516 = (long) *jme;
      size_3519 = ((1 - lbound_3515) + ubound_3516) * stride_3517;
      size_3519 = gf_max(size_3519, 0);
      D_14455 = size_3519 + -1;
      D_14456 = (ulong) (ulong) (size_3519) * 32;
      D_14457 = (ulong) (size_3519) * 4;
      offset_3518 = (-(lbound_3510) - lbound_3512 * stride_3514) - lbound_3515 * stride_3517;
      lbound_3842 = (long) *ims;
      ubound_3843 = (long) *ime;
      stride_3846 = (1 - lbound_3842) + ubound_3843;
      stride_3846 = gf_max(stride_3846, 0);
      lbound_3844 = (long) *kms;
      ubound_3845 = (long) *kme;
      stride_3849 = ((1 - lbound_3844) + ubound_3845) * stride_3846;
      stride_3849 = gf_max(stride_3849, 0);
      lbound_3847 = (long) *jms;
      ubound_3848 = (long) *jme;
      size_3851 = ((1 - lbound_3847) + ubound_3848) * stride_3849;
      size_3851 = gf_max(size_3851, 0);
      D_14452 = size_3851 + -1;
      D_14453 = (ulong) (ulong) (size_3851) * 32;
      D_14454 = (ulong) (size_3851) * 4;
      offset_3850 = (-(lbound_3842) - lbound_3844 * stride_3846) - lbound_3847 * stride_3849;
      lbound_3366 = (long) *ims;
      ubound_3367 = (long) *ime;
      stride_3370 = (1 - lbound_3366) + ubound_3367;
      stride_3370 = gf_max(stride_3370, 0);
      lbound_3368 = (long) *jms;
      ubound_3369 = (long) *jme;
      size_3372 = ((1 - lbound_3368) + ubound_3369) * stride_3370;
      size_3372 = gf_max(size_3372, 0);
      D_14449 = size_3372 + -1;
      D_14450 = (ulong) (ulong) (size_3372) * 32;
      D_14451 = (ulong) (size_3372) * 4;
      offset_3371 = -(lbound_3366) - lbound_3368 * stride_3370;
      lbound_3605 = (long) *ims;
      ubound_3606 = (long) *ime;
      stride_3609 = (1 - lbound_3605) + ubound_3606;
      stride_3609 = gf_max(stride_3609, 0);
      lbound_3607 = (long) *jms;
      ubound_3608 = (long) *jme;
      size_3611 = ((1 - lbound_3607) + ubound_3608) * stride_3609;
      size_3611 = gf_max(size_3611, 0);
      D_14446 = size_3611 + -1;
      D_14447 = (ulong) (ulong) (size_3611) * 32;
      D_14448 = (ulong) (size_3611) * 4;
      offset_3610 = -(lbound_3605) - lbound_3607 * stride_3609;
      lbound_3710 = (long) *ims;
      ubound_3711 = (long) *ime;
      stride_3714 = (1 - lbound_3710) + ubound_3711;
      stride_3714 = gf_max(stride_3714, 0);
      lbound_3712 = (long) *kms;
      ubound_3713 = (long) *kme;
      stride_3717 = ((1 - lbound_3712) + ubound_3713) * stride_3714;
      stride_3717 = gf_max(stride_3717, 0);
      lbound_3715 = (long) *jms;
      ubound_3716 = (long) *jme;
      size_3719 = ((1 - lbound_3715) + ubound_3716) * stride_3717;
      size_3719 = gf_max(size_3719, 0);
      D_14443 = size_3719 + -1;
      D_14444 = (ulong) (ulong) (size_3719) * 32;
      D_14445 = (ulong) (size_3719) * 4;
      offset_3718 = (-(lbound_3710) - lbound_3712 * stride_3714) - lbound_3715 * stride_3717;
      lbound_3680 = (long) *ims;
      ubound_3681 = (long) *ime;
      stride_3684 = (1 - lbound_3680) + ubound_3681;
      stride_3684 = gf_max(stride_3684, 0);
      lbound_3682 = (long) *kms;
      ubound_3683 = (long) *kme;
      stride_3687 = ((1 - lbound_3682) + ubound_3683) * stride_3684;
      stride_3687 = gf_max(stride_3687, 0);
      lbound_3685 = (long) *jms;
      ubound_3686 = (long) *jme;
      size_3689 = ((1 - lbound_3685) + ubound_3686) * stride_3687;
      size_3689 = gf_max(size_3689, 0);
      D_14440 = size_3689 + -1;
      D_14441 = (ulong) (ulong) (size_3689) * 32;
      D_14442 = (ulong) (size_3689) * 4;
      offset_3688 = (-(lbound_3680) - lbound_3682 * stride_3684) - lbound_3685 * stride_3687;
      lbound_3650 = (long) *ims;
      ubound_3651 = (long) *ime;
      stride_3654 = (1 - lbound_3650) + ubound_3651;
      stride_3654 = gf_max(stride_3654, 0);
      lbound_3652 = (long) *kms;
      ubound_3653 = (long) *kme;
      stride_3657 = ((1 - lbound_3652) + ubound_3653) * stride_3654;
      stride_3657 = gf_max(stride_3657, 0);
      lbound_3655 = (long) *jms;
      ubound_3656 = (long) *jme;
      size_3659 = ((1 - lbound_3655) + ubound_3656) * stride_3657;
      size_3659 = gf_max(size_3659, 0);
      D_14437 = size_3659 + -1;
      D_14438 = (ulong) (ulong) (size_3659) * 32;
      D_14439 = (ulong) (size_3659) * 4;
      offset_3658 = (-(lbound_3650) - lbound_3652 * stride_3654) - lbound_3655 * stride_3657;
      lbound_3660 = (long) *ims;
      ubound_3661 = (long) *ime;
      stride_3664 = (1 - lbound_3660) + ubound_3661;
      stride_3664 = gf_max(stride_3664, 0);
      lbound_3662 = (long) *kms;
      ubound_3663 = (long) *kme;
      stride_3667 = ((1 - lbound_3662) + ubound_3663) * stride_3664;
      stride_3667 = gf_max(stride_3667, 0);
      lbound_3665 = (long) *jms;
      ubound_3666 = (long) *jme;
      size_3669 = ((1 - lbound_3665) + ubound_3666) * stride_3667;
      size_3669 = gf_max(size_3669, 0);
      D_14434 = size_3669 + -1;
      D_14435 = (ulong) (ulong) (size_3669) * 32;
      D_14436 = (ulong) (size_3669) * 4;
      offset_3668 = (-(lbound_3660) - lbound_3662 * stride_3664) - lbound_3665 * stride_3667;
      lbound_3197 = (long) *its;
      ubound_3198 = (long) *ite;
      size_3200 = (1 - lbound_3197) + ubound_3198;
      size_3200 = gf_max(size_3200, 0);
      D_14430 = size_3200 + -1;
      D_14431 = (ulong) (ulong) (size_3200) * 32;
      D_14432 = (ulong) (size_3200) * 4;
      D_14433 = (device void *) workspace.allocate (gf_max((ulong) (size_3200 * 4), 1));
      aaeq = (device GFReal *) D_14433;
      offset_3199 = -(lbound_3197);
      lbound_3201 = (long) *its;
      ubound_3202 = (long) *ite;
      stride_3205 = (1 - lbound_3201) + ubound_3202;
      stride_3205 = gf_max(stride_3205, 0);
      lbound_3203 = (long) *jts;
      ubound_3204 = (long) *jte;
      size_3207 = ((1 - lbound_3203) + ubound_3204) * stride_3205;
      size_3207 = gf_max(size_3207, 0);
      D_14426 = size_3207 + -1;
      D_14427 = (ulong) (ulong) (size_3207) * 32;
      D_14428 = (ulong) (size_3207) * 4;
      D_14429 = (device void *) workspace.allocate (gf_max((ulong) (size_3207 * 4), 1));
      apri_as = (device GFReal *) D_14429;
      offset_3206 = -(lbound_3201) - lbound_3203 * stride_3205;
      lbound_3208 = (long) *its;
      ubound_3209 = (long) *ite;
      stride_3212 = (1 - lbound_3208) + ubound_3209;
      stride_3212 = gf_max(stride_3212, 0);
      lbound_3210 = (long) *jts;
      ubound_3211 = (long) *jte;
      size_3214 = ((1 - lbound_3210) + ubound_3211) * stride_3212;
      size_3214 = gf_max(size_3214, 0);
      D_14422 = size_3214 + -1;
      D_14423 = (ulong) (ulong) (size_3214) * 32;
      D_14424 = (ulong) (size_3214) * 4;
      D_14425 = (device void *) workspace.allocate (gf_max((ulong) (size_3214 * 4), 1));
      apri_capma = (device GFReal *) D_14425;
      offset_3213 = -(lbound_3208) - lbound_3210 * stride_3212;
      lbound_3215 = (long) *its;
      ubound_3216 = (long) *ite;
      stride_3219 = (1 - lbound_3215) + ubound_3216;
      stride_3219 = gf_max(stride_3219, 0);
      lbound_3217 = (long) *jts;
      ubound_3218 = (long) *jte;
      size_3221 = ((1 - lbound_3217) + ubound_3218) * stride_3219;
      size_3221 = gf_max(size_3221, 0);
      D_14418 = size_3221 + -1;
      D_14419 = (ulong) (ulong) (size_3221) * 32;
      D_14420 = (ulong) (size_3221) * 4;
      D_14421 = (device void *) workspace.allocate (gf_max((ulong) (size_3221 * 4), 1));
      apri_capme = (device GFReal *) D_14421;
      offset_3220 = -(lbound_3215) - lbound_3217 * stride_3219;
      lbound_3222 = (long) *its;
      ubound_3223 = (long) *ite;
      stride_3226 = (1 - lbound_3222) + ubound_3223;
      stride_3226 = gf_max(stride_3226, 0);
      lbound_3224 = (long) *jts;
      ubound_3225 = (long) *jte;
      size_3228 = ((1 - lbound_3224) + ubound_3225) * stride_3226;
      size_3228 = gf_max(size_3228, 0);
      D_14414 = size_3228 + -1;
      D_14415 = (ulong) (ulong) (size_3228) * 32;
      D_14416 = (ulong) (size_3228) * 4;
      D_14417 = (device void *) workspace.allocate (gf_max((ulong) (size_3228 * 4), 1));
      apri_capmi = (device GFReal *) D_14417;
      offset_3227 = -(lbound_3222) - lbound_3224 * stride_3226;
      lbound_3229 = (long) *its;
      ubound_3230 = (long) *ite;
      stride_3233 = (1 - lbound_3229) + ubound_3230;
      stride_3233 = gf_max(stride_3233, 0);
      lbound_3231 = (long) *jts;
      ubound_3232 = (long) *jte;
      size_3235 = ((1 - lbound_3231) + ubound_3232) * stride_3233;
      size_3235 = gf_max(size_3235, 0);
      D_14410 = size_3235 + -1;
      D_14411 = (ulong) (ulong) (size_3235) * 32;
      D_14412 = (ulong) (size_3235) * 4;
      D_14413 = (device void *) workspace.allocate (gf_max((ulong) (size_3235 * 4), 1));
      apri_gr = (device GFReal *) D_14413;
      offset_3234 = -(lbound_3229) - lbound_3231 * stride_3233;
      lbound_3236 = (long) *its;
      ubound_3237 = (long) *ite;
      stride_3240 = (1 - lbound_3236) + ubound_3237;
      stride_3240 = gf_max(stride_3240, 0);
      lbound_3238 = (long) *jts;
      ubound_3239 = (long) *jte;
      size_3242 = ((1 - lbound_3238) + ubound_3239) * stride_3240;
      size_3242 = gf_max(size_3242, 0);
      D_14406 = size_3242 + -1;
      D_14407 = (ulong) (ulong) (size_3242) * 32;
      D_14408 = (ulong) (size_3242) * 4;
      D_14409 = (device void *) workspace.allocate (gf_max((ulong) (size_3242 * 4), 1));
      apri_mc = (device GFReal *) D_14409;
      offset_3241 = -(lbound_3236) - lbound_3238 * stride_3240;
      lbound_3243 = (long) *its;
      ubound_3244 = (long) *ite;
      stride_3247 = (1 - lbound_3243) + ubound_3244;
      stride_3247 = gf_max(stride_3247, 0);
      lbound_3245 = (long) *jts;
      ubound_3246 = (long) *jte;
      size_3249 = ((1 - lbound_3245) + ubound_3246) * stride_3247;
      size_3249 = gf_max(size_3249, 0);
      D_14402 = size_3249 + -1;
      D_14403 = (ulong) (ulong) (size_3249) * 32;
      D_14404 = (ulong) (size_3249) * 4;
      D_14405 = (device void *) workspace.allocate (gf_max((ulong) (size_3249 * 4), 1));
      apri_st = (device GFReal *) D_14405;
      offset_3248 = -(lbound_3243) - lbound_3245 * stride_3247;
      lbound_3250 = (long) *its;
      ubound_3251 = (long) *ite;
      stride_3254 = (1 - lbound_3250) + ubound_3251;
      stride_3254 = gf_max(stride_3254, 0);
      lbound_3252 = (long) *jts;
      ubound_3253 = (long) *jte;
      size_3256 = ((1 - lbound_3252) + ubound_3253) * stride_3254;
      size_3256 = gf_max(size_3256, 0);
      D_14398 = size_3256 + -1;
      D_14399 = (ulong) (ulong) (size_3256) * 32;
      D_14400 = (ulong) (size_3256) * 4;
      D_14401 = (device void *) workspace.allocate (gf_max((ulong) (size_3256 * 4), 1));
      apri_w = (device GFReal *) D_14401;
      offset_3255 = -(lbound_3250) - lbound_3252 * stride_3254;
      lbound_3257 = (long) *its;
      ubound_3258 = (long) *ite;
      size_3260 = (1 - lbound_3257) + ubound_3258;
      size_3260 = gf_max(size_3260, 0);
      D_14394 = size_3260 + -1;
      D_14395 = (ulong) (ulong) (size_3260) * 32;
      D_14396 = (ulong) (size_3260) * 4;
      D_14397 = (device void *) workspace.allocate (gf_max((ulong) (size_3260 * 4), 1));
      area_loc = (device GFReal *) D_14397;
      offset_3259 = -(lbound_3257);
      lbound_3268 = (long) *its;
      ubound_3269 = (long) *ite;
      size_3271 = (1 - lbound_3268) + ubound_3269;
      size_3271 = gf_max(size_3271, 0);
      D_14390 = size_3271 + -1;
      D_14391 = (ulong) (ulong) (size_3271) * 32;
      D_14392 = (ulong) (size_3271) * 4;
      D_14393 = (device void *) workspace.allocate (gf_max((ulong) (size_3271 * 4), 1));
      ccn = (device GFReal *) D_14393;
      offset_3270 = -(lbound_3268);
      lbound_3272 = (long) *its;
      ubound_3273 = (long) *ite;
      stride_3276 = (1 - lbound_3272) + ubound_3273;
      stride_3276 = gf_max(stride_3276, 0);
      lbound_3274 = (long) *kts;
      ubound_3275 = (long) *kte;
      size_3278 = ((1 - lbound_3274) + ubound_3275) * stride_3276;
      size_3278 = gf_max(size_3278, 0);
      D_14386 = size_3278 + -1;
      D_14387 = (ulong) (ulong) (size_3278) * 32;
      D_14388 = (ulong) (size_3278) * 4;
      D_14389 = (device void *) workspace.allocate (gf_max((ulong) (size_3278 * 4), 1));
      cupclw = (device GFReal *) D_14389;
      offset_3277 = -(lbound_3272) - lbound_3274 * stride_3276;
      lbound_3279 = (long) *its;
      ubound_3280 = (long) *ite;
      stride_3283 = (1 - lbound_3279) + ubound_3280;
      stride_3283 = gf_max(stride_3283, 0);
      lbound_3281 = (long) *kts;
      ubound_3282 = (long) *kte;
      size_3285 = ((1 - lbound_3281) + ubound_3282) * stride_3283;
      size_3285 = gf_max(size_3285, 0);
      D_14382 = size_3285 + -1;
      D_14383 = (ulong) (ulong) (size_3285) * 32;
      D_14384 = (ulong) (size_3285) * 4;
      D_14385 = (device void *) workspace.allocate (gf_max((ulong) (size_3285 * 4), 1));
      cupclws = (device GFReal *) D_14385;
      offset_3284 = -(lbound_3279) - lbound_3281 * stride_3283;
      lbound_3286 = (long) *its;
      ubound_3287 = (long) *ite;
      size_3289 = (1 - lbound_3286) + ubound_3287;
      size_3289 = gf_max(size_3289, 0);
      D_14378 = size_3289 + -1;
      D_14379 = (ulong) (ulong) (size_3289) * 32;
      D_14380 = (ulong) (size_3289) * 4;
      D_14381 = (device void *) workspace.allocate (gf_max((ulong) (size_3289 * 4), 1));
      cuten = (device GFReal *) D_14381;
      offset_3288 = -(lbound_3286);
      lbound_3290 = (long) *its;
      ubound_3291 = (long) *ite;
      stride_3294 = (1 - lbound_3290) + ubound_3291;
      stride_3294 = gf_max(stride_3294, 0);
      lbound_3292 = (long) *kts;
      ubound_3293 = (long) *kte;
      size_3296 = ((1 - lbound_3292) + ubound_3293) * stride_3294;
      size_3296 = gf_max(size_3296, 0);
      D_14374 = size_3296 + -1;
      D_14375 = (ulong) (ulong) (size_3296) * 32;
      D_14376 = (ulong) (size_3296) * 4;
      D_14377 = (device void *) workspace.allocate (gf_max((ulong) (size_3296 * 4), 1));
      dhdt = (device GFReal *) D_14377;
      offset_3295 = -(lbound_3290) - lbound_3292 * stride_3294;
      lbound_3297 = (long) *its;
      ubound_3298 = (long) *ite;
      size_3300 = (1 - lbound_3297) + ubound_3298;
      size_3300 = gf_max(size_3300, 0);
      D_14370 = size_3300 + -1;
      D_14371 = (ulong) (ulong) (size_3300) * 32;
      D_14372 = (ulong) (size_3300) * 4;
      D_14373 = (device void *) workspace.allocate (gf_max((ulong) (size_3300 * 4), 1));
      dx_loc = (device GFReal *) D_14373;
      offset_3299 = -(lbound_3297);
      lbound_3318 = (long) *its;
      ubound_3319 = (long) *ite;
      stride_3322 = (1 - lbound_3318) + ubound_3319;
      stride_3322 = gf_max(stride_3322, 0);
      lbound_3320 = (long) *jts;
      ubound_3321 = (long) *jte;
      size_3324 = ((1 - lbound_3320) + ubound_3321) * stride_3322;
      size_3324 = gf_max(size_3324, 0);
      D_14366 = size_3324 + -1;
      D_14367 = (ulong) (ulong) (size_3324) * 32;
      D_14368 = (ulong) (size_3324) * 4;
      D_14369 = (device void *) workspace.allocate (gf_max((ulong) (size_3324 * 4), 1));
      edti_out = (device GFReal *) D_14369;
      offset_3323 = -(lbound_3318) - lbound_3320 * stride_3322;
      lbound_3352 = (long) *its;
      ubound_3353 = (long) *ite;
      stride_3356 = (1 - lbound_3352) + ubound_3353;
      stride_3356 = gf_max(stride_3356, 0);
      lbound_3354 = (long) *jts;
      ubound_3355 = (long) *jte;
      size_3358 = ((1 - lbound_3354) + ubound_3355) * stride_3356;
      size_3358 = gf_max(size_3358, 0);
      D_14362 = size_3358 + -1;
      D_14363 = (ulong) (ulong) (size_3358) * 32;
      D_14364 = (ulong) (size_3358) * 4;
      D_14365 = (device void *) workspace.allocate (gf_max((ulong) (size_3358 * 4), 1));
      gswi = (device GFReal *) D_14365;
      offset_3357 = -(lbound_3352) - lbound_3354 * stride_3356;
      lbound_3387 = (long) *its;
      ubound_3388 = (long) *ite;
      size_3390 = (1 - lbound_3387) + ubound_3388;
      size_3390 = gf_max(size_3390, 0);
      D_14358 = size_3390 + -1;
      D_14359 = (ulong) (ulong) (size_3390) * 32;
      D_14360 = (ulong) (size_3390) * 4;
      D_14361 = (device void *) workspace.allocate (gf_max((ulong) (size_3390 * 4), 1));
      ierr = (device int *) D_14361;
      offset_3389 = -(lbound_3387);
      lbound_3391 = (long) *its;
      ubound_3392 = (long) *ite;
      size_3394 = (1 - lbound_3391) + ubound_3392;
      size_3394 = gf_max(size_3394, 0);
      D_14354 = size_3394 + -1;
      D_14355 = (ulong) (ulong) (size_3394) * 400;
      D_14356 = (ulong) (size_3394) * 50;
      D_14357 = (device void *) workspace.allocate (gf_max((ulong) (size_3394 * 50), 1));
      ierrc = (device GFText *) D_14357;
      offset_3393 = -(lbound_3391);
      lbound_3395 = (long) *its;
      ubound_3396 = (long) *ite;
      size_3398 = (1 - lbound_3395) + ubound_3396;
      size_3398 = gf_max(size_3398, 0);
      D_14350 = size_3398 + -1;
      D_14351 = (ulong) (ulong) (size_3398) * 400;
      D_14352 = (ulong) (size_3398) * 50;
      D_14353 = (device void *) workspace.allocate (gf_max((ulong) (size_3398 * 50), 1));
      ierrcs = (device GFText *) D_14353;
      offset_3397 = -(lbound_3395);
      lbound_3399 = (long) *its;
      ubound_3400 = (long) *ite;
      size_3402 = (1 - lbound_3399) + ubound_3400;
      size_3402 = gf_max(size_3402, 0);
      D_14346 = size_3402 + -1;
      D_14347 = (ulong) (ulong) (size_3402) * 32;
      D_14348 = (ulong) (size_3402) * 4;
      D_14349 = (device void *) workspace.allocate (gf_max((ulong) (size_3402 * 4), 1));
      ierrs = (device int *) D_14349;
      offset_3401 = -(lbound_3399);
      lbound_3410 = (long) *its;
      ubound_3411 = (long) *ite;
      size_3413 = (1 - lbound_3410) + ubound_3411;
      size_3413 = gf_max(size_3413, 0);
      D_14342 = size_3413 + -1;
      D_14343 = (ulong) (ulong) (size_3413) * 32;
      D_14344 = (ulong) (size_3413) * 4;
      D_14345 = (device void *) workspace.allocate (gf_max((ulong) (size_3413 * 4), 1));
      k22s = (device int *) D_14345;
      offset_3412 = -(lbound_3410);
      lbound_3414 = (long) *its;
      ubound_3415 = (long) *ite;
      size_3417 = (1 - lbound_3414) + ubound_3415;
      size_3417 = gf_max(size_3417, 0);
      D_14338 = size_3417 + -1;
      D_14339 = (ulong) (ulong) (size_3417) * 32;
      D_14340 = (ulong) (size_3417) * 4;
      D_14341 = (device void *) workspace.allocate (gf_max((ulong) (size_3417 * 4), 1));
      kbcon = (device int *) D_14341;
      offset_3416 = -(lbound_3414);
      lbound_3425 = (long) *its;
      ubound_3426 = (long) *ite;
      size_3428 = (1 - lbound_3425) + ubound_3426;
      size_3428 = gf_max(size_3428, 0);
      D_14334 = size_3428 + -1;
      D_14335 = (ulong) (ulong) (size_3428) * 32;
      D_14336 = (ulong) (size_3428) * 4;
      D_14337 = (device void *) workspace.allocate (gf_max((ulong) (size_3428 * 4), 1));
      kbcons = (device int *) D_14337;
      offset_3427 = -(lbound_3425);
      lbound_3436 = (long) *its;
      ubound_3437 = (long) *ite;
      size_3439 = (1 - lbound_3436) + ubound_3437;
      size_3439 = gf_max(size_3439, 0);
      D_14330 = size_3439 + -1;
      D_14331 = (ulong) (ulong) (size_3439) * 32;
      D_14332 = (ulong) (size_3439) * 4;
      D_14333 = (device void *) workspace.allocate (gf_max((ulong) (size_3439 * 4), 1));
      kpbli = (device int *) D_14333;
      offset_3438 = -(lbound_3436);
      lbound_3440 = (long) *its;
      ubound_3441 = (long) *ite;
      size_3443 = (1 - lbound_3440) + ubound_3441;
      size_3443 = gf_max(size_3443, 0);
      D_14326 = size_3443 + -1;
      D_14327 = (ulong) (ulong) (size_3443) * 32;
      D_14328 = (ulong) (size_3443) * 4;
      D_14329 = (device void *) workspace.allocate (gf_max((ulong) (size_3443 * 4), 1));
      ktop = (device int *) D_14329;
      offset_3442 = -(lbound_3440);
      lbound_3451 = (long) *its;
      ubound_3452 = (long) *ite;
      size_3454 = (1 - lbound_3451) + ubound_3452;
      size_3454 = gf_max(size_3454, 0);
      D_14322 = size_3454 + -1;
      D_14323 = (ulong) (ulong) (size_3454) * 32;
      D_14324 = (ulong) (size_3454) * 4;
      D_14325 = (device void *) workspace.allocate (gf_max((ulong) (size_3454 * 4), 1));
      ktops = (device int *) D_14325;
      offset_3453 = -(lbound_3451);
      lbound_3455 = (long) *its;
      ubound_3456 = (long) *ite;
      stride_3457 = (1 - lbound_3455) + ubound_3456;
      stride_3457 = gf_max(stride_3457, 0);
      size_3459 = stride_3457 * 9;
      size_3459 = gf_max(size_3459, 0);
      D_14318 = size_3459 + -1;
      D_14319 = (ulong) (ulong) (size_3459) * 32;
      D_14320 = (ulong) (size_3459) * 4;
      D_14321 = (device void *) workspace.allocate (gf_max((ulong) (size_3459 * 4), 1));
      mconv = (device GFReal *) D_14321;
      offset_3458 = -(lbound_3455) - (stride_3457);
      lbound_3460 = (long) *its;
      ubound_3461 = (long) *ite;
      stride_3464 = (1 - lbound_3460) + ubound_3461;
      stride_3464 = gf_max(stride_3464, 0);
      lbound_3462 = (long) *kts;
      ubound_3463 = (long) *kte;
      stride_3465 = ((1 - lbound_3462) + ubound_3463) * stride_3464;
      stride_3465 = gf_max(stride_3465, 0);
      size_3467 = stride_3465 * 9;
      size_3467 = gf_max(size_3467, 0);
      D_14314 = size_3467 + -1;
      D_14315 = (ulong) (ulong) (size_3467) * 32;
      D_14316 = (ulong) (size_3467) * 4;
      D_14317 = (device void *) workspace.allocate (gf_max((ulong) (size_3467 * 4), 1));
      omeg = (device GFReal *) D_14317;
      offset_3466 = (-(lbound_3460) - lbound_3462 * stride_3464) - (stride_3465);
      lbound_3468 = (long) *its;
      ubound_3469 = (long) *ite;
      stride_3472 = (1 - lbound_3468) + ubound_3469;
      stride_3472 = gf_max(stride_3472, 0);
      lbound_3470 = (long) *kts;
      ubound_3471 = (long) *kte;
      size_3474 = ((1 - lbound_3470) + ubound_3471) * stride_3472;
      size_3474 = gf_max(size_3474, 0);
      D_14310 = size_3474 + -1;
      D_14311 = (ulong) (ulong) (size_3474) * 32;
      D_14312 = (ulong) (size_3474) * 4;
      D_14313 = (device void *) workspace.allocate (gf_max((ulong) (size_3474 * 4), 1));
      outq = (device GFReal *) D_14313;
      offset_3473 = -(lbound_3468) - lbound_3470 * stride_3472;
      lbound_3475 = (long) *its;
      ubound_3476 = (long) *ite;
      stride_3479 = (1 - lbound_3475) + ubound_3476;
      stride_3479 = gf_max(stride_3479, 0);
      lbound_3477 = (long) *kts;
      ubound_3478 = (long) *kte;
      size_3481 = ((1 - lbound_3477) + ubound_3478) * stride_3479;
      size_3481 = gf_max(size_3481, 0);
      D_14306 = size_3481 + -1;
      D_14307 = (ulong) (ulong) (size_3481) * 32;
      D_14308 = (ulong) (size_3481) * 4;
      D_14309 = (device void *) workspace.allocate (gf_max((ulong) (size_3481 * 4), 1));
      outqc = (device GFReal *) D_14309;
      offset_3480 = -(lbound_3475) - lbound_3477 * stride_3479;
      lbound_3482 = (long) *its;
      ubound_3483 = (long) *ite;
      stride_3486 = (1 - lbound_3482) + ubound_3483;
      stride_3486 = gf_max(stride_3486, 0);
      lbound_3484 = (long) *kts;
      ubound_3485 = (long) *kte;
      size_3488 = ((1 - lbound_3484) + ubound_3485) * stride_3486;
      size_3488 = gf_max(size_3488, 0);
      D_14302 = size_3488 + -1;
      D_14303 = (ulong) (ulong) (size_3488) * 32;
      D_14304 = (ulong) (size_3488) * 4;
      D_14305 = (device void *) workspace.allocate (gf_max((ulong) (size_3488 * 4), 1));
      outqcs = (device GFReal *) D_14305;
      offset_3487 = -(lbound_3482) - lbound_3484 * stride_3486;
      lbound_3489 = (long) *its;
      ubound_3490 = (long) *ite;
      stride_3493 = (1 - lbound_3489) + ubound_3490;
      stride_3493 = gf_max(stride_3493, 0);
      lbound_3491 = (long) *kts;
      ubound_3492 = (long) *kte;
      size_3495 = ((1 - lbound_3491) + ubound_3492) * stride_3493;
      size_3495 = gf_max(size_3495, 0);
      D_14298 = size_3495 + -1;
      D_14299 = (ulong) (ulong) (size_3495) * 32;
      D_14300 = (ulong) (size_3495) * 4;
      D_14301 = (device void *) workspace.allocate (gf_max((ulong) (size_3495 * 4), 1));
      outqs = (device GFReal *) D_14301;
      offset_3494 = -(lbound_3489) - lbound_3491 * stride_3493;
      lbound_3496 = (long) *its;
      ubound_3497 = (long) *ite;
      stride_3500 = (1 - lbound_3496) + ubound_3497;
      stride_3500 = gf_max(stride_3500, 0);
      lbound_3498 = (long) *kts;
      ubound_3499 = (long) *kte;
      size_3502 = ((1 - lbound_3498) + ubound_3499) * stride_3500;
      size_3502 = gf_max(size_3502, 0);
      D_14294 = size_3502 + -1;
      D_14295 = (ulong) (ulong) (size_3502) * 32;
      D_14296 = (ulong) (size_3502) * 4;
      D_14297 = (device void *) workspace.allocate (gf_max((ulong) (size_3502 * 4), 1));
      outt = (device GFReal *) D_14297;
      offset_3501 = -(lbound_3496) - lbound_3498 * stride_3500;
      lbound_3503 = (long) *its;
      ubound_3504 = (long) *ite;
      stride_3507 = (1 - lbound_3503) + ubound_3504;
      stride_3507 = gf_max(stride_3507, 0);
      lbound_3505 = (long) *kts;
      ubound_3506 = (long) *kte;
      size_3509 = ((1 - lbound_3505) + ubound_3506) * stride_3507;
      size_3509 = gf_max(size_3509, 0);
      D_14290 = size_3509 + -1;
      D_14291 = (ulong) (ulong) (size_3509) * 32;
      D_14292 = (ulong) (size_3509) * 4;
      D_14293 = (device void *) workspace.allocate (gf_max((ulong) (size_3509 * 4), 1));
      outts = (device GFReal *) D_14293;
      offset_3508 = -(lbound_3503) - lbound_3505 * stride_3507;
      lbound_3520 = (long) *its;
      ubound_3521 = (long) *ite;
      stride_3524 = (1 - lbound_3520) + ubound_3521;
      stride_3524 = gf_max(stride_3524, 0);
      lbound_3522 = (long) *kts;
      ubound_3523 = (long) *kte;
      size_3526 = ((1 - lbound_3522) + ubound_3523) * stride_3524;
      size_3526 = gf_max(size_3526, 0);
      D_14286 = size_3526 + -1;
      D_14287 = (ulong) (ulong) (size_3526) * 32;
      D_14288 = (ulong) (size_3526) * 4;
      D_14289 = (device void *) workspace.allocate (gf_max((ulong) (size_3526 * 4), 1));
      p2d = (device GFReal *) D_14289;
      offset_3525 = -(lbound_3520) - lbound_3522 * stride_3524;
      lbound_3537 = (long) *its;
      ubound_3538 = (long) *ite;
      stride_3541 = (1 - lbound_3537) + ubound_3538;
      stride_3541 = gf_max(stride_3541, 0);
      lbound_3539 = (long) *kts;
      ubound_3540 = (long) *kte;
      size_3543 = ((1 - lbound_3539) + ubound_3540) * stride_3541;
      size_3543 = gf_max(size_3543, 0);
      D_14282 = size_3543 + -1;
      D_14283 = (ulong) (ulong) (size_3543) * 32;
      D_14284 = (ulong) (size_3543) * 4;
      D_14285 = (device void *) workspace.allocate (gf_max((ulong) (size_3543 * 4), 1));
      phh = (device GFReal *) D_14285;
      offset_3542 = -(lbound_3537) - lbound_3539 * stride_3541;
      lbound_3554 = (long) *its;
      ubound_3555 = (long) *ite;
      size_3557 = (1 - lbound_3554) + ubound_3555;
      size_3557 = gf_max(size_3557, 0);
      D_14278 = size_3557 + -1;
      D_14279 = (ulong) (ulong) (size_3557) * 32;
      D_14280 = (ulong) (size_3557) * 4;
      D_14281 = (device void *) workspace.allocate (gf_max((ulong) (size_3557 * 4), 1));
      pmean = (device GFReal *) D_14281;
      offset_3556 = -(lbound_3554);
      lbound_3558 = (long) *its;
      ubound_3559 = (long) *ite;
      stride_3562 = (1 - lbound_3558) + ubound_3559;
      stride_3562 = gf_max(stride_3562, 0);
      lbound_3560 = (long) *kts;
      ubound_3561 = (long) *kte;
      size_3564 = ((1 - lbound_3560) + ubound_3561) * stride_3562;
      size_3564 = gf_max(size_3564, 0);
      D_14274 = size_3564 + -1;
      D_14275 = (ulong) (ulong) (size_3564) * 32;
      D_14276 = (ulong) (size_3564) * 4;
      D_14277 = (device void *) workspace.allocate (gf_max((ulong) (size_3564 * 4), 1));
      po = (device GFReal *) D_14277;
      offset_3563 = -(lbound_3558) - lbound_3560 * stride_3562;
      lbound_3572 = (long) *its;
      ubound_3573 = (long) *ite;
      size_3575 = (1 - lbound_3572) + ubound_3573;
      size_3575 = gf_max(size_3575, 0);
      D_14270 = size_3575 + -1;
      D_14271 = (ulong) (ulong) (size_3575) * 32;
      D_14272 = (ulong) (size_3575) * 4;
      D_14273 = (device void *) workspace.allocate (gf_max((ulong) (size_3575 * 4), 1));
      pret = (device GFReal *) D_14273;
      offset_3574 = -(lbound_3572);
      lbound_3576 = (long) *its;
      ubound_3577 = (long) *ite;
      stride_3580 = (1 - lbound_3576) + ubound_3577;
      stride_3580 = gf_max(stride_3580, 0);
      lbound_3578 = (long) *jts;
      ubound_3579 = (long) *jte;
      stride_3581 = ((1 - lbound_3578) + ubound_3579) * stride_3580;
      stride_3581 = gf_max(stride_3581, 0);
      size_3583 = stride_3581 * 16;
      size_3583 = gf_max(size_3583, 0);
      D_14266 = size_3583 + -1;
      D_14267 = (ulong) (ulong) (size_3583) * 32;
      D_14268 = (ulong) (size_3583) * 4;
      D_14269 = (device void *) workspace.allocate (gf_max((ulong) (size_3583 * 4), 1));
      pri_ens = (device GFReal *) D_14269;
      offset_3582 = (-(lbound_3576) - lbound_3578 * stride_3580) - (stride_3581);
      lbound_3584 = (long) *its;
      ubound_3585 = (long) *ite;
      size_3587 = (1 - lbound_3584) + ubound_3585;
      size_3587 = gf_max(size_3587, 0);
      D_14262 = size_3587 + -1;
      D_14263 = (ulong) (ulong) (size_3587) * 32;
      D_14264 = (ulong) (size_3587) * 4;
      D_14265 = (device void *) workspace.allocate (gf_max((ulong) (size_3587 * 4), 1));
      psur = (device GFReal *) D_14265;
      offset_3586 = -(lbound_3584);
      lbound_3598 = (long) *its;
      ubound_3599 = (long) *ite;
      stride_3602 = (1 - lbound_3598) + ubound_3599;
      stride_3602 = gf_max(stride_3602, 0);
      lbound_3600 = (long) *kts;
      ubound_3601 = (long) *kte;
      size_3604 = ((1 - lbound_3600) + ubound_3601) * stride_3602;
      size_3604 = gf_max(size_3604, 0);
      D_14258 = size_3604 + -1;
      D_14259 = (ulong) (ulong) (size_3604) * 32;
      D_14260 = (ulong) (size_3604) * 4;
      D_14261 = (device void *) workspace.allocate (gf_max((ulong) (size_3604 * 4), 1));
      q2d = (device GFReal *) D_14261;
      offset_3603 = -(lbound_3598) - lbound_3600 * stride_3602;
      lbound_3612 = (long) *its;
      ubound_3613 = (long) *ite;
      stride_3616 = (1 - lbound_3612) + ubound_3613;
      stride_3616 = gf_max(stride_3616, 0);
      lbound_3614 = (long) *kts;
      ubound_3615 = (long) *kte;
      size_3618 = ((1 - lbound_3614) + ubound_3615) * stride_3616;
      size_3618 = gf_max(size_3618, 0);
      D_14254 = size_3618 + -1;
      D_14255 = (ulong) (ulong) (size_3618) * 32;
      D_14256 = (ulong) (size_3618) * 4;
      D_14257 = (device void *) workspace.allocate (gf_max((ulong) (size_3618 * 4), 1));
      qo = (device GFReal *) D_14257;
      offset_3617 = -(lbound_3612) - lbound_3614 * stride_3616;
      lbound_3619 = (long) *its;
      ubound_3620 = (long) *ite;
      stride_3623 = (1 - lbound_3619) + ubound_3620;
      stride_3623 = gf_max(stride_3623, 0);
      lbound_3621 = (long) *kts;
      ubound_3622 = (long) *kte;
      size_3625 = ((1 - lbound_3621) + ubound_3622) * stride_3623;
      size_3625 = gf_max(size_3625, 0);
      D_14250 = size_3625 + -1;
      D_14251 = (ulong) (ulong) (size_3625) * 32;
      D_14252 = (ulong) (size_3625) * 4;
      D_14253 = (device void *) workspace.allocate (gf_max((ulong) (size_3625 * 4), 1));
      qshall = (device GFReal *) D_14253;
      offset_3624 = -(lbound_3619) - lbound_3621 * stride_3623;
      lbound_3643 = (long) *its;
      ubound_3644 = (long) *ite;
      stride_3647 = (1 - lbound_3643) + ubound_3644;
      stride_3647 = gf_max(stride_3647, 0);
      lbound_3645 = (long) *kts;
      ubound_3646 = (long) *kte;
      size_3649 = ((1 - lbound_3645) + ubound_3646) * stride_3647;
      size_3649 = gf_max(size_3649, 0);
      D_14246 = size_3649 + -1;
      D_14247 = (ulong) (ulong) (size_3649) * 32;
      D_14248 = (ulong) (size_3649) * 4;
      D_14249 = (device void *) workspace.allocate (gf_max((ulong) (size_3649 * 4), 1));
      rhoi = (device GFReal *) D_14249;
      offset_3648 = -(lbound_3643) - lbound_3645 * stride_3647;
      lbound_3740 = (long) *its;
      ubound_3741 = (long) *ite;
      stride_3744 = (1 - lbound_3740) + ubound_3741;
      stride_3744 = gf_max(stride_3744, 0);
      lbound_3742 = (long) *kts;
      ubound_3743 = (long) *kte;
      size_3746 = ((1 - lbound_3742) + ubound_3743) * stride_3744;
      size_3746 = gf_max(size_3746, 0);
      D_14242 = size_3746 + -1;
      D_14243 = (ulong) (ulong) (size_3746) * 32;
      D_14244 = (ulong) (size_3746) * 4;
      D_14245 = (device void *) workspace.allocate (gf_max((ulong) (size_3746 * 4), 1));
      subm = (device GFReal *) D_14245;
      offset_3745 = -(lbound_3740) - lbound_3742 * stride_3744;
      lbound_3747 = (long) *its;
      ubound_3748 = (long) *ite;
      stride_3751 = (1 - lbound_3747) + ubound_3748;
      stride_3751 = gf_max(stride_3751, 0);
      lbound_3749 = (long) *kts;
      ubound_3750 = (long) *kte;
      size_3753 = ((1 - lbound_3749) + ubound_3750) * stride_3751;
      size_3753 = gf_max(size_3753, 0);
      D_14238 = size_3753 + -1;
      D_14239 = (ulong) (ulong) (size_3753) * 32;
      D_14240 = (ulong) (size_3753) * 4;
      D_14241 = (device void *) workspace.allocate (gf_max((ulong) (size_3753 * 4), 1));
      subq = (device GFReal *) D_14241;
      offset_3752 = -(lbound_3747) - lbound_3749 * stride_3751;
      lbound_3754 = (long) *its;
      ubound_3755 = (long) *ite;
      stride_3758 = (1 - lbound_3754) + ubound_3755;
      stride_3758 = gf_max(stride_3758, 0);
      lbound_3756 = (long) *kts;
      ubound_3757 = (long) *kte;
      size_3760 = ((1 - lbound_3756) + ubound_3757) * stride_3758;
      size_3760 = gf_max(size_3760, 0);
      D_14234 = size_3760 + -1;
      D_14235 = (ulong) (ulong) (size_3760) * 32;
      D_14236 = (ulong) (size_3760) * 4;
      D_14237 = (device void *) workspace.allocate (gf_max((ulong) (size_3760 * 4), 1));
      subt = (device GFReal *) D_14237;
      offset_3759 = -(lbound_3754) - lbound_3756 * stride_3758;
      lbound_3771 = (long) *its;
      ubound_3772 = (long) *ite;
      stride_3775 = (1 - lbound_3771) + ubound_3772;
      stride_3775 = gf_max(stride_3775, 0);
      lbound_3773 = (long) *kts;
      ubound_3774 = (long) *kte;
      size_3777 = ((1 - lbound_3773) + ubound_3774) * stride_3775;
      size_3777 = gf_max(size_3777, 0);
      D_14230 = size_3777 + -1;
      D_14231 = (ulong) (ulong) (size_3777) * 32;
      D_14232 = (ulong) (size_3777) * 4;
      D_14233 = (device void *) workspace.allocate (gf_max((ulong) (size_3777 * 4), 1));
      t2d = (device GFReal *) D_14233;
      offset_3776 = -(lbound_3771) - lbound_3773 * stride_3775;
      lbound_3778 = (long) *its;
      ubound_3779 = (long) *ite;
      size_3781 = (1 - lbound_3778) + ubound_3779;
      size_3781 = gf_max(size_3781, 0);
      D_14226 = size_3781 + -1;
      D_14227 = (ulong) (ulong) (size_3781) * 32;
      D_14228 = (ulong) (size_3781) * 4;
      D_14229 = (device void *) workspace.allocate (gf_max((ulong) (size_3781 * 4), 1));
      ter11 = (device GFReal *) D_14229;
      offset_3780 = -(lbound_3778);
      lbound_3782 = (long) *its;
      ubound_3783 = (long) *ite;
      stride_3786 = (1 - lbound_3782) + ubound_3783;
      stride_3786 = gf_max(stride_3786, 0);
      lbound_3784 = (long) *kts;
      ubound_3785 = (long) *kte;
      size_3788 = ((1 - lbound_3784) + ubound_3785) * stride_3786;
      size_3788 = gf_max(size_3788, 0);
      D_14222 = size_3788 + -1;
      D_14223 = (ulong) (ulong) (size_3788) * 32;
      D_14224 = (ulong) (size_3788) * 4;
      D_14225 = (device void *) workspace.allocate (gf_max((ulong) (size_3788 * 4), 1));
      tn = (device GFReal *) D_14225;
      offset_3787 = -(lbound_3782) - lbound_3784 * stride_3786;
      lbound_3789 = (long) *its;
      ubound_3790 = (long) *ite;
      size_3792 = (1 - lbound_3789) + ubound_3790;
      size_3792 = gf_max(size_3792, 0);
      D_14218 = size_3792 + -1;
      D_14219 = (ulong) (ulong) (size_3792) * 32;
      D_14220 = (ulong) (size_3792) * 4;
      D_14221 = (device void *) workspace.allocate (gf_max((ulong) (size_3792 * 4), 1));
      tscl_kf = (device GFReal *) D_14221;
      offset_3791 = -(lbound_3789);
      lbound_3793 = (long) *its;
      ubound_3794 = (long) *ite;
      stride_3797 = (1 - lbound_3793) + ubound_3794;
      stride_3797 = gf_max(stride_3797, 0);
      lbound_3795 = (long) *kts;
      ubound_3796 = (long) *kte;
      size_3799 = ((1 - lbound_3795) + ubound_3796) * stride_3797;
      size_3799 = gf_max(size_3799, 0);
      D_14214 = size_3799 + -1;
      D_14215 = (ulong) (ulong) (size_3799) * 32;
      D_14216 = (ulong) (size_3799) * 4;
      D_14217 = (device void *) workspace.allocate (gf_max((ulong) (size_3799 * 4), 1));
      tshall = (device GFReal *) D_14217;
      offset_3798 = -(lbound_3793) - lbound_3795 * stride_3797;
      lbound_3810 = (long) *its;
      ubound_3811 = (long) *ite;
      size_3813 = (1 - lbound_3810) + ubound_3811;
      size_3813 = gf_max(size_3813, 0);
      D_14210 = size_3813 + -1;
      D_14211 = (ulong) (ulong) (size_3813) * 32;
      D_14212 = (ulong) (size_3813) * 4;
      D_14213 = (device void *) workspace.allocate (gf_max((ulong) (size_3813 * 4), 1));
      umean = (device GFReal *) D_14213;
      offset_3812 = -(lbound_3810);
      lbound_3814 = (long) *its;
      ubound_3815 = (long) *ite;
      stride_3818 = (1 - lbound_3814) + ubound_3815;
      stride_3818 = gf_max(stride_3818, 0);
      lbound_3816 = (long) *kts;
      ubound_3817 = (long) *kte;
      size_3820 = ((1 - lbound_3816) + ubound_3817) * stride_3818;
      size_3820 = gf_max(size_3820, 0);
      D_14206 = size_3820 + -1;
      D_14207 = (ulong) (ulong) (size_3820) * 32;
      D_14208 = (ulong) (size_3820) * 4;
      D_14209 = (device void *) workspace.allocate (gf_max((ulong) (size_3820 * 4), 1));
      us = (device GFReal *) D_14209;
      offset_3819 = -(lbound_3814) - lbound_3816 * stride_3818;
      lbound_3831 = (long) *its;
      ubound_3832 = (long) *ite;
      size_3834 = (1 - lbound_3831) + ubound_3832;
      size_3834 = gf_max(size_3834, 0);
      D_14202 = size_3834 + -1;
      D_14203 = (ulong) (ulong) (size_3834) * 32;
      D_14204 = (ulong) (size_3834) * 4;
      D_14205 = (device void *) workspace.allocate (gf_max((ulong) (size_3834 * 4), 1));
      vmean = (device GFReal *) D_14205;
      offset_3833 = -(lbound_3831);
      lbound_3835 = (long) *its;
      ubound_3836 = (long) *ite;
      stride_3839 = (1 - lbound_3835) + ubound_3836;
      stride_3839 = gf_max(stride_3839, 0);
      lbound_3837 = (long) *kts;
      ubound_3838 = (long) *kte;
      size_3841 = ((1 - lbound_3837) + ubound_3838) * stride_3839;
      size_3841 = gf_max(size_3841, 0);
      D_14198 = size_3841 + -1;
      D_14199 = (ulong) (ulong) (size_3841) * 32;
      D_14200 = (ulong) (size_3841) * 4;
      D_14201 = (device void *) workspace.allocate (gf_max((ulong) (size_3841 * 4), 1));
      vs = (device GFReal *) D_14201;
      offset_3840 = -(lbound_3835) - lbound_3837 * stride_3839;
      lbound_3852 = (long) *its;
      ubound_3853 = (long) *ite;
      stride_3856 = (1 - lbound_3852) + ubound_3853;
      stride_3856 = gf_max(stride_3856, 0);
      lbound_3854 = (long) *jts;
      ubound_3855 = (long) *jte;
      stride_3857 = ((1 - lbound_3854) + ubound_3855) * stride_3856;
      stride_3857 = gf_max(stride_3857, 0);
      size_3859 = stride_3857 * 16;
      size_3859 = gf_max(size_3859, 0);
      D_14194 = size_3859 + -1;
      D_14195 = (ulong) (ulong) (size_3859) * 32;
      D_14196 = (ulong) (size_3859) * 4;
      D_14197 = (device void *) workspace.allocate (gf_max((ulong) (size_3859 * 4), 1));
      xfi_ens = (device GFReal *) D_14197;
      offset_3858 = (-(lbound_3852) - lbound_3854 * stride_3856) - (stride_3857);
      lbound_3867 = (long) *its;
      ubound_3868 = (long) *ite;
      size_3870 = (1 - lbound_3867) + ubound_3868;
      size_3870 = gf_max(size_3870, 0);
      D_14190 = size_3870 + -1;
      D_14191 = (ulong) (ulong) (size_3870) * 32;
      D_14192 = (ulong) (size_3870) * 4;
      D_14193 = (device void *) workspace.allocate (gf_max((ulong) (size_3870 * 4), 1));
      xlandi = (device GFReal *) D_14193;
      offset_3869 = -(lbound_3867);
      lbound_3871 = (long) *its;
      ubound_3872 = (long) *ite;
      size_3874 = (1 - lbound_3871) + ubound_3872;
      size_3874 = gf_max(size_3874, 0);
      D_14186 = size_3874 + -1;
      D_14187 = (ulong) (ulong) (size_3874) * 32;
      D_14188 = (ulong) (size_3874) * 4;
      D_14189 = (device void *) workspace.allocate (gf_max((ulong) (size_3874 * 4), 1));
      xmb = (device GFReal *) D_14189;
      offset_3873 = -(lbound_3871);
      lbound_3889 = (long) *its;
      ubound_3890 = (long) *ite;
      size_3892 = (1 - lbound_3889) + ubound_3890;
      size_3892 = gf_max(size_3892, 0);
      D_14182 = size_3892 + -1;
      D_14183 = (ulong) (ulong) (size_3892) * 32;
      D_14184 = (ulong) (size_3892) * 4;
      D_14185 = (device void *) workspace.allocate (gf_max((ulong) (size_3892 * 4), 1));
      xmbs = (device GFReal *) D_14185;
      offset_3891 = -(lbound_3889);
      lbound_3893 = (long) *its;
      ubound_3894 = (long) *ite;
      stride_3897 = (1 - lbound_3893) + ubound_3894;
      stride_3897 = gf_max(stride_3897, 0);
      lbound_3895 = (long) *kts;
      ubound_3896 = (long) *kte;
      size_3899 = ((1 - lbound_3895) + ubound_3896) * stride_3897;
      size_3899 = gf_max(size_3899, 0);
      D_14178 = size_3899 + -1;
      D_14179 = (ulong) (ulong) (size_3899) * 32;
      D_14180 = (ulong) (size_3899) * 4;
      D_14181 = (device void *) workspace.allocate (gf_max((ulong) (size_3899 * 4), 1));
      zo = (device GFReal *) D_14181;
      offset_3898 = -(lbound_3893) - lbound_3895 * stride_3897;
      lbound_3900 = (long) *its;
      ubound_3901 = (long) *ite;
      size_3903 = (1 - lbound_3900) + ubound_3901;
      size_3903 = gf_max(size_3903, 0);
      D_14174 = size_3903 + -1;
      D_14175 = (ulong) (ulong) (size_3903) * 32;
      D_14176 = (ulong) (size_3903) * 4;
      D_14177 = (device void *) workspace.allocate (gf_max((ulong) (size_3903 * 4), 1));
      zqexec = (device GFReal *) D_14177;
      offset_3902 = -(lbound_3900);
      lbound_3904 = (long) *its;
      ubound_3905 = (long) *ite;
      size_3907 = (1 - lbound_3904) + ubound_3905;
      size_3907 = gf_max(size_3907, 0);
      D_14170 = size_3907 + -1;
      D_14171 = (ulong) (ulong) (size_3907) * 32;
      D_14172 = (ulong) (size_3907) * 4;
      D_14173 = (device void *) workspace.allocate (gf_max((ulong) (size_3907 * 4), 1));
      ztexec = (device GFReal *) D_14173;
      offset_3906 = -(lbound_3904);
      {
        int D_13952;
        int M_3908;
        int D_13954;

        D_13952 = *ite;
        M_3908 = D_13952;
        D_13954 = *ide + -1;
        M_3908 = gf_min(D_13954, M_3908);
        itf = M_3908;
      }
      {
        int D_13955;
        int M_3909;
        int D_13957;

        D_13955 = *kte;
        M_3909 = D_13955;
        D_13957 = *kde + -1;
        M_3909 = gf_min(D_13957, M_3909);
        ktf = M_3909;
      }
      {
        int D_13958;
        int M_3910;
        int D_13960;

        D_13958 = *jte;
        M_3910 = D_13958;
        D_13960 = *jde + -1;
        M_3910 = gf_min(D_13960, M_3910);
        jtf = M_3910;
      }
      ipr = *ite;
      jpr = *jte;
      {
        int D_13961;
        int D_13962;

        D_13961 = *jts;
        D_13962 = *jte;
        j = D_13961;
        while (1)
          {
            {
              bool D_13965;

              D_13965 = j > D_13962;
              if (D_13965) break;
              {
                int D_13966;
                int D_13967;

                D_13966 = *its;
                D_13967 = *ite;
                i = D_13966;
                while (1)
                  {
                    {
                      bool D_13970;

                      D_13970 = i > D_13967;
                      if (D_13970) break;
                      hbot[((long) j * stride_3363 + offset_3364) + (long) i] = (GFReal) *kte;
                      htop[((long) j * stride_3384 + offset_3385) + (long) i] = (GFReal) *kts;
                      xmb_total[((long) j * stride_3886 + offset_3887) + (long) i] = 0.0f;
                      raincv[((long) j * stride_3630 + offset_3631) + (long) i] = 0.0f;
                      pratec[((long) j * stride_3569 + offset_3570) + (long) i] = 0.0f;
                      k22_shallow[((long) j * stride_3407 + offset_3408) + (long) i] = 0;
                      kbcon_shallow[((long) j * stride_3422 + offset_3423) + (long) i] = 0;
                      ktop_shallow[((long) j * stride_3448 + offset_3449) + (long) i] = 0;
                      xmb_shallow[((long) j * stride_3879 + offset_3880) + (long) i] = 0.0f;

                      i = i + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
      if (gdc != nullptr)
        {
          {
            int D_13972;
            int D_13973;

            D_13972 = *jts;
            D_13973 = *jte;
            j = D_13972;
            while (1)
              {
                {
                  bool D_13976;

                  D_13976 = j > D_13973;
                  if (D_13976) break;
                  {
                    int D_13977;
                    int D_13978;

                    D_13977 = *jts;
                    D_13978 = *kte;
                    k = D_13977;
                    while (1)
                      {
                        {
                          bool D_13981;

                          D_13981 = k > D_13978;
                          if (D_13981) break;
                          {
                            int D_13982;
                            int D_13983;

                            D_13982 = *its;
                            D_13983 = *ite;
                            i = D_13982;
                            while (1)
                              {
                                {
                                  bool D_13986;

                                  D_13986 = i > D_13983;
                                  if (D_13986) break;
                                  gdc[(((long) j * stride_3332 + offset_3333) + (long) k * stride_3329) + (long) i] = 0.0f;

                                  i = i + 1;
                                }
                              }

                          }

                          k = k + 1;
                        }
                      }

                  }

                  j = j + 1;
                }
              }

          }
        }

      if (gdc2 != nullptr)
        {
          {
            int D_13988;
            int D_13989;

            D_13988 = *jts;
            D_13989 = *jte;
            j = D_13988;
            while (1)
              {
                {
                  bool D_13992;

                  D_13992 = j > D_13989;
                  if (D_13992) break;
                  {
                    int D_13993;
                    int D_13994;

                    D_13993 = *jts;
                    D_13994 = *kte;
                    k = D_13993;
                    while (1)
                      {
                        {
                          bool D_13997;

                          D_13997 = k > D_13994;
                          if (D_13997) break;
                          {
                            int D_13998;
                            int D_13999;

                            D_13998 = *its;
                            D_13999 = *ite;
                            i = D_13998;
                            while (1)
                              {
                                {
                                  bool D_14002;

                                  D_14002 = i > D_13999;
                                  if (D_14002) break;
                                  gdc2[(((long) j * stride_3342 + offset_3343) + (long) k * stride_3339) + (long) i] = 0.0f;

                                  i = i + 1;
                                }
                              }

                          }

                          k = k + 1;
                        }
                      }

                  }

                  j = j + 1;
                }
              }

          }
        }

      {
        int D_14003;
        int D_14004;

        D_14003 = *jts;
        D_14004 = jtf;
        j = D_14003;
        while (1)
          {
            {
              bool D_14007;

              D_14007 = j > D_14004;
              if (D_14007) break;
              {
                int D_14008;
                int D_14009;

                D_14008 = *its;
                D_14009 = itf;
                i = D_14008;
                while (1)
                  {
                    {
                      bool D_14012;

                      D_14012 = i > D_14009;
                      if (D_14012) break;
                      apri_gr[((long) j * stride_3233 + offset_3234) + (long) i] = 0.0f;
                      apri_w[((long) j * stride_3254 + offset_3255) + (long) i] = 0.0f;
                      apri_mc[((long) j * stride_3240 + offset_3241) + (long) i] = 0.0f;
                      apri_st[((long) j * stride_3247 + offset_3248) + (long) i] = 0.0f;
                      apri_as[((long) j * stride_3205 + offset_3206) + (long) i] = 0.0f;
                      apri_capma[((long) j * stride_3212 + offset_3213) + (long) i] = 0.0f;
                      apri_capme[((long) j * stride_3219 + offset_3220) + (long) i] = 0.0f;
                      apri_capmi[((long) j * stride_3226 + offset_3227) + (long) i] = 0.0f;
                      edti_out[((long) j * stride_3322 + offset_3323) + (long) i] = 0.0f;
                      gswi[((long) j * stride_3356 + offset_3357) + (long) i] = gsw[((long) j * stride_3349 + offset_3350) + (long) i];

                      i = i + 1;
                    }
                  }

              }
              n = 1;
              while (1)
                {
                  {
                    bool D_14015;

                    D_14015 = n > 16;
                    if (D_14015) break;
                    {
                      int D_14016;
                      int D_14017;

                      D_14016 = *its;
                      D_14017 = itf;
                      i = D_14016;
                      while (1)
                        {
                          {
                            bool D_14020;

                            D_14020 = i > D_14017;
                            if (D_14020) break;
                            pri_ens[(((long) n * stride_3581 + offset_3582) + (long) j * stride_3580) + (long) i] = 0.0f;
                            xfi_ens[(((long) n * stride_3857 + offset_3858) + (long) j * stride_3856) + (long) i] = 0.0f;

                            i = i + 1;
                          }
                        }

                    }

                    n = n + 1;
                  }
                }

              n = 1;
              while (1)
                {
                  {
                    bool D_14023;

                    D_14023 = n > 9;
                    if (D_14023) break;
                    {
                      int D_14024;
                      int D_14025;

                      D_14024 = *kts;
                      D_14025 = ktf;
                      k = D_14024;
                      while (1)
                        {
                          {
                            bool D_14028;

                            D_14028 = k > D_14025;
                            if (D_14028) break;
                            {
                              int D_14029;
                              int D_14030;

                              D_14029 = *its;
                              D_14030 = *ite;
                              i = D_14029;
                              while (1)
                                {
                                  {
                                    bool D_14033;

                                    D_14033 = i > D_14030;
                                    if (D_14033) break;
                                    omeg[(((long) n * stride_3465 + offset_3466) + (long) k * stride_3464) + (long) i] = 0.0f;

                                    i = i + 1;
                                  }
                                }

                            }

                            k = k + 1;
                          }
                        }

                    }
                    {
                      int D_14034;
                      int D_14035;

                      D_14034 = *its;
                      D_14035 = itf;
                      i = D_14034;
                      while (1)
                        {
                          {
                            bool D_14038;

                            D_14038 = i > D_14035;
                            if (D_14038) break;
                            mconv[((long) n * stride_3457 + offset_3458) + (long) i] = 0.0f;

                            i = i + 1;
                          }
                        }

                    }

                    n = n + 1;
                  }
                }

              {
                int D_14039;
                int D_14040;

                D_14039 = *its;
                D_14040 = itf;
                i = D_14039;
                while (1)
                  {
                    {
                      bool D_14043;

                      D_14043 = i > D_14040;
                      if (D_14043) break;
                      gf_memmove ((device char *)&ierrc[(long) i + offset_3393], "", 0);
                      gf_memset ((device char *)&ierrc[(long) i + offset_3393], 32, 50);
                      ierr[(long) i + offset_3389] = 0;
                      kbcon[(long) i + offset_3416] = 0;
                      ktop[(long) i + offset_3442] = 0;
                      xmb[(long) i + offset_3873] = 0.0f;
                      gf_memmove ((device char *)&ierrcs[(long) i + offset_3397], "", 0);
                      gf_memset ((device char *)&ierrcs[(long) i + offset_3397], 32, 50);
                      ierrs[(long) i + offset_3401] = 0;
                      kbcons[(long) i + offset_3427] = 0;
                      ktops[(long) i + offset_3453] = 0;
                      k22s[(long) i + offset_3412] = 0;
                      xmbs[(long) i + offset_3891] = 0.0f;
                      tscl_kf[(long) i + offset_3791] = dxcell[((long) j * stride_3305 + offset_3306) + (long) i] / 2.5e+1f;

                      i = i + 1;
                    }
                  }

              }
              {
                int D_14044;
                int D_14045;

                D_14044 = *its;
                D_14045 = itf;
                i = D_14044;
                while (1)
                  {
                    {
                      bool D_14048;

                      D_14048 = i > D_14045;
                      if (D_14048) break;
                      dx_loc[(long) i + offset_3299] = dxcell[((long) j * stride_3305 + offset_3306) + (long) i];
                      area_loc[(long) i + offset_3259] = areacell[((long) j * stride_3265 + offset_3266) + (long) i];
                      {
                        GFReal M_3911;
                        GFReal D_14050;

                        M_3911 = 0.0f;
                        D_14050 = ht[((long) j * stride_3377 + offset_3378) + (long) i];
                        M_3911 = gf_max(D_14050, M_3911);
                        ter11[(long) i + offset_3780] = M_3911;
                      }
                      zo[((long) *kts * stride_3897 + offset_3898) + (long) i] = ter11[(long) i + offset_3780] + dz8w[(((long) j * stride_3315 + offset_3316) + (stride_3312)) + (long) i] * 5.0e-1f;
                      {
                        int D_14051;
                        int D_14052;

                        D_14051 = *kts + 1;
                        D_14052 = ktf;
                        k = D_14051;
                        while (1)
                          {
                            {
                              bool D_14055;

                              D_14055 = k > D_14052;
                              if (D_14055) break;
                              zo[((long) k * stride_3897 + offset_3898) + (long) i] = zo[((long) (k + -1) * stride_3897 + offset_3898) + (long) i] + ((dz8w[(((long) j * stride_3315 + offset_3316) + (long) (k + -1) * stride_3312) + (long) i] + dz8w[(((long) j * stride_3315 + offset_3316) + (long) k * stride_3312) + (long) i])) * 5.0e-1f;

                              k = k + 1;
                            }
                          }

                      }
                      psur[(long) i + offset_3586] = p8w[(((long) j * stride_3534 + offset_3535) + (stride_3531)) + (long) i] * 9.99999977648258209228515625e-3f;
                      kpbli[(long) i + offset_3438] = kpbl[((long) j * stride_3433 + offset_3434) + (long) i];
                      xlandi[(long) i + offset_3869] = xland[((long) j * stride_3864 + offset_3865) + (long) i];
                      ccn[(long) i + offset_3270] = 1.5e+3f;
                      cuten[(long) i + offset_3288] = 0.0f;
                      umean[(long) i + offset_3812] = 0.0f;
                      vmean[(long) i + offset_3833] = 0.0f;
                      pmean[(long) i + offset_3556] = 0.0f;
                      pret[(long) i + offset_3574] = 0.0f;
                      aaeq[(long) i + offset_3199] = 0.0f;
                      zqexec[(long) i + offset_3902] = 0.0f;
                      ztexec[(long) i + offset_3906] = 0.0f;

                      i = i + 1;
                    }
                  }

              }
              {
                int D_14056;
                int D_14057;

                D_14056 = *kts;
                D_14057 = ktf;
                k = D_14056;
                while (1)
                  {
                    {
                      bool D_14060;

                      D_14060 = k > D_14057;
                      if (D_14060) break;
                      {
                        int D_14061;
                        int D_14062;

                        D_14061 = *its;
                        D_14062 = itf;
                        i = D_14061;
                        while (1)
                          {
                            {
                              bool D_14065;

                              D_14065 = i > D_14062;
                              if (D_14065) break;
                              us[((long) k * stride_3818 + offset_3819) + (long) i] = u[(((long) j * stride_3807 + offset_3808) + (long) k * stride_3804) + (long) i];
                              vs[((long) k * stride_3839 + offset_3840) + (long) i] = v[(((long) j * stride_3828 + offset_3829) + (long) k * stride_3825) + (long) i];
                              rhoi[((long) k * stride_3647 + offset_3648) + (long) i] = rho[(((long) j * stride_3640 + offset_3641) + (long) k * stride_3637) + (long) i];
                              t2d[((long) k * stride_3775 + offset_3776) + (long) i] = t[(((long) j * stride_3768 + offset_3769) + (long) k * stride_3765) + (long) i];
                              q2d[((long) k * stride_3602 + offset_3603) + (long) i] = q[(((long) j * stride_3595 + offset_3596) + (long) k * stride_3592) + (long) i];
                              if (q2d[((long) k * stride_3602 + offset_3603) + (long) i] < 9.99999993922529029077850282192230224609375e-9f)
                                {
                                  q2d[((long) k * stride_3602 + offset_3603) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                                }

                              tn[((long) k * stride_3786 + offset_3787) + (long) i] = t2d[((long) k * stride_3775 + offset_3776) + (long) i] + ((((rthften[(((long) j * stride_3727 + offset_3728) + (long) k * stride_3724) + (long) i] + rthraten[(((long) j * stride_3737 + offset_3738) + (long) k * stride_3734) + (long) i]) + rthblten[(((long) j * stride_3707 + offset_3708) + (long) k * stride_3704) + (long) i])) * pi[(((long) j * stride_3551 + offset_3552) + (long) k * stride_3548) + (long) i]) * *dt;
                              qo[((long) k * stride_3616 + offset_3617) + (long) i] = q2d[((long) k * stride_3602 + offset_3603) + (long) i] + ((rqvften[(((long) j * stride_3697 + offset_3698) + (long) k * stride_3694) + (long) i] + rqvblten[(((long) j * stride_3677 + offset_3678) + (long) k * stride_3674) + (long) i])) * *dt;
                              if (tn[((long) k * stride_3786 + offset_3787) + (long) i] < 2.0e+2f)
                                {
                                  tn[((long) k * stride_3786 + offset_3787) + (long) i] = t2d[((long) k * stride_3775 + offset_3776) + (long) i];
                                }

                              if (qo[((long) k * stride_3616 + offset_3617) + (long) i] < 9.99999993922529029077850282192230224609375e-9f)
                                {
                                  qo[((long) k * stride_3616 + offset_3617) + (long) i] = 9.99999993922529029077850282192230224609375e-9f;
                                }

                              phh[((long) k * stride_3541 + offset_3542) + (long) i] = p[(((long) j * stride_3517 + offset_3518) + (long) k * stride_3514) + (long) i];
                              po[((long) k * stride_3562 + offset_3563) + (long) i] = phh[((long) k * stride_3541 + offset_3542) + (long) i] * 9.99999977648258209228515625e-3f;
                              p2d[((long) k * stride_3524 + offset_3525) + (long) i] = po[((long) k * stride_3562 + offset_3563) + (long) i];
                              cupclw[((long) k * stride_3276 + offset_3277) + (long) i] = 0.0f;
                              outq[((long) k * stride_3472 + offset_3473) + (long) i] = 0.0f;
                              outqc[((long) k * stride_3479 + offset_3480) + (long) i] = 0.0f;
                              outt[((long) k * stride_3500 + offset_3501) + (long) i] = 0.0f;
                              subm[((long) k * stride_3744 + offset_3745) + (long) i] = 0.0f;
                              subq[((long) k * stride_3751 + offset_3752) + (long) i] = 0.0f;
                              subt[((long) k * stride_3758 + offset_3759) + (long) i] = 0.0f;
                              tshall[((long) k * stride_3797 + offset_3798) + (long) i] = t2d[((long) k * stride_3775 + offset_3776) + (long) i] + (rthblten[(((long) j * stride_3707 + offset_3708) + (long) k * stride_3704) + (long) i] * pi[(((long) j * stride_3551 + offset_3552) + (long) k * stride_3548) + (long) i]) * *dt;
                              qshall[((long) k * stride_3623 + offset_3624) + (long) i] = q2d[((long) k * stride_3602 + offset_3603) + (long) i] + rqvblten[(((long) j * stride_3677 + offset_3678) + (long) k * stride_3674) + (long) i] * *dt;
                              dhdt[((long) k * stride_3294 + offset_3295) + (long) i] = (*cp * rthblten[(((long) j * stride_3707 + offset_3708) + (long) k * stride_3704) + (long) i]) * pi[(((long) j * stride_3551 + offset_3552) + (long) k * stride_3548) + (long) i] + *xlv * rqvblten[(((long) j * stride_3677 + offset_3678) + (long) k * stride_3674) + (long) i];
                              cupclws[((long) k * stride_3283 + offset_3284) + (long) i] = 0.0f;
                              outqcs[((long) k * stride_3486 + offset_3487) + (long) i] = 0.0f;
                              outqs[((long) k * stride_3493 + offset_3494) + (long) i] = 0.0f;
                              outts[((long) k * stride_3507 + offset_3508) + (long) i] = 0.0f;

                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              n = 1;
              while (1)
                {
                  {
                    bool D_14071;

                    D_14071 = n > 9;
                    if (D_14071) break;
                    {
                      int D_14072;
                      int D_14073;

                      D_14072 = *kts + 1;
                      D_14073 = ktf;
                      k = D_14072;
                      while (1)
                        {
                          {
                            bool D_14076;

                            D_14076 = k > D_14073;
                            if (D_14076) break;
                            {
                              int D_14077;
                              int D_14078;

                              D_14077 = *its;
                              D_14078 = itf;
                              i = D_14077;
                              while (1)
                                {
                                  {
                                    bool D_14081;

                                    D_14081 = i > D_14078;
                                    if (D_14081) break;
                                    omeg[(((long) n * stride_3465 + offset_3466) + (long) k * stride_3464) + (long) i] = -(((*g * 5.0e-1f) * ((rho[(((long) j * stride_3640 + offset_3641) + (long) k * stride_3637) + (long) i] + rho[(((long) j * stride_3640 + offset_3641) + (long) (k + -1) * stride_3637) + (long) i]))) * w[(((long) j * stride_3849 + offset_3850) + (long) k * stride_3846) + (long) i]);

                                    i = i + 1;
                                  }
                                }

                            }

                            k = k + 1;
                          }
                        }

                    }
                    {
                      int D_14082;
                      int D_14083;

                      D_14082 = *kts + 1;
                      D_14083 = ktf;
                      k = D_14082;
                      while (1)
                        {
                          {
                            bool D_14086;

                            D_14086 = k > D_14083;
                            if (D_14086) break;
                            {
                              int D_14087;
                              int D_14088;

                              D_14087 = *its;
                              D_14088 = itf;
                              i = D_14087;
                              while (1)
                                {
                                  {
                                    bool D_14091;

                                    D_14091 = i > D_14088;
                                    if (D_14091) break;
                                    dq = ((q2d[((long) k * stride_3602 + offset_3603) + (long) i] - q2d[((long) (k + -1) * stride_3602 + offset_3603) + (long) i]));
                                    mconv[((long) n * stride_3457 + offset_3458) + (long) i] = mconv[((long) n * stride_3457 + offset_3458) + (long) i] + (omeg[(((long) n * stride_3465 + offset_3466) + (long) k * stride_3464) + (long) i] * dq) / *g;

                                    i = i + 1;
                                  }
                                }

                            }

                            k = k + 1;
                          }
                        }

                    }
                    {
                      int D_14092;
                      int D_14093;

                      D_14092 = *its;
                      D_14093 = itf;
                      i = D_14092;
                      while (1)
                        {
                          {
                            bool D_14096;

                            D_14096 = i > D_14093;
                            if (D_14096) break;
                            if (mconv[((long) n * stride_3457 + offset_3458) + (long) i] < 0.0f)
                              {
                                mconv[((long) n * stride_3457 + offset_3458) + (long) i] = 0.0f;
                              }


                            i = i + 1;
                          }
                        }

                    }

                    n = n + 1;
                  }
                }

              if (0)
                {
                  {
                    int D_14099;
                    int D_14100;

                    D_14099 = *its;
                    D_14100 = itf;
                    i = D_14099;
                    while (1)
                      {
                        {
                          bool D_14103;

                          D_14103 = i > D_14100;
                          if (D_14103) break;
                          zrho = (psur[(long) i + offset_3586] * 1.0e+2f) / ((((t2d[(offset_3776 + (stride_3775)) + (long) i] * ((q2d[(offset_3603 + (stride_3602)) + (long) i] * 6.07999980449676513671875e-1f + 1.0e+0f)))) * 2.87040008544921875e+2f));
                          pahfs = -hfx[((long) j * stride_3370 + offset_3371) + (long) i];
                          pqhfl = -(qfx[((long) j * stride_3609 + offset_3610) + (long) i] / *xlv);
                          zkhvfl = ((pahfs / 1.0046400146484375e+3f + (t2d[(offset_3776 + (stride_3775)) + (long) i] * 6.07999980449676513671875e-1f) * pqhfl)) / zrho;
                          pgeoh = zo[(offset_3898 + (stride_3897)) + (long) i] - ht[((long) j * stride_3377 + offset_3378) + (long) i];
                          {
                            GFReal M_3912;
                            GFReal D_14105;

                            M_3912 = 0.0f;
                            D_14105 = 1.000000047497451305389404296875e-3f - ((zkhvfl * 6.150000095367431640625e-1f) * pgeoh) / t2d[(offset_3776 + (stride_3775)) + (long) i];
                            M_3912 = gf_max(D_14105, M_3912);
                            zws = M_3912;
                          }
                          if (zws > 1.17549435082228750796873653722224567781866555677208752151e-38f)
                            {
                              {
                                GFReal D_14107;

                                D_14107 = zws;
                                zws = gf_pow (D_14107, 3.3329999446868896484375e-1f) * 1.2000000476837158203125e+0f;
                              }
                              {
                                GFReal D_14108;
                                GFReal M_3913;

                                D_14108 = -((pahfs * 1.5e+0f) / (((zrho * zws) * 1.0046400146484375e+3f)));
                                M_3913 = D_14108;
                                M_3913 = gf_max(M_3913, 0.0f);
                                ztexec[(long) i + offset_3906] = M_3913;
                              }
                              {
                                GFReal D_14110;
                                GFReal M_3914;

                                D_14110 = -((pqhfl * 1.5e+0f) / ((zrho * zws)));
                                M_3914 = D_14110;
                                M_3914 = gf_max(M_3914, 0.0f);
                                zqexec[(long) i + offset_3902] = M_3914;
                              }
                            }


                          i = i + 1;
                        }
                      }

                  }
                }

              {
                int D_14112;
                int D_14113;

                D_14112 = *kts + 1;
                D_14113 = *kte + -1;
                k = D_14112;
                while (1)
                  {
                    {
                      bool D_14116;

                      D_14116 = k > D_14113;
                      if (D_14116) break;
                      {
                        int D_14117;
                        int D_14118;

                        D_14117 = *its;
                        D_14118 = itf;
                        i = D_14117;
                        while (1)
                          {
                            {
                              bool D_14121;

                              D_14121 = i > D_14118;
                              if (D_14121) break;
                              if (((p2d[(offset_3525 + (stride_3524)) + (long) i] - p2d[((long) k * stride_3524 + offset_3525) + (long) i])) > 1.5e+2f && p2d[((long) k * stride_3524 + offset_3525) + (long) i] > 3.0e+2f)
                                {
                                  dp = -(((p2d[((long) (k + 1) * stride_3524 + offset_3525) + (long) i] - p2d[((long) (k + -1) * stride_3524 + offset_3525) + (long) i])) * 5.0e-1f);
                                  umean[(long) i + offset_3812] = umean[(long) i + offset_3812] + us[((long) k * stride_3818 + offset_3819) + (long) i] * dp;
                                  vmean[(long) i + offset_3833] = vmean[(long) i + offset_3833] + vs[((long) k * stride_3839 + offset_3840) + (long) i] * dp;
                                  pmean[(long) i + offset_3556] = pmean[(long) i + offset_3556] + dp;
                                }


                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }
              {
                GFReal C_14123 = 2.58e+2f;
                int C_14124 = 1;
                GFReal C_14125 = 2.5e+2f;
                int C_14126 = 1;
                int C_14127 = 1;
                int C_14128 = 1;
                int C_14129 = 16;
                int C_14130 = 16;
                int C_14131 = 9;
                GFReal C_14132 = 1.99999995529651641845703125e-2f;
                int C_14133 = 1;
                int C_14134 = 1;
                int C_14135 = 0;
                int C_14136 = 0;

                cup_gf (workspace, zo, outqc, &j, aaeq, t2d, q2d, ter11, subm, tn, qo, po, pret, p2d, outt, outq, dt, itimestep, psur, us, vs, &C_14123, &C_14124, ztexec, zqexec, ccn, &C_14125, rhoi, dx_loc, area_loc, mconv, omeg, &C_14126, &C_14127, &C_14128, &C_14129, &C_14130, apri_gr, apri_w, apri_mc, apri_st, apri_as, apri_capma, apri_capme, apri_capmi, kbcon, ktop, cupclw, xfi_ens, pri_ens, xlandi, gswi, subt, subq, xlv, r_v, cp, g, ichoice_deep, &ipr, &jpr, ierrc, &C_14131, &C_14132, &C_14133, &C_14134, &itf, &jtf, &ktf, &C_14135, xmb, &C_14136, its, ite, jts, jte, kts, kte, 50);
              }
              neg_check (workspace, &j, subt, subq, dt, q2d, outq, outt, outqc, pret, its, ite, kts, kte, &itf, &ktf);
              if (*ishallow_g3 == 1)
                {
                  {
                    GFReal C_14138 = 2.58e+2f;
                    GFReal C_14139 = 2.5e+2f;
                    int C_14140 = 0;
                    int C_14141 = 0;
                    int C_14142 = 1;
                    int C_14143 = 0;

                    cup_gf_sh (workspace, xmbs, zo, outqcs, &j, aaeq, t2d, q2d, ter11, tshall, qshall, p2d, pret, p2d, outts, outqs, dt, itimestep, psur, us, vs, &C_14138, ztexec, zqexec, ccn, &C_14139, rhoi, dx_loc, area_loc, dhdt, kpbli, kbcons, ktops, cupclws, k22s, xlandi, gswi, tscl_kf, xlv, r_v, cp, g, ichoice_shallow, &C_14140, &C_14141, ierrs, ierrcs, &C_14142, &itf, &jtf, &ktf, &C_14143, its, ite, jts, jte, kts, kte, 50);
                  }
                  {
                    int D_14144;
                    int D_14145;

                    D_14144 = *its;
                    D_14145 = *ite;
                    i = D_14144;
                    while (1)
                      {
                        {
                          bool D_14148;

                          D_14148 = i > D_14145;
                          if (D_14148) break;
                          xmb_shallow[((long) j * stride_3879 + offset_3880) + (long) i] = xmbs[(long) i + offset_3891];
                          k22_shallow[((long) j * stride_3407 + offset_3408) + (long) i] = k22s[(long) i + offset_3412];
                          kbcon_shallow[((long) j * stride_3422 + offset_3423) + (long) i] = kbcons[(long) i + offset_3427];
                          ktop_shallow[((long) j * stride_3448 + offset_3449) + (long) i] = ktops[(long) i + offset_3453];

                          i = i + 1;
                        }
                      }

                  }
                }

              {
                int D_14149;
                int D_14150;

                D_14149 = *its;
                D_14150 = *ite;
                i = D_14149;
                while (1)
                  {
                    {
                      bool D_14153;

                      D_14153 = i > D_14150;
                      if (D_14153) break;
                      xmb_total[((long) j * stride_3886 + offset_3887) + (long) i] = xmb[(long) i + offset_3873];
                      if (pret[(long) i + offset_3574] > 0.0f)
                        {
                          cuten[(long) i + offset_3288] = 1.0e+0f;
                          pratec[((long) j * stride_3569 + offset_3570) + (long) i] = pret[(long) i + offset_3574];
                          raincv[((long) j * stride_3630 + offset_3631) + (long) i] = pret[(long) i + offset_3574] * *dt;
                          if ((GFReal) ktop[(long) i + offset_3442] > htop[((long) j * stride_3384 + offset_3385) + (long) i])
                            {
                              htop[((long) j * stride_3384 + offset_3385) + (long) i] = (GFReal) ktop[(long) i + offset_3442] + 1.000000047497451305389404296875e-3f;
                            }

                          if ((GFReal) kbcon[(long) i + offset_3416] < hbot[((long) j * stride_3363 + offset_3364) + (long) i])
                            {
                              hbot[((long) j * stride_3363 + offset_3364) + (long) i] = (GFReal) kbcon[(long) i + offset_3416] + 1.000000047497451305389404296875e-3f;
                            }

                        }


                      i = i + 1;
                    }
                  }

              }
              {
                int D_14157;
                int D_14158;

                D_14157 = *kts;
                D_14158 = *kte;
                k = D_14157;
                while (1)
                  {
                    {
                      bool D_14161;

                      D_14161 = k > D_14158;
                      if (D_14161) break;
                      {
                        int D_14162;
                        int D_14163;

                        D_14162 = *its;
                        D_14163 = *ite;
                        i = D_14162;
                        while (1)
                          {
                            {
                              bool D_14166;

                              D_14166 = i > D_14163;
                              if (D_14166) break;
                              rthcuten[(((long) j * stride_3717 + offset_3718) + (long) k * stride_3714) + (long) i] = ((outts[((long) k * stride_3507 + offset_3508) + (long) i] + ((subt[((long) k * stride_3758 + offset_3759) + (long) i] + outt[((long) k * stride_3500 + offset_3501) + (long) i])) * cuten[(long) i + offset_3288])) / pi[(((long) j * stride_3551 + offset_3552) + (long) k * stride_3548) + (long) i];
                              rqvcuten[(((long) j * stride_3687 + offset_3688) + (long) k * stride_3684) + (long) i] = outqs[((long) k * stride_3493 + offset_3494) + (long) i] + ((subq[((long) k * stride_3751 + offset_3752) + (long) i] + outq[((long) k * stride_3472 + offset_3473) + (long) i])) * cuten[(long) i + offset_3288];
                              if (t2d[((long) k * stride_3775 + offset_3776) + (long) i] < 2.58e+2f)
                                {
                                  rqccuten[(((long) j * stride_3657 + offset_3658) + (long) k * stride_3654) + (long) i] = 0.0f;
                                  rqicuten[(((long) j * stride_3667 + offset_3668) + (long) k * stride_3664) + (long) i] = outqcs[((long) k * stride_3486 + offset_3487) + (long) i] + outqc[((long) k * stride_3479 + offset_3480) + (long) i] * cuten[(long) i + offset_3288];
                                  if (gdc2 != nullptr)
                                    {
                                      gdc2[(((long) j * stride_3342 + offset_3343) + (long) k * stride_3339) + (long) i] = cupclws[((long) k * stride_3283 + offset_3284) + (long) i] + cupclw[((long) k * stride_3276 + offset_3277) + (long) i] * cuten[(long) i + offset_3288];
                                    }

                                }
                              else
                                {
                                  rqicuten[(((long) j * stride_3667 + offset_3668) + (long) k * stride_3664) + (long) i] = 0.0f;
                                  rqccuten[(((long) j * stride_3657 + offset_3658) + (long) k * stride_3654) + (long) i] = outqcs[((long) k * stride_3486 + offset_3487) + (long) i] + outqc[((long) k * stride_3479 + offset_3480) + (long) i] * cuten[(long) i + offset_3288];
                                  if (gdc != nullptr)
                                    {
                                      gdc[(((long) j * stride_3332 + offset_3333) + (long) k * stride_3329) + (long) i] = cupclws[((long) k * stride_3283 + offset_3284) + (long) i] + cupclw[((long) k * stride_3276 + offset_3277) + (long) i] * cuten[(long) i + offset_3288];
                                    }

                                }


                              i = i + 1;
                            }
                          }

                      }

                      k = k + 1;
                    }
                  }

              }

              j = j + 1;
            }
          }

      }
    }
    {
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
      ; // reclaimed at procedure return
    }
}

// MPAS module_cu_gf.mpas.F:54-453; one complete column, original 1-based bounds.
// MPAS mpas_atmphys_manager.F:655-663: kme=kde=nz+1 and ktf=kte=nz.
// Payloads contain nz active levels; caller interface words are held.
// Native local-array index expressions, including its defects, are preserved.
static inline void gf_column(int nz,device GFReal* lev,device GFReal* surface,device int* isurface,
 device GFReal* outlev,device GFReal* outsurface,device int* outint,thread int* control,
 thread GFReal* constants,thread GFWorkspace &workspace){
 int one=1,two=2,nplus=nz+1;
 cu_grell_freitas(workspace,control,&surface[7],&surface[5],&surface[6],
 lev+0*nz,lev+1*nz,lev+3*nz,lev+2*nz,lev+4*nz,lev+5*nz,lev+6*nz,lev+7*nz,
 lev+8*nz,lev+9*nz,surface+3,surface+0,surface+1,surface+2,surface+4,
 lev+11*nz,lev+10*nz,lev+13*nz,lev+14*nz,lev+12*nz,isurface,
 constants+0,constants+1,constants+2,constants+3,
 control+1,control+2,control+3,
 outsurface+0,outsurface+1,outint+0,outint+1,outint+2,outsurface+2,outsurface+3,
 outsurface+4,outsurface+5,outlev+4*nz,outlev+5*nz,
 outlev+0*nz,outlev+1*nz,outlev+2*nz,outlev+3*nz,
 &one,&one,&one,&one,&one,&nplus,&one,&two,&one,&two,&one,&nplus,
 &one,&one,&one,&one,&one,&nz);
}

#include "hexv842_mpas_gf_column_f32_bindings.metalh"
kernel void hexv842_mpas_gf_column_f32(constant Hexv842MpasGfColumnF32Resources &r [[buffer(0)]], constant Hexv842MpasGfColumnF32Params &p [[buffer(1)]], uint lane [[thread_position_in_grid]]) {
 if(lane!=0)return; uint col=uint(p.column);
 int control[4]={p.itimestep,p.deep,p.shallow,1};GFReal constants[4]={p.xlv,p.cp,p.gravity,p.rv};
 ulong cap=ulong(p.scratch_words)*4ul;
 GFWorkspace ws={(device char*)r.scratch,cap,0,0,0};
 gf_column(p.nz,(device GFReal*)(r.levels+col*15*p.nz),(device GFReal*)(r.surface+col*8),r.kpbl+col,(device GFReal*)(r.outlev+col*6*p.nz),(device GFReal*)(r.outsurface+col*6),r.outint+col*3,control,constants,ws);
 r.highwater[col]=uint(ws.highwater);
}
