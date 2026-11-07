#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_seaice_math.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "exact_binary32.metalh"
#include "hexrrfs_mynnsfc_v600_bindings.metalh"
#pragma clang fp contract(off)
static inline bool operator<(Exact32 a,Exact32 b){return soft32_less(a.bits,b.bits);}
static inline bool operator<=(Exact32 a,Exact32 b){return soft32_less_equal(a.bits,b.bits);}
static inline bool operator==(Exact32 a,Exact32 b){return soft32_equal(a.bits,b.bits);}
static inline bool operator!=(Exact32 a,Exact32 b){return !soft32_equal(a.bits,b.bits);}
static inline Exact32 rr_add(Exact32 a,Exact32 b){return exact_add(a,b);}
static inline Exact32 rr_sub(Exact32 a,Exact32 b){return exact_sub(a,b);}
static inline Exact32 rr_mul(Exact32 a,Exact32 b){return exact_mul(a,b);}
static inline Exact32 rr_div(Exact32 a,Exact32 b){return exact_div(a,b);}
static inline Exact32 rr_sqrt(Exact32 a){return exact_sqrt(a);}
static inline Exact32 rr_abs(Exact32 a){return exact_abs(a);}
static inline Exact32 rr_min(Exact32 a,Exact32 b){return exact_min(a,b);}
static inline Exact32 rr_max(Exact32 a,Exact32 b){return exact_max(a,b);}
static inline Exact32 rr_atan(Exact32 a){return Exact32::from_bits(soft_glibc_solar_atanf(a.bits));}
static inline Exact32 rr_pow(Exact32 a,Exact32 b){return Exact32::from_bits(seaice_pow_bits(a.bits,b.bits));}
static inline Exact32 rr_exp(Exact32 a){return Exact32(seaice_exp(as_type<float>(a.bits)));}
static inline Exact32 rr_log(Exact32 x){
    uint ix=x.bits;
    if(ix==0x3f800000u)return Exact32::from_bits(0u);
    if(ix-0x00800000u>=0x7f800000u-0x00800000u){
        if(ix*2u==0u)return Exact32::from_bits(0xff800000u);
        if(ix==0x7f800000u)return x;
        if((ix&0x80000000u)||ix*2u>=0xff000000u)return Exact32::from_bits(0x7fc00000u);
        ix=soft32_mul(ix,0x4b000000u)-(23u<<23);
    }
    uint tmp=ix-0x3f330000u,i=(tmp>>19)&15u,iz=ix-(tmp&0xff800000u);
    int k=as_type<int>(tmp)>>23;
    ulong r=soft64_fma(soft64_from_f32_bits(iz),SOFT_GFK_LOGF_INVC[i],0xbff0000000000000ul);
    ulong y0=soft64_fma(soft64_from_i32(k),SOFT_GFK_LOGF_LN2,SOFT_GFK_LOGF_LOGC[i]);
    ulong r2=soft64_mul(r,r),y=soft64_fma(SOFT_GFK_LOGF_A1,r,SOFT_GFK_LOGF_A2);
    y=soft64_fma(r2,SOFT_GFK_LOGF_A0,y);
    return Exact32::from_bits(soft64_to_f32_bits(soft64_fma(r2,y,soft64_add(y0,r))));
}
// Retained interpolation indices are converted only for finite in-range inputs.
static inline int rr_int(Exact32 a){
    uint bits=a.bits, magnitude=bits&0x7fffffffu;
    int exponent=int(magnitude>>23)-127;
    if(exponent<0)return 0;
    uint significand=(magnitude&0x7fffffu)|0x800000u;
    uint integer=exponent<23?significand>>uint(23-exponent):significand<<uint(exponent-23);
    return (bits&0x80000000u)?-int(integer):int(integer);
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:120-903 prototype land_mynnsfc_land
static inline void m_land_mynnsfc_land(bool flag_iter, int itimestep, int i, int j, Exact32 dx, Exact32 xland, Exact32 u_1, Exact32 v_1, Exact32 t_1, Exact32 qv_1, Exact32 p_1, Exact32 dz8w_1, thread Exact32 &rho_1, Exact32 u_2, Exact32 v_2, Exact32 dz8w_2, Exact32 sigmaf, int vegtype, Exact32 shdmax, int ivegsrc, Exact32 z0pert, Exact32 ztpert, bool redrag, int sfc_z0_type, Exact32 pblh, thread Exact32 &znt, Exact32 psfcpa, Exact32 mavail, Exact32 tskin, Exact32 tsurf, Exact32 snowh, thread Exact32 &qgh, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs2, thread Exact32 &cqs, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &stress, thread Exact32 &qsfc, thread Exact32 &rmol, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &psim, thread Exact32 &psih, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &t2, thread Exact32 &q2, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &lh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &rb, thread Exact32 &cpm, thread Exact32 &ch, thread Exact32 &cm, Exact32 rstoch_1, thread Exact32 &wstar, thread Exact32 &qstar, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &psix, thread Exact32 &psit, thread Exact32 &psix10, thread Exact32 &psit2, int spp_sfc, int sf_mynn_sfcflux_land, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, device const float *psim_stab, device const float *psim_unstab, device const float *psih_stab, device const float *psih_unstab, thread int &errflg, bool has_sf_mynn_sfcflux_land, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:919-954 prototype land_zilitinkevich_1995
static inline void m_land_zilitinkevich_1995(Exact32 z_0, thread Exact32 &zt, thread Exact32 &zq, Exact32 restar, Exact32 ustar, Exact32 karman, int zt_opt, int lsm, int lsm_ruc, bool has_zt_opt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:963-984 prototype land_garratt_1992
static inline void m_land_garratt_1992(thread Exact32 &zt, thread Exact32 &zq, Exact32 z_0, Exact32 ren, Exact32 landsea);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1010-1041 prototype land_yang_2008
static inline void m_land_yang_2008(Exact32 z_0, thread Exact32 &zt, thread Exact32 &zq, Exact32 ustar, Exact32 tstar, Exact32 qst, Exact32 ren, Exact32 visc);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1045-1096 prototype land_gfs_z0_land
static inline void m_land_gfs_z0_land(thread Exact32 &z0max, Exact32 shdmax, Exact32 z1, int vegtype, int ivegsrc, Exact32 z0pert);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1100-1124 prototype land_gfs_zt_land
static inline void m_land_gfs_zt_land(thread Exact32 &ztmax, Exact32 z0max, Exact32 sigmaf, Exact32 ztpert, Exact32 ustar);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1129-1158 prototype land_psi_hogstrom_1996
static inline void m_land_psi_hogstrom_1996(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1165-1195 prototype land_psi_dyerhicks
static inline void m_land_psi_dyerhicks(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1201-1224 prototype land_psi_beljaars_holtslag_1991
static inline void m_land_psi_beljaars_holtslag_1991(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1231-1253 prototype land_psi_zilitinkevich_esau_2007
static inline void m_land_psi_zilitinkevich_esau_2007(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1258-1283 prototype land_psi_businger_1971
static inline void m_land_psi_businger_1971(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1289-1300 prototype land_psi_cb2005
static inline void m_land_psi_cb2005(thread Exact32 &psim1, thread Exact32 &psih1, Exact32 zl, Exact32 z0l);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1308-1364 prototype land_li_etal_2010
static inline void m_land_li_etal_2010(thread Exact32 &zl, Exact32 rib, Exact32 zaz0, Exact32 z0zt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1367-1420 prototype land_zolri
static inline Exact32 m_land_zolri(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1422-1462 prototype land_zolri2
static inline Exact32 m_land_zolri2(thread Exact32 &zol2, Exact32 ri2, Exact32 za, Exact32 z0, Exact32 zt, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1465-1539 prototype land_zolrib
static inline Exact32 m_land_zolrib(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 logz0, Exact32 logzt, thread Exact32 &zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1544-1551 prototype land_psim_stable_full
static inline Exact32 m_land_psim_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1554-1561 prototype land_psih_stable_full
static inline Exact32 m_land_psih_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1564-1579 prototype land_psim_unstable_full
static inline Exact32 m_land_psim_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1582-1596 prototype land_psih_unstable_full
static inline Exact32 m_land_psih_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1603-1612 prototype land_psim_stable_full_gfs
static inline Exact32 m_land_psim_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1616-1625 prototype land_psih_stable_full_gfs
static inline Exact32 m_land_psih_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1629-1645 prototype land_psim_unstable_full_gfs
static inline Exact32 m_land_psim_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1649-1665 prototype land_psih_unstable_full_gfs
static inline Exact32 m_land_psih_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1669-1686 prototype land_psim_stable
static inline Exact32 m_land_psim_stable(Exact32 zolf, int psi_opt, device const float *psim_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1689-1706 prototype land_psih_stable
static inline Exact32 m_land_psih_stable(Exact32 zolf, int psi_opt, device const float *psih_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1709-1726 prototype land_psim_unstable
static inline Exact32 m_land_psim_unstable(Exact32 zolf, int psi_opt, device const float *psim_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1729-1746 prototype land_psih_unstable
static inline Exact32 m_land_psih_unstable(Exact32 zolf, int psi_opt, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:122-919 prototype water_mynnsfc_water
static inline void m_water_mynnsfc_water(bool flag_iter, int itimestep, int i, int j, Exact32 dx, Exact32 xland, Exact32 lakemask, Exact32 wat_depth, Exact32 u_1, Exact32 v_1, Exact32 t_1, Exact32 qv_1, Exact32 p_1, Exact32 dz8w_1, thread Exact32 &rho_1, Exact32 u_2, Exact32 v_2, Exact32 dz8w_2, Exact32 sigmaf, int vegtype, Exact32 shdmax, int ivegsrc, Exact32 z0pert, Exact32 ztpert, bool redrag, int sfc_z0_type, Exact32 pblh, thread Exact32 &znt, Exact32 psfcpa, Exact32 mavail, Exact32 tskin, Exact32 tsurf, Exact32 snowh, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs2, thread Exact32 &cqs, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &stress, thread Exact32 &rmol, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &psim, thread Exact32 &psih, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &qsfc, thread Exact32 &t2, thread Exact32 &q2, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &lh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &rb, thread Exact32 &cpm, thread Exact32 &ch, thread Exact32 &cm, Exact32 rstoch_1, thread Exact32 &wstar, thread Exact32 &qstar, thread Exact32 &qgh, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &psix, thread Exact32 &psit, thread Exact32 &psix10, thread Exact32 &psit2, int spp_sfc, int sf_mynn_sfcflux_water, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int shalwater_z0, int iter, int lsm, int lsm_ruc, device const float *psim_stab, device const float *psim_unstab, device const float *psih_stab, device const float *psih_unstab, thread int &errflg, bool has_sf_mynn_sfcflux_water, bool has_shalwater_z0, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:924-950 prototype water_davis_etal_2008
static inline void m_water_davis_etal_2008(thread Exact32 &z_0, Exact32 ustar);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:955-974 prototype water_taylor_yelland_2001
static inline void m_water_taylor_yelland_2001(thread Exact32 &z_0, Exact32 ustar, Exact32 wsp10);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:979-995 prototype water_coare30_znt
static inline void m_water_coare30_znt(thread Exact32 &z_0, Exact32 ustar, Exact32 wsp10, Exact32 visc, Exact32 zu);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1002-1020 prototype water_edson_etal_2013
static inline void m_water_edson_etal_2013(thread Exact32 &z_0, Exact32 ustar, Exact32 wsp10, Exact32 visc, Exact32 zu);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1027-1051 prototype water_depth_dependent_z0
static inline Exact32 m_water_depth_dependent_z0(Exact32 water_depth, Exact32 z0, Exact32 ust);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1061-1082 prototype water_garratt_1992
static inline void m_water_garratt_1992(thread Exact32 &zt, thread Exact32 &zq, Exact32 z_0, Exact32 ren, Exact32 landsea);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1093-1127 prototype water_fairall_etal_2003
static inline void m_water_fairall_etal_2003(thread Exact32 &zt, thread Exact32 &zq, Exact32 ren, Exact32 ustar, Exact32 visc, Exact32 rstoch, int spp_sfc);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1135-1157 prototype water_fairall_etal_2014
static inline void m_water_fairall_etal_2014(thread Exact32 &zt, thread Exact32 &zq, Exact32 ren, Exact32 ustar, Exact32 visc, Exact32 rstoch, int spp_sfc);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1160-1209 prototype water_gfs_z0_wat
static inline void m_water_gfs_z0_wat(thread Exact32 &z0rl, thread Exact32 &ustar, Exact32 wspd, Exact32 z1, int sfc_z0_type, bool redrag);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1212-1257 prototype water_gfs_zt_wat
static inline void m_water_gfs_zt_wat(thread Exact32 &ztmax, Exact32 z0rl, Exact32 restar, Exact32 wspd, Exact32 z1, int sfc_z0_type, thread int &errflg);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1262-1304 prototype water_znot_m_v6
static inline void m_water_znot_m_v6(Exact32 uref, thread Exact32 &znotm);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1315-1368 prototype water_znot_t_v6
static inline void m_water_znot_t_v6(Exact32 uref, thread Exact32 &znott);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1378-1415 prototype water_znot_m_v7
static inline void m_water_znot_m_v7(Exact32 uref, thread Exact32 &znotm);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1424-1478 prototype water_znot_t_v7
static inline void m_water_znot_t_v7(Exact32 uref, thread Exact32 &znott);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1484-1513 prototype water_psi_hogstrom_1996
static inline void m_water_psi_hogstrom_1996(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1520-1550 prototype water_psi_dyerhicks
static inline void m_water_psi_dyerhicks(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1556-1579 prototype water_psi_beljaars_holtslag_1991
static inline void m_water_psi_beljaars_holtslag_1991(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1584-1609 prototype water_psi_businger_1971
static inline void m_water_psi_businger_1971(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1618-1640 prototype water_psi_suselj_sood_2010
static inline void m_water_psi_suselj_sood_2010(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1646-1657 prototype water_psi_cb2005
static inline void m_water_psi_cb2005(thread Exact32 &psim1, thread Exact32 &psih1, Exact32 zl, Exact32 z0l);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1663-1719 prototype water_li_etal_2010
static inline void m_water_li_etal_2010(thread Exact32 &zl, Exact32 rib, Exact32 zaz0, Exact32 z0zt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1722-1775 prototype water_zolri
static inline Exact32 m_water_zolri(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1777-1817 prototype water_zolri2
static inline Exact32 m_water_zolri2(thread Exact32 &zol2, Exact32 ri2, Exact32 za, Exact32 z0, Exact32 zt, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1820-1894 prototype water_zolrib
static inline Exact32 m_water_zolrib(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 logz0, Exact32 logzt, thread Exact32 &zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1899-1906 prototype water_psim_stable_full
static inline Exact32 m_water_psim_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1909-1916 prototype water_psih_stable_full
static inline Exact32 m_water_psih_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1919-1934 prototype water_psim_unstable_full
static inline Exact32 m_water_psim_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1937-1951 prototype water_psih_unstable_full
static inline Exact32 m_water_psih_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1958-1967 prototype water_psim_stable_full_gfs
static inline Exact32 m_water_psim_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1971-1980 prototype water_psih_stable_full_gfs
static inline Exact32 m_water_psih_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1984-2000 prototype water_psim_unstable_full_gfs
static inline Exact32 m_water_psim_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2004-2020 prototype water_psih_unstable_full_gfs
static inline Exact32 m_water_psih_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2024-2041 prototype water_psim_stable
static inline Exact32 m_water_psim_stable(Exact32 zolf, int psi_opt, device const float *psim_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2044-2061 prototype water_psih_stable
static inline Exact32 m_water_psih_stable(Exact32 zolf, int psi_opt, device const float *psih_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2064-2081 prototype water_psim_unstable
static inline Exact32 m_water_psim_unstable(Exact32 zolf, int psi_opt, device const float *psim_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2084-2101 prototype water_psih_unstable
static inline Exact32 m_water_psih_unstable(Exact32 zolf, int psi_opt, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:116-836 prototype ice_mynnsfc_ice
static inline void m_ice_mynnsfc_ice(bool flag_iter, int itimestep, int i, int j, Exact32 dx, Exact32 xland, Exact32 u_1, Exact32 v_1, Exact32 t_1, Exact32 qv_1, Exact32 p_1, Exact32 dz8w_1, thread Exact32 &rho_1, Exact32 u_2, Exact32 v_2, Exact32 dz8w_2, Exact32 pblh, thread Exact32 &znt, Exact32 psfcpa, Exact32 mavail, Exact32 tskin, Exact32 tsurf, Exact32 snowh, thread Exact32 &qgh, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs2, thread Exact32 &cqs, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &stress, thread Exact32 &rmol, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &psim, thread Exact32 &psih, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &qsfc, thread Exact32 &t2, thread Exact32 &q2, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &lh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &rb, thread Exact32 &cpm, thread Exact32 &ch, thread Exact32 &cm, Exact32 rstoch_1, thread Exact32 &wstar, thread Exact32 &qstar, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &psix, thread Exact32 &psit, thread Exact32 &psix10, thread Exact32 &psit2, int spp_sfc, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, device const float *psim_stab, device const float *psim_unstab, device const float *psih_stab, device const float *psih_unstab, thread int &errflg, bool has_ck, bool has_cka, bool has_cd, bool has_cda);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:846-898 prototype ice_andreas_2002
static inline void m_ice_andreas_2002(Exact32 z_0, Exact32 bvisc, Exact32 ustar, thread Exact32 &zt, thread Exact32 &zq, int spp_sfc, Exact32 rstoch);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:903-932 prototype ice_psi_hogstrom_1996
static inline void m_ice_psi_hogstrom_1996(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:939-969 prototype ice_psi_dyerhicks
static inline void m_ice_psi_dyerhicks(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:975-998 prototype ice_psi_beljaars_holtslag_1991
static inline void m_ice_psi_beljaars_holtslag_1991(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1005-1027 prototype ice_psi_zilitinkevich_esau_2007
static inline void m_ice_psi_zilitinkevich_esau_2007(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1032-1057 prototype ice_psi_businger_1971
static inline void m_ice_psi_businger_1971(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1066-1088 prototype ice_psi_suselj_sood_2010
static inline void m_ice_psi_suselj_sood_2010(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1094-1105 prototype ice_psi_cb2005
static inline void m_ice_psi_cb2005(thread Exact32 &psim1, thread Exact32 &psih1, Exact32 zl, Exact32 z0l);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1111-1167 prototype ice_li_etal_2010
static inline void m_ice_li_etal_2010(thread Exact32 &zl, Exact32 rib, Exact32 zaz0, Exact32 z0zt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1170-1223 prototype ice_zolri
static inline Exact32 m_ice_zolri(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1225-1265 prototype ice_zolri2
static inline Exact32 m_ice_zolri2(thread Exact32 &zol2, Exact32 ri2, Exact32 za, Exact32 z0, Exact32 zt, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1268-1342 prototype ice_zolrib
static inline Exact32 m_ice_zolrib(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 logz0, Exact32 logzt, thread Exact32 &zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1347-1354 prototype ice_psim_stable_full
static inline Exact32 m_ice_psim_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1357-1364 prototype ice_psih_stable_full
static inline Exact32 m_ice_psih_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1367-1382 prototype ice_psim_unstable_full
static inline Exact32 m_ice_psim_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1385-1399 prototype ice_psih_unstable_full
static inline Exact32 m_ice_psih_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1406-1415 prototype ice_psim_stable_full_gfs
static inline Exact32 m_ice_psim_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1419-1428 prototype ice_psih_stable_full_gfs
static inline Exact32 m_ice_psih_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1432-1448 prototype ice_psim_unstable_full_gfs
static inline Exact32 m_ice_psim_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1452-1468 prototype ice_psih_unstable_full_gfs
static inline Exact32 m_ice_psih_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1472-1489 prototype ice_psim_stable
static inline Exact32 m_ice_psim_stable(Exact32 zolf, int psi_opt, device const float *psim_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1492-1509 prototype ice_psih_stable
static inline Exact32 m_ice_psih_stable(Exact32 zolf, int psi_opt, device const float *psih_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1512-1529 prototype ice_psim_unstable
static inline Exact32 m_ice_psim_unstable(Exact32 zolf, int psi_opt, device const float *psim_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1532-1549 prototype ice_psih_unstable
static inline Exact32 m_ice_psih_unstable(Exact32 zolf, int psi_opt, device const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87 prototype drv_mynnsfc_init
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:91-101 prototype drv_mynnsfclay_finalize
static inline void m_drv_mynnsfclay_finalize(thread int &errflg);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740 prototype drv_mynnsfc_driver
static inline void m_drv_mynnsfc_driver(Exact32 u3d2, Exact32 v3d2, Exact32 dz8w2, Exact32 u3d, Exact32 v3d, Exact32 t3d, Exact32 qv3d, Exact32 p3d, Exact32 dz8w, Exact32 th3d, Exact32 rho3d, Exact32 sigmaf, int vegtype, Exact32 shdmax, int ivegsrc, Exact32 z0pert, Exact32 ztpert, thread bool &redrag, thread int &sfc_z0_type, Exact32 psfcpa, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs, thread Exact32 &cqs2, thread Exact32 &cpm, thread Exact32 &znt, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &pblh, Exact32 mavail, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &rmol, thread Exact32 &psim, thread Exact32 &psih, Exact32 xland, thread Exact32 &qgh, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &lh, Exact32 tsk, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &qsfc, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &t2, Exact32 lakemask, thread Exact32 &q2, Exact32 snowh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &br, Exact32 dx, thread Exact32 &ch, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &stress, thread Exact32 &hflx, thread Exact32 &qflx, thread Exact32 &cm, thread Exact32 &fm, thread Exact32 &fh, thread Exact32 &fm10, thread Exact32 &fh2, Exact32 tsurf, Exact32 water_depth, Exact32 xice, Exact32 xice_threshold, int spp_pbl, Exact32 pattern_spp_pbl, int sf_mynn_sfcflux_water, int sf_mynn_sfcflux_land, int shalwater_z0, int isfflx, bool restart, bool cycling, int initflag, thread bool &flag_iter, int flagc_lsm, int itimestep, int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, thread int &errflg, bool has_restart, bool has_cycling, bool has_initflag, bool has_sf_mynn_sfcflux_water, bool has_sf_mynn_sfcflux_land, bool has_shalwater_z0, bool has_flagc_lsm, bool has_spp_pbl, bool has_ivegsrc, bool has_sfc_z0_type, bool has_redrag, bool has_flag_iter, bool has_vegtype, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_pattern_spp_pbl, bool has_tsurf, bool has_lakemask, bool has_water_depth, bool has_cm, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm, bool has_fm, bool has_fh, bool has_fm10, bool has_fh2, bool has_hflx, bool has_qflx, bool has_stress, device const float *tables);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:745-788 prototype drv_psi_init
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:794-801 prototype drv_psim_stable_full
static inline Exact32 m_drv_psim_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:804-811 prototype drv_psih_stable_full
static inline Exact32 m_drv_psih_stable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:814-829 prototype drv_psim_unstable_full
static inline Exact32 m_drv_psim_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:832-846 prototype drv_psih_unstable_full
static inline Exact32 m_drv_psih_unstable_full(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:853-862 prototype drv_psim_stable_full_gfs
static inline Exact32 m_drv_psim_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:866-875 prototype drv_psih_stable_full_gfs
static inline Exact32 m_drv_psih_stable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:879-895 prototype drv_psim_unstable_full_gfs
static inline Exact32 m_drv_psim_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:899-915 prototype drv_psih_unstable_full_gfs
static inline Exact32 m_drv_psih_unstable_full_gfs(Exact32 zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:120-903 land_mynnsfc_land
static inline void m_land_mynnsfc_land(bool flag_iter, int itimestep, int i, int j, Exact32 dx, Exact32 xland, Exact32 u_1, Exact32 v_1, Exact32 t_1, Exact32 qv_1, Exact32 p_1, Exact32 dz8w_1, thread Exact32 &rho_1, Exact32 u_2, Exact32 v_2, Exact32 dz8w_2, Exact32 sigmaf, int vegtype, Exact32 shdmax, int ivegsrc, Exact32 z0pert, Exact32 ztpert, bool redrag, int sfc_z0_type, Exact32 pblh, thread Exact32 &znt, Exact32 psfcpa, Exact32 mavail, Exact32 tskin, Exact32 tsurf, Exact32 snowh, thread Exact32 &qgh, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs2, thread Exact32 &cqs, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &stress, thread Exact32 &qsfc, thread Exact32 &rmol, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &psim, thread Exact32 &psih, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &t2, thread Exact32 &q2, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &lh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &rb, thread Exact32 &cpm, thread Exact32 &ch, thread Exact32 &cm, Exact32 rstoch_1, thread Exact32 &wstar, thread Exact32 &qstar, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &psix, thread Exact32 &psit, thread Exact32 &psix10, thread Exact32 &psit2, int spp_sfc, int sf_mynn_sfcflux_land, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, device const float *psim_stab, device const float *psim_unstab, device const float *psih_stab, device const float *psih_unstab, thread int &errflg, bool has_sf_mynn_sfcflux_land, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda) {
Exact32 xka = 0x1.92a7380000000p-16f;
Exact32 prt = 0x1.0000000000000p+0f;
Exact32 za;
Exact32 za2;
Exact32 thv_1;
Exact32 th_1;
Exact32 tc_1;
Exact32 tv_1;
Exact32 qvmr;
Exact32 psih2;
Exact32 psim10;
Exact32 psih10;
Exact32 wspdi;
Exact32 govrth;
Exact32 psfc;
Exact32 qsfcmr;
Exact32 thcon;
Exact32 zratio;
Exact32 tsk;
Exact32 thsk;
Exact32 thvsk;
Exact32 gz1ozt;
Exact32 gz2oz0;
Exact32 gz2ozt;
Exact32 gz10oz0;
Exact32 gz10ozt;
Exact32 zntstoch;
Exact32 zt;
Exact32 zq;
Exact32 psiq;
Exact32 psiq2;
int n;
int k;
int l;
int yesno;
Exact32 pl;
Exact32 e1;
Exact32 tabs;
Exact32 dthvdz;
Exact32 dthvm;
Exact32 vconv;
Exact32 zol2;
Exact32 zol10;
Exact32 zolza;
Exact32 zolz0;
Exact32 zolzt;
Exact32 dtg;
Exact32 dtthx;
Exact32 psiq10;
Exact32 fluxc;
Exact32 vsgd;
Exact32 restar;
Exact32 visc;
Exact32 dqg;
Exact32 oldust;
Exact32 oldtst;
errflg = 0;
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
psfc = rr_div(psfcpa,0x1.f400000000000p+9f);
if (flag_iter) {
if ((itimestep==1)) {
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = rr_div(qsfc,rr_sub(0x1.0000000000000p+0f,qsfc));
} else {
tabs = rr_add(rr_mul(0x1.fae1480000000p-1f,tskin),rr_mul(0x1.47ae140000000p-7f,t_1));
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_div(rr_mul(0x1.1ab8520000000p+4f,rr_sub(tabs,0x1.1126660000000p+8f)),rr_sub(tabs,0x1.da66660000000p+4f))));
qsfc = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,rr_mul(0x1.8353dc0000000p-2f,e1)));
qsfcmr = rr_div(qsfc,rr_sub(0x1.0000000000000p+0f,qsfc));
}
} else {
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = rr_div(qsfc,rr_sub(0x1.0000000000000p+0f,qsfc));
} else {
tabs = tskin;
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_div(rr_mul(0x1.1ab8520000000p+4f,rr_sub(tabs,0x1.1126660000000p+8f)),rr_sub(tabs,0x1.da66660000000p+4f))));
qsfc = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,rr_mul(0x1.8353dc0000000p-2f,e1)));
qsfcmr = rr_div(qsfc,rr_sub(0x1.0000000000000p+0f,qsfc));
}
}
}
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_div(rr_mul(0x1.1ab8520000000p+4f,rr_sub(t_1,0x1.1126660000000p+8f)),rr_sub(t_1,0x1.da66660000000p+4f))));
pl = rr_div(p_1,0x1.f400000000000p+9f);
qgh = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(pl,e1));
qvmr = rr_div(qv_1,rr_sub(0x1.0000000000000p+0f,qv_1));
thcon = rr_pow(rr_div(0x1.86a0000000000p+16f,psfcpa),0x1.24924a0000000p-2f);
if (flag_iter) {
tsk = tskin;
thsk = rr_mul(tsk,thcon);
thvsk = rr_mul(thsk,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qsfc)));
}
th_1 = rr_mul(t_1,rr_pow(rr_div(0x1.86a0000000000p+16f,p_1),0x1.24924a0000000p-2f));
tc_1 = rr_sub(t_1,0x1.1126660000000p+8f);
thv_1 = rr_mul(th_1,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qv_1)));
tv_1 = rr_mul(t_1,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qv_1)));
rho_1 = rr_div(p_1,rr_mul(0x1.1f00000000000p+8f,tv_1));
za = rr_mul(0x1.0000000000000p-1f,dz8w_1);
za2 = rr_add(dz8w_1,rr_mul(0x1.0000000000000p-1f,dz8w_2));
govrth = rr_div(0x1.39cc100000000p+3f,th_1);
cpm = rr_mul(0x1.f640000000000p+9f,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.ae147a0000000p-1f,qv_1)));
if (flag_iter) {
wspd = rr_sqrt(rr_add(rr_mul(u_1,u_1),rr_mul(v_1,v_1)));
dthvdz = rr_sub(thv_1,thvsk);
fluxc = rr_max(rr_add(rr_div(rr_div(hfx,rho_1),0x1.f640000000000p+9f),rr_div(rr_mul(rr_mul(0x1.377b480000000p-1f,thvsk),qfx),rho_1)),0x0.0p+0f);
wstar = rr_mul(0x1.4000000000000p+0f,rr_pow(rr_mul(rr_mul(rr_div(0x1.39cc100000000p+3f,tsk),rr_min(rr_mul(0x1.8000000000000p+0f,pblh),0x1.f400000000000p+11f)),fluxc),0x1.5555560000000p-2f));
vsgd = rr_min(rr_mul(0x1.47ae140000000p-2f,rr_pow(rr_max(rr_sub(rr_div(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.5555560000000p-2f)),0x1.0000000000000p-1f);
wspd = rr_sqrt(rr_add(rr_add(rr_mul(wspd,wspd),rr_mul(wstar,wstar)),rr_mul(vsgd,vsgd)));
wspd = rr_max(wspd,0x1.99999a0000000p-4f);
rb = rr_div(rr_mul(rr_mul(govrth,za),dthvdz),rr_mul(wspd,wspd));
rb = rr_max(rb,-0x1.0000000000000p+1f);
rb = rr_min(rb,0x1.0000000000000p+1f);
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
}
if (flag_iter) {
visc = rr_mul(0x1.bcee880000000p-17f,rr_sub(rr_add(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.acbc8c0000000p-8f,tc_1)),rr_mul(rr_mul(0x1.16890c0000000p-17f,tc_1),tc_1)),rr_mul(rr_mul(rr_mul(0x1.4c9a2e0000000p-28f,tc_1),tc_1),tc_1)));
if ((sf_mynn_sfcflux_land==4)) {
m_land_gfs_z0_land(znt,shdmax,za,vegtype,ivegsrc,z0pert);
}
if ((spp_sfc==1)) {
zntstoch = rr_max(rr_add(znt,rr_mul(rr_mul(znt,0x1.0000000000000p+0f),rstoch_1)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
zntstoch = rr_min(zntstoch,rr_mul(dz8w_1,0x1.10cb2a0000000p-4f));
restar = rr_max(rr_div(rr_mul(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if (has_sf_mynn_sfcflux_land) {
if ((sf_mynn_sfcflux_land<=1)) {
m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,sf_mynn_sfcflux_land,lsm,lsm_ruc,has_sf_mynn_sfcflux_land);
} else if ((sf_mynn_sfcflux_land==2)) {
m_land_yang_2008(zntstoch,zt,zq,ust,mol,qstar,restar,visc);
} else if ((sf_mynn_sfcflux_land==3)) {
m_land_garratt_1992(zt,zq,zntstoch,restar,0x1.0000000000000p+0f);
} else if ((sf_mynn_sfcflux_land==4)) {
m_land_gfs_zt_land(zt,zntstoch,sigmaf,ztpert,ust);
zq = zt;
}
} else {
m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,0,lsm,lsm_ruc,true);
}
if ((spp_sfc==1)) {
zt = rr_add(zt,rr_mul(rr_mul(zt,0x1.8000000000000p-1f),rstoch_1));
zq = rr_add(zq,rr_mul(rr_mul(zq,0x1.8000000000000p-1f),rstoch_1));
zt = rr_min(zt,rr_mul(0x1.cccccc0000000p-1f,zntstoch));
zt = rr_max(zt,0x1.a36e2e0000000p-14f);
zq = rr_min(zq,rr_mul(0x1.cccccc0000000p-1f,zntstoch));
zq = rr_max(zq,0x1.a36e2e0000000p-14f);
}
gz1oz0 = rr_log(rr_div(rr_add(za,zntstoch),zntstoch));
gz1ozt = rr_log(rr_div(rr_add(za,zntstoch),zt));
gz2oz0 = rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zntstoch),zt));
gz10oz0 = rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zntstoch),zt));
zratio = rr_div(zntstoch,zt);
if ((rb>0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_land_li_etal_2010(zol,rb,rr_div(za,zntstoch),zratio);
zol = rr_max(zol,0x0.0p+0f);
zol = rr_min(zol,0x1.4000000000000p+4f);
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_land_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = rr_max(zol,0x0.0p+0f);
zol = rr_min(zol,0x1.4000000000000p+4f);
zolzt = rr_div(rr_mul(zol,zt),za);
zolz0 = rr_div(rr_mul(zol,zntstoch),za);
zolza = rr_div(rr_mul(zol,rr_add(za,zntstoch)),za);
zol10 = rr_div(rr_mul(zol,rr_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = rr_div(rr_mul(zol,rr_add(0x1.0000000000000p+1f,zntstoch)),za);
psim = rr_sub(m_land_psim_stable(zolza,psi_opt,psim_stab),m_land_psim_stable(zolz0,psi_opt,psim_stab));
psih = rr_sub(m_land_psih_stable(zolza,psi_opt,psih_stab),m_land_psih_stable(zolzt,psi_opt,psih_stab));
psim10 = rr_sub(m_land_psim_stable(zol10,psi_opt,psim_stab),m_land_psim_stable(zolz0,psi_opt,psim_stab));
psih10 = rr_sub(m_land_psih_stable(zol10,psi_opt,psih_stab),m_land_psih_stable(zolz0,psi_opt,psih_stab));
psih2 = rr_sub(m_land_psih_stable(zol2,psi_opt,psih_stab),m_land_psih_stable(zolzt,psi_opt,psih_stab));
rmol = rr_div(zol,za);
} else if ((rb==0x0.0p+0f)) {
psim = 0x0.0p+0f;
psih = psim;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = 0x0.0p+0f;
} else if ((rb<0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_land_li_etal_2010(zol,rb,rr_div(za,zntstoch),zratio);
zol = rr_max(zol,-0x1.4000000000000p+4f);
zol = rr_min(zol,0x0.0p+0f);
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_land_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = rr_max(zol,-0x1.4000000000000p+4f);
zol = rr_min(zol,0x0.0p+0f);
zolzt = rr_div(rr_mul(zol,zt),za);
zolz0 = rr_div(rr_mul(zol,zntstoch),za);
zolza = rr_div(rr_mul(zol,rr_add(za,zntstoch)),za);
zol10 = rr_div(rr_mul(zol,rr_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = rr_div(rr_mul(zol,rr_add(0x1.0000000000000p+1f,zntstoch)),za);
psim = rr_sub(m_land_psim_unstable(zolza,psi_opt,psim_unstab),m_land_psim_unstable(zolz0,psi_opt,psim_unstab));
psih = rr_sub(m_land_psih_unstable(zolza,psi_opt,psih_unstab),m_land_psih_unstable(zolzt,psi_opt,psih_unstab));
psim10 = rr_sub(m_land_psim_unstable(zol10,psi_opt,psim_unstab),m_land_psim_unstable(zolz0,psi_opt,psim_unstab));
psih10 = rr_sub(m_land_psih_unstable(zol10,psi_opt,psih_unstab),m_land_psih_unstable(zolz0,psi_opt,psih_unstab));
psih2 = rr_sub(m_land_psih_unstable(zol2,psi_opt,psih_unstab),m_land_psih_unstable(zolzt,psi_opt,psih_unstab));
psih = rr_min(psih,rr_mul(0x1.cccccc0000000p-1f,gz1ozt));
psim = rr_min(psim,rr_mul(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = rr_min(psih2,rr_mul(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = rr_min(psim10,rr_mul(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = rr_min(psih10,rr_mul(0x1.cccccc0000000p-1f,gz10ozt));
rmol = rr_div(zol,za);
}
psix = rr_max(rr_sub(gz1oz0,psim),0x1.0000000000000p+0f);
psix10 = rr_max(rr_sub(gz10oz0,psim10),0x1.0000000000000p+0f);
psit = rr_max(rr_sub(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = rr_max(rr_sub(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = rr_max(rr_sub(rr_log(rr_div(rr_add(za,zq),zq)),psih),0x1.0000000000000p+0f);
psiq2 = rr_max(rr_sub(rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zq),zq)),psih2),0x1.0000000000000p+0f);
psiq10 = rr_max(rr_sub(rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zq),zq)),psih10),0x1.0000000000000p+0f);
oldust = ust;
ust = rr_mul(0x1.0000000000000p-1f,rr_add(ust,rr_div(rr_mul(0x1.99999a0000000p-2f,wspd),psix)));
ust = rr_max(ust,0x1.47ae140000000p-8f);
stress = rr_mul(ust,ust);
ustm = ust;
dtg = rr_sub(thv_1,thvsk);
oldtst = mol;
mol = rr_div(rr_div(rr_mul(0x1.99999a0000000p-2f,dtg),psit),0x1.0000000000000p+0f);
dqg = rr_mul(rr_sub(qv_1,qsfc),0x1.f400000000000p+9f);
qstar = rr_div(rr_div(rr_mul(0x1.99999a0000000p-2f,dqg),psiq),0x1.0000000000000p+0f);
}
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
if (flag_iter) {
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
cm = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = rr_div(rr_mul(rr_mul(rr_mul(rho_1,mavail),ust),0x1.99999a0000000p-2f),psiq);
flhc = rr_div(rr_mul(rr_mul(rr_mul(rho_1,cpm),ust),0x1.99999a0000000p-2f),psit);
if (compute_flux) {
qfx = rr_mul(flqc,rr_sub(qsfc,qv_1));
qfx = rr_max(qfx,-0x1.47ae140000000p-6f);
lh = rr_mul(0x1.312d000000000p+21f,qfx);
hfx = rr_mul(rr_div(rr_mul(rr_div(rr_mul(rr_mul(rr_mul(rho_1,cpm),0x1.99999a0000000p-2f),wspd),psix),0x1.99999a0000000p-2f),psit),rr_sub(thsk,th_1));
hfx = rr_max(hfx,-0x1.f400000000000p+7f);
}
chs = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psit);
cqs = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psiq);
cqs2 = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psiq2);
chs2 = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = rr_mul(rr_div(0x1.99999a0000000p-2f,psix10),rr_div(0x1.99999a0000000p-2f,psiq10));
cd = rr_mul(rr_div(0x1.99999a0000000p-2f,psix10),rr_div(0x1.99999a0000000p-2f,psix10));
cka = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psiq));
cda = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psix));
}
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
ch = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psit));
cm = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psix));
}
}
if (compute_diag) {
if (flag_iter) {
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u_2;
v10 = v_2;
} else {
u10 = rr_div(rr_mul(u_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
v10 = rr_div(rr_mul(v_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = rr_div(rr_mul(u_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
v10 = rr_div(rr_mul(v_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
} else {
u10 = rr_div(rr_mul(u_1,psix10),psix);
v10 = rr_div(rr_mul(v_1,psix10),psix);
}
dtg = rr_sub(th_1,thsk);
th2 = rr_add(thsk,rr_div(rr_mul(dtg,psit2),psit));
if ((((th_1>thsk)&&((th2<thsk)||(th2>th_1)))||((th_1<thsk)&&((th2>thsk)||(th2<th_1))))) {
th2 = rr_add(thsk,rr_div(rr_mul(0x1.0000000000000p+1f,rr_sub(th_1,thsk)),za));
}
t2 = rr_mul(th2,rr_pow(rr_div(psfcpa,0x1.86a0000000000p+16f),0x1.24924a0000000p-2f));
q2 = rr_add(qsfc,rr_div(rr_mul(rr_sub(qv_1,qsfc),psiq2),psiq));
q2 = rr_max(q2,rr_min(qsfc,qv_1));
q2 = rr_min(q2,rr_mul(0x1.0ccccc0000000p+0f,qv_1));
}
}
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
yesno = 0;
if (compute_flux) {
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho_1<0x0.0p+0f)||(rho_1>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((rr_mul(qsfc,0x1.f400000000000p+9f)<0x0.0p+0f)||(rr_mul(qsfc,0x1.f400000000000p+9f)>0x1.4000000000000p+5f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:919-954 land_zilitinkevich_1995
static inline void m_land_zilitinkevich_1995(Exact32 z_0, thread Exact32 &zt, thread Exact32 &zq, Exact32 restar, Exact32 ustar, Exact32 karman, int zt_opt, int lsm, int lsm_ruc, bool has_zt_opt) {
Exact32 czil;
Exact32 znt;
if ((zt_opt==1)) {
if ((lsm!=lsm_ruc)) {
czil = rr_pow(0x1.4000000000000p+3f,rr_mul(-0x1.99999a0000000p-2f,rr_div(z_0,0x1.1eb8520000000p-4f)));
znt = z_0;
} else {
czil = rr_add(0x1.47ae140000000p-4f,rr_pow(0x1.4000000000000p+3f,rr_mul(-0x1.3333340000000p-1f,rr_div(rr_add(z_0,0x1.c28f5c0000000p-4f),0x1.eb851e0000000p-5f))));
znt = rr_min(0x1.3333340000000p-1f,z_0);
}
} else {
czil = 0x1.851eb80000000p-4f;
znt = z_0;
}
zt = rr_mul(znt,rr_exp(rr_mul(rr_mul((-karman),czil),rr_sqrt(restar))));
zt = rr_min(zt,rr_mul(0x1.8000000000000p-1f,znt));
zt = rr_max(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:963-984 land_garratt_1992
static inline void m_land_garratt_1992(thread Exact32 &zt, thread Exact32 &zq, Exact32 z_0, Exact32 ren, Exact32 landsea) {
Exact32 rq;
Exact32 e = 0x1.5bf0a80000000p+1f;
if ((rr_sub(landsea,0x1.8000000000000p+0f)>0x0.0p+0f)) {
zt = rr_mul(z_0,rr_exp(rr_sub(0x1.0000000000000p+1f,rr_mul(0x1.3d70a40000000p+1f,rr_pow(ren,0x1.0000000000000p-2f)))));
zq = rr_mul(z_0,rr_exp(rr_sub(0x1.0000000000000p+1f,rr_mul(0x1.23d70a0000000p+1f,rr_pow(ren,0x1.0000000000000p-2f)))));
zq = rr_min(zq,0x1.cd5f9a0000000p-15f);
zq = rr_max(zq,0x1.12e0be0000000p-29f);
zt = rr_min(zt,0x1.cd5f9a0000000p-15f);
zt = rr_max(zt,0x1.12e0be0000000p-29f);
} else {
zq = rr_div(z_0,0x1.d9999a0000000p+2f);
zt = zq;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1010-1041 land_yang_2008
static inline void m_land_yang_2008(Exact32 z_0, thread Exact32 &zt, thread Exact32 &zq, Exact32 ustar, Exact32 tstar, Exact32 qst, Exact32 ren, Exact32 visc) {
Exact32 ht;
Exact32 tstar2;
Exact32 qstar2;
Exact32 z_02;
Exact32 renc2;
Exact32 renc = 0x1.2c00000000000p+8f;
Exact32 beta = 0x1.8000000000000p+0f;
Exact32 m = 0x1.5400000000000p+7f;
Exact32 b = 0x1.5980000000000p+9f;
z_02 = rr_min(z_0,0x1.0000000000000p-1f);
z_02 = rr_max(z_02,0x1.47ae140000000p-5f);
renc2 = rr_add(0x1.5980000000000p+9f,rr_mul(0x1.5400000000000p+7f,rr_log(z_02)));
ht = rr_div(rr_mul(renc2,visc),rr_max(ustar,0x1.47ae140000000p-7f));
tstar2 = rr_min(tstar,-0x1.47ae140000000p-7f);
qstar2 = rr_min(qst,-0x1.47ae140000000p-7f);
zt = rr_mul(ht,rr_exp(rr_mul(rr_mul(-0x1.8000000000000p+0f,rr_sqrt(ustar)),rr_abs(tstar2))));
zt = rr_min(zt,rr_mul(0x1.8000000000000p-1f,z_0));
zt = rr_max(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1045-1096 land_gfs_z0_land
static inline void m_land_gfs_z0_land(thread Exact32 &z0max, Exact32 shdmax, Exact32 z1, int vegtype, int ivegsrc, Exact32 z0pert) {
Exact32 tem1;
Exact32 tem2;
z0max = rr_max(0x1.0c6f7a0000000p-20f,rr_min(z0max,z1));
tem1 = rr_sub(0x1.0000000000000p+0f,shdmax);
tem2 = rr_mul(tem1,tem1);
tem1 = rr_sub(0x1.0000000000000p+0f,tem2);
if ((ivegsrc==1)) {
if ((vegtype==10)) {
z0max = rr_exp(rr_add(rr_mul(tem2,-0x1.26bb1c0000000p+2f),rr_mul(tem1,-0x1.5462a20000000p+1f)));
} else if ((vegtype==6)) {
z0max = rr_exp(rr_add(rr_mul(tem2,-0x1.26bb1c0000000p+2f),rr_mul(tem1,-0x1.7f74280000000p+1f)));
} else if ((vegtype==7)) {
z0max = 0x1.47ae140000000p-7f;
} else if ((vegtype==16)) {
z0max = 0x1.47ae140000000p-7f;
} else {
z0max = rr_exp(rr_add(rr_mul(tem2,-0x1.26bb1c0000000p+2f),rr_mul(tem1,rr_log(z0max))));
}
} else if ((ivegsrc==2)) {
if ((vegtype==7)) {
z0max = rr_exp(rr_add(rr_mul(tem2,-0x1.26bb1c0000000p+2f),rr_mul(tem1,-0x1.5462a20000000p+1f)));
} else if ((vegtype==8)) {
z0max = rr_exp(rr_add(rr_mul(tem2,-0x1.26bb1c0000000p+2f),rr_mul(tem1,-0x1.7f74280000000p+1f)));
} else if ((vegtype==9)) {
z0max = 0x1.47ae140000000p-7f;
} else if ((vegtype==11)) {
z0max = 0x1.47ae140000000p-7f;
} else {
z0max = rr_exp(rr_add(rr_mul(tem2,-0x1.26bb1c0000000p+2f),rr_mul(tem1,rr_log(z0max))));
}
}
if ((z0pert!=0x0.0p+0f)) {
z0max = rr_mul(z0max,rr_pow(0x1.4000000000000p+3f,z0pert));
}
z0max = rr_max(z0max,0x1.0c6f7a0000000p-20f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1100-1124 land_gfs_zt_land
static inline void m_land_gfs_zt_land(thread Exact32 &ztmax, Exact32 z0max, Exact32 sigmaf, Exact32 ztpert, Exact32 ustar) {
Exact32 czilc;
Exact32 tem1;
Exact32 tem2;
Exact32 ca = 0x1.99999a0000000p-2f;
czilc = 0x1.99999a0000000p-1f;
tem1 = rr_sub(0x1.0000000000000p+0f,sigmaf);
ztmax = rr_mul(z0max,rr_exp(rr_mul(rr_mul(rr_mul(rr_mul((-tem1),tem1),czilc),0x1.99999a0000000p-2f),rr_sqrt(rr_mul(ustar,0x1.4d55560000000p+9f)))));
if ((ztpert!=0x0.0p+0f)) {
ztmax = rr_mul(ztmax,rr_pow(0x1.4000000000000p+3f,ztpert));
}
ztmax = rr_max(ztmax,0x1.0c6f7a0000000p-20f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1129-1158 land_psi_hogstrom_1996
static inline void m_land_psi_hogstrom_1996(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za) {
Exact32 x;
Exact32 x0;
Exact32 y;
Exact32 y0;
Exact32 zml;
Exact32 zhl;
zml = rr_div(rr_mul(z_0,zl),za);
zhl = rr_div(rr_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = rr_mul(-0x1.5333340000000p+2f,rr_sub(zl,zml));
psi_h = rr_mul(-0x1.0000000000000p+3f,rr_sub(zl,zhl));
} else {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,x),rr_add(0x1.0000000000000p+0f,x0)))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),rr_add(0x1.0000000000000p+0f,rr_mul(x0,x0))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),rr_mul(0x1.0000000000000p+1f,rr_atan(x0)));
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),rr_add(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1165-1195 land_psi_dyerhicks
static inline void m_land_psi_dyerhicks(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za) {
Exact32 x;
Exact32 x0;
Exact32 y;
Exact32 y0;
Exact32 zml;
Exact32 zhl;
zml = rr_div(rr_mul(z_0,zl),za);
zhl = rr_div(rr_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = rr_mul(-0x1.4000000000000p+2f,rr_sub(zl,zml));
psi_h = rr_mul(-0x1.4000000000000p+2f,rr_sub(zl,zhl));
} else {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,x),rr_add(0x1.0000000000000p+0f,x0)))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),rr_add(0x1.0000000000000p+0f,rr_mul(x0,x0))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),rr_mul(0x1.0000000000000p+1f,rr_atan(x0)));
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),rr_add(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1201-1224 land_psi_beljaars_holtslag_1991
static inline void m_land_psi_beljaars_holtslag_1991(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 a = 0x1.0000000000000p+0f;
Exact32 b = 0x1.54fdf40000000p-1f;
Exact32 c = 0x1.4000000000000p+2f;
Exact32 d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-rr_add(rr_add(rr_mul(0x1.0000000000000p+0f,zl),rr_mul(rr_mul(0x1.54fdf40000000p-1f,rr_sub(zl,0x1.c924920000000p+3f)),rr_exp(rr_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-rr_sub(rr_add(rr_add(rr_pow(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),rr_mul(rr_mul(0x1.54fdf40000000p-1f,rr_sub(zl,0x1.c924920000000p+3f)),rr_exp(rr_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1231-1253 land_psi_zilitinkevich_esau_2007
static inline void m_land_psi_zilitinkevich_esau_2007(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 cm = 0x1.8000000000000p+1f;
Exact32 ct = 0x1.4000000000000p+1f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = rr_mul(-0x1.8000000000000p+1f,rr_pow(zl,0x1.aaaaaa0000000p-1f));
psi_h = rr_mul(-0x1.4000000000000p+1f,rr_pow(zl,0x1.99999a0000000p-1f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1258-1283 land_psi_businger_1971
static inline void m_land_psi_businger_1971(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 x;
Exact32 y;
Exact32 pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_log(rr_mul(rr_div(rr_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),rr_div(rr_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),0x1.0000000000000p+1f))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = rr_mul(-0x1.2ccccc0000000p+2f,zl);
psi_h = rr_mul(-0x1.967c880000000p+2f,zl);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1289-1300 land_psi_cb2005
static inline void m_land_psi_cb2005(thread Exact32 &psim1, thread Exact32 &psih1, Exact32 zl, Exact32 z0l) {
psim1 = rr_sub(rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zl,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),rr_mul(0x1.8666660000000p+2f,rr_log(rr_add(z0l,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = rr_sub(rr_mul(-0x1.6000000000000p+2f,rr_log(rr_add(zl,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),rr_mul(0x1.6000000000000p+2f,rr_log(rr_add(z0l,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1308-1364 land_li_etal_2010
static inline void m_land_li_etal_2010(thread Exact32 &zl, Exact32 rib, Exact32 zaz0, Exact32 z0zt) {
Exact32 alfa;
Exact32 beta;
Exact32 zaz02;
Exact32 z0zt2;
Exact32 au11 = 0x1.70a3d80000000p-5f;
Exact32 bu11 = 0x1.89374c0000000p-9f;
Exact32 bu12 = 0x1.82a9940000000p-8f;
Exact32 bu21 = -0x1.5326180000000p-4f;
Exact32 bu22 = 0x1.c4dd300000000p-1f;
Exact32 bu31 = 0x1.6425ae0000000p-3f;
Exact32 bu32 = -0x1.d7b4a20000000p-1f;
Exact32 bu33 = -0x1.b0f27c0000000p-4f;
Exact32 aw11 = 0x1.25c91e0000000p-1f;
Exact32 aw12 = -0x1.c275260000000p-2f;
Exact32 aw21 = -0x1.39a9fc0000000p+2f;
Exact32 aw22 = 0x1.a400000000000p+5f;
Exact32 bw11 = -0x1.b98c7e0000000p-5f;
Exact32 bw12 = 0x1.8a3d700000000p+0f;
Exact32 bw21 = -0x1.56872c0000000p-1f;
Exact32 bw22 = -0x1.a418940000000p+1f;
Exact32 as11 = 0x1.817c1c0000000p-1f;
Exact32 as21 = 0x1.de147a0000000p+3f;
Exact32 bs11 = 0x1.4154ca0000000p-3f;
Exact32 bs21 = -0x1.3c84b60000000p-2f;
Exact32 bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = rr_log(zaz02);
beta = rr_log(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = rr_add(rr_mul(rr_mul(0x1.70a3d80000000p-5f,alfa),rr_mul(rib,rib)),rr_mul(rr_add(rr_add(rr_mul(rr_add(rr_mul(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),rr_mul(alfa,alfa)),rr_mul(rr_add(rr_mul(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),rr_add(rr_add(rr_mul(0x1.6425ae0000000p-3f,rr_mul(beta,beta)),rr_mul(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = rr_max(zl,-0x1.e000000000000p+3f);
zl = rr_min(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = rr_add(rr_mul(rr_add(rr_mul(rr_add(rr_mul(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),rr_add(rr_mul(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),rr_mul(rib,rib)),rr_mul(rr_add(rr_mul(rr_add(rr_mul(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),rr_add(rr_mul(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = rr_min(zl,0x1.0000000000000p+2f);
zl = rr_max(zl,0x0.0p+0f);
} else {
zl = rr_add(rr_add(rr_add(rr_mul(rr_add(rr_mul(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),rr_mul(0x1.4154ca0000000p-3f,alfa)),rr_mul(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = rr_min(zl,0x1.4000000000000p+4f);
zl = rr_max(zl,0x1.0000000000000p+0f);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1367-1420 land_zolri
static inline Exact32 m_land_zolri(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 x1;
Exact32 x2;
Exact32 fx1;
Exact32 fx2;
int n;
int nmax = 20;
if ((ri<0x0.0p+0f)) {
x1 = rr_sub(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = rr_add(zol1,0x1.47ae140000000p-6f);
}
n = 1;
fx1 = m_land_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
fx2 = m_land_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
while (((rr_abs(rr_sub(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((rr_abs(fx2)<rr_abs(fx1))) {
x1 = rr_sub(x1,rr_mul(rr_div(fx1,rr_sub(fx2,fx1)),rr_sub(x2,x1)));
fx1 = m_land_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x1;
} else {
x2 = rr_sub(x2,rr_mul(rr_div(fx2,rr_sub(fx2,fx1)),rr_sub(x2,x1)));
fx2 = m_land_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(rr_abs(rr_sub(x1,x2))>=0x1.47ae140000000p-7f))) {
m_land_li_etal_2010(result,ri,rr_div(za,z0),rr_div(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1422-1462 land_zolri2
static inline Exact32 m_land_zolri2(thread Exact32 &zol2, Exact32 ri2, Exact32 za, Exact32 z0, Exact32 zt, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 zol20;
Exact32 zol3;
Exact32 psim1;
Exact32 psih1;
Exact32 psix2;
Exact32 psit2;
Exact32 zolt;
if ((rr_mul(zol2,ri2)<0x0.0p+0f)) zol2 = 0x0.0p+0f;
zol20 = rr_div(rr_mul(zol2,z0),za);
zol3 = rr_add(zol2,zol20);
zolt = rr_div(rr_mul(zol2,zt),za);
if ((ri2<0)) {
psit2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),zt)),rr_sub(m_land_psih_unstable(zol3,psi_opt,psih_unstab),m_land_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),z0)),rr_sub(m_land_psim_unstable(zol3,psi_opt,psim_unstab),m_land_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),zt)),rr_sub(m_land_psih_stable(zol3,psi_opt,psih_stab),m_land_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),z0)),rr_sub(m_land_psim_stable(zol3,psi_opt,psim_stab),m_land_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = rr_sub(rr_div(rr_mul(zol2,psit2),rr_mul(psix2,psix2)),ri2);
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1465-1539 land_zolrib
static inline Exact32 m_land_zolrib(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 logz0, Exact32 logzt, thread Exact32 &zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 zol20;
Exact32 zol3;
Exact32 zolt;
Exact32 zolold;
int n;
int nmax = 20;
Exact32 psit2;
Exact32 psix2;
if ((rr_mul(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((rr_abs(rr_sub(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = rr_div(rr_mul(zolold,z0),za);
zol3 = rr_add(zolold,zol20);
zolt = rr_div(rr_mul(zolold,zt),za);
if ((ri<0)) {
psit2 = rr_max(rr_sub(logzt,rr_sub(m_land_psih_unstable(zol3,psi_opt,psih_unstab),m_land_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(logz0,rr_sub(m_land_psim_unstable(zol3,psi_opt,psim_unstab),m_land_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = rr_max(rr_sub(logzt,rr_sub(m_land_psih_stable(zol3,psi_opt,psih_stab),m_land_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(logz0,rr_sub(m_land_psim_stable(zol3,psi_opt,psim_stab),m_land_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = rr_div(rr_mul(ri,rr_mul(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(rr_abs(rr_sub(zolold,result))>0x1.47ae140000000p-7f))) {
m_land_li_etal_2010(result,ri,rr_div(za,z0),rr_div(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1544-1551 land_psim_stable_full
static inline Exact32 m_land_psim_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1554-1561 land_psih_stable_full
static inline Exact32 m_land_psih_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.5333340000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1564-1579 land_psim_unstable_full
static inline Exact32 m_land_psim_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 x;
Exact32 ym;
Exact32 psimc;
Exact32 psimk;
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,x)))),rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,rr_mul(x,x))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
ym = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psimk,rr_mul(rr_mul(zolf,zolf),psimc)),rr_add(1,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1582-1596 land_psih_unstable_full
static inline Exact32 m_land_psih_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 y;
Exact32 yh;
Exact32 psihc;
Exact32 psihk;
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psihk,rr_mul(rr_mul(zolf,zolf),psihc)),rr_add(0x1.0000000000000p+0f,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1603-1612 land_psim_stable_full_gfs
static inline Exact32 m_land_psim_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 aa;
aa = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,aa),rr_log(rr_add(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1616-1625 land_psih_stable_full_gfs
static inline Exact32 m_land_psih_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 bb;
bb = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,bb),rr_log(rr_add(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1629-1645 land_psim_unstable_full_gfs
static inline Exact32 m_land_psim_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0 = -0x1.fccccc0000000p+1f;
Exact32 a1 = 0x1.8a3d700000000p+3f;
Exact32 b1 = -0x1.f051ec0000000p+2f;
Exact32 b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fccccc0000000p+1f,rr_mul(0x1.8a3d700000000p+3f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.f051ec0000000p+2f,rr_mul(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_sub(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p+1f,rr_sqrt(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1649-1665 land_psih_unstable_full_gfs
static inline Exact32 m_land_psih_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0p = -0x1.fc39580000000p+2f;
Exact32 a1p = 0x1.8c00000000000p+4f;
Exact32 b1p = -0x1.168f5c0000000p+3f;
Exact32 b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fc39580000000p+2f,rr_mul(0x1.8c00000000000p+4f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.168f5c0000000p+3f,rr_mul(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_add(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1669-1686 land_psim_stable
static inline Exact32 m_land_psim_stable(Exact32 zolf, int psi_opt, device const float *psim_stab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul(zolf,0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psim_stab[nzol],rr_mul(rzol,rr_sub(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psim_stable_full(zolf);
} else {
result = m_land_psim_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1689-1706 land_psih_stable
static inline Exact32 m_land_psih_stable(Exact32 zolf, int psi_opt, device const float *psih_stab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul(zolf,0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psih_stab[nzol],rr_mul(rzol,rr_sub(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psih_stable_full(zolf);
} else {
result = m_land_psih_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1709-1726 land_psim_unstable
static inline Exact32 m_land_psim_unstable(Exact32 zolf, int psi_opt, device const float *psim_unstab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul((-zolf),0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psim_unstab[nzol],rr_mul(rzol,rr_sub(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psim_unstable_full(zolf);
} else {
result = m_land_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1729-1746 land_psih_unstable
static inline Exact32 m_land_psih_unstable(Exact32 zolf, int psi_opt, device const float *psih_unstab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul((-zolf),0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psih_unstab[nzol],rr_mul(rzol,rr_sub(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psih_unstable_full(zolf);
} else {
result = m_land_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:122-919 water_mynnsfc_water
static inline void m_water_mynnsfc_water(bool flag_iter, int itimestep, int i, int j, Exact32 dx, Exact32 xland, Exact32 lakemask, Exact32 wat_depth, Exact32 u_1, Exact32 v_1, Exact32 t_1, Exact32 qv_1, Exact32 p_1, Exact32 dz8w_1, thread Exact32 &rho_1, Exact32 u_2, Exact32 v_2, Exact32 dz8w_2, Exact32 sigmaf, int vegtype, Exact32 shdmax, int ivegsrc, Exact32 z0pert, Exact32 ztpert, bool redrag, int sfc_z0_type, Exact32 pblh, thread Exact32 &znt, Exact32 psfcpa, Exact32 mavail, Exact32 tskin, Exact32 tsurf, Exact32 snowh, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs2, thread Exact32 &cqs, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &stress, thread Exact32 &rmol, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &psim, thread Exact32 &psih, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &qsfc, thread Exact32 &t2, thread Exact32 &q2, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &lh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &rb, thread Exact32 &cpm, thread Exact32 &ch, thread Exact32 &cm, Exact32 rstoch_1, thread Exact32 &wstar, thread Exact32 &qstar, thread Exact32 &qgh, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &psix, thread Exact32 &psit, thread Exact32 &psix10, thread Exact32 &psit2, int spp_sfc, int sf_mynn_sfcflux_water, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int shalwater_z0, int iter, int lsm, int lsm_ruc, device const float *psim_stab, device const float *psim_unstab, device const float *psih_stab, device const float *psih_unstab, thread int &errflg, bool has_sf_mynn_sfcflux_water, bool has_shalwater_z0, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda) {
Exact32 xka = 0x1.92a7380000000p-16f;
Exact32 prt = 0x1.0000000000000p+0f;
Exact32 za;
Exact32 za2;
Exact32 thv_1;
Exact32 th_1;
Exact32 tc_1;
Exact32 tv_1;
Exact32 qvmr;
Exact32 psih2;
Exact32 psim10;
Exact32 psih10;
Exact32 wspdi;
Exact32 govrth;
Exact32 psfc;
Exact32 qsfcmr;
Exact32 thcon;
Exact32 zratio;
Exact32 tsk;
Exact32 thsk;
Exact32 thvsk;
Exact32 gz1ozt;
Exact32 gz2oz0;
Exact32 gz2ozt;
Exact32 gz10oz0;
Exact32 gz10ozt;
Exact32 zntstoch;
Exact32 zt;
Exact32 zq;
Exact32 psiq;
Exact32 psiq2;
int n;
int k;
int l;
int yesno;
Exact32 pl;
Exact32 e1;
Exact32 tabs;
Exact32 dthvdz;
Exact32 dthvm;
Exact32 vconv;
Exact32 zol2;
Exact32 zol10;
Exact32 zolza;
Exact32 zolz0;
Exact32 zolzt;
Exact32 dtg;
Exact32 dtthx;
Exact32 psiq10;
Exact32 fluxc;
Exact32 vsgd;
Exact32 restar;
Exact32 visc;
Exact32 dqg;
Exact32 oldust;
Exact32 oldtst;
errflg = 0;
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
psfc = rr_div(psfcpa,0x1.f400000000000p+9f);
if (flag_iter) {
tsk = tskin;
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_div(rr_mul(0x1.1ab8520000000p+4f,rr_sub(tsk,0x1.1126660000000p+8f)),rr_sub(tsk,0x1.da66660000000p+4f))));
if ((lakemask<0x1.0000000000000p-1f)) e1 = rr_mul(e1,0x1.f65fd80000000p-1f);
qsfc = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,rr_mul(0x1.8353dc0000000p-2f,e1)));
qsfcmr = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,e1));
}
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_div(rr_mul(0x1.1ab8520000000p+4f,rr_sub(t_1,0x1.1126660000000p+8f)),rr_sub(t_1,0x1.da66660000000p+4f))));
pl = rr_div(p_1,0x1.f400000000000p+9f);
qgh = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(pl,e1));
qvmr = rr_div(qv_1,rr_sub(0x1.0000000000000p+0f,qv_1));
thcon = rr_pow(rr_div(0x1.86a0000000000p+16f,psfcpa),0x1.24924a0000000p-2f);
if (flag_iter) {
tsk = tskin;
thsk = rr_mul(tsk,thcon);
thvsk = rr_mul(thsk,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qsfc)));
if (((thvsk<0x1.4000000000000p+7f)||(thvsk>0x1.8600000000000p+8f))) {
}
}
th_1 = rr_mul(t_1,rr_pow(rr_div(0x1.86a0000000000p+16f,p_1),0x1.24924a0000000p-2f));
tc_1 = rr_sub(t_1,0x1.1126660000000p+8f);
thv_1 = rr_mul(th_1,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qv_1)));
tv_1 = rr_mul(t_1,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qv_1)));
rho_1 = rr_div(p_1,rr_mul(0x1.1f00000000000p+8f,tv_1));
za = rr_mul(0x1.0000000000000p-1f,dz8w_1);
za2 = rr_add(dz8w_1,rr_mul(0x1.0000000000000p-1f,dz8w_2));
govrth = rr_div(0x1.39cc100000000p+3f,th_1);
cpm = rr_mul(0x1.f640000000000p+9f,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.ae147a0000000p-1f,qv_1)));
if (flag_iter) {
wspd = rr_sqrt(rr_add(rr_mul(u_1,u_1),rr_mul(v_1,v_1)));
dthvdz = rr_sub(thv_1,thvsk);
fluxc = rr_max(rr_add(rr_div(rr_div(hfx,rho_1),0x1.f640000000000p+9f),rr_div(rr_mul(rr_mul(0x1.377b480000000p-1f,thvsk),qfx),rho_1)),0x0.0p+0f);
wstar = rr_mul(0x1.4000000000000p+0f,rr_pow(rr_mul(rr_mul(rr_div(0x1.39cc100000000p+3f,tsk),pblh),fluxc),0x1.5555560000000p-2f));
vsgd = rr_min(rr_mul(0x1.0000000000000p-2f,rr_pow(rr_max(rr_sub(rr_div(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.5555560000000p-2f)),0x1.0000000000000p-1f);
wspd = rr_sqrt(rr_add(rr_add(rr_mul(wspd,wspd),rr_mul(wstar,wstar)),rr_mul(vsgd,vsgd)));
wspd = rr_max(wspd,0x1.99999a0000000p-4f);
rb = rr_div(rr_mul(rr_mul(govrth,za),dthvdz),rr_mul(wspd,wspd));
rb = rr_max(rb,-0x1.0000000000000p+1f);
rb = rr_min(rb,0x1.0000000000000p+1f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
}
if (flag_iter) {
visc = rr_mul(0x1.bcee880000000p-17f,rr_sub(rr_add(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.acbc8c0000000p-8f,tc_1)),rr_mul(rr_mul(0x1.16890c0000000p-17f,tc_1),tc_1)),rr_mul(rr_mul(rr_mul(0x1.4c9a2e0000000p-28f,tc_1),tc_1),tc_1)));
if ((sfc_z0_type>=0)) {
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (has_sf_mynn_sfcflux_water) {
if ((sf_mynn_sfcflux_water==0)) {
m_water_coare30_znt(znt,ust,wspd,visc,za);
} else if ((sf_mynn_sfcflux_water==1)) {
m_water_edson_etal_2013(znt,ust,wspd,visc,za);
} else if (((sf_mynn_sfcflux_water==2)||(sf_mynn_sfcflux_water==3))) {
m_water_davis_etal_2008(znt,ust);
} else if ((sf_mynn_sfcflux_water==4)) {
m_water_taylor_yelland_2001(znt,ust,wspd);
} else if ((sf_mynn_sfcflux_water==5)) {
m_water_gfs_z0_wat(znt,ust,wspd,za,sfc_z0_type,redrag);
}
} else {
m_water_edson_etal_2013(znt,ust,wspd,visc,za);
}
if (has_shalwater_z0) {
if ((shalwater_z0==1)) {
znt = m_water_depth_dependent_z0(wat_depth,znt,ust);
}
}
}
if ((spp_sfc==1)) {
zntstoch = rr_max(rr_add(znt,rr_mul(rr_mul(znt,0x1.0000000000000p+0f),rstoch_1)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = rr_max(rr_div(rr_mul(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (has_sf_mynn_sfcflux_water) {
if ((sf_mynn_sfcflux_water==0)) {
m_water_fairall_etal_2003(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==1)) {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==2)) {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==3)) {
m_water_garratt_1992(zt,zq,zntstoch,restar,0x1.0000000000000p+1f);
} else if ((sf_mynn_sfcflux_water==4)) {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==5)) {
m_water_gfs_zt_wat(zt,zntstoch,restar,wspd,za,sfc_z0_type,errflg);
if ((errflg!=0)) return;
zq = zt;
}
} else {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
gz1oz0 = rr_log(rr_div(rr_add(za,zntstoch),zntstoch));
gz1ozt = rr_log(rr_div(rr_add(za,zntstoch),zt));
gz2oz0 = rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zntstoch),zt));
gz10oz0 = rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zntstoch),zt));
zratio = rr_div(zntstoch,zt);
if ((rb>0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_water_li_etal_2010(zol,rb,rr_div(za,zntstoch),zratio);
zol = rr_max(zol,0x0.0p+0f);
zol = rr_min(zol,0x1.4000000000000p+4f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_water_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = rr_max(zol,0x0.0p+0f);
zol = rr_min(zol,0x1.4000000000000p+4f);
zolzt = rr_div(rr_mul(zol,zt),za);
zolz0 = rr_div(rr_mul(zol,zntstoch),za);
zolza = rr_div(rr_mul(zol,rr_add(za,zntstoch)),za);
zol10 = rr_div(rr_mul(zol,rr_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = rr_div(rr_mul(zol,rr_add(0x1.0000000000000p+1f,zntstoch)),za);
psim = rr_sub(m_water_psim_stable(zolza,psi_opt,psim_stab),m_water_psim_stable(zolz0,psi_opt,psim_stab));
psih = rr_sub(m_water_psih_stable(zolza,psi_opt,psih_stab),m_water_psih_stable(zolzt,psi_opt,psih_stab));
psim10 = rr_sub(m_water_psim_stable(zol10,psi_opt,psim_stab),m_water_psim_stable(zolz0,psi_opt,psim_stab));
psih10 = rr_sub(m_water_psih_stable(zol10,psi_opt,psih_stab),m_water_psih_stable(zolz0,psi_opt,psih_stab));
psih2 = rr_sub(m_water_psih_stable(zol2,psi_opt,psih_stab),m_water_psih_stable(zolzt,psi_opt,psih_stab));
rmol = rr_div(zol,za);
} else if ((rb==0x0.0p+0f)) {
psim = 0x0.0p+0f;
psih = psim;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = 0x0.0p+0f;
} else if ((rb<0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_water_li_etal_2010(zol,rb,rr_div(za,zntstoch),zratio);
zol = rr_max(zol,-0x1.4000000000000p+4f);
zol = rr_min(zol,0x0.0p+0f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_water_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = rr_max(zol,-0x1.4000000000000p+4f);
zol = rr_min(zol,0x0.0p+0f);
zolzt = rr_div(rr_mul(zol,zt),za);
zolz0 = rr_div(rr_mul(zol,zntstoch),za);
zolza = rr_div(rr_mul(zol,rr_add(za,zntstoch)),za);
zol10 = rr_div(rr_mul(zol,rr_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = rr_div(rr_mul(zol,rr_add(0x1.0000000000000p+1f,zntstoch)),za);
psim = rr_sub(m_water_psim_unstable(zolza,psi_opt,psim_unstab),m_water_psim_unstable(zolz0,psi_opt,psim_unstab));
psih = rr_sub(m_water_psih_unstable(zolza,psi_opt,psih_unstab),m_water_psih_unstable(zolzt,psi_opt,psih_unstab));
psim10 = rr_sub(m_water_psim_unstable(zol10,psi_opt,psim_unstab),m_water_psim_unstable(zolz0,psi_opt,psim_unstab));
psih10 = rr_sub(m_water_psih_unstable(zol10,psi_opt,psih_unstab),m_water_psih_unstable(zolz0,psi_opt,psih_unstab));
psih2 = rr_sub(m_water_psih_unstable(zol2,psi_opt,psih_unstab),m_water_psih_unstable(zolzt,psi_opt,psih_unstab));
psih = rr_min(psih,rr_mul(0x1.cccccc0000000p-1f,gz1ozt));
psim = rr_min(psim,rr_mul(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = rr_min(psih2,rr_mul(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = rr_min(psim10,rr_mul(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = rr_min(psih10,rr_mul(0x1.cccccc0000000p-1f,gz10ozt));
rmol = rr_div(zol,za);
}
psix = rr_max(rr_sub(gz1oz0,psim),0x1.0000000000000p+0f);
psix10 = rr_max(rr_sub(gz10oz0,psim10),0x1.0000000000000p+0f);
psit = rr_max(rr_sub(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = rr_max(rr_sub(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = rr_max(rr_sub(rr_log(rr_div(rr_add(za,zq),zq)),psih),0x1.0000000000000p+0f);
psiq2 = rr_max(rr_sub(rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zq),zq)),psih2),0x1.0000000000000p+0f);
psiq10 = rr_max(rr_sub(rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zq),zq)),psih10),0x1.0000000000000p+0f);
oldust = ust;
ust = rr_add(rr_mul(0x1.0000000000000p-1f,ust),rr_div(rr_mul(0x1.99999a0000000p-3f,wspd),psix));
stress = rr_mul(ust,ust);
wspdi = rr_max(rr_sqrt(rr_add(rr_mul(u_1,u_1),rr_mul(v_1,v_1))),0x1.99999a0000000p-4f);
ustm = rr_add(rr_mul(0x1.0000000000000p-1f,ustm),rr_div(rr_mul(0x1.99999a0000000p-3f,wspdi),psix));
dtg = rr_sub(thv_1,thvsk);
oldtst = mol;
mol = rr_div(rr_div(rr_mul(0x1.99999a0000000p-2f,dtg),psit),0x1.0000000000000p+0f);
dqg = rr_mul(rr_sub(qv_1,qsfc),0x1.f400000000000p+9f);
qstar = rr_div(rr_div(rr_mul(0x1.99999a0000000p-2f,dqg),psiq),0x1.0000000000000p+0f);
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (flag_iter) {
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
cqs = 0x0.0p+0f;
cm = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = rr_div(rr_mul(rr_mul(rr_mul(rho_1,mavail),ust),0x1.99999a0000000p-2f),psiq);
flhc = rr_div(rr_mul(rr_mul(rr_mul(rho_1,cpm),ust),0x1.99999a0000000p-2f),psit);
if (compute_flux) {
qfx = rr_mul(flqc,rr_sub(qsfc,qv_1));
qfx = rr_max(qfx,-0x1.47ae140000000p-6f);
lh = rr_mul(0x1.312d000000000p+21f,qfx);
hfx = rr_mul(rr_div(rr_mul(rr_div(rr_mul(rr_mul(rr_mul(rho_1,cpm),0x1.99999a0000000p-2f),wspd),psix),0x1.99999a0000000p-2f),psit),rr_sub(thsk,th_1));
if (has_sf_mynn_sfcflux_water) {
if ((sf_mynn_sfcflux_water>1)) {
hfx = rr_add(hfx,rr_mul(rr_mul(rr_mul(rho_1,ustm),ustm),wspdi));
}
}
}
chs = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psit);
cqs = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psiq);
cqs2 = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psiq2);
chs2 = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = rr_mul(rr_div(0x1.99999a0000000p-2f,psix10),rr_div(0x1.99999a0000000p-2f,psiq10));
cd = rr_mul(rr_div(0x1.99999a0000000p-2f,psix10),rr_div(0x1.99999a0000000p-2f,psix10));
cka = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psiq));
cda = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psix));
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
ch = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psit));
cm = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psix));
}
}
if (compute_diag) {
if (flag_iter) {
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u_2;
v10 = v_2;
} else {
u10 = rr_div(rr_mul(u_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
v10 = rr_div(rr_mul(v_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = rr_div(rr_mul(u_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
v10 = rr_div(rr_mul(v_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
} else {
u10 = rr_div(rr_mul(u_1,psix10),psix);
v10 = rr_div(rr_mul(v_1,psix10),psix);
}
dtg = rr_sub(th_1,thsk);
th2 = rr_add(thsk,rr_div(rr_mul(dtg,psit2),psit));
if ((((th_1>thsk)&&((th2<thsk)||(th2>th_1)))||((th_1<thsk)&&((th2>thsk)||(th2<th_1))))) {
th2 = rr_add(thsk,rr_div(rr_mul(0x1.0000000000000p+1f,rr_sub(th_1,thsk)),za));
}
t2 = rr_mul(th2,rr_pow(rr_div(psfcpa,0x1.86a0000000000p+16f),0x1.24924a0000000p-2f));
q2 = rr_add(qsfc,rr_div(rr_mul(rr_sub(qv_1,qsfc),psiq2),psiq));
q2 = rr_max(q2,rr_min(qsfc,qv_1));
q2 = rr_min(q2,rr_mul(0x1.0ccccc0000000p+0f,qv_1));
}
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
yesno = 0;
if (compute_flux) {
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho_1<0x0.0p+0f)||(rho_1>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:924-950 water_davis_etal_2008
static inline void m_water_davis_etal_2008(thread Exact32 &z_0, Exact32 ustar) {
Exact32 zw;
Exact32 zn1;
Exact32 zn2;
Exact32 ozo = 0x1.0ac1fa0000000p-16f;
zw = rr_min(rr_pow(rr_div(ustar,0x1.0f5c280000000p+0f),0x1.3333340000000p-2f),0x1.0000000000000p+0f);
zn1 = rr_add(rr_mul(rr_mul(rr_mul(0x1.6872b00000000p-7f,ustar),ustar),0x1.a1b2560000000p-4f),0x1.0ac1fa0000000p-16f);
zn2 = rr_add(rr_mul(0x1.4000000000000p+3f,rr_exp(rr_mul(-0x1.3000000000000p+3f,rr_pow(ustar,-0x1.5555560000000p-2f)))),rr_div(0x1.baeb220000000p-20f,rr_max(ustar,0x1.47ae140000000p-7f)));
z_0 = rr_add(rr_mul(rr_sub(0x1.0000000000000p+0f,zw),zn1),rr_mul(zw,zn2));
z_0 = rr_max(z_0,0x1.10bafe0000000p-23f);
z_0 = rr_min(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:955-974 water_taylor_yelland_2001
static inline void m_water_taylor_yelland_2001(thread Exact32 &z_0, Exact32 ustar, Exact32 wsp10) {
Exact32 pi = 0x1.921fb60000000p+1f;
Exact32 hs;
Exact32 tp;
Exact32 lp;
hs = rr_mul(0x1.9652be0000000p-6f,rr_mul(wsp10,wsp10));
tp = rr_mul(0x1.753f7c0000000p-1f,rr_max(wsp10,0x1.99999a0000000p-4f));
lp = rr_div(rr_mul(0x1.39cc100000000p+3f,rr_mul(tp,tp)),0x1.921fb60000000p+2f);
z_0 = rr_mul(rr_mul(0x1.2c00000000000p+10f,hs),rr_pow(rr_div(hs,lp),0x1.2000000000000p+2f));
z_0 = rr_max(z_0,0x1.10bafe0000000p-23f);
z_0 = rr_min(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:979-995 water_coare30_znt
static inline void m_water_coare30_znt(thread Exact32 &z_0, Exact32 ustar, Exact32 wsp10, Exact32 visc, Exact32 zu) {
Exact32 czo2 = 0x1.6872b00000000p-7f;
Exact32 czc;
Exact32 wsp10m;
wsp10m = rr_div(rr_mul(wsp10,0x1.7069e20000000p+3f),rr_log(rr_div(zu,0x1.a36e2e0000000p-14f)));
czc = rr_add(0x1.6872b00000000p-7f,rr_mul(0x1.cac0840000000p-8f,rr_min(rr_max(rr_div(rr_sub(wsp10m,0x1.4000000000000p+3f),0x1.0000000000000p+3f),0x0.0p+0f),0x1.0000000000000p+0f)));
z_0 = rr_add(rr_mul(rr_mul(rr_mul(czc,ustar),ustar),0x1.a1b2560000000p-4f),rr_div(rr_mul(0x1.c28f5c0000000p-4f,visc),rr_max(ustar,0x1.99999a0000000p-5f)));
z_0 = rr_max(z_0,0x1.10bafe0000000p-23f);
z_0 = rr_min(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1002-1020 water_edson_etal_2013
static inline void m_water_edson_etal_2013(thread Exact32 &z_0, Exact32 ustar, Exact32 wsp10, Exact32 visc, Exact32 zu) {
Exact32 m = 0x1.bda5120000000p-10f;
Exact32 b = -0x1.47ae140000000p-8f;
Exact32 czc;
Exact32 wsp10m;
wsp10m = rr_div(rr_mul(wsp10,0x1.7069e20000000p+3f),rr_log(rr_div(zu,0x1.a36e2e0000000p-14f)));
wsp10m = rr_min(0x1.3000000000000p+4f,wsp10m);
czc = rr_add(rr_mul(0x1.bda5120000000p-10f,wsp10m),-0x1.47ae140000000p-8f);
czc = rr_max(czc,0x0.0p+0f);
z_0 = rr_add(rr_mul(rr_mul(rr_mul(czc,ustar),ustar),0x1.a1b2560000000p-4f),rr_div(rr_mul(0x1.c28f5c0000000p-4f,visc),rr_max(ustar,0x1.99999a0000000p-5f)));
z_0 = rr_max(z_0,0x1.10bafe0000000p-23f);
z_0 = rr_min(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1027-1051 water_depth_dependent_z0
static inline Exact32 m_water_depth_dependent_z0(Exact32 water_depth, Exact32 z0, Exact32 ust) {
Exact32 result;
Exact32 depth_b;
Exact32 wt;
Exact32 effective_depth;
if ((water_depth<0x1.4000000000000p+3f)) {
effective_depth = 0x1.4000000000000p+3f;
} else if ((water_depth>0x1.9000000000000p+6f)) {
effective_depth = 0x1.9000000000000p+6f;
} else {
effective_depth = water_depth;
}
depth_b = rr_mul(0x1.1111120000000p-5f,rr_log(rr_div(0x1.3b00000000000p+10f,effective_depth)));
result = rr_exp(rr_div(rr_sub(rr_mul(0x1.59999a0000000p+1f,ust),rr_div(0x1.cccccc0000000p+0f,depth_b)),rr_add(ust,rr_div(0x1.5c28f60000000p-3f,depth_b))));
result = rr_min(result,0x1.99999a0000000p-4f);
wt = rr_min(0x1.0000000000000p+0f,rr_max(0x0.0p+0f,rr_div(rr_sub(water_depth,0x1.2c00000000000p+6f),0x1.2c00000000000p+6f)));
result = rr_add(rr_mul(rr_sub(0x1.0000000000000p+0f,wt),result),rr_mul(wt,z0));
return result;
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1061-1082 water_garratt_1992
static inline void m_water_garratt_1992(thread Exact32 &zt, thread Exact32 &zq, Exact32 z_0, Exact32 ren, Exact32 landsea) {
Exact32 rq;
Exact32 e = 0x1.5bf0a80000000p+1f;
if ((rr_sub(landsea,0x1.8000000000000p+0f)>0)) {
zt = rr_mul(z_0,rr_exp(rr_sub(0x1.0000000000000p+1f,rr_mul(0x1.3d70a40000000p+1f,rr_pow(ren,0x1.0000000000000p-2f)))));
zq = rr_mul(z_0,rr_exp(rr_sub(0x1.0000000000000p+1f,rr_mul(0x1.23d70a0000000p+1f,rr_pow(ren,0x1.0000000000000p-2f)))));
zq = rr_min(zq,0x1.cd5f9a0000000p-15f);
zq = rr_max(zq,0x1.12e0be0000000p-29f);
zt = rr_min(zt,0x1.cd5f9a0000000p-15f);
zt = rr_max(zt,0x1.12e0be0000000p-29f);
} else {
zq = rr_div(z_0,0x1.d8e64a0000000p+2f);
zt = zq;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1093-1127 water_fairall_etal_2003
static inline void m_water_fairall_etal_2003(thread Exact32 &zt, thread Exact32 &zq, Exact32 ren, Exact32 ustar, Exact32 visc, Exact32 rstoch, int spp_sfc) {
if ((ren<=0x1.0000000000000p+1f)) {
zt = rr_mul(0x1.cd5f9a0000000p-15f,rr_pow(ren,-0x1.3333340000000p-1f));
zq = zt;
} else {
zt = rr_mul(0x1.cd5f9a0000000p-15f,rr_pow(ren,-0x1.3333340000000p-1f));
zq = zt;
}
if ((spp_sfc==1)) {
zt = rr_add(zt,rr_mul(rr_mul(zt,0x1.0000000000000p-1f),rstoch));
zq = zt;
}
zt = rr_min(zt,0x1.a36e2e0000000p-14f);
zt = rr_max(zt,0x1.12e0be0000000p-29f);
zq = rr_min(zq,0x1.a36e2e0000000p-14f);
zq = rr_max(zq,0x1.12e0be0000000p-29f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1135-1157 water_fairall_etal_2014
static inline void m_water_fairall_etal_2014(thread Exact32 &zt, thread Exact32 &zq, Exact32 ren, Exact32 ustar, Exact32 visc, Exact32 rstoch, int spp_sfc) {
zt = rr_min(0x1.4f8b580000000p-13f,rr_div(0x1.e68a0e0000000p-15f,rr_pow(ren,0x1.70a3d80000000p-1f)));
zq = zt;
if ((spp_sfc==1)) {
zt = rr_add(zt,rr_mul(rr_mul(zt,0x1.0000000000000p-1f),rstoch));
zq = rr_add(zq,rr_mul(rr_mul(zq,0x1.0000000000000p-1f),rstoch));
}
zt = rr_min(zt,0x1.a36e2e0000000p-14f);
zt = rr_max(zt,0x1.12e0be0000000p-29f);
zq = rr_min(zq,0x1.a36e2e0000000p-14f);
zq = rr_max(zq,0x1.12e0be0000000p-29f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1160-1209 water_gfs_z0_wat
static inline void m_water_gfs_z0_wat(thread Exact32 &z0rl, thread Exact32 &ustar, Exact32 wspd, Exact32 z1, int sfc_z0_type, bool redrag) {
Exact32 z0;
Exact32 z0max;
Exact32 wind10m;
Exact32 charnock = 0x1.cac0840000000p-7f;
Exact32 z0s_max = 0x1.9f7f8c0000000p-9f;
z0 = z0rl;
z0max = rr_max(0x1.0c6f7a0000000p-20f,rr_min(z0,z1));
ustar = rr_sqrt(rr_div(rr_mul(0x1.39cc100000000p+3f,z0),0x1.cac0840000000p-7f));
wind10m = rr_div(rr_mul(wspd,0x1.7069e20000000p+3f),rr_log(rr_div(z1,0x1.a36e2e0000000p-14f)));
if ((sfc_z0_type>=0)) {
if ((sfc_z0_type==0)) {
z0 = rr_mul(rr_mul(0x1.7641960000000p-10f,ustar),ustar);
if (redrag) {
z0rl = rr_max(rr_min(z0,0x1.9f7f8c0000000p-9f),0x1.ad7f2a0000000p-24f);
} else {
z0rl = rr_max(rr_min(z0,0x1.99999a0000000p-4f),0x1.ad7f2a0000000p-24f);
}
} else if ((sfc_z0_type==6)) {
m_water_znot_m_v6(wind10m,z0);
} else if ((sfc_z0_type==7)) {
m_water_znot_m_v7(wind10m,z0);
} else {
z0rl = 0x1.0c6f7a0000000p-20f;
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1212-1257 water_gfs_zt_wat
static inline void m_water_gfs_zt_wat(thread Exact32 &ztmax, Exact32 z0rl, Exact32 restar, Exact32 wspd, Exact32 z1, int sfc_z0_type, thread int &errflg) {
Exact32 z0;
Exact32 z0max;
Exact32 wind10m;
Exact32 rat;
Exact32 ustar;
Exact32 charnock = 0x1.cac0840000000p-7f;
Exact32 z0s_max = 0x1.9f7f8c0000000p-9f;
errflg = 0;
z0 = z0rl;
z0max = rr_max(0x1.0c6f7a0000000p-20f,rr_min(z0,z1));
ustar = rr_sqrt(rr_div(rr_mul(0x1.39cc100000000p+3f,z0),0x1.cac0840000000p-7f));
wind10m = rr_div(rr_mul(wspd,0x1.7069e20000000p+3f),rr_log(rr_div(z1,0x1.a36e2e0000000p-14f)));
rat = rr_min(0x1.c000000000000p+2f,rr_sub(rr_mul(0x1.55c2900000000p+1f,rr_sqrt(rr_sqrt(restar))),0x1.48f5c20000000p+1f));
ztmax = rr_max(rr_mul(z0max,rr_exp((-rat))),0x1.0c6f7a0000000p-20f);
if ((sfc_z0_type==6)) {
m_water_znot_t_v6(wind10m,ztmax);
} else if ((sfc_z0_type==7)) {
m_water_znot_t_v7(wind10m,ztmax);
} else if ((sfc_z0_type>0)) {
errflg = 1;
return;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1262-1304 water_znot_m_v6
static inline void m_water_znot_m_v6(Exact32 uref, thread Exact32 &znotm) {
Exact32 p13 = -0x1.a8d8240000000p-7f;
Exact32 p12 = 0x1.246e940000000p-2f;
Exact32 p11 = -0x1.990fe00000000p+0f;
Exact32 p10 = -0x1.0cb4060000000p+3f;
Exact32 p25 = 0x1.a0ceda0000000p-32f;
Exact32 p24 = 0x1.c311da0000000p-29f;
Exact32 p23 = 0x1.a5659e0000000p-23f;
Exact32 p22 = -0x1.4cec960000000p-20f;
Exact32 p21 = 0x1.759c440000000p-23f;
Exact32 p20 = 0x1.68404a0000000p-16f;
Exact32 p35 = 0x1.8b3ab80000000p-23f;
Exact32 p34 = -0x1.d4baee0000000p-16f;
Exact32 p33 = 0x1.c6e6900000000p-10f;
Exact32 p32 = -0x1.f6eec80000000p-5f;
Exact32 p31 = 0x1.4165b00000000p+0f;
Exact32 p30 = -0x1.0a3d2e0000000p+4f;
Exact32 p40 = 0x1.e02e820000000p-12f;
if (((uref>=0x0.0p+0f)&&(uref<=0x1.a000000000000p+2f))) {
znotm = rr_exp(rr_add(-0x1.0cb4060000000p+3f,rr_mul(uref,rr_add(-0x1.990fe00000000p+0f,rr_mul(uref,rr_add(0x1.246e940000000p-2f,rr_mul(uref,-0x1.a8d8240000000p-7f)))))));
} else if (((uref>0x1.a000000000000p+2f)&&(uref<=0x1.f666660000000p+3f))) {
znotm = rr_add(0x1.68404a0000000p-16f,rr_mul(uref,rr_add(0x1.759c440000000p-23f,rr_mul(uref,rr_add(-0x1.4cec960000000p-20f,rr_mul(uref,rr_add(0x1.a5659e0000000p-23f,rr_mul(uref,rr_add(0x1.c311da0000000p-29f,rr_mul(uref,0x1.a0ceda0000000p-32f))))))))));
} else if (((uref>0x1.f666660000000p+3f)&&(uref<=0x1.a800000000000p+5f))) {
znotm = rr_exp(rr_add(-0x1.0a3d2e0000000p+4f,rr_mul(uref,rr_add(0x1.4165b00000000p+0f,rr_mul(uref,rr_add(-0x1.f6eec80000000p-5f,rr_mul(uref,rr_add(0x1.c6e6900000000p-10f,rr_mul(uref,rr_add(-0x1.d4baee0000000p-16f,rr_mul(uref,0x1.8b3ab80000000p-23f)))))))))));
} else if ((uref>0x1.a800000000000p+5f)) {
znotm = 0x1.e02e820000000p-12f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1315-1368 water_znot_t_v6
static inline void m_water_znot_t_v6(Exact32 uref, thread Exact32 &znott) {
Exact32 p00 = 0x1.cd5f9a0000000p-14f;
Exact32 p15 = -0x1.f6ba8c0000000p-31f;
Exact32 p14 = 0x1.2d85880000000p-24f;
Exact32 p13 = -0x1.2151ec0000000p-19f;
Exact32 p12 = 0x1.17a9dc0000000p-15f;
Exact32 p11 = -0x1.139e4c0000000p-12f;
Exact32 p10 = 0x1.c4ae900000000p-11f;
Exact32 p25 = -0x1.3080ec0000000p-37f;
Exact32 p24 = 0x1.52c10a0000000p-30f;
Exact32 p23 = -0x1.25afc20000000p-24f;
Exact32 p22 = 0x1.f65a2a0000000p-20f;
Exact32 p21 = -0x1.ac322c0000000p-16f;
Exact32 p20 = 0x1.2bad260000000p-13f;
Exact32 p35 = 0x1.c36aea0000000p-39f;
Exact32 p34 = -0x1.43e3d40000000p-32f;
Exact32 p33 = 0x1.2dfc9e0000000p-27f;
Exact32 p32 = -0x1.47a8ce0000000p-25f;
Exact32 p31 = -0x1.48b2700000000p-19f;
Exact32 p30 = 0x1.204b0a0000000p-15f;
Exact32 p45 = -0x1.3690060000000p-35f;
Exact32 p44 = 0x1.0ebee40000000p-28f;
Exact32 p43 = -0x1.a5d2da0000000p-27f;
Exact32 p42 = -0x1.cad8c40000000p-17f;
Exact32 p41 = 0x1.39f4ce0000000p-11f;
Exact32 p40 = -0x1.fba9280000000p-8f;
Exact32 p56 = -0x1.0b82860000000p-43f;
Exact32 p55 = 0x1.a663120000000p-35f;
Exact32 p54 = -0x1.1491840000000p-27f;
Exact32 p53 = 0x1.80edfa0000000p-21f;
Exact32 p52 = -0x1.2c744e0000000p-15f;
Exact32 p51 = 0x1.f247720000000p-11f;
Exact32 p50 = -0x1.53b2fa0000000p-7f;
Exact32 p60 = 0x1.8e90740000000p-15f;
if (((uref>=0x0.0p+0f)&&(uref<0x1.79999a0000000p+2f))) {
znott = 0x1.cd5f9a0000000p-14f;
} else if (((uref>=0x1.79999a0000000p+2f)&&(uref<=0x1.eccccc0000000p+3f))) {
znott = rr_add(0x1.c4ae900000000p-11f,rr_mul(uref,rr_add(-0x1.139e4c0000000p-12f,rr_mul(uref,rr_add(0x1.17a9dc0000000p-15f,rr_mul(uref,rr_add(-0x1.2151ec0000000p-19f,rr_mul(uref,rr_add(0x1.2d85880000000p-24f,rr_mul(uref,-0x1.f6ba8c0000000p-31f))))))))));
} else if (((uref>0x1.eccccc0000000p+3f)&&(uref<=0x1.59999a0000000p+4f))) {
znott = rr_add(0x1.2bad260000000p-13f,rr_mul(uref,rr_add(-0x1.ac322c0000000p-16f,rr_mul(uref,rr_add(0x1.f65a2a0000000p-20f,rr_mul(uref,rr_add(-0x1.25afc20000000p-24f,rr_mul(uref,rr_add(0x1.52c10a0000000p-30f,rr_mul(uref,-0x1.3080ec0000000p-37f))))))))));
} else if (((uref>0x1.59999a0000000p+4f)&&(uref<=0x1.51999a0000000p+5f))) {
znott = rr_add(0x1.204b0a0000000p-15f,rr_mul(uref,rr_add(-0x1.48b2700000000p-19f,rr_mul(uref,rr_add(-0x1.47a8ce0000000p-25f,rr_mul(uref,rr_add(0x1.2dfc9e0000000p-27f,rr_mul(uref,rr_add(-0x1.43e3d40000000p-32f,rr_mul(uref,0x1.c36aea0000000p-39f))))))))));
} else if (((uref>0x1.51999a0000000p+5f)&&(uref<=0x1.aa66660000000p+5f))) {
znott = rr_add(-0x1.fba9280000000p-8f,rr_mul(uref,rr_add(0x1.39f4ce0000000p-11f,rr_mul(uref,rr_add(-0x1.cad8c40000000p-17f,rr_mul(uref,rr_add(-0x1.a5d2da0000000p-27f,rr_mul(uref,rr_add(0x1.0ebee40000000p-28f,rr_mul(uref,-0x1.3690060000000p-35f))))))))));
} else if (((uref>0x1.aa66660000000p+5f)&&(uref<=0x1.4000000000000p+6f))) {
znott = rr_add(-0x1.53b2fa0000000p-7f,rr_mul(uref,rr_add(0x1.f247720000000p-11f,rr_mul(uref,rr_add(-0x1.2c744e0000000p-15f,rr_mul(uref,rr_add(0x1.80edfa0000000p-21f,rr_mul(uref,rr_add(-0x1.1491840000000p-27f,rr_mul(uref,rr_add(0x1.a663120000000p-35f,rr_mul(uref,-0x1.0b82860000000p-43f))))))))))));
} else if ((uref>0x1.4000000000000p+6f)) {
znott = 0x1.8e90740000000p-15f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1378-1415 water_znot_m_v7
static inline void m_water_znot_m_v7(Exact32 uref, thread Exact32 &znotm) {
Exact32 p13 = -0x1.a8d8240000000p-7f;
Exact32 p12 = 0x1.246e940000000p-2f;
Exact32 p11 = -0x1.990fe00000000p+0f;
Exact32 p10 = -0x1.0cb4060000000p+3f;
Exact32 p25 = 0x1.a0ceda0000000p-32f;
Exact32 p24 = 0x1.c311da0000000p-29f;
Exact32 p23 = 0x1.a5659e0000000p-23f;
Exact32 p22 = -0x1.4cec960000000p-20f;
Exact32 p21 = 0x1.759c440000000p-23f;
Exact32 p20 = 0x1.68404a0000000p-16f;
Exact32 p35 = 0x1.977e100000000p-23f;
Exact32 p34 = -0x1.fa965e0000000p-16f;
Exact32 p33 = 0x1.fa4d960000000p-10f;
Exact32 p32 = -0x1.166ac80000000p-4f;
Exact32 p31 = 0x1.58c51e0000000p+0f;
Exact32 p30 = -0x1.11416a0000000p+4f;
Exact32 p40 = 0x1.6185120000000p-12f;
if (((uref>=0x0.0p+0f)&&(uref<=0x1.a000000000000p+2f))) {
znotm = rr_exp(rr_add(-0x1.0cb4060000000p+3f,rr_mul(uref,rr_add(-0x1.990fe00000000p+0f,rr_mul(uref,rr_add(0x1.246e940000000p-2f,rr_mul(uref,-0x1.a8d8240000000p-7f)))))));
} else if (((uref>0x1.a000000000000p+2f)&&(uref<=0x1.f666660000000p+3f))) {
znotm = rr_add(0x1.68404a0000000p-16f,rr_mul(uref,rr_add(0x1.759c440000000p-23f,rr_mul(uref,rr_add(-0x1.4cec960000000p-20f,rr_mul(uref,rr_add(0x1.a5659e0000000p-23f,rr_mul(uref,rr_add(0x1.c311da0000000p-29f,rr_mul(uref,0x1.a0ceda0000000p-32f))))))))));
} else if (((uref>0x1.f666660000000p+3f)&&(uref<=0x1.a800000000000p+5f))) {
znotm = rr_exp(rr_add(-0x1.11416a0000000p+4f,rr_mul(uref,rr_add(0x1.58c51e0000000p+0f,rr_mul(uref,rr_add(-0x1.166ac80000000p-4f,rr_mul(uref,rr_add(0x1.fa4d960000000p-10f,rr_mul(uref,rr_add(-0x1.fa965e0000000p-16f,rr_mul(uref,0x1.977e100000000p-23f)))))))))));
} else if ((uref>0x1.a800000000000p+5f)) {
znotm = 0x1.6185120000000p-12f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1424-1478 water_znot_t_v7
static inline void m_water_znot_t_v7(Exact32 uref, thread Exact32 &znott) {
Exact32 p00 = 0x1.cd5f9a0000000p-14f;
Exact32 p15 = -0x1.f96ebc0000000p-31f;
Exact32 p14 = 0x1.2ee3f40000000p-24f;
Exact32 p13 = -0x1.225e840000000p-19f;
Exact32 p12 = 0x1.186d960000000p-15f;
Exact32 p11 = -0x1.1426440000000p-12f;
Exact32 p10 = 0x1.c53ef60000000p-11f;
Exact32 p25 = -0x1.4ad4320000000p-37f;
Exact32 p24 = 0x1.6c527e0000000p-30f;
Exact32 p23 = -0x1.397efc0000000p-24f;
Exact32 p22 = 0x1.0a23fa0000000p-19f;
Exact32 p21 = -0x1.c1adba0000000p-16f;
Exact32 p20 = 0x1.3749fc0000000p-13f;
Exact32 p35 = 0x1.16b60c0000000p-37f;
Exact32 p34 = -0x1.181bba0000000p-30f;
Exact32 p33 = 0x1.c3246a0000000p-25f;
Exact32 p32 = -0x1.671fcc0000000p-20f;
Exact32 p31 = 0x1.1669020000000p-16f;
Exact32 p30 = -0x1.3d0b320000000p-14f;
Exact32 p45 = -0x1.283fca0000000p-32f;
Exact32 p44 = 0x1.f3b52a0000000p-25f;
Exact32 p43 = -0x1.4f99520000000p-18f;
Exact32 p42 = 0x1.c196940000000p-13f;
Exact32 p41 = -0x1.2caf0e0000000p-8f;
Exact32 p40 = 0x1.417bba0000000p-5f;
Exact32 p56 = -0x1.76ae640000000p-43f;
Exact32 p55 = 0x1.27c3220000000p-34f;
Exact32 p54 = -0x1.833ea00000000p-27f;
Exact32 p53 = 0x1.0d6c960000000p-20f;
Exact32 p52 = -0x1.a47d280000000p-15f;
Exact32 p51 = 0x1.5c96d20000000p-10f;
Exact32 p50 = -0x1.db28080000000p-7f;
Exact32 p60 = 0x1.1eec900000000p-14f;
if (((uref>=0x0.0p+0f)&&(uref<0x1.79999a0000000p+2f))) {
znott = 0x1.cd5f9a0000000p-14f;
} else if (((uref>=0x1.79999a0000000p+2f)&&(uref<=0x1.eccccc0000000p+3f))) {
znott = rr_add(0x1.c53ef60000000p-11f,rr_mul(uref,rr_add(-0x1.1426440000000p-12f,rr_mul(uref,rr_add(0x1.186d960000000p-15f,rr_mul(uref,rr_add(-0x1.225e840000000p-19f,rr_mul(uref,rr_add(0x1.2ee3f40000000p-24f,rr_mul(uref,-0x1.f96ebc0000000p-31f))))))))));
} else if (((uref>0x1.eccccc0000000p+3f)&&(uref<=0x1.59999a0000000p+4f))) {
znott = rr_add(0x1.3749fc0000000p-13f,rr_mul(uref,rr_add(-0x1.c1adba0000000p-16f,rr_mul(uref,rr_add(0x1.0a23fa0000000p-19f,rr_mul(uref,rr_add(-0x1.397efc0000000p-24f,rr_mul(uref,rr_add(0x1.6c527e0000000p-30f,rr_mul(uref,-0x1.4ad4320000000p-37f))))))))));
} else if (((uref>0x1.59999a0000000p+4f)&&(uref<=0x1.54cccc0000000p+5f))) {
znott = rr_add(-0x1.3d0b320000000p-14f,rr_mul(uref,rr_add(0x1.1669020000000p-16f,rr_mul(uref,rr_add(-0x1.671fcc0000000p-20f,rr_mul(uref,rr_add(0x1.c3246a0000000p-25f,rr_mul(uref,rr_add(-0x1.181bba0000000p-30f,rr_mul(uref,0x1.16b60c0000000p-37f))))))))));
} else if (((uref>0x1.54cccc0000000p+5f)&&(uref<=0x1.a800000000000p+5f))) {
znott = rr_add(0x1.417bba0000000p-5f,rr_mul(uref,rr_add(-0x1.2caf0e0000000p-8f,rr_mul(uref,rr_add(0x1.c196940000000p-13f,rr_mul(uref,rr_add(-0x1.4f99520000000p-18f,rr_mul(uref,rr_add(0x1.f3b52a0000000p-25f,rr_mul(uref,-0x1.283fca0000000p-32f))))))))));
} else if (((uref>0x1.a800000000000p+5f)&&(uref<=0x1.4000000000000p+6f))) {
znott = rr_add(-0x1.db28080000000p-7f,rr_mul(uref,rr_add(0x1.5c96d20000000p-10f,rr_mul(uref,rr_add(-0x1.a47d280000000p-15f,rr_mul(uref,rr_add(0x1.0d6c960000000p-20f,rr_mul(uref,rr_add(-0x1.833ea00000000p-27f,rr_mul(uref,rr_add(0x1.27c3220000000p-34f,rr_mul(uref,-0x1.76ae640000000p-43f))))))))))));
} else if ((uref>0x1.4000000000000p+6f)) {
znott = 0x1.1eec900000000p-14f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1484-1513 water_psi_hogstrom_1996
static inline void m_water_psi_hogstrom_1996(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za) {
Exact32 x;
Exact32 x0;
Exact32 y;
Exact32 y0;
Exact32 zml;
Exact32 zhl;
zml = rr_div(rr_mul(z_0,zl),za);
zhl = rr_div(rr_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = rr_mul(-0x1.5333340000000p+2f,rr_sub(zl,zml));
psi_h = rr_mul(-0x1.0000000000000p+3f,rr_sub(zl,zhl));
} else {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,x),rr_add(0x1.0000000000000p+0f,x0)))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),rr_add(0x1.0000000000000p+0f,rr_mul(x0,x0))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),rr_mul(0x1.0000000000000p+1f,rr_atan(x0)));
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),rr_add(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1520-1550 water_psi_dyerhicks
static inline void m_water_psi_dyerhicks(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za) {
Exact32 x;
Exact32 x0;
Exact32 y;
Exact32 y0;
Exact32 zml;
Exact32 zhl;
zml = rr_div(rr_mul(z_0,zl),za);
zhl = rr_div(rr_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = rr_mul(-0x1.4000000000000p+2f,rr_sub(zl,zml));
psi_h = rr_mul(-0x1.4000000000000p+2f,rr_sub(zl,zhl));
} else {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,x),rr_add(0x1.0000000000000p+0f,x0)))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),rr_add(0x1.0000000000000p+0f,rr_mul(x0,x0))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),rr_mul(0x1.0000000000000p+1f,rr_atan(x0)));
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),rr_add(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1556-1579 water_psi_beljaars_holtslag_1991
static inline void m_water_psi_beljaars_holtslag_1991(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 a = 0x1.0000000000000p+0f;
Exact32 b = 0x1.54fdf40000000p-1f;
Exact32 c = 0x1.4000000000000p+2f;
Exact32 d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-rr_add(rr_add(rr_mul(0x1.0000000000000p+0f,zl),rr_mul(rr_mul(0x1.54fdf40000000p-1f,rr_sub(zl,0x1.c924920000000p+3f)),rr_exp(rr_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-rr_sub(rr_add(rr_add(rr_pow(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),rr_mul(rr_mul(0x1.54fdf40000000p-1f,rr_sub(zl,0x1.c924920000000p+3f)),rr_exp(rr_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1584-1609 water_psi_businger_1971
static inline void m_water_psi_businger_1971(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 x;
Exact32 y;
Exact32 pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_log(rr_mul(rr_div(rr_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),rr_div(rr_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),0x1.0000000000000p+1f))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = rr_mul(-0x1.2ccccc0000000p+2f,zl);
psi_h = rr_mul(-0x1.967c880000000p+2f,zl);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1618-1640 water_psi_suselj_sood_2010
static inline void m_water_psi_suselj_sood_2010(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 rfc = 0x1.851eb80000000p-3f;
Exact32 ric = 0x1.76c8b40000000p-3f;
Exact32 phit = 0x1.99999a0000000p-1f;
if ((zl>0x0.0p+0f)) {
psi_m = (-rr_add(rr_div(zl,0x1.851eb80000000p-3f),rr_mul(0x1.1f4f0e0000000p+0f,rr_exp(rr_sub(0x1.0000000000000p+0f,rr_div(0x1.aaa64c0000000p+0f,zl))))));
psi_h = (-rr_add(rr_div(rr_mul(zl,0x1.76c8b40000000p-3f),0x1.71a9fc0000000p-3f),rr_mul(0x1.c5c2900000000p+2f,rr_pow(zl,0x1.1bedfa0000000p+0f))));
} else {
psi_m = rr_mul(0x1.fb15b60000000p-1f,rr_log(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.c872b00000000p+3f,zl))));
psi_h = rr_mul(0x1.02a3060000000p+0f,rr_log(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.04e7d60000000p+4f,zl))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1646-1657 water_psi_cb2005
static inline void m_water_psi_cb2005(thread Exact32 &psim1, thread Exact32 &psih1, Exact32 zl, Exact32 z0l) {
psim1 = rr_sub(rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zl,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),rr_mul(0x1.8666660000000p+2f,rr_log(rr_add(z0l,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = rr_sub(rr_mul(-0x1.6000000000000p+2f,rr_log(rr_add(zl,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),rr_mul(0x1.6000000000000p+2f,rr_log(rr_add(z0l,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1663-1719 water_li_etal_2010
static inline void m_water_li_etal_2010(thread Exact32 &zl, Exact32 rib, Exact32 zaz0, Exact32 z0zt) {
Exact32 alfa;
Exact32 beta;
Exact32 zaz02;
Exact32 z0zt2;
Exact32 au11 = 0x1.70a3d80000000p-5f;
Exact32 bu11 = 0x1.89374c0000000p-9f;
Exact32 bu12 = 0x1.82a9940000000p-8f;
Exact32 bu21 = -0x1.5326180000000p-4f;
Exact32 bu22 = 0x1.c4dd300000000p-1f;
Exact32 bu31 = 0x1.6425ae0000000p-3f;
Exact32 bu32 = -0x1.d7b4a20000000p-1f;
Exact32 bu33 = -0x1.b0f27c0000000p-4f;
Exact32 aw11 = 0x1.25c91e0000000p-1f;
Exact32 aw12 = -0x1.c275260000000p-2f;
Exact32 aw21 = -0x1.39a9fc0000000p+2f;
Exact32 aw22 = 0x1.a400000000000p+5f;
Exact32 bw11 = -0x1.b98c7e0000000p-5f;
Exact32 bw12 = 0x1.8a3d700000000p+0f;
Exact32 bw21 = -0x1.56872c0000000p-1f;
Exact32 bw22 = -0x1.a418940000000p+1f;
Exact32 as11 = 0x1.817c1c0000000p-1f;
Exact32 as21 = 0x1.de147a0000000p+3f;
Exact32 bs11 = 0x1.4154ca0000000p-3f;
Exact32 bs21 = -0x1.3c84b60000000p-2f;
Exact32 bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = rr_log(zaz02);
beta = rr_log(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = rr_add(rr_mul(rr_mul(0x1.70a3d80000000p-5f,alfa),rr_mul(rib,rib)),rr_mul(rr_add(rr_add(rr_mul(rr_add(rr_mul(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),rr_mul(alfa,alfa)),rr_mul(rr_add(rr_mul(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),rr_add(rr_add(rr_mul(0x1.6425ae0000000p-3f,rr_mul(beta,beta)),rr_mul(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = rr_max(zl,-0x1.e000000000000p+3f);
zl = rr_min(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = rr_add(rr_mul(rr_add(rr_mul(rr_add(rr_mul(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),rr_add(rr_mul(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),rr_mul(rib,rib)),rr_mul(rr_add(rr_mul(rr_add(rr_mul(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),rr_add(rr_mul(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = rr_min(zl,0x1.0000000000000p+2f);
zl = rr_max(zl,0x0.0p+0f);
} else {
zl = rr_add(rr_add(rr_add(rr_mul(rr_add(rr_mul(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),rr_mul(0x1.4154ca0000000p-3f,alfa)),rr_mul(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = rr_min(zl,0x1.4000000000000p+4f);
zl = rr_max(zl,0x1.0000000000000p+0f);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1722-1775 water_zolri
static inline Exact32 m_water_zolri(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 x1;
Exact32 x2;
Exact32 fx1;
Exact32 fx2;
int n;
int nmax = 20;
if ((ri<0x0.0p+0f)) {
x1 = rr_sub(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = rr_add(zol1,0x1.47ae140000000p-6f);
}
n = 1;
fx1 = m_water_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
fx2 = m_water_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
while (((rr_abs(rr_sub(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((rr_abs(fx2)<rr_abs(fx1))) {
x1 = rr_sub(x1,rr_mul(rr_div(fx1,rr_sub(fx2,fx1)),rr_sub(x2,x1)));
fx1 = m_water_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x1;
} else {
x2 = rr_sub(x2,rr_mul(rr_div(fx2,rr_sub(fx2,fx1)),rr_sub(x2,x1)));
fx2 = m_water_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(rr_abs(rr_sub(x1,x2))>=0x1.47ae140000000p-7f))) {
m_water_li_etal_2010(result,ri,rr_div(za,z0),rr_div(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1777-1817 water_zolri2
static inline Exact32 m_water_zolri2(thread Exact32 &zol2, Exact32 ri2, Exact32 za, Exact32 z0, Exact32 zt, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 zol20;
Exact32 zol3;
Exact32 psim1;
Exact32 psih1;
Exact32 psix2;
Exact32 psit2;
Exact32 zolt;
if ((rr_mul(zol2,ri2)<0x0.0p+0f)) zol2 = 0x0.0p+0f;
zol20 = rr_div(rr_mul(zol2,z0),za);
zol3 = rr_add(zol2,zol20);
zolt = rr_div(rr_mul(zol2,zt),za);
if ((ri2<0)) {
psit2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),zt)),rr_sub(m_water_psih_unstable(zol3,psi_opt,psih_unstab),m_water_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),z0)),rr_sub(m_water_psim_unstable(zol3,psi_opt,psim_unstab),m_water_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),zt)),rr_sub(m_water_psih_stable(zol3,psi_opt,psih_stab),m_water_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),z0)),rr_sub(m_water_psim_stable(zol3,psi_opt,psim_stab),m_water_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = rr_sub(rr_div(rr_mul(zol2,psit2),rr_mul(psix2,psix2)),ri2);
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1820-1894 water_zolrib
static inline Exact32 m_water_zolrib(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 logz0, Exact32 logzt, thread Exact32 &zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 zol20;
Exact32 zol3;
Exact32 zolt;
Exact32 zolold;
int n;
int nmax = 20;
Exact32 psit2;
Exact32 psix2;
if ((rr_mul(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((rr_abs(rr_sub(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = rr_div(rr_mul(zolold,z0),za);
zol3 = rr_add(zolold,zol20);
zolt = rr_div(rr_mul(zolold,zt),za);
if ((ri<0)) {
psit2 = rr_max(rr_sub(logzt,rr_sub(m_water_psih_unstable(zol3,psi_opt,psih_unstab),m_water_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(logz0,rr_sub(m_water_psim_unstable(zol3,psi_opt,psim_unstab),m_water_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = rr_max(rr_sub(logzt,rr_sub(m_water_psih_stable(zol3,psi_opt,psih_stab),m_water_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(logz0,rr_sub(m_water_psim_stable(zol3,psi_opt,psim_stab),m_water_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = rr_div(rr_mul(ri,rr_mul(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(rr_abs(rr_sub(zolold,result))>0x1.47ae140000000p-7f))) {
m_water_li_etal_2010(result,ri,rr_div(za,z0),rr_div(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1899-1906 water_psim_stable_full
static inline Exact32 m_water_psim_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1909-1916 water_psih_stable_full
static inline Exact32 m_water_psih_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.5333340000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1919-1934 water_psim_unstable_full
static inline Exact32 m_water_psim_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 x;
Exact32 ym;
Exact32 psimc;
Exact32 psimk;
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,x)))),rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,rr_mul(x,x))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
ym = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psimk,rr_mul(rr_mul(zolf,zolf),psimc)),rr_add(1,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1937-1951 water_psih_unstable_full
static inline Exact32 m_water_psih_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 y;
Exact32 yh;
Exact32 psihc;
Exact32 psihk;
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psihk,rr_mul(rr_mul(zolf,zolf),psihc)),rr_add(0x1.0000000000000p+0f,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1958-1967 water_psim_stable_full_gfs
static inline Exact32 m_water_psim_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 aa;
aa = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,aa),rr_log(rr_add(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1971-1980 water_psih_stable_full_gfs
static inline Exact32 m_water_psih_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 bb;
bb = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,bb),rr_log(rr_add(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1984-2000 water_psim_unstable_full_gfs
static inline Exact32 m_water_psim_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0 = -0x1.fccccc0000000p+1f;
Exact32 a1 = 0x1.8a3d700000000p+3f;
Exact32 b1 = -0x1.f051ec0000000p+2f;
Exact32 b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fccccc0000000p+1f,rr_mul(0x1.8a3d700000000p+3f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.f051ec0000000p+2f,rr_mul(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_sub(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p+1f,rr_sqrt(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2004-2020 water_psih_unstable_full_gfs
static inline Exact32 m_water_psih_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0p = -0x1.fc39580000000p+2f;
Exact32 a1p = 0x1.8c00000000000p+4f;
Exact32 b1p = -0x1.168f5c0000000p+3f;
Exact32 b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fc39580000000p+2f,rr_mul(0x1.8c00000000000p+4f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.168f5c0000000p+3f,rr_mul(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_add(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2024-2041 water_psim_stable
static inline Exact32 m_water_psim_stable(Exact32 zolf, int psi_opt, device const float *psim_stab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul(zolf,0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psim_stab[nzol],rr_mul(rzol,rr_sub(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psim_stable_full(zolf);
} else {
result = m_water_psim_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2044-2061 water_psih_stable
static inline Exact32 m_water_psih_stable(Exact32 zolf, int psi_opt, device const float *psih_stab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul(zolf,0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psih_stab[nzol],rr_mul(rzol,rr_sub(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psih_stable_full(zolf);
} else {
result = m_water_psih_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2064-2081 water_psim_unstable
static inline Exact32 m_water_psim_unstable(Exact32 zolf, int psi_opt, device const float *psim_unstab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul((-zolf),0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psim_unstab[nzol],rr_mul(rzol,rr_sub(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psim_unstable_full(zolf);
} else {
result = m_water_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2084-2101 water_psih_unstable
static inline Exact32 m_water_psih_unstable(Exact32 zolf, int psi_opt, device const float *psih_unstab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul((-zolf),0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psih_unstab[nzol],rr_mul(rzol,rr_sub(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psih_unstable_full(zolf);
} else {
result = m_water_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:116-836 ice_mynnsfc_ice
static inline void m_ice_mynnsfc_ice(bool flag_iter, int itimestep, int i, int j, Exact32 dx, Exact32 xland, Exact32 u_1, Exact32 v_1, Exact32 t_1, Exact32 qv_1, Exact32 p_1, Exact32 dz8w_1, thread Exact32 &rho_1, Exact32 u_2, Exact32 v_2, Exact32 dz8w_2, Exact32 pblh, thread Exact32 &znt, Exact32 psfcpa, Exact32 mavail, Exact32 tskin, Exact32 tsurf, Exact32 snowh, thread Exact32 &qgh, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs2, thread Exact32 &cqs, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &stress, thread Exact32 &rmol, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &psim, thread Exact32 &psih, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &qsfc, thread Exact32 &t2, thread Exact32 &q2, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &lh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &rb, thread Exact32 &cpm, thread Exact32 &ch, thread Exact32 &cm, Exact32 rstoch_1, thread Exact32 &wstar, thread Exact32 &qstar, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &psix, thread Exact32 &psit, thread Exact32 &psix10, thread Exact32 &psit2, int spp_sfc, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, device const float *psim_stab, device const float *psim_unstab, device const float *psih_stab, device const float *psih_unstab, thread int &errflg, bool has_ck, bool has_cka, bool has_cd, bool has_cda) {
Exact32 xka = 0x1.92a7380000000p-16f;
Exact32 prt = 0x1.0000000000000p+0f;
Exact32 za;
Exact32 za2;
Exact32 thv_1;
Exact32 th_1;
Exact32 tc_1;
Exact32 tv_1;
Exact32 qvmr;
Exact32 psih2;
Exact32 psim10;
Exact32 psih10;
Exact32 wspdi;
Exact32 govrth;
Exact32 psfc;
Exact32 qsfcmr;
Exact32 thcon;
Exact32 zratio;
Exact32 tsk;
Exact32 thsk;
Exact32 thvsk;
Exact32 gz1ozt;
Exact32 gz2oz0;
Exact32 gz2ozt;
Exact32 gz10oz0;
Exact32 gz10ozt;
Exact32 zntstoch;
Exact32 zt;
Exact32 zq;
Exact32 psiq;
Exact32 psiq2;
int n;
int k;
int l;
int yesno;
Exact32 pl;
Exact32 e1;
Exact32 tabs;
Exact32 dthvdz;
Exact32 dthvm;
Exact32 vconv;
Exact32 zol2;
Exact32 zol10;
Exact32 zolza;
Exact32 zolz0;
Exact32 zolzt;
Exact32 dtg;
Exact32 dtthx;
Exact32 psiq10;
Exact32 fluxc;
Exact32 vsgd;
Exact32 restar;
Exact32 visc;
Exact32 dqg;
Exact32 oldust;
Exact32 oldtst;
errflg = 0;
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
psfc = rr_div(psfcpa,0x1.f400000000000p+9f);
if (flag_iter) {
if ((itimestep==1)) {
tsk = tskin;
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = rr_div(qsfc,rr_sub(0x1.0000000000000p+0f,qsfc));
} else {
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_add(rr_sub(rr_mul(0x1.2280000000000p+12f,rr_sub(0x1.dfda840000000p-9f,rr_div(0x1.0000000000000p+0f,tsk))),rr_mul(0x1.747ae20000000p+3f,rr_log(rr_div(0x1.1126660000000p+8f,tsk)))),rr_mul(0x1.7318fc0000000p-6f,rr_sub(0x1.1126660000000p+8f,tsk)))));
qsfc = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,rr_mul(0x1.8353dc0000000p-2f,e1)));
qsfcmr = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,e1));
}
} else {
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = rr_div(qsfc,rr_sub(0x1.0000000000000p+0f,qsfc));
} else {
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_add(rr_sub(rr_mul(0x1.2280000000000p+12f,rr_sub(0x1.dfda840000000p-9f,rr_div(0x1.0000000000000p+0f,tskin))),rr_mul(0x1.747ae20000000p+3f,rr_log(rr_div(0x1.1126660000000p+8f,tskin)))),rr_mul(0x1.7318fc0000000p-6f,rr_sub(0x1.1126660000000p+8f,tskin)))));
qsfc = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,rr_mul(0x1.8353dc0000000p-2f,e1)));
qsfcmr = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(psfc,e1));
}
}
}
e1 = rr_mul(0x1.38ef340000000p-1f,rr_exp(rr_add(rr_sub(rr_mul(0x1.2280000000000p+12f,rr_sub(0x1.dfda840000000p-9f,rr_div(0x1.0000000000000p+0f,t_1))),rr_mul(0x1.747ae20000000p+3f,rr_log(rr_div(0x1.1126660000000p+8f,t_1)))),rr_mul(0x1.7318fc0000000p-6f,rr_sub(0x1.1126660000000p+8f,t_1)))));
pl = rr_div(p_1,0x1.f400000000000p+9f);
qgh = rr_div(rr_mul(0x1.3e56120000000p-1f,e1),rr_sub(pl,e1));
qvmr = rr_div(qv_1,rr_sub(0x1.0000000000000p+0f,qv_1));
thcon = rr_pow(rr_div(0x1.86a0000000000p+16f,psfcpa),0x1.24924a0000000p-2f);
if (flag_iter) {
tsk = tskin;
thsk = rr_mul(tsk,thcon);
thvsk = rr_mul(thsk,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qsfc)));
if (((thvsk<0x1.4000000000000p+7f)||(thvsk>0x1.8600000000000p+8f))) {
}
}
th_1 = rr_mul(t_1,rr_pow(rr_div(0x1.86a0000000000p+16f,p_1),0x1.24924a0000000p-2f));
tc_1 = rr_sub(t_1,0x1.1126660000000p+8f);
thv_1 = rr_mul(th_1,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qv_1)));
tv_1 = rr_mul(t_1,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.377b480000000p-1f,qv_1)));
rho_1 = rr_div(p_1,rr_mul(0x1.1f00000000000p+8f,tv_1));
za = rr_mul(0x1.0000000000000p-1f,dz8w_1);
za2 = rr_add(dz8w_1,rr_mul(0x1.0000000000000p-1f,dz8w_2));
govrth = rr_div(0x1.39cc100000000p+3f,th_1);
cpm = rr_mul(0x1.f640000000000p+9f,rr_add(0x1.0000000000000p+0f,rr_mul(0x1.ae147a0000000p-1f,qv_1)));
if (flag_iter) {
wspd = rr_sqrt(rr_add(rr_mul(u_1,u_1),rr_mul(v_1,v_1)));
dthvdz = rr_sub(thv_1,thvsk);
fluxc = rr_max(rr_add(rr_div(rr_div(hfx,rho_1),0x1.f640000000000p+9f),rr_div(rr_mul(rr_mul(0x1.377b480000000p-1f,thvsk),qfx),rho_1)),0x0.0p+0f);
wstar = rr_mul(0x1.4000000000000p+0f,rr_pow(rr_mul(rr_mul(rr_div(0x1.39cc100000000p+3f,tsk),rr_min(rr_mul(0x1.8000000000000p+0f,pblh),0x1.f400000000000p+11f)),fluxc),0x1.5555560000000p-2f));
vsgd = rr_min(rr_mul(0x1.47ae140000000p-2f,rr_pow(rr_max(rr_sub(rr_div(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.5555560000000p-2f)),0x1.0000000000000p-1f);
wspd = rr_sqrt(rr_add(rr_add(rr_mul(wspd,wspd),rr_mul(wstar,wstar)),rr_mul(vsgd,vsgd)));
wspd = rr_max(wspd,0x1.99999a0000000p-4f);
rb = rr_div(rr_mul(rr_mul(govrth,za),dthvdz),rr_mul(wspd,wspd));
rb = rr_max(rb,-0x1.0000000000000p+1f);
rb = rr_min(rb,0x1.0000000000000p+1f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
}
if (flag_iter) {
visc = rr_mul(0x1.bcee880000000p-17f,rr_sub(rr_add(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.acbc8c0000000p-8f,tc_1)),rr_mul(rr_mul(0x1.16890c0000000p-17f,tc_1),tc_1)),rr_mul(rr_mul(rr_mul(0x1.4c9a2e0000000p-28f,tc_1),tc_1),tc_1)));
if ((spp_sfc==1)) {
zntstoch = rr_max(rr_add(znt,rr_mul(rr_mul(znt,0x1.0000000000000p+0f),rstoch_1)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = rr_max(rr_div(rr_mul(ust,zntstoch),visc),0x1.99999a0000000p-4f);
m_ice_andreas_2002(zntstoch,visc,ust,zt,zq,spp_sfc,rstoch_1);
gz1oz0 = rr_log(rr_div(rr_add(za,zntstoch),zntstoch));
gz1ozt = rr_log(rr_div(rr_add(za,zntstoch),zt));
gz2oz0 = rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zntstoch),zt));
gz10oz0 = rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zntstoch),zt));
zratio = rr_div(zntstoch,zt);
if ((rb>0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_ice_li_etal_2010(zol,rb,rr_div(za,zntstoch),zratio);
zol = rr_max(zol,0x0.0p+0f);
zol = rr_min(zol,0x1.4000000000000p+4f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_ice_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = rr_max(zol,0x0.0p+0f);
zol = rr_min(zol,0x1.4000000000000p+4f);
zolzt = rr_div(rr_mul(zol,zt),za);
zolz0 = rr_div(rr_mul(zol,zntstoch),za);
zolza = rr_div(rr_mul(zol,rr_add(za,zntstoch)),za);
zol10 = rr_div(rr_mul(zol,rr_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = rr_div(rr_mul(zol,rr_add(0x1.0000000000000p+1f,zntstoch)),za);
psim = rr_sub(m_ice_psim_stable(zolza,psi_opt,psim_stab),m_ice_psim_stable(zolz0,psi_opt,psim_stab));
psih = rr_sub(m_ice_psih_stable(zolza,psi_opt,psih_stab),m_ice_psih_stable(zolzt,psi_opt,psih_stab));
psim10 = rr_sub(m_ice_psim_stable(zol10,psi_opt,psim_stab),m_ice_psim_stable(zolz0,psi_opt,psim_stab));
psih10 = rr_sub(m_ice_psih_stable(zol10,psi_opt,psih_stab),m_ice_psih_stable(zolz0,psi_opt,psih_stab));
psih2 = rr_sub(m_ice_psih_stable(zol2,psi_opt,psih_stab),m_ice_psih_stable(zolzt,psi_opt,psih_stab));
rmol = rr_div(zol,za);
} else if ((rb==0x0.0p+0f)) {
psim = 0x0.0p+0f;
psih = 0x0.0p+0f;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = 0x0.0p+0f;
} else if ((rb<0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_ice_li_etal_2010(zol,rb,rr_div(za,zntstoch),zratio);
zol = rr_max(zol,-0x1.4000000000000p+4f);
zol = rr_min(zol,0x0.0p+0f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_ice_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = rr_max(zol,-0x1.4000000000000p+4f);
zol = rr_min(zol,0x0.0p+0f);
zolzt = rr_div(rr_mul(zol,zt),za);
zolz0 = rr_div(rr_mul(zol,zntstoch),za);
zolza = rr_div(rr_mul(zol,rr_add(za,zntstoch)),za);
zol10 = rr_div(rr_mul(zol,rr_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = rr_div(rr_mul(zol,rr_add(0x1.0000000000000p+1f,zntstoch)),za);
psim = rr_sub(m_ice_psim_unstable(zolza,psi_opt,psim_unstab),m_ice_psim_unstable(zolz0,psi_opt,psim_unstab));
psih = rr_sub(m_ice_psih_unstable(zolza,psi_opt,psih_unstab),m_ice_psih_unstable(zolzt,psi_opt,psih_unstab));
psim10 = rr_sub(m_ice_psim_unstable(zol10,psi_opt,psim_unstab),m_ice_psim_unstable(zolz0,psi_opt,psim_unstab));
psih10 = rr_sub(m_ice_psih_unstable(zol10,psi_opt,psih_unstab),m_ice_psih_unstable(zolz0,psi_opt,psih_unstab));
psih2 = rr_sub(m_ice_psih_unstable(zol2,psi_opt,psih_unstab),m_ice_psih_unstable(zolzt,psi_opt,psih_unstab));
psih = rr_min(psih,rr_mul(0x1.cccccc0000000p-1f,gz1ozt));
psim = rr_min(psim,rr_mul(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = rr_min(psih2,rr_mul(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = rr_min(psim10,rr_mul(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = rr_min(psih10,rr_mul(0x1.cccccc0000000p-1f,gz10ozt));
rmol = rr_div(zol,za);
}
psix = rr_max(rr_sub(gz1oz0,psim),0x1.0000000000000p+0f);
psix10 = rr_max(rr_sub(gz10oz0,psim10),0x1.0000000000000p+0f);
psit = rr_max(rr_sub(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = rr_max(rr_sub(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = rr_max(rr_sub(rr_log(rr_div(rr_add(za,zq),zq)),psih),0x1.0000000000000p+0f);
psiq2 = rr_max(rr_sub(rr_log(rr_div(rr_add(0x1.0000000000000p+1f,zq),zq)),psih2),0x1.0000000000000p+0f);
psiq10 = rr_max(rr_sub(rr_log(rr_div(rr_add(0x1.4000000000000p+3f,zq),zq)),psih10),0x1.0000000000000p+0f);
oldust = ust;
ust = rr_add(rr_mul(0x1.0000000000000p-1f,ust),rr_div(rr_mul(0x1.99999a0000000p-3f,wspd),psix));
ust = rr_max(ust,0x1.47ae140000000p-8f);
stress = rr_mul(ust,ust);
ustm = ust;
dtg = rr_sub(thv_1,thvsk);
oldtst = mol;
mol = rr_div(rr_div(rr_mul(0x1.99999a0000000p-2f,dtg),psit),0x1.0000000000000p+0f);
dqg = rr_mul(rr_sub(qv_1,qsfc),0x1.f400000000000p+9f);
qstar = rr_div(rr_div(rr_mul(0x1.99999a0000000p-2f,dqg),psiq),0x1.0000000000000p+0f);
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (flag_iter) {
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
cqs = 0x0.0p+0f;
cm = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = rr_div(rr_mul(rr_mul(rr_mul(rho_1,mavail),ust),0x1.99999a0000000p-2f),psiq);
flhc = rr_div(rr_mul(rr_mul(rr_mul(rho_1,cpm),ust),0x1.99999a0000000p-2f),psit);
if (compute_flux) {
qfx = rr_mul(flqc,rr_sub(qsfc,qv_1));
qfx = rr_max(qfx,-0x1.47ae140000000p-6f);
lh = rr_mul(0x1.55cc000000000p+18f,qfx);
hfx = rr_mul(rr_div(rr_mul(rr_div(rr_mul(rr_mul(rr_mul(rho_1,cpm),0x1.99999a0000000p-2f),wspd),psix),0x1.99999a0000000p-2f),psit),rr_sub(thsk,th_1));
hfx = rr_max(hfx,-0x1.f400000000000p+7f);
}
chs = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psit);
cqs = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psiq);
cqs2 = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psiq2);
chs2 = rr_div(rr_mul(ust,0x1.99999a0000000p-2f),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = rr_mul(rr_div(0x1.99999a0000000p-2f,psix10),rr_div(0x1.99999a0000000p-2f,psiq10));
cd = rr_mul(rr_div(0x1.99999a0000000p-2f,psix10),rr_div(0x1.99999a0000000p-2f,psix10));
cka = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psiq));
cda = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psix));
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
ch = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psit));
cm = rr_mul(rr_div(0x1.99999a0000000p-2f,psix),rr_div(0x1.99999a0000000p-2f,psix));
}
}
if (compute_diag) {
if (flag_iter) {
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u_2;
v10 = v_2;
} else {
u10 = rr_div(rr_mul(u_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
v10 = rr_div(rr_mul(v_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = rr_div(rr_mul(u_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
v10 = rr_div(rr_mul(v_1,rr_log(rr_div(0x1.4000000000000p+3f,zntstoch))),rr_log(rr_div(za,zntstoch)));
} else {
u10 = rr_div(rr_mul(u_1,psix10),psix);
v10 = rr_div(rr_mul(v_1,psix10),psix);
}
dtg = rr_sub(th_1,thsk);
th2 = rr_add(thsk,rr_div(rr_mul(dtg,psit2),psit));
if ((((th_1>thsk)&&((th2<thsk)||(th2>th_1)))||((th_1<thsk)&&((th2>thsk)||(th2<th_1))))) {
th2 = rr_add(thsk,rr_div(rr_mul(0x1.0000000000000p+1f,rr_sub(th_1,thsk)),za));
}
t2 = rr_mul(th2,rr_pow(rr_div(psfcpa,0x1.86a0000000000p+16f),0x1.24924a0000000p-2f));
q2 = rr_add(qsfc,rr_div(rr_mul(rr_sub(qv_1,qsfc),psiq2),psiq));
q2 = rr_max(q2,rr_min(qsfc,qv_1));
q2 = rr_min(q2,rr_mul(0x1.0ccccc0000000p+0f,qv_1));
}
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
yesno = 0;
if (compute_flux) {
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho_1<0x0.0p+0f)||(rho_1>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:846-898 ice_andreas_2002
static inline void m_ice_andreas_2002(Exact32 z_0, Exact32 bvisc, Exact32 ustar, thread Exact32 &zt, thread Exact32 &zq, int spp_sfc, Exact32 rstoch) {
Exact32 ren2;
Exact32 zntsno;
Exact32 bt0_s = 0x1.4000000000000p+0f;
Exact32 bt0_t = 0x1.3126ea0000000p-3f;
Exact32 bt0_r = 0x1.449ba60000000p-2f;
Exact32 bt1_s = 0x0.0p+0f;
Exact32 bt1_t = -0x1.19999a0000000p-1f;
Exact32 bt1_r = -0x1.2147ae0000000p-1f;
Exact32 bt2_s = 0x0.0p+0f;
Exact32 bt2_t = 0x0.0p+0f;
Exact32 bt2_r = -0x1.76c8b40000000p-3f;
Exact32 bq0_s = 0x1.9c28f60000000p+0f;
Exact32 bq0_t = 0x1.676c8c0000000p-2f;
Exact32 bq0_r = 0x1.9581060000000p-2f;
Exact32 bq1_s = 0x0.0p+0f;
Exact32 bq1_t = -0x1.4189380000000p-1f;
Exact32 bq1_r = -0x1.0624de0000000p-1f;
Exact32 bq2_s = 0x0.0p+0f;
Exact32 bq2_t = 0x0.0p+0f;
Exact32 bq2_r = -0x1.70a3d80000000p-3f;
zntsno = rr_add(rr_div(rr_mul(0x1.147ae20000000p-3f,bvisc),ustar),rr_mul(rr_mul(rr_mul(0x1.1eb8520000000p-5f,rr_mul(ustar,ustar)),0x1.a1b2560000000p-4f),rr_add(rr_mul(0x1.4000000000000p+2f,rr_exp(rr_mul(-0x1.0000000000000p+0f,rr_mul(rr_div(rr_sub(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f),rr_div(rr_sub(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f))))),0x1.0000000000000p+0f)));
ren2 = rr_div(rr_mul(ustar,zntsno),bvisc);
if ((ren2>0x1.f400000000000p+9f)) ren2 = 0x1.f400000000000p+9f;
if ((ren2<=0x1.147ae20000000p-3f)) {
zt = rr_mul(zntsno,rr_exp(rr_add(rr_add(0x1.4000000000000p+0f,rr_mul(0x0.0p+0f,rr_log(ren2))),rr_mul(0x0.0p+0f,rr_mul(rr_log(ren2),rr_log(ren2))))));
zq = rr_mul(zntsno,rr_exp(rr_add(rr_add(0x1.9c28f60000000p+0f,rr_mul(0x0.0p+0f,rr_log(ren2))),rr_mul(0x0.0p+0f,rr_mul(rr_log(ren2),rr_log(ren2))))));
} else if (((ren2>0x1.147ae20000000p-3f)&&(ren2<0x1.4000000000000p+1f))) {
zt = rr_mul(zntsno,rr_exp(rr_add(rr_add(0x1.3126ea0000000p-3f,rr_mul(-0x1.19999a0000000p-1f,rr_log(ren2))),rr_mul(0x0.0p+0f,rr_mul(rr_log(ren2),rr_log(ren2))))));
zq = rr_mul(zntsno,rr_exp(rr_add(rr_add(0x1.676c8c0000000p-2f,rr_mul(-0x1.4189380000000p-1f,rr_log(ren2))),rr_mul(0x0.0p+0f,rr_mul(rr_log(ren2),rr_log(ren2))))));
} else {
zt = rr_mul(zntsno,rr_exp(rr_add(rr_add(0x1.449ba60000000p-2f,rr_mul(-0x1.2147ae0000000p-1f,rr_log(ren2))),rr_mul(-0x1.76c8b40000000p-3f,rr_mul(rr_log(ren2),rr_log(ren2))))));
zq = rr_mul(zntsno,rr_exp(rr_add(rr_add(0x1.9581060000000p-2f,rr_mul(-0x1.0624de0000000p-1f,rr_log(ren2))),rr_mul(-0x1.70a3d80000000p-3f,rr_mul(rr_log(ren2),rr_log(ren2))))));
}
if ((spp_sfc==1)) {
zt = rr_add(zt,rr_mul(rr_mul(zt,0x1.0000000000000p-1f),rstoch));
zt = rr_max(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:903-932 ice_psi_hogstrom_1996
static inline void m_ice_psi_hogstrom_1996(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za) {
Exact32 x;
Exact32 x0;
Exact32 y;
Exact32 y0;
Exact32 zml;
Exact32 zhl;
zml = rr_div(rr_mul(z_0,zl),za);
zhl = rr_div(rr_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = rr_mul(-0x1.5333340000000p+2f,rr_sub(zl,zml));
psi_h = rr_mul(-0x1.0000000000000p+3f,rr_sub(zl,zhl));
} else {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,x),rr_add(0x1.0000000000000p+0f,x0)))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),rr_add(0x1.0000000000000p+0f,rr_mul(x0,x0))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),rr_mul(0x1.0000000000000p+1f,rr_atan(x0)));
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),rr_add(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:939-969 ice_psi_dyerhicks
static inline void m_ice_psi_dyerhicks(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl, Exact32 zt, Exact32 z_0, Exact32 za) {
Exact32 x;
Exact32 x0;
Exact32 y;
Exact32 y0;
Exact32 zml;
Exact32 zhl;
zml = rr_div(rr_mul(z_0,zl),za);
zhl = rr_div(rr_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = rr_mul(-0x1.4000000000000p+2f,rr_sub(zl,zml));
psi_h = rr_mul(-0x1.4000000000000p+2f,rr_sub(zl,zhl));
} else {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,x),rr_add(0x1.0000000000000p+0f,x0)))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),rr_add(0x1.0000000000000p+0f,rr_mul(x0,x0))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),rr_mul(0x1.0000000000000p+1f,rr_atan(x0)));
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),rr_add(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:975-998 ice_psi_beljaars_holtslag_1991
static inline void m_ice_psi_beljaars_holtslag_1991(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 a = 0x1.0000000000000p+0f;
Exact32 b = 0x1.54fdf40000000p-1f;
Exact32 c = 0x1.4000000000000p+2f;
Exact32 d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-rr_add(rr_add(rr_mul(0x1.0000000000000p+0f,zl),rr_mul(rr_mul(0x1.54fdf40000000p-1f,rr_sub(zl,0x1.c924920000000p+3f)),rr_exp(rr_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-rr_sub(rr_add(rr_add(rr_pow(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),rr_mul(rr_mul(0x1.54fdf40000000p-1f,rr_sub(zl,0x1.c924920000000p+3f)),rr_exp(rr_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1005-1027 ice_psi_zilitinkevich_esau_2007
static inline void m_ice_psi_zilitinkevich_esau_2007(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 cm = 0x1.8000000000000p+1f;
Exact32 ct = 0x1.4000000000000p+1f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = rr_mul(-0x1.8000000000000p+1f,rr_pow(zl,0x1.aaaaaa0000000p-1f));
psi_h = rr_mul(-0x1.4000000000000p+1f,rr_pow(zl,0x1.99999a0000000p-1f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1032-1057 ice_psi_businger_1971
static inline void m_ice_psi_businger_1971(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 x;
Exact32 y;
Exact32 pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = rr_add(rr_sub(rr_add(rr_log(rr_mul(rr_div(rr_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),rr_div(rr_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),rr_log(rr_div(rr_add(0x1.0000000000000p+0f,rr_mul(x,x)),0x1.0000000000000p+1f))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
psi_h = rr_mul(0x1.0000000000000p+1f,rr_log(rr_div(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = rr_mul(-0x1.2ccccc0000000p+2f,zl);
psi_h = rr_mul(-0x1.967c880000000p+2f,zl);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1066-1088 ice_psi_suselj_sood_2010
static inline void m_ice_psi_suselj_sood_2010(thread Exact32 &psi_m, thread Exact32 &psi_h, Exact32 zl) {
Exact32 rfc = 0x1.851eb80000000p-3f;
Exact32 ric = 0x1.76c8b40000000p-3f;
Exact32 phit = 0x1.99999a0000000p-1f;
if ((zl>0x0.0p+0f)) {
psi_m = (-rr_add(rr_div(zl,0x1.851eb80000000p-3f),rr_mul(0x1.1f4f0e0000000p+0f,rr_exp(rr_sub(0x1.0000000000000p+0f,rr_div(0x1.aaa64c0000000p+0f,zl))))));
psi_h = (-rr_add(rr_div(rr_mul(zl,0x1.76c8b40000000p-3f),0x1.71a9fc0000000p-3f),rr_mul(0x1.c5c2900000000p+2f,rr_pow(zl,0x1.1bedfa0000000p+0f))));
} else {
psi_m = rr_mul(0x1.fb15b60000000p-1f,rr_log(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.c872b00000000p+3f,zl))));
psi_h = rr_mul(0x1.02a3060000000p+0f,rr_log(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.04e7d60000000p+4f,zl))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1094-1105 ice_psi_cb2005
static inline void m_ice_psi_cb2005(thread Exact32 &psim1, thread Exact32 &psih1, Exact32 zl, Exact32 z0l) {
psim1 = rr_sub(rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zl,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),rr_mul(0x1.8666660000000p+2f,rr_log(rr_add(z0l,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = rr_sub(rr_mul(-0x1.6000000000000p+2f,rr_log(rr_add(zl,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),rr_mul(0x1.6000000000000p+2f,rr_log(rr_add(z0l,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1111-1167 ice_li_etal_2010
static inline void m_ice_li_etal_2010(thread Exact32 &zl, Exact32 rib, Exact32 zaz0, Exact32 z0zt) {
Exact32 alfa;
Exact32 beta;
Exact32 zaz02;
Exact32 z0zt2;
Exact32 au11 = 0x1.70a3d80000000p-5f;
Exact32 bu11 = 0x1.89374c0000000p-9f;
Exact32 bu12 = 0x1.82a9940000000p-8f;
Exact32 bu21 = -0x1.5326180000000p-4f;
Exact32 bu22 = 0x1.c4dd300000000p-1f;
Exact32 bu31 = 0x1.6425ae0000000p-3f;
Exact32 bu32 = -0x1.d7b4a20000000p-1f;
Exact32 bu33 = -0x1.b0f27c0000000p-4f;
Exact32 aw11 = 0x1.25c91e0000000p-1f;
Exact32 aw12 = -0x1.c275260000000p-2f;
Exact32 aw21 = -0x1.39a9fc0000000p+2f;
Exact32 aw22 = 0x1.a400000000000p+5f;
Exact32 bw11 = -0x1.b98c7e0000000p-5f;
Exact32 bw12 = 0x1.8a3d700000000p+0f;
Exact32 bw21 = -0x1.56872c0000000p-1f;
Exact32 bw22 = -0x1.a418940000000p+1f;
Exact32 as11 = 0x1.817c1c0000000p-1f;
Exact32 as21 = 0x1.de147a0000000p+3f;
Exact32 bs11 = 0x1.4154ca0000000p-3f;
Exact32 bs21 = -0x1.3c84b60000000p-2f;
Exact32 bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = rr_log(zaz02);
beta = rr_log(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = rr_add(rr_mul(rr_mul(0x1.70a3d80000000p-5f,alfa),rr_mul(rib,rib)),rr_mul(rr_add(rr_add(rr_mul(rr_add(rr_mul(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),rr_mul(alfa,alfa)),rr_mul(rr_add(rr_mul(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),rr_add(rr_add(rr_mul(0x1.6425ae0000000p-3f,rr_mul(beta,beta)),rr_mul(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = rr_max(zl,-0x1.e000000000000p+3f);
zl = rr_min(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = rr_add(rr_mul(rr_add(rr_mul(rr_add(rr_mul(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),rr_add(rr_mul(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),rr_mul(rib,rib)),rr_mul(rr_add(rr_mul(rr_add(rr_mul(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),rr_add(rr_mul(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = rr_min(zl,0x1.0000000000000p+2f);
zl = rr_max(zl,0x0.0p+0f);
} else {
zl = rr_add(rr_add(rr_add(rr_mul(rr_add(rr_mul(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),rr_mul(0x1.4154ca0000000p-3f,alfa)),rr_mul(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = rr_min(zl,0x1.4000000000000p+4f);
zl = rr_max(zl,0x1.0000000000000p+0f);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1170-1223 ice_zolri
static inline Exact32 m_ice_zolri(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 x1;
Exact32 x2;
Exact32 fx1;
Exact32 fx2;
int n;
int nmax = 20;
if ((ri<0x0.0p+0f)) {
x1 = rr_sub(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = rr_add(zol1,0x1.47ae140000000p-6f);
}
n = 1;
fx1 = m_ice_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
fx2 = m_ice_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
while (((rr_abs(rr_sub(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((rr_abs(fx2)<rr_abs(fx1))) {
x1 = rr_sub(x1,rr_mul(rr_div(fx1,rr_sub(fx2,fx1)),rr_sub(x2,x1)));
fx1 = m_ice_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x1;
} else {
x2 = rr_sub(x2,rr_mul(rr_div(fx2,rr_sub(fx2,fx1)),rr_sub(x2,x1)));
fx2 = m_ice_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(rr_abs(rr_sub(x1,x2))>=0x1.47ae140000000p-7f))) {
m_ice_li_etal_2010(result,ri,rr_div(za,z0),rr_div(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1225-1265 ice_zolri2
static inline Exact32 m_ice_zolri2(thread Exact32 &zol2, Exact32 ri2, Exact32 za, Exact32 z0, Exact32 zt, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 zol20;
Exact32 zol3;
Exact32 psim1;
Exact32 psih1;
Exact32 psix2;
Exact32 psit2;
Exact32 zolt;
if ((rr_mul(zol2,ri2)<0x0.0p+0f)) zol2 = 0x0.0p+0f;
zol20 = rr_div(rr_mul(zol2,z0),za);
zol3 = rr_add(zol2,zol20);
zolt = rr_div(rr_mul(zol2,zt),za);
if ((ri2<0)) {
psit2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),zt)),rr_sub(m_ice_psih_unstable(zol3,psi_opt,psih_unstab),m_ice_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),z0)),rr_sub(m_ice_psim_unstable(zol3,psi_opt,psim_unstab),m_ice_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),zt)),rr_sub(m_ice_psih_stable(zol3,psi_opt,psih_stab),m_ice_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(rr_log(rr_div(rr_add(za,z0),z0)),rr_sub(m_ice_psim_stable(zol3,psi_opt,psim_stab),m_ice_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = rr_sub(rr_div(rr_mul(zol2,psit2),rr_mul(psix2,psix2)),ri2);
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1268-1342 ice_zolrib
static inline Exact32 m_ice_zolrib(Exact32 ri, Exact32 za, Exact32 z0, Exact32 zt, Exact32 logz0, Exact32 logzt, thread Exact32 &zol1, int psi_opt, device const float *psim_stab, device const float *psih_stab, device const float *psim_unstab, device const float *psih_unstab) {
Exact32 result;
Exact32 zol20;
Exact32 zol3;
Exact32 zolt;
Exact32 zolold;
int n;
int nmax = 20;
Exact32 psit2;
Exact32 psix2;
if ((rr_mul(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((rr_abs(rr_sub(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = rr_div(rr_mul(zolold,z0),za);
zol3 = rr_add(zolold,zol20);
zolt = rr_div(rr_mul(zolold,zt),za);
if ((ri<0)) {
psit2 = rr_max(rr_sub(logzt,rr_sub(m_ice_psih_unstable(zol3,psi_opt,psih_unstab),m_ice_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(logz0,rr_sub(m_ice_psim_unstable(zol3,psi_opt,psim_unstab),m_ice_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = rr_max(rr_sub(logzt,rr_sub(m_ice_psih_stable(zol3,psi_opt,psih_stab),m_ice_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = rr_max(rr_sub(logz0,rr_sub(m_ice_psim_stable(zol3,psi_opt,psim_stab),m_ice_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = rr_div(rr_mul(ri,rr_mul(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(rr_abs(rr_sub(zolold,result))>0x1.47ae140000000p-7f))) {
m_ice_li_etal_2010(result,ri,rr_div(za,z0),rr_div(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1347-1354 ice_psim_stable_full
static inline Exact32 m_ice_psim_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1357-1364 ice_psih_stable_full
static inline Exact32 m_ice_psih_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.5333340000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1367-1382 ice_psim_unstable_full
static inline Exact32 m_ice_psim_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 x;
Exact32 ym;
Exact32 psimc;
Exact32 psimk;
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,x)))),rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,rr_mul(x,x))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
ym = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psimk,rr_mul(rr_mul(zolf,zolf),psimc)),rr_add(1,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1385-1399 ice_psih_unstable_full
static inline Exact32 m_ice_psih_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 y;
Exact32 yh;
Exact32 psihc;
Exact32 psihk;
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psihk,rr_mul(rr_mul(zolf,zolf),psihc)),rr_add(0x1.0000000000000p+0f,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1406-1415 ice_psim_stable_full_gfs
static inline Exact32 m_ice_psim_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 aa;
aa = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,aa),rr_log(rr_add(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1419-1428 ice_psih_stable_full_gfs
static inline Exact32 m_ice_psih_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 bb;
bb = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,bb),rr_log(rr_add(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1432-1448 ice_psim_unstable_full_gfs
static inline Exact32 m_ice_psim_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0 = -0x1.fccccc0000000p+1f;
Exact32 a1 = 0x1.8a3d700000000p+3f;
Exact32 b1 = -0x1.f051ec0000000p+2f;
Exact32 b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fccccc0000000p+1f,rr_mul(0x1.8a3d700000000p+3f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.f051ec0000000p+2f,rr_mul(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_sub(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p+1f,rr_sqrt(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1452-1468 ice_psih_unstable_full_gfs
static inline Exact32 m_ice_psih_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0p = -0x1.fc39580000000p+2f;
Exact32 a1p = 0x1.8c00000000000p+4f;
Exact32 b1p = -0x1.168f5c0000000p+3f;
Exact32 b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fc39580000000p+2f,rr_mul(0x1.8c00000000000p+4f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.168f5c0000000p+3f,rr_mul(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_add(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1472-1489 ice_psim_stable
static inline Exact32 m_ice_psim_stable(Exact32 zolf, int psi_opt, device const float *psim_stab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul(zolf,0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psim_stab[nzol],rr_mul(rzol,rr_sub(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psim_stable_full(zolf);
} else {
result = m_ice_psim_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1492-1509 ice_psih_stable
static inline Exact32 m_ice_psih_stable(Exact32 zolf, int psi_opt, device const float *psih_stab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul(zolf,0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psih_stab[nzol],rr_mul(rzol,rr_sub(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psih_stable_full(zolf);
} else {
result = m_ice_psih_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1512-1529 ice_psim_unstable
static inline Exact32 m_ice_psim_unstable(Exact32 zolf, int psi_opt, device const float *psim_unstab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul((-zolf),0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psim_unstab[nzol],rr_mul(rzol,rr_sub(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psim_unstable_full(zolf);
} else {
result = m_ice_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1532-1549 ice_psih_unstable
static inline Exact32 m_ice_psih_unstable(Exact32 zolf, int psi_opt, device const float *psih_unstab) {
Exact32 result;
int nzol;
Exact32 rzol;
nzol = rr_int(rr_mul((-zolf),0x1.9000000000000p+6f));
rzol = rr_sub(rr_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = rr_add(psih_unstab[nzol],rr_mul(rzol,rr_sub(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psih_unstable_full(zolf);
} else {
result = m_ice_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87 drv_mynnsfc_init

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:91-101 drv_mynnsfclay_finalize
static inline void m_drv_mynnsfclay_finalize(thread int &errflg) {
errflg = 0;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740 drv_mynnsfc_driver
static inline void m_drv_mynnsfc_driver(Exact32 u3d2, Exact32 v3d2, Exact32 dz8w2, Exact32 u3d, Exact32 v3d, Exact32 t3d, Exact32 qv3d, Exact32 p3d, Exact32 dz8w, Exact32 th3d, Exact32 rho3d, Exact32 sigmaf, int vegtype, Exact32 shdmax, int ivegsrc, Exact32 z0pert, Exact32 ztpert, thread bool &redrag, thread int &sfc_z0_type, Exact32 psfcpa, thread Exact32 &chs, thread Exact32 &chs2, thread Exact32 &cqs, thread Exact32 &cqs2, thread Exact32 &cpm, thread Exact32 &znt, thread Exact32 &ust, thread Exact32 &ustm, thread Exact32 &pblh, Exact32 mavail, thread Exact32 &zol, thread Exact32 &mol, thread Exact32 &rmol, thread Exact32 &psim, thread Exact32 &psih, Exact32 xland, thread Exact32 &qgh, thread Exact32 &hfx, thread Exact32 &qfx, thread Exact32 &lh, Exact32 tsk, thread Exact32 &flhc, thread Exact32 &flqc, thread Exact32 &qsfc, thread Exact32 &u10, thread Exact32 &v10, thread Exact32 &th2, thread Exact32 &t2, Exact32 lakemask, thread Exact32 &q2, Exact32 snowh, thread Exact32 &gz1oz0, thread Exact32 &wspd, thread Exact32 &br, Exact32 dx, thread Exact32 &ch, thread Exact32 &ck, thread Exact32 &cka, thread Exact32 &cd, thread Exact32 &cda, thread Exact32 &stress, thread Exact32 &hflx, thread Exact32 &qflx, thread Exact32 &cm, thread Exact32 &fm, thread Exact32 &fh, thread Exact32 &fm10, thread Exact32 &fh2, Exact32 tsurf, Exact32 water_depth, Exact32 xice, Exact32 xice_threshold, int spp_pbl, Exact32 pattern_spp_pbl, int sf_mynn_sfcflux_water, int sf_mynn_sfcflux_land, int shalwater_z0, int isfflx, bool restart, bool cycling, int initflag, thread bool &flag_iter, int flagc_lsm, int itimestep, int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, thread int &errflg, bool has_restart, bool has_cycling, bool has_initflag, bool has_sf_mynn_sfcflux_water, bool has_sf_mynn_sfcflux_land, bool has_shalwater_z0, bool has_flagc_lsm, bool has_spp_pbl, bool has_ivegsrc, bool has_sfc_z0_type, bool has_redrag, bool has_flag_iter, bool has_vegtype, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_pattern_spp_pbl, bool has_tsurf, bool has_lakemask, bool has_water_depth, bool has_cm, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm, bool has_fm, bool has_fh, bool has_fm10, bool has_fh2, bool has_hflx, bool has_qflx, bool has_stress, device const float *tables) {
    device const float *psim_stab=tables, *psim_unstab=tables+1001;
    device const float *psih_stab=tables+2002, *psih_unstab=tables+3003;

int flag_lsm;
int lsm_ruc = 3;
Exact32 snow_thresh = 0x1.99999a0000000p-5f;
int i;
int j;
int k;
int vegtype_1;
int iter;
int loc_z0_type;
int ncalls;
bool loc_redrag;
bool loc_iter;
bool loc_cycle;
Exact32 wstar;
Exact32 qstar;
Exact32 mavail_1;
Exact32 pblh_1;
Exact32 xland_1;
Exact32 tsk_1;
Exact32 psfcpa_1;
Exact32 snowh_1;
Exact32 dx_1;
Exact32 lakemask_1;
Exact32 wat_depth_1;
Exact32 xice_1;
Exact32 u_1;
Exact32 v_1;
Exact32 u_2;
Exact32 v_2;
Exact32 qv_1;
Exact32 p_1;
Exact32 t_1;
Exact32 rho_1;
Exact32 dz8w_1;
Exact32 dz8w_2;
Exact32 rstoch_1;
Exact32 qvmr;
Exact32 hfx_1;
Exact32 qfx_1;
Exact32 lh_1;
Exact32 mol_1;
Exact32 rmol_1;
Exact32 qsfc_1;
Exact32 qgh_1;
Exact32 znt_1;
Exact32 zol_1;
Exact32 ust_1;
Exact32 cpm_1;
Exact32 chs2_1;
Exact32 cqs_1;
Exact32 cqs2_1;
Exact32 chs_1;
Exact32 ch_1;
Exact32 flhc_1;
Exact32 flqc_1;
Exact32 gz1oz0_1;
Exact32 wspd_1;
Exact32 br_1;
Exact32 psim_1;
Exact32 psih_1;
Exact32 u10_1;
Exact32 v10_1;
Exact32 th2_1;
Exact32 t2_1;
Exact32 q2_1;
Exact32 cd_1;
Exact32 cda_1;
Exact32 ck_1;
Exact32 cka_1;
Exact32 ustm_1;
Exact32 wstar_1;
Exact32 qstar_1;
Exact32 tsurf_1;
Exact32 cm_1;
Exact32 stress_1;
Exact32 fm_1;
Exact32 fh_1;
Exact32 fm10_1;
Exact32 fh2_1;
Exact32 hflx_1;
Exact32 qflx_1;
Exact32 sigmaf_1;
Exact32 shdmax_1;
Exact32 z0pert_1;
Exact32 ztpert_1;
int debug_driver = 0;
errflg = 0;
iter = 1;
if ((flagc_lsm==1)) {
flag_lsm = 1;
} else if ((flagc_lsm==2)) {
flag_lsm = 2;
} else if ((flagc_lsm==3)) {
flag_lsm = 3;
} else {
flag_lsm = 4;
}
if (has_cycling) {
loc_cycle = cycling;
} else {
loc_cycle = false;
}
if (false) {
}
if (((itimestep==1)&&((!restart)||(!loc_cycle)))) {
{ i=its; j=jts; // independent MPAS column loop
{ i=its; j=jts; // independent MPAS column loop
if (((ust<=0x1.a36e2e0000000p-14f)||(ust>0x1.0000000000000p+2f))) {
u_1 = u3d;
v_1 = v3d;
ust = rr_max(rr_mul(0x1.47ae140000000p-5f,rr_sqrt(rr_add(rr_mul(u_1,u_1),rr_mul(v_1,v_1)))),0x1.0624de0000000p-10f);
}
pblh = rr_max(0x1.4000000000000p+3f,rr_mul(0x1.5e00000000000p+9f,ust));
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
mol = 0x0.0p+0f;
}
}
}
if ((itimestep==1)) {
qstar = 0x0.0p+0f;
}
{ i=its; j=jts; // independent MPAS column loop
{ i=its; j=jts; // independent MPAS column loop
qvmr = qv3d;
dz8w_1 = dz8w;
qv_1 = rr_max(0x1.5798ee0000000p-27f,rr_div(qvmr,rr_add(0x1.0000000000000p+0f,qvmr)));
p_1 = p3d;
t_1 = t3d;
rho_1 = rho3d;
dz8w_2 = dz8w2;
u_2 = u3d2;
v_2 = v3d2;
u_1 = u3d;
v_1 = v3d;
ust_1 = ust;
stress_1 = rr_mul(ust_1,ust_1);
mol_1 = mol;
qsfc_1 = rr_div(qsfc,rr_add(0x1.0000000000000p+0f,qvmr));
qgh_1 = rr_div(qgh,rr_add(0x1.0000000000000p+0f,qvmr));
qstar_1 = qstar;
if ((spp_pbl==1)) {
rstoch_1 = pattern_spp_pbl;
} else {
rstoch_1 = 0x0.0p+0f;
}
mavail_1 = mavail;
pblh_1 = pblh;
xland_1 = xland;
xice_1 = xice;
tsk_1 = tsk;
psfcpa_1 = psfcpa;
snowh_1 = snowh;
dx_1 = dx;
if (has_lakemask) {
lakemask_1 = lakemask;
} else {
lakemask_1 = 0x0.0p+0f;
}
if ((has_water_depth&&has_shalwater_z0)) {
wat_depth_1 = water_depth;
} else {
wat_depth_1 = 0x1.f400000000000p+9f;
}
hfx_1 = hfx;
qfx_1 = rr_div(qfx,rr_add(0x1.0000000000000p+0f,qvmr));
hflx_1 = rr_div(hfx_1,rr_mul(0x1.f640000000000p+9f,rho_1));
qflx_1 = rr_div(qfx_1,rho_1);
lh_1 = lh;
rmol_1 = rmol;
znt_1 = znt;
zol_1 = zol;
cpm_1 = cpm;
chs2_1 = chs2;
cqs2_1 = cqs2;
cqs_1 = cqs;
chs_1 = chs;
ch_1 = ch;
flhc_1 = flhc;
flqc_1 = flqc;
gz1oz0_1 = gz1oz0;
wspd_1 = wspd;
br_1 = br;
psim_1 = psim;
psih_1 = psih;
u10_1 = 0x0.0p+0f;
v10_1 = 0x0.0p+0f;
th2_1 = 0x0.0p+0f;
t2_1 = 0x0.0p+0f;
q2_1 = 0x0.0p+0f;
wstar_1 = 0x0.0p+0f;
loc_redrag = false;
if (has_redrag) loc_redrag = redrag;
loc_iter = true;
if (has_flag_iter) loc_iter = flag_iter;
loc_z0_type = 0;
if (has_sfc_z0_type) loc_z0_type = sfc_z0_type;
if (has_tsurf) {
tsurf_1 = tsurf;
} else {
tsurf_1 = tsk_1;
}
if (has_sigmaf) {
sigmaf_1 = sigmaf;
} else {
sigmaf_1 = 0x0.0p+0f;
}
if (has_shdmax) {
shdmax_1 = shdmax;
} else {
shdmax_1 = 0x0.0p+0f;
}
if (has_z0pert) {
z0pert_1 = z0pert;
} else {
z0pert_1 = 0x0.0p+0f;
}
if (has_ztpert) {
ztpert_1 = ztpert;
} else {
ztpert_1 = 0x0.0p+0f;
}
if (has_ztpert) {
vegtype_1 = vegtype;
} else {
vegtype_1 = 1;
}
if ((((has_ck&&has_cka)&&has_cd)&&has_cda)) {
ck_1 = 0x0.0p+0f;
cka_1 = 0x0.0p+0f;
cd_1 = 0x0.0p+0f;
cda_1 = 0x0.0p+0f;
}
if (has_ustm) {
ustm_1 = ustm;
} else {
ustm_1 = 0x0.0p+0f;
}
if ((((has_fm&&has_fh)&&has_fm10)&&has_fh2)) {
fm_1 = 0x0.0p+0f;
fh_1 = 0x0.0p+0f;
fm10_1 = 0x0.0p+0f;
fh2_1 = 0x0.0p+0f;
}
if (false) {
ncalls = 0;
}
if (((rr_sub(xland_1,0x1.8000000000000p+0f)<0x0.0p+0f)&&(snowh_1<0x1.99999a0000000p-5f))) {
if (false) {
ncalls = (ncalls+1);
}
m_land_mynnsfc_land(loc_iter,itimestep,i,j,dx_1,xland_1,u_1,v_1,t_1,qv_1,p_1,dz8w_1,rho_1,u_2,v_2,dz8w_2,sigmaf_1,vegtype_1,shdmax_1,ivegsrc,z0pert_1,ztpert_1,loc_redrag,loc_z0_type,pblh_1,znt_1,psfcpa_1,mavail_1,tsk_1,tsurf_1,snowh_1,qgh_1,chs_1,chs2_1,cqs2_1,cqs_1,ust_1,ustm_1,stress_1,qsfc_1,rmol_1,zol_1,mol_1,psim_1,psih_1,hfx_1,qfx_1,u10_1,v10_1,th2_1,t2_1,q2_1,flhc_1,flqc_1,lh_1,gz1oz0_1,wspd_1,br_1,cpm_1,ch_1,cm_1,rstoch_1,wstar_1,qstar_1,ck_1,cka_1,cd_1,cda_1,fm_1,fh_1,fm10_1,fh2_1,spp_pbl,sf_mynn_sfcflux_land,isfflx,restart,cycling,0,true,true,iter,flag_lsm,3,psim_stab,psim_unstab,psih_stab,psih_unstab,errflg,has_sf_mynn_sfcflux_land,true,true,true,true,true,true,true,true);
}
if ((rr_sub(xland_1,0x1.8000000000000p+0f)>=0x0.0p+0f)) {
if (false) {
ncalls = (ncalls+1);
}
m_water_mynnsfc_water(loc_iter,itimestep,i,j,dx_1,xland_1,lakemask_1,wat_depth_1,u_1,v_1,t_1,qv_1,p_1,dz8w_1,rho_1,u_2,v_2,dz8w_2,sigmaf_1,vegtype_1,shdmax_1,ivegsrc,z0pert_1,ztpert_1,loc_redrag,loc_z0_type,pblh_1,znt_1,psfcpa_1,mavail_1,tsk_1,tsurf_1,snowh_1,chs_1,chs2_1,cqs2_1,cqs_1,ust_1,ustm_1,stress_1,rmol_1,zol_1,mol_1,psim_1,psih_1,hfx_1,qfx_1,u10_1,v10_1,th2_1,qsfc_1,t2_1,q2_1,flhc_1,flqc_1,lh_1,gz1oz0_1,wspd_1,br_1,cpm_1,ch_1,cm_1,rstoch_1,wstar_1,qstar_1,qgh_1,ck_1,cka_1,cd_1,cda_1,fm_1,fh_1,fm10_1,fh2_1,spp_pbl,sf_mynn_sfcflux_water,isfflx,restart,cycling,0,true,true,shalwater_z0,iter,flag_lsm,3,psim_stab,psim_unstab,psih_stab,psih_unstab,errflg,has_sf_mynn_sfcflux_water,has_shalwater_z0,true,true,true,true,true,true,true,true);
}
if ((((rr_sub(xland_1,0x1.8000000000000p+0f)<0x0.0p+0f)&&(snowh_1>=0x1.99999a0000000p-5f))||((xice_1>=xice_threshold)&&(xice_1<0x1.0000000000000p+0f)))) {
if (false) {
ncalls = (ncalls+1);
}
m_ice_mynnsfc_ice(loc_iter,itimestep,i,j,dx_1,xland_1,u_1,v_1,t_1,qv_1,p_1,dz8w_1,rho_1,u_2,v_2,dz8w_2,pblh_1,znt_1,psfcpa_1,mavail_1,tsk_1,tsurf_1,snowh_1,qgh_1,chs_1,chs2_1,cqs2_1,cqs_1,ust_1,ustm_1,stress_1,rmol_1,zol_1,mol_1,psim_1,psih_1,hfx_1,qfx_1,u10_1,v10_1,th2_1,qsfc_1,t2_1,q2_1,flhc_1,flqc_1,lh_1,gz1oz0_1,wspd_1,br_1,cpm_1,ch_1,cm_1,rstoch_1,wstar_1,qstar_1,ck_1,cka_1,cd_1,cda_1,fm_1,fh_1,fm10_1,fh2_1,spp_pbl,isfflx,restart,cycling,0,true,true,iter,flag_lsm,3,psim_stab,psim_unstab,psih_stab,psih_unstab,errflg,true,true,true,true);
}
hfx = hfx_1;
qfx = rr_div(qfx_1,rr_sub(0x1.0000000000000p+0f,qv_1));
lh = rr_div(lh_1,rr_sub(0x1.0000000000000p+0f,qv_1));
mol = mol_1;
rmol = rmol_1;
qsfc = rr_div(qsfc_1,rr_sub(0x1.0000000000000p+0f,qv_1));
qgh = rr_div(qgh_1,rr_sub(0x1.0000000000000p+0f,qv_1));
znt = znt_1;
zol = zol_1;
ust = ust_1;
cpm = cpm_1;
chs2 = chs2_1;
cqs2 = cqs2_1;
cqs = cqs_1;
chs = chs_1;
ch = ch_1;
flhc = flhc_1;
flqc = flqc_1;
gz1oz0 = gz1oz0_1;
wspd = wspd_1;
br = br_1;
psim = psim_1;
psih = psih_1;
u10 = u10_1;
v10 = v10_1;
th2 = th2_1;
t2 = t2_1;
q2 = rr_div(q2_1,rr_sub(0x1.0000000000000p+0f,qv_1));
wstar = wstar_1;
qstar = qstar_1;
if (has_hflx) hflx = rr_div(hfx_1,rr_mul(0x1.f640000000000p+9f,rho_1));
if (has_qflx) qflx = rr_div(qfx_1,rho_1);
if ((((has_ck&&has_cka)&&has_cd)&&has_cda)) {
ck = ck_1;
cka = cka_1;
cd = cd_1;
cda = cda_1;
}
if ((((has_fm&&has_fh)&&has_fm10)&&has_fh2)) {
fm = fm_1;
fh = fh_1;
fm10 = fm10_1;
fh2 = fh2_1;
}
if (has_ustm) ustm = ustm_1;
if (has_stress) stress = stress_1;
if (has_cm) cm = cm_1;
if (false) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:745-788 drv_psi_init

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:794-801 drv_psim_stable_full
static inline Exact32 m_drv_psim_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.8666660000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:804-811 drv_psih_stable_full
static inline Exact32 m_drv_psih_stable_full(Exact32 zolf) {
Exact32 result;
result = rr_mul(-0x1.5333340000000p+2f,rr_log(rr_add(zolf,rr_pow(rr_add(0x1.0000000000000p+0f,rr_pow(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:814-829 drv_psim_unstable_full
static inline Exact32 m_drv_psim_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 x;
Exact32 ym;
Exact32 psimc;
Exact32 psimk;
x = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = rr_add(rr_sub(rr_add(rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,x)))),rr_log(rr_mul(0x1.0000000000000p-1f,rr_add(0x1.0000000000000p+0f,rr_mul(x,x))))),rr_mul(0x1.0000000000000p+1f,rr_atan(x))),0x1.921fb60000000p+0f);
ym = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psimk,rr_mul(rr_mul(zolf,zolf),psimc)),rr_add(1,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:832-846 drv_psih_unstable_full
static inline Exact32 m_drv_psih_unstable_full(Exact32 zolf) {
Exact32 result;
Exact32 y;
Exact32 yh;
Exact32 psihc;
Exact32 psihk;
y = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = rr_mul(0x1.0000000000000p+1f,rr_log(rr_mul(rr_add(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = rr_pow(rr_sub(0x1.0000000000000p+0f,rr_mul(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = rr_add(rr_sub(rr_mul(0x1.8000000000000p+0f,rr_log(rr_mul(rr_add(rr_add(rr_mul(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),rr_mul(0x1.bb67ae0000000p+0f,rr_atan(rr_div(rr_add(rr_mul(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = rr_div(rr_add(psihk,rr_mul(rr_mul(zolf,zolf),psihc)),rr_add(0x1.0000000000000p+0f,rr_mul(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:853-862 drv_psim_stable_full_gfs
static inline Exact32 m_drv_psim_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 aa;
aa = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,aa),rr_log(rr_add(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:866-875 drv_psih_stable_full_gfs
static inline Exact32 m_drv_psih_stable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 alpha4 = 0x1.4000000000000p+4f;
Exact32 bb;
bb = rr_sqrt(rr_add(0x1.0000000000000p+0f,rr_mul(0x1.4000000000000p+4f,zolf)));
result = rr_add(rr_mul(-0x1.0000000000000p+0f,bb),rr_log(rr_add(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:879-895 drv_psim_unstable_full_gfs
static inline Exact32 m_drv_psim_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0 = -0x1.fccccc0000000p+1f;
Exact32 a1 = 0x1.8a3d700000000p+3f;
Exact32 b1 = -0x1.f051ec0000000p+2f;
Exact32 b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fccccc0000000p+1f,rr_mul(0x1.8a3d700000000p+3f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.f051ec0000000p+2f,rr_mul(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_sub(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p+1f,rr_sqrt(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:899-915 drv_psih_unstable_full_gfs
static inline Exact32 m_drv_psih_unstable_full_gfs(Exact32 zolf) {
Exact32 result;
Exact32 hl1;
Exact32 tem1;
Exact32 a0p = -0x1.fc39580000000p+2f;
Exact32 a1p = 0x1.8c00000000000p+4f;
Exact32 b1p = -0x1.168f5c0000000p+3f;
Exact32 b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = rr_div(rr_mul(rr_add(-0x1.fc39580000000p+2f,rr_mul(0x1.8c00000000000p+4f,hl1)),hl1),rr_add(0x1.0000000000000p+0f,rr_mul(rr_add(-0x1.168f5c0000000p+3f,rr_mul(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = rr_div(0x1.0000000000000p+0f,rr_sqrt(hl1));
result = rr_add(rr_add(rr_log(hl1),rr_mul(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740

kernel void hexrrfs_mynnsfc_tables(constant HexrrfsMynnsfcTablesResources &r [[buffer(0)]],constant HexrrfsMynnsfcTablesParams &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
    if(c>=uint(p.ntable)) return;
    Exact32 z=rr_mul(Exact32(c),0x1.47ae140000000p-7f);
    r.tables[c]=as_type<float>(m_drv_psim_stable_full(z).bits); r.tables[2002+c]=as_type<float>(m_drv_psih_stable_full(z).bits);
    z=rr_mul(-Exact32(c),0x1.47ae140000000p-7f);
    r.tables[1001+c]=as_type<float>(m_drv_psim_unstable_full(z).bits); r.tables[3003+c]=as_type<float>(m_drv_psih_unstable_full(z).bits);
}
kernel void hexrrfs_mynnsfc_columns(constant HexrrfsMynnsfcColumnsResources &r [[buffer(0)]],constant HexrrfsMynnsfcColumnsParams &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
    if(c>=uint(p.ncol)) return;
    int error=0; ulong present=ulong(uint(p.present_lo))|(ulong(uint(p.present_hi))<<32);
    int o_ivegsrc=r.options[0]; bool o_redrag=r.options[1]; int o_sfc_z0_type=r.options[2];
    int o_spp_pbl=r.options[3],o_sf_mynn_sfcflux_water=r.options[4],o_sf_mynn_sfcflux_land=r.options[5];
    int o_shalwater_z0=r.options[6],o_isfflx=r.options[7]; bool o_restart=r.options[8],o_cycling=r.options[9];
    int o_initflag=p.init_override<0?r.options[10]:p.init_override; bool o_flag_iter=r.options[11];
    int o_flagc_lsm=r.options[12],o_itimestep=p.step_override<0?r.options[13]:p.step_override;
    Exact32 v0=r.fields[0*p.ncol+c];
    Exact32 v1=r.fields[1*p.ncol+c];
    Exact32 v2=r.fields[2*p.ncol+c];
    Exact32 v3=r.fields[3*p.ncol+c];
    Exact32 v4=r.fields[4*p.ncol+c];
    Exact32 v5=r.fields[5*p.ncol+c];
    Exact32 v6=r.fields[6*p.ncol+c];
    Exact32 v7=r.fields[7*p.ncol+c];
    Exact32 v8=r.fields[8*p.ncol+c];
    Exact32 v9=r.fields[9*p.ncol+c];
    Exact32 v10=r.fields[10*p.ncol+c];
    Exact32 v11=r.fields[11*p.ncol+c];
    Exact32 v12=r.fields[12*p.ncol+c];
    Exact32 v13=r.fields[13*p.ncol+c];
    Exact32 v14=r.fields[14*p.ncol+c];
    Exact32 v15=r.fields[15*p.ncol+c];
    Exact32 v16=r.fields[16*p.ncol+c];
    Exact32 v17=r.fields[17*p.ncol+c];
    Exact32 v18=r.fields[18*p.ncol+c];
    Exact32 v19=r.fields[19*p.ncol+c];
    Exact32 v20=r.fields[20*p.ncol+c];
    Exact32 v21=r.fields[21*p.ncol+c];
    Exact32 v22=r.fields[22*p.ncol+c];
    Exact32 v23=r.fields[23*p.ncol+c];
    Exact32 v24=r.fields[24*p.ncol+c];
    Exact32 v25=r.fields[25*p.ncol+c];
    Exact32 v26=r.fields[26*p.ncol+c];
    Exact32 v27=r.fields[27*p.ncol+c];
    Exact32 v28=r.fields[28*p.ncol+c];
    Exact32 v29=r.fields[29*p.ncol+c];
    Exact32 v30=r.fields[30*p.ncol+c];
    Exact32 v31=r.fields[31*p.ncol+c];
    Exact32 v32=r.fields[32*p.ncol+c];
    Exact32 v33=r.fields[33*p.ncol+c];
    Exact32 v34=r.fields[34*p.ncol+c];
    Exact32 v35=r.fields[35*p.ncol+c];
    Exact32 v36=r.fields[36*p.ncol+c];
    Exact32 v37=r.fields[37*p.ncol+c];
    Exact32 v38=r.fields[38*p.ncol+c];
    Exact32 v39=r.fields[39*p.ncol+c];
    Exact32 v40=r.fields[40*p.ncol+c];
    Exact32 v41=r.fields[41*p.ncol+c];
    Exact32 v42=r.fields[42*p.ncol+c];
    Exact32 v43=r.fields[43*p.ncol+c];
    Exact32 v44=r.fields[44*p.ncol+c];
    Exact32 v45=r.fields[45*p.ncol+c];
    Exact32 v46=r.fields[46*p.ncol+c];
    Exact32 v47=r.fields[47*p.ncol+c];
    Exact32 v48=r.fields[48*p.ncol+c];
    Exact32 v49=r.fields[49*p.ncol+c];
    Exact32 v50=r.fields[50*p.ncol+c];
    Exact32 v51=r.fields[51*p.ncol+c];
    Exact32 v52=r.fields[52*p.ncol+c];
    Exact32 v53=r.fields[53*p.ncol+c];
    Exact32 v54=r.fields[54*p.ncol+c];
    Exact32 v55=r.fields[55*p.ncol+c];
    Exact32 v56=r.fields[56*p.ncol+c];
    Exact32 v57=r.fields[57*p.ncol+c];
    Exact32 v58=r.fields[58*p.ncol+c];
    Exact32 v59=r.fields[59*p.ncol+c];
    Exact32 v60=r.fields[60*p.ncol+c];
    Exact32 v61=r.fields[61*p.ncol+c];
    Exact32 v62=r.fields[62*p.ncol+c];
    Exact32 v63=r.fields[63*p.ncol+c];
    Exact32 v64=r.fields[64*p.ncol+c];
    Exact32 v65=r.fields[65*p.ncol+c];
    Exact32 v66=r.fields[66*p.ncol+c];
    Exact32 v67=r.fields[67*p.ncol+c];
    m_drv_mynnsfc_driver(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,r.vegtype[c],v12,o_ivegsrc,v13,v14,o_redrag,o_sfc_z0_type,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,v51,v52,v53,v54,v55,v56,v57,v58,v59,v60,v61,v62,v63,v64,v65,v66,p.xice_threshold,o_spp_pbl,v67,o_sf_mynn_sfcflux_water,o_sf_mynn_sfcflux_land,o_shalwater_z0,o_isfflx,o_restart,o_cycling,o_initflag,o_flag_iter,o_flagc_lsm,o_itimestep,1,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,2,error,(present & (1ull<<0))!=0,(present & (1ull<<1))!=0,(present & (1ull<<2))!=0,(present & (1ull<<3))!=0,(present & (1ull<<4))!=0,(present & (1ull<<5))!=0,(present & (1ull<<6))!=0,(present & (1ull<<7))!=0,(present & (1ull<<8))!=0,(present & (1ull<<9))!=0,(present & (1ull<<10))!=0,(present & (1ull<<11))!=0,(present & (1ull<<12))!=0,(present & (1ull<<13))!=0,(present & (1ull<<14))!=0,(present & (1ull<<15))!=0,(present & (1ull<<16))!=0,(present & (1ull<<17))!=0,(present & (1ull<<18))!=0,(present & (1ull<<19))!=0,(present & (1ull<<20))!=0,(present & (1ull<<21))!=0,(present & (1ull<<22))!=0,(present & (1ull<<23))!=0,(present & (1ull<<24))!=0,(present & (1ull<<25))!=0,(present & (1ull<<26))!=0,(present & (1ull<<27))!=0,(present & (1ull<<28))!=0,(present & (1ull<<29))!=0,(present & (1ull<<30))!=0,(present & (1ull<<31))!=0,(present & (1ull<<32))!=0,(present & (1ull<<33))!=0,r.tables);
    r.fields[0*p.ncol+c]=as_type<float>(v0.bits);
    r.fields[1*p.ncol+c]=as_type<float>(v1.bits);
    r.fields[2*p.ncol+c]=as_type<float>(v2.bits);
    r.fields[3*p.ncol+c]=as_type<float>(v3.bits);
    r.fields[4*p.ncol+c]=as_type<float>(v4.bits);
    r.fields[5*p.ncol+c]=as_type<float>(v5.bits);
    r.fields[6*p.ncol+c]=as_type<float>(v6.bits);
    r.fields[7*p.ncol+c]=as_type<float>(v7.bits);
    r.fields[8*p.ncol+c]=as_type<float>(v8.bits);
    r.fields[9*p.ncol+c]=as_type<float>(v9.bits);
    r.fields[10*p.ncol+c]=as_type<float>(v10.bits);
    r.fields[11*p.ncol+c]=as_type<float>(v11.bits);
    r.fields[12*p.ncol+c]=as_type<float>(v12.bits);
    r.fields[13*p.ncol+c]=as_type<float>(v13.bits);
    r.fields[14*p.ncol+c]=as_type<float>(v14.bits);
    r.fields[15*p.ncol+c]=as_type<float>(v15.bits);
    r.fields[16*p.ncol+c]=as_type<float>(v16.bits);
    r.fields[17*p.ncol+c]=as_type<float>(v17.bits);
    r.fields[18*p.ncol+c]=as_type<float>(v18.bits);
    r.fields[19*p.ncol+c]=as_type<float>(v19.bits);
    r.fields[20*p.ncol+c]=as_type<float>(v20.bits);
    r.fields[21*p.ncol+c]=as_type<float>(v21.bits);
    r.fields[22*p.ncol+c]=as_type<float>(v22.bits);
    r.fields[23*p.ncol+c]=as_type<float>(v23.bits);
    r.fields[24*p.ncol+c]=as_type<float>(v24.bits);
    r.fields[25*p.ncol+c]=as_type<float>(v25.bits);
    r.fields[26*p.ncol+c]=as_type<float>(v26.bits);
    r.fields[27*p.ncol+c]=as_type<float>(v27.bits);
    r.fields[28*p.ncol+c]=as_type<float>(v28.bits);
    r.fields[29*p.ncol+c]=as_type<float>(v29.bits);
    r.fields[30*p.ncol+c]=as_type<float>(v30.bits);
    r.fields[31*p.ncol+c]=as_type<float>(v31.bits);
    r.fields[32*p.ncol+c]=as_type<float>(v32.bits);
    r.fields[33*p.ncol+c]=as_type<float>(v33.bits);
    r.fields[34*p.ncol+c]=as_type<float>(v34.bits);
    r.fields[35*p.ncol+c]=as_type<float>(v35.bits);
    r.fields[36*p.ncol+c]=as_type<float>(v36.bits);
    r.fields[37*p.ncol+c]=as_type<float>(v37.bits);
    r.fields[38*p.ncol+c]=as_type<float>(v38.bits);
    r.fields[39*p.ncol+c]=as_type<float>(v39.bits);
    r.fields[40*p.ncol+c]=as_type<float>(v40.bits);
    r.fields[41*p.ncol+c]=as_type<float>(v41.bits);
    r.fields[42*p.ncol+c]=as_type<float>(v42.bits);
    r.fields[43*p.ncol+c]=as_type<float>(v43.bits);
    r.fields[44*p.ncol+c]=as_type<float>(v44.bits);
    r.fields[45*p.ncol+c]=as_type<float>(v45.bits);
    r.fields[46*p.ncol+c]=as_type<float>(v46.bits);
    r.fields[47*p.ncol+c]=as_type<float>(v47.bits);
    r.fields[48*p.ncol+c]=as_type<float>(v48.bits);
    r.fields[49*p.ncol+c]=as_type<float>(v49.bits);
    r.fields[50*p.ncol+c]=as_type<float>(v50.bits);
    r.fields[51*p.ncol+c]=as_type<float>(v51.bits);
    r.fields[52*p.ncol+c]=as_type<float>(v52.bits);
    r.fields[53*p.ncol+c]=as_type<float>(v53.bits);
    r.fields[54*p.ncol+c]=as_type<float>(v54.bits);
    r.fields[55*p.ncol+c]=as_type<float>(v55.bits);
    r.fields[56*p.ncol+c]=as_type<float>(v56.bits);
    r.fields[57*p.ncol+c]=as_type<float>(v57.bits);
    r.fields[58*p.ncol+c]=as_type<float>(v58.bits);
    r.fields[59*p.ncol+c]=as_type<float>(v59.bits);
    r.fields[60*p.ncol+c]=as_type<float>(v60.bits);
    r.fields[61*p.ncol+c]=as_type<float>(v61.bits);
    r.fields[62*p.ncol+c]=as_type<float>(v62.bits);
    r.fields[63*p.ncol+c]=as_type<float>(v63.bits);
    r.fields[64*p.ncol+c]=as_type<float>(v64.bits);
    r.fields[65*p.ncol+c]=as_type<float>(v65.bits);
    r.fields[66*p.ncol+c]=as_type<float>(v66.bits);
    r.fields[67*p.ncol+c]=as_type<float>(v67.bits);
    r.errors[c]=error;
}
