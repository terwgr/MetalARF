#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_libm64.metalh"
#include "hexv842_mpas_noah_land_f32_bindings.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define MU(a,b) mpas_mul(a,b)
#define DV(a,b) cu_div(a,b)
static __attribute__((noinline)) float r_log(float x){return gfk_log(x);}
static __attribute__((noinline)) float r_exp(float x){return gfk_exp(x);}
static __attribute__((noinline)) float r_pow(float x,float y){return gfk_pow(x,y);}
static __attribute__((noinline)) float r_log10(float x){return as_type<float>(soft64_to_f32_bits(soft64_div(soft64_plm_log(soft64_from_f32_bits(as_type<uint>(x))),0x40026bb1bbb55516ul)));}

struct NoahTables {
float RD;
float SIGMA;
float CPH2O;
float CPICE;
float LSUBF;
float EMISSI_S;
int LUCATS;
int BARE;
int NATURAL;
int LCZ_1;
int LCZ_2;
int LCZ_3;
int LCZ_4;
int LCZ_5;
int LCZ_6;
int LCZ_7;
int LCZ_8;
int LCZ_9;
int LCZ_10;
int LCZ_11;
int NLUS;
char LUTYPE[256];
int NROTBL[50];
float SNUPTBL[50];
float RSTBL[50];
float RGLTBL[50];
float HSTBL[50];
float SHDTBL[50];
float MAXALB[50];
float EMISSMINTBL[50];
float EMISSMAXTBL[50];
float LAIMINTBL[50];
float LAIMAXTBL[50];
float Z0MINTBL[50];
float Z0MAXTBL[50];
float ALBEDOMINTBL[50];
float ALBEDOMAXTBL[50];
float ZTOPVTBL[50];
float ZBOTVTBL[50];
float TOPT_DATA;
float CMCMAX_DATA;
float CFACTR_DATA;
float RSMAX_DATA;
int SLCATS;
int NSLTYPE;
char SLTYPE[256];
float BB[30];
float DRYSMC[30];
float F11[30];
float MAXSMC[30];
float REFSMC[30];
float SATPSI[30];
float SATDK[30];
float SATDW[30];
float WLTSMC[30];
float QTZ[30];
int SLPCATS;
int NSLOPE;
float SLOPE_DATA[30];
float SBETA_DATA;
float FXEXP_DATA;
float CSOIL_DATA;
float SALP_DATA;
float REFDK_DATA;
float REFKDT_DATA;
float FRZK_DATA;
float ZBOT_DATA;
float SMLOW_DATA;
float SMHIGH_DATA;
float CZIL_DATA;
float LVCOEF_DATA;
char ERR_MESSAGE[256];
int ILOC;
int JLOC;
float CP;
float R_D;
float XLF;
float XLV;
float RHOWATER;
float STBOLT;
float KARMAN;
float GRAVITY;
int error;
};
template<class T> static __attribute__((noinline)) T noah_min(T a,T b){return a<b?a:b;}
template<class T> static __attribute__((noinline)) T noah_max(T a,T b){return a>b?a:b;}
template<class T> static __attribute__((noinline)) T noah_abs(T a){return a<0?-a:a;}


static inline float noah_powi_n(float a,int n){float r=1.0f;while(n){if(n&1)r=MU(r,a);n>>=1;if(n)a=MU(a,a);}return r;}
static __attribute__((noinline)) float noah_powi(float a,int n){if(n<0)return DV(1.0f,noah_powi_n(a,-n));return noah_powi_n(a,n);}
static __attribute__((noinline)) void noah_sflx(thread NoahTables & tables, thread int & IILOC, thread int & JJLOC, thread float & FFROZP, thread int & ISURBAN, thread float & DT, thread float & ZLVL, thread int & NSOIL, thread float * SLDPTH, thread bool & LOCAL, thread const char * LLANDUSE, thread const char * LSOIL, thread float & LWDN, thread float & SOLDN, thread float & SOLNET, thread float & SFCPRS, thread float & PRCP, thread float & SFCTMP, thread float & Q2, thread float & SFCSPD, thread float & COSZ, thread float & PRCPRAIN, thread float & SOLARDIRECT, thread float & TH2, thread float & Q2SAT, thread float & DQSDT2, thread int & VEGTYP, thread int & SOILTYP, thread int & SLOPETYP, thread float & SHDFAC, thread float & SHDMIN, thread float & SHDMAX, thread float & ALB, thread float & SNOALB, thread float & TBOT, thread float & Z0BRD, thread float & Z0, thread float & EMISSI, thread float & EMBRD, thread float & CMC, thread float & T1, thread float * STC, thread float * SMC, thread float * SH2O, thread float & SNOWH, thread float & SNEQV, thread float & ALBEDO, thread float & CH, thread float & CM, thread float & ETA, thread float & SHEAT, thread float & ETA_KINEMATIC, thread float & FDOWN, thread float & EC, thread float & EDIR, thread float * ET, thread float & ETT, thread float & ESNOW, thread float & DRIP, thread float & DEW, thread float & BETA, thread float & ETP, thread float & SSOIL, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & FLX4, thread float & FVB, thread float & FBUR, thread float & FGSN, thread bool & UA_PHYS, thread float & SNOMLT, thread float & SNCOVR, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & RC, thread float & PC, thread float & RSMIN, thread float & XLAI, thread float & RCS, thread float & RCT, thread float & RCQ, thread float & RCSOIL, thread float & SOILW, thread float & SOILM, thread float & Q1, thread float * SMAV, thread bool & RDLAI2D, thread bool & USEMONALB, thread float & SNOTIME1, thread float & RIBB, thread float & SMCWLT, thread float & SMCDRY, thread float & SMCREF, thread float & SMCMAX, thread int & NROOT, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & ETPND1, thread int & OPT_THCND, thread float & AOASIS, thread float & XSDA_QFX, thread float & HFX_PHY, thread float & QFX_PHY, thread float & XQNORM, thread int & FASDAS, thread float & HCPCT_FASDAS, thread float & IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL);
static __attribute__((noinline)) void noah_alcalc(thread NoahTables & tables, thread float & ALB, thread float & SNOALB, thread float & EMBRD, thread float & SHDFAC, thread float & SHDMIN, thread float & SNCOVR, thread float & TSNOW, thread float & ALBEDO, thread float & EMISSI, thread float & DT, thread bool & SNOWNG, thread float & SNOTIME1, thread float & LVCOEF);
static __attribute__((noinline)) void noah_canres(thread NoahTables & tables, thread float & SOLAR, thread float & CH, thread float & SFCTMP, thread float & Q2, thread float & SFCPRS, thread float * SMC, thread float * ZSOIL, thread int & NSOIL, thread float & SMCWLT, thread float & SMCREF, thread float & RSMIN, thread float & RC, thread float & PC, thread int & NROOT, thread float & Q2SAT, thread float & DQSDT2, thread float & TOPT, thread float & RSMAX, thread float & RGL, thread float & HS, thread float & XLAI, thread float & RCS, thread float & RCT, thread float & RCQ, thread float & RCSOIL, thread float & EMISSI);
static __attribute__((noinline)) void noah_csnow(thread NoahTables & tables, thread float & SNCOND, thread float & DSNOW);
static __attribute__((noinline)) void noah_devap(thread NoahTables & tables, thread float & EDIR, thread float & ETP1, thread float & SMC, thread float & ZSOIL, thread float & SHDFAC, thread float & SMCMAX, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SMCDRY, thread float & SMCREF, thread float & SMCWLT, thread float & FXEXP);
static __attribute__((noinline)) void noah_devap_hydro(thread NoahTables & tables, thread float & EDIR, thread float & ETP1, thread float & SMC, thread float & ZSOIL, thread float & SHDFAC, thread float & SMCMAX, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SMCDRY, thread float & SMCREF, thread float & SMCWLT, thread float & FXEXP, thread float & SFHEAD1RT, thread float & ETPND1, thread float & DT);
static __attribute__((noinline)) void noah_evapo(thread NoahTables & tables, thread float & ETA1, thread float * SMC, thread int & NSOIL, thread float & CMC, thread float & ETP1, thread float & DT, thread float * ZSOIL, thread float * SH2O, thread float & SMCMAX, thread float & BEXP, thread float & PC, thread float & SMCWLT, thread float & DKSAT, thread float & DWSAT, thread float & SMCREF, thread float & SHDFAC, thread float & CMCMAX, thread float & SMCDRY, thread float & CFACTR, thread float & EDIR, thread float & EC, thread float * ET, thread float & ETT, thread float & SFCTMP, thread float & Q2, thread int & NROOT, thread float * RTDIS, thread float & FXEXP, thread float & SFHEAD1RT, thread float & ETPND1);
static __attribute__((noinline)) void noah_fac2mit(thread NoahTables & tables, thread float & SMCMAX, thread float & FLIMIT);
static __attribute__((noinline)) void noah_frh2o(thread NoahTables & tables, thread float & FREE, thread float & TKELV, thread float & SMC, thread float & SH2O, thread float & SMCMAX, thread float & BEXP, thread float & PSIS);
static __attribute__((noinline)) void noah_hrt(thread NoahTables & tables, thread float * RHSTS, thread float * STC, thread float * SMC, thread float & SMCMAX, thread int & NSOIL, thread float * ZSOIL, thread float & YY, thread float & ZZ1, thread float & TBOT, thread float & ZBOT, thread float & PSISAT, thread float * SH2O, thread float & DT, thread float & BEXP, thread int & SOILTYP, thread int & OPT_THCND, thread float & F1, thread float & DF1, thread float & QUARTZ, thread float & CSOIL, thread float * AI, thread float * BI, thread float * CI, thread int & VEGTYP, thread int & ISURBAN, thread float & HCPCT_FASDAS);
static __attribute__((noinline)) void noah_hstep(thread NoahTables & tables, thread float * STCOUT, thread float * STCIN, thread float * RHSTS, thread float & DT, thread int & NSOIL, thread float * AI, thread float * BI, thread float * CI);
static __attribute__((noinline)) void noah_nopac(thread NoahTables & tables, thread float & ETP, thread float & ETA, thread float & PRCP, thread float * SMC, thread float & SMCMAX, thread float & SMCWLT, thread float & SMCREF, thread float & SMCDRY, thread float & CMC, thread float & CMCMAX, thread int & NSOIL, thread float & DT, thread float & SHDFAC, thread float & SBETA, thread float & Q2, thread float & T1, thread float & SFCTMP, thread float & T24, thread float & TH2, thread float & FDOWN, thread float & F1, thread float & EMISSI, thread float & SSOIL, thread float * STC, thread float & EPSCA, thread float & BEXP, thread float & PC, thread float & RCH, thread float & RR, thread float & CFACTR, thread float * SH2O, thread float & SLOPE, thread float & KDT, thread float & FRZFACT, thread float & PSISAT, thread float * ZSOIL, thread float & DKSAT, thread float & DWSAT, thread float & TBOT, thread float & ZBOT, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & EDIR, thread float & EC, thread float * ET, thread float & ETT, thread int & NROOT, thread float * RTDIS, thread float & QUARTZ, thread float & FXEXP, thread float & CSOIL, thread float & BETA, thread float & DRIP, thread float & DEW, thread float & FLX1, thread float & FLX3, thread int & VEGTYP, thread int & ISURBAN, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & ETPND1, thread int & SOILTYP, thread int & OPT_THCND, thread float & XSDA_QFX, thread float & QFX_PHY, thread float & XQNORM, thread int & FASDAS, thread float & HCPCT_FASDAS, thread float & IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL);
static __attribute__((noinline)) void noah_penman(thread NoahTables & tables, thread float & SFCTMP, thread float & SFCPRS, thread float & CH, thread float & T2V, thread float & TH2, thread float & PRCP, thread float & FDOWN, thread float & T24, thread float & SSOIL, thread float & Q2, thread float & Q2SAT, thread float & ETP, thread float & RCH, thread float & EPSCA, thread float & RR, thread bool & SNOWNG, thread bool & FRZGRA, thread float & DQSDT2, thread float & FLX2, thread float & EMISSI_IN, thread float & SNEQV, thread float & T1, thread float & SNCOVR, thread float & AOASIS, thread float & ALBEDO, thread float & SOLDN, thread float & FVB, thread float & GAMA, thread float & STC1, thread float & ETPN, thread float & FLX4, thread bool & UA_PHYS);
static __attribute__((noinline)) void noah_redprm(thread NoahTables & tables, thread int & VEGTYP, thread int & SOILTYP, thread int & SLOPETYP, thread float & CFACTR, thread float & CMCMAX, thread float & RSMAX, thread float & TOPT, thread float & REFKDT, thread float & KDT, thread float & SBETA, thread float & SHDFAC, thread float & RSMIN, thread float & RGL, thread float & HS, thread float & ZBOT, thread float & FRZX, thread float & PSISAT, thread float & SLOPE, thread float & SNUP, thread float & SALP, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SMCMAX, thread float & SMCWLT, thread float & SMCREF, thread float & SMCDRY, thread float & F1, thread float & QUARTZ, thread float & FXEXP, thread float * RTDIS, thread float * SLDPTH, thread float * ZSOIL, thread int & NROOT, thread int & NSOIL, thread float & CZIL, thread float & LAIMIN, thread float & LAIMAX, thread float & EMISSMIN, thread float & EMISSMAX, thread float & ALBEDOMIN, thread float & ALBEDOMAX, thread float & Z0MIN, thread float & Z0MAX, thread float & CSOIL, thread float & PTU, thread const char * LLANDUSE, thread const char * LSOIL, thread bool & LOCAL, thread float & LVCOEF, thread float & ZTOPV, thread float & ZBOTV);
static __attribute__((noinline)) void noah_rosr12(thread NoahTables & tables, thread float * P, thread float * A, thread float * B, thread float * C, thread float * D, thread float * DELTA, thread int & NSOIL);
static __attribute__((noinline)) void noah_shflx(thread NoahTables & tables, thread float & SSOIL, thread float * STC, thread float * SMC, thread float & SMCMAX, thread int & NSOIL, thread float & T1, thread float & DT, thread float & YY, thread float & ZZ1, thread float * ZSOIL, thread float & TBOT, thread float & ZBOT, thread float & SMCWLT, thread float & PSISAT, thread float * SH2O, thread float & BEXP, thread float & F1, thread float & DF1, thread float & QUARTZ, thread float & CSOIL, thread int & VEGTYP, thread int & ISURBAN, thread int & SOILTYP, thread int & OPT_THCND, thread float & HCPCT_FASDAS);
static __attribute__((noinline)) void noah_smflx(thread NoahTables & tables, thread float * SMC, thread int & NSOIL, thread float & CMC, thread float & DT, thread float & PRCP1, thread float * ZSOIL, thread float * SH2O, thread float & SLOPE, thread float & KDT, thread float & FRZFACT, thread float & SMCMAX, thread float & BEXP, thread float & SMCWLT, thread float & DKSAT, thread float & DWSAT, thread float & SHDFAC, thread float & CMCMAX, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & EDIR, thread float & EC, thread float * ET, thread float & DRIP, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL);
static __attribute__((noinline)) void noah_snfrac(thread NoahTables & tables, thread float & SNEQV, thread float & SNUP, thread float & SALP, thread float & SNOWH, thread float & SNCOVR, thread float & XLAI, thread float & SHDFAC, thread float & FVB, thread float & GAMA, thread float & FBUR, thread float & FGSN, thread float & ZTOPV, thread float & ZBOTV, thread bool & UA_PHYS);
static __attribute__((noinline)) void noah_snksrc(thread NoahTables & tables, thread float & TSNSR, thread float & TAVG, thread float & SMC, thread float & SH2O, thread float * ZSOIL, thread int & NSOIL, thread float & SMCMAX, thread float & PSISAT, thread float & BEXP, thread float & DT, thread int & K, thread float & QTOT);
static __attribute__((noinline)) void noah_snopac(thread NoahTables & tables, thread float & ETP, thread float & ETA, thread float & PRCP, thread float & PRCPF, thread bool & SNOWNG, thread float * SMC, thread float & SMCMAX, thread float & SMCWLT, thread float & SMCREF, thread float & SMCDRY, thread float & CMC, thread float & CMCMAX, thread int & NSOIL, thread float & DT, thread float & SBETA, thread float & DF1, thread float & Q2, thread float & T1, thread float & SFCTMP, thread float & T24, thread float & TH2, thread float & FDOWN, thread float & F1, thread float & SSOIL, thread float * STC, thread float & EPSCA, thread float & SFCPRS, thread float & BEXP, thread float & PC, thread float & RCH, thread float & RR, thread float & CFACTR, thread float & SNCOVR, thread float & ESD, thread float & SNDENS, thread float & SNOWH, thread float * SH2O, thread float & SLOPE, thread float & KDT, thread float & FRZFACT, thread float & PSISAT, thread float * ZSOIL, thread float & DWSAT, thread float & DKSAT, thread float & TBOT, thread float & ZBOT, thread float & SHDFAC, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & EDIR, thread float & EC, thread float * ET, thread float & ETT, thread int & NROOT, thread float & SNOMLT, thread float * RTDIS, thread float & QUARTZ, thread float & FXEXP, thread float & CSOIL, thread float & BETA, thread float & DRIP, thread float & DEW, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & ESNOW, thread float & ETNS, thread float & EMISSI, thread float & RIBB, thread float & SOLDN, thread int & ISURBAN, thread int & VEGTYP, thread float & ETPN, thread float & FLX4, thread bool & UA_PHYS, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & ETPND1, thread int & SOILTYP, thread int & OPT_THCND, thread float & QFX_PHY, thread int & FASDAS, thread float & HCPCT_FASDAS);
static __attribute__((noinline)) void noah_snowpack(thread NoahTables & tables, thread float & ESD, thread float & DTSEC, thread float & SNOWH, thread float & SNDENS, thread float & TSNOW, thread float & TSOIL, thread float & SNOMLT, thread bool & UA_PHYS);
static __attribute__((noinline)) void noah_snowz0(thread NoahTables & tables, thread float & SNCOVR, thread float & Z0, thread float & Z0BRD, thread float & SNOWH, thread float & FBUR, thread float & FGSN, thread float & SHDMAX, thread bool & UA_PHYS);
static __attribute__((noinline)) void noah_snow_new(thread NoahTables & tables, thread float & TEMP, thread float & NEWSN, thread float & SNOWH, thread float & SNDENS);
static __attribute__((noinline)) void noah_srt(thread NoahTables & tables, thread float * RHSTT, thread float & EDIR, thread float * ET, thread float * SH2O, thread float * SH2OA, thread int & NSOIL, thread float & PCPDRP, thread float * ZSOIL, thread float & DWSAT, thread float & DKSAT, thread float & SMCMAX, thread float & BEXP, thread float & RUNOFF1, thread float & RUNOFF2, thread float & DT, thread float & SMCWLT, thread float & SLOPE, thread float & KDT, thread float & FRZX, thread float * SICE, thread float * AI, thread float * BI, thread float * CI, thread float & SFHEAD1RT, thread float & INFXS1RT);
static __attribute__((noinline)) void noah_sstep(thread NoahTables & tables, thread float * SH2OOUT, thread float * SH2OIN, thread float & CMC, thread float * RHSTT, thread float & RHSCT, thread float & DT, thread int & NSOIL, thread float & SMCMAX, thread float & CMCMAX, thread float & RUNOFF3, thread float * ZSOIL, thread float * SMC, thread float * SICE, thread float * AI, thread float * BI, thread float * CI, thread float & INFXS1RT);
static __attribute__((noinline)) void noah_tbnd(thread NoahTables & tables, thread float & TU, thread float & TB, thread float * ZSOIL, thread float & ZBOT, thread int & K, thread int & NSOIL, thread float & TBND1);
static __attribute__((noinline)) void noah_tdfcnd(thread NoahTables & tables, thread float & DF, thread float & SMC, thread float & QZ, thread float & SMCMAX, thread float & SH2O, thread float & BEXP, thread float & PSISAT, thread int & SOILTYP, thread int & OPT_THCND);
static __attribute__((noinline)) void noah_tmpavg(thread NoahTables & tables, thread float & TAVG, thread float & TUP, thread float & TM, thread float & TDN, thread float * ZSOIL, thread int & NSOIL, thread int & K);
static __attribute__((noinline)) void noah_transp(thread NoahTables & tables, thread float * ET, thread int & NSOIL, thread float & ETP1, thread float * SMC, thread float & CMC, thread float * ZSOIL, thread float & SHDFAC, thread float & SMCWLT, thread float & CMCMAX, thread float & PC, thread float & CFACTR, thread float & SMCREF, thread float & SFCTMP, thread float & Q2, thread int & NROOT, thread float * RTDIS);
static __attribute__((noinline)) void noah_wdfcnd(thread NoahTables & tables, thread float & WDF, thread float & WCND, thread float & SMC, thread float & SMCMAX, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SICEMAX);
static __attribute__((noinline)) void noah_sfcdif_off(thread NoahTables & tables, thread float & ZLM, thread float & Z0, thread float & THZ0, thread float & THLM, thread float & SFCSPD, thread float & CZIL, thread float & AKMS, thread float & AKHS);
static __attribute__((noinline)) void gl_sflx_glacial(thread NoahTables & tables, thread int & IILOC, thread int & JJLOC, thread int & ISICE, thread float & FFROZP, thread float & DT, thread float & ZLVL, thread int & NSOIL, thread float * SLDPTH, thread float & LWDN, thread float & SOLNET, thread float & SFCPRS, thread float & PRCP, thread float & SFCTMP, thread float & Q2, thread float & TH2, thread float & Q2SAT, thread float & DQSDT2, thread float & ALB, thread float & SNOALB, thread float & TBOT, thread float & Z0BRD, thread float & Z0, thread float & EMISSI, thread float & EMBRD, thread float & T1, thread float * STC, thread float & SNOWH, thread float & SNEQV, thread float & ALBEDO, thread float & CH, thread float & ETA, thread float & SHEAT, thread float & ETA_KINEMATIC, thread float & FDOWN, thread float & ESNOW, thread float & DEW, thread float & ETP, thread float & SSOIL, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & SNOMLT, thread float & SNCOVR, thread float & RUNOFF1, thread float & Q1, thread float & SNOTIME1, thread float & RIBB);
static __attribute__((noinline)) void gl_alcalc(thread NoahTables & tables, thread float & ALB, thread float & SNOALB, thread float & EMBRD, thread float & TSNOW, thread float & ALBEDO, thread float & EMISSI, thread float & DT, thread bool & SNOWNG, thread float & SNOTIME1);
static __attribute__((noinline)) void gl_csnow(thread NoahTables & tables, thread float & SNCOND, thread float & DSNOW);
static __attribute__((noinline)) void gl_hrtice(thread NoahTables & tables, thread float * RHSTS, thread float * STC, thread float & TBOT, thread int & NSOIL, thread float * ZSOIL, thread float & YY, thread float & ZZ1, thread float & DF1, thread float * AI, thread float * BI, thread float * CI);
static __attribute__((noinline)) void gl_hstep(thread NoahTables & tables, thread float * STCOUT, thread float * STCIN, thread float * RHSTS, thread float & DT, thread int & NSOIL, thread float * AI, thread float * BI, thread float * CI);
static __attribute__((noinline)) void gl_penman(thread NoahTables & tables, thread float & SFCTMP, thread float & SFCPRS, thread float & CH, thread float & TH2, thread float & PRCP, thread float & FDOWN, thread float & T24, thread float & SSOIL, thread float & Q2, thread float & Q2SAT, thread float & ETP, thread float & RCH, thread float & RR, thread bool & SNOWNG, thread bool & FRZGRA, thread float & DQSDT2, thread float & FLX2, thread float & EMISSI, thread float & T1);
static __attribute__((noinline)) void gl_shflx(thread NoahTables & tables, thread float * STC, thread int & NSOIL, thread float & DT, thread float & YY, thread float & ZZ1, thread float * ZSOIL, thread float & TBOT, thread float & DF1);
static __attribute__((noinline)) void gl_snopac(thread NoahTables & tables, thread float & ETP, thread float & ETA, thread float & PRCP, thread float & PRCPF, thread bool & SNOWNG, thread int & NSOIL, thread float & DT, thread float & DF1, thread float & Q2, thread float & T1, thread float & SFCTMP, thread float & T24, thread float & TH2, thread float & FDOWN, thread float & SSOIL, thread float * STC, thread float & SFCPRS, thread float & RCH, thread float & RR, thread float & SNEQV, thread float & SNDENS, thread float & SNOWH, thread float * ZSOIL, thread float & TBOT, thread float & SNOMLT, thread float & DEW, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & ESNOW, thread float & EMISSI, thread float & RIBB);
static __attribute__((noinline)) void gl_snowpack(thread NoahTables & tables, thread float & SNEQV, thread float & DTSEC, thread float & SNOWH, thread float & SNDENS, thread float & TSNOW, thread float & TSOIL);
static __attribute__((noinline)) void gl_snowz0(thread NoahTables & tables, thread float & Z0, thread float & Z0BRD, thread float & SNOWH);
static __attribute__((noinline)) void gl_snow_new(thread NoahTables & tables, thread float & TEMP, thread float & NEWSN, thread float & SNOWH, thread float & SNDENS);
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:73-892
static __attribute__((noinline)) void noah_sflx(thread NoahTables & tables, thread int & IILOC, thread int & JJLOC, thread float & FFROZP, thread int & ISURBAN, thread float & DT, thread float & ZLVL, thread int & NSOIL, thread float * SLDPTH, thread bool & LOCAL, thread const char * LLANDUSE, thread const char * LSOIL, thread float & LWDN, thread float & SOLDN, thread float & SOLNET, thread float & SFCPRS, thread float & PRCP, thread float & SFCTMP, thread float & Q2, thread float & SFCSPD, thread float & COSZ, thread float & PRCPRAIN, thread float & SOLARDIRECT, thread float & TH2, thread float & Q2SAT, thread float & DQSDT2, thread int & VEGTYP, thread int & SOILTYP, thread int & SLOPETYP, thread float & SHDFAC, thread float & SHDMIN, thread float & SHDMAX, thread float & ALB, thread float & SNOALB, thread float & TBOT, thread float & Z0BRD, thread float & Z0, thread float & EMISSI, thread float & EMBRD, thread float & CMC, thread float & T1, thread float * STC, thread float * SMC, thread float * SH2O, thread float & SNOWH, thread float & SNEQV, thread float & ALBEDO, thread float & CH, thread float & CM, thread float & ETA, thread float & SHEAT, thread float & ETA_KINEMATIC, thread float & FDOWN, thread float & EC, thread float & EDIR, thread float * ET, thread float & ETT, thread float & ESNOW, thread float & DRIP, thread float & DEW, thread float & BETA, thread float & ETP, thread float & SSOIL, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & FLX4, thread float & FVB, thread float & FBUR, thread float & FGSN, thread bool & UA_PHYS, thread float & SNOMLT, thread float & SNCOVR, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & RC, thread float & PC, thread float & RSMIN, thread float & XLAI, thread float & RCS, thread float & RCT, thread float & RCQ, thread float & RCSOIL, thread float & SOILW, thread float & SOILM, thread float & Q1, thread float * SMAV, thread bool & RDLAI2D, thread bool & USEMONALB, thread float & SNOTIME1, thread float & RIBB, thread float & SMCWLT, thread float & SMCDRY, thread float & SMCREF, thread float & SMCMAX, thread int & NROOT, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & ETPND1, thread int & OPT_THCND, thread float & AOASIS, thread float & XSDA_QFX, thread float & HFX_PHY, thread float & QFX_PHY, thread float & XQNORM, thread int & FASDAS, thread float & HCPCT_FASDAS, thread float & IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL){
thread auto& ILOC=tables.ILOC;
thread auto& JLOC=tables.JLOC;
thread auto& CP=tables.CP;
bool FRZGRA;
bool SNOWNG;
int KZ;
int K;
int IOUT;
float RTDIS[20];
float ZSOIL[20];
float ZTOPV;
float ZBOTV;
float GAMA;
float FNET;
float ETPN;
float RU;
float BEXP;
float CFACTR;
float CMCMAX;
float CSOIL;
float CZIL;
float DF1;
float DF1H;
float DF1A;
float DKSAT;
float DWSAT;
float DSOIL;
float DTOT;
float FRCSNO;
float FRCSOI;
float EPSCA;
float F1;
float FXEXP;
float FRZX;
float HS;
float KDT;
float LVH2O = 0x1.314c400000000p+21f;
float PRCP1;
float PSISAT;
float QUARTZ;
float R = 0x1.1f0a3e0000000p+8f;
float RCH;
float REFKDT;
float RR;
float RGL;
float RSMAX;
float RSNOW;
float SNDENS;
float SNCOND;
float SBETA;
float SN_NEW;
float SLOPE;
float SNUP;
float SALP;
float SOILWM;
float SOILWW;
float T1V;
float T24;
float T2V;
float TH2V;
float TOPT;
float TFREEZ = 0x1.1126660000000p+8f;
float TSNOW;
float ZBOT;
float PRCPF;
float ETNS;
float PTU;
float LSUBS = 0x1.5975800000000p+21f;
float LVCOEF;
float INTERP_FRACTION;
float LAIMIN;
float LAIMAX;
float ALBEDOMIN;
float ALBEDOMAX;
float EMISSMIN;
float EMISSMAX;
float Z0MIN;
float Z0MAX;
 // module_sf_noahlsm.F:299
ILOC = IILOC; // module_sf_noahlsm.F:398
JLOC = JJLOC; // module_sf_noahlsm.F:399
RUNOFF1 = 0x0.0p+0f; // module_sf_noahlsm.F:401
RUNOFF2 = 0x0.0p+0f; // module_sf_noahlsm.F:402
RUNOFF3 = 0x0.0p+0f; // module_sf_noahlsm.F:403
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm.F:404
if ((!UA_PHYS)) { // module_sf_noahlsm.F:406
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:407
FVB = 0x0.0p+0f; // module_sf_noahlsm.F:408
FBUR = 0x0.0p+0f; // module_sf_noahlsm.F:409
FGSN = 0x0.0p+0f; // module_sf_noahlsm.F:410
} // module_sf_noahlsm.F:411
ZSOIL[(1)-(1)] = (-SLDPTH[(1)-(1)]); // module_sf_noahlsm.F:418
for (KZ=2; KZ<=NSOIL; KZ+=1) { // module_sf_noahlsm.F:419
ZSOIL[(KZ)-(1)] = AD((-SLDPTH[(KZ)-(1)]),ZSOIL[((KZ - 1))-(1)]); // module_sf_noahlsm.F:420
} // module_sf_noahlsm.F:421
{  noah_redprm(tables,VEGTYP,SOILTYP,SLOPETYP,CFACTR,CMCMAX,RSMAX,TOPT,REFKDT,KDT,SBETA,SHDFAC,RSMIN,RGL,HS,ZBOT,FRZX,PSISAT,SLOPE,SNUP,SALP,BEXP,DKSAT,DWSAT,SMCMAX,SMCWLT,SMCREF,SMCDRY,F1,QUARTZ,FXEXP,RTDIS,SLDPTH,ZSOIL,NROOT,NSOIL,CZIL,LAIMIN,LAIMAX,EMISSMIN,EMISSMAX,ALBEDOMIN,ALBEDOMAX,Z0MIN,Z0MAX,CSOIL,PTU,LLANDUSE,LSOIL,LOCAL,LVCOEF,ZTOPV,ZBOTV); if(tables.error)return; } // module_sf_noahlsm.F:427
if ((VEGTYP == ISURBAN)) { // module_sf_noahlsm.F:437
SHDFAC = 0x1.99999a0000000p-5f; // module_sf_noahlsm.F:438
RSMIN = 0x1.9000000000000p+8f; // module_sf_noahlsm.F:439
SMCMAX = 0x1.cccccc0000000p-2f; // module_sf_noahlsm.F:440
SMCREF = 0x1.ae147a0000000p-2f; // module_sf_noahlsm.F:441
SMCWLT = 0x1.99999a0000000p-2f; // module_sf_noahlsm.F:442
SMCDRY = 0x1.99999a0000000p-2f; // module_sf_noahlsm.F:443
} // module_sf_noahlsm.F:444
if ((SHDFAC >= SHDMAX)) { // module_sf_noahlsm.F:446
EMBRD = EMISSMAX; // module_sf_noahlsm.F:447
if ((!RDLAI2D)) { // module_sf_noahlsm.F:448
XLAI = LAIMAX; // module_sf_noahlsm.F:449
} // module_sf_noahlsm.F:450
if ((!USEMONALB)) { // module_sf_noahlsm.F:451
ALB = ALBEDOMIN; // module_sf_noahlsm.F:452
} // module_sf_noahlsm.F:453
Z0BRD = Z0MAX; // module_sf_noahlsm.F:454
} else if ((SHDFAC <= SHDMIN)) { // module_sf_noahlsm.F:455
EMBRD = EMISSMIN; // module_sf_noahlsm.F:456
if ((!RDLAI2D)) { // module_sf_noahlsm.F:457
XLAI = LAIMIN; // module_sf_noahlsm.F:458
} // module_sf_noahlsm.F:459
if ((!USEMONALB)) { // module_sf_noahlsm.F:460
ALB = ALBEDOMAX; // module_sf_noahlsm.F:461
} // module_sf_noahlsm.F:462
Z0BRD = Z0MIN; // module_sf_noahlsm.F:463
} else { // module_sf_noahlsm.F:464
if ((SHDMAX > SHDMIN)) { // module_sf_noahlsm.F:466
INTERP_FRACTION = DV(SU(SHDFAC,SHDMIN),SU(SHDMAX,SHDMIN)); // module_sf_noahlsm.F:468
INTERP_FRACTION = noah_min(INTERP_FRACTION,0x1.0000000000000p+0f); // module_sf_noahlsm.F:470
INTERP_FRACTION = noah_max(INTERP_FRACTION,0x0.0p+0f); // module_sf_noahlsm.F:471
EMBRD = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),EMISSMIN),MU(INTERP_FRACTION,EMISSMAX)); // module_sf_noahlsm.F:473
if ((!RDLAI2D)) { // module_sf_noahlsm.F:474
XLAI = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),LAIMIN),MU(INTERP_FRACTION,LAIMAX)); // module_sf_noahlsm.F:475
} // module_sf_noahlsm.F:476
if ((!USEMONALB)) { // module_sf_noahlsm.F:477
ALB = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),ALBEDOMAX),MU(INTERP_FRACTION,ALBEDOMIN)); // module_sf_noahlsm.F:478
} // module_sf_noahlsm.F:479
Z0BRD = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),Z0MIN),MU(INTERP_FRACTION,Z0MAX)); // module_sf_noahlsm.F:480
} else { // module_sf_noahlsm.F:482
EMBRD = AD(MU(0x1.0000000000000p-1f,EMISSMIN),MU(0x1.0000000000000p-1f,EMISSMAX)); // module_sf_noahlsm.F:484
if ((!RDLAI2D)) { // module_sf_noahlsm.F:485
XLAI = AD(MU(0x1.0000000000000p-1f,LAIMIN),MU(0x1.0000000000000p-1f,LAIMAX)); // module_sf_noahlsm.F:486
} // module_sf_noahlsm.F:487
if ((!USEMONALB)) { // module_sf_noahlsm.F:488
ALB = AD(MU(0x1.0000000000000p-1f,ALBEDOMIN),MU(0x1.0000000000000p-1f,ALBEDOMAX)); // module_sf_noahlsm.F:489
} // module_sf_noahlsm.F:490
Z0BRD = AD(MU(0x1.0000000000000p-1f,Z0MIN),MU(0x1.0000000000000p-1f,Z0MAX)); // module_sf_noahlsm.F:491
} // module_sf_noahlsm.F:493
} // module_sf_noahlsm.F:495
SNOWNG = false; // module_sf_noahlsm.F:499
FRZGRA = false; // module_sf_noahlsm.F:500
if ((SNEQV <= 0x1.ad7f2a0000000p-24f)) { // module_sf_noahlsm.F:507
SNEQV = 0x0.0p+0f; // module_sf_noahlsm.F:508
SNDENS = 0x0.0p+0f; // module_sf_noahlsm.F:509
SNOWH = 0x0.0p+0f; // module_sf_noahlsm.F:510
SNCOND = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:511
} else { // module_sf_noahlsm.F:512
SNDENS = DV(SNEQV,SNOWH); // module_sf_noahlsm.F:513
if ((SNDENS > 0x1.0000000000000p+0f)) { // module_sf_noahlsm.F:514
tables.error = 515; return; // module_sf_noahlsm.F:515
} // module_sf_noahlsm.F:516
{  noah_csnow(tables,SNCOND,SNDENS); if(tables.error)return; } // module_sf_noahlsm.F:517
} // module_sf_noahlsm.F:518
if ((PRCP > 0x0.0p+0f)) { // module_sf_noahlsm.F:525
if ((FFROZP > 0x1.0000000000000p-1f)) { // module_sf_noahlsm.F:528
SNOWNG = true; // module_sf_noahlsm.F:529
} else { // module_sf_noahlsm.F:530
if ((T1 <= TFREEZ)) { FRZGRA = true; } // module_sf_noahlsm.F:531
} // module_sf_noahlsm.F:532
} // module_sf_noahlsm.F:533
if ((SNOWNG || FRZGRA)) { // module_sf_noahlsm.F:541
SN_NEW = MU(MU(PRCP,DT),0x1.0624de0000000p-10f); // module_sf_noahlsm.F:542
SNEQV = AD(SNEQV,SN_NEW); // module_sf_noahlsm.F:543
PRCPF = 0x0.0p+0f; // module_sf_noahlsm.F:544
{  noah_snow_new(tables,SFCTMP,SN_NEW,SNOWH,SNDENS); if(tables.error)return; } // module_sf_noahlsm.F:550
{  noah_csnow(tables,SNCOND,SNDENS); if(tables.error)return; } // module_sf_noahlsm.F:551
} else { // module_sf_noahlsm.F:558
PRCPF = PRCP; // module_sf_noahlsm.F:559
} // module_sf_noahlsm.F:560
if ((SNEQV == 0x0.0p+0f)) { // module_sf_noahlsm.F:567
SNCOVR = 0x0.0p+0f; // module_sf_noahlsm.F:568
ALBEDO = ALB; // module_sf_noahlsm.F:569
EMISSI = EMBRD; // module_sf_noahlsm.F:570
if (UA_PHYS) { FGSN = 0x0.0p+0f; } // module_sf_noahlsm.F:571
if (UA_PHYS) { FVB = 0x0.0p+0f; } // module_sf_noahlsm.F:572
if (UA_PHYS) { FBUR = 0x0.0p+0f; } // module_sf_noahlsm.F:573
} else { // module_sf_noahlsm.F:574
{  noah_snfrac(tables,SNEQV,SNUP,SALP,SNOWH,SNCOVR,XLAI,SHDFAC,FVB,GAMA,FBUR,FGSN,ZTOPV,ZBOTV,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:579
if (UA_PHYS) { // module_sf_noahlsm.F:583
if ((SFCTMP <= T1)) { // module_sf_noahlsm.F:584
RU = 0x0.0p+0f; // module_sf_noahlsm.F:585
} else { // module_sf_noahlsm.F:586
RU = MU(MU(MU(MU(0x1.9000000000000p+6f,SHDFAC),FGSN),noah_min(DV(SU(SFCTMP,T1),0x1.4000000000000p+2f),0x1.0000000000000p+0f)),SU(0x1.0000000000000p+0f,r_exp((-XLAI)))); // module_sf_noahlsm.F:587
} // module_sf_noahlsm.F:588
CH = DV(CH,AD(0x1.0000000000000p+0f,MU(RU,CH))); // module_sf_noahlsm.F:589
} // module_sf_noahlsm.F:590
SNCOVR = noah_min(SNCOVR,0x1.f5c2900000000p-1f); // module_sf_noahlsm.F:592
{  noah_alcalc(tables,ALB,SNOALB,EMBRD,SHDFAC,SHDMIN,SNCOVR,T1,ALBEDO,EMISSI,DT,SNOWNG,SNOTIME1,LVCOEF); if(tables.error)return; } // module_sf_noahlsm.F:594
} // module_sf_noahlsm.F:596
{  noah_tdfcnd(tables,DF1,SMC[(1)-(1)],QUARTZ,SMCMAX,SH2O[(1)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:621
if ((VEGTYP == ISURBAN)) { DF1 = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:624
DF1 = MU(DF1,r_exp(MU(SBETA,SHDFAC))); // module_sf_noahlsm.F:626
if ((SNCOVR > 0x1.f0a3d80000000p-1f)) { // module_sf_noahlsm.F:631
DF1 = SNCOND; // module_sf_noahlsm.F:632
} // module_sf_noahlsm.F:633
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:641
if ((SNEQV == 0x0.0p+0f)) { // module_sf_noahlsm.F:642
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DSOIL); // module_sf_noahlsm.F:643
} else { // module_sf_noahlsm.F:644
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm.F:645
FRCSNO = DV(SNOWH,DTOT); // module_sf_noahlsm.F:646
FRCSOI = DV(DSOIL,DTOT); // module_sf_noahlsm.F:650
DF1H = DV(MU(SNCOND,DF1),AD(MU(FRCSOI,SNCOND),MU(FRCSNO,DF1))); // module_sf_noahlsm.F:653
DF1A = AD(MU(FRCSNO,SNCOND),MU(FRCSOI,DF1)); // module_sf_noahlsm.F:660
DF1 = AD(MU(DF1A,SNCOVR),MU(DF1,SU(0x1.0000000000000p+0f,SNCOVR))); // module_sf_noahlsm.F:667
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm.F:668
} // module_sf_noahlsm.F:669
if ((SNCOVR > 0x0.0p+0f)) { // module_sf_noahlsm.F:674
{  noah_snowz0(tables,SNCOVR,Z0,Z0BRD,SNOWH,FBUR,FGSN,SHDMAX,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:675
} else { // module_sf_noahlsm.F:676
Z0 = Z0BRD; // module_sf_noahlsm.F:677
if (UA_PHYS) { {  noah_snowz0(tables,SNCOVR,Z0,Z0BRD,SNOWH,FBUR,FGSN,SHDMAX,UA_PHYS); if(tables.error)return; } } // module_sf_noahlsm.F:678
} // module_sf_noahlsm.F:680
FDOWN = AD(SOLNET,LWDN); // module_sf_noahlsm.F:720
T2V = MU(SFCTMP,AD(0x1.0000000000000p+0f,MU(0x1.3851ec0000000p-1f,Q2))); // module_sf_noahlsm.F:724
IOUT = 0; // module_sf_noahlsm.F:726
if ((IOUT == 1)) { // module_sf_noahlsm.F:727
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:728
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:729
} // module_sf_noahlsm.F:737
{  noah_penman(tables,SFCTMP,SFCPRS,CH,T2V,TH2,PRCP,FDOWN,T24,SSOIL,Q2,Q2SAT,ETP,RCH,EPSCA,RR,SNOWNG,FRZGRA,DQSDT2,FLX2,EMISSI,SNEQV,T1,SNCOVR,AOASIS,ALBEDO,SOLDN,FVB,GAMA,STC[(1)-(1)],ETPN,FLX4,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:739
if (((SHDFAC > 0x0.0p+0f) && (XLAI > 0x0.0p+0f))) { // module_sf_noahlsm.F:753
{  noah_canres(tables,SOLDN,CH,SFCTMP,Q2,SFCPRS,SH2O,ZSOIL,NSOIL,SMCWLT,SMCREF,RSMIN,RC,PC,NROOT,Q2SAT,DQSDT2,TOPT,RSMAX,RGL,HS,XLAI,RCS,RCT,RCQ,RCSOIL,EMISSI); if(tables.error)return; } // module_sf_noahlsm.F:754
} else { // module_sf_noahlsm.F:758
RC = 0x0.0p+0f; // module_sf_noahlsm.F:759
} // module_sf_noahlsm.F:760
ESNOW = 0x0.0p+0f; // module_sf_noahlsm.F:765
if ((SNEQV == 0x0.0p+0f)) { // module_sf_noahlsm.F:766
{  noah_nopac(tables,ETP,ETA,PRCP,SMC,SMCMAX,SMCWLT,SMCREF,SMCDRY,CMC,CMCMAX,NSOIL,DT,SHDFAC,SBETA,Q2,T1,SFCTMP,T24,TH2,FDOWN,F1,EMISSI,SSOIL,STC,EPSCA,BEXP,PC,RCH,RR,CFACTR,SH2O,SLOPE,KDT,FRZX,PSISAT,ZSOIL,DKSAT,DWSAT,TBOT,ZBOT,RUNOFF1,RUNOFF2,RUNOFF3,EDIR,EC,ET,ETT,NROOT,RTDIS,QUARTZ,FXEXP,CSOIL,BETA,DRIP,DEW,FLX1,FLX3,VEGTYP,ISURBAN,SFHEAD1RT,INFXS1RT,ETPND1,SOILTYP,OPT_THCND,XSDA_QFX,QFX_PHY,XQNORM,FASDAS,HCPCT_FASDAS,IRRIGATION_CHANNEL,present_IRRIGATION_CHANNEL); if(tables.error)return; } // module_sf_noahlsm.F:767
ETA_KINEMATIC = ETA; // module_sf_noahlsm.F:781
} else { // module_sf_noahlsm.F:782
{  noah_snopac(tables,ETP,ETA,PRCP,PRCPF,SNOWNG,SMC,SMCMAX,SMCWLT,SMCREF,SMCDRY,CMC,CMCMAX,NSOIL,DT,SBETA,DF1,Q2,T1,SFCTMP,T24,TH2,FDOWN,F1,SSOIL,STC,EPSCA,SFCPRS,BEXP,PC,RCH,RR,CFACTR,SNCOVR,SNEQV,SNDENS,SNOWH,SH2O,SLOPE,KDT,FRZX,PSISAT,ZSOIL,DWSAT,DKSAT,TBOT,ZBOT,SHDFAC,RUNOFF1,RUNOFF2,RUNOFF3,EDIR,EC,ET,ETT,NROOT,SNOMLT,RTDIS,QUARTZ,FXEXP,CSOIL,BETA,DRIP,DEW,FLX1,FLX2,FLX3,ESNOW,ETNS,EMISSI,RIBB,SOLDN,ISURBAN,VEGTYP,ETPN,FLX4,UA_PHYS,SFHEAD1RT,INFXS1RT,ETPND1,SOILTYP,OPT_THCND,QFX_PHY,FASDAS,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:783
ETA_KINEMATIC = SU(AD(ESNOW,ETNS),MU(0x1.f400000000000p+9f,DEW)); // module_sf_noahlsm.F:799
} // module_sf_noahlsm.F:800
Q1 = AD(Q2,DV(MU(ETA_KINEMATIC,CP),RCH)); // module_sf_noahlsm.F:805
SHEAT = MU(DV((-MU(MU(CH,CP),SFCPRS)),MU(R,T2V)),SU(TH2,T1)); // module_sf_noahlsm.F:811
if (UA_PHYS) { SHEAT = AD(SHEAT,FLX4); } // module_sf_noahlsm.F:812
if ((FASDAS == 1)) { // module_sf_noahlsm.F:816
HFX_PHY = SHEAT; // module_sf_noahlsm.F:817
} // module_sf_noahlsm.F:818
EDIR = MU(EDIR,LVH2O); // module_sf_noahlsm.F:825
EC = MU(EC,LVH2O); // module_sf_noahlsm.F:826
for (K=1; K<=4; K+=1) { // module_sf_noahlsm.F:827
ET[(K)-(1)] = MU(ET[(K)-(1)],LVH2O); // module_sf_noahlsm.F:828
} // module_sf_noahlsm.F:829
ETT = MU(ETT,LVH2O); // module_sf_noahlsm.F:830
ETPND1 = MU(ETPND1,LVH2O); // module_sf_noahlsm.F:832
ESNOW = MU(ESNOW,LSUBS); // module_sf_noahlsm.F:834
ETP = MU(ETP,AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LVH2O),MU(SNCOVR,LSUBS))); // module_sf_noahlsm.F:835
if (UA_PHYS) { ETPN = MU(ETPN,AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LVH2O),MU(SNCOVR,LSUBS))); } // module_sf_noahlsm.F:836
if ((ETP > 0x0.0p+0f)) { // module_sf_noahlsm.F:837
ETA = AD(AD(AD(EDIR,EC),ETT),ESNOW); // module_sf_noahlsm.F:838
} else { // module_sf_noahlsm.F:839
ETA = ETP; // module_sf_noahlsm.F:840
} // module_sf_noahlsm.F:841
if ((ETP == 0x0.0p+0f)) { // module_sf_noahlsm.F:845
BETA = 0x0.0p+0f; // module_sf_noahlsm.F:846
} else { // module_sf_noahlsm.F:847
BETA = DV(ETA,ETP); // module_sf_noahlsm.F:848
} // module_sf_noahlsm.F:849
SSOIL = MU(-0x1.0000000000000p+0f,SSOIL); // module_sf_noahlsm.F:856
RUNOFF3 = DV(RUNOFF3,DT); // module_sf_noahlsm.F:864
RUNOFF2 = AD(RUNOFF2,RUNOFF3); // module_sf_noahlsm.F:865
SOILM = MU(MU(-0x1.0000000000000p+0f,SMC[(1)-(1)]),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:866
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:867
SOILM = AD(SOILM,MU(SMC[(K)-(1)],SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]))); // module_sf_noahlsm.F:868
} // module_sf_noahlsm.F:869
SOILWM = MU(MU(-0x1.0000000000000p+0f,SU(SMCMAX,SMCWLT)),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:870
SOILWW = MU(MU(-0x1.0000000000000p+0f,SU(SMC[(1)-(1)],SMCWLT)),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:871
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:873
SMAV[(K)-(1)] = DV(SU(SMC[(K)-(1)],SMCWLT),SU(SMCMAX,SMCWLT)); // module_sf_noahlsm.F:874
} // module_sf_noahlsm.F:875
if ((NROOT >= 2)) { // module_sf_noahlsm.F:877
for (K=2; K<=NROOT; K+=1) { // module_sf_noahlsm.F:878
SOILWM = AD(SOILWM,MU(SU(SMCMAX,SMCWLT),SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]))); // module_sf_noahlsm.F:879
SOILWW = AD(SOILWW,MU(SU(SMC[(K)-(1)],SMCWLT),SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]))); // module_sf_noahlsm.F:880
} // module_sf_noahlsm.F:881
} // module_sf_noahlsm.F:882
if ((SOILWM < 0x1.0c6f7a0000000p-20f)) { // module_sf_noahlsm.F:883
SOILWM = 0x0.0p+0f; // module_sf_noahlsm.F:884
SOILW = 0x0.0p+0f; // module_sf_noahlsm.F:885
SOILM = 0x0.0p+0f; // module_sf_noahlsm.F:886
} else { // module_sf_noahlsm.F:887
SOILW = DV(SOILWW,SOILWM); // module_sf_noahlsm.F:888
} // module_sf_noahlsm.F:889
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:895-1010
static __attribute__((noinline)) void noah_alcalc(thread NoahTables & tables, thread float & ALB, thread float & SNOALB, thread float & EMBRD, thread float & SHDFAC, thread float & SHDMIN, thread float & SNCOVR, thread float & TSNOW, thread float & ALBEDO, thread float & EMISSI, thread float & DT, thread bool & SNOWNG, thread float & SNOTIME1, thread float & LVCOEF){
thread auto& EMISSI_S=tables.EMISSI_S;
float SNOALB2;
float TM;
float SNOALB1;
float SNACCA = 0x1.e147ae0000000p-1f;
float SNACCB = 0x1.28f5c20000000p-1f;
float SNTHWA = 0x1.a3d70a0000000p-1f;
float SNTHWB = 0x1.d70a3e0000000p-2f;
 // module_sf_noahlsm.F:908
ALBEDO = AD(ALB,MU(SNCOVR,SU(SNOALB,ALB))); // module_sf_noahlsm.F:928
EMISSI = AD(EMBRD,MU(SNCOVR,SU(EMISSI_S,EMBRD))); // module_sf_noahlsm.F:929
SNOALB1 = AD(SNOALB,MU(LVCOEF,SU(0x1.b333340000000p-1f,SNOALB))); // module_sf_noahlsm.F:983
SNOALB2 = SNOALB1; // module_sf_noahlsm.F:984
if (SNOWNG) { // module_sf_noahlsm.F:986
SNOTIME1 = 0x0.0p+0f; // module_sf_noahlsm.F:987
} else { // module_sf_noahlsm.F:988
SNOTIME1 = AD(SNOTIME1,DT); // module_sf_noahlsm.F:989
SNOALB2 = MU(SNOALB1,r_pow(SNACCA,r_pow(DV(SNOTIME1,0x1.5180000000000p+16f),SNACCB))); // module_sf_noahlsm.F:991
} // module_sf_noahlsm.F:995
SNOALB2 = noah_max(SNOALB2,ALB); // module_sf_noahlsm.F:997
ALBEDO = AD(ALB,MU(SNCOVR,SU(SNOALB2,ALB))); // module_sf_noahlsm.F:998
if ((ALBEDO > SNOALB2)) { ALBEDO = SNOALB2; } // module_sf_noahlsm.F:999
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1013-1149
static __attribute__((noinline)) void noah_canres(thread NoahTables & tables, thread float & SOLAR, thread float & CH, thread float & SFCTMP, thread float & Q2, thread float & SFCPRS, thread float * SMC, thread float * ZSOIL, thread int & NSOIL, thread float & SMCWLT, thread float & SMCREF, thread float & RSMIN, thread float & RC, thread float & PC, thread int & NROOT, thread float & Q2SAT, thread float & DQSDT2, thread float & TOPT, thread float & RSMAX, thread float & RGL, thread float & HS, thread float & XLAI, thread float & RCS, thread float & RCT, thread float & RCQ, thread float & RCSOIL, thread float & EMISSI){
thread auto& RD=tables.RD;
thread auto& SIGMA=tables.SIGMA;
thread auto& CP=tables.CP;
int K;
float DELTA;
float FF;
float GX;
float P;
float RR;
float PART[20];
float SLV = 0x1.314c400000000p+21f;
 // module_sf_noahlsm.F:1054
RCS = 0x0.0p+0f; // module_sf_noahlsm.F:1070
RCT = 0x0.0p+0f; // module_sf_noahlsm.F:1071
RCQ = 0x0.0p+0f; // module_sf_noahlsm.F:1072
RCSOIL = 0x0.0p+0f; // module_sf_noahlsm.F:1073
RC = 0x0.0p+0f; // module_sf_noahlsm.F:1078
FF = DV(MU(0x1.19999a0000000p+0f,SOLAR),MU(RGL,XLAI)); // module_sf_noahlsm.F:1079
RCS = DV(AD(FF,DV(RSMIN,RSMAX)),AD(0x1.0000000000000p+0f,FF)); // module_sf_noahlsm.F:1080
RCS = noah_max(RCS,0x1.a36e2e0000000p-14f); // module_sf_noahlsm.F:1086
RCT = SU(0x1.0000000000000p+0f,MU(0x1.a36e2e0000000p-10f,MU(SU(TOPT,SFCTMP),SU(TOPT,SFCTMP)))); // module_sf_noahlsm.F:1087; strict GNU folds real exponent 2 into multiplication
RCT = noah_max(RCT,0x1.a36e2e0000000p-14f); // module_sf_noahlsm.F:1093
RCQ = DV(0x1.0000000000000p+0f,AD(0x1.0000000000000p+0f,MU(HS,SU(Q2SAT,Q2)))); // module_sf_noahlsm.F:1094
RCQ = noah_max(RCQ,0x1.47ae140000000p-7f); // module_sf_noahlsm.F:1100
GX = DV(SU(SMC[(1)-(1)],SMCWLT),SU(SMCREF,SMCWLT)); // module_sf_noahlsm.F:1101
if ((GX > 0x1.0000000000000p+0f)) { GX = 0x1.0000000000000p+0f; } // module_sf_noahlsm.F:1102
if ((GX < 0x0.0p+0f)) { GX = 0x0.0p+0f; } // module_sf_noahlsm.F:1103
PART[(1)-(1)] = MU(DV(ZSOIL[(1)-(1)],ZSOIL[(NROOT)-(1)]),GX); // module_sf_noahlsm.F:1112
for (K=2; K<=NROOT; K+=1) { // module_sf_noahlsm.F:1113
GX = DV(SU(SMC[(K)-(1)],SMCWLT),SU(SMCREF,SMCWLT)); // module_sf_noahlsm.F:1114
if ((GX > 0x1.0000000000000p+0f)) { GX = 0x1.0000000000000p+0f; } // module_sf_noahlsm.F:1115
if ((GX < 0x0.0p+0f)) { GX = 0x0.0p+0f; } // module_sf_noahlsm.F:1116
PART[(K)-(1)] = MU(DV(SU(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)]),ZSOIL[(NROOT)-(1)]),GX); // module_sf_noahlsm.F:1124
} // module_sf_noahlsm.F:1125
for (K=1; K<=NROOT; K+=1) { // module_sf_noahlsm.F:1126
RCSOIL = AD(RCSOIL,PART[(K)-(1)]); // module_sf_noahlsm.F:1127
} // module_sf_noahlsm.F:1128
RCSOIL = noah_max(RCSOIL,0x1.a36e2e0000000p-14f); // module_sf_noahlsm.F:1137
RC = DV(RSMIN,MU(MU(MU(MU(XLAI,RCS),RCT),RCQ),RCSOIL)); // module_sf_noahlsm.F:1139
RR = AD(DV(MU(DV(MU(MU(MU(0x1.0000000000000p+2f,EMISSI),SIGMA),RD),CP),r_pow(SFCTMP,0x1.0000000000000p+2f)),MU(SFCPRS,CH)),0x1.0000000000000p+0f); // module_sf_noahlsm.F:1141
DELTA = MU(DV(SLV,CP),DQSDT2); // module_sf_noahlsm.F:1144
PC = DV(AD(RR,DELTA),AD(MU(RR,AD(0x1.0000000000000p+0f,MU(RC,CH))),DELTA)); // module_sf_noahlsm.F:1146
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1152-1191
static __attribute__((noinline)) void noah_csnow(thread NoahTables & tables, thread float & SNCOND, thread float & DSNOW){
float C;
float UNIT = 0x1.dc67e00000000p-4f;
 // module_sf_noahlsm.F:1160
C = MU(0x1.4fdf3c0000000p-2f,r_pow(10,MU(0x1.2000000000000p+1f,DSNOW))); // module_sf_noahlsm.F:1171
SNCOND = MU(MU(0x1.0000000000000p+1f,UNIT),C); // module_sf_noahlsm.F:1188
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1193-1232
static __attribute__((noinline)) void noah_devap(thread NoahTables & tables, thread float & EDIR, thread float & ETP1, thread float & SMC, thread float & ZSOIL, thread float & SHDFAC, thread float & SMCMAX, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SMCDRY, thread float & SMCREF, thread float & SMCWLT, thread float & FXEXP){
float FX;
float SRATIO;
 // module_sf_noahlsm.F:1202
SRATIO = DV(SU(SMC,SMCDRY),SU(SMCMAX,SMCDRY)); // module_sf_noahlsm.F:1218
if ((SRATIO > 0x0.0p+0f)) { // module_sf_noahlsm.F:1219
FX = r_pow(SRATIO,FXEXP); // module_sf_noahlsm.F:1220
FX = noah_max(noah_min(FX,0x1.0000000000000p+0f),0x0.0p+0f); // module_sf_noahlsm.F:1221
} else { // module_sf_noahlsm.F:1222
FX = 0x0.0p+0f; // module_sf_noahlsm.F:1223
} // module_sf_noahlsm.F:1224
EDIR = MU(MU(FX,SU(0x1.0000000000000p+0f,SHDFAC)),ETP1); // module_sf_noahlsm.F:1229
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1234-1324

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1327-1425
static __attribute__((noinline)) void noah_evapo(thread NoahTables & tables, thread float & ETA1, thread float * SMC, thread int & NSOIL, thread float & CMC, thread float & ETP1, thread float & DT, thread float * ZSOIL, thread float * SH2O, thread float & SMCMAX, thread float & BEXP, thread float & PC, thread float & SMCWLT, thread float & DKSAT, thread float & DWSAT, thread float & SMCREF, thread float & SHDFAC, thread float & CMCMAX, thread float & SMCDRY, thread float & CFACTR, thread float & EDIR, thread float & EC, thread float * ET, thread float & ETT, thread float & SFCTMP, thread float & Q2, thread int & NROOT, thread float * RTDIS, thread float & FXEXP, thread float & SFHEAD1RT, thread float & ETPND1){
int I;
int K;
float CMC2MS;
 // module_sf_noahlsm.F:1344
EDIR = 0x0.0p+0f; // module_sf_noahlsm.F:1361
EC = 0x0.0p+0f; // module_sf_noahlsm.F:1362
ETT = 0x0.0p+0f; // module_sf_noahlsm.F:1363
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1364
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1365
} // module_sf_noahlsm.F:1366
if ((ETP1 > 0x0.0p+0f)) { // module_sf_noahlsm.F:1373
if ((SHDFAC < 0x1.0000000000000p+0f)) { // module_sf_noahlsm.F:1374
{  noah_devap(tables,EDIR,ETP1,SMC[(1)-(1)],ZSOIL[(1)-(1)],SHDFAC,SMCMAX,BEXP,DKSAT,DWSAT,SMCDRY,SMCREF,SMCWLT,FXEXP); if(tables.error)return; } // module_sf_noahlsm.F:1387
} // module_sf_noahlsm.F:1390
if ((SHDFAC > 0x0.0p+0f)) { // module_sf_noahlsm.F:1396
{  noah_transp(tables,ET,NSOIL,ETP1,SH2O,CMC,ZSOIL,SHDFAC,SMCWLT,CMCMAX,PC,CFACTR,SMCREF,SFCTMP,Q2,NROOT,RTDIS); if(tables.error)return; } // module_sf_noahlsm.F:1397
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1399
ETT = AD(ETT,ET[(K)-(1)]); // module_sf_noahlsm.F:1400
} // module_sf_noahlsm.F:1401
if ((CMC > 0x0.0p+0f)) { // module_sf_noahlsm.F:1406
EC = MU(MU(SHDFAC,r_pow(DV(CMC,CMCMAX),CFACTR)),ETP1); // module_sf_noahlsm.F:1407
} else { // module_sf_noahlsm.F:1408
EC = 0x0.0p+0f; // module_sf_noahlsm.F:1409
} // module_sf_noahlsm.F:1410
CMC2MS = DV(CMC,DT); // module_sf_noahlsm.F:1415
EC = noah_min(CMC2MS,EC); // module_sf_noahlsm.F:1416
} // module_sf_noahlsm.F:1417
} // module_sf_noahlsm.F:1418
ETA1 = AD(AD(EDIR,ETT),EC); // module_sf_noahlsm.F:1422
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1428-1448
static __attribute__((noinline)) void noah_fac2mit(thread NoahTables & tables, thread float & SMCMAX, thread float & FLIMIT){
 // module_sf_noahlsm.F:1429
FLIMIT = 0x1.cccccc0000000p-1f; // module_sf_noahlsm.F:1433
if ((SMCMAX == 0x1.947ae20000000p-2f)) { // module_sf_noahlsm.F:1435
FLIMIT = 0x1.2e147a0000000p-1f; // module_sf_noahlsm.F:1436
} else if (((SMCMAX == 0x1.bc6a7e0000000p-2f) || (SMCMAX == 0x1.9db22e0000000p-2f))) { // module_sf_noahlsm.F:1437
FLIMIT = 0x1.b333340000000p-1f; // module_sf_noahlsm.F:1438
} else if (((SMCMAX == 0x1.dc28f60000000p-2f) || (SMCMAX == 0x1.9fbe760000000p-2f))) { // module_sf_noahlsm.F:1439
FLIMIT = 0x1.b851ec0000000p-1f; // module_sf_noahlsm.F:1440
} else if (((SMCMAX == 0x1.e76c8c0000000p-2f) || (SMCMAX == 0x1.c189380000000p-2f))) { // module_sf_noahlsm.F:1441
FLIMIT = 0x1.7ae1480000000p-1f; // module_sf_noahlsm.F:1442
} else if (((SMCMAX == 0x1.99999a0000000p-3f) || (SMCMAX == 0x1.db22d00000000p-2f))) { // module_sf_noahlsm.F:1443
FLIMIT = 0x1.99999a0000000p-1f; // module_sf_noahlsm.F:1444
} // module_sf_noahlsm.F:1445
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1451-1589
static __attribute__((noinline)) void noah_frh2o(thread NoahTables & tables, thread float & FREE, thread float & TKELV, thread float & SMC, thread float & SH2O, thread float & SMCMAX, thread float & BEXP, thread float & PSIS){
float BX;
float DENOM;
float DF;
float DSWL;
float FK;
float SWL;
float SWLK;
int NLOG;
int KCOUNT;
float CK = 0x1.0000000000000p+3f;
float BLIM = 0x1.6000000000000p+2f;
float ERROR = 0x1.47ae140000000p-8f;
float HLICE = 0x1.45af000000000p+18f;
float GS = 0x1.39eb860000000p+3f;
float DICE = 0x1.cc00000000000p+9f;
float DH2O = 0x1.f400000000000p+9f;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:1482
BX = BEXP; // module_sf_noahlsm.F:1497
if ((BEXP > BLIM)) { BX = BLIM; } // module_sf_noahlsm.F:1502
NLOG = 0; // module_sf_noahlsm.F:1503
KCOUNT = 0; // module_sf_noahlsm.F:1508
if ((TKELV > SU(T0,0x1.0624de0000000p-10f))) { // module_sf_noahlsm.F:1510
FREE = SMC; // module_sf_noahlsm.F:1511
} else { // module_sf_noahlsm.F:1512
if ((CK != 0x0.0p+0f)) { // module_sf_noahlsm.F:1520
SWL = SU(SMC,SH2O); // module_sf_noahlsm.F:1521
if ((SWL > SU(SMC,0x1.47ae140000000p-6f))) { SWL = SU(SMC,0x1.47ae140000000p-6f); } // module_sf_noahlsm.F:1525
if ((SWL < 0x0.0p+0f)) { SWL = 0x0.0p+0f; } // module_sf_noahlsm.F:1530
while ((NLOG < 10) && (KCOUNT == 0)) {
NLOG = (NLOG + 1); // module_sf_noahlsm.F:1533
DF = SU(r_log(MU(MU(DV(MU(PSIS,GS),HLICE),MU(AD(0x1.0000000000000p+0f,MU(CK,SWL)),AD(0x1.0000000000000p+0f,MU(CK,SWL)))),r_pow(DV(SMCMAX,SU(SMC,SWL)),BX))),r_log(DV((-SU(TKELV,T0)),TKELV))); // module_sf_noahlsm.F:1534; strict GNU folds real exponent 2 into multiplication
DENOM = AD(DV(MU(0x1.0000000000000p+1f,CK),AD(0x1.0000000000000p+0f,MU(CK,SWL))),DV(BX,SU(SMC,SWL))); // module_sf_noahlsm.F:1537
SWLK = SU(SWL,DV(DF,DENOM)); // module_sf_noahlsm.F:1538
if ((SWLK > SU(SMC,0x1.47ae140000000p-6f))) { SWLK = SU(SMC,0x1.47ae140000000p-6f); } // module_sf_noahlsm.F:1542
if ((SWLK < 0x0.0p+0f)) { SWLK = 0x0.0p+0f; } // module_sf_noahlsm.F:1543
DSWL = noah_abs(SU(SWLK,SWL)); // module_sf_noahlsm.F:1548
SWL = SWLK; // module_sf_noahlsm.F:1554
if ((DSWL <= ERROR)) { // module_sf_noahlsm.F:1555
KCOUNT = (KCOUNT + 1); // module_sf_noahlsm.F:1556
} // module_sf_noahlsm.F:1557
}
FREE = SU(SMC,SWL); // module_sf_noahlsm.F:1566
} // module_sf_noahlsm.F:1567
if ((KCOUNT == 0)) { // module_sf_noahlsm.F:1576
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:1577
FK = MU(r_pow(MU(DV(HLICE,MU(GS,(-PSIS))),DV(SU(TKELV,T0),TKELV)),DV(-1,BX)),SMCMAX); // module_sf_noahlsm.F:1578
if ((FK < 0x1.47ae140000000p-6f)) { FK = 0x1.47ae140000000p-6f; } // module_sf_noahlsm.F:1581
FREE = noah_min(FK,SMC); // module_sf_noahlsm.F:1582
} // module_sf_noahlsm.F:1586
} // module_sf_noahlsm.F:1587
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1592-1854
static __attribute__((noinline)) void noah_hrt(thread NoahTables & tables, thread float * RHSTS, thread float * STC, thread float * SMC, thread float & SMCMAX, thread int & NSOIL, thread float * ZSOIL, thread float & YY, thread float & ZZ1, thread float & TBOT, thread float & ZBOT, thread float & PSISAT, thread float * SH2O, thread float & DT, thread float & BEXP, thread int & SOILTYP, thread int & OPT_THCND, thread float & F1, thread float & DF1, thread float & QUARTZ, thread float & CSOIL, thread float * AI, thread float * BI, thread float * CI, thread int & VEGTYP, thread int & ISURBAN, thread float & HCPCT_FASDAS){
bool ITAVG;
int I;
int K;
float DDZ;
float DDZ2;
float DENOM;
float DF1N;
float DF1K;
float DTSDZ;
float DTSDZ2;
float HCPCT;
float QTOT;
float SSOIL;
float SICE;
float TAVG;
float TBK;
float TBK1;
float TSNSR;
float TSURF;
float CSOIL_LOC;
float T0 = 0x1.1126660000000p+8f;
float CAIR = 0x1.f600000000000p+9f;
float CICE = 0x1.0114800000000p+21f;
float CH2O = 0x1.0059000000000p+22f;
 // module_sf_noahlsm.F:1604
if ((VEGTYP == ISURBAN)) { // module_sf_noahlsm.F:1632
CSOIL_LOC = 0x1.6e36000000000p+21f; // module_sf_noahlsm.F:1633
} else { // module_sf_noahlsm.F:1634
CSOIL_LOC = CSOIL; // module_sf_noahlsm.F:1635
} // module_sf_noahlsm.F:1636
ITAVG = true; // module_sf_noahlsm.F:1641
HCPCT = AD(AD(AD(MU(SH2O[(1)-(1)],CH2O),MU(SU(0x1.0000000000000p+0f,SMCMAX),CSOIL_LOC)),MU(SU(SMCMAX,SMC[(1)-(1)]),CAIR)),MU(SU(SMC[(1)-(1)],SH2O[(1)-(1)]),CICE)); // module_sf_noahlsm.F:1647
HCPCT_FASDAS = HCPCT; // module_sf_noahlsm.F:1653
DDZ = DV(0x1.0000000000000p+0f,MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:1660
AI[(1)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1661
CI[(1)-(1)] = DV(MU(DF1,DDZ),MU(ZSOIL[(1)-(1)],HCPCT)); // module_sf_noahlsm.F:1662
BI[(1)-(1)] = AD((-CI[(1)-(1)]),DV(DF1,MU(MU(MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZSOIL[(1)-(1)]),HCPCT),ZZ1))); // module_sf_noahlsm.F:1670
DTSDZ = DV(SU(STC[(1)-(1)],STC[(2)-(1)]),MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:1672
SSOIL = DV(MU(DF1,SU(STC[(1)-(1)],YY)),MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZZ1)); // module_sf_noahlsm.F:1673
DENOM = MU(ZSOIL[(1)-(1)],HCPCT); // module_sf_noahlsm.F:1675
RHSTS[(1)-(1)] = DV(SU(MU(DF1,DTSDZ),SSOIL),DENOM); // module_sf_noahlsm.F:1683
QTOT = MU(MU(-0x1.0000000000000p+0f,RHSTS[(1)-(1)]),DENOM); // module_sf_noahlsm.F:1688
SICE = SU(SMC[(1)-(1)],SH2O[(1)-(1)]); // module_sf_noahlsm.F:1700
if (ITAVG) { // module_sf_noahlsm.F:1701
TSURF = DV(AD(YY,MU(SU(ZZ1,1),STC[(1)-(1)])),ZZ1); // module_sf_noahlsm.F:1702
{ int tmp0_4=1; noah_tbnd(tables,STC[(1)-(1)],STC[(2)-(1)],ZSOIL,ZBOT,tmp0_4,NSOIL,TBK); if(tables.error)return; } // module_sf_noahlsm.F:1709
if (((((SICE > 0x0.0p+0f) || (STC[(1)-(1)] < T0)) || (TSURF < T0)) || (TBK < T0))) { // module_sf_noahlsm.F:1710
{ int tmp1_6=1; noah_tmpavg(tables,TAVG,TSURF,STC[(1)-(1)],TBK,ZSOIL,NSOIL,tmp1_6); if(tables.error)return; } // module_sf_noahlsm.F:1713
{ int tmp2_10=1; noah_snksrc(tables,TSNSR,TAVG,SMC[(1)-(1)],SH2O[(1)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,tmp2_10,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1714
RHSTS[(1)-(1)] = SU(RHSTS[(1)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1717
} // module_sf_noahlsm.F:1718
} else { // module_sf_noahlsm.F:1719
if (((SICE > 0x0.0p+0f) || (STC[(1)-(1)] < T0))) { // module_sf_noahlsm.F:1721
{ int tmp3_10=1; noah_snksrc(tables,TSNSR,STC[(1)-(1)],SMC[(1)-(1)],SH2O[(1)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,tmp3_10,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1722
RHSTS[(1)-(1)] = SU(RHSTS[(1)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1725
} // module_sf_noahlsm.F:1726
} // module_sf_noahlsm.F:1730
DDZ2 = 0x0.0p+0f; // module_sf_noahlsm.F:1735
DF1K = DF1; // module_sf_noahlsm.F:1736
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1744
HCPCT = AD(AD(AD(MU(SH2O[(K)-(1)],CH2O),MU(SU(0x1.0000000000000p+0f,SMCMAX),CSOIL_LOC)),MU(SU(SMCMAX,SMC[(K)-(1)]),CAIR)),MU(SU(SMC[(K)-(1)],SH2O[(K)-(1)]),CICE)); // module_sf_noahlsm.F:1745
if ((K != NSOIL)) { // module_sf_noahlsm.F:1752
{  noah_tdfcnd(tables,DF1N,SMC[(K)-(1)],QUARTZ,SMCMAX,SH2O[(K)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:1757
if ((VEGTYP == ISURBAN)) { DF1N = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:1760
DENOM = MU(0x1.0000000000000p-1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm.F:1762
DTSDZ2 = DV(SU(STC[(K)-(1)],STC[((K + 1))-(1)]),DENOM); // module_sf_noahlsm.F:1767
DDZ2 = DV(0x1.0000000000000p+1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm.F:1768
CI[(K)-(1)] = DV(MU((-DF1N),DDZ2),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm.F:1774
if (ITAVG) { // module_sf_noahlsm.F:1776
{  noah_tbnd(tables,STC[(K)-(1)],STC[((K + 1))-(1)],ZSOIL,ZBOT,K,NSOIL,TBK1); if(tables.error)return; } // module_sf_noahlsm.F:1777
} // module_sf_noahlsm.F:1778
} else { // module_sf_noahlsm.F:1780
{  noah_tdfcnd(tables,DF1N,SMC[(K)-(1)],QUARTZ,SMCMAX,SH2O[(K)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:1789
if ((VEGTYP == ISURBAN)) { DF1N = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:1793
DENOM = SU(MU(0x1.0000000000000p-1f,AD(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)])),ZBOT); // module_sf_noahlsm.F:1795
DTSDZ2 = DV(SU(STC[(K)-(1)],TBOT),DENOM); // module_sf_noahlsm.F:1800
CI[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1806
if (ITAVG) { // module_sf_noahlsm.F:1807
{  noah_tbnd(tables,STC[(K)-(1)],TBOT,ZSOIL,ZBOT,K,NSOIL,TBK1); if(tables.error)return; } // module_sf_noahlsm.F:1808
} // module_sf_noahlsm.F:1809
} // module_sf_noahlsm.F:1812
DENOM = MU(SU(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)]),HCPCT); // module_sf_noahlsm.F:1816
RHSTS[(K)-(1)] = DV(SU(MU(DF1N,DTSDZ2),MU(DF1K,DTSDZ)),DENOM); // module_sf_noahlsm.F:1817
QTOT = MU(MU(-0x1.0000000000000p+0f,DENOM),RHSTS[(K)-(1)]); // module_sf_noahlsm.F:1818
SICE = SU(SMC[(K)-(1)],SH2O[(K)-(1)]); // module_sf_noahlsm.F:1820
if (ITAVG) { // module_sf_noahlsm.F:1821
{  noah_tmpavg(tables,TAVG,TBK,STC[(K)-(1)],TBK1,ZSOIL,NSOIL,K); if(tables.error)return; } // module_sf_noahlsm.F:1822
if (((((SICE > 0x0.0p+0f) || (STC[(K)-(1)] < T0)) || (TBK < T0)) || (TBK1 < T0))) { // module_sf_noahlsm.F:1824
{  noah_snksrc(tables,TSNSR,TAVG,SMC[(K)-(1)],SH2O[(K)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,K,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1826
RHSTS[(K)-(1)] = SU(RHSTS[(K)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1828
} // module_sf_noahlsm.F:1829
} else { // module_sf_noahlsm.F:1830
if (((SICE > 0x0.0p+0f) || (STC[(K)-(1)] < T0))) { // module_sf_noahlsm.F:1832
{  noah_snksrc(tables,TSNSR,STC[(K)-(1)],SMC[(K)-(1)],SH2O[(K)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,K,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1833
RHSTS[(K)-(1)] = SU(RHSTS[(K)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1835
} // module_sf_noahlsm.F:1836
} // module_sf_noahlsm.F:1837
AI[(K)-(1)] = DV(MU((-DF1K),DDZ),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm.F:1842
BI[(K)-(1)] = (-AD(AI[(K)-(1)],CI[(K)-(1)])); // module_sf_noahlsm.F:1847
TBK = TBK1; // module_sf_noahlsm.F:1848
DF1K = DF1N; // module_sf_noahlsm.F:1849
DTSDZ = DTSDZ2; // module_sf_noahlsm.F:1850
DDZ = DDZ2; // module_sf_noahlsm.F:1851
} // module_sf_noahlsm.F:1852
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1857-1905
static __attribute__((noinline)) void noah_hstep(thread NoahTables & tables, thread float * STCOUT, thread float * STCIN, thread float * RHSTS, thread float & DT, thread int & NSOIL, thread float * AI, thread float * BI, thread float * CI){
int K;
float RHSTSIN[20];
float CIIN[20];
 // module_sf_noahlsm.F:1864
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1879
RHSTS[(K)-(1)] = MU(RHSTS[(K)-(1)],DT); // module_sf_noahlsm.F:1880
AI[(K)-(1)] = MU(AI[(K)-(1)],DT); // module_sf_noahlsm.F:1881
BI[(K)-(1)] = AD(0x1.0000000000000p+0f,MU(BI[(K)-(1)],DT)); // module_sf_noahlsm.F:1882
CI[(K)-(1)] = MU(CI[(K)-(1)],DT); // module_sf_noahlsm.F:1883
} // module_sf_noahlsm.F:1884
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1888
RHSTSIN[(K)-(1)] = RHSTS[(K)-(1)]; // module_sf_noahlsm.F:1889
} // module_sf_noahlsm.F:1890
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1891
CIIN[(K)-(1)] = CI[(K)-(1)]; // module_sf_noahlsm.F:1892
} // module_sf_noahlsm.F:1893
{  noah_rosr12(tables,CI,AI,BI,CIIN,RHSTSIN,RHSTS,NSOIL); if(tables.error)return; } // module_sf_noahlsm.F:1897
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1901
STCOUT[(K)-(1)] = AD(STCIN[(K)-(1)],CI[(K)-(1)]); // module_sf_noahlsm.F:1902
} // module_sf_noahlsm.F:1903
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1908-2196
static __attribute__((noinline)) void noah_nopac(thread NoahTables & tables, thread float & ETP, thread float & ETA, thread float & PRCP, thread float * SMC, thread float & SMCMAX, thread float & SMCWLT, thread float & SMCREF, thread float & SMCDRY, thread float & CMC, thread float & CMCMAX, thread int & NSOIL, thread float & DT, thread float & SHDFAC, thread float & SBETA, thread float & Q2, thread float & T1, thread float & SFCTMP, thread float & T24, thread float & TH2, thread float & FDOWN, thread float & F1, thread float & EMISSI, thread float & SSOIL, thread float * STC, thread float & EPSCA, thread float & BEXP, thread float & PC, thread float & RCH, thread float & RR, thread float & CFACTR, thread float * SH2O, thread float & SLOPE, thread float & KDT, thread float & FRZFACT, thread float & PSISAT, thread float * ZSOIL, thread float & DKSAT, thread float & DWSAT, thread float & TBOT, thread float & ZBOT, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & EDIR, thread float & EC, thread float * ET, thread float & ETT, thread int & NROOT, thread float * RTDIS, thread float & QUARTZ, thread float & FXEXP, thread float & CSOIL, thread float & BETA, thread float & DRIP, thread float & DEW, thread float & FLX1, thread float & FLX3, thread int & VEGTYP, thread int & ISURBAN, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & ETPND1, thread int & SOILTYP, thread int & OPT_THCND, thread float & XSDA_QFX, thread float & QFX_PHY, thread float & XQNORM, thread int & FASDAS, thread float & HCPCT_FASDAS, thread float & IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL){
thread auto& SIGMA=tables.SIGMA;
thread auto& CPH2O=tables.CPH2O;
int K;
float ET1[20];
float EC1;
float EDIR1;
float ETT1;
float DF1;
float ETA1;
float ETP1;
float PRCP1;
float YY;
float YYNUM;
float ZZ1;
float EFT[20];
float WETTY[20];
float EFDIR;
float EFC;
float EALL_NOW;
 // module_sf_noahlsm.F:1929
PRCP1 = MU(PRCP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:1969
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:1970
DEW = 0x0.0p+0f; // module_sf_noahlsm.F:1971
QFX_PHY = 0x0.0p+0f; // module_sf_noahlsm.F:1978
EDIR = 0x0.0p+0f; // module_sf_noahlsm.F:1982
EDIR1 = 0x0.0p+0f; // module_sf_noahlsm.F:1983
EC1 = 0x0.0p+0f; // module_sf_noahlsm.F:1984
EC = 0x0.0p+0f; // module_sf_noahlsm.F:1985
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1986
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1987
ET1[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1988
WETTY[(K)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:1992
} // module_sf_noahlsm.F:1996
ETT = 0x0.0p+0f; // module_sf_noahlsm.F:1997
ETT1 = 0x0.0p+0f; // module_sf_noahlsm.F:1998
ETPND1 = 0x0.0p+0f; // module_sf_noahlsm.F:2001
if ((ETP > 0x0.0p+0f)) { // module_sf_noahlsm.F:2004
{  noah_evapo(tables,ETA1,SMC,NSOIL,CMC,ETP1,DT,ZSOIL,SH2O,SMCMAX,BEXP,PC,SMCWLT,DKSAT,DWSAT,SMCREF,SHDFAC,CMCMAX,SMCDRY,CFACTR,EDIR1,EC1,ET1,ETT1,SFCTMP,Q2,NROOT,RTDIS,FXEXP,SFHEAD1RT,ETPND1); if(tables.error)return; } // module_sf_noahlsm.F:2005
if ((FASDAS == 1)) { // module_sf_noahlsm.F:2015
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2016
QFX_PHY = AD(QFX_PHY,ET1[(K)-(1)]); // module_sf_noahlsm.F:2017
if (((SMC[(K)-(1)] >= SMCREF) && (XSDA_QFX > 0x0.0p+0f))) { WETTY[(K)-(1)] = 0x0.0p+0f; } // module_sf_noahlsm.F:2019
} // module_sf_noahlsm.F:2020
QFX_PHY = AD(AD(EDIR1,EC1),QFX_PHY); // module_sf_noahlsm.F:2021
EALL_NOW = QFX_PHY; // module_sf_noahlsm.F:2022
QFX_PHY = MU(QFX_PHY,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2023
if ((EALL_NOW != 0x0.0p+0f)) { // module_sf_noahlsm.F:2025
EFDIR = MU(MU(MU(DV(EDIR1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2026
EFDIR = MU(EFDIR,WETTY[(1)-(1)]); // module_sf_noahlsm.F:2027
EDIR1 = AD(EDIR1,EFDIR); // module_sf_noahlsm.F:2029
EFC = MU(MU(MU(DV(EC1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2031
EC1 = AD(EC1,EFC); // module_sf_noahlsm.F:2033
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2036
EFT[(K)-(1)] = MU(MU(MU(DV(ET1[(K)-(1)],EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2037
EFT[(K)-(1)] = MU(EFT[(K)-(1)],WETTY[(K)-(1)]); // module_sf_noahlsm.F:2038
ET1[(K)-(1)] = AD(ET1[(K)-(1)],EFT[(K)-(1)]); // module_sf_noahlsm.F:2040
} // module_sf_noahlsm.F:2041
} // module_sf_noahlsm.F:2044
} else { // module_sf_noahlsm.F:2045
QFX_PHY = 0x0.0p+0f; // module_sf_noahlsm.F:2046
} // module_sf_noahlsm.F:2047
{  noah_smflx(tables,SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,EDIR1,EC1,ET1,DRIP,SFHEAD1RT,INFXS1RT,IRRIGATION_CHANNEL,present_IRRIGATION_CHANNEL); if(tables.error)return; } // module_sf_noahlsm.F:2051
ETA = MU(ETA1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2063
} else { // module_sf_noahlsm.F:2069
DEW = (-ETP1); // module_sf_noahlsm.F:2070
PRCP1 = AD(PRCP1,DEW); // module_sf_noahlsm.F:2076
if ((FASDAS == 1)) { // module_sf_noahlsm.F:2080
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2081
QFX_PHY = AD(QFX_PHY,ET1[(K)-(1)]); // module_sf_noahlsm.F:2082
if (((SMC[(K)-(1)] >= SMCREF) && (XSDA_QFX > 0x0.0p+0f))) { WETTY[(K)-(1)] = 0x0.0p+0f; } // module_sf_noahlsm.F:2084
} // module_sf_noahlsm.F:2085
QFX_PHY = AD(AD(EDIR1,EC1),QFX_PHY); // module_sf_noahlsm.F:2086
EALL_NOW = QFX_PHY; // module_sf_noahlsm.F:2087
QFX_PHY = MU(QFX_PHY,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2088
if ((EALL_NOW != 0x0.0p+0f)) { // module_sf_noahlsm.F:2090
EFDIR = MU(MU(MU(DV(EDIR1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2091
EFDIR = MU(EFDIR,WETTY[(1)-(1)]); // module_sf_noahlsm.F:2092
EDIR1 = AD(EDIR1,EFDIR); // module_sf_noahlsm.F:2094
EFC = MU(MU(MU(DV(EC1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2096
EC1 = AD(EC1,EFC); // module_sf_noahlsm.F:2098
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2100
EFT[(K)-(1)] = MU(MU(MU(DV(ET1[(K)-(1)],EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2101
EFT[(K)-(1)] = MU(EFT[(K)-(1)],WETTY[(K)-(1)]); // module_sf_noahlsm.F:2102
ET1[(K)-(1)] = AD(ET1[(K)-(1)],EFT[(K)-(1)]); // module_sf_noahlsm.F:2104
} // module_sf_noahlsm.F:2105
} // module_sf_noahlsm.F:2107
} else { // module_sf_noahlsm.F:2108
QFX_PHY = 0x0.0p+0f; // module_sf_noahlsm.F:2109
} // module_sf_noahlsm.F:2110
{  noah_smflx(tables,SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,EDIR1,EC1,ET1,DRIP,SFHEAD1RT,INFXS1RT,IRRIGATION_CHANNEL,present_IRRIGATION_CHANNEL); if(tables.error)return; } // module_sf_noahlsm.F:2114
} // module_sf_noahlsm.F:2126
if ((ETP <= 0x0.0p+0f)) { // module_sf_noahlsm.F:2132
BETA = 0x0.0p+0f; // module_sf_noahlsm.F:2133
ETA = ETP; // module_sf_noahlsm.F:2134
if ((ETP < 0x0.0p+0f)) { // module_sf_noahlsm.F:2135
BETA = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:2136
} // module_sf_noahlsm.F:2137
} else { // module_sf_noahlsm.F:2138
BETA = DV(ETA,ETP); // module_sf_noahlsm.F:2139
} // module_sf_noahlsm.F:2140
EDIR = MU(EDIR1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2145
EC = MU(EC1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2146
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2147
ET[(K)-(1)] = MU(ET1[(K)-(1)],0x1.f400000000000p+9f); // module_sf_noahlsm.F:2148
} // module_sf_noahlsm.F:2149
ETT = MU(ETT1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2150
{  noah_tdfcnd(tables,DF1,SMC[(1)-(1)],QUARTZ,SMCMAX,SH2O[(1)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:2158
if ((VEGTYP == ISURBAN)) { DF1 = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:2161
DF1 = MU(DF1,r_exp(MU(SBETA,SHDFAC))); // module_sf_noahlsm.F:2171
YYNUM = SU(FDOWN,MU(MU(EMISSI,SIGMA),T24)); // module_sf_noahlsm.F:2176
YY = AD(SFCTMP,DV(SU(SU(AD(DV(YYNUM,RCH),TH2),SFCTMP),MU(BETA,EPSCA)),RR)); // module_sf_noahlsm.F:2177
ZZ1 = AD(DV(DF1,MU(MU(MU(-0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),RCH),RR)),0x1.0000000000000p+0f); // module_sf_noahlsm.F:2179
{  noah_shflx(tables,SSOIL,STC,SMC,SMCMAX,NSOIL,T1,DT,YY,ZZ1,ZSOIL,TBOT,ZBOT,SMCWLT,PSISAT,SH2O,BEXP,F1,DF1,QUARTZ,CSOIL,VEGTYP,ISURBAN,SOILTYP,OPT_THCND,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:2182
FLX1 = MU(MU(CPH2O,PRCP),SU(T1,SFCTMP)); // module_sf_noahlsm.F:2192
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:2193
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2199-2316
static __attribute__((noinline)) void noah_penman(thread NoahTables & tables, thread float & SFCTMP, thread float & SFCPRS, thread float & CH, thread float & T2V, thread float & TH2, thread float & PRCP, thread float & FDOWN, thread float & T24, thread float & SSOIL, thread float & Q2, thread float & Q2SAT, thread float & ETP, thread float & RCH, thread float & EPSCA, thread float & RR, thread bool & SNOWNG, thread bool & FRZGRA, thread float & DQSDT2, thread float & FLX2, thread float & EMISSI_IN, thread float & SNEQV, thread float & T1, thread float & SNCOVR, thread float & AOASIS, thread float & ALBEDO, thread float & SOLDN, thread float & FVB, thread float & GAMA, thread float & STC1, thread float & ETPN, thread float & FLX4, thread bool & UA_PHYS){
thread auto& RD=tables.RD;
thread auto& SIGMA=tables.SIGMA;
thread auto& CPH2O=tables.CPH2O;
thread auto& CPICE=tables.CPICE;
thread auto& LSUBF=tables.LSUBF;
float A;
float DELTA;
float FNET;
float RAD;
float RHO;
float EMISSI;
float ELCP1;
float LVS;
float TOTABS;
float UCABS;
float SIGNCK;
float FNETN;
float RADN;
float EPSCAN;
float ELCP = 0x1.37199a0000000p+11f;
float LSUBC = 0x1.314c400000000p+21f;
float CP = 0x1.f64ccc0000000p+9f;
float LSUBS = 0x1.5975800000000p+21f;
float ALGDSN = 0x1.0000000000000p-1f;
float ALVGSN = 0x1.0a3d700000000p-3f;
 // module_sf_noahlsm.F:2211
EMISSI = EMISSI_IN; // module_sf_noahlsm.F:2235
ELCP1 = AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),ELCP),DV(MU(MU(SNCOVR,ELCP),LSUBS),LSUBC)); // module_sf_noahlsm.F:2236
LVS = AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LSUBC),MU(SNCOVR,LSUBS)); // module_sf_noahlsm.F:2237
FLX2 = 0x0.0p+0f; // module_sf_noahlsm.F:2239
DELTA = MU(ELCP1,DQSDT2); // module_sf_noahlsm.F:2241
T24 = MU(MU(MU(SFCTMP,SFCTMP),SFCTMP),SFCTMP); // module_sf_noahlsm.F:2242
RR = AD(DV(MU(MU(EMISSI,T24),0x1.16505a0000000p-24f),MU(SFCPRS,CH)),0x1.0000000000000p+0f); // module_sf_noahlsm.F:2244
RHO = DV(SFCPRS,MU(RD,T2V)); // module_sf_noahlsm.F:2245
RCH = MU(MU(RHO,CP),CH); // module_sf_noahlsm.F:2251
if ((!SNOWNG)) { // module_sf_noahlsm.F:2252
if ((PRCP > 0x0.0p+0f)) { RR = AD(RR,DV(MU(CPH2O,PRCP),RCH)); } // module_sf_noahlsm.F:2253
} else { // module_sf_noahlsm.F:2254
RR = AD(RR,DV(MU(CPICE,PRCP),RCH)); // module_sf_noahlsm.F:2255
} // module_sf_noahlsm.F:2256
FNET = SU(SU(FDOWN,MU(MU(EMISSI,SIGMA),T24)),SSOIL); // module_sf_noahlsm.F:2263
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:2265
if (UA_PHYS) { // module_sf_noahlsm.F:2266
if ((((SNEQV > 0x0.0p+0f) && (FNET > 0x0.0p+0f)) && (SOLDN > 0x0.0p+0f))) { // module_sf_noahlsm.F:2267
TOTABS = MU(MU(SU(0x1.0000000000000p+0f,ALBEDO),SOLDN),FVB); // module_sf_noahlsm.F:2268
UCABS = noah_min(TOTABS,MU(MU(MU(MU(SU(0x1.0000000000000p+0f,ALGDSN),SU(0x1.0000000000000p+0f,ALVGSN)),SOLDN),GAMA),FVB)); // module_sf_noahlsm.F:2270
FLX4 = noah_min(SU(TOTABS,UCABS),noah_min(0x1.f400000000000p+7f,MU(MU(0x1.0000000000000p-1f,SU(0x1.0000000000000p+0f,ALBEDO)),SOLDN))); // module_sf_noahlsm.F:2275
} // module_sf_noahlsm.F:2276
SIGNCK = MU(SU(STC1,0x1.1126660000000p+8f),SU(SFCTMP,0x1.1126660000000p+8f)); // module_sf_noahlsm.F:2278
if (((FLX4 > 0x0.0p+0f) && ((SIGNCK <= 0x0.0p+0f) || (STC1 < 0x1.1126660000000p+8f)))) { // module_sf_noahlsm.F:2280
if ((FNET >= FLX4)) { // module_sf_noahlsm.F:2281
FNETN = SU(FNET,FLX4); // module_sf_noahlsm.F:2282
} else { // module_sf_noahlsm.F:2283
FLX4 = FNET; // module_sf_noahlsm.F:2284
FNETN = 0x0.0p+0f; // module_sf_noahlsm.F:2285
} // module_sf_noahlsm.F:2286
} else { // module_sf_noahlsm.F:2287
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:2288
FNETN = 0x0.0p+0f; // module_sf_noahlsm.F:2289
} // module_sf_noahlsm.F:2290
} // module_sf_noahlsm.F:2291
if (FRZGRA) { // module_sf_noahlsm.F:2293
FLX2 = MU((-LSUBF),PRCP); // module_sf_noahlsm.F:2294
FNET = SU(FNET,FLX2); // module_sf_noahlsm.F:2295
if (UA_PHYS) { FNETN = SU(FNETN,FLX2); } // module_sf_noahlsm.F:2296
} // module_sf_noahlsm.F:2300
RAD = SU(AD(DV(FNET,RCH),TH2),SFCTMP); // module_sf_noahlsm.F:2301
A = MU(ELCP1,SU(Q2SAT,Q2)); // module_sf_noahlsm.F:2303
EPSCA = DV(AD(MU(A,RR),MU(RAD,DELTA)),AD(DELTA,RR)); // module_sf_noahlsm.F:2304
if ((EPSCA > 0x0.0p+0f)) { EPSCA = MU(EPSCA,AOASIS); } // module_sf_noahlsm.F:2306
ETP = DV(MU(EPSCA,RCH),LVS); // module_sf_noahlsm.F:2308
if (UA_PHYS) { // module_sf_noahlsm.F:2310
RADN = SU(AD(DV(FNETN,RCH),TH2),SFCTMP); // module_sf_noahlsm.F:2311
EPSCAN = DV(AD(MU(A,RR),MU(RADN,DELTA)),AD(DELTA,RR)); // module_sf_noahlsm.F:2312
ETPN = DV(MU(EPSCAN,RCH),LVS); // module_sf_noahlsm.F:2313
} // module_sf_noahlsm.F:2314
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2319-2539
static __attribute__((noinline)) void noah_redprm(thread NoahTables & tables, thread int & VEGTYP, thread int & SOILTYP, thread int & SLOPETYP, thread float & CFACTR, thread float & CMCMAX, thread float & RSMAX, thread float & TOPT, thread float & REFKDT, thread float & KDT, thread float & SBETA, thread float & SHDFAC, thread float & RSMIN, thread float & RGL, thread float & HS, thread float & ZBOT, thread float & FRZX, thread float & PSISAT, thread float & SLOPE, thread float & SNUP, thread float & SALP, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SMCMAX, thread float & SMCWLT, thread float & SMCREF, thread float & SMCDRY, thread float & F1, thread float & QUARTZ, thread float & FXEXP, thread float * RTDIS, thread float * SLDPTH, thread float * ZSOIL, thread int & NROOT, thread int & NSOIL, thread float & CZIL, thread float & LAIMIN, thread float & LAIMAX, thread float & EMISSMIN, thread float & EMISSMAX, thread float & ALBEDOMIN, thread float & ALBEDOMAX, thread float & Z0MIN, thread float & Z0MAX, thread float & CSOIL, thread float & PTU, thread const char * LLANDUSE, thread const char * LSOIL, thread bool & LOCAL, thread float & LVCOEF, thread float & ZTOPV, thread float & ZBOTV){
thread auto& LUCATS=tables.LUCATS;
thread auto& BARE=tables.BARE;
thread auto& NROTBL=tables.NROTBL;
thread auto& SNUPTBL=tables.SNUPTBL;
thread auto& RSTBL=tables.RSTBL;
thread auto& RGLTBL=tables.RGLTBL;
thread auto& HSTBL=tables.HSTBL;
thread auto& EMISSMINTBL=tables.EMISSMINTBL;
thread auto& EMISSMAXTBL=tables.EMISSMAXTBL;
thread auto& LAIMINTBL=tables.LAIMINTBL;
thread auto& LAIMAXTBL=tables.LAIMAXTBL;
thread auto& Z0MINTBL=tables.Z0MINTBL;
thread auto& Z0MAXTBL=tables.Z0MAXTBL;
thread auto& ALBEDOMINTBL=tables.ALBEDOMINTBL;
thread auto& ALBEDOMAXTBL=tables.ALBEDOMAXTBL;
thread auto& ZTOPVTBL=tables.ZTOPVTBL;
thread auto& ZBOTVTBL=tables.ZBOTVTBL;
thread auto& TOPT_DATA=tables.TOPT_DATA;
thread auto& CMCMAX_DATA=tables.CMCMAX_DATA;
thread auto& CFACTR_DATA=tables.CFACTR_DATA;
thread auto& RSMAX_DATA=tables.RSMAX_DATA;
thread auto& SLCATS=tables.SLCATS;
thread auto& BB=tables.BB;
thread auto& DRYSMC=tables.DRYSMC;
thread auto& F11=tables.F11;
thread auto& MAXSMC=tables.MAXSMC;
thread auto& REFSMC=tables.REFSMC;
thread auto& SATPSI=tables.SATPSI;
thread auto& SATDK=tables.SATDK;
thread auto& SATDW=tables.SATDW;
thread auto& WLTSMC=tables.WLTSMC;
thread auto& QTZ=tables.QTZ;
thread auto& SLPCATS=tables.SLPCATS;
thread auto& SLOPE_DATA=tables.SLOPE_DATA;
thread auto& SBETA_DATA=tables.SBETA_DATA;
thread auto& FXEXP_DATA=tables.FXEXP_DATA;
thread auto& CSOIL_DATA=tables.CSOIL_DATA;
thread auto& SALP_DATA=tables.SALP_DATA;
thread auto& REFDK_DATA=tables.REFDK_DATA;
thread auto& REFKDT_DATA=tables.REFKDT_DATA;
thread auto& FRZK_DATA=tables.FRZK_DATA;
thread auto& ZBOT_DATA=tables.ZBOT_DATA;
thread auto& CZIL_DATA=tables.CZIL_DATA;
thread auto& LVCOEF_DATA=tables.LVCOEF_DATA;
thread const char * ERR_MESSAGE=tables.ERR_MESSAGE;
int MAX_SLOPETYP = 30;
int MAX_SOILTYP = 30;
int MAX_VEGTYP = 30;
int I;
float FRZFACT;
float FRZK;
float REFDK;
 // module_sf_noahlsm.F:2329
if ((SOILTYP > SLCATS)) { // module_sf_noahlsm.F:2437
tables.error = 2438; return; // module_sf_noahlsm.F:2438
} // module_sf_noahlsm.F:2439
if ((VEGTYP > LUCATS)) { // module_sf_noahlsm.F:2440
tables.error = 2441; return; // module_sf_noahlsm.F:2441
} // module_sf_noahlsm.F:2442
if ((SLOPETYP > SLPCATS)) { // module_sf_noahlsm.F:2443
tables.error = 2444; return; // module_sf_noahlsm.F:2444
} // module_sf_noahlsm.F:2445
if (SOILTYP>30 || VEGTYP>50 || SLOPETYP>30) { tables.error=-1; return; }
CSOIL = CSOIL_DATA; // module_sf_noahlsm.F:2450
BEXP = BB[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2451
DKSAT = SATDK[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2452
DWSAT = SATDW[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2453
F1 = F11[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2454
PSISAT = SATPSI[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2455
QUARTZ = QTZ[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2456
SMCDRY = DRYSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2457
SMCMAX = MAXSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2458
SMCREF = REFSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2459
SMCWLT = WLTSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2460
ZBOT = ZBOT_DATA; // module_sf_noahlsm.F:2465
SALP = SALP_DATA; // module_sf_noahlsm.F:2466
SBETA = SBETA_DATA; // module_sf_noahlsm.F:2467
REFDK = REFDK_DATA; // module_sf_noahlsm.F:2468
FRZK = FRZK_DATA; // module_sf_noahlsm.F:2469
FXEXP = FXEXP_DATA; // module_sf_noahlsm.F:2470
REFKDT = REFKDT_DATA; // module_sf_noahlsm.F:2471
PTU = 0x0.0p+0f; // module_sf_noahlsm.F:2472
KDT = DV(MU(REFKDT,DKSAT),REFDK); // module_sf_noahlsm.F:2473
CZIL = CZIL_DATA; // module_sf_noahlsm.F:2474
SLOPE = SLOPE_DATA[(SLOPETYP)-(1)]; // module_sf_noahlsm.F:2475
LVCOEF = LVCOEF_DATA; // module_sf_noahlsm.F:2476
FRZFACT = MU(DV(SMCMAX,SMCREF),0x1.c2bc2c0000000p-1f); // module_sf_noahlsm.F:2481
FRZX = MU(FRZK,FRZFACT); // module_sf_noahlsm.F:2482
TOPT = TOPT_DATA; // module_sf_noahlsm.F:2487
CMCMAX = CMCMAX_DATA; // module_sf_noahlsm.F:2488
CFACTR = CFACTR_DATA; // module_sf_noahlsm.F:2489
RSMAX = RSMAX_DATA; // module_sf_noahlsm.F:2490
NROOT = NROTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2491
SNUP = SNUPTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2492
RSMIN = RSTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2493
RGL = RGLTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2494
HS = HSTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2495
EMISSMIN = EMISSMINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2496
EMISSMAX = EMISSMAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2497
LAIMIN = LAIMINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2498
LAIMAX = LAIMAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2499
Z0MIN = Z0MINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2500
Z0MAX = Z0MAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2501
ALBEDOMIN = ALBEDOMINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2502
ALBEDOMAX = ALBEDOMAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2503
ZTOPV = ZTOPVTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2504
ZBOTV = ZBOTVTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2505
if ((VEGTYP == BARE)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahlsm.F:2507
if ((NROOT > NSOIL)) { // module_sf_noahlsm.F:2508
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:2509
tables.error = 2511; return; // module_sf_noahlsm.F:2511
} // module_sf_noahlsm.F:2516
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:2517
RTDIS[(I)-(1)] = DV((-SLDPTH[(I)-(1)]),ZSOIL[(NROOT)-(1)]); // module_sf_noahlsm.F:2518
} // module_sf_noahlsm.F:2522
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2541-2600
static __attribute__((noinline)) void noah_rosr12(thread NoahTables & tables, thread float * P, thread float * A, thread float * B, thread float * C, thread float * D, thread float * DELTA, thread int & NSOIL){
int K;
int KK;
 // module_sf_noahlsm.F:2561
C[(NSOIL)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:2572
P[(1)-(1)] = DV((-C[(1)-(1)]),B[(1)-(1)]); // module_sf_noahlsm.F:2573
DELTA[(1)-(1)] = DV(D[(1)-(1)],B[(1)-(1)]); // module_sf_noahlsm.F:2581
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2582
P[(K)-(1)] = MU((-C[(K)-(1)]),DV(0x1.0000000000000p+0f,AD(B[(K)-(1)],MU(A[(K)-(1)],P[((K - 1))-(1)])))); // module_sf_noahlsm.F:2583
DELTA[(K)-(1)] = MU(SU(D[(K)-(1)],MU(A[(K)-(1)],DELTA[((K - 1))-(1)])),DV(0x1.0000000000000p+0f,AD(B[(K)-(1)],MU(A[(K)-(1)],P[((K - 1))-(1)])))); // module_sf_noahlsm.F:2584
} // module_sf_noahlsm.F:2586
P[(NSOIL)-(1)] = DELTA[(NSOIL)-(1)]; // module_sf_noahlsm.F:2590
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2595
KK = ((NSOIL - K) + 1); // module_sf_noahlsm.F:2596
P[(KK)-(1)] = AD(MU(P[(KK)-(1)],P[((KK + 1))-(1)]),DELTA[(KK)-(1)]); // module_sf_noahlsm.F:2597
} // module_sf_noahlsm.F:2598
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2604-2670
static __attribute__((noinline)) void noah_shflx(thread NoahTables & tables, thread float & SSOIL, thread float * STC, thread float * SMC, thread float & SMCMAX, thread int & NSOIL, thread float & T1, thread float & DT, thread float & YY, thread float & ZZ1, thread float * ZSOIL, thread float & TBOT, thread float & ZBOT, thread float & SMCWLT, thread float & PSISAT, thread float * SH2O, thread float & BEXP, thread float & F1, thread float & DF1, thread float & QUARTZ, thread float & CSOIL, thread int & VEGTYP, thread int & ISURBAN, thread int & SOILTYP, thread int & OPT_THCND, thread float & HCPCT_FASDAS){
int I;
float AI[20];
float BI[20];
float CI[20];
float STCF[20];
float RHSTS[20];
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:2616
{  noah_hrt(tables,RHSTS,STC,SMC,SMCMAX,NSOIL,ZSOIL,YY,ZZ1,TBOT,ZBOT,PSISAT,SH2O,DT,BEXP,SOILTYP,OPT_THCND,F1,DF1,QUARTZ,CSOIL,AI,BI,CI,VEGTYP,ISURBAN,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:2645
{  noah_hstep(tables,STCF,STC,RHSTS,DT,NSOIL,AI,BI,CI); if(tables.error)return; } // module_sf_noahlsm.F:2650
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm.F:2652
STC[(I)-(1)] = STCF[(I)-(1)]; // module_sf_noahlsm.F:2653
} // module_sf_noahlsm.F:2654
T1 = DV(AD(YY,MU(SU(ZZ1,0x1.0000000000000p+0f),STC[(1)-(1)])),ZZ1); // module_sf_noahlsm.F:2666
SSOIL = DV(MU(DF1,SU(STC[(1)-(1)],T1)),MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:2667
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2673-2817
static __attribute__((noinline)) void noah_smflx(thread NoahTables & tables, thread float * SMC, thread int & NSOIL, thread float & CMC, thread float & DT, thread float & PRCP1, thread float * ZSOIL, thread float * SH2O, thread float & SLOPE, thread float & KDT, thread float & FRZFACT, thread float & SMCMAX, thread float & BEXP, thread float & SMCWLT, thread float & DKSAT, thread float & DWSAT, thread float & SHDFAC, thread float & CMCMAX, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & EDIR, thread float & EC, thread float * ET, thread float & DRIP, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL){
int I;
int K;
float AI[20];
float BI[20];
float CI[20];
float STCF[20];
float RHSTS[20];
float RHSTT[20];
float SICE[20];
float SH2OA[20];
float SH2OFG[20];
float SH2OIN[20];
float DUMMY;
float EXCESS;
float PCPDRP;
float RHSCT;
float TRHSCT;
float FAC2;
float FLIMIT;
 // module_sf_noahlsm.F:2691
DUMMY = 0x0.0p+0f; // module_sf_noahlsm.F:2716
RHSCT = SU(MU(SHDFAC,PRCP1),EC); // module_sf_noahlsm.F:2723
DRIP = 0x0.0p+0f; // module_sf_noahlsm.F:2724
TRHSCT = MU(DT,RHSCT); // module_sf_noahlsm.F:2725
EXCESS = AD(CMC,TRHSCT); // module_sf_noahlsm.F:2726
if ((EXCESS > CMCMAX)) { DRIP = SU(EXCESS,CMCMAX); } // module_sf_noahlsm.F:2732
PCPDRP = AD(MU(SU(0x1.0000000000000p+0f,SHDFAC),PRCP1),DV(DRIP,DT)); // module_sf_noahlsm.F:2733
if (present_IRRIGATION_CHANNEL) { // module_sf_noahlsm.F:2734
if ((IRRIGATION_CHANNEL != 0x0.0p+0f)) { PCPDRP = AD(PCPDRP,MU(0x1.0624de0000000p-10f,IRRIGATION_CHANNEL)); } // module_sf_noahlsm.F:2735
} // module_sf_noahlsm.F:2736
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm.F:2740
SICE[(I)-(1)] = SU(SMC[(I)-(1)],SH2O[(I)-(1)]); // module_sf_noahlsm.F:2741
} // module_sf_noahlsm.F:2742
FAC2 = 0x0.0p+0f; // module_sf_noahlsm.F:2766
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm.F:2767
FAC2 = noah_max(FAC2,DV(SH2O[(I)-(1)],SMCMAX)); // module_sf_noahlsm.F:2768
} // module_sf_noahlsm.F:2769
{  noah_fac2mit(tables,SMCMAX,FLIMIT); if(tables.error)return; } // module_sf_noahlsm.F:2770
if (((MU(PCPDRP,DT) > MU(MU(0x1.9999980000000p-4f,(-ZSOIL[(1)-(1)])),SMCMAX)) || (FAC2 > FLIMIT))) { // module_sf_noahlsm.F:2785
{  noah_srt(tables,RHSTT,EDIR,ET,SH2O,SH2O,NSOIL,PCPDRP,ZSOIL,DWSAT,DKSAT,SMCMAX,BEXP,RUNOFF1,RUNOFF2,DT,SMCWLT,SLOPE,KDT,FRZFACT,SICE,AI,BI,CI,SFHEAD1RT,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2787
{  noah_sstep(tables,SH2OFG,SH2O,DUMMY,RHSTT,RHSCT,DT,NSOIL,SMCMAX,CMCMAX,RUNOFF3,ZSOIL,SMC,SICE,AI,BI,CI,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2791
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2793
SH2OA[(K)-(1)] = MU(AD(SH2O[(K)-(1)],SH2OFG[(K)-(1)]),0x1.0000000000000p-1f); // module_sf_noahlsm.F:2794
} // module_sf_noahlsm.F:2795
{  noah_srt(tables,RHSTT,EDIR,ET,SH2O,SH2OA,NSOIL,PCPDRP,ZSOIL,DWSAT,DKSAT,SMCMAX,BEXP,RUNOFF1,RUNOFF2,DT,SMCWLT,SLOPE,KDT,FRZFACT,SICE,AI,BI,CI,SFHEAD1RT,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2796
for(int ac=1;ac<=NSOIL;++ac) SH2OIN[ac-(1)] = SH2O[ac-(1)]; // module_sf_noahlsm.F:2800
{  noah_sstep(tables,SH2O,SH2OIN,CMC,RHSTT,RHSCT,DT,NSOIL,SMCMAX,CMCMAX,RUNOFF3,ZSOIL,SMC,SICE,AI,BI,CI,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2801
} else { // module_sf_noahlsm.F:2804
{  noah_srt(tables,RHSTT,EDIR,ET,SH2O,SH2O,NSOIL,PCPDRP,ZSOIL,DWSAT,DKSAT,SMCMAX,BEXP,RUNOFF1,RUNOFF2,DT,SMCWLT,SLOPE,KDT,FRZFACT,SICE,AI,BI,CI,SFHEAD1RT,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2805
for(int ac=1;ac<=NSOIL;++ac) SH2OIN[ac-(1)] = SH2O[ac-(1)]; // module_sf_noahlsm.F:2809
{  noah_sstep(tables,SH2O,SH2OIN,CMC,RHSTT,RHSCT,DT,NSOIL,SMCMAX,CMCMAX,RUNOFF3,ZSOIL,SMC,SICE,AI,BI,CI,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2810
} // module_sf_noahlsm.F:2814
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2821-2923
static __attribute__((noinline)) void noah_snfrac(thread NoahTables & tables, thread float & SNEQV, thread float & SNUP, thread float & SALP, thread float & SNOWH, thread float & SNCOVR, thread float & XLAI, thread float & SHDFAC, thread float & FVB, thread float & GAMA, thread float & FBUR, thread float & FGSN, thread float & ZTOPV, thread float & ZBOTV, thread bool & UA_PHYS){
float RSNOW;
float Z0N;
float SNUPGRD = 0x1.47ae140000000p-6f;
 // module_sf_noahlsm.F:2834
if ((SNEQV < SNUP)) { // module_sf_noahlsm.F:2855
RSNOW = DV(SNEQV,SNUP); // module_sf_noahlsm.F:2856
SNCOVR = SU(0x1.0000000000000p+0f,SU(r_exp(MU((-SALP),RSNOW)),MU(RSNOW,r_exp((-SALP))))); // module_sf_noahlsm.F:2857
} else { // module_sf_noahlsm.F:2858
SNCOVR = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:2859
} // module_sf_noahlsm.F:2860
if (UA_PHYS) { // module_sf_noahlsm.F:2870
if ((SNEQV < SNUPGRD)) { // module_sf_noahlsm.F:2875
FGSN = DV(SNEQV,SNUPGRD); // module_sf_noahlsm.F:2876
} else { // module_sf_noahlsm.F:2877
FGSN = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:2878
} // module_sf_noahlsm.F:2879
if (((ZBOTV > 0x0.0p+0f) && (SNOWH > ZBOTV))) { // module_sf_noahlsm.F:2886
if ((ZBOTV <= 0x1.0000000000000p-1f)) { // module_sf_noahlsm.F:2887
FBUR = DV(SU(SNOWH,MU(0x1.99999a0000000p-2f,ZBOTV)),MU(0x1.99999a0000000p-2f,SU(ZTOPV,ZBOTV))); // module_sf_noahlsm.F:2888
} else { // module_sf_noahlsm.F:2889
FBUR = DV(SU(SNOWH,ZBOTV),SU(ZTOPV,ZBOTV)); // module_sf_noahlsm.F:2890
} // module_sf_noahlsm.F:2891
} else { // module_sf_noahlsm.F:2892
FBUR = 0x0.0p+0f; // module_sf_noahlsm.F:2893
} // module_sf_noahlsm.F:2894
FBUR = noah_min(noah_max(FBUR,0x0.0p+0f),0x1.0000000000000p+0f); // module_sf_noahlsm.F:2896
XLAI = MU(XLAI,SU(0x1.0000000000000p+0f,FBUR)); // module_sf_noahlsm.F:2899
FVB = MU(MU(SHDFAC,FGSN),SU(0x1.0000000000000p+0f,FBUR)); // module_sf_noahlsm.F:2909
GAMA = r_exp(MU(-0x1.0000000000000p+0f,XLAI)); // module_sf_noahlsm.F:2913
} else { // module_sf_noahlsm.F:2914
FVB = 0x0.0p+0f; // module_sf_noahlsm.F:2916
GAMA = 0x0.0p+0f; // module_sf_noahlsm.F:2917
FBUR = 0x0.0p+0f; // module_sf_noahlsm.F:2918
FGSN = 0x0.0p+0f; // module_sf_noahlsm.F:2919
} // module_sf_noahlsm.F:2920
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2926-3011
static __attribute__((noinline)) void noah_snksrc(thread NoahTables & tables, thread float & TSNSR, thread float & TAVG, thread float & SMC, thread float & SH2O, thread float * ZSOIL, thread int & NSOIL, thread float & SMCMAX, thread float & PSISAT, thread float & BEXP, thread float & DT, thread int & K, thread float & QTOT){
float DF;
float DZ;
float DZH;
float FREE;
float TDN;
float TM;
float TUP;
float TZ;
float X0;
float XDN;
float XH2O;
float XUP;
float DH2O = 0x1.f400000000000p+9f;
float HLICE = 0x1.45af000000000p+18f;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:2934
if ((K == 1)) { // module_sf_noahlsm.F:2949
DZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:2950
} else { // module_sf_noahlsm.F:2951
DZ = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:2952
} // module_sf_noahlsm.F:2953
{  noah_frh2o(tables,FREE,TAVG,SMC,SH2O,SMCMAX,BEXP,PSISAT); if(tables.error)return; } // module_sf_noahlsm.F:2972
XH2O = AD(SH2O,DV(MU(QTOT,DT),MU(MU(DH2O,HLICE),DZ))); // module_sf_noahlsm.F:2979
if (((XH2O < SH2O) && (XH2O < FREE))) { // module_sf_noahlsm.F:2980
if ((FREE > SH2O)) { // module_sf_noahlsm.F:2981
XH2O = SH2O; // module_sf_noahlsm.F:2982
} else { // module_sf_noahlsm.F:2983
XH2O = FREE; // module_sf_noahlsm.F:2984
} // module_sf_noahlsm.F:2985
} // module_sf_noahlsm.F:2986
if (((XH2O > SH2O) && (XH2O > FREE))) { // module_sf_noahlsm.F:2992
if ((FREE < SH2O)) { // module_sf_noahlsm.F:2993
XH2O = SH2O; // module_sf_noahlsm.F:2994
} else { // module_sf_noahlsm.F:2995
XH2O = FREE; // module_sf_noahlsm.F:2996
} // module_sf_noahlsm.F:2997
} // module_sf_noahlsm.F:2998
if ((XH2O < 0x0.0p+0f)) { XH2O = 0x0.0p+0f; } // module_sf_noahlsm.F:3005
if ((XH2O > SMC)) { XH2O = SMC; } // module_sf_noahlsm.F:3006
TSNSR = DV(MU(MU(MU((-DH2O),HLICE),DZ),SU(XH2O,SH2O)),DT); // module_sf_noahlsm.F:3007
SH2O = XH2O; // module_sf_noahlsm.F:3008
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3014-3417
static __attribute__((noinline)) void noah_snopac(thread NoahTables & tables, thread float & ETP, thread float & ETA, thread float & PRCP, thread float & PRCPF, thread bool & SNOWNG, thread float * SMC, thread float & SMCMAX, thread float & SMCWLT, thread float & SMCREF, thread float & SMCDRY, thread float & CMC, thread float & CMCMAX, thread int & NSOIL, thread float & DT, thread float & SBETA, thread float & DF1, thread float & Q2, thread float & T1, thread float & SFCTMP, thread float & T24, thread float & TH2, thread float & FDOWN, thread float & F1, thread float & SSOIL, thread float * STC, thread float & EPSCA, thread float & SFCPRS, thread float & BEXP, thread float & PC, thread float & RCH, thread float & RR, thread float & CFACTR, thread float & SNCOVR, thread float & ESD, thread float & SNDENS, thread float & SNOWH, thread float * SH2O, thread float & SLOPE, thread float & KDT, thread float & FRZFACT, thread float & PSISAT, thread float * ZSOIL, thread float & DWSAT, thread float & DKSAT, thread float & TBOT, thread float & ZBOT, thread float & SHDFAC, thread float & RUNOFF1, thread float & RUNOFF2, thread float & RUNOFF3, thread float & EDIR, thread float & EC, thread float * ET, thread float & ETT, thread int & NROOT, thread float & SNOMLT, thread float * RTDIS, thread float & QUARTZ, thread float & FXEXP, thread float & CSOIL, thread float & BETA, thread float & DRIP, thread float & DEW, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & ESNOW, thread float & ETNS, thread float & EMISSI, thread float & RIBB, thread float & SOLDN, thread int & ISURBAN, thread int & VEGTYP, thread float & ETPN, thread float & FLX4, thread bool & UA_PHYS, thread float & SFHEAD1RT, thread float & INFXS1RT, thread float & ETPND1, thread int & SOILTYP, thread int & OPT_THCND, thread float & QFX_PHY, thread int & FASDAS, thread float & HCPCT_FASDAS){
thread auto& SIGMA=tables.SIGMA;
thread auto& CPH2O=tables.CPH2O;
thread auto& CPICE=tables.CPICE;
thread auto& LSUBF=tables.LSUBF;
int K;
int IT16;
float ET1[20];
float DENOM;
float DSOIL;
float DTOT;
float EC1;
float EDIR1;
float ESDFLX;
float ETT1;
float ESNOW1;
float ESNOW2;
float ETA1;
float ETP1;
float ETP2;
float ETP3;
float ETNS1;
float ETANRG;
float ETAX;
float EX;
float FLX3X;
float FRCSNO;
float FRCSOI;
float PRCP1;
float QSAT;
float RSNOW;
float SEH;
float SNCOND;
float SSOIL1;
float T11;
float T12;
float T12A;
float T12AX;
float T12B;
float T14;
float YY;
float ZZ1;
float T15;
float T16;
float DTOT2;
float ESDMIN = 0x1.0c6f7a0000000p-20f;
float LSUBC = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
float TFREEZ = 0x1.1126660000000p+8f;
float SNOEXP = 0x1.0000000000000p+1f;
float ETP1N;
 // module_sf_noahlsm.F:3037
DEW = 0x0.0p+0f; // module_sf_noahlsm.F:3110
EDIR = 0x0.0p+0f; // module_sf_noahlsm.F:3111
EDIR1 = 0x0.0p+0f; // module_sf_noahlsm.F:3112
EC1 = 0x0.0p+0f; // module_sf_noahlsm.F:3113
EC = 0x0.0p+0f; // module_sf_noahlsm.F:3114
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3117
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3118
ET1[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3119
} // module_sf_noahlsm.F:3120
ETT = 0x0.0p+0f; // module_sf_noahlsm.F:3121
ETT1 = 0x0.0p+0f; // module_sf_noahlsm.F:3122
ETPND1 = 0x0.0p+0f; // module_sf_noahlsm.F:3125
ETNS = 0x0.0p+0f; // module_sf_noahlsm.F:3128
ETNS1 = 0x0.0p+0f; // module_sf_noahlsm.F:3129
ESNOW = 0x0.0p+0f; // module_sf_noahlsm.F:3130
ESNOW1 = 0x0.0p+0f; // module_sf_noahlsm.F:3131
ESNOW2 = 0x0.0p+0f; // module_sf_noahlsm.F:3132
PRCP1 = MU(PRCPF,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3137
BETA = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3141
if ((ETP <= 0x0.0p+0f)) { // module_sf_noahlsm.F:3142
if (((RIBB >= 0x1.99999a0000000p-4f) && (FDOWN > 0x1.2c00000000000p+7f))) { // module_sf_noahlsm.F:3143
ETP = DV(AD(DV(MU(noah_min(MU(ETP,SU(0x1.0000000000000p+0f,RIBB)),0x0.0p+0f),SNCOVR),0x1.f5c2900000000p-1f),MU(ETP,SU(0x1.f5c2900000000p-1f,SNCOVR))),0x1.f5c2900000000p-1f); // module_sf_noahlsm.F:3144
} // module_sf_noahlsm.F:3145
if ((ETP == 0x0.0p+0f)) { BETA = 0x0.0p+0f; } // module_sf_noahlsm.F:3146
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3147
if (UA_PHYS) { ETP1N = MU(ETPN,0x1.0624de0000000p-10f); } // module_sf_noahlsm.F:3148
DEW = (-ETP1); // module_sf_noahlsm.F:3149
ESNOW2 = MU(ETP1,DT); // module_sf_noahlsm.F:3150
ETANRG = MU(ETP,AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LSUBC),MU(SNCOVR,LSUBS))); // module_sf_noahlsm.F:3151
} else { // module_sf_noahlsm.F:3152
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3153
if (UA_PHYS) { ETP1N = MU(ETPN,0x1.0624de0000000p-10f); } // module_sf_noahlsm.F:3154
if ((SNCOVR < 0x1.0000000000000p+0f)) { // module_sf_noahlsm.F:3156
{  noah_evapo(tables,ETNS1,SMC,NSOIL,CMC,ETP1,DT,ZSOIL,SH2O,SMCMAX,BEXP,PC,SMCWLT,DKSAT,DWSAT,SMCREF,SHDFAC,CMCMAX,SMCDRY,CFACTR,EDIR1,EC1,ET1,ETT1,SFCTMP,Q2,NROOT,RTDIS,FXEXP,SFHEAD1RT,ETPND1); if(tables.error)return; } // module_sf_noahlsm.F:3157
EDIR1 = MU(EDIR1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3165
EC1 = MU(EC1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3166
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3167
ET1[(K)-(1)] = MU(ET1[(K)-(1)],SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3168
} // module_sf_noahlsm.F:3169
ETT1 = MU(ETT1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3170
ETNS1 = MU(ETNS1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3172
EDIR = MU(EDIR1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3174
EC = MU(EC1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3175
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3176
ET[(K)-(1)] = MU(ET1[(K)-(1)],0x1.f400000000000p+9f); // module_sf_noahlsm.F:3177
} // module_sf_noahlsm.F:3178
if ((FASDAS == 1)) { // module_sf_noahlsm.F:3182
QFX_PHY = AD(EDIR,EC); // module_sf_noahlsm.F:3183
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3184
QFX_PHY = AD(QFX_PHY,ET[(K)-(1)]); // module_sf_noahlsm.F:3185
} // module_sf_noahlsm.F:3186
} // module_sf_noahlsm.F:3187
ETT = MU(ETT1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3191
ETNS = MU(ETNS1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3192
ETPND1 = MU(ETPND1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3196
} // module_sf_noahlsm.F:3201
ESNOW = MU(ETP,SNCOVR); // module_sf_noahlsm.F:3202
if (UA_PHYS) { ESNOW = MU(ETPN,SNCOVR); } // module_sf_noahlsm.F:3203
ESNOW1 = MU(ESNOW,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3204
ESNOW2 = MU(ESNOW1,DT); // module_sf_noahlsm.F:3205
ETANRG = AD(MU(ESNOW,LSUBS),MU(ETNS,LSUBC)); // module_sf_noahlsm.F:3206
} // module_sf_noahlsm.F:3207
FLX1 = 0x0.0p+0f; // module_sf_noahlsm.F:3215
if (SNOWNG) { // module_sf_noahlsm.F:3216
FLX1 = MU(MU(CPICE,PRCP),SU(T1,SFCTMP)); // module_sf_noahlsm.F:3217
} else { // module_sf_noahlsm.F:3218
if ((PRCP > 0x0.0p+0f)) { FLX1 = MU(MU(CPH2O,PRCP),SU(T1,SFCTMP)); } // module_sf_noahlsm.F:3219
} // module_sf_noahlsm.F:3228
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:3229
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm.F:3230
DENOM = AD(0x1.0000000000000p+0f,DV(DF1,MU(MU(DTOT,RR),RCH))); // module_sf_noahlsm.F:3231
T12A = DV(SU(SU(AD(DV(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T24)),RCH),TH2),SFCTMP),DV(ETANRG,RCH)),RR); // module_sf_noahlsm.F:3236
T12B = DV(MU(DF1,STC[(1)-(1)]),MU(MU(DTOT,RR),RCH)); // module_sf_noahlsm.F:3239
T12 = DV(AD(AD(SFCTMP,T12A),T12B),DENOM); // module_sf_noahlsm.F:3253
if ((T12 <= TFREEZ)) { // module_sf_noahlsm.F:3254
T1 = T12; // module_sf_noahlsm.F:3255
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm.F:3256
ESD = noah_max(0x0.0p+0f,SU(ESD,ESNOW2)); // module_sf_noahlsm.F:3258
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:3259
EX = 0x0.0p+0f; // module_sf_noahlsm.F:3260
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm.F:3262
if (UA_PHYS) { FLX4 = 0x0.0p+0f; } // module_sf_noahlsm.F:3263
} else { // module_sf_noahlsm.F:3279
// GNU strict folds real PARAMETER SNOEXP=2 to one rounded multiply; module_sf_noahlsm.F:3282.
T1 = AD(MU(TFREEZ,noah_max(0x1.47ae140000000p-7f,MU(SNCOVR,SNCOVR))),MU(T12,SU(0x1.0000000000000p+0f,noah_max(0x1.47ae140000000p-7f,MU(SNCOVR,SNCOVR))))); // module_sf_noahlsm.F:3282
BETA = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3283
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm.F:3290
if ((SU(ESD,ESNOW2) <= ESDMIN)) { // module_sf_noahlsm.F:3291
ESD = 0x0.0p+0f; // module_sf_noahlsm.F:3292
EX = 0x0.0p+0f; // module_sf_noahlsm.F:3293
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm.F:3294
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:3295
if (UA_PHYS) { FLX4 = 0x0.0p+0f; } // module_sf_noahlsm.F:3296
} else { // module_sf_noahlsm.F:3301
ESD = SU(ESD,ESNOW2); // module_sf_noahlsm.F:3302
ETP3 = MU(ETP,LSUBC); // module_sf_noahlsm.F:3303
SEH = MU(RCH,SU(T1,TH2)); // module_sf_noahlsm.F:3304
T14 = MU(T1,T1); // module_sf_noahlsm.F:3305
T14 = MU(T14,T14); // module_sf_noahlsm.F:3306
FLX3 = SU(SU(SU(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T14)),SSOIL),SEH),ETANRG); // module_sf_noahlsm.F:3310
if ((FLX3 <= 0x0.0p+0f)) { FLX3 = 0x0.0p+0f; } // module_sf_noahlsm.F:3311
if (((UA_PHYS && (FLX4 > 0x0.0p+0f)) && (FLX3 > 0x0.0p+0f))) { // module_sf_noahlsm.F:3313
if ((FLX3 >= FLX4)) { // module_sf_noahlsm.F:3314
FLX3 = SU(FLX3,FLX4); // module_sf_noahlsm.F:3315
} else { // module_sf_noahlsm.F:3316
FLX4 = FLX3; // module_sf_noahlsm.F:3317
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:3318
} // module_sf_noahlsm.F:3319
} else { // module_sf_noahlsm.F:3320
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:3321
} // module_sf_noahlsm.F:3322
EX = DV(MU(FLX3,0x1.0624de0000000p-10f),LSUBF); // module_sf_noahlsm.F:3327
SNOMLT = MU(EX,DT); // module_sf_noahlsm.F:3333
if ((SU(ESD,SNOMLT) >= ESDMIN)) { // module_sf_noahlsm.F:3334
ESD = SU(ESD,SNOMLT); // module_sf_noahlsm.F:3335
} else { // module_sf_noahlsm.F:3339
EX = DV(ESD,DT); // module_sf_noahlsm.F:3340
FLX3 = MU(MU(EX,0x1.f400000000000p+9f),LSUBF); // module_sf_noahlsm.F:3341
SNOMLT = ESD; // module_sf_noahlsm.F:3342
ESD = 0x0.0p+0f; // module_sf_noahlsm.F:3344
} // module_sf_noahlsm.F:3348
} // module_sf_noahlsm.F:3349
PRCP1 = AD(PRCP1,EX); // module_sf_noahlsm.F:3361
} // module_sf_noahlsm.F:3369
{ float tmp1_26=0; noah_smflx(tables,SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,EDIR1,EC1,ET1,DRIP,SFHEAD1RT,INFXS1RT,tmp1_26,false); if(tables.error)return; } // module_sf_noahlsm.F:3370
ZZ1 = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3385
YY = SU(STC[(1)-(1)],DV(MU(MU(MU(0x1.0000000000000p-1f,SSOIL),ZSOIL[(1)-(1)]),ZZ1),DF1)); // module_sf_noahlsm.F:3386
T11 = T1; // module_sf_noahlsm.F:3395
{  noah_shflx(tables,SSOIL1,STC,SMC,SMCMAX,NSOIL,T11,DT,YY,ZZ1,ZSOIL,TBOT,ZBOT,SMCWLT,PSISAT,SH2O,BEXP,F1,DF1,QUARTZ,CSOIL,VEGTYP,ISURBAN,SOILTYP,OPT_THCND,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:3396
if ((ESD > 0x0.0p+0f)) { // module_sf_noahlsm.F:3406
{  noah_snowpack(tables,ESD,DT,SNOWH,SNDENS,T1,YY,SNOMLT,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:3407
} else { // module_sf_noahlsm.F:3408
ESD = 0x0.0p+0f; // module_sf_noahlsm.F:3409
SNOWH = 0x0.0p+0f; // module_sf_noahlsm.F:3410
SNDENS = 0x0.0p+0f; // module_sf_noahlsm.F:3411
SNCOND = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3412
SNCOVR = 0x0.0p+0f; // module_sf_noahlsm.F:3413
} // module_sf_noahlsm.F:3414
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3421-3553
static __attribute__((noinline)) void noah_snowpack(thread NoahTables & tables, thread float & ESD, thread float & DTSEC, thread float & SNOWH, thread float & SNDENS, thread float & TSNOW, thread float & TSOIL, thread float & SNOMLT, thread bool & UA_PHYS){
int IPOL;
int J;
float BFAC;
float DSX;
float DTHR;
float DW;
float SNOWHC;
float PEXP;
float TAVGC;
float TSNOWC;
float TSOILC;
float ESDC;
float ESDCX;
float C1 = 0x1.47ae140000000p-7f;
float C2 = 0x1.5000000000000p+4f;
float G = 0x1.39eb860000000p+3f;
float KN = 0x1.f400000000000p+11f;
float SNOMLTC;
 // module_sf_noahlsm.F:3440
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3455
ESDC = MU(ESD,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3456
if (UA_PHYS) { SNOMLTC = MU(SNOMLT,0x1.9000000000000p+6f); } // module_sf_noahlsm.F:3457
DTHR = DV(DTSEC,0x1.c200000000000p+11f); // module_sf_noahlsm.F:3458
TSNOWC = SU(TSNOW,0x1.1126660000000p+8f); // module_sf_noahlsm.F:3459
TSOILC = SU(TSOIL,0x1.1126660000000p+8f); // module_sf_noahlsm.F:3460
TAVGC = MU(0x1.0000000000000p-1f,AD(TSNOWC,TSOILC)); // module_sf_noahlsm.F:3474
if ((ESDC > 0x1.47ae140000000p-7f)) { // module_sf_noahlsm.F:3475
ESDCX = ESDC; // module_sf_noahlsm.F:3476
} else { // module_sf_noahlsm.F:3477
ESDCX = 0x1.47ae140000000p-7f; // module_sf_noahlsm.F:3478
} // module_sf_noahlsm.F:3479
BFAC = MU(MU(DTHR,C1),r_exp(SU(MU(0x1.47ae140000000p-4f,TAVGC),MU(C2,SNDENS)))); // module_sf_noahlsm.F:3499
IPOL = 4; // module_sf_noahlsm.F:3500
PEXP = 0x0.0p+0f; // module_sf_noahlsm.F:3501
for (J=IPOL; J>=1; J+=-1) { // module_sf_noahlsm.F:3503
PEXP = DV(MU(MU(AD(0x1.0000000000000p+0f,PEXP),BFAC),ESDCX),((float)((J + 1)))); // module_sf_noahlsm.F:3504
} // module_sf_noahlsm.F:3505
PEXP = AD(PEXP,0x1.0000000000000p+0f); // module_sf_noahlsm.F:3507
DSX = MU(SNDENS,PEXP); // module_sf_noahlsm.F:3528
if ((DSX > 0x1.99999a0000000p-2f)) { DSX = 0x1.99999a0000000p-2f; } // module_sf_noahlsm.F:3529
if ((DSX < 0x1.99999a0000000p-5f)) { DSX = 0x1.99999a0000000p-5f; } // module_sf_noahlsm.F:3530
SNDENS = DSX; // module_sf_noahlsm.F:3536
if ((TSNOWC >= 0x0.0p+0f)) { // module_sf_noahlsm.F:3537
DW = DV(MU(0x1.0a3d700000000p-3f,DTHR),0x1.8000000000000p+4f); // module_sf_noahlsm.F:3538
if ((UA_PHYS && (TSOILC >= 0x0.0p+0f))) { // module_sf_noahlsm.F:3539
DW = noah_min(DW,DV(MU(0x1.0a3d700000000p-3f,SNOMLTC),AD(ESDCX,MU(0x1.0a3d700000000p-3f,SNOMLTC)))); // module_sf_noahlsm.F:3540
} // module_sf_noahlsm.F:3541
SNDENS = AD(MU(SNDENS,SU(0x1.0000000000000p+0f,DW)),DW); // module_sf_noahlsm.F:3542
if ((SNDENS >= 0x1.99999a0000000p-2f)) { SNDENS = 0x1.99999a0000000p-2f; } // module_sf_noahlsm.F:3543
} // module_sf_noahlsm.F:3548
SNOWHC = DV(ESDC,SNDENS); // module_sf_noahlsm.F:3549
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm.F:3550
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3556-3601
static __attribute__((noinline)) void noah_snowz0(thread NoahTables & tables, thread float & SNCOVR, thread float & Z0, thread float & Z0BRD, thread float & SNOWH, thread float & FBUR, thread float & FGSN, thread float & SHDMAX, thread bool & UA_PHYS){
float Z0S = 0x1.0624de0000000p-10f;
float BURIAL;
float Z0EFF;
float Z0G = 0x1.47ae140000000p-7f;
float FV;
float A1;
float A2;
 // module_sf_noahlsm.F:3566
if (UA_PHYS) { // module_sf_noahlsm.F:3580
FV = MU(SHDMAX,SU(0x1.0000000000000p+0f,FBUR)); // module_sf_noahlsm.F:3582
A1 = MU(noah_powi(SU(0x1.0000000000000p+0f,FV),2),AD(MU(SU(0x1.0000000000000p+0f,noah_powi(FGSN,2)),r_log(Z0G)),MU(noah_powi(FGSN,2),r_log(Z0S)))); // module_sf_noahlsm.F:3583
A2 = MU(SU(0x1.0000000000000p+0f,noah_powi(SU(0x1.0000000000000p+0f,FV),2)),r_log(Z0BRD)); // module_sf_noahlsm.F:3584
Z0 = r_exp(AD(A1,A2)); // module_sf_noahlsm.F:3585
} else { // module_sf_noahlsm.F:3587
BURIAL = SU(MU(0x1.c000000000000p+2f,Z0BRD),SNOWH); // module_sf_noahlsm.F:3590
if ((BURIAL <= 0x1.6f00680000000p-11f)) { // module_sf_noahlsm.F:3591
Z0EFF = Z0S; // module_sf_noahlsm.F:3592
} else { // module_sf_noahlsm.F:3593
Z0EFF = DV(BURIAL,0x1.c000000000000p+2f); // module_sf_noahlsm.F:3594
} // module_sf_noahlsm.F:3595
Z0 = AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),Z0BRD),MU(SNCOVR,Z0EFF)); // module_sf_noahlsm.F:3597
} // module_sf_noahlsm.F:3599
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3605-3654
static __attribute__((noinline)) void noah_snow_new(thread NoahTables & tables, thread float & TEMP, thread float & NEWSN, thread float & SNOWH, thread float & SNDENS){
float DSNEW;
float HNEWC;
float SNOWHC;
float NEWSNC;
float TEMPC;
 // module_sf_noahlsm.F:3618
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3626
NEWSNC = MU(NEWSN,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3627
TEMPC = SU(TEMP,0x1.1126660000000p+8f); // module_sf_noahlsm.F:3635
if ((TEMPC <= -0x1.e000000000000p+3f)) { // module_sf_noahlsm.F:3636
DSNEW = 0x1.99999a0000000p-5f; // module_sf_noahlsm.F:3637
} else { // module_sf_noahlsm.F:3638
DSNEW = AD(0x1.99999a0000000p-5f,MU(0x1.bda5120000000p-10f,r_pow(AD(TEMPC,0x1.e000000000000p+3f),0x1.8000000000000p+0f))); // module_sf_noahlsm.F:3639
} // module_sf_noahlsm.F:3640
HNEWC = DV(NEWSNC,DSNEW); // module_sf_noahlsm.F:3644
if ((AD(SNOWHC,HNEWC) < 0x1.0624de0000000p-10f)) { // module_sf_noahlsm.F:3645
SNDENS = noah_max(DSNEW,SNDENS); // module_sf_noahlsm.F:3646
} else { // module_sf_noahlsm.F:3647
SNDENS = DV(AD(MU(SNOWHC,SNDENS),MU(HNEWC,DSNEW)),AD(SNOWHC,HNEWC)); // module_sf_noahlsm.F:3648
} // module_sf_noahlsm.F:3649
SNOWHC = AD(SNOWHC,HNEWC); // module_sf_noahlsm.F:3650
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm.F:3651
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3657-3951
static __attribute__((noinline)) void noah_srt(thread NoahTables & tables, thread float * RHSTT, thread float & EDIR, thread float * ET, thread float * SH2O, thread float * SH2OA, thread int & NSOIL, thread float & PCPDRP, thread float * ZSOIL, thread float & DWSAT, thread float & DKSAT, thread float & SMCMAX, thread float & BEXP, thread float & RUNOFF1, thread float & RUNOFF2, thread float & DT, thread float & SMCWLT, thread float & SLOPE, thread float & KDT, thread float & FRZX, thread float * SICE, thread float * AI, thread float * BI, thread float * CI, thread float & SFHEAD1RT, thread float & INFXS1RT){
int IALP1;
int IOHINF;
int J;
int JJ;
int K;
int KS;
float SFCWATR;
float CHCKSM;
float DMAX[20];
float ACRT;
float DD;
float DDT;
float DDZ;
float DDZ2;
float DENOM;
float DENOM2;
float DICE;
float DSMDZ;
float DSMDZ2;
float DT1;
float FCR;
float INFMAX;
float MXSMC;
float MXSMC2;
float NUMER;
float PDDUM;
float PX;
float SICEMAX;
float SLOPX;
float SMCAV;
float SSTT;
float SUM;
float VAL;
float WCND;
float WCND2;
float WDF;
float WDF2;
int CVFRZ = 3;
 // module_sf_noahlsm.F:3669
IOHINF = 1; // module_sf_noahlsm.F:3714
SICEMAX = 0x0.0p+0f; // module_sf_noahlsm.F:3715
for (KS=1; KS<=NSOIL; KS+=1) { // module_sf_noahlsm.F:3716
if ((SICE[(KS)-(1)] > SICEMAX)) { SICEMAX = SICE[(KS)-(1)]; } // module_sf_noahlsm.F:3717
} // module_sf_noahlsm.F:3721
PDDUM = PCPDRP; // module_sf_noahlsm.F:3732
RUNOFF1 = 0x0.0p+0f; // module_sf_noahlsm.F:3733
if ((PCPDRP != 0x0.0p+0f)) { // module_sf_noahlsm.F:3748
DT1 = DV(DT,0x1.5180000000000p+16f); // module_sf_noahlsm.F:3750
SMCAV = SU(SMCMAX,SMCWLT); // module_sf_noahlsm.F:3751
DMAX[(1)-(1)] = MU((-ZSOIL[(1)-(1)]),SMCAV); // module_sf_noahlsm.F:3756
DICE = MU((-ZSOIL[(1)-(1)]),SICE[(1)-(1)]); // module_sf_noahlsm.F:3758
DMAX[(1)-(1)] = MU(DMAX[(1)-(1)],SU(0x1.0000000000000p+0f,DV(SU(AD(SH2OA[(1)-(1)],SICE[(1)-(1)]),SMCWLT),SMCAV))); // module_sf_noahlsm.F:3759
DD = DMAX[(1)-(1)]; // module_sf_noahlsm.F:3762
for (KS=2; KS<=NSOIL; KS+=1) { // module_sf_noahlsm.F:3767
DICE = AD(DICE,MU(SU(ZSOIL[((KS - 1))-(1)],ZSOIL[(KS)-(1)]),SICE[(KS)-(1)])); // module_sf_noahlsm.F:3769
DMAX[(KS)-(1)] = MU(SU(ZSOIL[((KS - 1))-(1)],ZSOIL[(KS)-(1)]),SMCAV); // module_sf_noahlsm.F:3770
DMAX[(KS)-(1)] = MU(DMAX[(KS)-(1)],SU(0x1.0000000000000p+0f,DV(SU(AD(SH2OA[(KS)-(1)],SICE[(KS)-(1)]),SMCWLT),SMCAV))); // module_sf_noahlsm.F:3771
DD = AD(DD,DMAX[(KS)-(1)]); // module_sf_noahlsm.F:3773
} // module_sf_noahlsm.F:3778
VAL = SU(0x1.0000000000000p+0f,r_exp(MU((-KDT),DT1))); // module_sf_noahlsm.F:3779
DDT = MU(DD,VAL); // module_sf_noahlsm.F:3780
PX = MU(PCPDRP,DT); // module_sf_noahlsm.F:3786
if ((PX < 0x0.0p+0f)) { PX = 0x0.0p+0f; } // module_sf_noahlsm.F:3788
INFMAX = DV(MU(PX,DV(DDT,AD(PX,DDT))),DT); // module_sf_noahlsm.F:3796
FCR = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3797
if ((DICE > 0x1.47ae140000000p-7f)) { // module_sf_noahlsm.F:3798
ACRT = DV(MU(CVFRZ,FRZX),DICE); // module_sf_noahlsm.F:3799
SUM = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3800
IALP1 = (CVFRZ - 1); // module_sf_noahlsm.F:3801
for (J=1; J<=IALP1; J+=1) { // module_sf_noahlsm.F:3802
K = 1; // module_sf_noahlsm.F:3803
for (JJ=(J + 1); JJ<=IALP1; JJ+=1) { // module_sf_noahlsm.F:3804
K = (K * JJ); // module_sf_noahlsm.F:3805
} // module_sf_noahlsm.F:3806
SUM = AD(SUM,DV(noah_powi(ACRT,(CVFRZ - J)),((float)(K)))); // module_sf_noahlsm.F:3807
} // module_sf_noahlsm.F:3808
FCR = SU(0x1.0000000000000p+0f,MU(r_exp((-ACRT)),SUM)); // module_sf_noahlsm.F:3809
} // module_sf_noahlsm.F:3810
INFMAX = MU(INFMAX,FCR); // module_sf_noahlsm.F:3818
MXSMC = SH2OA[(1)-(1)]; // module_sf_noahlsm.F:3820
{  noah_wdfcnd(tables,WDF,WCND,MXSMC,SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3821
INFMAX = noah_max(INFMAX,WCND); // module_sf_noahlsm.F:3823
INFMAX = noah_min(INFMAX,DV(PX,DT)); // module_sf_noahlsm.F:3825
if ((PCPDRP > INFMAX)) { // module_sf_noahlsm.F:3833
RUNOFF1 = SU(PCPDRP,INFMAX); // module_sf_noahlsm.F:3834
INFXS1RT = MU(MU(RUNOFF1,DT),0x1.f400000000000p+9f); // module_sf_noahlsm.F:3836
PDDUM = INFMAX; // module_sf_noahlsm.F:3837
} // module_sf_noahlsm.F:3838
} // module_sf_noahlsm.F:3845
MXSMC = SH2OA[(1)-(1)]; // module_sf_noahlsm.F:3847
{  noah_wdfcnd(tables,WDF,WCND,MXSMC,SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3848
DDZ = DV(0x1.0000000000000p+0f,MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:3853
AI[(1)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3854
BI[(1)-(1)] = DV(MU(WDF,DDZ),(-ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:3855
CI[(1)-(1)] = (-BI[(1)-(1)]); // module_sf_noahlsm.F:3861
DSMDZ = DV(SU(SH2O[(1)-(1)],SH2O[(2)-(1)]),MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:3862
RHSTT[(1)-(1)] = DV(AD(AD(SU(AD(MU(WDF,DSMDZ),WCND),PDDUM),EDIR),ET[(1)-(1)]),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:3863
SSTT = AD(AD(AD(MU(WDF,DSMDZ),WCND),EDIR),ET[(1)-(1)]); // module_sf_noahlsm.F:3868
DDZ2 = 0x0.0p+0f; // module_sf_noahlsm.F:3873
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3874
DENOM2 = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:3875
if ((K != NSOIL)) { // module_sf_noahlsm.F:3876
SLOPX = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3883
MXSMC2 = SH2OA[(K)-(1)]; // module_sf_noahlsm.F:3885
{  noah_wdfcnd(tables,WDF2,WCND2,MXSMC2,SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3886
DENOM = SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)]); // module_sf_noahlsm.F:3891
DSMDZ2 = DV(SU(SH2O[(K)-(1)],SH2O[((K + 1))-(1)]),MU(DENOM,0x1.0000000000000p-1f)); // module_sf_noahlsm.F:3896
DDZ2 = DV(0x1.0000000000000p+1f,DENOM); // module_sf_noahlsm.F:3897
CI[(K)-(1)] = DV(MU((-WDF2),DDZ2),DENOM2); // module_sf_noahlsm.F:3898
} else { // module_sf_noahlsm.F:3900
SLOPX = SLOPE; // module_sf_noahlsm.F:3909
{  noah_wdfcnd(tables,WDF2,WCND2,SH2OA[(NSOIL)-(1)],SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3910
DSMDZ2 = 0x0.0p+0f; // module_sf_noahlsm.F:3920
CI[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3921
} // module_sf_noahlsm.F:3925
NUMER = AD(SU(SU(AD(MU(WDF2,DSMDZ2),MU(SLOPX,WCND2)),MU(WDF,DSMDZ)),WCND),ET[(K)-(1)]); // module_sf_noahlsm.F:3926
RHSTT[(K)-(1)] = DV(NUMER,(-DENOM2)); // module_sf_noahlsm.F:3932
AI[(K)-(1)] = DV(MU((-WDF),DDZ),DENOM2); // module_sf_noahlsm.F:3933
BI[(K)-(1)] = (-AD(AI[(K)-(1)],CI[(K)-(1)])); // module_sf_noahlsm.F:3939
if ((K == NSOIL)) { // module_sf_noahlsm.F:3940
RUNOFF2 = MU(SLOPX,WCND2); // module_sf_noahlsm.F:3941
} // module_sf_noahlsm.F:3942
if ((K != NSOIL)) { // module_sf_noahlsm.F:3943
WDF = WDF2; // module_sf_noahlsm.F:3944
WCND = WCND2; // module_sf_noahlsm.F:3945
DSMDZ = DSMDZ2; // module_sf_noahlsm.F:3946
DDZ = DDZ2; // module_sf_noahlsm.F:3947
} // module_sf_noahlsm.F:3948
} // module_sf_noahlsm.F:3949
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3954-4081
static __attribute__((noinline)) void noah_sstep(thread NoahTables & tables, thread float * SH2OOUT, thread float * SH2OIN, thread float & CMC, thread float * RHSTT, thread float & RHSCT, thread float & DT, thread int & NSOIL, thread float & SMCMAX, thread float & CMCMAX, thread float & RUNOFF3, thread float * ZSOIL, thread float * SMC, thread float * SICE, thread float * AI, thread float * BI, thread float * CI, thread float & INFXS1RT){
int I;
int K;
int KK11;
float AVAIL;
float RHSTTIN[20];
float CIIN[20];
float DDZ;
float STOT;
float WPLUS;
 // module_sf_noahlsm.F:3964
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3987
RHSTT[(K)-(1)] = MU(RHSTT[(K)-(1)],DT); // module_sf_noahlsm.F:3988
AI[(K)-(1)] = MU(AI[(K)-(1)],DT); // module_sf_noahlsm.F:3989
BI[(K)-(1)] = AD(0x1.0000000000000p+0f,MU(BI[(K)-(1)],DT)); // module_sf_noahlsm.F:3990
CI[(K)-(1)] = MU(CI[(K)-(1)],DT); // module_sf_noahlsm.F:3991
} // module_sf_noahlsm.F:3992
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3996
RHSTTIN[(K)-(1)] = RHSTT[(K)-(1)]; // module_sf_noahlsm.F:3997
} // module_sf_noahlsm.F:3998
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3999
CIIN[(K)-(1)] = CI[(K)-(1)]; // module_sf_noahlsm.F:4000
} // module_sf_noahlsm.F:4001
{  noah_rosr12(tables,CI,AI,BI,CIIN,RHSTTIN,RHSTT,NSOIL); if(tables.error)return; } // module_sf_noahlsm.F:4005
WPLUS = 0x0.0p+0f; // module_sf_noahlsm.F:4011
RUNOFF3 = 0x0.0p+0f; // module_sf_noahlsm.F:4012
DDZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:4014
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:4015
if ((K != 1)) { DDZ = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); } // module_sf_noahlsm.F:4016
SH2OOUT[(K)-(1)] = AD(AD(SH2OIN[(K)-(1)],CI[(K)-(1)]),DV(WPLUS,DDZ)); // module_sf_noahlsm.F:4017
STOT = AD(SH2OOUT[(K)-(1)],SICE[(K)-(1)]); // module_sf_noahlsm.F:4018
if ((STOT > SMCMAX)) { // module_sf_noahlsm.F:4019
if ((K == 1)) { // module_sf_noahlsm.F:4020
DDZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:4021
} else { // module_sf_noahlsm.F:4022
KK11 = (K - 1); // module_sf_noahlsm.F:4023
DDZ = AD((-ZSOIL[(K)-(1)]),ZSOIL[(KK11)-(1)]); // module_sf_noahlsm.F:4024
} // module_sf_noahlsm.F:4025
WPLUS = MU(SU(STOT,SMCMAX),DDZ); // module_sf_noahlsm.F:4026
} else { // module_sf_noahlsm.F:4027
WPLUS = 0x0.0p+0f; // module_sf_noahlsm.F:4028
} // module_sf_noahlsm.F:4029
SMC[(K)-(1)] = noah_max(noah_min(STOT,SMCMAX),0x1.47ae140000000p-6f); // module_sf_noahlsm.F:4030
SH2OOUT[(K)-(1)] = noah_max(SU(SMC[(K)-(1)],SICE[(K)-(1)]),0x0.0p+0f); // module_sf_noahlsm.F:4031
} // module_sf_noahlsm.F:4032
RUNOFF3 = WPLUS; // module_sf_noahlsm.F:4075
CMC = AD(CMC,MU(DT,RHSCT)); // module_sf_noahlsm.F:4076
if ((CMC < 0x1.79ca100000000p-67f)) { CMC = 0x0.0p+0f; } // module_sf_noahlsm.F:4077
CMC = noah_min(CMC,CMCMAX); // module_sf_noahlsm.F:4078
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4084-4124
static __attribute__((noinline)) void noah_tbnd(thread NoahTables & tables, thread float & TU, thread float & TB, thread float * ZSOIL, thread float & ZBOT, thread int & K, thread int & NSOIL, thread float & TBND1){
float ZB;
float ZUP;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:4092
if ((K == 1)) { // module_sf_noahlsm.F:4104
ZUP = 0x0.0p+0f; // module_sf_noahlsm.F:4105
} else { // module_sf_noahlsm.F:4106
ZUP = ZSOIL[((K - 1))-(1)]; // module_sf_noahlsm.F:4107
} // module_sf_noahlsm.F:4108
if ((K == NSOIL)) { // module_sf_noahlsm.F:4113
ZB = SU(MU(0x1.0000000000000p+1f,ZBOT),ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:4114
} else { // module_sf_noahlsm.F:4115
ZB = ZSOIL[((K + 1))-(1)]; // module_sf_noahlsm.F:4116
} // module_sf_noahlsm.F:4117
TBND1 = AD(TU,DV(MU(SU(TB,TU),SU(ZUP,ZSOIL[(K)-(1)])),SU(ZUP,ZB))); // module_sf_noahlsm.F:4122
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4128-4251
static __attribute__((noinline)) void noah_tdfcnd(thread NoahTables & tables, thread float & DF, thread float & SMC, thread float & QZ, thread float & SMCMAX, thread float & SH2O, thread float & BEXP, thread float & PSISAT, thread int & SOILTYP, thread int & OPT_THCND){
float AKE;
float GAMMD;
float THKDRY;
float THKICE;
float THKO;
float THKQTZ;
float THKSAT;
float THKS;
float THKW;
float SATRATIO;
float XU;
float XUNFROZ;
float AKEI;
float AKEL;
float PSIF;
float PF;
 // module_sf_noahlsm.F:4139
if (((OPT_THCND == 1) || ((OPT_THCND == 2) && ((SOILTYP != 4) && (SOILTYP != 3))))) { // module_sf_noahlsm.F:4177
SATRATIO = DV(SMC,SMCMAX); // module_sf_noahlsm.F:4184
THKICE = 0x1.19999a0000000p+1f; // module_sf_noahlsm.F:4186
THKW = 0x1.23d70a0000000p-1f; // module_sf_noahlsm.F:4188
THKO = 0x1.0000000000000p+1f; // module_sf_noahlsm.F:4191
THKQTZ = 0x1.eccccc0000000p+2f; // module_sf_noahlsm.F:4193
THKS = MU(r_pow(THKQTZ,QZ),r_pow(THKO,SU(0x1.0000000000000p+0f,QZ))); // module_sf_noahlsm.F:4195
XUNFROZ = DV(SH2O,SMC); // module_sf_noahlsm.F:4198
XU = MU(XUNFROZ,SMCMAX); // module_sf_noahlsm.F:4200
THKSAT = MU(MU(r_pow(THKS,SU(0x1.0000000000000p+0f,SMCMAX)),r_pow(THKICE,SU(SMCMAX,XU))),r_pow(THKW,XU)); // module_sf_noahlsm.F:4203
GAMMD = MU(SU(0x1.0000000000000p+0f,SMCMAX),0x1.5180000000000p+11f); // module_sf_noahlsm.F:4207
THKDRY = DV(AD(MU(0x1.147ae20000000p-3f,GAMMD),0x1.02cccc0000000p+6f),SU(0x1.5180000000000p+11f,MU(0x1.e4dd300000000p-1f,GAMMD))); // module_sf_noahlsm.F:4210
AKEI = SATRATIO; // module_sf_noahlsm.F:4212
if ((SATRATIO > 0x1.99999a0000000p-4f)) { // module_sf_noahlsm.F:4220
AKEL = AD(r_log10(SATRATIO),0x1.0000000000000p+0f); // module_sf_noahlsm.F:4222
} else { // module_sf_noahlsm.F:4225
AKEL = 0x0.0p+0f; // module_sf_noahlsm.F:4227
} // module_sf_noahlsm.F:4228
AKE = DV(AD(MU(SU(SMC,SH2O),AKEI),MU(SH2O,AKEL)),SMC); // module_sf_noahlsm.F:4229
DF = AD(MU(AKE,SU(THKSAT,THKDRY)),THKDRY); // module_sf_noahlsm.F:4233
} else { // module_sf_noahlsm.F:4235
PSIF = MU(MU(PSISAT,0x1.9000000000000p+6f),r_pow(DV(SMCMAX,SMC),BEXP)); // module_sf_noahlsm.F:4239
PF = r_log10(noah_abs(PSIF)); // module_sf_noahlsm.F:4241
if ((PF <= 0x1.4666660000000p+2f)) { // module_sf_noahlsm.F:4243
DF = MU(0x1.a400000000000p+8f,r_exp((-AD(PF,0x1.59999a0000000p+1f)))); // module_sf_noahlsm.F:4244
} else { // module_sf_noahlsm.F:4245
DF = 0x1.652bd40000000p-3f; // module_sf_noahlsm.F:4246
} // module_sf_noahlsm.F:4247
} // module_sf_noahlsm.F:4249
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4254-4356
static __attribute__((noinline)) void noah_tmpavg(thread NoahTables & tables, thread float & TAVG, thread float & TUP, thread float & TM, thread float & TDN, thread float * ZSOIL, thread int & NSOIL, thread int & K){
float DZ;
float DZH;
float T0 = 0x1.1126660000000p+8f;
float X0;
float XDN;
float XUP;
 // module_sf_noahlsm.F:4264
if ((K == 1)) { // module_sf_noahlsm.F:4283
DZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:4284
} else { // module_sf_noahlsm.F:4285
DZ = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:4286
} // module_sf_noahlsm.F:4287
DZH = MU(DZ,0x1.0000000000000p-1f); // module_sf_noahlsm.F:4289
if ((TUP < T0)) { // module_sf_noahlsm.F:4290
if ((TM < T0)) { // module_sf_noahlsm.F:4291
if ((TDN < T0)) { // module_sf_noahlsm.F:4295
TAVG = DV(AD(AD(TUP,MU(0x1.0000000000000p+1f,TM)),TDN),0x1.0000000000000p+2f); // module_sf_noahlsm.F:4296
} else { // module_sf_noahlsm.F:4300
X0 = DV(MU(SU(T0,TM),DZH),SU(TDN,TM)); // module_sf_noahlsm.F:4301
TAVG = DV(MU(0x1.0000000000000p-1f,AD(AD(MU(TUP,DZH),MU(TM,AD(DZH,X0))),MU(T0,SU(MU(0x1.0000000000000p+1f,DZH),X0)))),DZ); // module_sf_noahlsm.F:4302
} // module_sf_noahlsm.F:4304
} else { // module_sf_noahlsm.F:4305
if ((TDN < T0)) { // module_sf_noahlsm.F:4309
XUP = DV(MU(SU(T0,TUP),DZH),SU(TM,TUP)); // module_sf_noahlsm.F:4310
XDN = SU(DZH,DV(MU(SU(T0,TM),DZH),SU(TDN,TM))); // module_sf_noahlsm.F:4311
TAVG = DV(MU(0x1.0000000000000p-1f,AD(AD(MU(TUP,XUP),MU(T0,SU(SU(MU(0x1.0000000000000p+1f,DZ),XUP),XDN))),MU(TDN,XDN))),DZ); // module_sf_noahlsm.F:4312
} else { // module_sf_noahlsm.F:4317
XUP = DV(MU(SU(T0,TUP),DZH),SU(TM,TUP)); // module_sf_noahlsm.F:4318
TAVG = DV(MU(0x1.0000000000000p-1f,AD(MU(TUP,XUP),MU(T0,SU(MU(0x1.0000000000000p+1f,DZ),XUP)))),DZ); // module_sf_noahlsm.F:4319
} // module_sf_noahlsm.F:4320
} // module_sf_noahlsm.F:4321
} else { // module_sf_noahlsm.F:4322
if ((TM < T0)) { // module_sf_noahlsm.F:4323
if ((TDN < T0)) { // module_sf_noahlsm.F:4327
XUP = SU(DZH,DV(MU(SU(T0,TUP),DZH),SU(TM,TUP))); // module_sf_noahlsm.F:4328
TAVG = DV(MU(0x1.0000000000000p-1f,AD(AD(MU(T0,SU(DZ,XUP)),MU(TM,AD(DZH,XUP))),MU(TDN,DZH))),DZ); // module_sf_noahlsm.F:4329
} else { // module_sf_noahlsm.F:4334
XUP = SU(DZH,DV(MU(SU(T0,TUP),DZH),SU(TM,TUP))); // module_sf_noahlsm.F:4335
XDN = DV(MU(SU(T0,TM),DZH),SU(TDN,TM)); // module_sf_noahlsm.F:4336
TAVG = DV(MU(0x1.0000000000000p-1f,AD(MU(T0,SU(SU(MU(0x1.0000000000000p+1f,DZ),XUP),XDN)),MU(TM,AD(XUP,XDN)))),DZ); // module_sf_noahlsm.F:4337
} // module_sf_noahlsm.F:4339
} else { // module_sf_noahlsm.F:4340
if ((TDN < T0)) { // module_sf_noahlsm.F:4344
XDN = SU(DZH,DV(MU(SU(T0,TM),DZH),SU(TDN,TM))); // module_sf_noahlsm.F:4345
TAVG = DV(AD(MU(T0,SU(DZ,XDN)),MU(MU(0x1.0000000000000p-1f,AD(T0,TDN)),XDN)),DZ); // module_sf_noahlsm.F:4346
} else { // module_sf_noahlsm.F:4350
TAVG = DV(AD(AD(TUP,MU(0x1.0000000000000p+1f,TM)),TDN),0x1.0000000000000p+2f); // module_sf_noahlsm.F:4351
} // module_sf_noahlsm.F:4352
} // module_sf_noahlsm.F:4353
} // module_sf_noahlsm.F:4354
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4359-4462
static __attribute__((noinline)) void noah_transp(thread NoahTables & tables, thread float * ET, thread int & NSOIL, thread float & ETP1, thread float * SMC, thread float & CMC, thread float * ZSOIL, thread float & SHDFAC, thread float & SMCWLT, thread float & CMCMAX, thread float & PC, thread float & CFACTR, thread float & SMCREF, thread float & SFCTMP, thread float & Q2, thread int & NROOT, thread float * RTDIS){
int I;
int K;
float DENOM;
float ETP1A;
float GX[20];
float RTX;
float SGX;
 // module_sf_noahlsm.F:4368
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:4398
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:4399
} // module_sf_noahlsm.F:4407
if ((CMC != 0x0.0p+0f)) { // module_sf_noahlsm.F:4408
ETP1A = MU(MU(MU(SHDFAC,PC),ETP1),SU(0x1.0000000000000p+0f,r_pow(DV(CMC,CMCMAX),CFACTR))); // module_sf_noahlsm.F:4409
} else { // module_sf_noahlsm.F:4410
ETP1A = MU(MU(SHDFAC,PC),ETP1); // module_sf_noahlsm.F:4411
} // module_sf_noahlsm.F:4412
SGX = 0x0.0p+0f; // module_sf_noahlsm.F:4413
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:4414
GX[(I)-(1)] = DV(SU(SMC[(I)-(1)],SMCWLT),SU(SMCREF,SMCWLT)); // module_sf_noahlsm.F:4415
GX[(I)-(1)] = noah_max(noah_min(GX[(I)-(1)],0x1.0000000000000p+0f),0x0.0p+0f); // module_sf_noahlsm.F:4416
SGX = AD(SGX,GX[(I)-(1)]); // module_sf_noahlsm.F:4417
} // module_sf_noahlsm.F:4418
SGX = DV(SGX,NROOT); // module_sf_noahlsm.F:4420
DENOM = 0x0.0p+0f; // module_sf_noahlsm.F:4421
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:4422
RTX = SU(AD(RTDIS[(I)-(1)],GX[(I)-(1)]),SGX); // module_sf_noahlsm.F:4423
GX[(I)-(1)] = MU(GX[(I)-(1)],noah_max(RTX,0x0.0p+0f)); // module_sf_noahlsm.F:4424
DENOM = AD(DENOM,GX[(I)-(1)]); // module_sf_noahlsm.F:4425
} // module_sf_noahlsm.F:4426
if ((DENOM <= 0x0.0p+0f)) { DENOM = 0x1.0000000000000p+0f; } // module_sf_noahlsm.F:4428
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:4429
ET[(I)-(1)] = DV(MU(ETP1A,GX[(I)-(1)]),DENOM); // module_sf_noahlsm.F:4430
} // module_sf_noahlsm.F:4460
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4465-4523
static __attribute__((noinline)) void noah_wdfcnd(thread NoahTables & tables, thread float & WDF, thread float & WCND, thread float & SMC, thread float & SMCMAX, thread float & BEXP, thread float & DKSAT, thread float & DWSAT, thread float & SICEMAX){
float EXPON;
float FACTR1;
float FACTR2;
float VKWGT;
 // module_sf_noahlsm.F:4473
FACTR1 = DV(0x1.99999a0000000p-5f,SMCMAX); // module_sf_noahlsm.F:4490
FACTR2 = DV(SMC,SMCMAX); // module_sf_noahlsm.F:4495
FACTR1 = noah_min(FACTR1,FACTR2); // module_sf_noahlsm.F:4496
EXPON = AD(BEXP,0x1.0000000000000p+1f); // module_sf_noahlsm.F:4497
WDF = MU(DWSAT,r_pow(FACTR2,EXPON)); // module_sf_noahlsm.F:4511
if ((SICEMAX > 0x0.0p+0f)) { // module_sf_noahlsm.F:4512
VKWGT = DV(0x1.0000000000000p+0f,AD(0x1.0000000000000p+0f,r_pow(MU(0x1.f400000000000p+8f,SICEMAX),0x1.8000000000000p+1f))); // module_sf_noahlsm.F:4513
WDF = AD(MU(VKWGT,WDF),MU(MU(SU(0x1.0000000000000p+0f,VKWGT),DWSAT),r_pow(FACTR1,EXPON))); // module_sf_noahlsm.F:4514
} // module_sf_noahlsm.F:4518
EXPON = AD(MU(0x1.0000000000000p+1f,BEXP),0x1.8000000000000p+1f); // module_sf_noahlsm.F:4519
WCND = MU(DKSAT,r_pow(FACTR2,EXPON)); // module_sf_noahlsm.F:4520
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4526-4760

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:35-413

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:416-527

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:530-566

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:569-711

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:714-759

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:762-825

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:828-857

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:860-1077

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1080-1203

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1206-1230

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1233-1281

// MPAS v8.4.2 physics_wrf/module_sf_noahdrv.F:44-1590; MPAS urban=0 slice.

// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207. FNDSNOWH=.true. in MPAS.


kernel void hexv842_mpas_noah_land_f32(
constant Hexv842MpasNoahLandF32Resources &r [[buffer(0)]],
constant Hexv842MpasNoahLandF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
if(c>=uint(p.n)) return;
thread float x[110]; thread int ix[14]; for(int j=0;j<110;j++) x[j]=r.floats[c*110+j]; for(int j=0;j<14;j++) ix[j]=r.integers[c*14+j];
thread NoahTables tables; thread uint *t=reinterpret_cast<thread uint*>(&tables); for(uint j=0;j<sizeof(NoahTables)/4;j++)t[j]=r.tables[j]; tables.error=0;
// Four packed soil slots: prevent out-of-bounds module_sf_noahlsm.F:426 access.
if(ix[3]!=4){r.status[c]=426;return;}
if(ix[5]<1 || ix[6]<1 || ix[7]<1){r.status[c]=-1;return;}
int IILOC_arg=ix[0];int JJLOC_arg=ix[1];float FFROZP_arg=x[0];int ISURBAN_arg=ix[2];float DT_arg=x[1];float ZLVL_arg=x[2];int NSOIL_arg=ix[3];bool LOCAL_arg=ix[4];float LWDN_arg=x[7];float SOLDN_arg=x[8];float SOLNET_arg=x[9];float SFCPRS_arg=x[10];float PRCP_arg=x[11];float SFCTMP_arg=x[12];float Q2_arg=x[13];float SFCSPD_arg=x[14];float COSZ_arg=x[15];float PRCPRAIN_arg=x[16];float SOLARDIRECT_arg=x[17];float TH2_arg=x[18];float Q2SAT_arg=x[19];float DQSDT2_arg=x[20];int VEGTYP_arg=ix[5];int SOILTYP_arg=ix[6];int SLOPETYP_arg=ix[7];float SHDFAC_arg=x[21];float SHDMIN_arg=x[22];float SHDMAX_arg=x[23];float ALB_arg=x[24];float SNOALB_arg=x[25];float TBOT_arg=x[26];float Z0BRD_arg=x[27];float Z0_arg=x[28];float EMISSI_arg=x[29];float EMBRD_arg=x[30];float CMC_arg=x[31];float T1_arg=x[32];float SNOWH_arg=x[45];float SNEQV_arg=x[46];float ALBEDO_arg=x[47];float CH_arg=x[48];float CM_arg=x[49];float ETA_arg=x[50];float SHEAT_arg=x[51];float ETA_KINEMATIC_arg=x[52];float FDOWN_arg=x[53];float EC_arg=x[54];float EDIR_arg=x[55];float ETT_arg=x[60];float ESNOW_arg=x[61];float DRIP_arg=x[62];float DEW_arg=x[63];float BETA_arg=x[64];float ETP_arg=x[65];float SSOIL_arg=x[66];float FLX1_arg=x[67];float FLX2_arg=x[68];float FLX3_arg=x[69];float FLX4_arg=x[70];float FVB_arg=x[71];float FBUR_arg=x[72];float FGSN_arg=x[73];bool UA_PHYS_arg=ix[8];float SNOMLT_arg=x[74];float SNCOVR_arg=x[75];float RUNOFF1_arg=x[76];float RUNOFF2_arg=x[77];float RUNOFF3_arg=x[78];float RC_arg=x[79];float PC_arg=x[80];float RSMIN_arg=x[81];float XLAI_arg=x[82];float RCS_arg=x[83];float RCT_arg=x[84];float RCQ_arg=x[85];float RCSOIL_arg=x[86];float SOILW_arg=x[87];float SOILM_arg=x[88];float Q1_arg=x[89];bool RDLAI2D_arg=ix[9];bool USEMONALB_arg=ix[10];float SNOTIME1_arg=x[94];float RIBB_arg=x[95];float SMCWLT_arg=x[96];float SMCDRY_arg=x[97];float SMCREF_arg=x[98];float SMCMAX_arg=x[99];int NROOT_arg=ix[11];float SFHEAD1RT_arg=x[100];float INFXS1RT_arg=x[101];float ETPND1_arg=x[102];int OPT_THCND_arg=ix[12];float AOASIS_arg=x[103];float XSDA_QFX_arg=x[104];float HFX_PHY_arg=x[105];float QFX_PHY_arg=x[106];float XQNORM_arg=x[107];int FASDAS_arg=ix[13];float HCPCT_FASDAS_arg=x[108];float IRRIGATION_CHANNEL_arg=x[109];
noah_sflx(tables,IILOC_arg,JJLOC_arg,FFROZP_arg,ISURBAN_arg,DT_arg,ZLVL_arg,NSOIL_arg,x+3,LOCAL_arg,tables.LUTYPE,tables.SLTYPE,LWDN_arg,SOLDN_arg,SOLNET_arg,SFCPRS_arg,PRCP_arg,SFCTMP_arg,Q2_arg,SFCSPD_arg,COSZ_arg,PRCPRAIN_arg,SOLARDIRECT_arg,TH2_arg,Q2SAT_arg,DQSDT2_arg,VEGTYP_arg,SOILTYP_arg,SLOPETYP_arg,SHDFAC_arg,SHDMIN_arg,SHDMAX_arg,ALB_arg,SNOALB_arg,TBOT_arg,Z0BRD_arg,Z0_arg,EMISSI_arg,EMBRD_arg,CMC_arg,T1_arg,x+33,x+37,x+41,SNOWH_arg,SNEQV_arg,ALBEDO_arg,CH_arg,CM_arg,ETA_arg,SHEAT_arg,ETA_KINEMATIC_arg,FDOWN_arg,EC_arg,EDIR_arg,x+56,ETT_arg,ESNOW_arg,DRIP_arg,DEW_arg,BETA_arg,ETP_arg,SSOIL_arg,FLX1_arg,FLX2_arg,FLX3_arg,FLX4_arg,FVB_arg,FBUR_arg,FGSN_arg,UA_PHYS_arg,SNOMLT_arg,SNCOVR_arg,RUNOFF1_arg,RUNOFF2_arg,RUNOFF3_arg,RC_arg,PC_arg,RSMIN_arg,XLAI_arg,RCS_arg,RCT_arg,RCQ_arg,RCSOIL_arg,SOILW_arg,SOILM_arg,Q1_arg,x+90,RDLAI2D_arg,USEMONALB_arg,SNOTIME1_arg,RIBB_arg,SMCWLT_arg,SMCDRY_arg,SMCREF_arg,SMCMAX_arg,NROOT_arg,SFHEAD1RT_arg,INFXS1RT_arg,ETPND1_arg,OPT_THCND_arg,AOASIS_arg,XSDA_QFX_arg,HFX_PHY_arg,QFX_PHY_arg,XQNORM_arg,FASDAS_arg,HCPCT_FASDAS_arg,IRRIGATION_CHANNEL_arg,true);
ix[0]=IILOC_arg;ix[1]=JJLOC_arg;x[0]=FFROZP_arg;ix[2]=ISURBAN_arg;x[1]=DT_arg;x[2]=ZLVL_arg;ix[3]=NSOIL_arg;ix[4]=LOCAL_arg;x[7]=LWDN_arg;x[8]=SOLDN_arg;x[9]=SOLNET_arg;x[10]=SFCPRS_arg;x[11]=PRCP_arg;x[12]=SFCTMP_arg;x[13]=Q2_arg;x[14]=SFCSPD_arg;x[15]=COSZ_arg;x[16]=PRCPRAIN_arg;x[17]=SOLARDIRECT_arg;x[18]=TH2_arg;x[19]=Q2SAT_arg;x[20]=DQSDT2_arg;ix[5]=VEGTYP_arg;ix[6]=SOILTYP_arg;ix[7]=SLOPETYP_arg;x[21]=SHDFAC_arg;x[22]=SHDMIN_arg;x[23]=SHDMAX_arg;x[24]=ALB_arg;x[25]=SNOALB_arg;x[26]=TBOT_arg;x[27]=Z0BRD_arg;x[28]=Z0_arg;x[29]=EMISSI_arg;x[30]=EMBRD_arg;x[31]=CMC_arg;x[32]=T1_arg;x[45]=SNOWH_arg;x[46]=SNEQV_arg;x[47]=ALBEDO_arg;x[48]=CH_arg;x[49]=CM_arg;x[50]=ETA_arg;x[51]=SHEAT_arg;x[52]=ETA_KINEMATIC_arg;x[53]=FDOWN_arg;x[54]=EC_arg;x[55]=EDIR_arg;x[60]=ETT_arg;x[61]=ESNOW_arg;x[62]=DRIP_arg;x[63]=DEW_arg;x[64]=BETA_arg;x[65]=ETP_arg;x[66]=SSOIL_arg;x[67]=FLX1_arg;x[68]=FLX2_arg;x[69]=FLX3_arg;x[70]=FLX4_arg;x[71]=FVB_arg;x[72]=FBUR_arg;x[73]=FGSN_arg;ix[8]=UA_PHYS_arg;x[74]=SNOMLT_arg;x[75]=SNCOVR_arg;x[76]=RUNOFF1_arg;x[77]=RUNOFF2_arg;x[78]=RUNOFF3_arg;x[79]=RC_arg;x[80]=PC_arg;x[81]=RSMIN_arg;x[82]=XLAI_arg;x[83]=RCS_arg;x[84]=RCT_arg;x[85]=RCQ_arg;x[86]=RCSOIL_arg;x[87]=SOILW_arg;x[88]=SOILM_arg;x[89]=Q1_arg;ix[9]=RDLAI2D_arg;ix[10]=USEMONALB_arg;x[94]=SNOTIME1_arg;x[95]=RIBB_arg;x[96]=SMCWLT_arg;x[97]=SMCDRY_arg;x[98]=SMCREF_arg;x[99]=SMCMAX_arg;ix[11]=NROOT_arg;x[100]=SFHEAD1RT_arg;x[101]=INFXS1RT_arg;x[102]=ETPND1_arg;ix[12]=OPT_THCND_arg;x[103]=AOASIS_arg;x[104]=XSDA_QFX_arg;x[105]=HFX_PHY_arg;x[106]=QFX_PHY_arg;x[107]=XQNORM_arg;ix[13]=FASDAS_arg;x[108]=HCPCT_FASDAS_arg;x[109]=IRRIGATION_CHANNEL_arg;
r.status[c]=tables.error;

for(int j=0;j<110;j++)r.floats[c*110+j]=x[j]; for(int j=0;j<14;j++)r.integers[c*14+j]=ix[j];
}
