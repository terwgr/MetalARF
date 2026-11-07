#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_noah_glacier_f32_bindings.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define MU(a,b) mpas_mul(a,b)
#define DV(a,b) cu_div(a,b)
static __attribute__((noinline)) float r_log(float x){return gfk_log(x);}
static __attribute__((noinline)) float r_exp(float x){return gfk_exp(x);}
static __attribute__((noinline)) float r_pow(float x,float y){return gfk_pow(x,y);}

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

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:895-1010

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1013-1149

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1152-1191

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1193-1232

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1234-1324

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1327-1425

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1428-1448

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1451-1589

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1592-1854

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1857-1905

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1908-2196

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2199-2316

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2319-2539

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

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2673-2817

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2821-2923

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2926-3011

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3014-3417

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3421-3553

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3556-3601

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3605-3654

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3657-3951

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3954-4081

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4084-4124

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4128-4251

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4254-4356

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4359-4462

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4465-4523

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4526-4760

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:35-413
static __attribute__((noinline)) void gl_sflx_glacial(thread NoahTables & tables, thread int & IILOC, thread int & JJLOC, thread int & ISICE, thread float & FFROZP, thread float & DT, thread float & ZLVL, thread int & NSOIL, thread float * SLDPTH, thread float & LWDN, thread float & SOLNET, thread float & SFCPRS, thread float & PRCP, thread float & SFCTMP, thread float & Q2, thread float & TH2, thread float & Q2SAT, thread float & DQSDT2, thread float & ALB, thread float & SNOALB, thread float & TBOT, thread float & Z0BRD, thread float & Z0, thread float & EMISSI, thread float & EMBRD, thread float & T1, thread float * STC, thread float & SNOWH, thread float & SNEQV, thread float & ALBEDO, thread float & CH, thread float & ETA, thread float & SHEAT, thread float & ETA_KINEMATIC, thread float & FDOWN, thread float & ESNOW, thread float & DEW, thread float & ETP, thread float & SSOIL, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & SNOMLT, thread float & SNCOVR, thread float & RUNOFF1, thread float & Q1, thread float & SNOTIME1, thread float & RIBB){
thread auto& RD=tables.RD;
thread auto& ILOC=tables.ILOC;
thread auto& JLOC=tables.JLOC;
thread auto& CP=tables.CP;
bool FRZGRA;
bool SNOWNG;
int KZ;
float ZSOIL[20];
float DF1;
float DSOIL;
float DTOT;
float FRCSNO;
float FRCSOI;
float PRCP1;
float RCH;
float RR;
float RSNOW;
float SNDENS;
float SNCOND;
float SN_NEW;
float T1V;
float T24;
float T2V;
float TH2V;
float TSNOW;
float PRCPF;
float RHO;
float TFREEZ = 0x1.1126660000000p+8f;
float LVH2O = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
float R = 0x1.1f0a3e0000000p+8f;
 // module_sf_noahlsm_glacial_only.F:165
ILOC = IILOC; // module_sf_noahlsm_glacial_only.F:209
JLOC = JJLOC; // module_sf_noahlsm_glacial_only.F:210
ZSOIL[(1)-(1)] = (-SLDPTH[(1)-(1)]); // module_sf_noahlsm_glacial_only.F:212
for (KZ=2; KZ<=NSOIL; KZ+=1) { // module_sf_noahlsm_glacial_only.F:213
ZSOIL[(KZ)-(1)] = AD((-SLDPTH[(KZ)-(1)]),ZSOIL[((KZ - 1))-(1)]); // module_sf_noahlsm_glacial_only.F:214
} // module_sf_noahlsm_glacial_only.F:215
if ((SNEQV < 0x1.99999a0000000p-4f)) { // module_sf_noahlsm_glacial_only.F:221
SNEQV = 0x1.99999a0000000p-4f; // module_sf_noahlsm_glacial_only.F:222
SNOWH = 0x1.0000000000000p-1f; // module_sf_noahlsm_glacial_only.F:223
} // module_sf_noahlsm_glacial_only.F:224
SNDENS = DV(SNEQV,SNOWH); // module_sf_noahlsm_glacial_only.F:229
if ((SNDENS > 0x1.0000000000000p+0f)) { // module_sf_noahlsm_glacial_only.F:230
tables.error = 231; return; // module_sf_noahlsm_glacial_only.F:231
} // module_sf_noahlsm_glacial_only.F:232
{  gl_csnow(tables,SNCOND,SNDENS); } // module_sf_noahlsm_glacial_only.F:234
SNOWNG = false; // module_sf_noahlsm_glacial_only.F:242
FRZGRA = false; // module_sf_noahlsm_glacial_only.F:243
if ((PRCP > 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:244
if ((FFROZP > 0x1.0000000000000p-1f)) { // module_sf_noahlsm_glacial_only.F:249
SNOWNG = true; // module_sf_noahlsm_glacial_only.F:250
} else { // module_sf_noahlsm_glacial_only.F:251
if ((T1 <= TFREEZ)) { FRZGRA = true; } // module_sf_noahlsm_glacial_only.F:252
} // module_sf_noahlsm_glacial_only.F:253
} // module_sf_noahlsm_glacial_only.F:254
if ((SNOWNG || FRZGRA)) { // module_sf_noahlsm_glacial_only.F:262
SN_NEW = MU(MU(PRCP,DT),0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:263
SNEQV = AD(SNEQV,SN_NEW); // module_sf_noahlsm_glacial_only.F:264
PRCPF = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:265
{  gl_snow_new(tables,SFCTMP,SN_NEW,SNOWH,SNDENS); } // module_sf_noahlsm_glacial_only.F:271
if ((SNCOVR > 0x1.fae1480000000p-1f)) { // module_sf_noahlsm_glacial_only.F:279
if ((STC[(1)-(1)] < SU(TFREEZ,0x1.4000000000000p+2f))) { SNDENS = 0x1.99999a0000000p-3f; } // module_sf_noahlsm_glacial_only.F:280
if (((SNOWNG && (T1 < 0x1.1100000000000p+8f)) && (SFCTMP < 0x1.1100000000000p+8f))) { SNDENS = 0x1.99999a0000000p-3f; } // module_sf_noahlsm_glacial_only.F:281
} // module_sf_noahlsm_glacial_only.F:282
{  gl_csnow(tables,SNCOND,SNDENS); } // module_sf_noahlsm_glacial_only.F:284
} else { // module_sf_noahlsm_glacial_only.F:290
PRCPF = PRCP; // module_sf_noahlsm_glacial_only.F:291
} // module_sf_noahlsm_glacial_only.F:292
SNCOVR = 0x1.0000000000000p+0f; // module_sf_noahlsm_glacial_only.F:299
{  gl_alcalc(tables,ALB,SNOALB,EMBRD,T1,ALBEDO,EMISSI,DT,SNOWNG,SNOTIME1); } // module_sf_noahlsm_glacial_only.F:305
DF1 = SNCOND; // module_sf_noahlsm_glacial_only.F:311
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm_glacial_only.F:313
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm_glacial_only.F:314
FRCSNO = DV(SNOWH,DTOT); // module_sf_noahlsm_glacial_only.F:315
FRCSOI = DV(DSOIL,DTOT); // module_sf_noahlsm_glacial_only.F:319
DF1 = AD(MU(FRCSNO,SNCOND),MU(FRCSOI,DF1)); // module_sf_noahlsm_glacial_only.F:323
if ((DTOT > MU(0x1.0000000000000p+1f,DSOIL))) { // module_sf_noahlsm_glacial_only.F:330
DTOT = MU(0x1.0000000000000p+1f,DSOIL); // module_sf_noahlsm_glacial_only.F:331
} // module_sf_noahlsm_glacial_only.F:332
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm_glacial_only.F:333
{  gl_snowz0(tables,Z0,Z0BRD,SNOWH); } // module_sf_noahlsm_glacial_only.F:340
FDOWN = AD(SOLNET,LWDN); // module_sf_noahlsm_glacial_only.F:347
T2V = MU(SFCTMP,AD(0x1.0000000000000p+0f,MU(0x1.3851ec0000000p-1f,Q2))); // module_sf_noahlsm_glacial_only.F:354
RHO = DV(SFCPRS,MU(RD,T2V)); // module_sf_noahlsm_glacial_only.F:355
RCH = MU(MU(RHO,0x1.f64ccc0000000p+9f),CH); // module_sf_noahlsm_glacial_only.F:356
T24 = MU(MU(MU(SFCTMP,SFCTMP),SFCTMP),SFCTMP); // module_sf_noahlsm_glacial_only.F:357
{  gl_penman(tables,SFCTMP,SFCPRS,CH,TH2,PRCP,FDOWN,T24,SSOIL,Q2,Q2SAT,ETP,RCH,RR,SNOWNG,FRZGRA,DQSDT2,FLX2,EMISSI,T1); } // module_sf_noahlsm_glacial_only.F:366
{  gl_snopac(tables,ETP,ETA,PRCP,PRCPF,SNOWNG,NSOIL,DT,DF1,Q2,T1,SFCTMP,T24,TH2,FDOWN,SSOIL,STC,SFCPRS,RCH,RR,SNEQV,SNDENS,SNOWH,ZSOIL,TBOT,SNOMLT,DEW,FLX1,FLX2,FLX3,ESNOW,EMISSI,RIBB); } // module_sf_noahlsm_glacial_only.F:370
ETA_KINEMATIC = ETP; // module_sf_noahlsm_glacial_only.F:376
Q1 = AD(Q2,DV(MU(ETA_KINEMATIC,CP),RCH)); // module_sf_noahlsm_glacial_only.F:381
SHEAT = MU(DV((-MU(MU(CH,CP),SFCPRS)),MU(R,T2V)),SU(TH2,T1)); // module_sf_noahlsm_glacial_only.F:386
ESNOW = MU(ESNOW,LSUBS); // module_sf_noahlsm_glacial_only.F:391
ETP = MU(ETP,LSUBS); // module_sf_noahlsm_glacial_only.F:392
if ((ETP > 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:393
ETA = ESNOW; // module_sf_noahlsm_glacial_only.F:394
} else { // module_sf_noahlsm_glacial_only.F:395
ETA = ETP; // module_sf_noahlsm_glacial_only.F:396
} // module_sf_noahlsm_glacial_only.F:397
SSOIL = MU(-0x1.0000000000000p+0f,SSOIL); // module_sf_noahlsm_glacial_only.F:404
RUNOFF1 = DV(SNOMLT,DT); // module_sf_noahlsm_glacial_only.F:410
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:416-527
static __attribute__((noinline)) void gl_alcalc(thread NoahTables & tables, thread float & ALB, thread float & SNOALB, thread float & EMBRD, thread float & TSNOW, thread float & ALBEDO, thread float & EMISSI, thread float & DT, thread bool & SNOWNG, thread float & SNOTIME1){
thread auto& EMISSI_S=tables.EMISSI_S;
thread auto& LVCOEF_DATA=tables.LVCOEF_DATA;
float SNOALB2;
float TM;
float SNOALB1;
float SNACCA = 0x1.e147ae0000000p-1f;
float SNACCB = 0x1.28f5c20000000p-1f;
float SNTHWA = 0x1.a3d70a0000000p-1f;
float SNTHWB = 0x1.d70a3e0000000p-2f;
 // module_sf_noahlsm_glacial_only.F:426
ALBEDO = AD(ALB,SU(SNOALB,ALB)); // module_sf_noahlsm_glacial_only.F:445
EMISSI = AD(EMBRD,SU(EMISSI_S,EMBRD)); // module_sf_noahlsm_glacial_only.F:446
SNOALB1 = AD(SNOALB,MU(LVCOEF_DATA,SU(0x1.b333340000000p-1f,SNOALB))); // module_sf_noahlsm_glacial_only.F:500
SNOALB2 = SNOALB1; // module_sf_noahlsm_glacial_only.F:501
if (SNOWNG) { // module_sf_noahlsm_glacial_only.F:503
SNOTIME1 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:504
} else { // module_sf_noahlsm_glacial_only.F:505
SNOTIME1 = AD(SNOTIME1,DT); // module_sf_noahlsm_glacial_only.F:506
SNOALB2 = MU(SNOALB1,r_pow(SNACCA,r_pow(DV(SNOTIME1,0x1.5180000000000p+16f),SNACCB))); // module_sf_noahlsm_glacial_only.F:508
} // module_sf_noahlsm_glacial_only.F:512
SNOALB2 = noah_max(SNOALB2,ALB); // module_sf_noahlsm_glacial_only.F:514
ALBEDO = AD(ALB,SU(SNOALB2,ALB)); // module_sf_noahlsm_glacial_only.F:515
if ((ALBEDO > SNOALB2)) { ALBEDO = SNOALB2; } // module_sf_noahlsm_glacial_only.F:516
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:530-566
static __attribute__((noinline)) void gl_csnow(thread NoahTables & tables, thread float & SNCOND, thread float & DSNOW){
float C;
float UNIT = 0x1.dc67e00000000p-4f;
 // module_sf_noahlsm_glacial_only.F:535
C = MU(0x1.4fdf3c0000000p-2f,r_pow(10,MU(0x1.2000000000000p+1f,DSNOW))); // module_sf_noahlsm_glacial_only.F:546
SNCOND = MU(MU(0x1.0000000000000p+1f,UNIT),C); // module_sf_noahlsm_glacial_only.F:563
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:569-711
static __attribute__((noinline)) void gl_hrtice(thread NoahTables & tables, thread float * RHSTS, thread float * STC, thread float & TBOT, thread int & NSOIL, thread float * ZSOIL, thread float & YY, thread float & ZZ1, thread float & DF1, thread float * AI, thread float * BI, thread float * CI){
int K;
float DDZ;
float DDZ2;
float DENOM;
float DTSDZ;
float DTSDZ2;
float SSOIL;
float HCPCT;
float DF1K;
float DF1N;
float ZMD;
float ZBOT = -0x1.9000000000000p+4f;
 // module_sf_noahlsm_glacial_only.F:580
HCPCT = MU(0x1.e848000000000p+19f,SU(0x1.a388660000000p-1f,MU(0x1.0c154c0000000p-4f,ZSOIL[(1)-(1)]))); // module_sf_noahlsm_glacial_only.F:604
DF1K = DF1; // module_sf_noahlsm_glacial_only.F:605
DDZ = DV(0x1.0000000000000p+0f,MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm_glacial_only.F:619
AI[(1)-(1)] = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:620
CI[(1)-(1)] = DV(MU(DF1,DDZ),MU(ZSOIL[(1)-(1)],HCPCT)); // module_sf_noahlsm_glacial_only.F:621
BI[(1)-(1)] = AD((-CI[(1)-(1)]),DV(DF1,MU(MU(MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZSOIL[(1)-(1)]),HCPCT),ZZ1))); // module_sf_noahlsm_glacial_only.F:628
DTSDZ = DV(SU(STC[(1)-(1)],STC[(2)-(1)]),MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm_glacial_only.F:630
SSOIL = DV(MU(DF1,SU(STC[(1)-(1)],YY)),MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZZ1)); // module_sf_noahlsm_glacial_only.F:631
RHSTS[(1)-(1)] = DV(SU(MU(DF1,DTSDZ),SSOIL),MU(ZSOIL[(1)-(1)],HCPCT)); // module_sf_noahlsm_glacial_only.F:636
DDZ2 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:641
DF1K = DF1; // module_sf_noahlsm_glacial_only.F:642
DF1N = DF1; // module_sf_noahlsm_glacial_only.F:643
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:644
ZMD = MU(0x1.0000000000000p-1f,AD(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)])); // module_sf_noahlsm_glacial_only.F:646
HCPCT = MU(0x1.e848000000000p+19f,SU(0x1.a388660000000p-1f,MU(0x1.0c154c0000000p-3f,ZMD))); // module_sf_noahlsm_glacial_only.F:656
DF1N = SU(0x1.4b17040000000p-2f,MU(0x1.9c97100000000p-4f,ZMD)); // module_sf_noahlsm_glacial_only.F:665
if ((K != NSOIL)) { // module_sf_noahlsm_glacial_only.F:669
DENOM = MU(0x1.0000000000000p-1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm_glacial_only.F:670
DTSDZ2 = DV(SU(STC[(K)-(1)],STC[((K + 1))-(1)]),DENOM); // module_sf_noahlsm_glacial_only.F:675
DDZ2 = DV(0x1.0000000000000p+1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm_glacial_only.F:676
CI[(K)-(1)] = DV(MU((-DF1N),DDZ2),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm_glacial_only.F:677
} else { // module_sf_noahlsm_glacial_only.F:682
DTSDZ2 = DV(SU(STC[(K)-(1)],TBOT),SU(MU(0x1.0000000000000p-1f,AD(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)])),ZBOT)); // module_sf_noahlsm_glacial_only.F:687
CI[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:689
} // module_sf_noahlsm_glacial_only.F:693
DENOM = MU(SU(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)]),HCPCT); // module_sf_noahlsm_glacial_only.F:694
RHSTS[(K)-(1)] = DV(SU(MU(DF1N,DTSDZ2),MU(DF1K,DTSDZ)),DENOM); // module_sf_noahlsm_glacial_only.F:699
AI[(K)-(1)] = DV(MU((-DF1K),DDZ),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm_glacial_only.F:700
BI[(K)-(1)] = (-AD(AI[(K)-(1)],CI[(K)-(1)])); // module_sf_noahlsm_glacial_only.F:705
DF1K = DF1N; // module_sf_noahlsm_glacial_only.F:706
DTSDZ = DTSDZ2; // module_sf_noahlsm_glacial_only.F:707
DDZ = DDZ2; // module_sf_noahlsm_glacial_only.F:708
} // module_sf_noahlsm_glacial_only.F:709
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:714-759
static __attribute__((noinline)) void gl_hstep(thread NoahTables & tables, thread float * STCOUT, thread float * STCIN, thread float * RHSTS, thread float & DT, thread int & NSOIL, thread float * AI, thread float * BI, thread float * CI){
float RHSTSIN[20];
float CIIN[20];
int K;
 // module_sf_noahlsm_glacial_only.F:719
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:733
RHSTS[(K)-(1)] = MU(RHSTS[(K)-(1)],DT); // module_sf_noahlsm_glacial_only.F:734
AI[(K)-(1)] = MU(AI[(K)-(1)],DT); // module_sf_noahlsm_glacial_only.F:735
BI[(K)-(1)] = AD(0x1.0000000000000p+0f,MU(BI[(K)-(1)],DT)); // module_sf_noahlsm_glacial_only.F:736
CI[(K)-(1)] = MU(CI[(K)-(1)],DT); // module_sf_noahlsm_glacial_only.F:737
} // module_sf_noahlsm_glacial_only.F:738
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:742
RHSTSIN[(K)-(1)] = RHSTS[(K)-(1)]; // module_sf_noahlsm_glacial_only.F:743
} // module_sf_noahlsm_glacial_only.F:744
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:745
CIIN[(K)-(1)] = CI[(K)-(1)]; // module_sf_noahlsm_glacial_only.F:746
} // module_sf_noahlsm_glacial_only.F:747
{  noah_rosr12(tables,CI,AI,BI,CIIN,RHSTSIN,RHSTS,NSOIL); if(tables.error)return; } // module_sf_noahlsm_glacial_only.F:751
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:755
STCOUT[(K)-(1)] = AD(STCIN[(K)-(1)],CI[(K)-(1)]); // module_sf_noahlsm_glacial_only.F:756
} // module_sf_noahlsm_glacial_only.F:757
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:762-825
static __attribute__((noinline)) void gl_penman(thread NoahTables & tables, thread float & SFCTMP, thread float & SFCPRS, thread float & CH, thread float & TH2, thread float & PRCP, thread float & FDOWN, thread float & T24, thread float & SSOIL, thread float & Q2, thread float & Q2SAT, thread float & ETP, thread float & RCH, thread float & RR, thread bool & SNOWNG, thread bool & FRZGRA, thread float & DQSDT2, thread float & FLX2, thread float & EMISSI, thread float & T1){
thread auto& SIGMA=tables.SIGMA;
thread auto& CPH2O=tables.CPH2O;
thread auto& CPICE=tables.CPICE;
thread auto& LSUBF=tables.LSUBF;
float A;
float DELTA;
float FNET;
float RAD;
float ELCP1;
float LVS;
float EPSCA;
float ELCP = 0x1.37199a0000000p+11f;
float LSUBC = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
 // module_sf_noahlsm_glacial_only.F:771
if ((T1 > 0x1.1126660000000p+8f)) { // module_sf_noahlsm_glacial_only.F:785
ELCP1 = ELCP; // module_sf_noahlsm_glacial_only.F:786
LVS = LSUBC; // module_sf_noahlsm_glacial_only.F:787
} else { // module_sf_noahlsm_glacial_only.F:788
ELCP1 = DV(MU(ELCP,LSUBS),LSUBC); // module_sf_noahlsm_glacial_only.F:789
LVS = LSUBS; // module_sf_noahlsm_glacial_only.F:790
} // module_sf_noahlsm_glacial_only.F:791
DELTA = MU(ELCP1,DQSDT2); // module_sf_noahlsm_glacial_only.F:792
A = MU(ELCP1,SU(Q2SAT,Q2)); // module_sf_noahlsm_glacial_only.F:793
RR = AD(DV(MU(MU(EMISSI,T24),0x1.16505a0000000p-24f),MU(SFCPRS,CH)),0x1.0000000000000p+0f); // module_sf_noahlsm_glacial_only.F:794
if ((!SNOWNG)) { // module_sf_noahlsm_glacial_only.F:800
if ((PRCP > 0x0.0p+0f)) { RR = AD(RR,DV(MU(CPH2O,PRCP),RCH)); } // module_sf_noahlsm_glacial_only.F:801
} else { // module_sf_noahlsm_glacial_only.F:802
RR = AD(RR,DV(MU(CPICE,PRCP),RCH)); // module_sf_noahlsm_glacial_only.F:803
} // module_sf_noahlsm_glacial_only.F:804
if (FRZGRA) { // module_sf_noahlsm_glacial_only.F:810
FLX2 = MU((-LSUBF),PRCP); // module_sf_noahlsm_glacial_only.F:811
} else { // module_sf_noahlsm_glacial_only.F:812
FLX2 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:813
} // module_sf_noahlsm_glacial_only.F:814
FNET = SU(SU(SU(FDOWN,MU(MU(EMISSI,SIGMA),T24)),SSOIL),FLX2); // module_sf_noahlsm_glacial_only.F:815
RAD = SU(AD(DV(FNET,RCH),TH2),SFCTMP); // module_sf_noahlsm_glacial_only.F:820
EPSCA = DV(AD(MU(A,RR),MU(RAD,DELTA)),AD(DELTA,RR)); // module_sf_noahlsm_glacial_only.F:821
ETP = DV(MU(EPSCA,RCH),LVS); // module_sf_noahlsm_glacial_only.F:822
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:828-857
static __attribute__((noinline)) void gl_shflx(thread NoahTables & tables, thread float * STC, thread int & NSOIL, thread float & DT, thread float & YY, thread float & ZZ1, thread float * ZSOIL, thread float & TBOT, thread float & DF1){
float AI[20];
float BI[20];
float CI[20];
float STCF[20];
float RHSTS[20];
int I;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm_glacial_only.F:834
{  gl_hrtice(tables,RHSTS,STC,TBOT,NSOIL,ZSOIL,YY,ZZ1,DF1,AI,BI,CI); } // module_sf_noahlsm_glacial_only.F:849
{  gl_hstep(tables,STCF,STC,RHSTS,DT,NSOIL,AI,BI,CI); } // module_sf_noahlsm_glacial_only.F:851
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm_glacial_only.F:853
STC[(I)-(1)] = STCF[(I)-(1)]; // module_sf_noahlsm_glacial_only.F:854
} // module_sf_noahlsm_glacial_only.F:855
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:860-1077
static __attribute__((noinline)) void gl_snopac(thread NoahTables & tables, thread float & ETP, thread float & ETA, thread float & PRCP, thread float & PRCPF, thread bool & SNOWNG, thread int & NSOIL, thread float & DT, thread float & DF1, thread float & Q2, thread float & T1, thread float & SFCTMP, thread float & T24, thread float & TH2, thread float & FDOWN, thread float & SSOIL, thread float * STC, thread float & SFCPRS, thread float & RCH, thread float & RR, thread float & SNEQV, thread float & SNDENS, thread float & SNOWH, thread float * ZSOIL, thread float & TBOT, thread float & SNOMLT, thread float & DEW, thread float & FLX1, thread float & FLX2, thread float & FLX3, thread float & ESNOW, thread float & EMISSI, thread float & RIBB){
thread auto& SIGMA=tables.SIGMA;
thread auto& CPH2O=tables.CPH2O;
thread auto& CPICE=tables.CPICE;
thread auto& LSUBF=tables.LSUBF;
float ET1[20];
int K;
float DENOM;
float DSOIL;
float DTOT;
float ESDFLX;
float ESNOW1;
float ESNOW2;
float ETA1;
float ETP1;
float ETP2;
float ETP3;
float ETANRG;
float EX;
float FRCSNO;
float FRCSOI;
float PRCP1;
float QSAT;
float RSNOW;
float SEH;
float SNCOND;
float T12;
float T12A;
float T12B;
float T14;
float YY;
float ZZ1;
float ESDMIN = 0x1.0c6f7a0000000p-20f;
float LSUBC = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
float TFREEZ = 0x1.1126660000000p+8f;
float SNOEXP = 0x1.0000000000000p+1f;
 // module_sf_noahlsm_glacial_only.F:870
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:907
DEW = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:908
ESNOW = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:909
ESNOW1 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:910
ESNOW2 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:911
PRCP1 = MU(PRCPF,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:916
if ((ETP <= 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:920
if (((RIBB >= 0x1.99999a0000000p-4f) && (FDOWN > 0x1.2c00000000000p+7f))) { // module_sf_noahlsm_glacial_only.F:921
ETP = DV(AD(DV(noah_min(MU(ETP,SU(0x1.0000000000000p+0f,RIBB)),0x0.0p+0f),0x1.f5c2900000000p-1f),MU(ETP,-0x1.47ae000000000p-6f)),0x1.f5c2900000000p-1f); // module_sf_noahlsm_glacial_only.F:922
} // module_sf_noahlsm_glacial_only.F:923
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:924
DEW = (-ETP1); // module_sf_noahlsm_glacial_only.F:925
ESNOW2 = MU(ETP1,DT); // module_sf_noahlsm_glacial_only.F:926
ETANRG = MU(ETP,LSUBS); // module_sf_noahlsm_glacial_only.F:927
} else { // module_sf_noahlsm_glacial_only.F:928
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:929
ESNOW = ETP; // module_sf_noahlsm_glacial_only.F:930
ESNOW1 = MU(ESNOW,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:931
ESNOW2 = MU(ESNOW1,DT); // module_sf_noahlsm_glacial_only.F:932
ETANRG = MU(ESNOW,LSUBS); // module_sf_noahlsm_glacial_only.F:933
} // module_sf_noahlsm_glacial_only.F:934
FLX1 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:942
if (SNOWNG) { // module_sf_noahlsm_glacial_only.F:943
FLX1 = MU(MU(CPICE,PRCP),SU(T1,SFCTMP)); // module_sf_noahlsm_glacial_only.F:944
} else { // module_sf_noahlsm_glacial_only.F:945
if ((PRCP > 0x0.0p+0f)) { FLX1 = MU(MU(CPH2O,PRCP),SU(T1,SFCTMP)); } // module_sf_noahlsm_glacial_only.F:946
} // module_sf_noahlsm_glacial_only.F:947
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm_glacial_only.F:956
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm_glacial_only.F:957
DENOM = AD(0x1.0000000000000p+0f,DV(DF1,MU(MU(DTOT,RR),RCH))); // module_sf_noahlsm_glacial_only.F:958
T12A = DV(SU(SU(AD(DV(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T24)),RCH),TH2),SFCTMP),DV(ETANRG,RCH)),RR); // module_sf_noahlsm_glacial_only.F:959
T12B = DV(MU(DF1,STC[(1)-(1)]),MU(MU(DTOT,RR),RCH)); // module_sf_noahlsm_glacial_only.F:961
T12 = DV(AD(AD(SFCTMP,T12A),T12B),DENOM); // module_sf_noahlsm_glacial_only.F:963
if ((T12 <= TFREEZ)) { // module_sf_noahlsm_glacial_only.F:964
T1 = T12; // module_sf_noahlsm_glacial_only.F:978
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm_glacial_only.F:979
SNEQV = noah_max(0x0.0p+0f,SU(SNEQV,ESNOW2)); // module_sf_noahlsm_glacial_only.F:980
FLX3 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:981
EX = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:982
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:983
} else { // module_sf_noahlsm_glacial_only.F:984
T1 = TFREEZ; // module_sf_noahlsm_glacial_only.F:1000
if ((DTOT > MU(0x1.0000000000000p+1f,DSOIL))) { // module_sf_noahlsm_glacial_only.F:1001
DTOT = MU(0x1.0000000000000p+1f,DSOIL); // module_sf_noahlsm_glacial_only.F:1002
} // module_sf_noahlsm_glacial_only.F:1003
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm_glacial_only.F:1004
if ((SU(SNEQV,ESNOW2) <= ESDMIN)) { // module_sf_noahlsm_glacial_only.F:1005
SNEQV = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1006
EX = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1007
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1008
FLX3 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1009
} else { // module_sf_noahlsm_glacial_only.F:1014
SNEQV = SU(SNEQV,ESNOW2); // module_sf_noahlsm_glacial_only.F:1015
ETP3 = MU(ETP,LSUBC); // module_sf_noahlsm_glacial_only.F:1016
SEH = MU(RCH,SU(T1,TH2)); // module_sf_noahlsm_glacial_only.F:1017
T14 = MU(MU(T1,T1),MU(T1,T1)); // module_sf_noahlsm_glacial_only.F:1018
FLX3 = SU(SU(SU(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T14)),SSOIL),SEH),ETANRG); // module_sf_noahlsm_glacial_only.F:1019
if ((FLX3 <= 0x0.0p+0f)) { FLX3 = 0x0.0p+0f; } // module_sf_noahlsm_glacial_only.F:1020
EX = DV(MU(FLX3,0x1.0624de0000000p-10f),LSUBF); // module_sf_noahlsm_glacial_only.F:1021
SNOMLT = MU(EX,DT); // module_sf_noahlsm_glacial_only.F:1022
if ((SU(SNEQV,SNOMLT) >= ESDMIN)) { // module_sf_noahlsm_glacial_only.F:1027
SNEQV = SU(SNEQV,SNOMLT); // module_sf_noahlsm_glacial_only.F:1028
} else { // module_sf_noahlsm_glacial_only.F:1029
EX = DV(SNEQV,DT); // module_sf_noahlsm_glacial_only.F:1033
FLX3 = MU(MU(EX,0x1.f400000000000p+9f),LSUBF); // module_sf_noahlsm_glacial_only.F:1034
SNOMLT = SNEQV; // module_sf_noahlsm_glacial_only.F:1035
SNEQV = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1037
} // module_sf_noahlsm_glacial_only.F:1038
} // module_sf_noahlsm_glacial_only.F:1039
} // module_sf_noahlsm_glacial_only.F:1047
ZZ1 = 0x1.0000000000000p+0f; // module_sf_noahlsm_glacial_only.F:1056
YY = SU(STC[(1)-(1)],DV(MU(MU(MU(0x1.0000000000000p-1f,SSOIL),ZSOIL[(1)-(1)]),ZZ1),DF1)); // module_sf_noahlsm_glacial_only.F:1057
{  gl_shflx(tables,STC,NSOIL,DT,YY,ZZ1,ZSOIL,TBOT,DF1); } // module_sf_noahlsm_glacial_only.F:1062
if ((SNEQV >= 0x1.99999a0000000p-4f)) { // module_sf_noahlsm_glacial_only.F:1068
{  gl_snowpack(tables,SNEQV,DT,SNOWH,SNDENS,T1,YY); } // module_sf_noahlsm_glacial_only.F:1069
} else { // module_sf_noahlsm_glacial_only.F:1070
SNEQV = 0x1.99999a0000000p-4f; // module_sf_noahlsm_glacial_only.F:1071
SNOWH = 0x1.0000000000000p-1f; // module_sf_noahlsm_glacial_only.F:1072
} // module_sf_noahlsm_glacial_only.F:1075
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1080-1203
static __attribute__((noinline)) void gl_snowpack(thread NoahTables & tables, thread float & SNEQV, thread float & DTSEC, thread float & SNOWH, thread float & SNDENS, thread float & TSNOW, thread float & TSOIL){
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
 // module_sf_noahlsm_glacial_only.F:1097
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1109
ESDC = MU(SNEQV,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1110
DTHR = DV(DTSEC,0x1.c200000000000p+11f); // module_sf_noahlsm_glacial_only.F:1111
TSNOWC = SU(TSNOW,0x1.1126660000000p+8f); // module_sf_noahlsm_glacial_only.F:1112
TSOILC = SU(TSOIL,0x1.1126660000000p+8f); // module_sf_noahlsm_glacial_only.F:1113
TAVGC = MU(0x1.0000000000000p-1f,AD(TSNOWC,TSOILC)); // module_sf_noahlsm_glacial_only.F:1127
if ((ESDC > 0x1.47ae140000000p-7f)) { // module_sf_noahlsm_glacial_only.F:1128
ESDCX = ESDC; // module_sf_noahlsm_glacial_only.F:1129
} else { // module_sf_noahlsm_glacial_only.F:1130
ESDCX = 0x1.47ae140000000p-7f; // module_sf_noahlsm_glacial_only.F:1131
} // module_sf_noahlsm_glacial_only.F:1132
BFAC = MU(MU(DTHR,C1),r_exp(SU(MU(0x1.47ae140000000p-4f,TAVGC),MU(C2,SNDENS)))); // module_sf_noahlsm_glacial_only.F:1152
IPOL = 4; // module_sf_noahlsm_glacial_only.F:1153
PEXP = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1154
for (J=IPOL; J>=1; J+=-1) { // module_sf_noahlsm_glacial_only.F:1156
PEXP = DV(MU(MU(AD(0x1.0000000000000p+0f,PEXP),BFAC),ESDCX),((float)((J + 1)))); // module_sf_noahlsm_glacial_only.F:1157
} // module_sf_noahlsm_glacial_only.F:1158
PEXP = AD(PEXP,0x1.0000000000000p+0f); // module_sf_noahlsm_glacial_only.F:1160
DSX = MU(SNDENS,PEXP); // module_sf_noahlsm_glacial_only.F:1181
if ((DSX > 0x1.99999a0000000p-2f)) { DSX = 0x1.99999a0000000p-2f; } // module_sf_noahlsm_glacial_only.F:1182
if ((DSX < 0x1.99999a0000000p-5f)) { DSX = 0x1.99999a0000000p-5f; } // module_sf_noahlsm_glacial_only.F:1183
SNDENS = DSX; // module_sf_noahlsm_glacial_only.F:1189
if ((TSNOWC >= 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:1190
DW = DV(MU(0x1.0a3d700000000p-3f,DTHR),0x1.8000000000000p+4f); // module_sf_noahlsm_glacial_only.F:1191
SNDENS = AD(MU(SNDENS,SU(0x1.0000000000000p+0f,DW)),DW); // module_sf_noahlsm_glacial_only.F:1192
if ((SNDENS >= 0x1.99999a0000000p-2f)) { SNDENS = 0x1.99999a0000000p-2f; } // module_sf_noahlsm_glacial_only.F:1193
} // module_sf_noahlsm_glacial_only.F:1198
SNOWHC = DV(ESDC,SNDENS); // module_sf_noahlsm_glacial_only.F:1199
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm_glacial_only.F:1200
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1206-1230
static __attribute__((noinline)) void gl_snowz0(thread NoahTables & tables, thread float & Z0, thread float & Z0BRD, thread float & SNOWH){
float Z0S = 0x1.0624de0000000p-10f;
float BURIAL;
float Z0EFF;
 // module_sf_noahlsm_glacial_only.F:1212
BURIAL = SU(MU(0x1.c000000000000p+2f,Z0BRD),SNOWH); // module_sf_noahlsm_glacial_only.F:1220
if ((BURIAL <= 0x1.6f00680000000p-11f)) { // module_sf_noahlsm_glacial_only.F:1221
Z0EFF = Z0S; // module_sf_noahlsm_glacial_only.F:1222
} else { // module_sf_noahlsm_glacial_only.F:1223
Z0EFF = DV(BURIAL,0x1.c000000000000p+2f); // module_sf_noahlsm_glacial_only.F:1224
} // module_sf_noahlsm_glacial_only.F:1225
Z0 = Z0EFF; // module_sf_noahlsm_glacial_only.F:1227
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1233-1281
static __attribute__((noinline)) void gl_snow_new(thread NoahTables & tables, thread float & TEMP, thread float & NEWSN, thread float & SNOWH, thread float & SNDENS){
float DSNEW;
float HNEWC;
float SNOWHC;
float NEWSNC;
float TEMPC;
 // module_sf_noahlsm_glacial_only.F:1244
TEMPC = SU(TEMP,0x1.1126660000000p+8f); // module_sf_noahlsm_glacial_only.F:1255
if ((TEMPC <= -0x1.e000000000000p+3f)) { // module_sf_noahlsm_glacial_only.F:1256
DSNEW = 0x1.99999a0000000p-5f; // module_sf_noahlsm_glacial_only.F:1257
} else { // module_sf_noahlsm_glacial_only.F:1258
DSNEW = AD(0x1.99999a0000000p-5f,MU(0x1.bda5120000000p-10f,r_pow(AD(TEMPC,0x1.e000000000000p+3f),0x1.8000000000000p+0f))); // module_sf_noahlsm_glacial_only.F:1259
} // module_sf_noahlsm_glacial_only.F:1260
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1265
NEWSNC = MU(NEWSN,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1266
HNEWC = DV(NEWSNC,DSNEW); // module_sf_noahlsm_glacial_only.F:1271
if ((AD(SNOWHC,HNEWC) < 0x1.0624de0000000p-10f)) { // module_sf_noahlsm_glacial_only.F:1272
SNDENS = noah_max(DSNEW,SNDENS); // module_sf_noahlsm_glacial_only.F:1273
} else { // module_sf_noahlsm_glacial_only.F:1274
SNDENS = DV(AD(MU(SNOWHC,SNDENS),MU(HNEWC,DSNEW)),AD(SNOWHC,HNEWC)); // module_sf_noahlsm_glacial_only.F:1275
} // module_sf_noahlsm_glacial_only.F:1276
SNOWHC = AD(SNOWHC,HNEWC); // module_sf_noahlsm_glacial_only.F:1277
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm_glacial_only.F:1278
}
// MPAS v8.4.2 physics_wrf/module_sf_noahdrv.F:44-1590; MPAS urban=0 slice.

// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207. FNDSNOWH=.true. in MPAS.


kernel void hexv842_mpas_noah_glacier_f32(
constant Hexv842MpasNoahGlacierF32Resources &r [[buffer(0)]],
constant Hexv842MpasNoahGlacierF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
if(c>=uint(p.n)) return;
thread float x[49]; thread int ix[4]; for(int j=0;j<49;j++) x[j]=r.floats[c*49+j]; for(int j=0;j<4;j++) ix[j]=r.integers[c*4+j];
thread NoahTables tables; thread uint *t=reinterpret_cast<thread uint*>(&tables); for(uint j=0;j<sizeof(NoahTables)/4;j++)t[j]=r.tables[j]; tables.error=0;
// Four packed soil slots: prevent out-of-bounds module_sf_noahlsm_glacial_only.F:211 access.
if(ix[3]!=4){r.status[c]=211;return;}
int IILOC_arg=ix[0];int JJLOC_arg=ix[1];int ISICE_arg=ix[2];float FFROZP_arg=x[0];float DT_arg=x[1];float ZLVL_arg=x[2];int NSOIL_arg=ix[3];float LWDN_arg=x[7];float SOLNET_arg=x[8];float SFCPRS_arg=x[9];float PRCP_arg=x[10];float SFCTMP_arg=x[11];float Q2_arg=x[12];float TH2_arg=x[13];float Q2SAT_arg=x[14];float DQSDT2_arg=x[15];float ALB_arg=x[16];float SNOALB_arg=x[17];float TBOT_arg=x[18];float Z0BRD_arg=x[19];float Z0_arg=x[20];float EMISSI_arg=x[21];float EMBRD_arg=x[22];float T1_arg=x[23];float SNOWH_arg=x[28];float SNEQV_arg=x[29];float ALBEDO_arg=x[30];float CH_arg=x[31];float ETA_arg=x[32];float SHEAT_arg=x[33];float ETA_KINEMATIC_arg=x[34];float FDOWN_arg=x[35];float ESNOW_arg=x[36];float DEW_arg=x[37];float ETP_arg=x[38];float SSOIL_arg=x[39];float FLX1_arg=x[40];float FLX2_arg=x[41];float FLX3_arg=x[42];float SNOMLT_arg=x[43];float SNCOVR_arg=x[44];float RUNOFF1_arg=x[45];float Q1_arg=x[46];float SNOTIME1_arg=x[47];float RIBB_arg=x[48];
gl_sflx_glacial(tables,IILOC_arg,JJLOC_arg,ISICE_arg,FFROZP_arg,DT_arg,ZLVL_arg,NSOIL_arg,x+3,LWDN_arg,SOLNET_arg,SFCPRS_arg,PRCP_arg,SFCTMP_arg,Q2_arg,TH2_arg,Q2SAT_arg,DQSDT2_arg,ALB_arg,SNOALB_arg,TBOT_arg,Z0BRD_arg,Z0_arg,EMISSI_arg,EMBRD_arg,T1_arg,x+24,SNOWH_arg,SNEQV_arg,ALBEDO_arg,CH_arg,ETA_arg,SHEAT_arg,ETA_KINEMATIC_arg,FDOWN_arg,ESNOW_arg,DEW_arg,ETP_arg,SSOIL_arg,FLX1_arg,FLX2_arg,FLX3_arg,SNOMLT_arg,SNCOVR_arg,RUNOFF1_arg,Q1_arg,SNOTIME1_arg,RIBB_arg);
ix[0]=IILOC_arg;ix[1]=JJLOC_arg;ix[2]=ISICE_arg;x[0]=FFROZP_arg;x[1]=DT_arg;x[2]=ZLVL_arg;ix[3]=NSOIL_arg;x[7]=LWDN_arg;x[8]=SOLNET_arg;x[9]=SFCPRS_arg;x[10]=PRCP_arg;x[11]=SFCTMP_arg;x[12]=Q2_arg;x[13]=TH2_arg;x[14]=Q2SAT_arg;x[15]=DQSDT2_arg;x[16]=ALB_arg;x[17]=SNOALB_arg;x[18]=TBOT_arg;x[19]=Z0BRD_arg;x[20]=Z0_arg;x[21]=EMISSI_arg;x[22]=EMBRD_arg;x[23]=T1_arg;x[28]=SNOWH_arg;x[29]=SNEQV_arg;x[30]=ALBEDO_arg;x[31]=CH_arg;x[32]=ETA_arg;x[33]=SHEAT_arg;x[34]=ETA_KINEMATIC_arg;x[35]=FDOWN_arg;x[36]=ESNOW_arg;x[37]=DEW_arg;x[38]=ETP_arg;x[39]=SSOIL_arg;x[40]=FLX1_arg;x[41]=FLX2_arg;x[42]=FLX3_arg;x[43]=SNOMLT_arg;x[44]=SNCOVR_arg;x[45]=RUNOFF1_arg;x[46]=Q1_arg;x[47]=SNOTIME1_arg;x[48]=RIBB_arg;
r.status[c]=tables.error;

for(int j=0;j<49;j++)r.floats[c*49+j]=x[j]; for(int j=0;j<4;j++)r.integers[c*4+j]=ix[j];
}
