#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_noah_init_f32_bindings.metalh"
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

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1857-1905

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1908-2196

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2199-2316

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2319-2539

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2541-2600

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
static __attribute__((noinline)) void noah_init_column(thread NoahTables & tables, thread float * TSLB, thread float * SMOIS, thread float * SH2O, thread float & SNOALB, thread float & SNOW, thread float & SNOWH, thread int & IVGTYP, thread int & ISLTYP, thread bool & RESTART, thread bool & INPUT_SFC_ALBEDO){
thread auto& MAXALB=tables.MAXALB;
thread auto& BB=tables.BB;
thread auto& MAXSMC=tables.MAXSMC;
thread auto& SATPSI=tables.SATPSI;
thread auto& GRAVITY=tables.GRAVITY;
int NS;
int NSOILLEVELS = 4;
float BX;
float FK;
float SMCMAX;
float PSISAT;
float FREE;
float BLIM = 0x1.6000000000000p+2f;
float HLICE = 0x1.45af000000000p+18f;
float T0 = 0x1.1126660000000p+8f;
if (RESTART) { return; } // mpas_atmphys_lsm_noahinit.F:136
if ((ISLTYP < 1)) { tables.error = 140; return; } // mpas_atmphys_lsm_noahinit.F:140
if ((!INPUT_SFC_ALBEDO)) { SNOALB = MU(MAXALB[(IVGTYP)-(1)],0x1.47ae140000000p-7f); } // mpas_atmphys_lsm_noahinit.F:146
BX = BB[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:154
SMCMAX = MAXSMC[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:155
PSISAT = SATPSI[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:156
if ((((BX > 0x0.0p+0f) && (SMCMAX > 0x0.0p+0f)) && (PSISAT > 0x0.0p+0f))) { // mpas_atmphys_lsm_noahinit.F:157
for (NS=1; NS<=NSOILLEVELS; NS+=1) { // mpas_atmphys_lsm_noahinit.F:158
if ((TSLB[(NS)-(1)] < 0x1.1126240000000p+8f)) { // mpas_atmphys_lsm_noahinit.F:161
BX = BB[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:166
SMCMAX = MAXSMC[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:167
PSISAT = SATPSI[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:168
if ((BX > BLIM)) { BX = BLIM; } // mpas_atmphys_lsm_noahinit.F:169
FK = MU(r_pow(MU(DV(HLICE,MU(GRAVITY,(-PSISAT))),DV(SU(TSLB[(NS)-(1)],T0),TSLB[(NS)-(1)])),DV(-1,BX)),SMCMAX); // mpas_atmphys_lsm_noahinit.F:170
if ((FK < 0x1.47ae140000000p-6f)) { FK = 0x1.47ae140000000p-6f; } // mpas_atmphys_lsm_noahinit.F:172
SH2O[(NS)-(1)] = noah_min(FK,SMOIS[(NS)-(1)]); // mpas_atmphys_lsm_noahinit.F:173
{  noah_frh2o(tables,FREE,TSLB[(NS)-(1)],SMOIS[(NS)-(1)],SH2O[(NS)-(1)],SMCMAX,BX,PSISAT); if(tables.error)return; } // mpas_atmphys_lsm_noahinit.F:178
SH2O[(NS)-(1)] = FREE; // mpas_atmphys_lsm_noahinit.F:180
} else { // mpas_atmphys_lsm_noahinit.F:181
SH2O[(NS)-(1)] = SMOIS[(NS)-(1)]; // mpas_atmphys_lsm_noahinit.F:184
} // mpas_atmphys_lsm_noahinit.F:186
} // mpas_atmphys_lsm_noahinit.F:187
} else { // mpas_atmphys_lsm_noahinit.F:188
for (NS=1; NS<=NSOILLEVELS; NS+=1) { // mpas_atmphys_lsm_noahinit.F:189
SH2O[(NS)-(1)] = SMOIS[(NS)-(1)]; // mpas_atmphys_lsm_noahinit.F:190
} // mpas_atmphys_lsm_noahinit.F:191
} // mpas_atmphys_lsm_noahinit.F:192
}

kernel void hexv842_mpas_noah_init_f32(
constant Hexv842MpasNoahInitF32Resources &r [[buffer(0)]],
constant Hexv842MpasNoahInitF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
if(c>=uint(p.n)) return;
thread float x[15]; thread int ix[4]; for(int j=0;j<15;j++) x[j]=r.floats[c*15+j]; for(int j=0;j<4;j++) ix[j]=r.integers[c*4+j];
thread NoahTables tables; thread uint *t=reinterpret_cast<thread uint*>(&tables); for(uint j=0;j<sizeof(NoahTables)/4;j++)t[j]=r.tables[j]; tables.error=0;
// mpas_atmphys_lsm_noahinit.F:140-150: native rejects only ISLTYP < 1.
// Preserve its unchecked table reads; no additional upper/category refusal.
if(!ix[2] && ix[1]<1){r.status[c]=140;return;}
if(!ix[2] && (ix[1]>30 || (!ix[3] && (ix[0]<1 || ix[0]>50)))){r.status[c]=-1;return;}
bool restart=ix[2]!=0,albedo=ix[3]!=0;
noah_init_column(tables,x,x+4,x+8,x[12],x[13],x[14],ix[0],ix[1],restart,albedo);
r.status[c]=tables.error;

for(int j=0;j<15;j++)r.floats[c*15+j]=x[j];
}
