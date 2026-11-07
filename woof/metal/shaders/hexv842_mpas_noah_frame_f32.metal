#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define MU(a,b) mpas_mul(a,b)
#define DV(a,b) cu_div(a,b)
static __attribute__((noinline)) float r_exp(float x){return gfk_exp(x);}
static __attribute__((noinline)) float r_pow(float x,float y){return gfk_pow(x,y);}
static inline float noah_powi(float a,int n){unsigned m=n<0?-n:n;float r=1;while(m){if(m&1)r=MU(r,a);m>>=1;if(m)a=MU(a,a);}return n<0?DV(1.0f,r):r;}
#include "hexv842_mpas_noah_frame_f32_bindings.metalh"
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

struct NoahDriverFrame {
bool UA_PHYS;
float FLX4;
float FVB;
float FBUR;
float FGSN;
float SMAV[4];
float ETP;
float SSOIL;
float FLX1;
float FLX2;
float FLX3;
float XLAI;
float FFROZP;
int NSOIL;
int NS;
float LWDN;
float PRCP;
float ALBBRD;
float ALBEDOK;
float ETA;
float ETA_KINEMATIC;
float Z0K;
float RUNOFF1;
float RUNOFF2;
float SHEAT;
float SOLNET;
float FDTLIW;
float FDTW;
float EMISSI;
float SNCOVR;
float SNEQV;
float SNOWHK;
float CMC;
float SNOMLT;
float SOILM;
float SOILW;
float Q1;
float T1;
float SNOTIME1;
float Z0BRD;
float STC[4];
float SMC[4];
float SWC[4];
};
static_assert(sizeof(NoahDriverFrame)==220, "Noah frame ABI");
static inline void frame_pre(thread NoahTables & tables, thread float * DZ8W, thread float * QV3D, thread float * P8W3D, thread float * T3D, thread float & TSK, thread float & HFX, thread float & QFX, thread float & LH, thread float & GRDFLX, thread float & QGH, thread float & SWDOWN, thread float & GLW, thread float & SMSTAV, thread float & SMSTOT, thread float & SFCRUNOFF, thread float & UDRUNOFF, thread int & IVGTYP, thread int & ISLTYP, thread int & ISURBAN, thread int & ISICE, thread float & VEGFRA, thread float & ALBEDO, thread float & ALBBCK, thread float & ZNT, thread float & Z0, thread float & TMN, thread float & XLAND, thread float & XICE, thread float & EMISS, thread float & EMBCK, thread float & SNOWC, thread float & QSFC, thread float & RAINBL, thread float & DT, thread float * DZS, thread int & ITIMESTEP, thread float * SMOIS, thread float * TSLB, thread float & SNOW, thread float & CANWAT, thread float & CHS, thread float & CHS2, thread float & CQS2, thread float & SR, thread float & CHKLOWQ, thread float & LAI, thread float & QZ0, thread float * SH2O, thread float & SNOWH, thread float & SNOALB, thread float & SHDMIN, thread float & SHDMAX, thread float & SNOTIME, thread float & ACSNOM, thread float & ACSNOW, thread float & SNOPCX, thread float & POTEVP, thread float * SMCREL, thread float & XICE_THRESHOLD, thread bool & USEMONALB, thread float & RIB, thread float & NOAHRES, thread int & OPT_THCND, thread float & FLX4_2D, thread float & FVB_2D, thread float & FBUR_2D, thread float & FGSN_2D, int I, int J, thread int &route, thread NoahDriverFrame &f, device float *land, device int *landi, device float *glacier, device int *glacieri, uint c){

thread auto& LCZ_1=tables.LCZ_1;
thread auto& LCZ_2=tables.LCZ_2;
thread auto& LCZ_3=tables.LCZ_3;
thread auto& LCZ_4=tables.LCZ_4;
thread auto& LCZ_5=tables.LCZ_5;
thread auto& LCZ_6=tables.LCZ_6;
thread auto& LCZ_7=tables.LCZ_7;
thread auto& LCZ_8=tables.LCZ_8;
thread auto& LCZ_9=tables.LCZ_9;
thread auto& LCZ_10=tables.LCZ_10;
thread auto& LCZ_11=tables.LCZ_11;
thread const char * LUTYPE=tables.LUTYPE;
thread const char * SLTYPE=tables.SLTYPE;
thread auto& CP=tables.CP;
thread auto& R_D=tables.R_D;
thread auto& XLF=tables.XLF;
thread auto& XLV=tables.XLV;
thread auto& RHOWATER=tables.RHOWATER;
thread auto& STBOLT=tables.STBOLT;
int KTS = 1;
float ETPND1_HYDRO;
float SFCHEADRT_HYDRO;
float INFXSRT_HYDRO;
int NUM_SOIL_LAYERS = 4;
thread bool &UA_PHYS=f.UA_PHYS;
UA_PHYS = false;
thread float &FLX4=f.FLX4;
thread float &FVB=f.FVB;
thread float &FBUR=f.FBUR;
thread float &FGSN=f.FGSN;
float ET[4];
thread float *SMAV=f.SMAV;
float BETA;
thread float &ETP=f.ETP;
thread float &SSOIL=f.SSOIL;
float EC;
float EDIR;
float ESNOW;
float ETT;
thread float &FLX1=f.FLX1;
thread float &FLX2=f.FLX2;
thread float &FLX3=f.FLX3;
float DRIP;
float DEW;
float FDOWN;
float RC;
float PC;
float RSMIN;
thread float &XLAI=f.XLAI;
float RCS;
float RCT;
float RCQ;
float RCSOIL;
thread float &FFROZP=f.FFROZP;
bool MYJ = false;
bool FRPCPN = false;
bool LOCAL = false;
bool IPRINT;
int ICE;
thread int &NSOIL=f.NSOIL;
int SLOPETYP;
int SOILTYP;
int VEGTYP;
int NROOT;
thread int &NS=f.NS;
float SHMIN;
float SHMAX;
float DQSDT2;
thread float &LWDN=f.LWDN;
thread float &PRCP=f.PRCP;
float Q2SAT;
float Q2SATI;
float SFCPRS;
float SFCTMP;
float SHDFAC;
float SNOALB1;
float SOLDN;
float TBOT;
float ZLVL;
float Q2K;
thread float &ALBBRD=f.ALBBRD;
thread float &ALBEDOK=f.ALBEDOK;
thread float &ETA=f.ETA;
thread float &ETA_KINEMATIC=f.ETA_KINEMATIC;
float EMBRD;
thread float &Z0K=f.Z0K;
thread float &RUNOFF1=f.RUNOFF1;
thread float &RUNOFF2=f.RUNOFF2;
float RUNOFF3;
thread float &SHEAT=f.SHEAT;
thread float &SOLNET=f.SOLNET;
float E2SAT;
float SFCTSNO;
float SATFLG;
thread float &FDTLIW=f.FDTLIW;
float RIBB;
thread float &FDTW=f.FDTW;
thread float &EMISSI=f.EMISSI;
thread float &SNCOVR=f.SNCOVR;
thread float &SNEQV=f.SNEQV;
thread float &SNOWHK=f.SNOWHK;
thread float &CMC=f.CMC;
float CHK;
float TH2;
float SMCDRY;
float SMCMAX;
float SMCREF;
float SMCWLT;
thread float &SNOMLT=f.SNOMLT;
thread float &SOILM=f.SOILM;
thread float &SOILW=f.SOILW;
thread float &Q1=f.Q1;
thread float &T1=f.T1;
thread float &SNOTIME1=f.SNOTIME1;
float DUMMY;
thread float &Z0BRD=f.Z0BRD;
float SLDPTH[4];
thread float *STC=f.STC;
thread float *SMC=f.SMC;
thread float *SWC=f.SWC;
float TRESH = 0x1.e666660000000p-1f;
float A2 = 0x1.1ab8520000000p+4f;
float A3 = 0x1.1126660000000p+8f;
float A4 = 0x1.da66660000000p+4f;
float T0 = 0x1.1128f60000000p+8f;
float ELWV = 0x1.312d000000000p+21f;
float A23M4 = MU(A2,SU(A3,A4));
float ROW = 0x1.f400000000000p+9f;
float ELIW = XLF;
float ROWLIW = MU(ROW,ELIW);
bool RDLAI2D = false;
float AOASIS;
float SIGMA_SB;
float CAPA = DV(R_D,CP);
float APELM;
float APES;
float SFCTH2;
float PSFC;
int FASDAS = 0;
float XSDA_HFX;
float XSDA_QFX;
float XQNORM = 0.0f;
float HFX_PHY = 0.0f;
float QFX_PHY = 0.0f;
float HCPCT_FASDAS;
float IRRIGATION_CHANNEL = 0.0f;
XSDA_HFX = 0x0.0p+0f; // module_sf_noahdrv.F:719
XSDA_QFX = 0x0.0p+0f; // module_sf_noahdrv.F:720
SIGMA_SB = 0x1.e70c9e0000000p-25f; // module_sf_noahdrv.F:724
FDTLIW = DV(DT,ROWLIW); // module_sf_noahdrv.F:727
FDTW = DV(DT,MU(XLV,RHOWATER)); // module_sf_noahdrv.F:729
IPRINT = false; // module_sf_noahdrv.F:731
SLOPETYP = 1; // module_sf_noahdrv.F:734
NSOIL = NUM_SOIL_LAYERS; // module_sf_noahdrv.F:738
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:740
SLDPTH[(NS)-(1)] = DZS[(NS)-(1)]; // module_sf_noahdrv.F:741
} // module_sf_noahdrv.F:742
if ((ITIMESTEP == 1)) { // module_sf_noahdrv.F:746
if ((SU(XLAND,0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // module_sf_noahdrv.F:759
SMSTAV = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:765
SMSTOT = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:766
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:767
SMOIS[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:768
TSLB[(NS)-(1)] = 0x1.1128f60000000p+8f; // module_sf_noahdrv.F:769
SMCREL[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:770
} // module_sf_noahdrv.F:771
} else { // module_sf_noahdrv.F:772
if ((XICE >= XICE_THRESHOLD)) { // module_sf_noahdrv.F:773
SMSTAV = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:775
SMSTOT = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:776
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:777
SMOIS[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:778
SMCREL[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:779
} // module_sf_noahdrv.F:780
} // module_sf_noahdrv.F:781
} // module_sf_noahdrv.F:782

} // module_sf_noahdrv.F:785
PSFC = P8W3D[(1)-(1)]; // module_sf_noahdrv.F:790
SFCPRS = MU(AD(P8W3D[((KTS + 1))-(1)],P8W3D[(KTS)-(1)]),0x1.0000000000000p-1f); // module_sf_noahdrv.F:792
Q2K = DV(QV3D[(1)-(1)],AD(0x1.0000000000000p+0f,QV3D[(1)-(1)])); // module_sf_noahdrv.F:794
SOILW = 0x0.0p+0f; // module_sf_noahdrv.F:796
FLX4 = 0x0.0p+0f; // module_sf_noahdrv.F:797
FVB = 0x0.0p+0f; // module_sf_noahdrv.F:798
FBUR = 0x0.0p+0f; // module_sf_noahdrv.F:799
FGSN = 0x0.0p+0f; // module_sf_noahdrv.F:800
Q2SAT = DV(QGH,AD(0x1.0000000000000p+0f,QGH)); // module_sf_noahdrv.F:803
if (((MYJ && (Q2K >= MU(Q2SAT,TRESH))) && (Q2K < QZ0))) { // module_sf_noahdrv.F:806
SATFLG = 0x0.0p+0f; // module_sf_noahdrv.F:807
CHKLOWQ = 0x0.0p+0f; // module_sf_noahdrv.F:808
} else { // module_sf_noahdrv.F:809
SATFLG = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:810
CHKLOWQ = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:811
} // module_sf_noahdrv.F:812
SFCTMP = T3D[(1)-(1)]; // module_sf_noahdrv.F:814
ZLVL = MU(0x1.0000000000000p-1f,DZ8W[(1)-(1)]); // module_sf_noahdrv.F:815
APES = r_pow(DV(0x1.86a0000000000p+16f,PSFC),CAPA); // module_sf_noahdrv.F:819
APELM = r_pow(DV(0x1.86a0000000000p+16f,SFCPRS),CAPA); // module_sf_noahdrv.F:820
SFCTH2 = MU(SFCTMP,APELM); // module_sf_noahdrv.F:821
TH2 = DV(SFCTH2,APES); // module_sf_noahdrv.F:822
EMISSI = EMISS; // module_sf_noahdrv.F:824
LWDN = MU(GLW,EMISSI); // module_sf_noahdrv.F:825
SOLDN = SWDOWN; // module_sf_noahdrv.F:827
SOLNET = MU(SOLDN,SU(0x1.0000000000000p+0f,ALBEDO)); // module_sf_noahdrv.F:831
PRCP = DV(RAINBL,DT); // module_sf_noahdrv.F:832
VEGTYP = IVGTYP; // module_sf_noahdrv.F:840
SOILTYP = ISLTYP; // module_sf_noahdrv.F:841
SHDFAC = DV(VEGFRA,0x1.9000000000000p+6f); // module_sf_noahdrv.F:842
T1 = TSK; // module_sf_noahdrv.F:843
CHK = CHS; // module_sf_noahdrv.F:844
SHMIN = DV(SHDMIN,0x1.9000000000000p+6f); // module_sf_noahdrv.F:845
SHMAX = DV(SHDMAX,0x1.9000000000000p+6f); // module_sf_noahdrv.F:846
SNEQV = MU(SNOW,0x1.0624de0000000p-10f); // module_sf_noahdrv.F:848
SNOWHK = SNOWH; // module_sf_noahdrv.F:850
SNCOVR = SNOWC; // module_sf_noahdrv.F:851
if (FRPCPN) { // module_sf_noahdrv.F:856
FFROZP = SR; // module_sf_noahdrv.F:857
} else { // module_sf_noahdrv.F:858
if ((SFCTMP <= 0x1.1126660000000p+8f)) { // module_sf_noahdrv.F:859
FFROZP = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:860
} else { // module_sf_noahdrv.F:861
FFROZP = 0x0.0p+0f; // module_sf_noahdrv.F:862
} // module_sf_noahdrv.F:863
} // module_sf_noahdrv.F:864
if ((SU(XLAND,0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // module_sf_noahdrv.F:866
} else { // module_sf_noahdrv.F:875
if ((XICE >= XICE_THRESHOLD)) { // module_sf_noahdrv.F:878
ICE = 1; // module_sf_noahdrv.F:880
} else if ((VEGTYP == ISICE)) { // module_sf_noahdrv.F:881
ICE = -1; // module_sf_noahdrv.F:883
} else { // module_sf_noahdrv.F:884
ICE = 0; // module_sf_noahdrv.F:886
} // module_sf_noahdrv.F:887
DQSDT2 = DV(MU(Q2SAT,A23M4),noah_powi(SU(SFCTMP,A4),2)); // module_sf_noahdrv.F:888
if ((SNOW > 0x0.0p+0f)) { // module_sf_noahdrv.F:890
SFCTSNO = SFCTMP; // module_sf_noahdrv.F:892
E2SAT = MU(0x1.31999a0000000p+9f,r_exp(MU(0x1.81e0000000000p+12f,SU(0x1.dfda840000000p-9f,DV(0x1.0000000000000p+0f,SFCTSNO))))); // module_sf_noahdrv.F:893
Q2SATI = DV(MU(0x1.3e76c80000000p-1f,E2SAT),SU(SFCPRS,E2SAT)); // module_sf_noahdrv.F:894
Q2SATI = DV(Q2SATI,AD(0x1.0000000000000p+0f,Q2SATI)); // module_sf_noahdrv.F:895
if ((T1 > 0x1.1123d80000000p+8f)) { // module_sf_noahdrv.F:896
Q2SAT = AD(MU(Q2SAT,SU(0x1.0000000000000p+0f,SNOWC)),MU(Q2SATI,SNOWC)); // module_sf_noahdrv.F:898
DQSDT2 = AD(MU(DQSDT2,SU(0x1.0000000000000p+0f,SNOWC)),MU(DV(MU(Q2SATI,0x1.81e0000000000p+12f),noah_powi(SFCTSNO,2)),SNOWC)); // module_sf_noahdrv.F:899
} else { // module_sf_noahdrv.F:900
Q2SAT = Q2SATI; // module_sf_noahdrv.F:902
DQSDT2 = DV(MU(Q2SATI,0x1.81e0000000000p+12f),noah_powi(SFCTSNO,2)); // module_sf_noahdrv.F:903
} // module_sf_noahdrv.F:904
if ((((T1 > 0x1.1100000000000p+8f) && (SNOWC > 0x0.0p+0f)) && (SWDOWN > 0x1.4000000000000p+3f))) { DQSDT2 = MU(DQSDT2,SU(0x1.0000000000000p+0f,SNOWC)); } // module_sf_noahdrv.F:907
} // module_sf_noahdrv.F:908
TBOT = TMN; // module_sf_noahdrv.F:911
if ((ISURBAN == 1)) { // module_sf_noahdrv.F:913
if ((VEGTYP == 25)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahdrv.F:915
if ((VEGTYP == 26)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahdrv.F:916
if ((VEGTYP == 27)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahdrv.F:917
} // module_sf_noahdrv.F:918
if (((SOILTYP == 14) && (XICE == 0x0.0p+0f))) { // module_sf_noahdrv.F:919
SOILTYP = 7; // module_sf_noahdrv.F:924
} // module_sf_noahdrv.F:925
SNOALB1 = SNOALB; // module_sf_noahdrv.F:926
CMC = DV(CANWAT,0x1.f400000000000p+9f); // module_sf_noahdrv.F:928
ALBBRD = ALBBCK; // module_sf_noahdrv.F:941
Z0BRD = Z0; // module_sf_noahdrv.F:942
EMBRD = EMBCK; // module_sf_noahdrv.F:943
SNOTIME1 = SNOTIME; // module_sf_noahdrv.F:944
RIBB = RIB; // module_sf_noahdrv.F:945
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:948
SMC[(NS)-(1)] = SMOIS[(NS)-(1)]; // module_sf_noahdrv.F:949
STC[(NS)-(1)] = TSLB[(NS)-(1)]; // module_sf_noahdrv.F:950
SWC[(NS)-(1)] = SH2O[(NS)-(1)]; // module_sf_noahdrv.F:951
} // module_sf_noahdrv.F:952
if ((((SNEQV != 0x0.0p+0f) && (SNOWHK == 0x0.0p+0f)) || (SNOWHK <= SNEQV))) { // module_sf_noahdrv.F:954
SNOWHK = MU(0x1.4000000000000p+2f,SNEQV); // module_sf_noahdrv.F:955
} // module_sf_noahdrv.F:956
if (((((((((((((IVGTYP == ISURBAN) || (IVGTYP == LCZ_1)) || (IVGTYP == LCZ_2)) || (IVGTYP == LCZ_3)) || (IVGTYP == LCZ_4)) || (IVGTYP == LCZ_5)) || (IVGTYP == LCZ_6)) || (IVGTYP == LCZ_7)) || (IVGTYP == LCZ_8)) || (IVGTYP == LCZ_9)) || (IVGTYP == LCZ_10)) || (IVGTYP == LCZ_11))) { // module_sf_noahdrv.F:986
VEGTYP = ISURBAN; // module_sf_noahdrv.F:990
} // module_sf_noahdrv.F:991
AOASIS = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:996
if ((ICE == 1)) { // module_sf_noahdrv.F:1064
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1068
SH2O[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1069
} // module_sf_noahdrv.F:1070
LAI = 0x1.47ae140000000p-7f; // module_sf_noahdrv.F:1071
return; // module_sf_noahdrv.F:1073
} else if ((ICE == 0)) { // module_sf_noahdrv.F:1075
ETPND1_HYDRO = 0x0.0p+0f; // module_sf_noahdrv.F:1098
SFCHEADRT_HYDRO = 0x0.0p+0f; // module_sf_noahdrv.F:1099
INFXSRT_HYDRO = 0x0.0p+0f; // module_sf_noahdrv.F:1100
{  land[c*110+0]=FFROZP;
land[c*110+1]=DT;
land[c*110+2]=ZLVL;
land[c*110+3]=SLDPTH[0];
land[c*110+4]=SLDPTH[1];
land[c*110+5]=SLDPTH[2];
land[c*110+6]=SLDPTH[3];
land[c*110+7]=LWDN;
land[c*110+8]=SOLDN;
land[c*110+9]=SOLNET;
land[c*110+10]=SFCPRS;
land[c*110+11]=PRCP;
land[c*110+12]=SFCTMP;
land[c*110+13]=Q2K;
land[c*110+18]=TH2;
land[c*110+19]=Q2SAT;
land[c*110+20]=DQSDT2;
land[c*110+21]=SHDFAC;
land[c*110+22]=SHMIN;
land[c*110+23]=SHMAX;
land[c*110+24]=ALBBRD;
land[c*110+25]=SNOALB1;
land[c*110+26]=TBOT;
land[c*110+27]=Z0BRD;
land[c*110+29]=EMISSI;
land[c*110+30]=EMBRD;
land[c*110+31]=CMC;
land[c*110+32]=T1;
land[c*110+33]=STC[0];
land[c*110+34]=STC[1];
land[c*110+35]=STC[2];
land[c*110+36]=STC[3];
land[c*110+37]=SMC[0];
land[c*110+38]=SMC[1];
land[c*110+39]=SMC[2];
land[c*110+40]=SMC[3];
land[c*110+41]=SWC[0];
land[c*110+42]=SWC[1];
land[c*110+43]=SWC[2];
land[c*110+44]=SWC[3];
land[c*110+45]=SNOWHK;
land[c*110+46]=SNEQV;
land[c*110+48]=CHK;
land[c*110+70]=FLX4;
land[c*110+71]=FVB;
land[c*110+72]=FBUR;
land[c*110+73]=FGSN;
land[c*110+75]=SNCOVR;
land[c*110+87]=SOILW;
land[c*110+94]=SNOTIME1;
land[c*110+95]=RIBB;
land[c*110+100]=SFCHEADRT_HYDRO;
land[c*110+101]=INFXSRT_HYDRO;
land[c*110+102]=ETPND1_HYDRO;
land[c*110+103]=AOASIS;
land[c*110+104]=XSDA_QFX;
land[c*110+105]=HFX_PHY;
land[c*110+106]=QFX_PHY;
land[c*110+107]=XQNORM;
land[c*110+109]=IRRIGATION_CHANNEL;
landi[c*14+0]=I;
landi[c*14+1]=J;
landi[c*14+2]=ISURBAN;
landi[c*14+3]=NSOIL;
landi[c*14+4]=LOCAL;
landi[c*14+5]=VEGTYP;
landi[c*14+6]=SOILTYP;
landi[c*14+7]=SLOPETYP;
landi[c*14+8]=UA_PHYS;
landi[c*14+9]=RDLAI2D;
landi[c*14+10]=USEMONALB;
landi[c*14+12]=OPT_THCND;
landi[c*14+13]=FASDAS;
route=1; return; } // module_sf_noahdrv.F:1102
} else if ((ICE == -1)) { // module_sf_noahdrv.F:1137
SOILM = 0x0.0p+0f; // module_sf_noahdrv.F:1143
XLAI = 0x1.47ae140000000p-7f; // module_sf_noahdrv.F:1144
RUNOFF2 = 0x0.0p+0f; // module_sf_noahdrv.F:1145
RUNOFF3 = 0x0.0p+0f; // module_sf_noahdrv.F:1146
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1147
SWC[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1148
SMC[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1149
SMAV[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1150
} // module_sf_noahdrv.F:1151
{  glacier[c*49+0]=FFROZP;
glacier[c*49+1]=DT;
glacier[c*49+2]=ZLVL;
glacier[c*49+3]=SLDPTH[0];
glacier[c*49+4]=SLDPTH[1];
glacier[c*49+5]=SLDPTH[2];
glacier[c*49+6]=SLDPTH[3];
glacier[c*49+7]=LWDN;
glacier[c*49+8]=SOLNET;
glacier[c*49+9]=SFCPRS;
glacier[c*49+10]=PRCP;
glacier[c*49+11]=SFCTMP;
glacier[c*49+12]=Q2K;
glacier[c*49+13]=TH2;
glacier[c*49+14]=Q2SAT;
glacier[c*49+15]=DQSDT2;
glacier[c*49+16]=ALBBRD;
glacier[c*49+17]=SNOALB1;
glacier[c*49+18]=TBOT;
glacier[c*49+19]=Z0BRD;
glacier[c*49+21]=EMISSI;
glacier[c*49+22]=EMBRD;
glacier[c*49+23]=T1;
glacier[c*49+24]=STC[0];
glacier[c*49+25]=STC[1];
glacier[c*49+26]=STC[2];
glacier[c*49+27]=STC[3];
glacier[c*49+28]=SNOWHK;
glacier[c*49+29]=SNEQV;
glacier[c*49+31]=CHK;
glacier[c*49+44]=SNCOVR;
glacier[c*49+47]=SNOTIME1;
glacier[c*49+48]=RIBB;
glacieri[c*4+0]=I;
glacieri[c*4+1]=J;
glacieri[c*4+2]=ISICE;
glacieri[c*4+3]=NSOIL;
route=2; return; } // module_sf_noahdrv.F:1167
} // module_sf_noahdrv.F:1182
}
}
static inline void frame_post(thread NoahTables & tables, thread float * DZ8W, thread float * QV3D, thread float * P8W3D, thread float * T3D, thread float & TSK, thread float & HFX, thread float & QFX, thread float & LH, thread float & GRDFLX, thread float & QGH, thread float & SWDOWN, thread float & GLW, thread float & SMSTAV, thread float & SMSTOT, thread float & SFCRUNOFF, thread float & UDRUNOFF, thread int & IVGTYP, thread int & ISLTYP, thread int & ISURBAN, thread int & ISICE, thread float & VEGFRA, thread float & ALBEDO, thread float & ALBBCK, thread float & ZNT, thread float & Z0, thread float & TMN, thread float & XLAND, thread float & XICE, thread float & EMISS, thread float & EMBCK, thread float & SNOWC, thread float & QSFC, thread float & RAINBL, thread float & DT, thread float * DZS, thread int & ITIMESTEP, thread float * SMOIS, thread float * TSLB, thread float & SNOW, thread float & CANWAT, thread float & CHS, thread float & CHS2, thread float & CQS2, thread float & SR, thread float & CHKLOWQ, thread float & LAI, thread float & QZ0, thread float * SH2O, thread float & SNOWH, thread float & SNOALB, thread float & SHDMIN, thread float & SHDMAX, thread float & SNOTIME, thread float & ACSNOM, thread float & ACSNOW, thread float & SNOPCX, thread float & POTEVP, thread float * SMCREL, thread float & XICE_THRESHOLD, thread bool & USEMONALB, thread float & RIB, thread float & NOAHRES, thread int & OPT_THCND, thread float & FLX4_2D, thread float & FVB_2D, thread float & FBUR_2D, thread float & FGSN_2D, int I, int J, thread int &route, thread NoahDriverFrame &f, device float *land, device int *landi, device float *glacier, device int *glacieri, uint c){

thread auto& LCZ_1=tables.LCZ_1;
thread auto& LCZ_2=tables.LCZ_2;
thread auto& LCZ_3=tables.LCZ_3;
thread auto& LCZ_4=tables.LCZ_4;
thread auto& LCZ_5=tables.LCZ_5;
thread auto& LCZ_6=tables.LCZ_6;
thread auto& LCZ_7=tables.LCZ_7;
thread auto& LCZ_8=tables.LCZ_8;
thread auto& LCZ_9=tables.LCZ_9;
thread auto& LCZ_10=tables.LCZ_10;
thread auto& LCZ_11=tables.LCZ_11;
thread const char * LUTYPE=tables.LUTYPE;
thread const char * SLTYPE=tables.SLTYPE;
thread auto& CP=tables.CP;
thread auto& R_D=tables.R_D;
thread auto& XLF=tables.XLF;
thread auto& XLV=tables.XLV;
thread auto& RHOWATER=tables.RHOWATER;
thread auto& STBOLT=tables.STBOLT;
thread bool &UA_PHYS=f.UA_PHYS;
thread float &FLX4=f.FLX4;
thread float &FVB=f.FVB;
thread float &FBUR=f.FBUR;
thread float &FGSN=f.FGSN;
thread float *SMAV=f.SMAV;
thread float &ETP=f.ETP;
thread float &SSOIL=f.SSOIL;
thread float &FLX1=f.FLX1;
thread float &FLX2=f.FLX2;
thread float &FLX3=f.FLX3;
thread float &XLAI=f.XLAI;
thread float &FFROZP=f.FFROZP;
thread int &NSOIL=f.NSOIL;
thread int &NS=f.NS;
thread float &LWDN=f.LWDN;
thread float &PRCP=f.PRCP;
thread float &ALBBRD=f.ALBBRD;
thread float &ALBEDOK=f.ALBEDOK;
thread float &ETA=f.ETA;
thread float &ETA_KINEMATIC=f.ETA_KINEMATIC;
thread float &Z0K=f.Z0K;
thread float &RUNOFF1=f.RUNOFF1;
thread float &RUNOFF2=f.RUNOFF2;
thread float &SHEAT=f.SHEAT;
thread float &SOLNET=f.SOLNET;
thread float &FDTLIW=f.FDTLIW;
thread float &FDTW=f.FDTW;
thread float &EMISSI=f.EMISSI;
thread float &SNCOVR=f.SNCOVR;
thread float &SNEQV=f.SNEQV;
thread float &SNOWHK=f.SNOWHK;
thread float &CMC=f.CMC;
thread float &SNOMLT=f.SNOMLT;
thread float &SOILM=f.SOILM;
thread float &SOILW=f.SOILW;
thread float &Q1=f.Q1;
thread float &T1=f.T1;
thread float &SNOTIME1=f.SNOTIME1;
thread float &Z0BRD=f.Z0BRD;
thread float *STC=f.STC;
thread float *SMC=f.SMC;
thread float *SWC=f.SWC;
if(route==1){
FFROZP=land[c*110+0];
DT=land[c*110+1];
LWDN=land[c*110+7];
SOLNET=land[c*110+9];
PRCP=land[c*110+11];
ALBBRD=land[c*110+24];
Z0BRD=land[c*110+27];
Z0K=land[c*110+28];
EMISSI=land[c*110+29];
CMC=land[c*110+31];
T1=land[c*110+32];
STC[0]=land[c*110+33];
STC[1]=land[c*110+34];
STC[2]=land[c*110+35];
STC[3]=land[c*110+36];
SMC[0]=land[c*110+37];
SMC[1]=land[c*110+38];
SMC[2]=land[c*110+39];
SMC[3]=land[c*110+40];
SWC[0]=land[c*110+41];
SWC[1]=land[c*110+42];
SWC[2]=land[c*110+43];
SWC[3]=land[c*110+44];
SNOWHK=land[c*110+45];
SNEQV=land[c*110+46];
ALBEDOK=land[c*110+47];
ETA=land[c*110+50];
SHEAT=land[c*110+51];
ETA_KINEMATIC=land[c*110+52];
ETP=land[c*110+65];
SSOIL=land[c*110+66];
FLX1=land[c*110+67];
FLX2=land[c*110+68];
FLX3=land[c*110+69];
FLX4=land[c*110+70];
FVB=land[c*110+71];
FBUR=land[c*110+72];
FGSN=land[c*110+73];
SNOMLT=land[c*110+74];
SNCOVR=land[c*110+75];
RUNOFF1=land[c*110+76];
RUNOFF2=land[c*110+77];
XLAI=land[c*110+82];
SOILW=land[c*110+87];
SOILM=land[c*110+88];
Q1=land[c*110+89];
SMAV[0]=land[c*110+90];
SMAV[1]=land[c*110+91];
SMAV[2]=land[c*110+92];
SMAV[3]=land[c*110+93];
SNOTIME1=land[c*110+94];
ISURBAN=landi[c*14+2];
NSOIL=landi[c*14+3];
UA_PHYS=landi[c*14+8];
USEMONALB=landi[c*14+10];
OPT_THCND=landi[c*14+12];
}else if(route==2){
FFROZP=glacier[c*49+0];
DT=glacier[c*49+1];
LWDN=glacier[c*49+7];
SOLNET=glacier[c*49+8];
PRCP=glacier[c*49+10];
ALBBRD=glacier[c*49+16];
Z0BRD=glacier[c*49+19];
Z0K=glacier[c*49+20];
EMISSI=glacier[c*49+21];
T1=glacier[c*49+23];
STC[0]=glacier[c*49+24];
STC[1]=glacier[c*49+25];
STC[2]=glacier[c*49+26];
STC[3]=glacier[c*49+27];
SNOWHK=glacier[c*49+28];
SNEQV=glacier[c*49+29];
ALBEDOK=glacier[c*49+30];
ETA=glacier[c*49+32];
SHEAT=glacier[c*49+33];
ETA_KINEMATIC=glacier[c*49+34];
ETP=glacier[c*49+38];
SSOIL=glacier[c*49+39];
FLX1=glacier[c*49+40];
FLX2=glacier[c*49+41];
FLX3=glacier[c*49+42];
SNOMLT=glacier[c*49+43];
SNCOVR=glacier[c*49+44];
RUNOFF1=glacier[c*49+45];
Q1=glacier[c*49+46];
SNOTIME1=glacier[c*49+47];
ISICE=glacieri[c*4+2];
NSOIL=glacieri[c*4+3];
}else{return;}
LAI = XLAI; // module_sf_noahdrv.F:1184
CANWAT = MU(CMC,0x1.f400000000000p+9f); // module_sf_noahdrv.F:1217
SNOW = MU(SNEQV,0x1.f400000000000p+9f); // module_sf_noahdrv.F:1218
SNOWH = SNOWHK; // module_sf_noahdrv.F:1220
ALBEDO = ALBEDOK; // module_sf_noahdrv.F:1221
ALBBCK = ALBBRD; // module_sf_noahdrv.F:1223
Z0 = Z0BRD; // module_sf_noahdrv.F:1224
EMISS = EMISSI; // module_sf_noahdrv.F:1225
ZNT = Z0K; // module_sf_noahdrv.F:1228
TSK = T1; // module_sf_noahdrv.F:1245
HFX = SHEAT; // module_sf_noahdrv.F:1252
POTEVP = AD(POTEVP,MU(ETP,FDTW)); // module_sf_noahdrv.F:1255
QFX = ETA_KINEMATIC; // module_sf_noahdrv.F:1256
LH = ETA; // module_sf_noahdrv.F:1267
GRDFLX = SSOIL; // module_sf_noahdrv.F:1269
SNOWC = SNCOVR; // module_sf_noahdrv.F:1271
CHS2 = CQS2; // module_sf_noahdrv.F:1272
SNOTIME = SNOTIME1; // module_sf_noahdrv.F:1273
QSFC = DV(Q1,SU(0x1.0000000000000p+0f,Q1)); // module_sf_noahdrv.F:1283
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1288
SMOIS[(NS)-(1)] = SMC[(NS)-(1)]; // module_sf_noahdrv.F:1289
TSLB[(NS)-(1)] = STC[(NS)-(1)]; // module_sf_noahdrv.F:1290
SH2O[(NS)-(1)] = SWC[(NS)-(1)]; // module_sf_noahdrv.F:1291
} // module_sf_noahdrv.F:1292
FLX4_2D = FLX4; // module_sf_noahdrv.F:1295
FVB_2D = FVB; // module_sf_noahdrv.F:1296
FBUR_2D = FBUR; // module_sf_noahdrv.F:1297
FGSN_2D = FGSN; // module_sf_noahdrv.F:1298
if (UA_PHYS) { // module_sf_noahdrv.F:1304
NOAHRES = SU(SU(SU(SU(SU(SU(AD(SU(AD(SOLNET,LWDN),SHEAT),SSOIL),ETA),MU(MU(EMISSI,STBOLT),noah_powi(T1,4))),FLX1),FLX2),FLX3),FLX4); // module_sf_noahdrv.F:1305
} else { // module_sf_noahdrv.F:1308
NOAHRES = SU(SU(SU(SU(SU(AD(SU(AD(SOLNET,LWDN),SHEAT),SSOIL),ETA),MU(MU(EMISSI,STBOLT),noah_powi(T1,4))),FLX1),FLX2),FLX3); // module_sf_noahdrv.F:1309
} // module_sf_noahdrv.F:1311
SMSTAV = SOILW; // module_sf_noahdrv.F:1570
SMSTOT = MU(SOILM,0x1.f400000000000p+9f); // module_sf_noahdrv.F:1571
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1572
SMCREL[(NS)-(1)] = SMAV[(NS)-(1)]; // module_sf_noahdrv.F:1573
} // module_sf_noahdrv.F:1574
SFCRUNOFF = AD(SFCRUNOFF,MU(MU(RUNOFF1,DT),0x1.f400000000000p+9f)); // module_sf_noahdrv.F:1578
UDRUNOFF = AD(UDRUNOFF,MU(MU(RUNOFF2,DT),0x1.f400000000000p+9f)); // module_sf_noahdrv.F:1579
if ((FFROZP > 0x1.0000000000000p-1f)) { // module_sf_noahdrv.F:1581
ACSNOW = AD(ACSNOW,MU(PRCP,DT)); // module_sf_noahdrv.F:1582
} // module_sf_noahdrv.F:1583
if ((SNOW > 0x0.0p+0f)) { // module_sf_noahdrv.F:1584
ACSNOM = AD(ACSNOM,MU(SNOMLT,0x1.f400000000000p+9f)); // module_sf_noahdrv.F:1585
SNOPCX = SU(SNOPCX,DV(SNOMLT,FDTLIW)); // module_sf_noahdrv.F:1587
} // module_sf_noahdrv.F:1588
}
kernel void hexv842_mpas_noah_frame_f32(
constant Hexv842MpasNoahFrameF32Resources &r [[buffer(0)]],
constant Hexv842MpasNoahFrameF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
device float *xs=r.floats; device int *ixs=r.integers; device const uint *ts=r.tables;
device NoahDriverFrame *frames=reinterpret_cast<device NoahDriverFrame*>(r.frames);
device int *routes=r.routes; device float *land=r.land; device int *landi=r.landi;
device float *glacier=r.glacier; device int *glacieri=r.glacieri;
device const int *land_status=r.land_status; device const int *glacier_status=r.glacier_status;
int mode=p.mode; int n=p.n;
if(c>=uint(n))return;
thread float x[79];thread int ix[7];for(int j=0;j<79;j++)x[j]=xs[c*79+j];for(int j=0;j<7;j++)ix[j]=ixs[c*7+j];
thread NoahTables tables;thread uint *t=reinterpret_cast<thread uint*>(&tables);for(uint j=0;j<sizeof(NoahTables)/4;j++)t[j]=ts[j];tables.error=0;
thread NoahDriverFrame f={};int route=0;
float TSK_arg=x[8];float HFX_arg=x[9];float QFX_arg=x[10];float LH_arg=x[11];float GRDFLX_arg=x[12];float QGH_arg=x[13];float SWDOWN_arg=x[14];float GLW_arg=x[15];float SMSTAV_arg=x[16];float SMSTOT_arg=x[17];float SFCRUNOFF_arg=x[18];float UDRUNOFF_arg=x[19];int IVGTYP_arg=ix[0];int ISLTYP_arg=ix[1];int ISURBAN_arg=ix[2];int ISICE_arg=ix[3];float VEGFRA_arg=x[20];float ALBEDO_arg=x[21];float ALBBCK_arg=x[22];float ZNT_arg=x[23];float Z0_arg=x[24];float TMN_arg=x[25];float XLAND_arg=x[26];float XICE_arg=x[27];float EMISS_arg=x[28];float EMBCK_arg=x[29];float SNOWC_arg=x[30];float QSFC_arg=x[31];float RAINBL_arg=x[32];float DT_arg=x[33];int ITIMESTEP_arg=ix[4];float SNOW_arg=x[46];float CANWAT_arg=x[47];float CHS_arg=x[48];float CHS2_arg=x[49];float CQS2_arg=x[50];float SR_arg=x[51];float CHKLOWQ_arg=x[52];float LAI_arg=x[53];float QZ0_arg=x[54];float SNOWH_arg=x[59];float SNOALB_arg=x[60];float SHDMIN_arg=x[61];float SHDMAX_arg=x[62];float SNOTIME_arg=x[63];float ACSNOM_arg=x[64];float ACSNOW_arg=x[65];float SNOPCX_arg=x[66];float POTEVP_arg=x[67];float XICE_THRESHOLD_arg=x[72];bool USEMONALB_arg=ix[5];float RIB_arg=x[73];float NOAHRES_arg=x[74];int OPT_THCND_arg=ix[6];float FLX4_2D_arg=x[75];float FVB_2D_arg=x[76];float FBUR_2D_arg=x[77];float FGSN_2D_arg=x[78];

if(mode==0){
frame_pre(tables,x+0,x+2,x+4,x+6,TSK_arg,HFX_arg,QFX_arg,LH_arg,GRDFLX_arg,QGH_arg,SWDOWN_arg,GLW_arg,SMSTAV_arg,SMSTOT_arg,SFCRUNOFF_arg,UDRUNOFF_arg,IVGTYP_arg,ISLTYP_arg,ISURBAN_arg,ISICE_arg,VEGFRA_arg,ALBEDO_arg,ALBBCK_arg,ZNT_arg,Z0_arg,TMN_arg,XLAND_arg,XICE_arg,EMISS_arg,EMBCK_arg,SNOWC_arg,QSFC_arg,RAINBL_arg,DT_arg,x+34,ITIMESTEP_arg,x+38,x+42,SNOW_arg,CANWAT_arg,CHS_arg,CHS2_arg,CQS2_arg,SR_arg,CHKLOWQ_arg,LAI_arg,QZ0_arg,x+55,SNOWH_arg,SNOALB_arg,SHDMIN_arg,SHDMAX_arg,SNOTIME_arg,ACSNOM_arg,ACSNOW_arg,SNOPCX_arg,POTEVP_arg,x+68,XICE_THRESHOLD_arg,USEMONALB_arg,RIB_arg,NOAHRES_arg,OPT_THCND_arg,FLX4_2D_arg,FVB_2D_arg,FBUR_2D_arg,FGSN_2D_arg,r.locations[2*c],r.locations[2*c+1],route,f,land,landi,glacier,glacieri,c);
frames[c]=f;routes[c]=route;r.status[c]=0;
}else{route=routes[c];f=frames[c];if(route==1 && land_status[c]){r.status[c]=land_status[c];return;}
if(route==2 && glacier_status[c]){r.status[c]=glacier_status[c];return;}
frame_post(tables,x+0,x+2,x+4,x+6,TSK_arg,HFX_arg,QFX_arg,LH_arg,GRDFLX_arg,QGH_arg,SWDOWN_arg,GLW_arg,SMSTAV_arg,SMSTOT_arg,SFCRUNOFF_arg,UDRUNOFF_arg,IVGTYP_arg,ISLTYP_arg,ISURBAN_arg,ISICE_arg,VEGFRA_arg,ALBEDO_arg,ALBBCK_arg,ZNT_arg,Z0_arg,TMN_arg,XLAND_arg,XICE_arg,EMISS_arg,EMBCK_arg,SNOWC_arg,QSFC_arg,RAINBL_arg,DT_arg,x+34,ITIMESTEP_arg,x+38,x+42,SNOW_arg,CANWAT_arg,CHS_arg,CHS2_arg,CQS2_arg,SR_arg,CHKLOWQ_arg,LAI_arg,QZ0_arg,x+55,SNOWH_arg,SNOALB_arg,SHDMIN_arg,SHDMAX_arg,SNOTIME_arg,ACSNOM_arg,ACSNOW_arg,SNOPCX_arg,POTEVP_arg,x+68,XICE_THRESHOLD_arg,USEMONALB_arg,RIB_arg,NOAHRES_arg,OPT_THCND_arg,FLX4_2D_arg,FVB_2D_arg,FBUR_2D_arg,FGSN_2D_arg,r.locations[2*c],r.locations[2*c+1],route,f,land,landi,glacier,glacieri,c);
}
x[8]=TSK_arg;x[9]=HFX_arg;x[10]=QFX_arg;x[11]=LH_arg;x[12]=GRDFLX_arg;x[13]=QGH_arg;x[14]=SWDOWN_arg;x[15]=GLW_arg;x[16]=SMSTAV_arg;x[17]=SMSTOT_arg;x[18]=SFCRUNOFF_arg;x[19]=UDRUNOFF_arg;ix[0]=IVGTYP_arg;ix[1]=ISLTYP_arg;ix[2]=ISURBAN_arg;ix[3]=ISICE_arg;x[20]=VEGFRA_arg;x[21]=ALBEDO_arg;x[22]=ALBBCK_arg;x[23]=ZNT_arg;x[24]=Z0_arg;x[25]=TMN_arg;x[26]=XLAND_arg;x[27]=XICE_arg;x[28]=EMISS_arg;x[29]=EMBCK_arg;x[30]=SNOWC_arg;x[31]=QSFC_arg;x[32]=RAINBL_arg;x[33]=DT_arg;ix[4]=ITIMESTEP_arg;x[46]=SNOW_arg;x[47]=CANWAT_arg;x[48]=CHS_arg;x[49]=CHS2_arg;x[50]=CQS2_arg;x[51]=SR_arg;x[52]=CHKLOWQ_arg;x[53]=LAI_arg;x[54]=QZ0_arg;x[59]=SNOWH_arg;x[60]=SNOALB_arg;x[61]=SHDMIN_arg;x[62]=SHDMAX_arg;x[63]=SNOTIME_arg;x[64]=ACSNOM_arg;x[65]=ACSNOW_arg;x[66]=SNOPCX_arg;x[67]=POTEVP_arg;x[72]=XICE_THRESHOLD_arg;ix[5]=USEMONALB_arg;x[73]=RIB_arg;x[74]=NOAHRES_arg;ix[6]=OPT_THCND_arg;x[75]=FLX4_2D_arg;x[76]=FVB_2D_arg;x[77]=FBUR_2D_arg;x[78]=FGSN_2D_arg;

for(int j=0;j<79;j++)xs[c*79+j]=x[j];for(int j=0;j<7;j++)ixs[c*7+j]=ix[j];
}
