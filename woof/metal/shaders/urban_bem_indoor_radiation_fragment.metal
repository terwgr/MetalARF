#include <metal_stdlib>
#include "urban_bem_indoor_radiation_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_atan.metalh"
#include "soft_gfk_log.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmir_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_sqrt(float a) { return as_type<float>(soft32_sqrt(as_type<uint>(a))); }
static inline float ubmir_atan(float a) { return as_type<float>(soft_gfk_atanf(as_type<uint>(a))); }
static inline float ubmir_log(float a) { return as_type<float>(soft_gfk_log(as_type<uint>(a))); }
static inline float ubmir_pow4(float a) { const float h = ubmir_mul(a, a); return ubmir_mul(h, h); }

static inline float ubmir_fprl(float vx, float vy) {
    const float vx2 = ubmir_mul(vx, vx), vy2 = ubmir_mul(vy, vy);
    const float sx = ubmir_sqrt(ubmir_add(1.0f, vx2));
    const float sy = ubmir_sqrt(ubmir_add(1.0f, vy2));
    const float denom = ubmir_add(ubmir_add(1.0f, vx2), vy2);
    const float inside = ubmir_div(ubmir_mul(ubmir_add(1.0f, vx2), ubmir_add(1.0f, vy2)), denom);
    float g = ubmir_log(ubmir_sqrt(inside));
    g = ubmir_add(g, ubmir_mul(ubmir_mul(vy, sx), ubmir_atan(ubmir_div(vy, sx))));
    g = ubmir_add(g, ubmir_mul(ubmir_mul(vx, sy), ubmir_atan(ubmir_div(vx, sy))));
    g = ubmir_sub(g, ubmir_mul(vy, ubmir_atan(vy)));
    g = ubmir_sub(g, ubmir_mul(vx, ubmir_atan(vx)));
    return ubmir_mul(ubmir_div(2.0f, ubmir_mul(ubmir_mul(3.141592653f, vx), vy)), g);
}

static inline float ubmir_fnrm(float wx, float wy, float wz) {
    const float wx2 = ubmir_mul(wx, wx), wy2 = ubmir_mul(wy, wy);
    const float rootz = ubmir_sqrt(wz);
    float g = ubmir_add(ubmir_mul(wy, ubmir_atan(ubmir_div(1.0f, wy))), ubmir_mul(wx, ubmir_atan(ubmir_div(1.0f, wx))));
    g = ubmir_sub(g, ubmir_mul(rootz, ubmir_atan(ubmir_div(1.0f, rootz))));
    const float log0 = ubmir_log(ubmir_div(ubmir_mul(ubmir_add(1.0f, wx2), ubmir_add(1.0f, wy2)), ubmir_add(1.0f, wz)));
    const float logy = ubmir_log(ubmir_div(ubmir_mul(wy2, ubmir_add(1.0f, wz)), ubmir_mul(wz, ubmir_add(1.0f, wy2))));
    const float logx = ubmir_log(ubmir_div(ubmir_mul(wx2, ubmir_add(1.0f, wz)), ubmir_mul(wz, ubmir_add(1.0f, wx2))));
    g = ubmir_add(g, ubmir_mul(0.25f, ubmir_add(ubmir_add(log0, ubmir_mul(wy2, logy)), ubmir_mul(wx2, logx))));
    return ubmir_mul(ubmir_div(1.0f, ubmir_mul(3.141592653f, wy)), g);
}

static inline bool ubmir_solve(thread float *a, thread float *b) {
    int ipiv[6];
    for (int j = 0; j < 6; ++j) ipiv[j] = 0;
    for (int i = 0; i < 6; ++i) {
        float big = 0.0f; int irow = -1, icol = -1;
        for (int j = 0; j < 6; ++j) if (ipiv[j] != 1)
            for (int k = 0; k < 6; ++k) {
                if (ipiv[k] == 0) { const float v = fabs(a[j * 6 + k]); if (v >= big) { big = v; irow = j; icol = k; } }
                else if (ipiv[k] > 1) return false;
            }
        if (irow < 0 || icol < 0) return false;
        ++ipiv[icol];
        if (irow != icol) { for (int l = 0; l < 6; ++l) { const float t = a[irow * 6 + l]; a[irow * 6 + l] = a[icol * 6 + l]; a[icol * 6 + l] = t; } const float t = b[irow]; b[irow] = b[icol]; b[icol] = t; }
        const float pivot = a[icol * 6 + icol]; if (pivot == 0.0f || !isfinite(pivot)) return false;
        const float inv = ubmir_div(1.0f, pivot); a[icol * 6 + icol] = 1.0f;
        for (int l = 0; l < 6; ++l) a[icol * 6 + l] = ubmir_mul(a[icol * 6 + l], inv);
        b[icol] = ubmir_mul(b[icol], inv);
        for (int ll = 0; ll < 6; ++ll) if (ll != icol) { const float d = a[ll * 6 + icol]; a[ll * 6 + icol] = 0.0f; for (int l = 0; l < 6; ++l) a[ll * 6 + l] = ubmir_sub(a[ll * 6 + l], ubmir_mul(a[icol * 6 + l], d)); b[ll] = ubmir_sub(b[ll], ubmir_mul(b[icol], d)); }
    }
    return true;
}

// These algebra routines are the source-order `bem_algebra_short` and
// `bem_algebra_long` from module_sf_bem.F; arrays are row-major in this ABI.
static inline void ubmir_algebra_short(float rstr, float albwal, float albwin, float aw, float bw, float zw, float pwin, thread float *a, thread float *b) {
    for (int i = 0; i < 36; ++i) a[i] = 0.0f;
    for (int i = 0; i < 6; ++i) b[i] = -rstr;
    const float albm = ubmir_add(ubmir_mul(pwin, albwin), ubmir_mul(ubmir_sub(1.0f, pwin), albwal)); float q;
    a[0] = -1.0f; a[1] = ubmir_mul(albm, ubmir_fprl(ubmir_div(aw,bw),ubmir_div(zw,bw)));
    q=ubmir_fnrm(ubmir_div(aw,zw),ubmir_div(bw,zw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(bw,bw)),ubmir_mul(zw,zw))); a[2]=ubmir_mul(ubmir_mul(albm,ubmir_div(bw,aw)),q); a[3]=a[2];
    q=ubmir_fnrm(ubmir_div(zw,aw),ubmir_div(bw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw))); a[4]=ubmir_mul(ubmir_mul(albwal,ubmir_div(bw,zw)),q); a[5]=a[4];
    a[6]=a[1]; a[7]=-1.0f; a[8]=a[2]; a[9]=a[3]; a[10]=a[4]; a[11]=a[5];
    q=ubmir_fnrm(ubmir_div(bw,zw),ubmir_div(aw,zw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(aw,aw)),ubmir_mul(zw,zw))); a[12]=ubmir_mul(ubmir_mul(albm,ubmir_div(aw,bw)),q); a[13]=a[12]; a[14]=-1.0f;
    a[15]=ubmir_mul(albm,ubmir_fprl(ubmir_div(zw,aw),ubmir_div(bw,aw)));
    q=ubmir_fnrm(ubmir_div(zw,bw),ubmir_div(aw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw))); a[16]=ubmir_mul(ubmir_mul(albwal,ubmir_div(aw,zw)),q); a[17]=a[16];
    a[18]=a[12]; a[19]=a[13]; a[20]=a[15]; a[21]=-1.0f; a[22]=a[16]; a[23]=a[17];
    q=ubmir_fnrm(ubmir_div(bw,aw),ubmir_div(zw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw))); a[24]=ubmir_mul(ubmir_mul(albm,ubmir_div(zw,bw)),q); a[25]=a[24];
    q=ubmir_fnrm(ubmir_div(aw,bw),ubmir_div(zw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw))); a[26]=ubmir_mul(ubmir_mul(albm,ubmir_div(zw,aw)),q); a[27]=a[26]; a[28]=-1.0f;
    a[29]=ubmir_mul(albwal,ubmir_fprl(ubmir_div(aw,zw),ubmir_div(bw,zw))); a[30]=a[24]; a[31]=a[24]; a[32]=a[26]; a[33]=a[27]; a[34]=a[29]; a[35]=-1.0f;
}

static inline void ubmir_algebra_long(float emwal, float emwin, float sigma, thread const float *twal, thread const float *twin, float pwin, float aw, float bw, float zw, thread float *a, thread float *b) {
    // The equivalent of `bem_algebra_long` is intentionally delegated to the
    // exact same helper body as the registered radiation fragment. This body
    // is kept explicit below to avoid a runtime or host-side fallback.
    float wall[6], wind[6], t4w[6], t4n[4]; for (int i=0;i<36;++i) a[i]=0.0f;
    for (int i=0;i<6;++i) { wall[i]=0.0f; wind[i]=0.0f; t4w[i]=ubmir_pow4(twal[i]); }
    for (int i=0;i<4;++i) t4n[i]=ubmir_pow4(twin[i]);
    const float emwal_av=ubmir_mul(ubmir_sub(1.0f,pwin),emwal), emwin_av=ubmir_mul(pwin,emwin), em_av=ubmir_add(emwal_av,emwin_av);
    float fp,fx,fy,sum;
    fp=ubmir_fprl(ubmir_div(aw,zw),ubmir_div(bw,zw)); fx=ubmir_fnrm(ubmir_div(aw,bw),ubmir_div(zw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw))); fy=ubmir_fnrm(ubmir_div(bw,aw),ubmir_div(zw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw)));
    for(int row=0;row<2;++row){int wi=4+row;float t1=ubmir_mul(ubmir_mul(ubmir_mul(emwal,sigma),t4w[wi]),fp);sum=ubmir_add(ubmir_mul(emwal_av,t4w[2]),ubmir_mul(emwal_av,t4w[3]));float t2=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(zw,aw)),fx);sum=ubmir_add(ubmir_mul(emwal_av,t4w[0]),ubmir_mul(emwal_av,t4w[1]));float t3=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(zw,bw)),fy);wall[row]=ubmir_add(ubmir_add(t1,t2),t3);}
    fp=ubmir_fprl(ubmir_div(zw,aw),ubmir_div(bw,aw));fx=ubmir_fnrm(ubmir_div(bw,zw),ubmir_div(aw,zw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(aw,aw)),ubmir_mul(zw,zw)));fy=ubmir_fnrm(ubmir_div(zw,bw),ubmir_div(aw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw)));
    for(int row=2;row<4;++row){int wi=row==2?3:2;float t1=ubmir_mul(ubmir_mul(ubmir_mul(emwal_av,sigma),t4w[wi]),fp);sum=ubmir_add(ubmir_mul(emwal_av,t4w[1]),ubmir_mul(emwal_av,t4w[0]));float t2=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(aw,bw)),fx);sum=ubmir_add(ubmir_mul(emwal,t4w[4]),ubmir_mul(emwal,t4w[5]));float t3=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(aw,zw)),fy);wall[row]=ubmir_add(ubmir_add(t1,t2),t3);}
    fp=ubmir_fprl(ubmir_div(aw,bw),ubmir_div(zw,bw));fx=ubmir_fnrm(ubmir_div(aw,zw),ubmir_div(bw,zw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(bw,bw)),ubmir_mul(zw,zw)));fy=ubmir_fnrm(ubmir_div(zw,aw),ubmir_div(bw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw)));
    for(int row=4;row<6;++row){int wi=row==4?1:0;float t1=ubmir_mul(ubmir_mul(ubmir_mul(emwal_av,sigma),t4w[wi]),fp);sum=ubmir_add(ubmir_mul(emwal_av,t4w[2]),ubmir_mul(emwal_av,t4w[3]));float t2=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(bw,aw)),fx);sum=ubmir_add(ubmir_mul(emwal,t4w[4]),ubmir_mul(emwal,t4w[5]));float t3=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(bw,zw)),fy);wall[row]=ubmir_add(ubmir_add(t1,t2),t3);}
    fx=ubmir_fnrm(ubmir_div(aw,bw),ubmir_div(zw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw)));fy=ubmir_fnrm(ubmir_div(bw,aw),ubmir_div(zw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw)));
    for(int row=0;row<2;++row){sum=ubmir_add(ubmir_mul(emwin_av,t4n[2]),ubmir_mul(emwin_av,t4n[3]));float t1=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(zw,aw)),fx);sum=ubmir_add(ubmir_mul(emwin_av,t4n[0]),ubmir_mul(emwin_av,t4n[1]));float t2=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(zw,bw)),fy);wind[row]=ubmir_add(t1,t2);}
    fp=ubmir_fprl(ubmir_div(zw,aw),ubmir_div(bw,aw));fx=ubmir_fnrm(ubmir_div(bw,zw),ubmir_div(aw,zw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(aw,aw)),ubmir_mul(zw,zw)));
    for(int row=2;row<4;++row){int wi=row==2?3:2;float t1=ubmir_mul(ubmir_mul(ubmir_mul(emwin_av,sigma),t4n[wi]),fp);sum=ubmir_add(ubmir_mul(emwin_av,t4n[1]),ubmir_mul(emwin_av,t4n[0]));float t2=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(aw,bw)),fx);wind[row]=ubmir_add(t1,t2);}
    fp=ubmir_fprl(ubmir_div(aw,bw),ubmir_div(zw,bw));fy=ubmir_fnrm(ubmir_div(aw,zw),ubmir_div(bw,zw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(bw,bw)),ubmir_mul(zw,zw)));
    for(int row=4;row<6;++row){int wi=row==4?1:0;float t1=ubmir_mul(ubmir_mul(ubmir_mul(emwin_av,sigma),t4n[wi]),fp);sum=ubmir_add(ubmir_mul(emwin_av,t4n[2]),ubmir_mul(emwin_av,t4n[3]));float t2=ubmir_mul(ubmir_mul(ubmir_mul(sigma,sum),ubmir_div(bw,aw)),fy);wind[row]=ubmir_add(t1,t2);}
    for(int i=0;i<6;++i)b[i]=ubmir_add(wall[i],wind[i]);
    fy=ubmir_fnrm(ubmir_div(bw,aw),ubmir_div(zw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw)));float m11=ubmir_mul(ubmir_mul(ubmir_sub(em_av,1.0f),ubmir_div(zw,bw)),fy);
    fx=ubmir_fnrm(ubmir_div(aw,bw),ubmir_div(zw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw)));float m13=ubmir_mul(ubmir_mul(ubmir_sub(em_av,1.0f),ubmir_div(zw,aw)),fx);fp=ubmir_fprl(ubmir_div(aw,zw),ubmir_div(bw,zw));float m15=ubmir_mul(ubmir_sub(emwal,1.0f),fp);
    a[0]=m11;a[1]=m11;a[2]=m13;a[3]=m13;a[4]=m15;a[5]=1.0f;a[6]=m11;a[7]=m11;a[8]=m13;a[9]=m13;a[10]=1.0f;a[11]=m15;
    fx=ubmir_fnrm(ubmir_div(bw,zw),ubmir_div(aw,zw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(aw,aw)),ubmir_mul(zw,zw)));float m31=ubmir_mul(ubmir_mul(ubmir_sub(em_av,1.0f),ubmir_div(aw,bw)),fx);fp=ubmir_fprl(ubmir_div(zw,aw),ubmir_div(bw,aw));float m34=ubmir_mul(ubmir_sub(em_av,1.0f),fp);fy=ubmir_fnrm(ubmir_div(zw,bw),ubmir_div(aw,bw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(zw,zw)),ubmir_mul(bw,bw)));float m35=ubmir_mul(ubmir_mul(ubmir_sub(emwal,1.0f),ubmir_div(aw,zw)),fy);
    a[12]=m31;a[13]=m31;a[14]=1.0f;a[15]=m34;a[16]=m35;a[17]=m35;a[18]=m31;a[19]=m31;a[20]=m34;a[21]=1.0f;a[22]=m35;a[23]=m35;
    fp=ubmir_fprl(ubmir_div(aw,bw),ubmir_div(zw,bw));float m52=ubmir_mul(ubmir_sub(em_av,1.0f),fp);fx=ubmir_fnrm(ubmir_div(aw,zw),ubmir_div(bw,zw),ubmir_div(ubmir_add(ubmir_mul(aw,aw),ubmir_mul(bw,bw)),ubmir_mul(zw,zw)));float m53=ubmir_mul(ubmir_mul(ubmir_sub(em_av,1.0f),ubmir_div(bw,aw)),fx);fy=ubmir_fnrm(ubmir_div(zw,aw),ubmir_div(bw,aw),ubmir_div(ubmir_add(ubmir_mul(bw,bw),ubmir_mul(zw,zw)),ubmir_mul(aw,aw)));float m55=ubmir_mul(ubmir_mul(ubmir_sub(emwal,1.0f),ubmir_div(bw,zw)),fy);
    a[24]=1.0f;a[25]=m52;a[26]=m53;a[27]=m53;a[28]=m55;a[29]=m55;a[30]=m52;a[31]=1.0f;a[32]=m53;a[33]=m53;a[34]=m55;a[35]=m55;
}

// One work item is (column, building slot, level). WRF first gathers both
// urban directions into the four wall faces, then runs bem_bem once per
// active building. Flat layers are row-major, with NDM's final state selected
// exactly as the preceding bb_bep1d gather leaves it (direction index 1).
kernel void urban_bem_indoor_radiation_fragment(
    constant UrbanBemIndoorRadiationFragmentResources &r [[buffer(0)]],
    constant UrbanBemIndoorRadiationFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const int *active=r.active_cols, *utype=r.utype, *nlev=r.building_nlev;
    device const float *cf=r.class_float, *bs=r.bs, *rsw=r.rsw, *tw=r.tw, *twlev=r.twlev;
    device const float *tflev=r.tflev, *tglev=r.tglev, *tr=r.tr;
    device float *rsint=r.rsint, *rlint=r.rlint; device int *error=r.error;
    if(item>=uint(p.nitem))return;
    const uint lev=item%18u, b=(item/18u)%15u, col=item/(18u*15u);
    error[item]=0; const ulong out6=ulong(item)*6ul;
    for(uint i=0;i<6;++i){rsint[out6+i]=0.0f;rlint[out6+i]=0.0f;}
    if(col>=uint(p.ncol)||active[col]==0)return;
    const int nl=nlev[ulong(col)*15ul+b]; if(nl<=int(lev))return;
    const int cls=utype[col]-1; if(cls<0||cls>=p.nurbm){error[item]=3811;return;}
    const float bw=bs[ulong(col)*2ul], bl=bs[ulong(col)*2ul+1ul], zw=p.dz_u;
    const float albwal=cf[77+cls], albwin=cf[1052+cls], emwal=cf[110+cls], emwin=cf[1063+cls], pwin=cf[1085+cls];
    if(!(bw>0.0f&&bl>0.0f&&zw>0.0f)||!isfinite(bw)||!isfinite(bl)||!isfinite(zw)||!(pwin>=0.0f&&pwin<=1.0f)||!isfinite(pwin)){error[item]=3812;return;}
    const ulong base_tw=((((ulong(col)*15ul+b)*10ul)*18ul+lev)*4ul);
    const ulong base_twin=(((ulong(col)*15ul+b)*18ul+lev)*4ul);
    thread float wall_t[6], window_t[4], area[6], rswall[4];
    for(uint f=0;f<4;++f){wall_t[f]=tw[base_tw+f];window_t[f]=twlev[base_twin+f];rswall[f]=rsw[(ulong(col)*4ul+f)*18ul+lev];}
    if(lev==0u){
        for(uint f=0;f<6;++f)area[f]=1.0f;
        area[0]=area[1]=ubmir_mul(zw,bw); area[2]=area[3]=ubmir_mul(zw,bl); area[4]=area[5]=ubmir_mul(bw,bl);
        const ulong base_tg=((ulong(col)*15ul+b)*10ul+9ul)*2ul+1ul;
        wall_t[4]=tglev[base_tg];
        if(nl==1) wall_t[5]=tr[(ulong(col)*10ul)*18ul*2ul+1ul];
        else wall_t[5]=tflev[((((ulong(col)*15ul+b)*10ul)*17ul)*2ul)+1ul];
    } else if(lev+1u==uint(nl)) {
        for(uint f=0;f<6;++f)area[f]=1.0f;
        area[0]=area[1]=ubmir_mul(zw,bw); area[2]=area[3]=ubmir_mul(zw,bl); area[4]=area[5]=ubmir_mul(bw,bl);
        const ulong floor_last=((((ulong(col)*15ul+b)*10ul+9ul)*17ul+(lev-1u))*2ul)+1ul;
        wall_t[4]=tflev[floor_last]; wall_t[5]=tr[(ulong(col)*10ul)*18ul*2ul+1ul];
    } else {
        for(uint f=0;f<6;++f)area[f]=1.0f;
        area[0]=area[1]=ubmir_mul(zw,bw); area[2]=area[3]=ubmir_mul(zw,bl); area[4]=area[5]=ubmir_mul(bw,bl);
        wall_t[4]=tflev[((((ulong(col)*15ul+b)*10ul+9ul)*17ul+(lev-1u))*2ul)+1ul];
        wall_t[5]=tflev[((((ulong(col)*15ul+b)*10ul)*17ul+lev)*2ul)+1ul];
    }
    if(!(nl>0)){error[item]=3816;return;}
    float area_total=0.0f,rstr=0.0f;
    for(uint f=0;f<4;++f){const float term=ubmir_mul(ubmir_mul(area[f],pwin),ubmir_mul(ubmir_sub(1.0f,albwin),rswall[f]));rstr=ubmir_add(rstr,term);}
    for(uint f=0;f<6;++f)area_total=ubmir_add(area_total,area[f]);
    if(!(area_total>0.0f)||!isfinite(area_total)){error[item]=3813;return;} rstr=ubmir_div(rstr,area_total);
    float asw[36],bsw[6]; ubmir_algebra_short(rstr,albwal,albwin,bw,bl,zw,pwin,asw,bsw);
    if(!ubmir_solve(asw,bsw)){error[item]=3814;return;}
    float alw[36],blw[6]; ubmir_algebra_long(emwal,emwin,5.67e-8f,wall_t,window_t,pwin,bw,bl,zw,alw,blw);
    if(!ubmir_solve(alw,blw)){error[item]=3815;return;}
    for(uint f=0;f<6;++f){rsint[out6+f]=bsw[f];rlint[out6+f]=blw[f];}
}
