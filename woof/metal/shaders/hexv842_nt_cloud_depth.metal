#include "hexv842_nt_cloud_depth_bindings.metalh"
#include <metal_stdlib>
#include "soft_binary32.metalh"
using namespace metal;

static float a32(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static float s32(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
// Selected ntiedtke_cloud_depth: input levels paph,pdmfup; surface
// ldcum,kcbot,kctop; in/out type,ictop0,prfl and five downdraft arrays.
kernel void hexv842_nt_cloud_depth(constant Hexv842NtCloudDepthResources &r [[buffer(0)]], constant Hexv842NtCloudDepthParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    device const float *in=r.in;
    device const int *sf=r.sf;
    device float *out=r.out;
    device int *sout=r.sout;

    if (i>=uint(p.ncol)) return;
    int n=p.ncol, klev=p.klev, st=(klev+2)*n;
    device const float *paph=in,*pdmfup=in+st;
    device const int *ldcum=sf,*kcbot=sf+n,*kctop=sf+2*n;
    device int *ktype=sout,*ictop0=sout+n;
    device float *prfl=out+5*st;
    if (ldcum[i]) {
        int ikb=kcbot[i],itopm2=kctop[i];
        float zpbmpt=s32(paph[ikb*n+i],paph[itopm2*n+i]);
        if (ktype[i]==1 && zpbmpt<20000.0f) ktype[i]=2;
        if (ktype[i]==2 && zpbmpt>=20000.0f) ktype[i]=1;
        ictop0[i]=itopm2;
    }
    float zrfl=pdmfup[n+i];
    for (int jk=2;jk<=klev;++jk) zrfl=a32(zrfl,pdmfup[jk*n+i]);
    prfl[i]=zrfl;
    for (int jk=1;jk<=klev;++jk)
        for (int f=0;f<5;++f) out[f*st+jk*n+i]=0.0f;
}
