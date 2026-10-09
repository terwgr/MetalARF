// needs integration: append to urban_bep_family.metal after its bep_* RN
// helpers. The final solver fragment calls these shared flat and wall flux
// equations; no typed entrypoint or generated ABI is introduced here.
using namespace metal;

static inline float urban_bep_sin(float x) {
    return precise::sin(x);
}
static inline float urban_bep_cos(float x) {
    return precise::cos(x);
}

static inline void urban_bep_flux_wall(
        float ua, float va, float pt, float da, float ptw, float drst,
        float dt,
        float &uva, float &vva, float &uvb, float &vvb,
        float &tva, float &tvb, float &evb) {
    constexpr float cdrag=0.4f, cp_u=1004.0f;
    const float sinr=urban_bep_sin(drst), cosr=urban_bep_cos(drst);
    const float speed=as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(bep_add(bep_mul(ua,ua),bep_mul(va,va))),
        as_type<uint>(0.5f)));
    const float ort=fabs(bep_sub(bep_mul(cosr,ua),bep_mul(sinr,va)));
    uva=bep_mul(bep_mul(bep_div(bep_mul(-cdrag,ort),2.0f),cosr),cosr);
    vva=bep_mul(bep_mul(bep_div(bep_mul(-cdrag,ort),2.0f),sinr),sinr);
    uvb=bep_mul(bep_mul(bep_mul(bep_div(bep_mul(cdrag,ort),2.0f),sinr),cosr),va);
    vvb=bep_mul(bep_mul(bep_mul(bep_div(bep_mul(cdrag,ort),2.0f),sinr),cosr),ua);
    float hc=bep_mul(5.678f,bep_add(1.09f,bep_mul(0.23f,bep_div(speed,0.3048f))));
    hc=min(hc,bep_div(bep_mul(da,cp_u),dt));
    tvb=bep_sub(bep_div(bep_div(bep_mul(hc,ptw),da),cp_u),
                bep_mul(bep_div(bep_div(hc,da),cp_u),pt));
    tva=0.0f;
    evb=bep_div(bep_mul(cdrag,as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(ort),as_type<uint>(3.0f)))),2.0f);
}

static inline void urban_bep_flux_flat(
        float dz, float z0, float ua, float va, float pt, float pt0,
        float ptg, float &uhb, float &vhb, float &thb, float &ehb) {
    constexpr float g_u=9.81f, vk=0.40f;
    constexpr float b=9.4f, cm=7.4f, ch=5.3f, rr=0.74f;
    float utot=as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(bep_add(bep_mul(ua,ua),bep_mul(va,va))),
        as_type<uint>(0.5f)));
    const float zz=bep_mul(dz,0.5f);
    utot=max(utot,0.01f);
    const float ric=bep_div(bep_mul(bep_mul(bep_mul(2.0f,g_u),zz),
        bep_sub(pt,ptg)),bep_mul(bep_add(pt,ptg),bep_mul(utot,utot)));
    const float aa=bep_div(vk,bep_log(bep_div(zz,z0)));
    float fm,fh,c;
    if(ric>0.0f) {
        const float br=bep_mul(0.5f,b);
        fm=bep_div(1.0f,bep_mul(bep_add(1.0f,bep_mul(br,ric)),
                                 bep_add(1.0f,bep_mul(br,ric))));
        fh=fm;
    } else {
        c=bep_mul(bep_mul(bep_mul(b,cm),bep_mul(aa,aa)),
            as_type<float>(soft_gfk_pow_bits(as_type<uint>(bep_div(zz,z0)),
                                             as_type<uint>(0.5f))));
        const float root=as_type<float>(soft_gfk_pow_bits(
            as_type<uint>(-ric),as_type<uint>(0.5f)));
        fm=bep_sub(1.0f,bep_div(bep_mul(b,ric),bep_add(1.0f,bep_mul(c,root))));
        c=bep_div(bep_mul(c,ch),cm);
        fh=bep_sub(1.0f,bep_div(bep_mul(b,ric),bep_add(1.0f,bep_mul(c,root))));
    }
    const float fbuw=bep_mul(bep_mul(bep_mul(bep_mul(-aa,aa),utot),utot),fm);
    const float fbpt=bep_div(bep_mul(bep_mul(bep_mul(bep_mul(-aa,aa),utot),
        bep_sub(pt,ptg)),fh),rr);
    const float ustar=as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(-fbuw),as_type<uint>(0.5f)));
    const float tstar=bep_div(-fbpt,ustar);
    const float buu=bep_mul(bep_mul(bep_div(-g_u,pt0),ustar),tstar);
    uhb=bep_div(bep_mul(bep_mul(-ustar,ustar),ua),utot);
    vhb=bep_div(bep_mul(bep_mul(-ustar,ustar),va),utot);
    thb=bep_mul(-ustar,tstar);
    ehb=buu;
}
