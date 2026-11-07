#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_ozone_time_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_ozone_time_f32(
    constant Hexv842MpasOzoneTimeF32Resources &r [[buffer(0)]],
    constant Hexv842MpasOzoneTimeF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    int i=int(tid),plane=p.nlev*p.ncells;
    if(i>=plane || i%p.ncells>=p.nsolve) return;
    const int dates[12]={16,45,75,105,136,166,197,228,258,289,319,350};
    float shifted=mpas_add(p.julian,1.0f);
    int daynum=int(shifted), original=daynum;
    ulong day=soft64_sub(soft64_from_f32_bits(as_type<uint>(shifted)),soft64_from_i32(original));
    daynum=daynum%365; if(daynum==0) daynum=365;
    day=soft64_add(day,soft64_from_i32(daynum));
    int next=0;
    for(int m=0;m<12;m++) if(soft64_less(day,soft64_from_i32(dates[m]))) {next=m;break;}
    int prev=next>0?next-1:11;
    ulong plus=soft64_from_i32(dates[next]), minus=soft64_from_i32(dates[prev]);
    ulong f1,f2;
    if(next==0) {
        ulong year=soft64_from_i32(365), end=soft64_add(plus,year),delta=soft64_sub(end,minus);
        if(soft64_less(plus,day)) {
            f1=soft64_div(soft64_sub(end,day),delta);
            f2=soft64_div(soft64_sub(day,minus),delta);
        } else {
            f1=soft64_div(soft64_sub(plus,day),delta);
            f2=soft64_div(soft64_sub(soft64_add(day,year),minus),delta);
        }
    } else {
        ulong delta=soft64_sub(plus,minus);
        f1=soft64_div(soft64_sub(plus,day),delta);
        f2=soft64_div(soft64_sub(day,minus),delta);
    }
    float a=as_type<float>(soft64_to_f32_bits(f1)),b=as_type<float>(soft64_to_f32_bits(f2));
    r.out[i]=mpas_add(mpas_mul(a,r.monthly[prev*plane+i]),mpas_mul(b,r.monthly[next*plane+i]));
}
