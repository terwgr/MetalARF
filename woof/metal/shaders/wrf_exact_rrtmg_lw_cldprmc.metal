#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
static inline float add32(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float sub32(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float mul32(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float div32(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}
#define RLW_AD(a,b) add32(a,b)
#define RLW_SU(a,b) sub32(a,b)
#define RLW_MU(a,b) mul32(a,b)
#define RLW_DV(a,b) div32(a,b)
struct Resources {
device const float* cldfmc [[id(0)]];
device const float* ciwpmc [[id(1)]];
device const float* clwpmc [[id(2)]];
device const float* cswpmc [[id(3)]];
device const float* reicmc [[id(4)]];
device const float* relqmc [[id(5)]];
device const float* resnmc [[id(6)]];
device const float* absice1 [[id(7)]];
device const float* absice2 [[id(8)]];
device const float* absice3 [[id(9)]];
device const float* absice0 [[id(10)]];
device const float* absliq1 [[id(11)]];
device const int* ngb [[id(12)]];
device float* taucmc [[id(13)]];
device atomic_int* ncb_flag [[id(14)]];
device atomic_int* err_flag [[id(15)]];
};
struct Params { int ncol,nl,inflag,iceflag,liqflag; float absliq0; };
kernel void wrf_exact_rrtmg_lw_cldprmc(constant Resources &r [[buffer(0)]],constant Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int ncol=p.ncol,nl=p.nl,inflag=p.inflag,iceflag=p.iceflag,liqflag=p.liqflag; float absliq0=p.absliq0;
device const float* cldfmc=r.cldfmc;
device const float* ciwpmc=r.ciwpmc;
device const float* clwpmc=r.clwpmc;
device const float* cswpmc=r.cswpmc;
device const float* reicmc=r.reicmc;
device const float* relqmc=r.relqmc;
device const float* resnmc=r.resnmc;
device const float* absice1=r.absice1;
device const float* absice2=r.absice2;
device const float* absice3=r.absice3;
device const float* absice0=r.absice0;
device const float* absliq1=r.absliq1;
device const int* ngb=r.ngb;
device float* taucmc=r.taucmc;
device atomic_int* ncb_flag=r.ncb_flag;
device atomic_int* err_flag=r.err_flag;

    long tid = (long)gid;
    if (tid >= (long)ncol * NGPTLW) return;
    int col = (int)(tid / NGPTLW);
    int ig = (int)(tid % NGPTLW) + 1;
    const float cldmin = 1.e-20f;
    const int icb[16] = {1,2,3,3,3,4,4,4,5,5,5,5,5,5,5,5};
#define MC(a, l) a[((long)col * NGPTLW + (ig - 1)) * nl + ((l) - 1)]
#define AB2(tab, ld, i, b) tab[((i) - 1) + ((b) - 1) * (ld)]

    for (int lay = 1; lay <= nl; ++lay) {
        float cwp = RLW_AD(RLW_AD(MC(ciwpmc, lay), MC(clwpmc, lay)),
                           MC(cswpmc, lay));
        if (!(soft32_greater_equal(as_type<uint>(MC(cldfmc, lay)), as_type<uint>(cldmin))
              && (soft32_greater_equal(as_type<uint>(cwp), as_type<uint>(cldmin)) || soft32_greater_equal(as_type<uint>(MC(taucmc, lay)), as_type<uint>(cldmin)))))
            continue;
        if (inflag == 0) return;             // taucmc already set
        if (inflag == 1) { atomic_exchange_explicit(err_flag, 1, memory_order_relaxed); return; }

        float radice = reicmc[(long)col * nl + (lay - 1)];
        float abscoice, abscosno;
        float ice_plus_snow = RLW_AD(MC(ciwpmc, lay), MC(cswpmc, lay));
        if (soft32_equal(as_type<uint>(ice_plus_snow),0u)) {
            abscoice = 0.0f;
            abscosno = 0.0f;
        } else if (iceflag == 0) {
            if (radice < 10.0f) { atomic_exchange_explicit(err_flag, 2, memory_order_relaxed); return; }
            abscoice = RLW_AD(absice0[0], RLW_DV(absice0[1], radice));
            abscosno = 0.0f;
        } else if (iceflag == 1) {
            if (radice < 13.0f || radice > 130.0f) {
                atomic_exchange_explicit(err_flag, 3, memory_order_relaxed); return;
            }
            atomic_exchange_explicit(&ncb_flag[col], 1, memory_order_relaxed);
            int ib = icb[ngb[ig - 1] - 1];
            abscoice = RLW_AD(AB2(absice1, 2, 1, ib),
                              RLW_DV(AB2(absice1, 2, 2, ib), radice));
            abscosno = 0.0f;
        } else if (iceflag == 2) {
            if (radice < 5.0f || radice > 131.0f) {
                atomic_exchange_explicit(err_flag, 4, memory_order_relaxed); return;
            }
            atomic_exchange_explicit(&ncb_flag[col], 1, memory_order_relaxed);
            float factor = RLW_DV(RLW_SU(radice, 2.0f), 3.0f);
            int index = (int)factor;
            if (index == 43) index = 42;
            float fint = RLW_SU(factor, (float)index);
            int ib = ngb[ig - 1];
            abscoice = RLW_AD(AB2(absice2, 43, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absice2, 43, index + 1, ib),
                                    AB2(absice2, 43, index, ib))));
            abscosno = 0.0f;
        } else {                              // iceflag >= 3
            if (radice < 5.0f || radice > 140.0f) {
                atomic_exchange_explicit(err_flag, 5, memory_order_relaxed); return;
            }
            atomic_exchange_explicit(&ncb_flag[col], 1, memory_order_relaxed);
            float factor = RLW_DV(RLW_SU(radice, 2.0f), 3.0f);
            int index = (int)factor;
            if (index == 46) index = 45;
            float fint = RLW_SU(factor, (float)index);
            int ib = ngb[ig - 1];
            abscoice = RLW_AD(AB2(absice3, 46, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absice3, 46, index + 1, ib),
                                    AB2(absice3, 46, index, ib))));
            abscosno = 0.0f;
        }

        if (soft32_greater(as_type<uint>(MC(cswpmc, lay)),0u) && iceflag == 5) {
            float radsno = resnmc[(long)col * nl + (lay - 1)];
            if (radsno < 5.0f || radsno > 140.0f) {
                atomic_exchange_explicit(err_flag, 6, memory_order_relaxed); return;
            }
            atomic_exchange_explicit(&ncb_flag[col], 1, memory_order_relaxed);
            float factor = RLW_DV(RLW_SU(radsno, 2.0f), 3.0f);
            int index = (int)factor;
            if (index == 46) index = 45;
            float fint = RLW_SU(factor, (float)index);
            int ib = ngb[ig - 1];
            abscosno = RLW_AD(AB2(absice3, 46, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absice3, 46, index + 1, ib),
                                    AB2(absice3, 46, index, ib))));
        }

        float abscoliq;
        if (soft32_equal(as_type<uint>(MC(clwpmc, lay)),0u)) {
            abscoliq = 0.0f;
        } else if (liqflag == 0) {
            abscoliq = absliq0;
        } else {                              // liqflag == 1
            float radliq = relqmc[(long)col * nl + (lay - 1)];
            if (radliq < 2.5f || radliq > 60.0f) {
                atomic_exchange_explicit(err_flag, 7, memory_order_relaxed); return;
            }
            int index = (int)RLW_SU(radliq, 1.5f);
            if (index == 0) index = 1;
            if (index == 58) index = 57;
            float fint = RLW_SU(RLW_SU(radliq, 1.5f), (float)index);
            int ib = ngb[ig - 1];
            abscoliq = RLW_AD(AB2(absliq1, 58, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absliq1, 58, index + 1, ib),
                                    AB2(absliq1, 58, index, ib))));
        }

        MC(taucmc, lay) = RLW_AD(
            RLW_AD(RLW_MU(MC(ciwpmc, lay), abscoice),
                   RLW_MU(MC(clwpmc, lay), abscoliq)),
            RLW_MU(MC(cswpmc, lay), abscosno));
    }
#undef MC
#undef AB2

}
