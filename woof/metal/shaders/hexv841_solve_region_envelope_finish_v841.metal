// Hex v8.4.1 Metal transcription of CUDA kernel solve_region_envelope_finish_v841 (woof/hex/cuda_solve_region_v841.py:214).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_solve_region_envelope_finish_v841_bindings.metalh"
#pragma clang fp contract(off)

#define RC2(k,c,nc) ((k)*(nc) + (c))
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RQI3(t,k,c,nlev,nc) (((t)*((nlev)+1) + (k))*(nc) + (c))
#define RCES(c,s,me) ((c)*(me) + (s))
#define RADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
#define REGIONAL_N_RELAX_ZONE 5
#define REGIONAL_N_SPEC_ZONE 2
#define MPAS_FTZ_FALLBACK_ENABLED 1



#define SR_PART 5
#define SR_OUT 4

struct sr_env {
    float vmin;
    float vmax;
    int nan;
    float amax;
    long aidx;
};

static inline void sr_env_init(thread sr_env &e) {
    e.vmin = as_type<float>(0x7f800000);   /* +inf */
    e.vmax = as_type<float>(0xff800000u);  /* -inf */
    e.nan = 0;
    e.amax = -1.0f;
    e.aidx = 0x7fffffffffffffffL;
}

static inline void sr_env_merge(thread sr_env &a, thread const sr_env &b) {
    a.vmin = fmin(a.vmin, b.vmin);
    a.vmax = fmax(a.vmax, b.vmax);
    a.nan |= b.nan;
    if (b.amax > a.amax || (b.amax == a.amax && b.aidx < a.aidx)) {
        a.amax = b.amax;
        a.aidx = b.aidx;
    }
}

static inline void sr_env_take(thread sr_env &e, float v, long index) {
    if (v != v) {
        e.nan = 1;
    } else {
        e.vmin = fmin(e.vmin, v);
        e.vmax = fmax(e.vmax, v);
    }
    const float a = fabs(v);
    if (a > e.amax || (a == e.amax && index < e.aidx)) {
        e.amax = a;
        e.aidx = index;
    }
}

template <typename Row>
static inline void sr_env_store(Row row, thread const sr_env &e) {
    row[0] = long(as_type<int>(e.vmin));
    row[1] = long(as_type<int>(e.vmax));
    row[2] = long(e.nan);
    row[3] = e.aidx;
    row[4] = long(as_type<int>(e.amax));
}

template <typename Row>
static inline void sr_env_load(thread sr_env &e, Row row) {
    e.vmin = as_type<float>(int(row[0]));
    e.vmax = as_type<float>(int(row[1]));
    e.nan = int(row[2]);
    e.aidx = row[3];
    e.amax = as_type<float>(int(row[4]));
}

template <typename Row>
static inline void sr_env_block_reduce(
    thread sr_env &mine, Row out_row, uint t, uint block_dim,
    threadgroup float *s_min, threadgroup float *s_max,
    threadgroup int *s_nan, threadgroup float *s_amax,
    threadgroup long *s_aidx)
{
    s_min[t] = mine.vmin;
    s_max[t] = mine.vmax;
    s_nan[t] = mine.nan;
    s_amax[t] = mine.amax;
    s_aidx[t] = mine.aidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = block_dim / 2; width > 0; width >>= 1) {
        if (t < width) {
            sr_env other;
            other.vmin = s_min[t + width];
            other.vmax = s_max[t + width];
            other.nan = s_nan[t + width];
            other.amax = s_amax[t + width];
            other.aidx = s_aidx[t + width];
            sr_env self;
            self.vmin = s_min[t];
            self.vmax = s_max[t];
            self.nan = s_nan[t];
            self.amax = s_amax[t];
            self.aidx = s_aidx[t];
            sr_env_merge(self, other);
            s_min[t] = self.vmin;
            s_max[t] = self.vmax;
            s_nan[t] = self.nan;
            s_amax[t] = self.amax;
            s_aidx[t] = self.aidx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        sr_env total;
        total.vmin = s_min[0];
        total.vmax = s_max[0];
        total.nan = s_nan[0];
        total.amax = s_amax[0];
        total.aidx = s_aidx[0];
        sr_env_store(out_row, total);
    }
}

kernel void hexv841_solve_region_envelope_finish_v841(
    constant Hexv841SolveRegionEnvelopeFinishV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841SolveRegionEnvelopeFinishV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint bx [[threadgroup_position_in_grid]],
    uint tgs [[threads_per_threadgroup]]) {
    const int n_fields = prm__.n_fields;
    const int blocks_per_field = prm__.blocks_per_field;
    device const long *partials = rsc__.partials;
    device long *out = rsc__.out;

    threadgroup float s_min[256];
    threadgroup float s_max[256];
    threadgroup int s_nan[256];
    threadgroup float s_amax[256];
    threadgroup long s_aidx[256];
    threadgroup long s_row[SR_PART];
    const int f = int(bx);
    if (f >= n_fields) return;
    sr_env mine;
    sr_env_init(mine);
    for (int b = int(t); b < blocks_per_field; b += int(tgs)) {
        sr_env other;
        sr_env_load(other, partials + ((long)f * blocks_per_field + b) * SR_PART);
        sr_env_merge(mine, other);
    }
    sr_env_block_reduce(mine, s_row, t, tgs, s_min, s_max, s_nan, s_amax, s_aidx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        sr_env total;
        sr_env_load(total, s_row);
        if (total.nan) {
            /* cupy.min and cupy.max propagate a NaN; so does this. */
            out[f * SR_OUT + 0] = (long)0x7fc00000;
            out[f * SR_OUT + 1] = (long)0x7fc00000;
        } else {
            out[f * SR_OUT + 0] = long(as_type<int>(total.vmin));
            out[f * SR_OUT + 1] = long(as_type<int>(total.vmax));
        }
        out[f * SR_OUT + 2] = (long)total.nan;
        out[f * SR_OUT + 3] = total.aidx;
    }
}
