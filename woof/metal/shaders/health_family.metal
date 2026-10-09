#include <metal_stdlib>
#include "health_partial_bindings.metalh"
#include "health_final_bindings.metalh"
#include "health_partial_tile_bindings.metalh"


#include "health_grouped_scan_bindings.metalh"
#include "health_grouped_final_bindings.metalh"

using namespace metal;

// Match the CUDA source's float operation boundaries in the CFL calculation.
#pragma clang fp contract(off)

static inline void health_update(float value, ulong index,
                                 thread float &maximum,
                                 thread ulong &max_index) {
    if (value > maximum || (value == maximum && index < max_index)) {
        maximum = value;
        max_index = index;
    }
}

static inline void health_update(float value, ulong index,
                                 threadgroup float &maximum,
                                 threadgroup ulong &max_index) {
    if (value > maximum || (value == maximum && index < max_index)) {
        maximum = value;
        max_index = index;
    }
}

static inline void health_group_reduce(
        threadgroup float *values, threadgroup ulong *indices,
        threadgroup uint *masks, uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[lane] = max(values[lane], values[lane + offset]);
            health_update(values[256 + lane + offset],
                          indices[lane + offset],
                          values[256 + lane], indices[lane]);
            values[512 + lane] = max(values[512 + lane],
                                     values[512 + lane + offset]);
            values[768 + lane] = max(values[768 + lane],
                                     values[768 + lane + offset]);
            values[1024 + lane] = max(values[1024 + lane],
                                      values[1024 + lane + offset]);
            values[1280 + lane] = max(values[1280 + lane],
                                      values[1280 + lane + offset]);
            masks[lane] |= masks[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void health_partial(
        constant HealthPartialResources &r [[buffer(0)]],
        constant HealthPartialParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]],
        uint lane [[thread_position_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    if (group >= uint(q.nblocks)) return;
    const ulong stride = ulong(q.nblocks) * 256ul;
    float umax = 0.0f, wmax = 0.0f, thmax = 0.0f;
    float edge = 0.0f, interior = 0.0f, vertical_rate = 0.0f;
    ulong windex = 0xfffffffffffffffful;
    uint mask = 0u;
    for (ulong ix = gid; ix < ulong(q.nu); ix += stride) {
        const float value = fabs(r.u[ix]);
        if (isnan(value)) mask |= 1u;
        else umax = max(value, umax);
    }
    const ulong plane = ulong(q.ny) * ulong(q.nx);
    for (ulong ix = gid; ix < ulong(q.nw); ix += stride) {
        const float value = fabs(r.w[ix]);
        bool boundary = false;
        if (q.width > 0) {
            const ulong c = ix % plane;
            const int j = int(c / ulong(q.nx));
            const int i = int(c - ulong(j) * ulong(q.nx));
            boundary = j < q.width || j >= q.ny - q.width
                    || i < q.width || i >= q.nx - q.width;
        }
        if (isnan(value)) {
            mask |= 2u;
            if (q.width > 0) mask |= boundary ? 8u : 16u;
        } else {
            health_update(value, ix, wmax, windex);
            if (q.width > 0) {
                if (boundary) edge = max(edge, value);
                else interior = max(interior, value);
            }
        }
    }
    for (ulong ix = gid; ix < ulong(q.nth); ix += stride) {
        const float value = fabs(r.thp[ix]);
        if (isnan(value)) mask |= 4u;
        else thmax = max(value, thmax);
    }
    for (ulong ix = gid; ix < ulong(q.ncells); ix += stride) {
        const ulong k = ix / plane;
        const ulong upper = ix + plane;
        const float base_lower = q.phb_full ? r.phb[ix] : r.phb[k];
        const float base_upper = q.phb_full ? r.phb[upper] : r.phb[k + 1];
        const float dz = ((r.ph[upper] + base_upper)
                         - (r.ph[ix] + base_lower)) / q.gravity;
        const float speed = fabs(r.w[upper]);
        if (!isfinite(dz) || dz <= 0.0f || !isfinite(speed)) mask |= 32u;
        else vertical_rate = max(vertical_rate, speed / dz);
    }

    threadgroup float values[6 * 256];
    threadgroup ulong indices[256];
    threadgroup uint masks[256];
    values[lane] = umax;
    values[256 + lane] = wmax;
    values[512 + lane] = thmax;
    values[768 + lane] = edge;
    values[1024 + lane] = interior;
    values[1280 + lane] = vertical_rate;
    indices[lane] = windex;
    masks[lane] = mask;
    health_group_reduce(values, indices, masks, lane);
    if (lane == 0) {
        const ulong out = ulong(group) * 9ul;
        for (uint field = 0; field < 6; ++field)
            r.partial[out + field] = values[ulong(field) * 256ul];
        r.partial[out + 6] = as_type<float>(uint(indices[0]));
        r.partial[out + 7] = as_type<float>(uint(indices[0] >> 32));
        r.partial[out + 8] = as_type<float>(masks[0]);
    }
}

kernel void health_final(
        constant HealthFinalResources &r [[buffer(0)]],
        constant HealthFinalParams &q [[buffer(1)]],
        uint lane [[thread_position_in_threadgroup]]) {
    float maxima[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ulong windex = 0xfffffffffffffffful;
    uint mask = 0u;
    for (uint block = lane; block < uint(q.nblocks); block += 256) {
        const ulong src = ulong(block) * 9ul;
        maxima[0] = max(maxima[0], r.partial[src]);
        const ulong index = ulong(as_type<uint>(r.partial[src + 6]))
                          | (ulong(as_type<uint>(r.partial[src + 7])) << 32);
        health_update(r.partial[src + 1], index, maxima[1], windex);
        for (uint field = 2; field < 6; ++field)
            maxima[field] = max(maxima[field], r.partial[src + field]);
        mask |= as_type<uint>(r.partial[src + 8]);
    }
    threadgroup float values[6 * 256];
    threadgroup ulong indices[256];
    threadgroup uint masks[256];
    for (uint field = 0; field < 6; ++field)
        values[ulong(field) * 256ul + lane] = maxima[field];
    indices[lane] = windex;
    masks[lane] = mask;
    health_group_reduce(values, indices, masks, lane);
    if (lane == 0) {
        for (uint field = 0; field < 6; ++field) {
            const uint bit = 1u << field;
            r.result[field] = (masks[0] & bit)
                ? as_type<float>(0x7fc00000u)
                : values[ulong(field) * 256ul];
        }
        r.result[6] = as_type<float>(uint(indices[0]));
        r.result[7] = as_type<float>(uint(indices[0] >> 32));
    }
}

kernel void health_partial_tile(
        constant HealthPartialTileResources &r [[buffer(0)]],
        constant HealthPartialTileParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]],
        uint lane [[thread_position_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    if (group >= uint(q.nblocks)) return;
    const ulong stride = ulong(q.nblocks) * 256ul;
    const ulong bplane = ulong(q.bny) * ulong(q.bnx);
    const ulong uplane = ulong(q.bny) * ulong(q.bnx + 1);
    const ulong area = ulong(q.iny) * ulong(q.inx);
    const ulong ucols = ulong(q.inx + q.u_extra);
    const ulong uarea = ulong(q.iny) * ucols;
    float umax = 0.0f, wmax = 0.0f, thmax = 0.0f;
    float edge = 0.0f, interior = 0.0f, vertical_rate = 0.0f;
    ulong windex = 0xfffffffffffffffful;
    uint mask = 0u;

    for (ulong t = gid; t < ulong(q.nz) * uarea; t += stride) {
        const ulong k = t / uarea, rem = t - k * uarea;
        const ulong jj = rem / ucols, ii = rem - jj * ucols;
        const float value = fabs(r.u[k * uplane + ulong(q.jb + int(jj))
                                  * ulong(q.bnx + 1) + ulong(q.ib + int(ii))]);
        if (isnan(value)) mask |= 1u;
        else umax = max(umax, value);
    }
    for (ulong t = gid; t < ulong(q.nz + 1) * area; t += stride) {
        const ulong k = t / area, rem = t - k * area;
        const ulong jj = rem / ulong(q.inx), ii = rem - jj * ulong(q.inx);
        const float value = fabs(r.w[k * bplane + ulong(q.jb + int(jj))
                                  * ulong(q.bnx) + ulong(q.ib + int(ii))]);
        const int j = q.jd + int(jj), i = q.id + int(ii);
        const ulong dindex = (k * ulong(q.dny) + ulong(j)) * ulong(q.dnx)
                           + ulong(i);
        const bool boundary = q.width > 0
            && (j < q.width || j >= q.dny - q.width
                || i < q.width || i >= q.dnx - q.width);
        if (isnan(value)) {
            mask |= 2u;
            if (q.width > 0) mask |= boundary ? 8u : 16u;
        } else {
            health_update(value, dindex, wmax, windex);
            if (q.width > 0) {
                if (boundary) edge = max(edge, value);
                else interior = max(interior, value);
            }
        }
    }
    for (ulong t = gid; t < ulong(q.nz) * area; t += stride) {
        const ulong k = t / area, rem = t - k * area;
        const ulong jj = rem / ulong(q.inx), ii = rem - jj * ulong(q.inx);
        const float value = fabs(r.thp[k * bplane + ulong(q.jb + int(jj))
                                      * ulong(q.bnx) + ulong(q.ib + int(ii))]);
        if (isnan(value)) mask |= 4u;
        else thmax = max(thmax, value);
    }
    if (q.have_geopotential) {
        for (ulong t = gid; t < ulong(q.nz) * area; t += stride) {
            const ulong k = t / area, rem = t - k * area;
            const ulong jj = rem / ulong(q.inx), ii = rem - jj * ulong(q.inx);
            const ulong bix = k * bplane + ulong(q.jb + int(jj)) * ulong(q.bnx)
                            + ulong(q.ib + int(ii));
            const ulong upper = bix + bplane;
            const float base_lower = q.phb_full ? r.phb[bix] : r.phb[k];
            const float base_upper = q.phb_full ? r.phb[upper] : r.phb[k + 1];
            const float dz = ((r.ph[upper] + base_upper)
                             - (r.ph[bix] + base_lower)) / q.gravity;
            const float speed = fabs(r.w[upper]);
            if (!isfinite(dz) || dz <= 0.0f || !isfinite(speed)) mask |= 32u;
            else vertical_rate = max(vertical_rate, speed / dz);
        }
    }
    threadgroup float values[6 * 256];
    threadgroup ulong indices[256];
    threadgroup uint masks[256];
    values[lane] = umax;
    values[256 + lane] = wmax;
    values[512 + lane] = thmax;
    values[768 + lane] = edge;
    values[1024 + lane] = interior;
    values[1280 + lane] = vertical_rate;
    indices[lane] = windex;
    masks[lane] = mask;
    health_group_reduce(values, indices, masks, lane);
    if (lane == 0) {
        const ulong out = ulong(q.block_offset + int(group)) * 9ul;
        for (uint field = 0; field < 6; ++field)
            r.partial[out + field] = values[ulong(field) * 256ul];
        r.partial[out + 6] = as_type<float>(uint(indices[0]));
        r.partial[out + 7] = as_type<float>(uint(indices[0] >> 32));
        r.partial[out + 8] = as_type<float>(masks[0]);
    }
}


// Descriptor-free Metal equivalent of validate_full_state. The 30 cases are
// selected uniformly per block after the flattened group prefix lookup.
static inline device const uint *health_value_resource(
        constant HealthGroupedScanResources &r, uint slot) {
    switch (slot) {
        case 0: return r.v0;
        case 1: return r.v1;
        case 2: return r.v2;
        case 3: return r.v3;
        case 4: return r.v4;
        case 5: return r.v5;
        case 6: return r.v6;
        case 7: return r.v7;
        case 8: return r.v8;
        case 9: return r.v9;
        case 10: return r.v10;
        case 11: return r.v11;
        case 12: return r.v12;
        case 13: return r.v13;
        case 14: return r.v14;
        case 15: return r.v15;
        case 16: return r.v16;
        case 17: return r.v17;
        case 18: return r.v18;
        case 19: return r.v19;
        case 20: return r.v20;
        case 21: return r.v21;
        case 22: return r.v22;
        case 23: return r.v23;
        case 24: return r.v24;
        case 25: return r.v25;
        case 26: return r.v26;
        case 27: return r.v27;
        case 28: return r.v28;
        case 29: return r.v29;
        default: return r.v0;
    }
}
static inline device const uint *health_aux_resource(
        constant HealthGroupedScanResources &r, uint slot) {
    switch (slot) {
        case 0: return r.a0;
        case 1: return r.a1;
        case 2: return r.a2;
        case 3: return r.a3;
        case 4: return r.a4;
        case 5: return r.a5;
        case 6: return r.a6;
        case 7: return r.a7;
        case 8: return r.a8;
        case 9: return r.a9;
        case 10: return r.a10;
        case 11: return r.a11;
        case 12: return r.a12;
        case 13: return r.a13;
        case 14: return r.a14;
        case 15: return r.a15;
        case 16: return r.a16;
        case 17: return r.a17;
        case 18: return r.a18;
        case 19: return r.a19;
        case 20: return r.a20;
        case 21: return r.a21;
        case 22: return r.a22;
        case 23: return r.a23;
        case 24: return r.a24;
        case 25: return r.a25;
        case 26: return r.a26;
        case 27: return r.a27;
        case 28: return r.a28;
        case 29: return r.a29;
        default: return r.a0;
    }
}

kernel void health_grouped_scan(
        constant HealthGroupedScanResources &r [[buffer(0)]],
        constant HealthGroupedScanParams &q [[buffer(1)]],
        uint gid [[threadgroup_position_in_grid]],
        uint lane [[thread_position_in_threadgroup]]) {
    if (gid >= uint(q.groups)) return;
    const ulong fields = r.meta64[0];
    const ulong field_base = r.meta64[2];
    const ulong chunk_size = r.meta64[3];
    const ulong partial_base = r.meta64[4];
    // Find the field whose half-open prefix span owns this flattened group.
    uint lo = 0u, hi = uint(fields);
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        const ulong prefix = r.meta64[5ul + ulong(mid) * 5ul + 3ul];
        const ulong count = r.meta64[5ul + ulong(mid) * 5ul + 4ul];
        if (ulong(gid) < prefix + count) hi = mid;
        else lo = mid + 1u;
    }
    const uint field = lo;
    const ulong meta = 5ul + ulong(field) * 5ul;
    const ulong size = r.meta64[meta];
    const ulong aux_extent = r.meta64[meta + 1ul];
    const ulong plane = r.meta64[meta + 2ul];
    const ulong prefix = r.meta64[meta + 3ul];
    const ulong local_chunk = ulong(gid) - prefix;
    const ulong begin = local_chunk * chunk_size;
    const ulong end = min(size, begin + chunk_size);
    const uint m = field * 4u;
    const uint flags = r.meta32[m];
    const uint status_bit = r.meta32[m + 1u];
    const float lower = as_type<float>(r.meta32[m + 2u]);
    const float upper = as_type<float>(r.meta32[m + 3u]);
    device const uint *values = health_value_resource(r, field);
    device const uint *auxiliary = health_aux_resource(r, field);
    uint status = 0u;
    ulong first = 0xfffffffffffffffful;
    for (ulong ix = begin + ulong(lane); ix < end; ix += 256ul) {
        const uint word = values[ix];
        float value = (flags & 32u) != 0u
                    ? float(as_type<int>(word)) : as_type<float>(word);
        if ((flags & 8u) != 0u) {
            const ulong aux_index = (flags & 16u) != 0u ? ix / plane : ix;
            value = value + as_type<float>(auxiliary[aux_index]);
        }
        bool bad = !isfinite(value);
        if ((flags & 1u) != 0u)
            bad = bad || (((flags & 4u) != 0u)
                          ? value <= lower : value < lower);
        if ((flags & 2u) != 0u) bad = bad || value > upper;
        if (bad) { status |= status_bit; first = min(first, ix); }
    }
    threadgroup uint statuses[256];
    threadgroup ulong indices[256];
    statuses[lane] = status;
    indices[lane] = first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            statuses[lane] |= statuses[lane + offset];
            indices[lane] = min(indices[lane], indices[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        const ulong out = (partial_base + ulong(gid)) * 2ul;
        const ulong packed = indices[0] == 0xfffffffffffffffful
            ? 0xfffffffffffffffful
            : ((field_base + ulong(field)) << 48) | indices[0];
        r.partial[out] = ulong(statuses[0]);
        r.partial[out + 1ul] = packed;
    }
    (void)aux_extent;
}

kernel void health_grouped_final(
        constant HealthGroupedFinalResources &r [[buffer(0)]],
        constant HealthGroupedFinalParams &q [[buffer(1)]],
        uint lane [[thread_position_in_threadgroup]]) {
    uint status = 0u;
    ulong first = 0xfffffffffffffffful;
    for (uint record = lane; record < uint(q.records); record += 256u) {
        const ulong offset = ulong(record) * 2ul;
        status |= uint(r.partial[offset]);
        first = min(first, r.partial[offset + 1ul]);
    }
    threadgroup uint statuses[256];
    threadgroup ulong indices[256];
    statuses[lane] = status;
    indices[lane] = first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            statuses[lane] |= statuses[lane + offset];
            indices[lane] = min(indices[lane], indices[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { r.result[0] = ulong(statuses[0]); r.result[1] = indices[0]; }
}
