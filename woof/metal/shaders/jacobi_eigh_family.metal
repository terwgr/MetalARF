#include <metal_stdlib>
#include "jacobi_eigh_batched_bindings.metalh"

using namespace metal;

// One matrix per threadgroup. v_out temporarily stores the accumulated real
// eigenvectors, leaving only the padded working matrix in threadgroup memory.
// Separate primitive FP32 statements keep CUDA's operation ordering visible.
#pragma clang fp contract(off)

static constexpr float JACOBI_EPS = 1.1920928955078125e-07f;
static constexpr uint JACOBI_MAX_M = 64;

static inline int2 jacobi_pair(int m, int sweep_round, int pair_index) {
    const int a = pair_index == 0
        ? 0 : ((pair_index - 1 + sweep_round) % (m - 1) + 1);
    const int b = (m - 2 - pair_index + sweep_round) % (m - 1) + 1;
    return a < b ? int2(a, b) : int2(b, a);
}

kernel void jacobi_eigh_batched(
    constant JacobiEighBatchedResources &r [[buffer(0)]],
    constant JacobiEighBatchedParams &p [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float work[JACOBI_MAX_M * JACOBI_MAX_M];
    threadgroup float2 rotations[JACOBI_MAX_M / 2];
    threadgroup int source_index[JACOBI_MAX_M];
    threadgroup atomic_uint flag;

    const uint mat = group.x;
    if (mat >= uint(p.n_matrices)) return;
    const uint k = uint(p.k);
    const uint m = uint(p.m);
    const uint tpb = uint(p.tpb);
    const ulong input_base = ulong(mat) * ulong(k) * ulong(k);
    const ulong output_base = ulong(mat) * ulong(k) * ulong(k);
    const ulong values_base = ulong(mat) * ulong(k);

    if (tid == 0) atomic_store_explicit(&flag, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < m * m; idx += tpb) {
        const uint row = idx / m;
        const uint col = idx - row * m;
        float value;
        if (row < k && col < k) {
            value = r.a_in[input_base + ulong(row) * k + col];
            if (!isfinite(value))
                atomic_store_explicit(&flag, 1u, memory_order_relaxed);
        } else {
            value = row == col ? 1.0f : 0.0f;
        }
        work[idx] = value;
        if (row < k && col < k) {
            r.v_out[output_base + ulong(row) * k + col] = row == col ? 1.0f : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&flag, memory_order_relaxed) != 0u) {
        const float qnan = as_type<float>(0x7fc00000u);
        for (uint idx = tid; idx < k; idx += tpb)
            r.w_out[values_base + idx] = qnan;
        for (uint idx = tid; idx < k * k; idx += tpb)
            r.v_out[output_base + idx] = qnan;
        if (tid == 0) r.status_out[mat] = -2.0f;
        return;
    }

    uint sweeps = 0;
    bool converged = false;
    for (uint sweep = 0; sweep < uint(p.sweep_cap); ++sweep) {
        if (tid == 0) atomic_store_explicit(&flag, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint sweep_round = 0; sweep_round < m - 1; ++sweep_round) {
            for (uint i = tid; i < m / 2; i += tpb) {
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                const uint pi = uint(pair.x), qi = uint(pair.y);
                const float app = work[pi * m + pi];
                const float aqq = work[qi * m + qi];
                const float apq = work[pi * m + qi];
                float cs = 1.0f;
                float sn = 0.0f;
                const float threshold = JACOBI_EPS * sqrt(fabs(app) * fabs(aqq));
                if (fabs(apq) > threshold) {
                    const float tau = (aqq - app) / (apq + apq);
                    const float root = sqrt(1.0f + tau * tau);
                    const float t = tau >= 0.0f
                        ? 1.0f / (tau + root)
                        : -1.0f / (root - tau);
                    cs = 1.0f / sqrt(1.0f + t * t);
                    sn = t * cs;
                    atomic_store_explicit(&flag, 1u, memory_order_relaxed);
                }
                rotations[i] = float2(cs, sn);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Left rotation A <- J^T A; every pair owns two distinct rows.
            for (uint idx = tid; idx < (m / 2) * m; idx += tpb) {
                const uint i = idx / m;
                const uint j = idx - i * m;
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                const float cs = rotations[i].x, sn = rotations[i].y;
                const float ap = work[uint(pair.x) * m + j];
                const float aq = work[uint(pair.y) * m + j];
                work[uint(pair.x) * m + j] = cs * ap - sn * aq;
                work[uint(pair.y) * m + j] = sn * ap + cs * aq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Right rotation A <- A J, and the same rotation on V. Padding
            // partners are exact identity rotations and have no V column.
            for (uint idx = tid; idx < (m / 2) * m; idx += tpb) {
                const uint i = idx / m;
                const uint row = idx - i * m;
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                const uint pi = uint(pair.x), qi = uint(pair.y);
                const float cs = rotations[i].x, sn = rotations[i].y;
                const float ap = work[row * m + pi];
                const float aq = work[row * m + qi];
                work[row * m + pi] = cs * ap - sn * aq;
                work[row * m + qi] = sn * ap + cs * aq;
                if (row < k && pi < k && qi < k) {
                    const ulong row_base = output_base + ulong(row) * k;
                    const float vp = r.v_out[row_base + pi];
                    const float vq = r.v_out[row_base + qi];
                    r.v_out[row_base + pi] = cs * vp - sn * vq;
                    r.v_out[row_base + qi] = sn * vp + cs * vq;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Store the exact zero targeted by the Jacobi rotation.
            for (uint i = tid; i < m / 2; i += tpb) {
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                work[uint(pair.x) * m + uint(pair.y)] = 0.0f;
                work[uint(pair.y) * m + uint(pair.x)] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Symmetrise the two triangles after each cyclic sweep, matching the
        // CUDA source's explicit FP32 add followed by multiply by one half.
        for (uint idx = tid; idx < m * m; idx += tpb) {
            const uint row = idx / m, col = idx - row * m;
            if (row < col) {
                const float mean = (work[row * m + col] + work[col * m + row]) * 0.5f;
                work[row * m + col] = mean;
                work[col * m + row] = mean;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sweeps = sweep + 1;
        if (atomic_load_explicit(&flag, memory_order_relaxed) == 0u) {
            converged = true;
            break;
        }
    }

    // Sort eigenvalues in ascending order using the same odd-even network.
    // Swapping V columns alongside their eigenvalues avoids an extra k*k
    // scratch matrix; source_index preserves the source-column tie break.
    for (uint i = tid; i < k; i += tpb) {
        r.w_out[values_base + i] = work[i * m + i];
        source_index[i] = int(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint phase = 0; phase < k; ++phase) {
        const uint parity = phase & 1u;
        for (uint i = tid; i < (k + 1) / 2; i += tpb) {
            const uint left = 2 * i + parity;
            const uint right = left + 1;
            if (right < k) {
                const int src_left = source_index[left];
                const int src_right = source_index[right];
                const float eval_left = r.w_out[values_base + left];
                const float eval_right = r.w_out[values_base + right];
                if (eval_right < eval_left
                        || (eval_right == eval_left && src_right < src_left)) {
                    r.w_out[values_base + left] = eval_right;
                    r.w_out[values_base + right] = eval_left;
                    source_index[left] = src_right;
                    source_index[right] = src_left;
                    for (uint row = 0; row < k; ++row) {
                        const ulong row_base = output_base + ulong(row) * k;
                        const float vl = r.v_out[row_base + left];
                        r.v_out[row_base + left] = r.v_out[row_base + right];
                        r.v_out[row_base + right] = vl;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Canonical sign: the largest-magnitude component is positive; strict
    // greater-than retains the first (lowest row) component on magnitude ties.
    for (uint col = tid; col < k; col += tpb) {
        uint best = 0;
        float best_magnitude = fabs(r.v_out[output_base + col]);
        for (uint row = 1; row < k; ++row) {
            const float magnitude = fabs(
                r.v_out[output_base + ulong(row) * k + col]);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                best = row;
            }
        }
        if (r.v_out[output_base + ulong(best) * k + col] < 0.0f) {
            for (uint row = 0; row < k; ++row) {
                const ulong index = output_base + ulong(row) * k + col;
                r.v_out[index] = -r.v_out[index];
            }
        }
    }
    if (tid == 0)
        r.status_out[mat] = converged ? float(sweeps) : -1.0f;
}
