#include <metal_stdlib>
#include "jacobi_eigh_fp64_batched_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;

// Software binary64 Jacobi solver. All numerical values are raw IEEE-754
// words and every primitive operation is explicitly rounded by soft64_*.
// This avoids relying on native FP64 support in Metal. CUDA's contraction
// behavior for the source expressions has not been established, so this
// kernel does not claim cross-backend bitwise identity.

constant uint JACOBI_MAX_M = 64u;
constant ulong JACOBI_F64_EPS = 0x3cb0000000000000ul;
constant ulong JACOBI_ONE = 0x3ff0000000000000ul;
constant ulong JACOBI_HALF = 0x3fe0000000000000ul;
constant ulong JACOBI_ZERO = 0x0000000000000000ul;
constant ulong JACOBI_NEG_ONE = 0xbff0000000000000ul;
constant ulong JACOBI_QNAN = 0x7ff8000000000000ul;

static inline int2 jacobi_pair(int m, int sweep_round, int pair_index) {
    const int a = pair_index == 0
        ? 0 : ((pair_index - 1 + sweep_round) % (m - 1) + 1);
    const int b = (m - 2 - pair_index + sweep_round) % (m - 1) + 1;
    return a < b ? int2(a, b) : int2(b, a);
}

static inline ulong jacobi_abs(ulong value) {
    return soft64_abs_bits(value);
}

kernel void jacobi_eigh_fp64_batched(
    constant JacobiEighFp64BatchedResources &r [[buffer(0)]],
    constant JacobiEighFp64BatchedParams &p [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    // The caller allocates exactly (batch,m,m) words. Device scratch allows
    // k=64 without exceeding the per-threadgroup memory ceiling.
    threadgroup ulong rotations[JACOBI_MAX_M / 2][2];
    threadgroup int source_index[JACOBI_MAX_M];
    threadgroup atomic_uint flag;

    const uint mat = group.x;
    const uint k = uint(p.k);
    const uint m = uint(p.m);
    const uint tpb = uint(p.tpb);
    const ulong input_base = ulong(mat) * ulong(k) * ulong(k);
    const ulong scratch_base = ulong(mat) * ulong(m) * ulong(m);
    const ulong output_base = ulong(mat) * ulong(k) * ulong(k);
    const ulong values_base = ulong(mat) * ulong(k);

    if (tid == 0u) atomic_store_explicit(&flag, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < m * m; idx += tpb) {
        const uint row = idx / m;
        const uint col = idx - row * m;
        ulong value;
        if (row < k && col < k) {
            value = r.a_in[input_base + ulong(row) * k + col];
            if (!soft64_is_finite(value))
                atomic_store_explicit(&flag, 1u, memory_order_relaxed);
        } else {
            value = row == col ? JACOBI_ONE : JACOBI_ZERO;
        }
        r.work[scratch_base + idx] = value;
        if (row < k && col < k)
            r.v_out[output_base + ulong(row) * k + col] = row == col
                ? JACOBI_ONE : JACOBI_ZERO;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&flag, memory_order_relaxed) != 0u) {
        for (uint idx = tid; idx < k; idx += tpb)
            r.w_out[values_base + idx] = JACOBI_QNAN;
        for (uint idx = tid; idx < k * k; idx += tpb)
            r.v_out[output_base + idx] = JACOBI_QNAN;
        if (tid == 0u) r.status_out[mat] = -2;
        return;
    }

    uint sweeps = 0u;
    bool converged = false;
    for (uint sweep = 0u; sweep < uint(p.sweep_cap); ++sweep) {
        if (tid == 0u) atomic_store_explicit(&flag, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        for (uint sweep_round = 0u; sweep_round < m - 1u; ++sweep_round) {
            // Rotation coefficients are produced in threadgroup memory from
            // device scratch, then consumed only after a two-space fence.
            for (uint i = tid; i < m / 2u; i += tpb) {
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                const uint pi = uint(pair.x), qi = uint(pair.y);
                const ulong app = r.work[scratch_base + pi * m + pi];
                const ulong aqq = r.work[scratch_base + qi * m + qi];
                const ulong apq = r.work[scratch_base + pi * m + qi];
                ulong cs = JACOBI_ONE;
                ulong sn = JACOBI_ZERO;
                const ulong threshold = soft64_mul(JACOBI_F64_EPS,
                    soft64_sqrt(soft64_mul(jacobi_abs(app), jacobi_abs(aqq))));
                if (soft64_greater(jacobi_abs(apq), threshold)) {
                    const ulong tau = soft64_div(soft64_sub(aqq, app),
                                                  soft64_add(apq, apq));
                    const ulong root = soft64_sqrt(soft64_add(JACOBI_ONE,
                                                        soft64_mul(tau, tau)));
                    const ulong t = soft64_greater_equal(tau, JACOBI_ZERO)
                        ? soft64_div(JACOBI_ONE, soft64_add(tau, root))
                        : soft64_div(JACOBI_NEG_ONE, soft64_sub(root, tau));
                    cs = soft64_div(JACOBI_ONE,
                                    soft64_sqrt(soft64_add(JACOBI_ONE,
                                                        soft64_mul(t, t))));
                    sn = soft64_mul(t, cs);
                    atomic_store_explicit(&flag, 1u, memory_order_relaxed);
                }
                rotations[i][0] = cs;
                rotations[i][1] = sn;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // Left congruence pass A <- J^T A. Pair-owned rows are disjoint.
            for (uint idx = tid; idx < (m / 2u) * m; idx += tpb) {
                const uint i = idx / m;
                const uint j = idx - i * m;
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                const ulong cs = rotations[i][0], sn = rotations[i][1];
                const ulong ap = r.work[scratch_base + uint(pair.x) * m + j];
                const ulong aq = r.work[scratch_base + uint(pair.y) * m + j];
                r.work[scratch_base + uint(pair.x) * m + j] = soft64_sub(
                    soft64_mul(cs, ap), soft64_mul(sn, aq));
                r.work[scratch_base + uint(pair.y) * m + j] = soft64_add(
                    soft64_mul(sn, ap), soft64_mul(cs, aq));
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // Right pass A <- A J and accumulated eigenvectors V <- V J.
            for (uint idx = tid; idx < (m / 2u) * m; idx += tpb) {
                const uint i = idx / m;
                const uint row = idx - i * m;
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                const uint pi = uint(pair.x), qi = uint(pair.y);
                const ulong cs = rotations[i][0], sn = rotations[i][1];
                const ulong ap = r.work[scratch_base + row * m + pi];
                const ulong aq = r.work[scratch_base + row * m + qi];
                r.work[scratch_base + row * m + pi] = soft64_sub(
                    soft64_mul(cs, ap), soft64_mul(sn, aq));
                r.work[scratch_base + row * m + qi] = soft64_add(
                    soft64_mul(sn, ap), soft64_mul(cs, aq));
                if (row < k && pi < k && qi < k) {
                    const ulong row_base = output_base + ulong(row) * k;
                    const ulong vp = r.v_out[row_base + pi];
                    const ulong vq = r.v_out[row_base + qi];
                    r.v_out[row_base + pi] = soft64_sub(
                        soft64_mul(cs, vp), soft64_mul(sn, vq));
                    r.v_out[row_base + qi] = soft64_add(
                        soft64_mul(sn, vp), soft64_mul(cs, vq));
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // Force exact annihilation of the selected symmetric pair.
            for (uint i = tid; i < m / 2u; i += tpb) {
                const int2 pair = jacobi_pair(int(m), int(sweep_round), int(i));
                r.work[scratch_base + uint(pair.x) * m + uint(pair.y)] = JACOBI_ZERO;
                r.work[scratch_base + uint(pair.y) * m + uint(pair.x)] = JACOBI_ZERO;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }

        // Average the two triangles once per complete cyclic sweep.
        for (uint idx = tid; idx < m * m; idx += tpb) {
            const uint row = idx / m, col = idx - row * m;
            if (row < col) {
                const ulong mean = soft64_mul(soft64_add(
                    r.work[scratch_base + row * m + col],
                    r.work[scratch_base + col * m + row]), JACOBI_HALF);
                r.work[scratch_base + row * m + col] = mean;
                r.work[scratch_base + col * m + row] = mean;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweeps = sweep + 1u;
        if (atomic_load_explicit(&flag, memory_order_relaxed) == 0u) {
            converged = true;
            break;
        }
    }

    // Ascending odd-even sorting network, with source-column tie breaking.
    for (uint i = tid; i < k; i += tpb) {
        r.w_out[values_base + i] = r.work[scratch_base + i * m + i];
        source_index[i] = int(i);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint phase = 0u; phase < k; ++phase) {
        const uint parity = phase & 1u;
        for (uint i = tid; i < (k + 1u) / 2u; i += tpb) {
            const uint left = 2u * i + parity;
            const uint right = left + 1u;
            if (right < k) {
                const int src_left = source_index[left];
                const int src_right = source_index[right];
                const ulong eval_left = r.w_out[values_base + left];
                const ulong eval_right = r.w_out[values_base + right];
                if (soft64_less(eval_right, eval_left)
                        || (soft64_equal(eval_right, eval_left)
                            && src_right < src_left)) {
                    r.w_out[values_base + left] = eval_right;
                    r.w_out[values_base + right] = eval_left;
                    source_index[left] = src_right;
                    source_index[right] = src_left;
                    for (uint row = 0u; row < k; ++row) {
                        const ulong row_base = output_base + ulong(row) * k;
                        const ulong vl = r.v_out[row_base + left];
                        r.v_out[row_base + left] = r.v_out[row_base + right];
                        r.v_out[row_base + right] = vl;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    // Canonical signs: largest-magnitude component positive; strict greater
    // retains the lowest row for magnitude ties.
    for (uint col = tid; col < k; col += tpb) {
        uint best = 0u;
        ulong best_magnitude = jacobi_abs(r.v_out[output_base + col]);
        for (uint row = 1u; row < k; ++row) {
            const ulong magnitude = jacobi_abs(
                r.v_out[output_base + ulong(row) * k + col]);
            if (soft64_greater(magnitude, best_magnitude)) {
                best_magnitude = magnitude;
                best = row;
            }
        }
        if (soft64_less(r.v_out[output_base + ulong(best) * k + col],
                        JACOBI_ZERO)) {
            for (uint row = 0u; row < k; ++row) {
                const ulong index = output_base + ulong(row) * k + col;
                r.v_out[index] ^= 0x8000000000000000ul;
            }
        }
    }
    if (tid == 0u) r.status_out[mat] = converged ? int(sweeps) : -1;
}
