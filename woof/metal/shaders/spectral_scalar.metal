#include <metal_stdlib>
#include "spectral_reflect_extend_bindings.metalh"
#include "spectral_transfer_bindings.metalh"
#include "spectral_pair_stats_bindings.metalh"
#include "spectral_recenter_crop_bindings.metalh"
#include "spectral_final_stats_bindings.metalh"
#include "spectral_stats_reduce_bindings.metalh"
#include "spectral_plane_correction_bindings.metalh"
using namespace metal;

kernel void spectral_reflect_extend(constant SpectralReflectExtendResources& r [[buffer(0)]], constant SpectralReflectExtendParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint total = p.batches * p.ey * p.ex;
    if (tid >= total) return;
    uint x = tid % p.ex, y = (tid / p.ex) % p.ey, b = tid / (p.ex * p.ey);
    uint sy = y < uint(p.ny) ? y : uint(p.ey) - 1 - y;
    uint sx = x < uint(p.nx) ? x : uint(p.ex) - 1 - x;
    r.output[tid] = r.input[(b * p.ny + sy) * p.nx + sx];
}

kernel void spectral_transfer(constant SpectralTransferResources& r [[buffer(0)]], constant SpectralTransferParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint width = p.ex / 2 + 1, plane = p.ey * width;
    if (tid >= p.batches * plane * 2) return;
    uint bin = tid / 2;
    uint x = bin % width, y = (bin / width) % p.ey;
    int yfreq = y <= uint(p.ey) / 2 ? int(y) : int(y) - p.ey;
    float ky = 6.2831853071795864769f * float(yfreq) / (float(p.ey) * p.dy);
    float kx = 6.2831853071795864769f * float(x) / (float(p.ex) * p.dx);
    float magnitude = sqrt(ky * ky + kx * kx);
    float kref = 6.2831853071795864769f / p.reference_wavelength;
    float ratio = magnitude / kref, activation = 1.0f;
    if (p.protect_wavelength > 0.0f) {
        float kp = 6.2831853071795864769f / p.protect_wavelength;
        float t = clamp((magnitude - kp) / (kref - kp), 0.0f, 1.0f);
        activation = 0.5f - 0.5f * cos(3.14159265358979323846f * t);
        if (magnitude <= kp) activation = 0.0f;
        if (magnitude >= kref) activation = 1.0f;
    }
    float power = 1.0f;
    for (int i = 0; i < 2 * p.order; ++i) power *= ratio;
    float exponent = max(-(p.dt / p.efold) * activation * power, -80.0f);
    float transfer = max(exp(exponent), 1.0f - p.max_damping);
    if (magnitude == 0.0f) transfer = 1.0f;
    r.coefficients[tid] *= transfer;
}

kernel void spectral_pair_stats(constant SpectralPairStatsResources& r [[buffer(0)]], constant SpectralPairStatsParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint total = uint(p.batches) * uint(p.partial_count);
    if (tid >= total) return;
    uint batch = tid / uint(p.partial_count), block = tid % uint(p.partial_count);
    uint start = block * 256, end = min(start + 256, uint(p.plane_count));
    float a = 0.0f, b = 0.0f;
    uint finite = 1;
    for (uint i = start; i < end; ++i) {
        uint original_index = batch * uint(p.plane_count) + i;
        uint y = i / uint(p.nx), x = i % uint(p.nx);
        uint filtered_index = (batch * uint(p.ey) + y) * uint(p.ex) + x;
        float av = r.original[original_index], bv = r.filtered[filtered_index];
        a += av; b += bv;
        finite &= isfinite(av) && isfinite(bv);
    }
    uint output = tid * 3;
    r.stats[output] = a; r.stats[output + 1] = b;
    r.stats[output + 2] = finite ? 1.0f : 0.0f;
}

kernel void spectral_recenter_crop(constant SpectralRecenterCropResources& r [[buffer(0)]], constant SpectralRecenterCropParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint total = p.batches * p.ny * p.nx;
    if (tid >= total) return;
    uint x = tid % p.nx, y = (tid / p.nx) % p.ny, b = tid / (p.nx * p.ny);
    uint source = (b * p.ey + y) * p.ex + x;
    r.output[tid] = r.filtered[source] + r.corrections[b];
}

kernel void spectral_plane_correction(constant SpectralPlaneCorrectionResources& r [[buffer(0)]], constant SpectralPlaneCorrectionParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint finite = 1;
    for (int batch = 0; batch < p.batches; ++batch) {
        uint offset = uint(batch) * 3;
        float correction = p.preserve_mean
            ? (r.partial_sums[offset] - r.partial_sums[offset + 1]) / float(p.plane_count)
            : 0.0f;
        r.corrections[batch] = correction;
        finite &= r.partial_sums[offset + 2] == 1.0f;
    }
    r.status[0] = finite ? 1.0f : 0.0f;
}

kernel void spectral_final_stats(constant SpectralFinalStatsResources& r [[buffer(0)]], constant SpectralFinalStatsParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.partial_count)) return;
    uint start = tid * 256, end = min(start + 256, uint(p.count));
    float sums[7] = {0.0f, 0.0f, 0.0f, 0.0f, INFINITY, INFINITY, 1.0f};
    uint finite = 1;
    for (uint i = start; i < end; ++i) {
        float a = r.original[i], b = r.output[i], d = b - a;
        finite &= isfinite(a) && isfinite(b);
        sums[0] += a; sums[1] += b; sums[2] += d * d;
        sums[3] = max(sums[3], abs(d));
        sums[4] = min(sums[4], a); sums[5] = min(sums[5], b);
    }
    sums[6] = finite ? 1.0f : 0.0f;
    for (uint j = 0; j < 7; ++j) r.stats[tid * 7 + j] = sums[j];
}

kernel void spectral_stats_reduce(constant SpectralStatsReduceResources& r [[buffer(0)]], constant SpectralStatsReduceParams& p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint total = uint(p.batches) * uint(p.output_count);
    if (tid >= total) return;
    uint batch = tid / uint(p.output_count), block = tid % uint(p.output_count);
    uint start = block * 256, end = min(start + 256, uint(p.input_count));
    float values[7] = {0.0f, 0.0f, 0.0f, 0.0f, INFINITY, INFINITY, 1.0f};
    if (p.width == 3) values[2] = 1.0f;
    for (uint i = start; i < end; ++i) {
        uint offset = (batch * uint(p.input_count) + i) * uint(p.width);
        if (p.width == 3) {
            values[0] += r.partials[offset];
            values[1] += r.partials[offset + 1];
            values[2] = min(values[2], r.partials[offset + 2]);
        } else {
            values[0] += r.partials[offset]; values[1] += r.partials[offset + 1];
            values[2] += r.partials[offset + 2];
            values[3] = max(values[3], r.partials[offset + 3]);
            values[4] = min(values[4], r.partials[offset + 4]);
            values[5] = min(values[5], r.partials[offset + 5]);
            values[6] = min(values[6], r.partials[offset + 6]);
        }
    }
    uint output = (batch * uint(p.output_count) + block) * uint(p.width);
    if (p.width == 3) {
        r.reduced[output] = values[0]; r.reduced[output + 1] = values[1];
        r.reduced[output + 2] = values[2];
    } else {
        for (uint j = 0; j < 7; ++j) r.reduced[output + j] = values[j];
    }
}
