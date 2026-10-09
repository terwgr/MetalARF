#include "complex64_add_bindings.metalh"
#include "complex64_multiply_bindings.metalh"
#include "complex64_conjugate_bindings.metalh"
#include "complex64_squared_magnitude_bindings.metalh"

// Complex64 is represented as interleaved [real, imaginary] float32 words.
// These entrypoints expose componentwise binary32 operations and do not
// implement NumPy's exceptional complex recovery rules for Inf/NaN pairs.
// M2 probing observed subnormal multiplication results flushing to signed
// zero. Callers requiring subnormal preservation must refuse these kernels.

kernel void complex64_add(
    constant Complex64AddResources& r [[buffer(0)]],
    constant Complex64AddParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (p.n <= 0 || p.n2 <= 0 || uint(p.n2) != 2u * uint(p.n)
            || i >= uint(p.n)) return;
    uint j = 2u * i;
    r.dst[j] = r.a[j] + r.b[j];
    r.dst[j + 1u] = r.a[j + 1u] + r.b[j + 1u];
}

kernel void complex64_multiply(
    constant Complex64MultiplyResources& r [[buffer(0)]],
    constant Complex64MultiplyParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (p.n <= 0 || p.n2 <= 0 || uint(p.n2) != 2u * uint(p.n)
            || i >= uint(p.n)) return;
    uint j = 2u * i;
    float ar = r.a[j];
    float ai = r.a[j + 1u];
    float br = r.b[j];
    float bi = r.b[j + 1u];
    // Build with FP contraction disabled so each product rounds before the
    // final add/subtract, matching NumPy's ordinary finite complex64 path.
    float real_product = ar * br;
    float imag_product = ai * bi;
    float real_value = real_product - imag_product;
    float cross_real = ar * bi;
    float cross_imag = ai * br;
    float imag_value = cross_real + cross_imag;
    r.dst[j] = real_value;
    r.dst[j + 1u] = imag_value;
}

kernel void complex64_conjugate(
    constant Complex64ConjugateResources& r [[buffer(0)]],
    constant Complex64ConjugateParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (p.n <= 0 || p.n2 <= 0 || uint(p.n2) != 2u * uint(p.n)
            || i >= uint(p.n)) return;
    uint j = 2u * i;
    r.dst[j] = r.src[j];
    r.dst[j + 1u] = -r.src[j + 1u];
}

kernel void complex64_squared_magnitude(
    constant Complex64SquaredMagnitudeResources& r [[buffer(0)]],
    constant Complex64SquaredMagnitudeParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (p.n <= 0 || p.n2 <= 0 || uint(p.n2) != 2u * uint(p.n)
            || i >= uint(p.n)) return;
    uint j = 2u * i;
    float real_square = r.src[j] * r.src[j];
    float imag_square = r.src[j + 1u] * r.src[j + 1u];
    r.dst[i] = real_square + imag_square;
}
