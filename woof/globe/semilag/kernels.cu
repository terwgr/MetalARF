/* Semi-Lagrangian departure points and 3-D interpolation on a Gaussian grid.
 *
 * Two entry families, both written once as a templated device body and
 * exposed through extern "C" wrappers, so there is one compiled kernel per
 * (scalar type, quasi-monotone, deficit-reporting) triple and no name
 * mangling to reason about.
 *
 *   sl_gather_*      tricubic 4x4x4 gather of F fields that share one index
 *                    and weight computation.  The measured shape: the
 *                    vertical tap loop is PARTIALLY unrolled.  Fully
 *                    unrolling it holds 64 stencil values live at once and
 *                    read 1.75x slow on the 5070 Ti; not unrolling at all
 *                    spilled the weights to local memory and read 3.4x slow.
 *   sl_gather_qmd_*  the same, plus the signed mass the quasi-monotone
 *                    limiter moved at every point.  The additive mass
 *                    fixer needs it to put a species' correction back
 *                    where the limiter took it rather than spreading it
 *                    over every point that holds the species.
 *   sl_gather5_*     the QUINTIC-horizontal gather: six-point Lagrange in
 *                    the zonal and meridional directions and the same
 *                    four-point cubic in the vertical (6x6x4 = 144 taps
 *                    against 64), for the dynamical bundle.  MEASURED
 *                    2026-09-06 on the T255 forecast day: the cubic gather
 *                    every 300 s step kept 17 percent of the Eulerian
 *                    core's 500 hPa vorticity power at total wavenumbers
 *                    181 to 230, and the four-point Lagrange weights at a
 *                    fifth of a cell lose 7.6 percent of a 3.8-cell wave's
 *                    amplitude per pass, which over 288 passes a day is
 *                    an effective drain 25 times the shipped
 *                    hyperdiffusion's at that scale.  The six-point
 *                    weights lose about half of that.  Same template, a
 *                    second table set (three reflected rows a pole rather
 *                    than two), and the cubic entry points untouched.
 *   sl_departure_*   the spherical fixed-point departure-point search, all
 *                    of its iterations in one launch, with the wind read as
 *                    geocentric Cartesian components so the poles need no
 *                    local basis and no sign convention.
 *
 * A POSITIVE-DEFINITE shape, [0, cell maximum] instead of [cell minimum,
 * cell maximum], was built here and MEASURED on 2026-09-06 and is not in
 * this file: on a condensate-shaped field (blobs over a zero background)
 * it returned the same array as the quasi-monotone shape at every point,
 * because every undershoot that shape lifts is already below zero there,
 * and on a field with a positive floor its mass error was 2.15e-3 against
 * the quasi-monotone shape's 8.04e-5.  Identical where it would have
 * helped and 27 times worse where it would not.
 *
 * Grid conventions this file assumes, every one of them checked on the
 * Python side before a launch:
 *   - longitudes are uniform, lon[i] = i * dlam, dlam = 2*pi/ni, ni EVEN
 *     (the polar reflection shifts by exactly ni/2 columns)
 *   - latitudes are the Gauss-Legendre nodes, ASCENDING, handed in as an
 *     EXTENDED table lat_ext[nj+4] whose first two entries are the two rows
 *     reflected through the south pole and whose last two are the two rows
 *     reflected through the north pole, so the table is strictly increasing
 *     over its whole length and a 4-point stencil always fits
 *   - rowoff[nj+4] and rowshift[nj+4] map an extended row to the offset of
 *     the DATA row it reads and to the zonal shift (0 or ni/2) it reads with
 *   - full level k sits at continuous index k, k in [0, nk-1], nk >= 4
 *
 * The weights of all three directions are NORMALIZED by their own sum.  That
 * is what makes the zero-displacement identity exact: at a grid point three
 * of the four Lagrange weights are exactly zero (each carries the vanishing
 * factor) and the fourth is divided by itself.  Without the normalization
 * the meridional weight at a node is 1 + O(eps) rather than 1, because it is
 * a product of node differences and their reciprocals rather than a closed
 * form, and the identity would fail by a few ulp at every point.
 */

#define SL_HALF_PI_D 1.57079632679489661923

template <typename T> struct sl_traits;

template <> struct sl_traits<float> {
    __device__ static float floor_(float x) { return floorf(x); }
    __device__ static float asin_(float x) { return asinf(x); }
    __device__ static float atan2_(float y, float x) { return atan2f(y, x); }
    __device__ static void sincos_(float x, float* s, float* c) { sincosf(x, s, c); }
    __device__ static float rsqrt_(float x) { return rsqrtf(x); }
    __device__ static float sqrt_(float x) { return sqrtf(x); }
    __device__ static float abs_(float x) { return fabsf(x); }
    __device__ static float big() { return 3.402823466e38f; }
};

template <> struct sl_traits<double> {
    __device__ static double floor_(double x) { return floor(x); }
    __device__ static double asin_(double x) { return asin(x); }
    __device__ static double atan2_(double y, double x) { return atan2(y, x); }
    __device__ static void sincos_(double x, double* s, double* c) { sincos(x, s, c); }
    __device__ static double rsqrt_(double x) { return rsqrt(x); }
    __device__ static double sqrt_(double x) { return sqrt(x); }
    __device__ static double abs_(double x) { return fabs(x); }
    __device__ static double big() { return 1.7976931348623157e308; }
};

/* Lagrange weights on the equispaced nodes 0,1,2,3 evaluated at t, then
   normalized.  At t = 0,1,2,3 exactly one weight is 1 and the rest are 0,
   in floating point as well as in exact arithmetic. */
template <typename T>
__device__ __forceinline__ void sl_cubic_equispaced(T t, T* w)
{
    const T a = t;
    const T b = t - (T)1;
    const T c = t - (T)2;
    const T d = t - (T)3;
    w[0] = -b * c * d * (T)(1.0 / 6.0);
    w[1] =  a * c * d * (T)0.5;
    w[2] = -a * b * d * (T)0.5;
    w[3] =  a * b * c * (T)(1.0 / 6.0);
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= inv; w[1] *= inv; w[2] *= inv; w[3] *= inv;
}

/* Lagrange weights on the equispaced nodes 0..5 evaluated at t, normalized
   the same way: at an integer t exactly one weight survives. */
template <typename T>
__device__ __forceinline__ void sl_quintic_equispaced(T t, T* w)
{
    const T d0 = t;
    const T d1 = t - (T)1;
    const T d2 = t - (T)2;
    const T d3 = t - (T)3;
    const T d4 = t - (T)4;
    const T d5 = t - (T)5;
    w[0] = -d1 * d2 * d3 * d4 * d5 * (T)(1.0 / 120.0);
    w[1] =  d0 * d2 * d3 * d4 * d5 * (T)(1.0 / 24.0);
    w[2] = -d0 * d1 * d3 * d4 * d5 * (T)(1.0 / 12.0);
    w[3] =  d0 * d1 * d2 * d4 * d5 * (T)(1.0 / 12.0);
    w[4] = -d0 * d1 * d2 * d3 * d5 * (T)(1.0 / 24.0);
    w[5] =  d0 * d1 * d2 * d3 * d4 * (T)(1.0 / 120.0);
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]);
    w[0] *= inv; w[1] *= inv; w[2] *= inv;
    w[3] *= inv; w[4] *= inv; w[5] *= inv;
}

/* Latitude of a unit vector, by atan2 against the horizontal radius rather
   than asin of the vertical component.  d(asin)/dz is 1/cos(phi), which is
   156 on the outermost T255 ring, so the last bit of a float32 unit vector
   becomes 120 m of latitude there and the departure point of a resting
   atmosphere is not the arrival point.  atan2(z, hypot(x,y)) carries the
   same relative precision at every latitude, including exactly at a pole. */
template <typename T>
__device__ __forceinline__ T sl_latitude(T x, T y, T z)
{
    return sl_traits<T>::atan2_(z, sl_traits<T>::sqrt_(x * x + y * y));
}

/* The extended-row bracket m0 with lat_ext[m0] <= p < lat_ext[m0+1], clamped
   so the 4-point stencil m0-1 .. m0+2 stays inside the table.  The lookup
   table lands within a row or so; the two loops then make it exact for any
   node distribution, so a change of quadrature cannot silently mislocate. */
template <typename T>
__device__ __forceinline__ int sl_bracket(
    T p, const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    int nb, int nj, T lut_scale)
{
    int b = (int)((p + (T)SL_HALF_PI_D) * lut_scale);
    b = min(max(b, 0), nb - 1);
    int m0 = mlut[b];
    while (m0 > 1 && p < lat_ext[m0]) --m0;
    while (m0 < nj + 1 && p >= lat_ext[m0 + 1]) ++m0;
    return m0;
}

/* Meridional Lagrange weights on the true nodes, normalized. */
template <typename T>
__device__ __forceinline__ void sl_meridional_weights(
    T p, int sst, const T* __restrict__ lat_ext,
    const T* __restrict__ mrden, T* w)
{
    const T d0 = p - lat_ext[sst];
    const T d1 = p - lat_ext[sst + 1];
    const T d2 = p - lat_ext[sst + 2];
    const T d3 = p - lat_ext[sst + 3];
    const T* r = mrden + 4 * sst;
    w[0] = d1 * d2 * d3 * r[0];
    w[1] = d0 * d2 * d3 * r[1];
    w[2] = d0 * d1 * d3 * r[2];
    w[3] = d0 * d1 * d2 * r[3];
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= inv; w[1] *= inv; w[2] *= inv; w[3] *= inv;
}

/* The six-point bracket: lat_ext6[m0] <= p < lat_ext6[m0+1], m0 clamped to
   [2, nj+2] so the stencil m0-2 .. m0+3 stays inside the wider table. */
template <typename T>
__device__ __forceinline__ int sl_bracket6(
    T p, const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    int nb, int nj, T lut_scale)
{
    int b = (int)((p + (T)SL_HALF_PI_D) * lut_scale);
    b = min(max(b, 0), nb - 1);
    int m0 = mlut[b];
    while (m0 > 2 && p < lat_ext[m0]) --m0;
    while (m0 < nj + 2 && p >= lat_ext[m0 + 1]) ++m0;
    return m0;
}

/* Six-point meridional Lagrange weights on the true nodes, normalized. */
template <typename T>
__device__ __forceinline__ void sl_meridional_weights6(
    T p, int sst, const T* __restrict__ lat_ext,
    const T* __restrict__ mrden, T* w)
{
    T d[6];
#pragma unroll
    for (int m = 0; m < 6; ++m) d[m] = p - lat_ext[sst + m];
    const T* r = mrden + 6 * sst;
#pragma unroll
    for (int m = 0; m < 6; ++m) {
        T prod = r[m];
#pragma unroll
        for (int q = 0; q < 6; ++q) if (q != m) prod *= d[q];
        w[m] = prod;
    }
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]);
#pragma unroll
    for (int m = 0; m < 6; ++m) w[m] *= inv;
}

/* One body for both horizontal widths.  NH = 4 is the tricubic gather the
   gates of record were measured on and its arithmetic is untouched: the
   NH = 6 branches are compile-time and the four-point weights, bracket and
   address arithmetic are the same expressions in the same order. */
template <typename T, int QM, int UNROLL, int DEFICIT, int NH>
__device__ __forceinline__ void sl_gather_body(
    const T* __restrict__ src,
    const T* __restrict__ xis, const T* __restrict__ phi,
    const T* __restrict__ zk,
    const T* __restrict__ lat_ext, const T* __restrict__ mrden,
    const int* __restrict__ mlut, const int* __restrict__ rowoff,
    const int* __restrict__ rowshift,
    T* __restrict__ out, T* __restrict__ dev,
    int F, int nk, int nj, int ni, int nb, T lut_scale)
{
    const long long n = (long long)nk * (long long)nj * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    T p = phi[gid];
    p = min(max(p, -(T)SL_HALF_PI_D), (T)SL_HALF_PI_D);
    T xk = zk[gid];
    xk = min(max(xk, (T)0), (T)(nk - 1));
    const T xi = xis[gid];

    /* zonal: uniform nodes, periodic, never clamped */
    const T fi = sl_traits<T>::floor_(xi);
    const int i0 = (int)fi - (NH / 2 - 1);
    T wx[NH];
    if (NH == 6) sl_quintic_equispaced<T>(xi - (T)i0, wx);
    else         sl_cubic_equispaced<T>(xi - (T)i0, wx);
    int a[NH], b[NH];
    a[0] = i0 % ni; if (a[0] < 0) a[0] += ni;
#pragma unroll
    for (int q = 1; q < NH; ++q) { a[q] = a[q - 1] + 1; if (a[q] >= ni) a[q] -= ni; }
    const int half = ni >> 1;
#pragma unroll
    for (int q = 0; q < NH; ++q) { b[q] = a[q] + half; if (b[q] >= ni) b[q] -= ni; }

    /* meridional: true Gauss-Legendre nodes, poles by reflection */
    int sst;
    T wy[NH];
    if (NH == 6) {
        const int m0 = sl_bracket6<T>(p, lat_ext, mlut, nb, nj, lut_scale);
        sst = m0 - 2;
        sl_meridional_weights6<T>(p, sst, lat_ext, mrden, wy);
    } else {
        const int m0 = sl_bracket<T>(p, lat_ext, mlut, nb, nj, lut_scale);
        sst = m0 - 1;
        sl_meridional_weights<T>(p, sst, lat_ext, mrden, wy);
    }

    /* vertical: uniform in the level index, clamped to the data range */
    const int kb = min(max((int)sl_traits<T>::floor_(xk), 0), nk - 2);
    int k0 = kb - 1;
    k0 = min(max(k0, 0), nk - 4);
    const int kbox = kb - k0;
    T wz[4];
    sl_cubic_equispaced<T>(xk - (T)k0, wz);

    int adr[NH * NH];
#pragma unroll
    for (int m = 0; m < NH; ++m) {
        const int ro = rowoff[sst + m];
        const int sh = rowshift[sst + m];
#pragma unroll
        for (int q = 0; q < NH; ++q) adr[NH * m + q] = ro + (sh ? b[q] : a[q]);
    }
    /* the limiter's cell: the two rows and two columns around the point */
    const int mid = NH / 2 - 1;

    const long long plane = (long long)nj * (long long)ni;
    for (int f = 0; f < F; ++f) {
        const T* s = src + (long long)f * n;
        T acc = (T)0;
        T lo = sl_traits<T>::big();
        T hi = -sl_traits<T>::big();
#pragma unroll UNROLL
        for (int c = 0; c < 4; ++c) {
            const T* sp = s + (long long)(k0 + c) * plane;
            T rowsum = (T)0;
#pragma unroll
            for (int m = 0; m < NH; ++m) {
                T v[NH];
#pragma unroll
                for (int q = 0; q < NH; ++q) v[q] = sp[adr[NH * m + q]];
                T colsum = wx[0] * v[0] + wx[1] * v[1]
                         + wx[2] * v[2] + wx[3] * v[3];
                if (NH == 6) colsum += wx[4] * v[4] + wx[5] * v[5];
                rowsum += wy[m] * colsum;
                if (QM && (c == kbox || c == kbox + 1)
                        && (m == mid || m == mid + 1)) {
                    lo = min(lo, min(v[mid], v[mid + 1]));
                    hi = max(hi, max(v[mid], v[mid + 1]));
                }
            }
            acc += wz[c] * rowsum;
        }
        T kept = acc;
        if (QM) kept = min(max(acc, lo), hi);
        out[(long long)f * n + gid] = kept;
        /* The signed mass the clip moved at this point, raw minus kept:
         * positive where the limiter cut an overshoot away (mass the
         * gather lost) and negative where it lifted an undershoot (mass
         * the gather gained).  It is the ONLY place the interpolation
         * knows where its mass went, and the additive mass fixer puts the
         * correction back exactly there instead of spreading it over
         * every point that holds the species. */
        if (DEFICIT) dev[(long long)f * n + gid] = acc - kept;
    }
}

extern "C" __global__ void sl_gather_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 0, 2, 0, 4>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                 rowoff, rowshift, out, (float*)0, F, nk, nj,
                                 ni, nb, lut_scale); }

extern "C" __global__ void sl_gather_qm_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 1, 2, 0, 4>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                 rowoff, rowshift, out, (float*)0, F, nk, nj,
                                 ni, nb, lut_scale); }

extern "C" __global__ void sl_gather_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 0, 2, 0, 4>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                  rowoff, rowshift, out, (double*)0, F, nk, nj,
                                  ni, nb, lut_scale); }

extern "C" __global__ void sl_gather_qm_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 1, 2, 0, 4>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                  rowoff, rowshift, out, (double*)0, F, nk, nj,
                                  ni, nb, lut_scale); }

/* The quasi-monotone gather that also reports where its limiter moved
 * mass.  Same arithmetic in ``out``, bit for bit, as sl_gather_qm_*: the
 * clip is computed either way and this variant writes the difference it
 * would otherwise have discarded.  The gate that says so is
 * test_the_deficit_gather_returns_the_same_values_as_the_plain_one. */
extern "C" __global__ void sl_gather_qmd_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out, float* dev,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 1, 2, 1, 4>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                 rowoff, rowshift, out, dev, F, nk, nj, ni,
                                 nb, lut_scale); }

extern "C" __global__ void sl_gather_qmd_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out, double* dev,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 1, 2, 1, 4>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                  rowoff, rowshift, out, dev, F, nk, nj, ni,
                                  nb, lut_scale); }

/* The quintic-horizontal gather.  Its table arguments are the SIX-POINT
 * tables (lat_ext6, mrden6, mlut6, rowoff6, rowshift6, their bin count and
 * scale); handing it the four-point tables reads the wrong rows, which is
 * why interpolate.gather_batch selects the set from the order and never
 * from the caller. */
extern "C" __global__ void sl_gather5_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 0, 2, 0, 6>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                    rowoff, rowshift, out, (float*)0, F, nk,
                                    nj, ni, nb, lut_scale); }

extern "C" __global__ void sl_gather5_qm_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 1, 2, 0, 6>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                    rowoff, rowshift, out, (float*)0, F, nk,
                                    nj, ni, nb, lut_scale); }

extern "C" __global__ void sl_gather5_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 0, 2, 0, 6>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                     rowoff, rowshift, out, (double*)0, F, nk,
                                     nj, ni, nb, lut_scale); }

extern "C" __global__ void sl_gather5_qm_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 1, 2, 0, 6>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                     rowoff, rowshift, out, (double*)0, F, nk,
                                     nj, ni, nb, lut_scale); }

/* ------------------------------------------------------------------ */
/* trilinear gather of four fields at one point, used by the search    */

template <typename T>
__device__ __forceinline__ void sl_trilinear4(
    const T* __restrict__ f0, const T* __restrict__ f1,
    const T* __restrict__ f2, const T* __restrict__ f3,
    T lamd, T phid, T xkd,
    const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    const int* __restrict__ rowoff, const int* __restrict__ rowshift,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,
    T* out)
{
    const T p = min(max(phid, -(T)SL_HALF_PI_D), (T)SL_HALF_PI_D);
    const T xk = min(max(xkd, (T)0), (T)(nk - 1));
    const T xi = lamd * inv_dlam;

    const T fi = sl_traits<T>::floor_(xi);
    const T ax = xi - fi;
    int c0 = ((int)fi) % ni; if (c0 < 0) c0 += ni;
    int c1 = c0 + 1; if (c1 >= ni) c1 -= ni;
    const int half = ni >> 1;
    int d0 = c0 + half; if (d0 >= ni) d0 -= ni;
    int d1 = c1 + half; if (d1 >= ni) d1 -= ni;

    const int m0 = sl_bracket<T>(p, lat_ext, mlut, nb, nj, lut_scale);
    const T la = lat_ext[m0];
    const T lb = lat_ext[m0 + 1];
    const T ay = (p - la) / (lb - la);
    const int r0 = rowoff[m0], s0 = rowshift[m0];
    const int r1 = rowoff[m0 + 1], s1 = rowshift[m0 + 1];
    const int i00 = r0 + (s0 ? d0 : c0), i01 = r0 + (s0 ? d1 : c1);
    const int i10 = r1 + (s1 ? d0 : c0), i11 = r1 + (s1 ? d1 : c1);

    int k0 = (int)sl_traits<T>::floor_(xk);
    k0 = min(max(k0, 0), nk - 2);
    const T az = xk - (T)k0;

    const long long plane = (long long)nj * (long long)ni;
    const long long o0 = (long long)k0 * plane;
    const long long o1 = o0 + plane;
    const T bx = (T)1 - ax, by = (T)1 - ay, bz = (T)1 - az;

    const T* fs[4] = {f0, f1, f2, f3};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const T* s = fs[q];
        const T t0 = by * (bx * s[o0 + i00] + ax * s[o0 + i01])
                   + ay * (bx * s[o0 + i10] + ax * s[o0 + i11]);
        const T t1 = by * (bx * s[o1 + i00] + ax * s[o1 + i01])
                   + ay * (bx * s[o1 + i10] + ax * s[o1 + i11]);
        out[q] = bz * t0 + az * t1;
    }
}

/* The spherical fixed-point departure-point search, all iterations in one
 * launch.  The wind arrives as geocentric Cartesian components of the
 * horizontal (tangent) velocity in m/s, and the vertical rate as level
 * indices per second at full levels.  The extrapolated fields are read at
 * the ARRIVAL point, so they need no interpolation at all.
 *
 * Writes the departure longitude and latitude in radians, the departure
 * continuous level index, and the size of the LAST iteration's move in
 * metres and in level indices, which is what the convergence gate reads.
 */
template <typename T>
__device__ __forceinline__ void sl_departure_body(
    const T* __restrict__ vx, const T* __restrict__ vy,
    const T* __restrict__ vz, const T* __restrict__ sv,
    const T* __restrict__ vxe, const T* __restrict__ vye,
    const T* __restrict__ vze, const T* __restrict__ sve,
    const T* __restrict__ lat_ext,
    const int* __restrict__ mlut, const int* __restrict__ rowoff,
    const int* __restrict__ rowshift,
    T* __restrict__ xi_d, T* __restrict__ phi_d, T* __restrict__ zk_d,
    T* __restrict__ move_m, T* __restrict__ move_k,
    int iters, T dt, T radius, T dlam,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale)
{
    const long long n = (long long)nk * (long long)nj * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const long long plane = (long long)nj * (long long)ni;
    const int i = (int)(gid % ni);
    const int j = (int)((gid / ni) % nj);
    const int k = (int)(gid / plane);

    const T phia = lat_ext[j + 2];
    const T lama = (T)i * dlam;
    T sph, cph, sla, cla;
    sl_traits<T>::sincos_(phia, &sph, &cph);
    sl_traits<T>::sincos_(lama, &sla, &cla);
    const T ax = cph * cla, ay = cph * sla, az = sph;

    const T ex = vxe[gid], ey = vye[gid], ez = vze[gid], es = sve[gid];
    const T inv_r = (T)1 / radius;

    /* first guess: one explicit backward step on the time-n wind at the
       arrival point, which is exactly what the arrival index holds */
    T qx = ax - dt * vx[gid] * inv_r;
    T qy = ay - dt * vy[gid] * inv_r;
    T qz = az - dt * vz[gid] * inv_r;
    T rn = sl_traits<T>::rsqrt_(qx * qx + qy * qy + qz * qz);
    qx *= rn; qy *= rn; qz *= rn;
    T qk = (T)k - dt * sv[gid];
    qk = min(max(qk, (T)0), (T)(nk - 1));

    T last_m = (T)0, last_k = (T)0;
    const T hdt = (T)0.5 * dt;
    for (int it = 0; it < iters; ++it) {
        const T lamd = sl_traits<T>::atan2_(qy, qx);
        const T phid = sl_latitude<T>(qx, qy, qz);
        T w[4];
        sl_trilinear4<T>(vx, vy, vz, sv, lamd, phid, qk, lat_ext, mlut,
                         rowoff, rowshift, nb, nk, nj, ni, inv_dlam,
                         lut_scale, w);
        T px = ax - hdt * (ex + w[0]) * inv_r;
        T py = ay - hdt * (ey + w[1]) * inv_r;
        T pz = az - hdt * (ez + w[2]) * inv_r;
        rn = sl_traits<T>::rsqrt_(px * px + py * py + pz * pz);
        px *= rn; py *= rn; pz *= rn;
        T pk = (T)k - hdt * (es + w[3]);
        pk = min(max(pk, (T)0), (T)(nk - 1));

        const T cx = px - qx, cy = py - qy, cz = pz - qz;
        const T chord = sl_traits<T>::sqrt_(cx * cx + cy * cy + cz * cz);
        last_m = radius * (T)2 * sl_traits<T>::asin_(
            min(max((T)0.5 * chord, (T)0), (T)1));
        last_k = sl_traits<T>::abs_(pk - qk);
        qx = px; qy = py; qz = pz; qk = pk;
    }

    xi_d[gid] = sl_traits<T>::atan2_(qy, qx) * inv_dlam;
    phi_d[gid] = sl_latitude<T>(qx, qy, qz);
    zk_d[gid] = qk;
    move_m[gid] = last_m;
    move_k[gid] = last_k;
}

extern "C" __global__ void sl_departure_f32(
    const float* vx, const float* vy, const float* vz, const float* sv,
    const float* vxe, const float* vye, const float* vze, const float* sve,
    const float* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift,
    float* xi_d, float* phi_d, float* zk_d, float* move_m, float* move_k,
    int iters, float dt, float radius, float dlam,
    int nb, int nk, int nj, int ni, float inv_dlam, float lut_scale)
{ sl_departure_body<float>(vx, vy, vz, sv, vxe, vye, vze, sve, lat_ext,
                           mlut, rowoff, rowshift, xi_d, phi_d, zk_d,
                           move_m, move_k, iters, dt, radius, dlam, nb, nk,
                           nj, ni, inv_dlam, lut_scale); }

extern "C" __global__ void sl_departure_f64(
    const double* vx, const double* vy, const double* vz, const double* sv,
    const double* vxe, const double* vye, const double* vze, const double* sve,
    const double* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift,
    double* xi_d, double* phi_d, double* zk_d, double* move_m, double* move_k,
    int iters, double dt, double radius, double dlam,
    int nb, int nk, int nj, int ni, double inv_dlam, double lut_scale)
{ sl_departure_body<double>(vx, vy, vz, sv, vxe, vye, vze, sve, lat_ext,
                            mlut, rowoff, rowshift, xi_d, phi_d, zk_d,
                            move_m, move_k, iters, dt, radius, dlam, nb, nk,
                            nj, ni, inv_dlam, lut_scale); }
