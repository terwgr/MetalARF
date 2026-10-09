/* WRF module_surface_driver.F MYJ post-call surface wind diagnostic. */
extern "C" __global__ void myj_driver_wspd(
    const float *u, const float *v, float *wspd, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float uu = u[i], vv = v[i];
        const float speed = sqrtf(uu * uu + vv * vv);
        wspd[i] = fmaxf(speed, 0.001f);
    }
}
