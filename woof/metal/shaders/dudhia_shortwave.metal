#include <metal_stdlib>
#include "dudhia_shortwave_bindings.metalh"
using namespace metal;

constant float ALBTAB[20] = {
    0.f, 69.f, 90.f, 94.f, 96.f,
    0.f, 58.f, 80.f, 90.f, 92.f,
    0.f, 40.f, 70.f, 82.f, 85.f,
    0.f, 15.f, 60.f, 78.f, 80.f
};
constant float ABSTAB[20] = {
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 2.5f, 2.6f, 3.3f, 3.7f,
    0.f, 4.f, 7.f, 10.f, 10.f,
    0.f, 5.f, 10.f, 14.f, 15.f
};

kernel void dudhia_shortwave_columns(
    constant DudhiaShortwaveColumnsResources &r [[buffer(0)]],
    constant DudhiaShortwaveColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const int nz = q.nz, ny = q.ny, nx = q.nx;
    const uint columns = uint(ny * nx);
    if (column >= columns) return;
    const float mu = r.cosine_zenith[column];
    const bool daylight = mu > 1.0e-9f;
    const float mu_safe = daylight ? mu : 1.f;
    const float surface_albedo = r.albedo[column];
    float sdown = daylight ? q.solcon * mu_safe : 0.f;
    const float top_flux = sdown;
    float ww = 0.f, uv = 0.f, oldalb = 0.f, oldabc = 0.f;
    float totabs = 0.f, dsca = 0.f, dabs = 0.f, dscld = 0.f;
    const float cssca = q.swrad_scat * 1.0e-5f;
    for (int k = 0; k < nz; ++k) {
        const int index = (nz - 1 - k) * int(columns) + int(column);
        const float t = r.temperature[index];
        const float p = r.pressure[index];
        const float layer_dz = r.dz[index];
        const float rho = p / (287.f * t);
        const float vapor_path = rho * max(r.qv[index], 0.f) * layer_dz * 1000.f;
        const float air_path = rho * layer_dz;
        float cloud_path = 0.f;
        if (q.icloud != 0) {
            cloud_path = rho * 1000.f * layer_dz * (
                max(r.qc[index], 0.f) + 0.1f * max(r.qi[index], 0.f)
                + 0.05f * max(r.qr[index], 0.f)
                + 0.02f * max(r.qs[index], 0.f)
                + 0.05f * max(r.qg[index], 0.f));
        }
        ww += cloud_path;
        uv += vapor_path;
        const float wgm = ww / mu_safe;
        const float ugcm = uv * 0.0001f / mu_safe;
        const float oldabs = totabs;
        totabs = 2.9f * ugcm /
            (pow(1.f + 141.5f * ugcm, 0.635f) + 5.925f * ugcm);
        const float xsca = cssca * air_path / mu_safe;
        const float safe_sdown = daylight ? sdown : 1.f;
        float xabs = (totabs - oldabs)
            * (top_flux - dscld - dsca) / safe_sdown;
        xabs = max(xabs, 0.f);
        const float alw = min(log10(wgm + 1.f), 3.999f);
        const int jlo = int(floor(alw));
        const float wy = alw - float(jlo);
        const int ilo = mu_safe > 0.5f ? 2 : (mu_safe > 0.2f ? 1 : 0);
        const float xmu_lo = ilo == 0 ? 0.f : (ilo == 1 ? 0.2f : 0.5f);
        const float xmu_hi = ilo == 0 ? 0.2f : (ilo == 1 ? 0.5f : 1.f);
        const float wx = (mu_safe - xmu_lo) / (xmu_hi - xmu_lo);
        const int ihi = ilo + 1, jhi = jlo + 1;
        const float a00 = ALBTAB[ilo * 5 + jlo];
        const float a10 = ALBTAB[ihi * 5 + jlo];
        const float a01 = ALBTAB[ilo * 5 + jhi];
        const float a11 = ALBTAB[ihi * 5 + jhi];
        const float b00 = ABSTAB[ilo * 5 + jlo];
        const float b10 = ABSTAB[ihi * 5 + jlo];
        const float b01 = ABSTAB[ilo * 5 + jhi];
        const float b11 = ABSTAB[ihi * 5 + jhi];
        const float alba = (1.f-wx)*(1.f-wy)*a00 + wx*(1.f-wy)*a10
                         + (1.f-wx)*wy*a01 + wx*wy*a11;
        const float absc = (1.f-wx)*(1.f-wy)*b00 + wx*(1.f-wy)*b10
                         + (1.f-wx)*wy*b01 + wx*wy*b11;
        const float clear_available = top_flux - dsca - dabs;
        const float xalb = max((alba - oldalb) * clear_available / safe_sdown, 0.f);
        float xabsc = max((absc - oldabc) * clear_available / safe_sdown, 0.f);
        dscld += (xalb + xabsc) * sdown * 0.01f;
        dsca += xsca * sdown;
        dabs += xabs * sdown;
        oldalb = alba;
        oldabc = absc;
        float trans = 100.f - xalb - xabsc - 100.f*xabs - 100.f*xsca;
        if (trans < 1.f) {
            const float attenuation = xalb + xabsc + 100.f*xabs + 100.f*xsca;
            const float ff = 99.f / max(attenuation, 1.0e-30f);
            xabsc *= ff;
            xabs *= ff;
            trans = 1.f;
        }
        r.heating[index] = daylight
            ? (sdown * (xabsc + 100.f*xabs) * 0.01f
               / (rho * 1004.5f * layer_dz)) / r.exner[index]
            : 0.f;
        sdown = daylight ? max(1.0e-9f, sdown * trans * 0.01f) : 0.f;
    }
    r.swdown[column] = sdown;
    r.gsw[column] = (1.f - surface_albedo) * sdown;
    r.coszen[column] = mu;
}
