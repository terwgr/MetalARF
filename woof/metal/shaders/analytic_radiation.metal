#include <metal_stdlib>
#include "analytic_clear_sky_fluxes_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void analytic_clear_sky_fluxes(
    constant AnalyticClearSkyFluxesResources &r [[buffer(0)]],
    constant AnalyticClearSkyFluxesParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const float solar_minutes = p.solar_minutes0 + 4.0f * r.longitude[i];
    const float hour_angle = (solar_minutes / 4.0f - 180.0f) * (M_PI_F / 180.0f);
    const float latitude = r.latitude[i] * (M_PI_F / 180.0f);
    const float cosine = sin(latitude) * p.sin_declination
        + cos(latitude) * p.cos_declination * cos(hour_angle);
    const float sw = p.solar_flux * max(cosine, 0.0f);
    const float qv = max(r.qv[i], 0.0f);
    const float vapor = max(qv * r.pressure[i] / (0.622f + qv) / 100.0f, 0.01f);
    const float emissivity = 0.70f + 0.09f * log10(vapor);
    const float temperature = p.potential_temperature
        ? r.temperature[i] * pow(r.pressure[i] / 100000.0f, float(287.0 / 1004.0))
        : r.temperature[i];
    r.swdown[i] = sw;
    r.glw[i] = (emissivity * p.sigma) * pow(temperature, 4.0f);
    r.gsw[i] = sw * (1.0f - r.albedo[i]);
}
