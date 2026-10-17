// Hex v8.4.1 Metal transcription of CUDA kernel validate_prep_v841_f32 (woof/hex/cuda_physics_prep_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_validate_prep_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)

static inline uint prep_positive_q_bits_v841(const float raw)
{
    const uint bits = as_type<uint>(raw);
    return (bits & 0x80000000u) == 0u ? bits : 0u;
}

static inline float prep_positive_q_v841(const float raw)
{
    return as_type<float>(prep_positive_q_bits_v841(raw));
}

static inline bool prep_finite_bits_v841(const float raw)
{
    return (as_type<uint>(raw) & 0x7fffffffu) < 0x7f800000u;
}

static inline bool prep_negative_nonzero_bits_v841(const float raw)
{
    const uint bits = as_type<uint>(raw);
    return (bits & 0x80000000u) != 0u
        && (bits & 0x7fffffffu) != 0u;
}


kernel void hexv841_validate_prep_v841_f32(
    constant Hexv841ValidatePrepV841F32Resources &r [[buffer(0)]],
    constant Hexv841ValidatePrepV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int maxedges = p.maxedges;
    const int wsm6_ready = p.wsm6_ready;
    const float gravity = p.gravity;
    const float rvord = p.rvord;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const float *coeffs_reconstruct = r.coeffs_reconstruct;
    device const float *lat_cell = r.lat_cell;
    device const float *lon_cell = r.lon_cell;
    device const float *source_scalars = r.source_scalars;
    device const float *source_theta_m = r.source_theta_m;
    device const float *source_exner = r.source_exner;
    const int nspecies = p.nspecies;
    device const float *qspecies = r.qspecies;
    device const float *u_p = r.u_p;
    device const float *v_p = r.v_p;
    device const float *zz_p = r.zz_p;
    device const float *rho_dry = r.rho_dry;
    device const float *rho_p = r.rho_p;
    device const float *th_p = r.th_p;
    device const float *t_p = r.t_p;
    device const float *pi_p = r.pi_p;
    device const float *pres_p = r.pres_p;
    device const float *zmid_p = r.zmid_p;
    device const float *dz_p = r.dz_p;
    device const float *pres_hyd_p = r.pres_hyd_p;
    device const float *pres_hydd_p = r.pres_hydd_p;
    device const float *znu_p = r.znu_p;
    device const float *znu_hyd_p = r.znu_hyd_p;
    device const float *w_p = r.w_p;
    device const float *z_p = r.z_p;
    device const float *t2_p = r.t2_p;
    device const float *pres2_p = r.pres2_p;
    device const float *pres2_hyd_p = r.pres2_hyd_p;
    device const float *pres2_hydd_p = r.pres2_hydd_p;
    device const float *psfc_p = r.psfc_p;
    device const float *psfc_hyd_p = r.psfc_hyd_p;
    device const float *psfc_hydd_p = r.psfc_hydd_p;
    device const float *plrad = r.plrad;
    device const float *noah_dz8w = r.noah_dz8w;
    device const float *noah_qv_curr = r.noah_qv_curr;
    device const float *noah_t_phy = r.noah_t_phy;
    device const float *noah_u_phy = r.noah_u_phy;
    device const float *noah_v_phy = r.noah_v_phy;
    device const float *noah_p8w = r.noah_p8w;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    const int plane = nlev * ncells;
    const int count = n_edges_on_cell[cell];
    if (count < 0 || count > maxedges || !isfinite(lat_cell[cell])
            || !isfinite(lon_cell[cell])) atomicExch(invalid, 1);
    for (int slot = 0; slot < maxedges; ++slot) {
        const int edge = edges_on_cell[cell * maxedges + slot];
        if (slot < count) {
            if (edge < 0 || edge >= nedges) atomicExch(invalid, 1);
            const int ci = (cell * maxedges + slot) * 3;
            if (!isfinite(coeffs_reconstruct[ci])
                    || !isfinite(coeffs_reconstruct[ci + 1])
                    || !isfinite(coeffs_reconstruct[ci + 2]))
                atomicExch(invalid, 1);
        } else if (edge != -1) {
            atomicExch(invalid, 1);
        }
    }
    // The 15 non-species mass-level fields.  The species are checked from
    // qspecies, whose length is the row's, so this table no longer grows
    // with the scheme.
    device const float *mass[15] = {
        u_p, v_p, zz_p,
        rho_dry, rho_p, th_p, t_p, pi_p, pres_p, zmid_p, dz_p,
        pres_hyd_p, pres_hydd_p, znu_p, znu_hyd_p};
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        for (int f = 0; f < 15; ++f)
            if (!isfinite(mass[f][i])) atomicExch(invalid, 1);
        for (int q = 0; q < nspecies; ++q) {
            if (!isfinite(qspecies[q * plane + i])) atomicExch(invalid, 1);
            const float raw = source_scalars[q * plane + i];
            const float value = qspecies[q * plane + i];
            const uint expected = prep_positive_q_bits_v841(raw);
            if (!prep_finite_bits_v841(raw)
                    || as_type<uint>(value) != expected)
                atomicExch(invalid, 1);
            if ((as_type<uint>(value) & 0x80000000u) != 0u)
                atomicExch(invalid, 1);
            if (wsm6_ready && prep_negative_nonzero_bits_v841(raw))
                atomicExch(invalid, 1);
        }
        const float expected_rho = mpas_mul(
            rho_dry[i], mpas_add(1.0f, qspecies[0 * plane + i]));
        const float expected_theta = mpas_div(
            source_theta_m[i],
            mpas_add(1.0f, mpas_mul(rvord, qspecies[0 * plane + i])));
        const float expected_temp = mpas_mul(expected_theta, source_exner[i]);
        const float expected_noah_temp = mpas_mul(
            mpas_div(
                source_theta_m[i],
                mpas_add(1.0f, mpas_mul(rvord, source_scalars[i]))),
            source_exner[i]);
        const float expected_dz = mpas_sub(z_p[i + ncells], z_p[i]);
        const float expected_zmid = mpas_mul(
            0.5f, mpas_add(z_p[i + ncells], z_p[i]));
        if (as_type<uint>(rho_p[i]) != as_type<uint>(expected_rho)
                || as_type<uint>(th_p[i]) != as_type<uint>(expected_theta)
                || as_type<uint>(t_p[i]) != as_type<uint>(expected_temp)
                || as_type<uint>(pi_p[i])
                    != as_type<uint>(source_exner[i])
                || as_type<uint>(dz_p[i]) != as_type<uint>(expected_dz)
                || as_type<uint>(zmid_p[i])
                    != as_type<uint>(expected_zmid))
            atomicExch(invalid, 1);
        if (as_type<uint>(noah_qv_curr[i])
                    != as_type<uint>(source_scalars[i])
                || as_type<uint>(noah_t_phy[i])
                    != as_type<uint>(expected_noah_temp)
                || as_type<uint>(noah_dz8w[i])
                    != as_type<uint>(dz_p[i])
                || as_type<uint>(noah_u_phy[i]) != as_type<uint>(u_p[i])
                || as_type<uint>(noah_v_phy[i]) != as_type<uint>(v_p[i])
                || as_type<uint>(noah_p8w[i])
                    != as_type<uint>(pres2_p[i]))
            atomicExch(invalid, 1);
        if (rho_dry[i] <= 0.0f || rho_p[i] <= 0.0f
                || th_p[i] <= 0.0f || t_p[i] <= 0.0f
                || pi_p[i] <= 0.0f || pres_p[i] <= 0.0f
                || pres_hyd_p[i] <= 0.0f || pres_hydd_p[i] <= 0.0f
                || dz_p[i] <= 0.0f || znu_p[i] <= 0.0f
                || znu_hyd_p[i] <= 0.0f) atomicExch(invalid, 1);
        if (!isfinite(noah_t_phy[i]) || noah_t_phy[i] <= 0.0f
                || !isfinite(noah_p8w[i]) || noah_p8w[i] <= 0.0f)
            atomicExch(invalid, 1);
        // MPAS detects exactly this pressure inversion in MPAS_to_physics and
        // does NOT refuse.  mpas_atmphys_interface.F:458-474 tests only pres_p,
        // only across the two lowest levels
        //   :462  if(pres_p(i,1,j) .le. pres_p(i,2,j)) then
        // writes diagnostic lines at :463-472, and its single fatal escalation
        // at :473 is COMMENTED OUT in the official source:
        //   !      call mpas_log_write('pressure increasing with height', &
        //   !                          messageType=MPAS_LOG_CRIT)
        // so execution continues past :474 unconditionally.  Refusing the step
        // was stricter than the authority, and stricter at more sites: native
        // never tests pres_hyd_p or pres_hydd_p for monotonicity at all.
        // Non-finite and non-positive pressure remain refused above.
    }

    device const float *interfaces[6] = {
        w_p, z_p, t2_p, pres2_p, pres2_hyd_p, pres2_hydd_p};
    for (int k = 0; k <= nlev; ++k) {
        const int i = k * ncells + cell;
        for (int f = 0; f < 6; ++f)
            if (!isfinite(interfaces[f][i])) atomicExch(invalid, 1);
        if (t2_p[i] <= 0.0f || pres2_p[i] <= 0.0f
                || pres2_hyd_p[i] <= 0.0f || pres2_hydd_p[i] <= 0.0f)
            atomicExch(invalid, 1);
        if (k < nlev) {
            const int above = i + ncells;
            // z_p ordering is RETAINED: the t2_p/fzm/fzp reconstruction below
            // divides by (z_p[k+1] - z_p[k-1]), which is native's own
            // expression at mpas_atmphys_interface.F:479
            //   tem1 = 1./(zgrid(k+1,i)-zgrid(k-1,i))
            // so a non-monotonic column is a genuine division hazard rather
            // than a state native tolerates.  The interface PRESSURE ordering
            // refusals beside it are removed for the reason cited above:
            // native has no interface pressure monotonicity test whatsoever.
            if (!(z_p[above] > z_p[i])) atomicExch(invalid, 1);
        }
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float inverse = mpas_div(
            1.0f, mpas_sub(z_p[i + ncells], z_p[i - ncells]));
        const float fzm = mpas_mul(
            mpas_sub(z_p[i], z_p[i - ncells]), inverse);
        const float fzp = mpas_mul(
            mpas_sub(z_p[i + ncells], z_p[i]), inverse);
        const float expected_t2 = mpas_add(
            mpas_mul(fzm, t_p[i]), mpas_mul(fzp, t_p[i - ncells]));
        const float expected_p2 = mpas_add(
            mpas_mul(fzm, pres_p[i]), mpas_mul(fzp, pres_p[i - ncells]));
        if (as_type<uint>(t2_p[i]) != as_type<uint>(expected_t2)
                || as_type<uint>(pres2_p[i])
                    != as_type<uint>(expected_p2))
            atomicExch(invalid, 1);
    }
    const int top = nlev * ncells + cell;
    float z0 = z_p[top];
    float z1 = mpas_mul(0.5f, mpas_add(z_p[top], z_p[top - ncells]));
    float z2 = mpas_mul(
        0.5f, mpas_add(z_p[top - ncells], z_p[top - 2 * ncells]));
    float w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    float w2 = mpas_sub(1.0f, w1);
    const float expected_top_t2 = mpas_add(
        mpas_mul(w1, t_p[top - ncells]),
        mpas_mul(w2, t_p[top - 2 * ncells]));
    const float expected_top_p2 = precise::exp(mpas_add(
        mpas_mul(w1, precise::log(pres_p[top - ncells])),
        mpas_mul(w2, precise::log(pres_p[top - 2 * ncells]))));
    z0 = z_p[cell];
    z1 = mpas_mul(0.5f, mpas_add(z_p[cell], z_p[ncells + cell]));
    z2 = mpas_mul(
        0.5f, mpas_add(z_p[ncells + cell], z_p[2 * ncells + cell]));
    w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    w2 = mpas_sub(1.0f, w1);
    const float expected_bottom_t2 = mpas_add(
        mpas_mul(w1, t_p[cell]), mpas_mul(w2, t_p[ncells + cell]));
    const float expected_bottom_p2 = mpas_add(
        mpas_mul(w1, pres_p[cell]), mpas_mul(w2, pres_p[ncells + cell]));
    if (as_type<uint>(t2_p[top]) != as_type<uint>(expected_top_t2)
            || as_type<uint>(pres2_p[top])
                != as_type<uint>(expected_top_p2)
            || as_type<uint>(t2_p[cell])
                != as_type<uint>(expected_bottom_t2)
            || as_type<uint>(pres2_p[cell])
                != as_type<uint>(expected_bottom_p2))
        atomicExch(invalid, 1);

    if (as_type<uint>(pres2_hyd_p[top])
                != as_type<uint>(pres2_p[top])
            || as_type<uint>(pres2_hydd_p[top])
                != as_type<uint>(pres2_p[top]))
        atomicExch(invalid, 1);
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        const float rho_a = mpas_div(
            rho_p[i], mpas_add(1.0f, qspecies[0 * plane + i]));
        const float expected_p2_hyd = mpas_add(
            pres2_hyd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_p[i]), dz_p[i]));
        const float expected_p2_hydd = mpas_add(
            pres2_hydd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_a), dz_p[i]));
        const float expected_p_hyd = mpas_mul(
            0.5f, mpas_add(expected_p2_hyd, pres2_hyd_p[i + ncells]));
        const float expected_p_hydd = mpas_mul(
            0.5f, mpas_add(expected_p2_hydd, pres2_hydd_p[i + ncells]));
        const float expected_znu = mpas_div(pres_p[i], psfc_p[cell]);
        const float expected_znu_hyd = mpas_div(
            expected_p_hyd, psfc_hyd_p[cell]);
        if (as_type<uint>(pres2_hyd_p[i])
                    != as_type<uint>(expected_p2_hyd)
                || as_type<uint>(pres2_hydd_p[i])
                    != as_type<uint>(expected_p2_hydd)
                || as_type<uint>(pres_hyd_p[i])
                    != as_type<uint>(expected_p_hyd)
                || as_type<uint>(pres_hydd_p[i])
                    != as_type<uint>(expected_p_hydd)
                || as_type<uint>(znu_p[i])
                    != as_type<uint>(expected_znu)
                || as_type<uint>(znu_hyd_p[i])
                    != as_type<uint>(expected_znu_hyd))
            atomicExch(invalid, 1);
    }
    if (!isfinite(psfc_p[cell]) || psfc_p[cell] <= 0.0f
            || !isfinite(psfc_hyd_p[cell]) || psfc_hyd_p[cell] <= 0.0f
            || !isfinite(psfc_hydd_p[cell]) || psfc_hydd_p[cell] <= 0.0f
            || !isfinite(plrad[cell]) || plrad[cell] <= 0.0f
            || as_type<uint>(psfc_hyd_p[cell])
                != as_type<uint>(pres2_hyd_p[cell])
            || as_type<uint>(psfc_hydd_p[cell])
                != as_type<uint>(pres2_hydd_p[cell])
            || as_type<uint>(plrad[cell])
                != as_type<uint>(pres2_p[top]))
        atomicExch(invalid, 1);
}
