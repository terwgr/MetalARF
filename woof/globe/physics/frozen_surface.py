"""Heat conduction column for the surfaces the land surface skips as frozen.

The Noah port skips two kinds of column before SFLX exactly as WRF's
driver does: sea ice (``xice >= xice_threshold``, noah.cu:969; WRF hands
those to ``seaice_noah`` in module_sf_noah_seaice_drv.F) and land ice
(``ivgtyp == isice``, noah.cu:976; WRF's SFLX_GLACIAL, not ported).
Before the cold-start seeding no column was sea ice, and the land-ice
columns (Greenland, Antarctica) kept the analysis skin temperature for
the whole run: a surface that never responded to its own radiation or
to the air above it.  This module gives both kinds the same defined
behaviour in place of the two unported WRF schemes: a four-node heat
conduction column through the medium that is actually there, snow over
ice, whose top node is the skin,

    C_0 dT_0/dt = SW_down (1 - albedo) + eps (LW_down - sigma T_0^4)
                  - H - L_s E - G_01 (T_0 - T_1)
    C_k dT_k/dt = G_{k-1,k} (T_{k-1} - T_k) - G_{k,k+1} (T_k - T_{k+1})

Each node takes the conductivity and heat capacity of the medium at its
midpoint: snow (0.31 W m-1 K-1, 300 kg m-3) down to the analysed snow
depth, ice (2.03 W m-1 K-1, 917 kg m-3) below it.  A sea-ice column is
the snow plus the analysed ice thickness with its bottom held at the
freezing point of sea water; a land-ice column is the top two metres of
the sheet (Noah's layer thicknesses with a thin skin layer) coupled to
the deep-soil climatology at Noah's 8 m through six metres of firn read
as ice.  The longwave emission is linearised about the old skin and the
whole column is stepped implicitly (a tridiagonal solve per column,
vectorised over the grid); the turbulent fluxes are the surface layer's
of the same call; every node is capped at the melting point, and the
energy above it melts snow or ice the water ledger does not carry (the
same stated limit as the ocean, which has no mixed layer here).  The
column lives in the surface state's four soil-temperature layers (TSLB),
so the checkpoint, the export and a restart carry it without a new
array.  On a sea-ice column whose analysed fraction is below one, the
skin the surface layer and the radiation see is the fraction-weighted
blend of the ice skin and open water at the freezing point, the reading
a fractional pack presents to the air; the column itself is the ice.

The lane's first two graded arms named what this column must do.  A
single 0.1 m skin slab conducting to the bottom through the whole ice
(arm 1) cooled the Antarctic ice sheet twice as fast as the reference
over the day and lost the open-water warmth of a partial pack.  Four
ice nodes under the snow taken as one lumped resistance (arm 2) fixed
the pack (its skin deficit fell from 4.0 to 3.0 K, the 2 m air's rmse
from 4.3 to 3.3 K) and left the ice sheets untouched (the interior skin
still 5.9 K too cold under one to two metres of analysed snow), because
a skin of ice sitting on a metre of insulation is as decoupled as the
slab was.  Snow as the top nodes, with its own capacity and conduction,
is what the sheet's skin was missing.

The partial pack.  The surface layer runs one column per cell and sees
the fraction-weighted skin, so its sensible and latent fluxes are the
cell's; the seeding lane's column charged those whole-cell fluxes to
the ice, and the open water's heat loss (the leads sit at the freezing
point, 15 to 20 K above a winter pack's air) was paid by the ice skin,
which cooled until the extra conduction from the ocean balanced it
(the ladder on the merged tip at f024: the Antarctic partial pack drew
40 W/m2 up through the ice against the reference's 13, skin 2.3 K cold,
the full pack 0.6 K).  The column now receives the fluxes of its own
skin: the cell's flux corrected by the surface layer's own exchange
coefficient for the difference between the ice skin and the skin the
fluxes were evaluated at,

    H_ice(T_0) = H_cell + rho c_p C_h (T_0 - T_ref)
    E_ice(T_0) = E_cell + rho m C_h (q_sat,ice(T_0) - q_sat,ice(T_ref)),

implicit in the new skin (the same linearisation as the emission; on a
full pack or an ice sheet T_ref is the skin itself and only the implicit
coupling remains).  What the open water exchanges with the air stays
with the water, held at the freezing point, the stated limit of an
ocean without a mixed layer.  The atmosphere sees the composite surface
WRF's fractional sea ice presents (module_physics_init.F landuse_init:
albedo f a_ice + (1 - f) 0.08, emissivity f e_ice + (1 - f) 0.98), the
ice column its own albedo and the sea-ice emissivity WRF's SFLX_SEAICE
sets (0.98, module_sf_noah_seaice.F), and a cold start reads the
analysed composite skin back to the ice skin it blends from.

Stated divergence from WRF, recorded in the adapter contract: WRF's
seaice_noah integrates its four ice layers with its own snowpack and
SFLX_GLACIAL a glacier column with Noah's full surface energy balance;
this column is the same conduction with a simpler surface balance.
Every constant is named here so the receipt and a later port can
compare.  Arithmetic is float32 on the array module the runtime hands
in; a column outside the frozen mask is never touched, so an ice-free
planet is bit-for-bit the pre-seeding run.
"""
from __future__ import annotations

#: Melting point of ice; no node of a frozen column can exceed it.
MELT_POINT_K = 273.15
#: Freezing point of sea water, the temperature at the ice bottom and of
#: the open water in a partial pack (module_sf_noah_seaice.F holds the
#: ice base at 271.36 K).
SEA_ICE_BOTTOM_K = 271.36
#: Thermal conductivity (W m-1 K-1) and volumetric heat capacity
#: (J m-3 K-1) of the two media: ice (sea ice and firn read as ice,
#: 917 kg m-3 x 2100 J kg-1 K-1) and settled snow (300 kg m-3).
ICE_CONDUCTIVITY_W_M_K = 2.03
SNOW_CONDUCTIVITY_W_M_K = 0.31
ICE_VOLUMETRIC_HEAT_CAPACITY_J_M3_K = 917.0 * 2100.0
SNOW_VOLUMETRIC_HEAT_CAPACITY_J_M3_K = 300.0 * 2100.0
#: Number of nodes in the column: the skin and three layers below it.
COLUMN_NODES = 4
#: Albedo of a sea-ice column (WRF's seaice_albedo_default) and the snow
#: water above which a land-ice column shows the statics' maximum snow
#: albedo (real.exe's SNOWC rule) instead of the ice class's table value.
SEA_ICE_ALBEDO = 0.65
SNOW_COVERED_KG_M2 = 10.0
#: Emissivity of the sea-ice surface (WRF SFLX_SEAICE, module_sf_noah_
#: seaice.F: EMISSI = 0.98 whatever LANDUSE.TBL gives the ice class) and
#: the open-water albedo and emissivity WRF's fractional sea ice blends
#: in for the atmosphere (module_physics_init.F landuse_init, 0.08 and
#: 0.98).
SEA_ICE_EMISSIVITY = 0.98
OPEN_WATER_ALBEDO = 0.08
OPEN_WATER_EMISSIVITY = 0.98
#: Land ice: the four node thicknesses (m), Noah's soil layers with a
#: thin skin layer, and the depth of the deep-soil climatology the
#: bottom node conducts to (Noah's ZBOT).
LAND_ICE_LAYER_THICKNESS_M = (0.05, 0.25, 0.70, 1.00)
LAND_ICE_BOTTOM_DEPTH_M = 8.0
#: Sea ice: the analysed thickness is floored and capped before it is
#: split over the nodes with the snow on it (a column at or above the
#: ice threshold with a thickness of zero, the analysis's thickness
#: plane lagging its fraction plane, gets the floor rather than an
#: infinite conductance); the skin node is at most this thick, the rest
#: is split evenly.  Snow deeper than the cap is read as the cap.
MIN_SEA_ICE_THICKNESS_M = 0.10
MAX_SEA_ICE_THICKNESS_M = 5.0
SEA_ICE_SKIN_LAYER_MAX_M = 0.10
MAX_SNOW_DEPTH_M = 5.0
#: Snow shallower than this on the pack keeps the bare-ice node layout
#: (the snow-following layout would give the skin node half the snow,
#: less thermal inertia than the diurnal damping depth of snow; the
#: Arctic pack's 3 to 5 cm of September snow read 0.17 K colder at 2 m
#: with a 2 cm skin node); the midpoint rule still reads a shallower
#: cover as snow in the skin node where it reaches the node's midpoint.
SNOW_LAYER_MIN_M = 0.10
#: Latent heat of sublimation (the frozen surface's latent flux is
#: vapour to ice), Stefan-Boltzmann constant.
LATENT_HEAT_SUBLIMATION_J_KG = 2.834e6
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
#: Saturation vapour pressure over ice (Buck 1981, Pa) and the dry-air
#: to vapour molecular mass ratio the surface layer's qsfc is built with.
_BUCK_ICE_A = 611.15
_BUCK_ICE_B = 22.452
_BUCK_ICE_C = 272.55
EPSILON = 0.622


def frozen_columns(xland, sea_ice, landuse_category, ice_category: int, xp):
    """Boolean plane of the columns Noah skips as frozen: land (xland
    below 1.5) that is sea ice (fraction at or above the threshold the
    kernel applies, 0.5) or carries the land-ice class."""
    land = xp.asarray(xland, dtype=xp.float32) < xp.float32(1.5)
    seaice = xp.asarray(sea_ice, dtype=xp.float32) >= xp.float32(0.5)
    landice = xp.rint(xp.asarray(landuse_category, dtype=xp.float32)) == xp.float32(int(ice_category))
    return land & (seaice | landice)


def _snow_depth(snow_depth_m, xp):
    return xp.clip(
        xp.asarray(snow_depth_m, dtype=xp.float32), xp.float32(0.0), xp.float32(MAX_SNOW_DEPTH_M)
    )


def sea_ice_layer_thickness(thickness_m, snow_depth_m, xp):
    """The four node thicknesses (m, shape ``(4, ...)``) of a sea-ice
    column: the snow on it plus the analysed ice thickness (floored and
    capped).  With snow at least :data:`SNOW_LAYER_MIN_M` deep the nodes
    follow the snow: a skin node of half the snow, at most
    :data:`SEA_ICE_SKIN_LAYER_MAX_M`, the rest of the snow as the second
    node, and the ice split evenly over the two nodes below, so the whole
    snow depth insulates the ice as WRF's SFLX_SEAICE snowpack does (the
    earlier layout put a third of the ice in the second node and read a
    0.25 m snow cover as 0.10 m of snow over ice: the merged tip's full
    Antarctic pack drew 20 W/m2 up through its snow against the
    reference's 15).  Bare ice keeps the skin node of at most the cap and
    the rest split evenly over the three nodes below."""
    ice = xp.clip(
        xp.asarray(thickness_m, dtype=xp.float32),
        xp.float32(MIN_SEA_ICE_THICKNESS_M), xp.float32(MAX_SEA_ICE_THICKNESS_M),
    )
    snow = _snow_depth(snow_depth_m, xp)
    snowy = snow >= xp.float32(SNOW_LAYER_MIN_M)
    # bare ice: the skin node at most the cap, the rest split evenly
    h = ice + snow
    top_bare = xp.minimum(h / xp.float32(COLUMN_NODES), xp.float32(SEA_ICE_SKIN_LAYER_MAX_M))
    rest_bare = (h - top_bare) / xp.float32(COLUMN_NODES - 1)
    # snow-covered ice: the nodes follow the snow
    top_snow = xp.minimum(xp.float32(0.5) * snow, xp.float32(SEA_ICE_SKIN_LAYER_MAX_M))
    second_snow = snow - top_snow
    half_ice = xp.float32(0.5) * ice
    dz = xp.stack([
        xp.where(snowy, top_snow, top_bare),
        xp.where(snowy, second_snow, rest_bare),
        xp.where(snowy, half_ice, rest_bare),
        xp.where(snowy, half_ice, rest_bare),
    ])
    return dz.astype(xp.float32)


def land_ice_layer_thickness(like, xp):
    """The four node thicknesses (m, shape ``(4, ...)``) of a land-ice
    column, :data:`LAND_ICE_LAYER_THICKNESS_M` on every column of ``like``."""
    plane = xp.asarray(like, dtype=xp.float32)
    return xp.stack([
        xp.full_like(plane, xp.float32(dz)) for dz in LAND_ICE_LAYER_THICKNESS_M
    ]).astype(xp.float32)


def column_properties(dz, snow_depth_m, xp):
    """Per-node conductivity (W m-1 K-1) and volumetric heat capacity
    (J m-3 K-1), shape ``(4, ...)``: snow where the node's midpoint lies
    within the snow depth, ice below."""
    dz = xp.asarray(dz, dtype=xp.float32)
    midpoint = xp.cumsum(dz, axis=0) - xp.float32(0.5) * dz
    snow = midpoint < _snow_depth(snow_depth_m, xp)[None]
    conductivity = xp.where(
        snow, xp.float32(SNOW_CONDUCTIVITY_W_M_K), xp.float32(ICE_CONDUCTIVITY_W_M_K)
    ).astype(xp.float32)
    capacity = xp.where(
        snow, xp.float32(SNOW_VOLUMETRIC_HEAT_CAPACITY_J_M3_K),
        xp.float32(ICE_VOLUMETRIC_HEAT_CAPACITY_J_M3_K),
    ).astype(xp.float32)
    return conductivity, capacity


def initial_sea_ice_column(skin_k, thickness_m, snow_depth_m, xp):
    """The four node temperatures a sea-ice column starts from: linear in
    depth from the analysed skin at the surface to the freezing point at
    the ice bottom, at the node midpoints of :func:`sea_ice_layer_thickness`."""
    dz = sea_ice_layer_thickness(thickness_m, snow_depth_m, xp)
    h = dz.sum(axis=0)
    skin = xp.asarray(skin_k, dtype=xp.float32)
    bottom = xp.float32(SEA_ICE_BOTTOM_K)
    depth = xp.cumsum(dz, axis=0) - xp.float32(0.5) * dz
    return (skin[None] + (bottom - skin)[None] * depth / h[None]).astype(xp.float32)


def blended_surface_temperature(ice_skin_k, sea_ice_fraction, xp):
    """The skin a partial pack presents to the air: the ice skin weighted
    by the analysed fraction, open water at the freezing point for the
    rest (a fraction of one is the ice skin itself)."""
    f = xp.clip(xp.asarray(sea_ice_fraction, dtype=xp.float32), xp.float32(0.0), xp.float32(1.0))
    skin = xp.asarray(ice_skin_k, dtype=xp.float32)
    return (f * skin + (xp.float32(1.0) - f) * xp.float32(SEA_ICE_BOTTOM_K)).astype(xp.float32)


def ice_skin_from_composite(composite_k, sea_ice_fraction, xp, threshold: float = 0.5):
    """The inverse of :func:`blended_surface_temperature` on the columns
    the pack freezes over: the ice skin whose blend with open water at the
    freezing point is the analysed composite skin (the GFS surface
    temperature over a partial pack is that composite), capped at the
    melting point; a column below the threshold is returned as given."""
    f = xp.clip(xp.asarray(sea_ice_fraction, dtype=xp.float32), xp.float32(0.0), xp.float32(1.0))
    skin = xp.asarray(composite_k, dtype=xp.float32)
    frozen = f >= xp.float32(threshold)
    safe = xp.where(frozen, f, xp.float32(1.0))
    ice = (skin - (xp.float32(1.0) - safe) * xp.float32(SEA_ICE_BOTTOM_K)) / safe
    ice = xp.minimum(ice, xp.float32(MELT_POINT_K))
    return xp.where(frozen, ice, skin).astype(xp.float32)


def composite_albedo(ice_albedo, sea_ice_fraction, xp):
    """The albedo the atmosphere sees over a partial pack, WRF's
    fractional-sea-ice blend with open water at 0.08."""
    f = xp.clip(xp.asarray(sea_ice_fraction, dtype=xp.float32), xp.float32(0.0), xp.float32(1.0))
    a = xp.asarray(ice_albedo, dtype=xp.float32)
    return (f * a + (xp.float32(1.0) - f) * xp.float32(OPEN_WATER_ALBEDO)).astype(xp.float32)


def composite_emissivity(ice_emissivity, sea_ice_fraction, xp):
    """The emissivity the atmosphere sees over a partial pack, WRF's
    fractional-sea-ice blend with open water at 0.98."""
    f = xp.clip(xp.asarray(sea_ice_fraction, dtype=xp.float32), xp.float32(0.0), xp.float32(1.0))
    e = xp.asarray(ice_emissivity, dtype=xp.float32)
    return (f * e + (xp.float32(1.0) - f) * xp.float32(OPEN_WATER_EMISSIVITY)).astype(xp.float32)


def column_step(*, layers, dz, snow_depth_m, bottom_k, bottom_extra_resistance,
                swdown, albedo, glw, emissivity, hfx, qfx, dt_s, xp,
                exchange_heat_w_m2_k=None, exchange_moisture_kg_m2_s=None,
                skin_reference_k=None, psfc_pa=None):
    """One implicit step of the four-node column; returns the new nodes
    (float32, shape ``(4, ...)``), the first of which is the skin.

    ``layers`` and ``dz`` are ``(4, ...)`` node temperatures (K) and
    thicknesses (m); ``snow_depth_m`` says which nodes are snow (see
    :func:`column_properties`); ``bottom_k`` is the temperature the
    bottom node conducts to through half its own thickness plus
    ``bottom_extra_resistance`` (m2 K W-1: zero for sea ice, the firn
    between 2 m and Noah's 8 m for land ice).  ``hfx`` W m-2 and ``qfx``
    kg m-2 s-1 are the surface layer's fluxes, positive upward (WRF
    HFX/QFX), evaluated at ``skin_reference_k`` (the skin the surface
    layer saw: the blended skin on a partial pack; the old skin itself
    when None).  ``exchange_heat_w_m2_k`` (rho c_p C_h, W m-2 K-1) and
    ``exchange_moisture_kg_m2_s`` (rho m C_h, kg m-2 s-1) are the surface
    layer's own flux slopes: the ice skin's fluxes are the cell's plus
    the slope times the skin difference, implicit in the new skin (the
    saturation humidity over ice needs ``psfc_pa``).  With no slopes the
    fluxes are taken as given, explicit.  Longwave emission is linearised
    about the old skin; every node is capped at the melting point.
    """
    f32 = xp.float32
    t = xp.asarray(layers, dtype=xp.float32)
    dz = xp.asarray(dz, dtype=xp.float32)
    conductivity, volumetric = column_properties(dz, snow_depth_m, xp)
    inv_dt = f32(1.0 / float(dt_s))
    capacity = (volumetric * dz) * inv_dt  # C_k / dt
    # Conductances between consecutive nodes, W m-2 K-1: each node's half
    # thickness over its own conductivity, in series.
    half = f32(0.5) * dz / conductivity
    g = [
        f32(1.0) / (half[0] + half[1]),
        f32(1.0) / (half[1] + half[2]),
        f32(1.0) / (half[2] + half[3]),
    ]
    g_bottom = f32(1.0) / (half[3] + xp.asarray(bottom_extra_resistance, dtype=xp.float32))
    # Surface forcing at the old skin, and the emission slope.
    t0 = t[0]
    eps = xp.asarray(emissivity, dtype=xp.float32)
    sigma = f32(STEFAN_BOLTZMANN_W_M2_K4)
    t0_3 = t0 * t0 * t0
    slope = f32(4.0) * eps * sigma * t0_3
    hfx = xp.asarray(hfx, dtype=xp.float32)
    qfx = xp.asarray(qfx, dtype=xp.float32)
    ls = f32(LATENT_HEAT_SUBLIMATION_J_KG)
    # The ice skin's own fluxes: the cell's, corrected by the surface
    # layer's slopes for the skin difference at the old skin (explicit
    # part) and for the step's change (implicit part, on the diagonal).
    coupling = f32(0.0)
    if exchange_heat_w_m2_k is not None:
        rcc = xp.asarray(exchange_heat_w_m2_k, dtype=xp.float32)
        tref = t0 if skin_reference_k is None else xp.asarray(skin_reference_k, dtype=xp.float32)
        hfx = hfx + rcc * (t0 - tref)
        coupling = coupling + rcc
    if exchange_moisture_kg_m2_s is not None:
        if psfc_pa is None:
            raise ValueError("the moisture exchange slope needs the surface pressure for the saturation over ice")
        rqc = xp.asarray(exchange_moisture_kg_m2_s, dtype=xp.float32)
        tref = t0 if skin_reference_k is None else xp.asarray(skin_reference_k, dtype=xp.float32)
        qfx = qfx + rqc * (
            saturation_specific_humidity_over_ice(t0, psfc_pa, xp)
            - saturation_specific_humidity_over_ice(tref, psfc_pa, xp)
        )
        coupling = coupling + ls * rqc * saturation_slope_over_ice(t0, psfc_pa, xp)
    net = (
        xp.asarray(swdown, dtype=xp.float32) * (f32(1.0) - xp.asarray(albedo, dtype=xp.float32))
        + eps * xp.asarray(glw, dtype=xp.float32)
        - eps * sigma * t0_3 * t0
        - hfx
        - ls * qfx
    )
    tb = xp.asarray(bottom_k, dtype=xp.float32)
    # Tridiagonal system a_k T_{k-1} + b_k T_k + c_k T_{k+1} = d_k.
    a = [None, -g[0], -g[1], -g[2]]
    b = [
        capacity[0] + slope + coupling + g[0],
        capacity[1] + g[0] + g[1],
        capacity[2] + g[1] + g[2],
        capacity[3] + g[2] + g_bottom,
    ]
    c = [-g[0], -g[1], -g[2], None]
    d = [
        capacity[0] * t0 + net + (slope + coupling) * t0,
        capacity[1] * t[1],
        capacity[2] * t[2],
        capacity[3] * t[3] + g_bottom * tb,
    ]
    # Thomas algorithm, vectorised over the grid.
    cp = [None] * COLUMN_NODES
    dp = [None] * COLUMN_NODES
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, COLUMN_NODES):
        denom = b[k] - a[k] * cp[k - 1]
        cp[k] = (c[k] / denom) if c[k] is not None else None
        dp[k] = (d[k] - a[k] * dp[k - 1]) / denom
    out = [None] * COLUMN_NODES
    out[-1] = dp[-1]
    for k in range(COLUMN_NODES - 2, -1, -1):
        out[k] = dp[k] - cp[k] * out[k + 1]
    new = xp.stack(out)
    return xp.minimum(new, f32(MELT_POINT_K)).astype(xp.float32)


def skin_conduction_w_m2(layers, dz, snow_depth_m, xp):
    """Conductive heat flux from the skin node into the node below it,
    W m-2 positive downward: the frozen column's G, formed with the same
    node thicknesses and media as :func:`column_step` (the surface-energy
    instrument's reading of a checkpoint, not a term the step uses)."""
    t = xp.asarray(layers, dtype=xp.float32)
    dz = xp.asarray(dz, dtype=xp.float32)
    conductivity, _ = column_properties(dz, snow_depth_m, xp)
    half = xp.float32(0.5) * dz / conductivity
    g01 = xp.float32(1.0) / (half[0] + half[1])
    return (g01 * (t[0] - t[1])).astype(xp.float32)


def saturation_specific_humidity_over_ice(tsk, psfc_pa, xp):
    """q_sat over ice at the skin, kg kg-1 (Buck 1981 over ice), the
    value the surface layer's qsfc is refreshed with on frozen columns."""
    t = xp.asarray(tsk, dtype=xp.float32)
    p = xp.asarray(psfc_pa, dtype=xp.float32)
    celsius = t - xp.float32(273.15)
    es = xp.float32(_BUCK_ICE_A) * xp.exp(
        xp.float32(_BUCK_ICE_B) * celsius / (celsius + xp.float32(_BUCK_ICE_C))
    )
    es = xp.minimum(es, xp.float32(0.9) * p)
    return (xp.float32(EPSILON) * es / (p - (xp.float32(1.0) - xp.float32(EPSILON)) * es)).astype(xp.float32)


def saturation_slope_over_ice(tsk, psfc_pa, xp):
    """d q_sat / dT over ice at the skin, kg kg-1 K-1: the derivative of
    :func:`saturation_specific_humidity_over_ice` (Buck 1981 over ice),
    the slope the implicit latent coupling of the skin is built with."""
    t = xp.asarray(tsk, dtype=xp.float32)
    p = xp.asarray(psfc_pa, dtype=xp.float32)
    celsius = t - xp.float32(273.15)
    b, c = xp.float32(_BUCK_ICE_B), xp.float32(_BUCK_ICE_C)
    es = xp.float32(_BUCK_ICE_A) * xp.exp(b * celsius / (celsius + c))
    capped = es > xp.float32(0.9) * p
    des = es * b * c / ((celsius + c) * (celsius + c))
    eps = xp.float32(EPSILON)
    denominator = p - (xp.float32(1.0) - eps) * es
    dq = eps * p * des / (denominator * denominator)
    return xp.where(capped, xp.float32(0.0), dq).astype(xp.float32)


__all__ = [
    "COLUMN_NODES",
    "ICE_CONDUCTIVITY_W_M_K",
    "ICE_VOLUMETRIC_HEAT_CAPACITY_J_M3_K",
    "LAND_ICE_BOTTOM_DEPTH_M",
    "LAND_ICE_LAYER_THICKNESS_M",
    "MAX_SEA_ICE_THICKNESS_M",
    "MAX_SNOW_DEPTH_M",
    "MELT_POINT_K",
    "MIN_SEA_ICE_THICKNESS_M",
    "OPEN_WATER_ALBEDO",
    "OPEN_WATER_EMISSIVITY",
    "SEA_ICE_ALBEDO",
    "SEA_ICE_BOTTOM_K",
    "SEA_ICE_EMISSIVITY",
    "SEA_ICE_SKIN_LAYER_MAX_M",
    "SNOW_COVERED_KG_M2",
    "SNOW_LAYER_MIN_M",
    "SNOW_CONDUCTIVITY_W_M_K",
    "SNOW_VOLUMETRIC_HEAT_CAPACITY_J_M3_K",
    "blended_surface_temperature",
    "column_properties",
    "column_step",
    "composite_albedo",
    "composite_emissivity",
    "frozen_columns",
    "ice_skin_from_composite",
    "initial_sea_ice_column",
    "land_ice_layer_thickness",
    "saturation_slope_over_ice",
    "saturation_specific_humidity_over_ice",
    "sea_ice_layer_thickness",
    "skin_conduction_w_m2",
]
