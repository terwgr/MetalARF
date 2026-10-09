"""One explicit Metal caller slice for the Hex CPU preview driver.

This module offloads only MPAS horizontal mass-flux divergence.  Mesh fields
and ``rho_u`` are uploaded for the call, the cell tendency is read back, and
the rest of the preview integration remains NumPy.  It therefore provides a
real opt-in dispatch boundary, not a GPU-resident forecast claim.

The launcher requires the generated Metal registry and a bundle containing
the matching shader; it refuses by capability if the entrypoint is absent.
"""

from __future__ import annotations

import numpy as np

from .errors import ConfigurationRefusal


ENTRYPOINT = "hex_mass_flux_divergence"
THREADS_PER_GROUP = 128
_HEX_ACOUSTIC_EDGE_UPDATE = "hex_acoustic_edge_update"
_HEX_THETA_DIAGNOSTIC = "hex_theta_diagnostic"
_HEX_EXNER_PRESSURE_DIAGNOSTIC = "hex_exner_pressure_diagnostic"


def submit_mass_flux_divergence(
    context,
    *,
    rho_u,
    edge_indices,
    edge_counts,
    edge_signs,
    dv_edge,
    area_cell,
    inv_area_cell=None,
):
    """Dispatch one FP32 finite-volume divergence and return host FP32 data.

    The row-wise accumulation order is the source loop order in
    ``dynamics.mass_flux_divergence``.  FP64 meshes, extents that cannot fit
    the shader's signed parameters, and an unregistered source fragment are
    refused explicitly; no precision conversion fallback is made.
    """
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext

    if not isinstance(context, MetalContext):
        raise TypeError("Hex preview Metal slice requires a MetalContext")
    from .metal_admission import require_hex_metal

    admission = require_hex_metal(context)
    rho_u = np.asarray(rho_u)
    edge_indices = np.asarray(edge_indices)
    edge_counts = np.asarray(edge_counts)
    edge_signs = np.asarray(edge_signs)
    dv_edge = np.asarray(dv_edge)
    area_cell = np.asarray(area_cell)
    if rho_u.dtype != np.dtype(np.float32):
        raise ConfigurationRefusal(
            "hex_metal_precision", str(rho_u.dtype),
            "the first Hex Metal slice has only an FP32 typed kernel; a cast "
            "would change the preview driver's RKIND arithmetic",
            "an FP32 Hex preview state or a source-faithful FP64 Metal port",
        )
    if rho_u.ndim != 2 or edge_indices.ndim != 2:
        raise ValueError("Hex mass-flux inputs must be level/edge and cell/slot arrays")
    nlev, nedges = rho_u.shape
    ncells, max_edges = edge_indices.shape
    if (edge_counts.shape != (ncells,) or edge_signs.shape != edge_indices.shape
            or dv_edge.shape != (nedges,) or area_cell.shape != (ncells,)):
        raise ValueError("Hex mass-flux mesh fields have inconsistent dimensions")
    if (max(nlev, nedges, ncells, max_edges) > 0x7FFFFFFF
            or nlev * ncells > 0xFFFFFFFF
            or min(nlev, nedges, ncells, max_edges) <= 0):
        raise ConfigurationRefusal(
            "hex_metal_extent", (nlev, ncells, nedges, max_edges),
            "the preview slice uses int32 shape parameters and a uint32 grid",
            "mesh extents fitting int32 parameters and a uint32 launch grid",
        )
    if np.any(edge_counts < 0) or np.any(edge_counts > max_edges):
        raise ValueError("Hex edge counts are outside edgesOnCell row bounds")
    used_slots = np.arange(max_edges)[None, :] < edge_counts[:, None]
    used_edges = edge_indices[used_slots]
    if np.any(used_edges < 0) or np.any(used_edges >= nedges):
        raise ValueError("Hex mass-flux used edge slots cannot contain sentinels")

    use_inverse = int(inv_area_cell is not None)
    if inv_area_cell is None:
        inv_area_cell = np.zeros((ncells,), dtype=np.float32)
    else:
        inv_area_cell = np.asarray(inv_area_cell)
        if inv_area_cell.shape != (ncells,) or inv_area_cell.dtype != rho_u.dtype:
            raise ValueError("Hex inverse cell area must match FP32 cell shape")

    from woof.metal.generated import kernel_specs

    if not hasattr(kernel_specs, f"{ENTRYPOINT.upper()}_RESOURCES"):
        raise ConfigurationRefusal(
            "hex_metal_kernel", ENTRYPOINT,
            "the Hex preview shader and signature fragment are present, but "
            "the shared generated Metal registry does not yet include this "
            "entry point",
            "use or build a Metal bundle containing the generated Hex entry "
            "point and its shader",
        )

    xp = MetalArrayModule(context)
    arrays = {
        "rho_u": xp.asarray(rho_u, dtype=np.float32),
        "edge_indices": xp.asarray(edge_indices, dtype=np.int32),
        "edge_counts": xp.asarray(edge_counts, dtype=np.int32),
        "edge_signs": xp.asarray(edge_signs, dtype=np.float32),
        "dv_edge": xp.asarray(dv_edge, dtype=np.float32),
        "area_cell": xp.asarray(area_cell, dtype=np.float32),
        "inv_area_cell": xp.asarray(inv_area_cell, dtype=np.float32),
        "divergence": xp.empty((nlev, ncells), dtype=np.float32),
    }
    params = {
        "nlev": int(nlev), "ncells": int(ncells),
        "nedges": int(nedges), "max_edges": int(max_edges),
        "use_inverse": use_inverse,
    }
    try:
        context.submit_kernel(
            ENTRYPOINT, arrays, params, exact_policy=False,
            grid_size=nlev * ncells,
            threads_per_group=THREADS_PER_GROUP,
        )
    except ValueError as error:
        if f"unregistered Metal kernel entry point {ENTRYPOINT!r}" not in str(error):
            raise
        raise ConfigurationRefusal(
            "hex_metal_kernel", ENTRYPOINT,
            "the Hex preview shader and signature fragment are present, but "
            "the shared generated Metal registry does not yet include this "
            "entry point",
            "use or build a Metal bundle containing the generated Hex entry "
            "point and its shader",
        ) from error
    # Keep the most recent Hex-specific admission attached to this context so
    # a caller building an execution receipt can record the exact device and
    # verified runtime/shader identity.  This does not change the numerical
    # return contract of the preview helper.
    context.hex_last_admission = admission.as_dict()
    return arrays["divergence"].to_numpy()


def submit_acoustic_edge_update(
    context, *, ru_p, ru_avg, tend_ru, rtheta_pp, rho_pp, zz, exner, zxu,
    cqu, cells_on_edge, dc_edge, specified_zone_edge, small_step, dts,
    c2, gravity,
):
    """Dispatch the FP32 edge-owned acoustic pressure/momentum update."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    arrays_host = {
        "ru_p": np.asarray(ru_p), "ru_avg": np.asarray(ru_avg),
        "tend_ru": np.asarray(tend_ru),
        "rtheta_pp": np.asarray(rtheta_pp), "rho_pp": np.asarray(rho_pp),
        "zz": np.asarray(zz), "exner": np.asarray(exner),
        "zxu": np.asarray(zxu), "cqu": np.asarray(cqu),
        "cells_on_edge": np.asarray(cells_on_edge),
        "dc_edge": np.asarray(dc_edge),
        "specified_zone_edge": np.asarray(specified_zone_edge),
    }
    cells_host = arrays_host["cells_on_edge"]
    if cells_host.dtype != np.dtype(np.int32):
        if (np.any(cells_host < np.iinfo(np.int32).min)
                or np.any(cells_host > np.iinfo(np.int32).max)):
            raise ValueError("Hex acoustic cell indices exceed int32 ABI")
        arrays_host["cells_on_edge"] = cells_host.astype(np.int32)
    nlev, nedges = arrays_host["ru_p"].shape
    ncells = arrays_host["rho_pp"].shape[1]
    for name in ("ru_p", "ru_avg", "tend_ru", "zxu", "cqu"):
        if arrays_host[name].shape != (nlev, nedges):
            raise ValueError(f"Hex acoustic {name} shape mismatch")
    for name in ("rtheta_pp", "rho_pp", "zz", "exner"):
        if arrays_host[name].shape != (nlev, ncells):
            raise ValueError(f"Hex acoustic {name} shape mismatch")
    if (arrays_host["cells_on_edge"].shape != (nedges, 2)
            or arrays_host["dc_edge"].shape != (nedges,)
            or arrays_host["specified_zone_edge"].shape != (nedges,)):
        raise ValueError("Hex acoustic edge geometry shape mismatch")
    for name, value in arrays_host.items():
        expected = np.dtype(np.int32 if name == "cells_on_edge" else np.float32)
        if value.dtype != expected:
            raise TypeError(f"Hex acoustic {name} must be {expected}")
    metal = MetalArrayModule(context)
    arrays = {name: metal.asarray(np.ascontiguousarray(value))
              for name, value in arrays_host.items()}
    resources = {
        **arrays,
    }
    context.submit_kernel(
        _HEX_ACOUSTIC_EDGE_UPDATE, resources,
        {"nlev": int(nlev), "ncells": int(ncells),
         "nedges": int(nedges), "small_step": int(small_step),
         "dts": float(dts), "c2": float(c2), "gravity": float(gravity)},
        exact_policy=False, grid_size=int(nedges),
        threads_per_group=kernel_specs.HEX_ACOUSTIC_EDGE_UPDATE_THREADS_PER_GROUP)
    return arrays["ru_p"].to_numpy(), arrays["ru_avg"].to_numpy()


def submit_acoustic_postsolve_damping(
    context, *, rw_p, rw_save, rw, dss, w, rho_zz, zz, fzm, fzp, dts,
):
    """Dispatch the interface-velocity damping after the vertical solve."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    values = {name: np.asarray(value, dtype=np.float32) for name, value in {
        "rw_p": rw_p, "rw_save": rw_save, "rw": rw, "dss": dss,
        "w": w, "rho_zz": rho_zz, "zz": zz, "fzm": fzm, "fzp": fzp,
    }.items()}
    nlev, ncells = values["rho_zz"].shape
    for name in ("dss", "w"):
        if values[name].shape == (nlev, ncells):
            values[name] = np.pad(values[name], ((0, 1), (0, 0)))
    for name in ("fzm", "fzp"):
        if values[name].shape == (nlev,):
            values[name] = np.pad(values[name], (0, 1))
    if (nlev < 2 or values["zz"].shape != (nlev, ncells)
            or any(values[name].shape != (nlev + 1, ncells)
                   for name in ("rw_p", "rw_save", "rw", "dss", "w"))
            or values["fzm"].shape != (nlev + 1,)
            or values["fzp"].shape != (nlev + 1,)):
        raise ValueError("Hex acoustic postsolve damping arrays mismatch: "
                         + str({name: value.shape
                                for name, value in values.items()}))
    metal = MetalArrayModule(context)
    arrays = {name: metal.asarray(np.ascontiguousarray(value))
              for name, value in values.items()}
    context.submit_kernel(
        "hex_acoustic_postsolve_damping", arrays,
        {"nlev": int(nlev), "ncells": int(ncells), "dts": float(dts)},
        exact_policy=False, grid_size=(nlev - 1) * ncells,
        threads_per_group=kernel_specs.HEX_ACOUSTIC_POSTSOLVE_DAMPING_THREADS_PER_GROUP)
    return arrays["rw_p"].to_numpy()


def submit_pressure_diagnostics(context, *, rho_theta, rho, zz, rgas,
                                cp, reference_pressure):
    """Dispatch the integrated theta, Exner and pressure diagnostic stages."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    rho_theta = np.ascontiguousarray(np.asarray(rho_theta))
    rho = np.ascontiguousarray(np.asarray(rho))
    zz = np.ascontiguousarray(np.asarray(zz))
    if rho_theta.dtype != np.float32 or rho.dtype != np.float32:
        raise ConfigurationRefusal(
            "hex_metal_precision", (rho.dtype, rho_theta.dtype),
            "integrated Hex pressure diagnostics currently have FP32 resources",
            "an FP32 Hex preview state or a typed FP64 Metal diagnostic",
        )
    nlev, ncells = rho.shape
    if zz.shape == rho.shape:
        if not np.all(zz == zz[:, :1]):
            raise ConfigurationRefusal(
                "hex_metal_vertical_coordinate", zz.shape,
                "the integrated diagnostic ABI has one zz value per level",
                "a Hex mesh with level-uniform zz or a cellwise zz kernel",
            )
        zz = np.ascontiguousarray(zz[:, 0])
    if rho_theta.shape != rho.shape or zz.shape != (nlev,):
        raise ValueError("Hex pressure diagnostic inputs have inconsistent shapes")
    xp = MetalArrayModule(context)
    theta = xp.empty(rho.shape, dtype=np.float32)
    exner = xp.empty(rho.shape, dtype=np.float32)
    pressure = xp.empty(rho.shape, dtype=np.float32)
    fields = {"rho_theta": xp.asarray(rho_theta), "rho": xp.asarray(rho),
              "theta_m": theta}
    context.submit_kernel(
        _HEX_THETA_DIAGNOSTIC, fields, {"nlev": nlev, "ncells": ncells},
        exact_policy=False, grid_size=nlev * ncells,
        threads_per_group=kernel_specs.HEX_THETA_DIAGNOSTIC_THREADS_PER_GROUP,
    )
    fields = {"exner": exner, "pressure": pressure,
              "zz": xp.asarray(zz), "rho_theta": xp.asarray(rho_theta)}
    context.submit_kernel(
        _HEX_EXNER_PRESSURE_DIAGNOSTIC, fields,
        {"nlev": nlev, "ncells": ncells,
         "rgas_over_reference_pressure": float(rgas / reference_pressure),
         "exner_exponent": float(rgas / (cp - rgas)), "rgas": float(rgas)},
        exact_policy=False, grid_size=nlev * ncells,
        threads_per_group=kernel_specs.HEX_EXNER_PRESSURE_DIAGNOSTIC_THREADS_PER_GROUP,
    )
    return pressure.to_numpy(), exner.to_numpy(), theta.to_numpy()


def submit_vertical_scalar_flux3(context, *, q_im2, q_im1, q_i, q_ip1,
                                 vertical_velocity, coefficient):
    """Dispatch contiguous level/cell flux3 stencils to the integrated kernel."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    values = {name: np.ascontiguousarray(np.asarray(value)) for name, value in {
        "q_im2": q_im2, "q_im1": q_im1, "q_i": q_i,
        "q_ip1": q_ip1, "vertical_velocity": vertical_velocity,
    }.items()}
    shape = values["q_i"].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("Hex flux3 stencil fields must share a shape")
    if any(value.dtype != np.dtype(np.float32) for value in values.values()):
        raise TypeError("Hex vertical scalar flux3 currently requires FP32")
    count = int(values["q_i"].size)
    xp = MetalArrayModule(context)
    resources = {name: xp.asarray(value.reshape(-1))
                 for name, value in values.items()}
    resources["flux"] = xp.empty((count,), dtype=np.float32)
    context.submit_kernel(
        "hex_vertical_scalar_flux3", resources,
        {"nitems": count, "third_order_coefficient": float(coefficient)},
        exact_policy=False, grid_size=count,
        threads_per_group=kernel_specs.HEX_VERTICAL_SCALAR_FLUX3_THREADS_PER_GROUP,
    )
    return resources["flux"].to_numpy().reshape(shape)


def submit_vertical_momentum_transport(context, *, normal_velocity, rho_w,
                                       cells_on_edge, fzm, fzp, rdzw,
                                       coefficient=1.0):
    """Submit the vertical normal-momentum flux divergence on Metal."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    u = np.asarray(normal_velocity)
    rw = np.asarray(rho_w)
    edges = np.asarray(cells_on_edge)
    fzm = np.asarray(fzm)
    fzp = np.asarray(fzp)
    rdzw = np.asarray(rdzw)
    if u.dtype != np.dtype(np.float32) or u.ndim != 2:
        raise TypeError("Hex Metal vertical momentum expects an FP32 2-D velocity")
    nlev, nedges = map(int, u.shape)
    if (rw.dtype != u.dtype or rw.ndim != 2 or rw.shape[0] != nlev + 1
            or edges.shape != (nedges, 2)
            or fzm.shape != (nlev,) or fzp.shape != (nlev,)
            or rdzw.shape != (nlev,)):
        raise ValueError("Hex vertical momentum inputs have incompatible extents")
    ncells = int(rw.shape[1])
    module = MetalArrayModule(context)
    resources = {
        "normal_velocity": module.asarray(np.ascontiguousarray(u).reshape(-1)),
        "rho_w": module.asarray(np.ascontiguousarray(rw).reshape(-1)),
        "cells_on_edge": module.asarray(np.ascontiguousarray(
            edges, dtype=np.int32).reshape(-1)),
        "fzm": module.asarray(np.ascontiguousarray(fzm, dtype=np.float32)),
        "fzp": module.asarray(np.ascontiguousarray(fzp, dtype=np.float32)),
        "rdzw": module.asarray(np.ascontiguousarray(rdzw, dtype=np.float32)),
        "tendency": module.empty((nlev * nedges,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_vertical_momentum_transport_fragment", resources,
        {"nlev": nlev, "nedges": nedges, "ncells": ncells,
         "coefficient": float(coefficient)},
        exact_policy=False, grid_size=nlev * nedges,
        threads_per_group=(
            kernel_specs.HEX_VERTICAL_MOMENTUM_TRANSPORT_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["tendency"].to_numpy().reshape(nlev, nedges)


def submit_vertical_velocity_tendency(
    context, *, normal_velocity, rho_w, rho_u, edges_on_cell,
    n_edges_on_cell, edge_signs, adv_coefs, adv_coefs_3rd,
    adv_cells_for_edge, n_adv_cells_for_edge, fzm, fzp, rdzu,
    area_cell, inv_area_cell=None, skip_cells=None, coefficient=0.25,
):
    """Submit the complete horizontal-plus-vertical w tendency on Metal."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    floating = {name: np.asarray(value) for name, value in {
        "normal_velocity": normal_velocity, "rho_w": rho_w,
        "rho_u": rho_u, "edge_signs": edge_signs,
        "adv_coefs": adv_coefs, "adv_coefs_3rd": adv_coefs_3rd,
        "fzm": fzm, "fzp": fzp, "rdzu": rdzu,
        "area_cell": area_cell,
    }.items()}
    integers = {name: np.asarray(value) for name, value in {
        "edges_on_cell": edges_on_cell,
        "n_edges_on_cell": n_edges_on_cell,
        "adv_cells_for_edge": adv_cells_for_edge,
        "n_adv_cells_for_edge": n_adv_cells_for_edge,
    }.items()}
    w = floating["normal_velocity"]
    if w.dtype != np.dtype(np.float32) or w.ndim != 2:
        raise TypeError("Hex Metal w tendency requires FP32 interface velocity")
    nlev = int(floating["rho_u"].shape[0])
    ncells = int(w.shape[1])
    nedges = int(floating["rho_u"].shape[1])
    max_edges = int(integers["edges_on_cell"].shape[1])
    max_adv = int(floating["adv_coefs"].shape[1])
    inv = (np.zeros(ncells, dtype=np.float32) if inv_area_cell is None
           else np.asarray(inv_area_cell))
    skip = (np.zeros(ncells, dtype=np.int32) if skip_cells is None
            else np.asarray(skip_cells, dtype=np.int32))
    if (w.shape != (nlev + 1, ncells)
            or floating["rho_w"].shape != w.shape
            or floating["rho_u"].shape != (nlev, nedges)
            or integers["edges_on_cell"].shape != (ncells, max_edges)
            or integers["n_edges_on_cell"].shape != (ncells,)
            or floating["edge_signs"].shape != (ncells, max_edges)
            or floating["adv_coefs"].shape != (nedges, max_adv)
            or floating["adv_coefs_3rd"].shape != (nedges, max_adv)
            or integers["adv_cells_for_edge"].shape != (nedges, max_adv)
            or integers["n_adv_cells_for_edge"].shape != (nedges,)
            or floating["fzm"].size < nlev
            or floating["fzp"].size < nlev
            or floating["rdzu"].size < nlev
            or floating["area_cell"].shape != (ncells,)
            or inv.shape != (ncells,) or skip.shape != (ncells,)):
        raise ValueError("Hex Metal w-tendency inputs have incompatible shapes")
    if any(value.dtype != np.dtype(np.float32) for value in floating.values()):
        raise TypeError("Hex Metal w-tendency floating inputs must be FP32")
    if any(value.dtype.kind not in "iu" for value in integers.values()):
        raise TypeError("Hex Metal w-tendency connectivity must be integer")

    module = MetalArrayModule(context)
    resources = {
        name: module.asarray(np.ascontiguousarray(value).reshape(-1))
        for name, value in floating.items()
    }
    resources.update({
        name: module.asarray(np.ascontiguousarray(value, dtype=np.int32).reshape(-1))
        for name, value in integers.items()
    })
    resources["inv_area_cell"] = module.asarray(np.ascontiguousarray(inv))
    resources["skip_cell"] = module.asarray(np.ascontiguousarray(skip))
    resources["tendency"] = module.empty(((nlev + 1) * ncells,), dtype=np.float32)
    context.submit_kernel(
        "hex_vertical_velocity_tendency_fragment", resources,
        {"nlev": nlev, "ncells": ncells, "nedges": nedges,
         "max_edges": max_edges, "max_adv": max_adv,
         "coefficient": float(coefficient),
         "use_inverse_area": int(inv_area_cell is not None)},
        exact_policy=False, grid_size=(nlev + 1) * ncells,
        threads_per_group=(
            kernel_specs.HEX_VERTICAL_VELOCITY_TENDENCY_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["tendency"].to_numpy().reshape(nlev + 1, ncells)


def submit_vertical_coefficients_v841(context, *, dts, gravity, rgas, cp,
                                      zz, cqw, exner, theta, rho_base,
                                      rho_theta_base, exner_base,
                                      rho_theta_perturbation, qtot, rdzw, fzm,
                                      fzp, rdzu, etp, ewp):
    """Assemble v8.4.1 acoustic coefficients and Thomas factors on Metal."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    from .acoustic_v841 import VerticalImplicitCoefficientsV841

    floating = {name: np.asarray(value) for name, value in {
        "zz": zz, "cqw": cqw, "exner": exner, "theta": theta,
        "rho_base": rho_base, "rho_theta_base": rho_theta_base,
        "exner_base": exner_base,
        "rho_theta_perturbation": rho_theta_perturbation, "qtot": qtot,
        "rdzw": rdzw, "fzm": fzm, "fzp": fzp, "rdzu": rdzu,
        "etp": etp, "ewp": ewp,
    }.items()}
    vertical = floating["zz"]
    if vertical.dtype != np.dtype(np.float32) or vertical.ndim != 2:
        raise TypeError("Hex v8.4.1 Metal coefficients require FP32 zz")
    nlev, ncells = map(int, vertical.shape)
    if any(value.dtype != np.dtype(np.float32) for value in floating.values()):
        raise TypeError("Hex v8.4.1 Metal coefficient inputs must be FP32")
    expected_shapes = {
        "cqw": (nlev, ncells), "exner": (nlev, ncells),
        "theta": (nlev, ncells), "rho_base": (nlev, ncells),
        "rho_theta_base": (nlev, ncells), "exner_base": (nlev, ncells),
        "rho_theta_perturbation": (nlev, ncells), "qtot": (nlev, ncells),
        "rdzw": (nlev,), "fzm": (nlev,), "fzp": (nlev,),
        "rdzu": (nlev,), "etp": (nlev,), "ewp": (nlev + 1,),
    }
    if any(floating[name].shape != shape
           for name, shape in expected_shapes.items()):
        raise ValueError("Hex v8.4.1 Metal coefficient inputs have invalid shapes")

    module = MetalArrayModule(context)
    resources = {
        name: module.asarray(np.ascontiguousarray(value).reshape(-1))
        for name, value in floating.items()
    }
    cell_shape = (nlev, ncells)
    resources.update({
        name: module.empty((int(np.prod(shape)),), dtype=np.float32)
        for name, shape in {
            "cofwr": cell_shape, "cofwz": cell_shape,
            "coftz": (nlev + 1, ncells), "cofwt": cell_shape,
            "cofrz": (nlev,), "a_tri": cell_shape, "b_tri": cell_shape,
            "c_tri": cell_shape, "alpha_tri": cell_shape,
            "gamma_tri": cell_shape,
        }.items()
    })
    resources["errors"] = module.zeros((ncells,), dtype=np.int32)
    context.submit_kernel(
        "hex_vertical_coefficients_v841_fragment", resources,
        {"nlev": nlev, "ncells": ncells, "dts": float(dts),
         "gravity": float(gravity), "rgas": float(rgas), "cp": float(cp)},
        exact_policy=False, grid_size=ncells,
        threads_per_group=(
            kernel_specs.HEX_VERTICAL_COEFFICIENTS_V841_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    errors = resources["errors"].to_numpy()
    singular = np.flatnonzero(errors)
    if singular.size:
        level = int(errors[int(singular[0])])
        raise FloatingPointError(
            f"singular vertical acoustic system at level {level}"
        )
    values = {
        name: resources[name].to_numpy().reshape(shape)
        for name, shape in {
            "cofwr": cell_shape, "cofwz": cell_shape,
            "coftz": (nlev + 1, ncells), "cofwt": cell_shape,
            "cofrz": (nlev,), "a_tri": cell_shape, "b_tri": cell_shape,
            "c_tri": cell_shape, "alpha_tri": cell_shape,
            "gamma_tri": cell_shape,
        }.items()
    }
    return VerticalImplicitCoefficientsV841(**values)


def submit_mesh_edge_diagnostics(context, *, rho, rho_u, cells_on_edge):
    """Recover edge density and normal velocity using the mesh Metal stage."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    density = np.asarray(rho)
    momentum = np.asarray(rho_u)
    cells = np.asarray(cells_on_edge)
    if density.dtype != np.dtype(np.float32) or momentum.dtype != density.dtype:
        raise TypeError("Hex Metal edge diagnostics require FP32 state")
    if density.ndim != 2 or momentum.ndim != 2:
        raise ValueError("Hex Metal edge diagnostics require 2-D state arrays")
    nlev, ncells = map(int, density.shape)
    if momentum.shape[0] != nlev or cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("Hex Metal edge diagnostics have incompatible extents")
    nedges = int(momentum.shape[1])
    if cells.shape != (nedges, 2):
        raise ValueError("cells_on_edge must have one pair per momentum edge")
    cells = np.asarray(cells, dtype=np.int32)
    if np.any(cells < 0) or np.any(cells >= ncells):
        raise ValueError("cells_on_edge references a cell outside the density field")
    count = nlev * nedges
    module = MetalArrayModule(context)
    resources = {
        "rho": module.asarray(np.ascontiguousarray(density).reshape(-1)),
        "rho_u": module.asarray(np.ascontiguousarray(momentum).reshape(-1)),
        "cells_on_edge": module.asarray(np.ascontiguousarray(cells).reshape(-1)),
        "rho_edge": module.empty((count,), dtype=np.float32),
        "normal_velocity": module.empty((count,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_mesh_diagnostics_fragment", resources,
        {"nlev": nlev, "ncells": ncells, "nedges": nedges},
        exact_policy=False, grid_size=count,
        threads_per_group=(
            kernel_specs.HEX_MESH_DIAGNOSTICS_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    rho_edge = resources["rho_edge"].to_numpy().reshape(nlev, nedges)
    if np.any(rho_edge == 0.0):
        raise FloatingPointError("zero edge-density denominator")
    return rho_edge, resources["normal_velocity"].to_numpy().reshape(nlev, nedges)


def submit_moist_coefficients_v841(context, *, scalars, moist_indices,
                                   cells_on_edge_remapped, nlev):
    """Build qtot/cqw/cqu for the v8.4.1 moist acoustic path on Metal."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    values = np.asarray(scalars)
    cells = np.asarray(cells_on_edge_remapped)
    indices = np.asarray(tuple(int(index) for index in moist_indices),
                         dtype=np.int32)
    if values.dtype != np.dtype(np.float32) or values.ndim != 3:
        raise TypeError("Hex Metal moist coefficients require FP32 scalar state")
    nscalars, scalar_nlev, ncells = map(int, values.shape)
    nlev = int(nlev)
    if scalar_nlev != nlev:
        raise ValueError("moist scalar vertical extent differs from nlev")
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cells_on_edge_remapped must have one pair per edge")
    nedges = int(cells.shape[0])
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= nscalars):
        raise ValueError("moist_indices must select existing scalar slots")
    cells = np.ascontiguousarray(cells, dtype=np.int32)
    cell_values, edge_values = nlev * ncells, nlev * nedges
    module = MetalArrayModule(context)
    resources = {
        "scalars": module.asarray(np.ascontiguousarray(values).reshape(-1)),
        "moist_indices": module.asarray(indices),
        "cells_on_edge": module.asarray(cells.reshape(-1)),
        "qtot": module.empty((cell_values,), dtype=np.float32),
        "cqw": module.empty((cell_values,), dtype=np.float32),
        "cqu": module.empty((edge_values,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_moist_coefficients_v841_fragment", resources,
        {"nlev": nlev, "ncells": ncells, "nedges": nedges,
         "nscalars": nscalars, "nmoist": int(indices.size)},
        exact_policy=False, grid_size=max(cell_values, edge_values),
        threads_per_group=(
            kernel_specs.HEX_MOIST_COEFFICIENTS_V841_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return (
        resources["qtot"].to_numpy().reshape(nlev, ncells),
        resources["cqw"].to_numpy().reshape(nlev, ncells),
        resources["cqu"].to_numpy().reshape(nlev, nedges),
    )


def submit_vector_invariant_momentum(
    context, *, normal_velocity, rho_edge, pv_edge, kinetic_energy,
    horizontal_divergence, cells_on_edge, edges_on_edge, edge_counts,
    weights_on_edge, inv_dc,
):
    """Submit the FP32 vector-invariant horizontal momentum tendency."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    arrays = {
        "normal_velocity": np.asarray(normal_velocity),
        "rho_edge": np.asarray(rho_edge),
        "pv_edge": np.asarray(pv_edge),
        "kinetic_energy": np.asarray(kinetic_energy),
        "horizontal_divergence": np.asarray(horizontal_divergence),
        "cells_on_edge": np.asarray(cells_on_edge),
        "edges_on_edge": np.asarray(edges_on_edge),
        "edge_counts": np.asarray(edge_counts),
        "weights_on_edge": np.asarray(weights_on_edge),
        "inv_dc_edge": np.asarray(inv_dc),
    }
    u = arrays["normal_velocity"]
    if u.dtype != np.dtype(np.float32) or u.ndim != 2:
        raise TypeError("Hex Metal vector-invariant momentum requires FP32 2-D velocity")
    nlev, nedges = map(int, u.shape)
    cells = arrays["cells_on_edge"]
    neighbors = arrays["edges_on_edge"]
    counts = arrays["edge_counts"]
    weights = arrays["weights_on_edge"]
    if (cells.shape != (nedges, 2) or neighbors.ndim != 2
            or neighbors.shape[0] != nedges or counts.shape != (nedges,)
            or weights.shape != neighbors.shape
            or arrays["rho_edge"].shape != u.shape
            or arrays["pv_edge"].shape != u.shape
            or arrays["kinetic_energy"].shape != arrays["horizontal_divergence"].shape
            or arrays["kinetic_energy"].shape[0] != nlev
            or arrays["inv_dc_edge"].shape != (nedges,)):
        raise ValueError("Hex vector-invariant momentum inputs have incompatible extents")
    ncells = int(arrays["kinetic_energy"].shape[1])
    if np.any(counts < 0) or np.any(counts > neighbors.shape[1]):
        raise ValueError("Hex edgesOnEdge counts exceed their row bounds")
    used = np.arange(neighbors.shape[1])[None, :] < counts[:, None]
    if (np.any(cells < 0) or np.any(cells >= ncells)
            or np.any(neighbors[used] < 0) or np.any(neighbors[used] >= nedges)):
        raise ValueError("Hex used momentum mesh indices contain sentinels")
    if any(value.dtype != np.dtype(np.float32)
           for name, value in arrays.items()
           if name not in ("cells_on_edge", "edges_on_edge", "edge_counts")):
        raise TypeError("Hex Metal vector-invariant momentum inputs must be FP32")
    if any(value.dtype.kind not in "iu" for name, value in arrays.items()
           if name in ("cells_on_edge", "edges_on_edge", "edge_counts")):
        raise TypeError("Hex momentum mesh indices and counts must be integers")

    metal = MetalArrayModule(context)
    resources = {
        "normal_velocity": metal.asarray(np.ascontiguousarray(u).reshape(-1)),
        "rho_edge": metal.asarray(np.ascontiguousarray(arrays["rho_edge"]).reshape(-1)),
        "pv_edge": metal.asarray(np.ascontiguousarray(arrays["pv_edge"]).reshape(-1)),
        "kinetic_energy": metal.asarray(np.ascontiguousarray(
            arrays["kinetic_energy"]).reshape(-1)),
        "horizontal_divergence": metal.asarray(np.ascontiguousarray(
            arrays["horizontal_divergence"]).reshape(-1)),
        "cells_on_edge": metal.asarray(np.ascontiguousarray(
            cells, dtype=np.int32).reshape(-1)),
        "edges_on_edge": metal.asarray(np.ascontiguousarray(
            neighbors, dtype=np.int32).reshape(-1)),
        "edge_counts": metal.asarray(np.ascontiguousarray(
            counts, dtype=np.int32)),
        "weights_on_edge": metal.asarray(np.ascontiguousarray(
            weights).reshape(-1)),
        "inv_dc": metal.asarray(
            np.ascontiguousarray(arrays["inv_dc_edge"])),
        "tendency": metal.empty((nlev * nedges,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_vector_invariant_momentum_fragment", resources,
        {"nlev": nlev, "nedges": nedges, "ncells": ncells,
         "max_edges": int(neighbors.shape[1])},
        exact_policy=False, grid_size=nlev * nedges,
        threads_per_group=(
            kernel_specs.HEX_VECTOR_INVARIANT_MOMENTUM_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["tendency"].to_numpy().reshape(nlev, nedges)


def submit_vector_invariant_momentum_v841(
    context, *, normal_velocity, rho_edge, pv_edge, kinetic_energy,
    horizontal_divergence, cells_on_edge, edges_on_edge, edge_counts,
    weights_on_edge, inv_dc_edge, angle_edge, f_edge, u_init, v_init,
):
    """Submit the v8.4.1 reference-wind vector-invariant tendency on Metal."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    arrays = {
        "normal_velocity": np.asarray(normal_velocity),
        "rho_edge": np.asarray(rho_edge),
        "pv_edge": np.asarray(pv_edge),
        "kinetic_energy": np.asarray(kinetic_energy),
        "horizontal_divergence": np.asarray(horizontal_divergence),
        "cells_on_edge": np.asarray(cells_on_edge),
        "edges_on_edge": np.asarray(edges_on_edge),
        "edge_counts": np.asarray(edge_counts),
        "weights_on_edge": np.asarray(weights_on_edge),
        "inv_dc_edge": np.asarray(inv_dc_edge),
        "angle_edge": np.asarray(angle_edge),
        "f_edge": np.asarray(f_edge),
        "u_init": np.asarray(u_init),
        "v_init": np.asarray(v_init),
    }
    u = arrays["normal_velocity"]
    if u.dtype != np.dtype(np.float32) or u.ndim != 2:
        raise TypeError("Hex Metal v8.4.1 momentum requires FP32 2-D velocity")
    nlev, nedges = map(int, u.shape)
    cells = arrays["cells_on_edge"]
    neighbors = arrays["edges_on_edge"]
    counts = arrays["edge_counts"]
    weights = arrays["weights_on_edge"]
    ncells = int(arrays["kinetic_energy"].shape[1])
    if (cells.shape != (nedges, 2) or neighbors.ndim != 2
            or neighbors.shape[0] != nedges or counts.shape != (nedges,)
            or weights.shape != neighbors.shape
            or arrays["rho_edge"].shape != u.shape
            or arrays["pv_edge"].shape != u.shape
            or arrays["kinetic_energy"].shape !=
            arrays["horizontal_divergence"].shape
            or arrays["kinetic_energy"].shape[0] != nlev
            or arrays["inv_dc_edge"].shape != (nedges,)
            or arrays["angle_edge"].shape != (nedges,)
            or arrays["f_edge"].shape != (nedges,)
            or arrays["u_init"].shape != (nlev,)
            or arrays["v_init"].shape != (nlev,)):
        raise ValueError("Hex v8.4.1 momentum inputs have incompatible extents")
    if any(value.dtype != np.dtype(np.float32)
           for name, value in arrays.items()
           if name not in ("cells_on_edge", "edges_on_edge", "edge_counts")):
        raise TypeError("Hex Metal v8.4.1 momentum inputs must be FP32")
    if any(value.dtype.kind not in "iu"
           for name, value in arrays.items()
           if name in ("cells_on_edge", "edges_on_edge", "edge_counts")):
        raise TypeError("Hex momentum mesh indices and counts must be integers")
    if np.any(counts < 0) or np.any(counts > neighbors.shape[1]):
        raise ValueError("Hex edgesOnEdge counts exceed their row bounds")
    used = np.arange(neighbors.shape[1])[None, :] < counts[:, None]
    if (np.any(cells < 0) or np.any(cells >= ncells)
            or np.any(neighbors[used] < 0) or np.any(neighbors[used] >= nedges)):
        raise ValueError("Hex used momentum mesh indices contain sentinels")

    metal = MetalArrayModule(context)
    resources = {}
    for name, value in arrays.items():
        if name in ("cells_on_edge", "edges_on_edge", "edge_counts"):
            value = np.ascontiguousarray(value, dtype=np.int32)
        else:
            value = np.ascontiguousarray(value, dtype=np.float32)
        resources[name] = metal.asarray(value.reshape(-1))
    resources["tendency"] = metal.empty((nlev * nedges,), dtype=np.float32)
    context.submit_kernel(
        "hex_vector_invariant_momentum_v841_fragment", resources,
        {"nlev": nlev, "nedges": nedges, "ncells": ncells,
         "max_edges": int(neighbors.shape[1])}, exact_policy=False,
        grid_size=nlev * nedges,
        threads_per_group=(
            kernel_specs.HEX_VECTOR_INVARIANT_MOMENTUM_V841_FRAGMENT_THREADS_PER_GROUP),
    )
    return resources["tendency"].to_numpy().reshape(nlev, nedges)


def submit_omega_terrain_conversion(
    context, *, w_tendency, u_tendency, zz, fzm, fzp, zb_cell, zb3_cell,
    edges_on_cell, edge_counts, edge_signs, boundary_mask, relaxation_zone,
):
    """Submit the FP32 terrain-slope conversion from w to omega tendency."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    arrays = {name: np.asarray(value) for name, value in {
        "w_tendency": w_tendency, "u_tendency": u_tendency, "zz": zz,
        "fzm": fzm, "fzp": fzp, "zb_cell": zb_cell, "zb3_cell": zb3_cell,
        "edges_on_cell": edges_on_cell, "edge_counts": edge_counts,
        "edge_signs": edge_signs, "boundary_mask": boundary_mask,
    }.items()}
    w = arrays["w_tendency"]
    u = arrays["u_tendency"]
    if w.dtype != np.dtype(np.float32) or u.dtype != w.dtype or w.ndim != 2 or u.ndim != 2:
        raise TypeError("Hex Metal omega terrain conversion requires FP32 2-D tendencies")
    nlev, nedges = map(int, u.shape)
    ncell = int(w.shape[1])
    edges, counts = arrays["edges_on_cell"], arrays["edge_counts"]
    if (w.shape != (nlev + 1, ncell) or arrays["zz"].shape != (nlev, ncell)
            or arrays["fzm"].shape != (nlev,) or arrays["fzp"].shape != (nlev,)
            or edges.ndim != 2 or edges.shape[0] != ncell
            or counts.shape != (ncell,) or arrays["edge_signs"].shape != edges.shape
            or arrays["zb_cell"].shape != (nlev + 1, ncell, edges.shape[1])
            or arrays["zb3_cell"].shape != arrays["zb_cell"].shape
            or arrays["boundary_mask"].shape != (ncell,)):
        raise ValueError("Hex omega terrain inputs have incompatible extents")
    numeric = ("zz", "fzm", "fzp", "zb_cell", "zb3_cell", "edge_signs")
    if any(arrays[name].dtype != w.dtype for name in numeric):
        raise TypeError("Hex omega terrain coefficients must match FP32 tendencies")
    if (edges.dtype.kind not in "iu" or counts.dtype.kind not in "iu"
            or arrays["boundary_mask"].dtype.kind not in "iu"):
        raise TypeError("Hex omega terrain indices, counts and mask must be integers")
    max_edges = int(edges.shape[1])
    if np.any(counts < 0) or np.any(counts > max_edges):
        raise ValueError("Hex cell edge counts exceed edgesOnCell row bounds")
    used = np.arange(max_edges)[None, :] < counts[:, None]
    if np.any(edges[used] < 0) or np.any(edges[used] >= nedges):
        raise ValueError("Hex active terrain edge indices contain sentinels")

    metal = MetalArrayModule(context)
    resources = {
        "w_tendency": metal.asarray(np.ascontiguousarray(w).reshape(-1)),
        "u_tendency": metal.asarray(np.ascontiguousarray(u).reshape(-1)),
        "zz": metal.asarray(np.ascontiguousarray(arrays["zz"]).reshape(-1)),
        "fzm": metal.asarray(np.ascontiguousarray(arrays["fzm"])),
        "fzp": metal.asarray(np.ascontiguousarray(arrays["fzp"])),
        "zb_cell": metal.asarray(np.ascontiguousarray(arrays["zb_cell"]).reshape(-1)),
        "zb3_cell": metal.asarray(np.ascontiguousarray(arrays["zb3_cell"]).reshape(-1)),
        "edges_on_cell": metal.asarray(np.ascontiguousarray(edges, dtype=np.int32).reshape(-1)),
        "edge_counts": metal.asarray(np.ascontiguousarray(counts, dtype=np.int32)),
        "edge_signs": metal.asarray(np.ascontiguousarray(arrays["edge_signs"]).reshape(-1)),
        "boundary_mask": metal.asarray(np.ascontiguousarray(
            arrays["boundary_mask"], dtype=np.int32)),
        "omega_tendency": metal.empty((w.size,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_omega_terrain_conversion_fragment", resources,
        {"nlev": nlev, "ncells": ncell, "nedges": nedges,
         "max_edges": max_edges, "relaxation_zone": int(relaxation_zone)},
        exact_policy=False, grid_size=(nlev + 1) * ncell,
        threads_per_group=(
            kernel_specs.HEX_OMEGA_TERRAIN_CONVERSION_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["omega_tendency"].to_numpy().reshape(w.shape)


def submit_pressure_gradient(
    context, *, pressure, dpdz, cqu, zz, zxu, cells_on_edge, inv_dc,
):
    """Submit the FP32 terrain-coordinate pressure-gradient tendency."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    pp = np.asarray(pressure)
    arrays = {name: np.asarray(value) for name, value in {
        "dpdz": dpdz, "cqu": cqu, "zz": zz, "zxu": zxu,
        "cells_on_edge": cells_on_edge, "inv_dc_edge": inv_dc,
    }.items()}
    if pp.dtype != np.dtype(np.float32) or pp.ndim != 2:
        raise TypeError("Hex Metal pressure gradient requires FP32 2-D pressure")
    nlev, ncells = map(int, pp.shape)
    cells = arrays["cells_on_edge"]
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("Hex pressure-gradient cellsOnEdge must have two columns")
    nedges = int(cells.shape[0])
    if (any(arrays[name].shape != pp.shape for name in ("dpdz", "zz"))
            or arrays["cqu"].shape != (nlev, nedges)
            or arrays["zxu"].shape != (nlev, nedges)
            or arrays["inv_dc_edge"].shape != (nedges,)):
        raise ValueError("Hex pressure-gradient input shapes are inconsistent")
    if any(arrays[name].dtype != pp.dtype
           for name in ("dpdz", "cqu", "zz", "zxu", "inv_dc_edge")):
        raise TypeError("Hex pressure-gradient fields must use FP32")
    if cells.dtype.kind not in "iu" or np.any(cells < 0) or np.any(cells >= ncells):
        raise ValueError("Hex pressure-gradient cell indices are invalid")
    metal = MetalArrayModule(context)
    resources = {
        "pressure_perturbation": metal.asarray(np.ascontiguousarray(pp)),
        "dpdz": metal.asarray(np.ascontiguousarray(arrays["dpdz"])),
        "cqu": metal.asarray(np.ascontiguousarray(arrays["cqu"])),
        "zz": metal.asarray(np.ascontiguousarray(arrays["zz"])),
        "zxu": metal.asarray(np.ascontiguousarray(arrays["zxu"])),
        "cells_on_edge": metal.asarray(np.ascontiguousarray(cells, dtype=np.int32)),
        "inv_dc_edge": metal.asarray(np.ascontiguousarray(arrays["inv_dc_edge"])),
        "tendency": metal.empty((nlev, nedges), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_pressure_gradient_euler", resources,
        {"nlev": nlev, "nedges": nedges, "ncells": ncells},
        exact_policy=False, grid_size=nlev * nedges,
        threads_per_group=kernel_specs.HEX_PRESSURE_GRADIENT_EULER_THREADS_PER_GROUP,
    )
    return resources["tendency"].to_numpy()


def submit_vertical_pressure_euler(
    context, *, pressure_perturbation, dpdz, cqw, rdzu, fzm, fzp,
):
    """Submit the FP32 vertical Euler pressure tendency for Hex w."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    pp, dpdz = np.asarray(pressure_perturbation), np.asarray(dpdz)
    if pp.dtype != np.dtype(np.float32) or pp.ndim != 2 or dpdz.shape != pp.shape:
        raise TypeError("Hex vertical pressure Euler requires matching FP32 cell fields")
    nlev, ncells = map(int, pp.shape)
    cqw = (np.ones((nlev, ncells), dtype=np.float32) if cqw is None
           else np.asarray(cqw))
    rdzu, fzm, fzp = (np.asarray(value) for value in (rdzu, fzm, fzp))
    if (cqw.dtype != pp.dtype or cqw.shape != (nlev, ncells)
            or any(value.dtype != pp.dtype or value.ndim != 1 or value.size < nlev
                   for value in (rdzu, fzm, fzp))):
        raise ValueError("Hex vertical pressure Euler coefficients have invalid shapes or dtypes")
    metal = MetalArrayModule(context)
    resources = {
        "pressure_perturbation": metal.asarray(np.ascontiguousarray(pp).reshape(-1)),
        "dpdz": metal.asarray(np.ascontiguousarray(dpdz).reshape(-1)),
        "cqw": metal.asarray(np.ascontiguousarray(cqw).reshape(-1)),
        "rdzu": metal.asarray(np.ascontiguousarray(rdzu)),
        "fzm": metal.asarray(np.ascontiguousarray(fzm)),
        "fzp": metal.asarray(np.ascontiguousarray(fzp)),
        "tendency": metal.empty(((nlev + 1) * ncells,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_vertical_pressure_euler_fragment", resources,
        {"nlev": nlev, "ncells": ncells}, exact_policy=False,
        grid_size=(nlev + 1) * ncells,
        threads_per_group=(
            kernel_specs.HEX_VERTICAL_PRESSURE_EULER_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["tendency"].to_numpy().reshape(nlev + 1, ncells)


def submit_density_tendency(
    context, *, horizontal, rho_w, rdzw, physics_tendency=None,
):
    """Fuse the Hex density tendency from horizontal and vertical fluxes."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    horizontal, rho_w = np.asarray(horizontal), np.asarray(rho_w)
    rdzw = np.asarray(rdzw, dtype=horizontal.dtype)
    if (horizontal.dtype != np.dtype(np.float32) or horizontal.ndim != 2
            or rho_w.dtype != horizontal.dtype or rho_w.shape !=
            (horizontal.shape[0] + 1, horizontal.shape[1])
            or rdzw.shape != (horizontal.shape[0],)):
        raise ValueError("Hex density tendency inputs have invalid shapes or dtypes")
    if physics_tendency is None:
        physics = np.zeros_like(horizontal)
    else:
        physics = np.asarray(physics_tendency, dtype=horizontal.dtype)
        if physics.shape != horizontal.shape:
            raise ValueError("Hex density physics tendency shape mismatch")
    nlev, ncells = map(int, horizontal.shape)
    metal = MetalArrayModule(context)
    resources = {
        "horizontal": metal.asarray(np.ascontiguousarray(horizontal).reshape(-1)),
        "rho_w": metal.asarray(np.ascontiguousarray(rho_w).reshape(-1)),
        "rdzw": metal.asarray(np.ascontiguousarray(rdzw)),
        "physics_tendency": metal.asarray(np.ascontiguousarray(physics).reshape(-1)),
        "tendency": metal.empty((nlev * ncells,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_density_tendency_fragment", resources,
        {"nlev": nlev, "ncells": ncells}, exact_policy=False,
        grid_size=nlev * ncells,
        threads_per_group=kernel_specs.HEX_DENSITY_TENDENCY_FRAGMENT_THREADS_PER_GROUP,
    )
    return resources["tendency"].to_numpy().reshape(nlev, ncells)


def submit_dpdz_tendency(
    context, *, density_perturbation, gravity, rho_base=None, qtot=None,
):
    """Form the saved vertical pressure gradient for dry or moist Hex state."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    perturbation = np.asarray(density_perturbation)
    if perturbation.dtype != np.dtype(np.float32) or perturbation.ndim != 2:
        raise TypeError("Hex dpdz Metal stage requires FP32 2-D density perturbation")
    nlev, ncells = map(int, perturbation.shape)
    use_moist = int(qtot is not None)
    if use_moist:
        base, total = np.asarray(rho_base), np.asarray(qtot)
        if base.shape != perturbation.shape or total.shape != perturbation.shape:
            raise ValueError("Hex moist dpdz fields must match density perturbation")
    else:
        base = np.zeros_like(perturbation)
        total = np.zeros_like(perturbation)
    if base.dtype != perturbation.dtype or total.dtype != perturbation.dtype:
        raise TypeError("Hex dpdz fields must use FP32")
    metal = MetalArrayModule(context)
    resources = {
        "density_perturbation": metal.asarray(np.ascontiguousarray(perturbation).reshape(-1)),
        "rho_base": metal.asarray(np.ascontiguousarray(base).reshape(-1)),
        "qtot": metal.asarray(np.ascontiguousarray(total).reshape(-1)),
        "dpdz": metal.empty((nlev * ncells,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_dpdz_tendency_fragment", resources,
        {"nlev": nlev, "ncells": ncells, "gravity": float(gravity),
         "use_moist": use_moist},
        exact_policy=False, grid_size=nlev * ncells,
        threads_per_group=kernel_specs.HEX_DPDZ_TENDENCY_FRAGMENT_THREADS_PER_GROUP,
    )
    return resources["dpdz"].to_numpy().reshape(nlev, ncells)


def submit_horizontal_theta_tendency(
    context, *, theta, theta_saved, rho_u, rho_u_saved, adv_coefs,
    adv_coefs_3rd, adv_cells_for_edge, n_adv_cells_for_edge,
    edges_on_cell, edge_counts, edge_signs, cells_on_edge, dv_edge,
    inv_area_cell, skip_cells, rk_step, third_order_coefficient,
):
    """Submit the FP32 cell-owned horizontal rho-theta flux tendency."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    floats = {name: np.asarray(value) for name, value in {
        "theta": theta, "theta_saved": theta_saved, "rho_u": rho_u,
        "rho_u_saved": rho_u_saved, "adv_coefs": adv_coefs,
        "adv_coefs_3rd": adv_coefs_3rd, "edge_signs": edge_signs,
        "dv_edge": dv_edge, "inv_area_cell": inv_area_cell,
    }.items()}
    ints = {name: np.asarray(value) for name, value in {
        "adv_cells_for_edge": adv_cells_for_edge,
        "n_adv_cells_for_edge": n_adv_cells_for_edge,
        "edges_on_cell": edges_on_cell, "edge_counts": edge_counts,
        "cells_on_edge": cells_on_edge,
    }.items()}
    th = floats["theta"]
    if th.dtype != np.dtype(np.float32) or th.ndim != 2:
        raise TypeError("Hex horizontal theta Metal stage requires FP32 2-D theta")
    nlev, ncells = map(int, th.shape)
    nedges = int(floats["rho_u"].shape[1])
    edge_array, counts = ints["edges_on_cell"], ints["edge_counts"]
    adv_cells = ints["adv_cells_for_edge"]
    if (floats["theta_saved"].shape != th.shape
            or floats["rho_u"].shape != (nlev, nedges)
            or floats["rho_u_saved"].shape != (nlev, nedges)
            or floats["adv_coefs"].ndim != 2
            or floats["adv_coefs_3rd"].shape != floats["adv_coefs"].shape
            or adv_cells.shape != floats["adv_coefs"].shape
            or ints["n_adv_cells_for_edge"].shape != (nedges,)
            or edge_array.shape[0] != ncells or edge_array.ndim != 2
            or counts.shape != (ncells,) or floats["edge_signs"].shape != edge_array.shape
            or ints["cells_on_edge"].shape != (nedges, 2)
            or floats["dv_edge"].shape != (nedges,)
            or floats["inv_area_cell"].shape != (ncells,)):
        raise ValueError("Hex horizontal theta Metal input shapes are inconsistent")
    max_cell_edges = int(edge_array.shape[1])
    max_adv = int(floats["adv_coefs"].shape[1])
    if (np.any(counts < 0) or np.any(counts > max_cell_edges)
            or np.any(ints["n_adv_cells_for_edge"] < 0)
            or np.any(ints["n_adv_cells_for_edge"] > max_adv)):
        raise ValueError("Hex horizontal theta connectivity counts exceed row bounds")
    used_edges = np.arange(max_cell_edges)[None, :] < counts[:, None]
    used_adv = np.arange(max_adv)[None, :] < ints["n_adv_cells_for_edge"][:, None]
    if (np.any(edge_array[used_edges] < 0) or np.any(edge_array[used_edges] >= nedges)
            or np.any(adv_cells[used_adv] < 0) or np.any(adv_cells[used_adv] >= ncells)):
        raise ValueError("Hex horizontal theta active connectivity contains sentinels")
    if any(value.dtype != th.dtype for value in floats.values()):
        raise TypeError("Hex horizontal theta floating inputs must be FP32")
    if any(value.dtype.kind not in "iu" for value in ints.values()):
        raise TypeError("Hex horizontal theta connectivity must be integer arrays")
    skipped = (np.zeros((ncells,), dtype=np.int32) if skip_cells is None else
               np.ascontiguousarray(np.asarray(skip_cells, dtype=np.int32)))
    if skipped.shape != (ncells,):
        raise ValueError("Hex horizontal theta skip mask must match the cell count")

    metal = MetalArrayModule(context)
    resources = {
        "theta": metal.asarray(np.ascontiguousarray(th).reshape(-1)),
        "theta_saved": metal.asarray(np.ascontiguousarray(floats["theta_saved"]).reshape(-1)),
        "rho_u": metal.asarray(np.ascontiguousarray(floats["rho_u"]).reshape(-1)),
        "rho_u_saved": metal.asarray(np.ascontiguousarray(floats["rho_u_saved"]).reshape(-1)),
        "adv_coefs": metal.asarray(np.ascontiguousarray(floats["adv_coefs"]).reshape(-1)),
        "adv_coefs_3rd": metal.asarray(np.ascontiguousarray(floats["adv_coefs_3rd"]).reshape(-1)),
        "adv_cells_for_edge": metal.asarray(np.ascontiguousarray(adv_cells, dtype=np.int32).reshape(-1)),
        "n_adv_cells_for_edge": metal.asarray(np.ascontiguousarray(
            ints["n_adv_cells_for_edge"], dtype=np.int32)),
        "edges_on_cell": metal.asarray(np.ascontiguousarray(edge_array, dtype=np.int32).reshape(-1)),
        "edge_counts": metal.asarray(np.ascontiguousarray(counts, dtype=np.int32)),
        "edge_signs": metal.asarray(np.ascontiguousarray(floats["edge_signs"]).reshape(-1)),
        "cells_on_edge": metal.asarray(np.ascontiguousarray(
            ints["cells_on_edge"], dtype=np.int32).reshape(-1)),
        "dv_edge": metal.asarray(np.ascontiguousarray(floats["dv_edge"])),
        "inv_area_cell": metal.asarray(np.ascontiguousarray(floats["inv_area_cell"])),
        "skip_cell": metal.asarray(skipped),
        "tendency": metal.empty((nlev * ncells,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_horizontal_theta_tendency_fragment", resources,
        {"nlev": nlev, "ncells": ncells, "nedges": nedges,
         "max_cell_edges": max_cell_edges, "max_adv_cells": max_adv,
         "rk_step": int(rk_step),
         "third_order_coefficient": float(third_order_coefficient)},
        exact_policy=False, grid_size=nlev * ncells,
        threads_per_group=(
            kernel_specs.HEX_HORIZONTAL_THETA_TENDENCY_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["tendency"].to_numpy().reshape(nlev, ncells)


def submit_vertical_theta_tendency(
    context, *, horizontal, theta, theta_saved, rho_w, rho_w_saved,
    fzm, fzp, rdzw, coefficient,
):
    """Submit the FP32 vertical rho-theta flux divergence and combine it."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    arrays = {name: np.asarray(value) for name, value in {
        "horizontal": horizontal, "theta": theta, "theta_saved": theta_saved,
        "rho_w": rho_w, "rho_w_saved": rho_w_saved,
        "fzm": fzm, "fzp": fzp, "rdzw": rdzw,
    }.items()}
    th = arrays["theta"]
    if th.dtype != np.dtype(np.float32) or th.ndim != 2:
        raise TypeError("Hex vertical theta Metal stage requires FP32 2-D theta")
    nlev, ncells = map(int, th.shape)
    if (any(arrays[name].shape != th.shape for name in
            ("horizontal", "theta_saved"))
            or arrays["rho_w"].shape != (nlev + 1, ncells)
            or arrays["rho_w_saved"].shape != (nlev + 1, ncells)
            or arrays["fzm"].ndim != 1 or arrays["fzp"].shape != arrays["fzm"].shape
            or arrays["fzm"].size < nlev or arrays["rdzw"].shape != (nlev,)):
        raise ValueError("Hex vertical theta Metal input shapes are inconsistent")
    if any(value.dtype != th.dtype for value in arrays.values()):
        raise TypeError("Hex vertical theta Metal fields must use FP32")
    metal = MetalArrayModule(context)
    resources = {
        "horizontal": metal.asarray(np.ascontiguousarray(arrays["horizontal"]).reshape(-1)),
        "theta": metal.asarray(np.ascontiguousarray(th).reshape(-1)),
        "theta_saved": metal.asarray(np.ascontiguousarray(arrays["theta_saved"]).reshape(-1)),
        "rho_w": metal.asarray(np.ascontiguousarray(arrays["rho_w"]).reshape(-1)),
        "rho_w_saved": metal.asarray(np.ascontiguousarray(arrays["rho_w_saved"]).reshape(-1)),
        "fzm": metal.asarray(np.ascontiguousarray(arrays["fzm"])),
        "fzp": metal.asarray(np.ascontiguousarray(arrays["fzp"])),
        "rdzw": metal.asarray(np.ascontiguousarray(arrays["rdzw"])),
        "tendency": metal.empty((nlev * ncells,), dtype=np.float32),
    }
    context.submit_kernel(
        "hex_vertical_theta_tendency_fragment", resources,
        {"nlev": nlev, "ncells": ncells,
         "coefficient": float(coefficient)}, exact_policy=False,
        grid_size=nlev * ncells,
        threads_per_group=(
            kernel_specs.HEX_VERTICAL_THETA_TENDENCY_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return resources["tendency"].to_numpy().reshape(nlev, ncells)


def submit_pressure_perturbation(context, *, exner, rho_theta_perturbation,
                                 rho_theta_base, exner_base, zz, rgas):
    """Dispatch the integrated FP32 pressure-perturbation diagnostic."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.generated import kernel_specs

    values = {name: np.ascontiguousarray(np.asarray(value)) for name, value in {
        "exner": exner, "rho_theta_perturbation": rho_theta_perturbation,
        "rho_theta_base": rho_theta_base, "exner_base": exner_base,
    }.items()}
    nlev, ncells = values["exner"].shape
    z = np.ascontiguousarray(np.asarray(zz))
    if z.shape == (nlev, ncells):
        if not np.all(z == z[:, :1]):
            raise ConfigurationRefusal(
                "hex_metal_vertical_coordinate", z.shape,
                "the integrated pressure-perturbation ABI has one zz per level",
                "a level-uniform zz grid or cellwise pressure-perturbation kernel",
            )
        z = np.ascontiguousarray(z[:, 0])
    if z.shape != (nlev,) or any(value.shape != (nlev, ncells)
                                 for value in values.values()):
        raise ValueError("Hex pressure-perturbation input shapes are inconsistent")
    if any(value.dtype != np.dtype(np.float32) for value in values.values()) or z.dtype != np.float32:
        raise TypeError("Hex pressure perturbation currently requires FP32")
    metal = MetalArrayModule(context)
    resources = {name: metal.asarray(value)
                 for name, value in values.items()}
    resources["zz"] = metal.asarray(z)
    resources["pressure_perturbation"] = metal.empty((nlev, ncells), dtype=np.float32)
    context.submit_kernel(
        "hex_pressure_perturbation", resources,
        {"nlev": nlev, "ncells": ncells, "rgas": float(rgas)},
        exact_policy=False, grid_size=nlev * ncells,
        threads_per_group=kernel_specs.HEX_PRESSURE_PERTURBATION_THREADS_PER_GROUP,
    )
    return resources["pressure_perturbation"].to_numpy()
