"""Column integration for Metal data assimilation with explicit host staging."""
import numpy as np

from .array import MetalArrayModule


def integrate(context, arrays, constants, phi, pressure, *, option, mass, species, apply_php=None):
    """Fill host outputs using the Rust operator's field order and binary64 words."""
    from woof.da.hydrostatic import HydrostaticError, _apply_php_host

    levels, ny, nx = pressure.shape
    columns = ny * nx
    budget = min(int(context.recommended_working_set_size) // 16,
                 int(context.max_buffer_length))
    fixed_bytes = 8 * (11 * (levels + 1) + constants.size)
    field_count = 6 + 2 * len(species)
    column_bytes = 8 * ((field_count + 3) * levels + 4) + 4
    # Leave room for at least one solver column and keep the established
    # host writer when the full transactional device output does not fit.
    device_apply = (apply_php is not None
                    and apply_php.shape == (levels + 1, ny, nx)
                    and apply_php.dtype == np.dtype(np.float32)
                    and columns <= np.iinfo(np.int32).max
                    and apply_php.size <= np.iinfo(np.uint32).max
                    and fixed_bytes + column_bytes + apply_php.nbytes <= budget)
    if device_apply:
        fixed_bytes += apply_php.nbytes
    chunk = min(columns, (budget - fixed_bytes) // column_bytes,
                np.iinfo(np.int32).max // (field_count * levels))
    if chunk < 1:
        raise MemoryError("Metal hydrostatic analysis cannot fit one column")
    xp = MetalArrayModule(context)
    result = xp.empty(apply_php.shape, dtype=np.float32) if device_apply else None
    profiles = np.zeros((11, levels + 1), dtype=np.float64)
    for row, values in zip(profiles, arrays[11:]):
        row[:values.size] = values
    profiles_dev = xp.asarray(profiles.view(np.uint64))
    constants_dev = xp.asarray(constants.view(np.uint64))
    fields = [arrays[i].reshape(levels, columns) for i in (0, 1, 2, 3, 4, 7)]
    masses = [values.reshape(columns) for values in arrays[8:11]]
    phi = phi.reshape(levels + 1, columns)
    pressure = pressure.reshape(levels, columns)
    bad_arithmetic = False
    bad_hydro = False
    for start in range(0, columns, chunk):
        end = min(columns, start + chunk)
        count = end - start
        rows, cols = np.divmod(np.arange(start, end), nx)
        field_words = np.stack(
            [field[:, start:end] for field in fields]
            + [values[:, rows, cols] for pair in species for values in pair],
            dtype=np.float64,
        )
        mass_words = np.stack([field[start:end] for field in masses])
        resources = dict(
            field3=xp.asarray(field_words.view(np.uint64)),
            mass2=xp.asarray(mass_words.view(np.uint64)),
            profiles=profiles_dev, constants=constants_dev,
            phi=xp.empty((levels + 1, count), dtype=np.uint64),
            pressure=xp.empty((levels, count), dtype=np.uint64),
            scratch=xp.empty((levels, count), dtype=np.uint64),
            status=xp.empty((count,), dtype=np.uint32),
        )
        context.submit_kernel(
            "da_hydrostatic_column", resources,
            dict(levels=levels, columns=count, option=option, mass=mass, field_count=field_count),
            exact_policy=False, grid_size=count, threads_per_group=32,
        )
        status = resources["status"].to_numpy()
        bad_arithmetic |= bool((status == 2).any())
        bad_hydro |= bool((status == 1).any())
        if not status.any():
            if result is not None:
                context.submit_kernel(
                    "da_hydrostatic_apply",
                    dict(phi=resources["phi"], php=apply_php.view((levels + 1, columns), np.uint32),
                         result=result.view((levels + 1, columns), np.uint32)),
                    dict(levels=levels, columns=count, total_columns=columns, start=start),
                    exact_policy=False, grid_size=count, threads_per_group=32,
                )
            phi[:, start:end] = resources["phi"].to_numpy().view(np.float64)
            pressure[:, start:end] = resources["pressure"].to_numpy().view(np.float64)
    if bad_arithmetic:
        raise ValueError("analysis arithmetic is not finite")
    if bad_hydro:
        raise HydrostaticError(
            "the analysed hydrostatic pressure is not finite and positive "
            "or the geopotential increment is not finite")
    if result is not None:
        apply_php[...] = result
    elif apply_php is not None:
        dphp = phi.reshape(levels + 1, ny, nx)
        _apply_php_host(apply_php, dphp, dphp != 0.0)
