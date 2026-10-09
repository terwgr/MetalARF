"""Typed spectral prognostic states and arithmetic helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass
class ShallowWaterState:
    vorticity: object
    divergence: object
    geopotential: object
    time_s: float = 0.0
    step: int = 0

    def fields(self) -> tuple[object, ...]:
        return self.vorticity, self.divergence, self.geopotential

    def with_fields(self, fields: Iterable[object], *, time_s: float | None = None, step: int | None = None):
        z, d, p = tuple(fields)
        return replace(
            self,
            vorticity=z,
            divergence=d,
            geopotential=p,
            time_s=self.time_s if time_s is None else float(time_s),
            step=self.step if step is None else int(step),
        )


@dataclass
class PrimitiveDryState:
    vorticity: object
    divergence: object
    temperature: object
    log_surface_pressure: object
    time_s: float = 0.0
    step: int = 0

    def fields(self) -> tuple[object, ...]:
        return (
            self.vorticity,
            self.divergence,
            self.temperature,
            self.log_surface_pressure,
        )

    def with_fields(self, fields: Iterable[object], *, time_s: float | None = None, step: int | None = None):
        z, d, t, p = tuple(fields)
        return replace(
            self,
            vorticity=z,
            divergence=d,
            temperature=t,
            log_surface_pressure=p,
            time_s=self.time_s if time_s is None else float(time_s),
            step=self.step if step is None else int(step),
        )


def state_add(state, tendency, factor: float, *, time_s: float | None = None):
    fields = []
    for a, b in zip(state.fields(), tendency.fields()):
        from woof.metal.array import MetalArray

        if not isinstance(a, MetalArray) and not isinstance(b, MetalArray):
            fields.append(a + factor * b)
            continue
        from woof.metal.array import MetalArrayModule
        from woof.metal.generated import kernel_specs
        import numpy as np

        context = (a.buffer.context if isinstance(a, MetalArray)
                   else b.buffer.context)
        if isinstance(a, MetalArray):
            if (a.buffer.context is not context
                    or a.dtype != np.dtype(np.complex128)):
                raise TypeError("Global state addition requires resident complex128")
            base = a
        else:
            host_a = np.ascontiguousarray(a, dtype=np.complex128)
            base = MetalArrayModule(context).asarray(host_a)
        if isinstance(b, MetalArray):
            if (b.buffer.context is not context
                    or b.dtype != np.dtype(np.complex128)
                    or b.shape != base.shape):
                raise TypeError("Global tendency must match resident complex128 state")
            increment = b
        else:
            host_b = np.ascontiguousarray(b, dtype=np.complex128)
            if host_b.shape != base.shape:
                raise ValueError("Global state and tendency shapes differ")
            increment = MetalArrayModule(context).asarray(host_b)
        metal = MetalArrayModule(context)
        count = base.size * 2
        base_words = base.view((count,), dtype=np.float64)
        increment_words = increment.view((count,), dtype=np.float64)
        previous_scale = metal.asarray(np.asarray([0.0], dtype=np.float64))
        current_scale = metal.asarray(np.asarray([float(factor)], dtype=np.float64))
        destination = metal.empty((count,), dtype=np.float64)
        context.submit_kernel(
            "global_semilag_add_scaled_fragment",
            {"base": base_words, "increment": increment_words,
             "previous": increment_words, "current_scale": current_scale,
             "previous_scale": previous_scale, "destination": destination},
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        fields.append(destination.view(tuple(base.shape), dtype=np.complex128))
    return state.with_fields(fields, time_s=time_s)


def state_linear_combination(template, terms: list[tuple[float, object]], *, time_s: float | None = None):
    fields = []
    for index in range(len(template.fields())):
        values = [(float(weight), state.fields()[index])
                  for weight, state in terms]
        from woof.metal.array import MetalArray

        if not any(isinstance(value, MetalArray) for _, value in values):
            result = None
            for weight, value in values:
                term = weight * value
                result = term if result is None else result + term
            fields.append(result)
            continue

        from woof.metal.array import MetalArrayModule
        from woof.metal.generated import kernel_specs
        import numpy as np

        metal_value = next(value for _, value in values
                           if isinstance(value, MetalArray))
        context = metal_value.buffer.context
        if metal_value.dtype != np.dtype(np.complex128):
            raise TypeError("Global linear combination requires complex128 Metal")
        metal = MetalArrayModule(context)
        accumulator = None
        for weight, value in values:
            if isinstance(value, MetalArray):
                if (value.buffer.context is not context
                        or value.dtype != np.dtype(np.complex128)
                        or value.shape != metal_value.shape):
                    raise TypeError(
                        "Global linear combination terms must share complex128 Metal layout"
                    )
                source = value
            else:
                host = np.ascontiguousarray(value, dtype=np.complex128)
                if host.shape != metal_value.shape:
                    raise ValueError("Global linear combination term shapes differ")
                source = metal.asarray(host)

            count = int(source.size * 2)
            source_words = source.view((count,), dtype=np.float64)
            destination = metal.empty((count,), dtype=np.float64)
            if accumulator is None:
                base = increment = previous = source_words
                current_scale = metal.asarray(
                    np.asarray([weight - 1.0], dtype=np.float64)
                )
            else:
                base = accumulator.view((count,), dtype=np.float64)
                increment = previous = source_words
                current_scale = metal.asarray(
                    np.asarray([weight], dtype=np.float64)
                )
            previous_scale = metal.asarray(np.asarray([0.0], dtype=np.float64))
            context.submit_kernel(
                "global_semilag_add_scaled_fragment",
                {"base": base, "increment": increment,
                 "previous": previous, "current_scale": current_scale,
                 "previous_scale": previous_scale,
                 "destination": destination},
                {"count": count}, exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            accumulator = destination.view(
                tuple(metal_value.shape), dtype=np.complex128
            )
        fields.append(accumulator)
    return template.with_fields(fields, time_s=time_s)
