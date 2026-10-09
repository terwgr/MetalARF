"""Explicit research time steppers for typed spectral states."""
from __future__ import annotations

from .state import state_add, state_linear_combination


def ssprk3_step(state, dt_s: float, rhs):
    # Each tendency and intermediate state is released as soon as the
    # stage that reads it has run, so a right-hand side is evaluated
    # beside at most the base state and the stage it acts on (memory
    # only: the third evaluation otherwise ran beside six spectral
    # states, 7.1 GiB at T533 with 40 levels, and the peak of a step is
    # inside rhs).
    t0 = state.time_s
    k1 = rhs(state)
    y1 = state_add(state, k1, dt_s, time_s=t0 + dt_s)
    del k1

    k2 = rhs(y1)
    e2 = state_add(y1, k2, dt_s, time_s=t0 + dt_s)
    del k2, y1
    y2 = state_linear_combination(
        state, [(0.75, state), (0.25, e2)], time_s=t0 + 0.5 * dt_s
    )
    del e2

    k3 = rhs(y2)
    e3 = state_add(y2, k3, dt_s, time_s=t0 + dt_s)
    del k3, y2
    return state_linear_combination(
        state, [(1.0 / 3.0, state), (2.0 / 3.0, e3)], time_s=t0 + dt_s
    )


def rk4_step(state, dt_s: float, rhs):
    t0 = state.time_s
    k1 = rhs(state)
    y2 = state_add(state, k1, 0.5 * dt_s, time_s=t0 + 0.5 * dt_s)
    k2 = rhs(y2)
    y3 = state_add(state, k2, 0.5 * dt_s, time_s=t0 + 0.5 * dt_s)
    k3 = rhs(y3)
    y4 = state_add(state, k3, dt_s, time_s=t0 + dt_s)
    k4 = rhs(y4)
    fields = [
        a + (dt_s / 6.0) * (b + 2.0 * c + 2.0 * d + e)
        for a, b, c, d, e in zip(
            state.fields(), k1.fields(), k2.fields(), k3.fields(), k4.fields()
        )
    ]
    return state.with_fields(fields, time_s=t0 + dt_s)


def step_with_scheme(state, dt_s: float, rhs, scheme: str):
    key = str(scheme).lower()
    if key == "ssprk3":
        return ssprk3_step(state, dt_s, rhs)
    if key == "rk4":
        return rk4_step(state, dt_s, rhs)
    raise ValueError(f"unknown time integrator {scheme!r}")
