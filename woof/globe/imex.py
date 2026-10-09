"""IMEX Runge-Kutta time integration of the semi-implicit split.

The dycore's own integrator: the explicit right-hand side ``E = F - L``
(``rhs``, which subtracts the semi-implicit scheme's linear gravity-wave
operator ``L``) and the operator ``L`` itself are advanced by ONE
Runge-Kutta method with two tableaux that share every abscissa, so a
balanced state is a fixed point of the step exactly, not to first order.

WHY.  The shipped symmetric split ``H R3 H`` (``H = sqrt(CN(dt L))``,
semi_implicit.py) is not a fixed point at a balanced state: with ``F(y_b)
= 0`` the half map moves ``y_b`` (``L y_b`` is as large as the terrain's
own pressure-gradient force), the explicit step then integrates ``-L``
from a state that is no longer balanced, and the second half map does not
undo that beyond first order: the per-step residual is the commutator
``dt^2 [J, L] y_b / 2`` of the coded Jacobian with the reference operator,
which is zero without terrain and first order in dt over it.  Measured at
rest over a 2 km mountain (T42, 20-level pressure_blend, rhs residual
4.27e-7 m/s^2): 8.0e-3 m/s and 0.25 K after 60 min at dt = 60 s, 0.18 m/s
and 1.5 K at dt = 300 s, while the fully explicit step measures 4.19e-4
m/s and 5.5e-4 K at both steps (the rhs residual's own response, the floor
of any consistent scheme).  The IMEX step measures 4.19e-4 / 5.5e-4 at
dt = 60 s and 4.20e-4 / 5.5e-4 at dt = 300 s: its own contribution is
1e-7 m/s and 4e-6 K (tests/test_arwen_global_imex.py, gate b).

THE METHOD.  For ``y' = E(y) + L y`` with stage values ``Y_i``:

    Y_i     = y_n + dt sum_{j<i} at_ij E(Y_j) + dt sum_{j<=i} a_ij L Y_j
    y_{n+1} = y_n + dt sum_i bt_i E(Y_i) + dt sum_i b_i L Y_i

Explicit tableau (``at``, ``bt``): the third-order SSP Runge-Kutta the
model already integrates with, in Butcher form,

    c  = (0, 1, 1/2)
    at = [[0, 0, 0], [1, 0, 0], [1/4, 1/4, 0]]
    bt = (1/6, 1/6, 2/3)

Implicit tableau (``a``, ``b``), diagonally implicit with an explicit
first stage, the SAME abscissae ``c`` and the SAME weights ``b = bt``:

    a  = [[0,        0,    0   ],
          [1 - a22,  a22,  0   ],
          [a31,      a32,  a33 ]],   a31 = 1/2 - a32 - a33

a three-parameter family ``(a22, a32, a33)``; the shipped member is

    a  = [[0, 0, 0], [7/10, 3/10, 0], [1/10, 1/10, 3/10]]

(``ssp3_tableau(0.3, 0.1, 0.3)``).  Two identical solves of ``(I - 3 dt
L / 10)`` per step, one cached Helmholtz inverse per total degree.

FIXED-POINT PROOF.  Let ``F(y_b) = 0`` for the full nonlinear right-hand
side, so ``E(y_b) = -L y_b`` exactly (``rhs`` subtracts the very operator
the solve integrates; tests/test_arwen_global_vertical_modes.py holds the
subtraction to 1e-12).  Induction on the stages with ``y_n = y_b``:
``Y_1 = y_b``; if ``Y_j = y_b`` for all ``j < i`` then

    Y_i - y_b = dt (sum_{j<i} a_ij - sum_{j<i} at_ij) L y_b + dt a_ii L Y_i
              = -dt a_ii L y_b + dt a_ii L Y_i

because ``sum_{j<i} at_ij = c_i = sum_{j<=i} a_ij``; so ``(I - dt a_ii L)
(Y_i - y_b) = 0`` and ``Y_i = y_b`` (``I - tau L`` is invertible for every
real ``tau``: the spectrum of ``L`` is imaginary).  Then ``y_{n+1} - y_b =
dt (sum b_i - sum bt_i) L y_b = 0``.  Nothing in the argument uses
linearity of ``E`` or smallness of ``L y_b``: every balanced state of the
nonlinear model, over any terrain, is a fixed point of the step to
roundoff, at every dt (gate a: 6.4e-17 relative on a uniform isothermal
column, 3.0e-18 1/s divergence increment).  The symmetric split fails
this at the first line: ``H y_b != y_b``.

WHY ``b = bt``.  Near balance, ``E = -L y + d`` with ``d`` the slow
tendency.  In the scalar model ``R(z_E, z_L) = 1 + (z_E + z_L) sum_i b_i
Y_i`` when ``b = bt``, and every ``Y_i = 1`` on the balanced line, so the
step's response to the slow tendency is ``1 + d`` exactly to first order
at EVERY ``omega dt``: the slow manifold is neither retarded nor damped by
the stiffness of ``L``.  A stiffly accurate implicit tableau (``b`` = its
last row) divides the slow response by ``(1 - a_ss z_L)`` instead.

WHY NOT A NEUTRAL IMPLICIT PART.  The natural first choice, an implicit
stability function that is a product of Crank-Nicolson factors (``a22 +
a33 = 1/2`` with the cubic and quadratic numerator terms matched so
``|R(iy)| = 1`` for every ``y``), is unstable at every dt when the
explicit residual has the OPPOSITE sign to the implicit operator, which
is exactly the over-covering regime the split requires (the reference
modes are faster than the atmosphere's, so ``E = J - L`` acts as ``-r L``
with ``0 < r < 1``): in the scalar model ``|R(-r i y, i y)| - 1 = +8e-5``
at ``r = 0.1, y = 0.5`` for ``a22 = a33 = 1/4``, growing as ``y^4``, and
on the coded operator the linearized rest ceilings were 5 to 20 s at
T533 for every member of that family (the shipped split: 395 s).  The
ARS(2,3,2) pair measured before this lane (+7e-5 per step at rest) fails
the same way.  Stability therefore requires an implicit part that DAMPS
what the explicit residual would amplify; the family above gives it up
in the numerator's cubic term.

THE MEMBER, BY MEASUREMENT.  For ``a22 = a33 = g`` the implicit stability
function is ``R(z) = N(z) / (1 - g z)^2`` with ``N`` cubic; its cubic
coefficient is ``(2/3) (a32 - 3g/4 + 3g^2/2)``.  A scalar-model search
over ``(a22, a32, a33)`` for zero growth on ``r y <= 1.6``, Doppler ``u
<= 0.75``, ``y <= 40`` converged on ``g = 0.335, a32 = 0.0825``, the
cubic-free member near ``g = 1/3, a32 = 1/12``; on the coded operator
(whole-rhs Jacobian about resting 286 / 300 / 320 / 340 K columns,
shipped hyperdiffusion, spectral radius over eight degrees, the ladder of
semi_implicit.py) that member measured 517 s at T533 on the 40-level
default but 15.8 s on the 20-level pressure_blend stack: every
cubic-free member (bounded at infinity) grows at EVERY small dt on that
stack without diffusion (+2.4e-4 per step at dt = 60 s, +1.8e-8 at 5 s,
the dt^4 coefficient of the coupled non-commuting step being positive),
and the scalar model cannot see it.  Moving ``a32`` 0.01 above the
cubic-free line removes the small-dt growth on both stacks and both
columns (roundoff, 1e-12, at every dt from 5 to 240 s) for ``g`` from
0.29 to 1/3; 0.02 above costs the ceiling (356 s at T533).  Along that
offset ``g = 0.30`` reads, minimum over the four columns with the shipped
hyperdiffusion:

    40-level surface_stretched (default)  T533  582.9 s  (shipped 395.3)
                                          T255 1247.8 s  (shipped 868.2)
    20-level pressure_blend               T533  452.9 s  (shipped 445.3)

Doppler growth at T533, dt = 60 s on the 20-level stack: 1.000000 at
U = 150 m/s (shipped 1.000745), 1.000018 at U = 50 (shipped 1.000020);
on the 40-level default 1.000000 at U = 150 (shipped 1.000100).

STIFF LIMIT.  ``R(z) = (1 + 2z/5 - z^2/100 + z^3/150) / (1 - 3z/10)^2``:
``|R(iy)|^2 = 1 - y^4/75 + O(y^6)`` (neutral to fourth order for resolved
modes, 0.6% at ``y = 1``, 5.9% at ``y = 2``, 1/3 at ``y = 10``) and
``|R(iy)| > 1`` beyond ``y = 17.33`` (IMPLICIT_NEUTRAL_LIMIT), where the
cubic term takes over and the implicit part alone amplifies the fastest
treated mode at the truncation; ``dynamics.step`` refuses a step with
``c_max k_T dt`` past that number by name (T21: dt above 14500 s; T533:
above 580 s, which is the ceiling above).

COST.  Three explicit right-hand sides per step, as SSPRK3; two implicit
solves per step, each one cached Helmholtz inverse per total degree in
the operator's modal basis plus two small matrix contractions; no
square-root maps.  Explicit tendencies are folded into the later stages
as they are produced, so at most two full accumulators live beside the
stage value.

PRECISION.  The stage values, the accumulators and the operator terms
are formed in the state's own precision: every tableau factor reaches
the arrays as a Python float, so a complex64 state stays complex64
through the step.  Before 2026-09-04 the factors were numpy float64
scalars and, under NumPy 2 promotion, silently lifted every accumulator
and stage value to complex128 (the stage solve then ran its Helmholtz
contractions in double, 26 ms of the 415 ms step at T255 on the RTX
5090, and the arrays were twice their size); under NumPy 1 the same
code stayed complex64.  The precision of the integrator is now defined
by the state, not by the numpy release; a run before this change and
one after differ in the last bits of the dynamics and are graded, not
bit-compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import SPECTRAL_FIELDS
from .profile import NULL_PROFILER
from .state import MoistHybridState


@dataclass(frozen=True)
class ImexTableau:
    """One IMEX Runge-Kutta pair with shared abscissae."""

    name: str
    explicit_a: tuple[tuple[float, ...], ...]
    explicit_b: tuple[float, ...]
    implicit_a: tuple[tuple[float, ...], ...]
    implicit_b: tuple[float, ...]

    def __post_init__(self) -> None:
        at = np.asarray(self.explicit_a, dtype=np.float64)
        a = np.asarray(self.implicit_a, dtype=np.float64)
        bt = np.asarray(self.explicit_b, dtype=np.float64)
        b = np.asarray(self.implicit_b, dtype=np.float64)
        stages = at.shape[0]
        if at.shape != (stages, stages) or a.shape != (stages, stages):
            raise ValueError("IMEX tableaux must be square and of one size")
        if bt.shape != (stages,) or b.shape != (stages,):
            raise ValueError("IMEX weights must have one entry per stage")
        if np.any(np.triu(at) != 0.0):
            raise ValueError("the explicit tableau must be strictly lower triangular")
        if np.any(np.triu(a, 1) != 0.0):
            raise ValueError("the implicit tableau must be lower triangular")
        if a[0, 0] != 0.0:
            raise ValueError("the first stage must be explicit (a_11 = 0)")
        if np.any(np.diag(a) < 0.0):
            raise ValueError("implicit diagonal entries must be nonnegative")
        # Matching abscissae: the fixed-point property of the module
        # docstring rests on every stage's explicit and implicit row sums
        # being equal, and on the two weight vectors summing to one.
        if not np.allclose(at.sum(axis=1), a.sum(axis=1), rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "IMEX tableaux must share every abscissa: a balanced state "
                "is otherwise not a fixed point of the step"
            )
        if abs(bt.sum() - 1.0) > 1.0e-14 or abs(b.sum() - 1.0) > 1.0e-14:
            raise ValueError("IMEX weights must each sum to one")

    @property
    def stages(self) -> int:
        return len(self.explicit_b)

    @property
    def abscissae(self) -> np.ndarray:
        return np.asarray(self.explicit_a, dtype=np.float64).sum(axis=1)

    def implicit_stability_function(self, z: complex) -> complex:
        """``R(z) = 1 + z b (I - z a)^-1 1`` of the implicit tableau."""
        a = np.asarray(self.implicit_a, dtype=np.float64)
        b = np.asarray(self.implicit_b, dtype=np.float64)
        s = self.stages
        w = np.linalg.solve(np.eye(s) - z * a, np.ones(s))
        return complex(1.0 + z * (b @ w))

    def implicit_neutral_limit(self) -> float:
        """The largest ``y`` with ``|R(i y')| <= 1`` for every ``y' <= y``
        (scanned to 1e3 in steps of 1e-2): the range of ``omega dt`` over
        which the implicit part alone never amplifies a mode of ``L``.
        ``inf`` when the whole axis is neutral or damping."""
        ys = np.arange(0.01, 1.0e3, 0.01)
        moduli = np.array([abs(self.implicit_stability_function(1j * y)) for y in ys])
        above = np.nonzero(moduli > 1.0 + 1.0e-12)[0]
        if above.size == 0:
            return float("inf")
        return float(ys[above[0]])

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stages": self.stages,
            "abscissae": [float(v) for v in self.abscissae],
            "explicit_a": [[float(v) for v in row] for row in self.explicit_a],
            "explicit_b": [float(v) for v in self.explicit_b],
            "implicit_a": [[float(v) for v in row] for row in self.implicit_a],
            "implicit_b": [float(v) for v in self.implicit_b],
            "implicit_neutral_limit": self.implicit_neutral_limit(),
        }


def ssp3_tableau(a22: float, a32: float, a33: float, *, name: str) -> ImexTableau:
    """Explicit SSPRK3 with an implicit tableau of matching abscissae and
    the same weights, parametrized by ``(a22, a32, a33)`` (``a21 = 1 -
    a22``, ``a31 = 1/2 - a32 - a33``): the search space of the module
    docstring."""
    a22 = float(a22)
    a32 = float(a32)
    a33 = float(a33)
    return ImexTableau(
        name=name,
        explicit_a=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.25, 0.25, 0.0)),
        explicit_b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        implicit_a=((0.0, 0.0, 0.0), (1.0 - a22, a22, 0.0), (0.5 - a32 - a33, a32, a33)),
        implicit_b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    )


#: The member the model integrates (chosen by measurement; module
#: docstring): a22 = a33 = 3/10, a32 = a31 = 1/10, a21 = 7/10.  Two
#: identical solves of (I - 3 dt/10 L) per step, one cached inverse;
#: implicit stability function (1 + 2z/5 - z^2/100 + z^3/150) /
#: (1 - 3z/10)^2, |R(iy)|^2 = 1 - y^4/75 + O(y^6): neutral to fourth
#: order, damping up to omega dt = 17.3 (IMPLICIT_NEUTRAL_LIMIT), the
#: refusal in dynamics.step beyond it.
IMEX_SSP3 = ssp3_tableau(0.3, 0.1, 0.3, name="imex_ssp3")

IMEX_TABLEAUX: dict[str, ImexTableau] = {IMEX_SSP3.name: IMEX_SSP3}
IMEX_INTEGRATORS = tuple(IMEX_TABLEAUX)
#: The [time] integrator a config gets when it names none.
DEFAULT_INTEGRATOR = IMEX_SSP3.name
#: omega dt beyond which the shipped member's implicit part amplifies
#: (module docstring): 17.3 for a22 = a33 = 3/10, a32 = 1/10.
IMPLICIT_NEUTRAL_LIMIT = {name: tableau.implicit_neutral_limit() for name, tableau in IMEX_TABLEAUX.items()}

_OPERATOR_FIELDS = ("divergence", "theta", "log_surface_pressure")
_OPERATOR_INDEX = tuple(SPECTRAL_FIELDS.index(name) for name in _OPERATOR_FIELDS)


def _add_operator(fields: list, linear, factor: float) -> None:
    """``fields[k] += factor * (L y)_k`` on the three operator rows."""
    if linear is None:
        return
    parts = (linear.divergence, linear.theta, linear.log_surface_pressure)
    factor = float(factor)
    for index, part in zip(_OPERATOR_INDEX, parts):
        if part is not None:
            fields[index] = fields[index] + factor * part


def imex_step(
    state: MoistHybridState,
    dt_s: float,
    rhs,
    scheme,
    transform,
    vertical,
    tableau: ImexTableau = IMEX_SSP3,
    *,
    profiler=None,
    mark=None,
):
    """One IMEX step of ``rhs`` (explicit, ``E = F - L``) and the scheme's
    operator ``L`` (implicit through ``scheme.solve_shifted``).

    Returns the advanced state and the semi-implicit metric: the largest
    divergence increment the implicit part contributed over the step,
    ``max |dt sum_i b_i (L Y_i)_D|``, the same reading the split schemes
    report for their maps.  ``profiler`` (woof.globe.profile)
    wraps each stage's solve, right-hand side, operator and
    accumulation in named sections; None is the no-op.

    ``mark(name, state)``, when given, is the energy ledger's observer
    (insitu.energy, 2026-09-04): the step's update is the sum of the
    explicit tendencies ``dt sum_i bt_i E(Y_i)`` and the implicit operator
    ``dt sum_i b_i L Y_i``, and the ledger reads the two apart by marking
    ``y_n`` plus the explicit sum alone as ``dynamics_explicit`` before the
    caller marks the advanced state (the implicit sum's net).  The
    decomposition costs one extra accumulator on sampled steps and changes
    no bit of the advanced state: ``final`` is accumulated exactly as
    without the observer.
    """
    prof = NULL_PROFILER if profiler is None else profiler
    dt = float(dt_s)
    at = np.asarray(tableau.explicit_a, dtype=np.float64)
    bt = np.asarray(tableau.explicit_b, dtype=np.float64)
    a = np.asarray(tableau.implicit_a, dtype=np.float64)
    b = np.asarray(tableau.implicit_b, dtype=np.float64)
    c = tableau.abscissae
    stages = tableau.stages
    t0 = state.time_s
    base = list(state.fields())
    # Accumulators: one per later stage plus the final combination.  Each
    # starts as the step's initial fields; tendencies are folded in as
    # they are produced so no stage tendency outlives its last use.
    accumulators = [None] + [list(base) for _ in range(1, stages)]
    final = list(base)
    explicit_only = list(base) if mark is not None else None
    implicit_divergence = None
    stage = state
    for i in range(stages):
        if i > 0:
            fields = accumulators[i]
            accumulators[i] = None
            stage = state.with_fields(fields, time_s=t0 + c[i] * dt)
            if a[i, i] != 0.0:
                with prof.section(f"stage{i}_solve"):
                    stage = scheme.solve_shifted(stage, transform, vertical, dt * a[i, i])
        with prof.section(f"stage{i}_rhs"):
            explicit = rhs(stage)
        # The operator at the stage, only where a later coefficient reads it.
        needs_operator = b[i] != 0.0 or any(a[k, i] != 0.0 for k in range(i + 1, stages))
        with prof.section(f"stage{i}_linear"):
            linear = scheme.linear_tendencies(stage, transform, vertical) if needs_operator else None
        accumulate = prof.section(f"stage{i}_accumulate")
        accumulate.__enter__()
        # Python floats: a numpy float64 scalar would lift a complex64
        # array to complex128 under NumPy 2 promotion (module docstring,
        # PRECISION).
        for k in range(i + 1, stages):
            target = accumulators[k]
            if at[k, i] != 0.0:
                factor = float(dt * at[k, i])
                for index, tendency in enumerate(explicit.fields()):
                    target[index] = target[index] + factor * tendency
            if a[k, i] != 0.0:
                _add_operator(target, linear, float(dt * a[k, i]))
        if bt[i] != 0.0:
            factor = float(dt * bt[i])
            for index, tendency in enumerate(explicit.fields()):
                final[index] = final[index] + factor * tendency
                if explicit_only is not None:
                    explicit_only[index] = explicit_only[index] + factor * tendency
        if b[i] != 0.0 and linear is not None:
            _add_operator(final, linear, float(dt * b[i]))
            contribution = float(dt * b[i]) * linear.divergence
            implicit_divergence = (
                contribution if implicit_divergence is None
                else implicit_divergence + contribution
            )
        accumulate.__exit__(None, None, None)
    if explicit_only is not None:
        mark("dynamics_explicit", state.with_fields(explicit_only, time_s=t0 + dt))
        del explicit_only
    advanced = state.with_fields(final, time_s=t0 + dt)
    if implicit_divergence is None:
        maximum = 0.0
    else:
        # The modulus and its maximum are reduced where the spectrum lives
        # and one scalar crosses: the host read copied the whole
        # divergence spectrum back (21 MB at T255) every step.  The
        # reading is a receipt metric, not a state; the device modulus of
        # a complex coefficient may differ from numpy's in its last bit.
        xp = transform.backend.xp
        maximum = float(transform.backend.to_numpy(
            xp.max(xp.abs(implicit_divergence))
        ))
    return advanced, {"semi_implicit_max_divergence_increment_s1": maximum}


__all__ = [
    "DEFAULT_INTEGRATOR",
    "IMEX_INTEGRATORS",
    "IMPLICIT_NEUTRAL_LIMIT",
    "IMEX_SSP3",
    "IMEX_TABLEAUX",
    "ImexTableau",
    "imex_step",
    "ssp3_tableau",
]
