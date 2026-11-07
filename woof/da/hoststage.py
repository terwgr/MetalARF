"""Host staging of Metal fields for the data-assimilation arithmetic.

The DA modules do their algebra on NumPy or CuPy.  A Metal state keeps its
fields in device storage that exposes only explicit transfers, so these
helpers read a field to the host (``to_numpy``) and the callers write the
result back through the array's own ``__setitem__``.  Float32/float64
elementwise arithmetic is IEEE on both hosts, so the words are the NumPy
reference's.
"""

from __future__ import annotations


def is_metal(array) -> bool:
    return type(array).__module__ == "woof.metal.array"


def host(array):
    """``array`` as a host array when it is Metal storage, else itself."""
    return array.to_numpy() if is_metal(array) else array


class HostView:
    """A read-only view of a state whose Metal fields are host copies."""

    def __init__(self, state):
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        return host(getattr(object.__getattribute__(self, "_state"), name))


def view(state):
    """``state`` itself unless it holds Metal storage."""
    return HostView(state) if is_metal(getattr(state, "p", None)) else state


class StagedState:
    """A state whose Metal fields are host copies that are written back.

    Reads of a Metal field return one cached host array, so in-place edits
    persist for the whole call; :meth:`commit` writes back the fields whose
    words changed.  Everything else is the state's own attribute.
    """

    def __init__(self, state):
        self._state = state
        self._copies = {}
        self._original = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        value = getattr(self._state, name)
        if not is_metal(value):
            return value
        if name not in self._copies:
            self._copies[name] = value.to_numpy()
            self._original[name] = self._copies[name].copy()
        return self._copies[name]

    def has_field(self, name: str) -> bool:
        """Check the field inventory without reading device words."""
        return getattr(self._state, name, None) is not None

    def total_theta(self):
        thb = self.thb
        return (thb if thb.ndim == 3 else thb[:, None, None]) + self.thp

    def total_mu(self):
        return self.mub2d + self.mup

    def commit(self) -> None:
        import numpy as np

        for name, copy in self._copies.items():
            if not np.array_equal(copy.view(np.uint8), self._original[name].view(np.uint8)):
                getattr(self._state, name)[...] = copy


def metal_context(state):
    """Return the device that owns a live or host-staged analysis state."""
    if isinstance(state, StagedState):
        state = state._state
    pressure = getattr(state, "p", None)
    return pressure.buffer.context if is_metal(pressure) else None
