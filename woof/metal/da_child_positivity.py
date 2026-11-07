"""Single-child positivity, with host-ordered scientific receipts."""
import numpy as np

from .array import MetalArray, MetalArrayModule
from woof.da.positivity import _receipt


def bound(prior, increments, policy):
    """Return bounded arrays and receipt, or None for unsupported inputs."""
    if policy not in ("mean-preserving", "clip") or not increments:
        return None
    arrays = list(prior.values()) + list(increments.values())
    if any(not isinstance(a, MetalArray) or a.dtype != np.dtype(np.float32)
           or not a.flags.c_contiguous or a.ndim not in (2, 3)
           or not a.size or a.size > np.iinfo(np.int32).max for a in arrays):
        return None
    first = arrays[0]
    ctx = first.buffer.context
    if any(a.shape != first.shape or a.buffer.context is not ctx for a in arrays):
        return None
    # One output per field and one temporary binary64 diagnostic per point.
    if (first.size * 8 > int(ctx.max_buffer_length)
            or first.size * (4 * len(increments) + 8)
            > int(ctx.recommended_working_set_size) // 16):
        return None
    xp = MetalArrayModule(ctx)
    adjusted = {}
    per_field = []
    total_points = 0
    total_mass = 0.0
    for name, increment in increments.items():
        background = prior[name]
        out = xp.empty(first.shape, dtype=np.float32)
        negative = xp.empty(first.shape, dtype=np.uint64)
        ctx.submit_kernel(
            "da_child_positivity",
            {"background": background.view((first.size,)),
             "increment": increment.view((first.size,)),
             "out": out.view((first.size,)),
             "negative": negative.view((first.size,))},
            {"n": first.size}, exact_policy=False,
            grid_size=first.size, threads_per_group=128)
        words = negative.to_numpy().reshape(-1)
        if np.any(words == np.uint64(0x7ff8000000000000)):
            return None
        values = words[words != 0].view(np.float64)
        count = int(values.size)
        mass = float(-values.sum()) if count else 0.0
        worst = float(values.min()) if count else 0.0
        if policy == "clip":
            entry = {"field": name, "policy": policy,
                     "negative_points": count, "mass_added_by_clip": mass,
                     "mass_left_negative": 0.0, "total_points": first.size,
                     "worst_negative": worst}
        else:
            entry = {"field": name, "policy": policy, "member_axis": False,
                     "negative_points": count, "total_points": first.size,
                     "worst_negative": worst, "mass_clip_would_add": mass,
                     "mass_added": mass, "mass_redistributed": 0.0,
                     "cells_touched": count, "cells_mean_negative": count,
                     "rounding_floor_points": 0}
            if count:
                entry["semantics"] = ("one member: no ensemble mean to keep, "
                                      "so the rule is the clip")
        adjusted[name] = out if count else increment
        per_field.append(entry)
        total_points += count
        total_mass += mass
    names = tuple(increments)
    return adjusted, _receipt(policy, names, names, per_field,
                              total_points, total_mass)
