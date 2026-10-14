"""Persistent, signed-object tracking for the WK82 split diagnostic.

The persistent WK82 split measurement uses fixed identities seeded only after
a unique coherent opposite-sign pair persists for three samples. Canonical qualification uses the source
instantaneous two-strongest measurement. Persistent identities provide
additional diagnostics and do not replace it or select its members.

Inputs are host NumPy planes on the same mass grid. ``flow_u`` and ``flow_v``
are the layer-mean 0--6 km steering winds in m/s.  ``w5`` and ``zeta5`` are
the source-selected 5-km planes; ``uh2d`` is the source 2--5 km UH integral.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np


@dataclass
class _Object:
    component_id: int
    cells: np.ndarray
    mask: np.ndarray
    rotation_cells: np.ndarray
    x_m: float
    y_m: float
    peak_w: float
    sign: int
    zeta_sign: int
    uh_sign: int
    zeta_dominance: float
    uh_coherence: float
    flow_u: float
    flow_v: float
    flow_spread: float
    radius_m: float

    def public(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "peak_w": self.peak_w,
            "zeta_sign": self.zeta_sign,
            "uh_sign": self.uh_sign,
            "zeta_dominance": self.zeta_dominance,
            "uh_coherence": self.uh_coherence,
            "rotation_confident": self.sign != 0,
            "rotation_support_cells": len(self.rotation_cells),
            "flow_u": self.flow_u,
            "flow_v": self.flow_v,
            "radius_m": self.radius_m,
        }


@dataclass
class _Track:
    label: str
    sign: int
    obj: _Object
    time_s: float
    previous_obj: _Object | None = None
    previous_time_s: float | None = None


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _components(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    """Return 4-connected components as ``(n, 2)`` row/column arrays."""
    seen = np.zeros(mask.shape, dtype=bool)
    found: list[np.ndarray] = []
    ny, nx = mask.shape
    for j0, i0 in zip(*np.nonzero(mask)):
        if seen[j0, i0]:
            continue
        seen[j0, i0] = True
        stack = [(int(j0), int(i0))]
        cells: list[tuple[int, int]] = []
        while stack:
            j, i = stack.pop()
            cells.append((j, i))
            for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
                if (0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]
                        and not seen[jj, ii]):
                    seen[jj, ii] = True
                    stack.append((jj, ii))
        if len(cells) >= minimum:
            found.append(np.asarray(cells, dtype=np.int64))
    return found


class _SingleStormTracker:
    """Track one persistent opposite-sign pair through successive frames.

    A seed must be the sole eligible positive and negative component for
    ``seed_samples`` consecutive frames.  Before seeding, ambiguous or absent
    candidates simply restart confirmation. Once seeded, net vorticity and UH
    signs must remain unchanged, while footprint continuity can preserve an
    identity through weaker instantaneous coherence. Analysis checkpoints
    require the original seed confidence. Lost, merged, reversed or ambiguous
    identities fail closed.

    Motion gates are dimensional: expected displacement blends supplied
    layer-mean flow with the prior observed centroid velocity; residual
    allowance is the two component radii plus one cell diagonal for
    rasterization. Footprint overlap is computed after cellwise flow
    advection. No fixed-distance radius or peak-rank term is used.
    """

    def __init__(self, dx: float, dy: float, shear_uv, *, threshold: float = 15.0,
                 min_cells: int = 3, seed_samples: int = 3,
                 max_seed_gap_s: float = 180.0,
                 uh_coherence_min: float = 0.5,
                 zeta_dominance_min: float = 0.75):
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive")
        shear = np.asarray(shear_uv, dtype=np.float64)
        if shear.shape != (2,) or not np.isfinite(shear).all() or np.hypot(*shear) == 0:
            raise ValueError("shear_uv must be a finite nonzero (u, v) vector")
        if min_cells < 1 or seed_samples < 1:
            raise ValueError("min_cells and seed_samples must be positive")
        self.dx, self.dy = float(dx), float(dy)
        self.shear_uv = shear
        self.threshold = float(threshold)
        self.min_cells = int(min_cells)
        self.seed_samples = int(seed_samples)
        self.max_seed_gap_s = float(max_seed_gap_s)
        self.uh_coherence_min = float(uh_coherence_min)
        self.zeta_dominance_min = float(zeta_dominance_min)
        self.tracks: dict[str, _Track] = {}
        self._seed_pair: tuple[_Object, _Object] | None = None
        self._seed_count = 0
        self._seed_last_t: float | None = None
        self._failed = False
        self.trace: list[dict[str, Any]] = []

    def _extract(self, w, zeta, uh, fu, fv) -> list[_Object]:
        arrays = [np.asarray(a, dtype=np.float64) for a in (w, zeta, uh, fu, fv)]
        if any(a.ndim != 2 for a in arrays) or any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("w5, zeta5, uh2d, flow_u, and flow_v must be same-shape 2-D planes")
        w, zeta, uh, fu, fv = arrays
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("tracker input planes must contain only finite values")
        objects = []
        for cid, cells in enumerate(_components(w > self.threshold, self.min_cells)):
            rows, cols = cells[:, 0], cells[:, 1]
            weights = w[rows, cols]
            total_w = float(weights.sum())
            if total_w <= 0:
                continue
            # Match the legacy source centroid: thresholded w values, not w-15.
            x = float(np.sum(cols * weights) / total_w) * self.dx
            y = float(np.sum(rows * weights) / total_w) * self.dy
            zz = zeta[rows, cols]
            h = uh[rows, cols]
            znet, hnet = float(zz.sum()), float(h.sum())
            zs = _sign(znet)
            zdom = (float(np.count_nonzero(np.sign(zz) == zs) / len(zz))
                    if zs != 0 and len(zz) else 0.0)
            hcoh = abs(hnet) / float(np.abs(h).sum()) if np.abs(h).sum() else 0.0
            hs = _sign(hnet)
            sign = zs if (zs != 0 and zs == hs
                          and zdom >= self.zeta_dominance_min
                          and hcoh >= self.uh_coherence_min) else 0
            flows_u, flows_v = fu[rows, cols], fv[rows, cols]
            mean_u = float(np.average(flows_u, weights=weights))
            mean_v = float(np.average(flows_v, weights=weights))
            spread = float(np.sqrt(np.average(
                (flows_u - mean_u) ** 2 + (flows_v - mean_v) ** 2,
                weights=weights)))
            mask = np.zeros(w.shape, dtype=bool)
            mask[rows, cols] = True
            rotation_cells = cells[(np.sign(zz) == zs) & (np.sign(h) == zs)]
            if zs == 0:
                rotation_cells = cells[:0]
            radius = math.sqrt(len(cells) / math.pi) * math.sqrt(self.dx * self.dy)
            objects.append(_Object(
                cid, cells, mask, rotation_cells, x, y, float(weights.max()), sign, zs, hs,
                zdom, hcoh, mean_u, mean_v, spread, radius))
        return objects

    def _advected_mask(self, obj: _Object, dt: float, fu, fv, *, cells=None) -> np.ndarray:
        cells = obj.cells if cells is None else cells
        rows, cols = cells[:, 0], cells[:, 1]
        # Cellwise flow shift preserves footprint deformation from shear.
        new_cols = np.rint(cols + fu[rows, cols] * dt / self.dx).astype(int)
        new_rows = np.rint(rows + fv[rows, cols] * dt / self.dy).astype(int)
        valid = ((new_rows >= 0) & (new_rows < obj.mask.shape[0])
                 & (new_cols >= 0) & (new_cols < obj.mask.shape[1]))
        out = np.zeros_like(obj.mask)
        out[new_rows[valid], new_cols[valid]] = True
        return out

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        union = int(np.count_nonzero(a | b))
        return float(np.count_nonzero(a & b) / union) if union else 0.0

    def _prediction(self, track: _Track, t: float) -> tuple[float, float, np.ndarray, float]:
        old = track.obj
        dt = t - track.time_s
        flow_u, flow_v = old.flow_u, old.flow_v
        uncertainty = old.radius_m + math.sqrt(self.dx ** 2 + self.dy ** 2)
        if track.previous_obj is not None and track.previous_time_s is not None:
            observed_dt = track.time_s - track.previous_time_s
            if observed_dt > 0:
                observed_u = (old.x_m - track.previous_obj.x_m) / observed_dt
                observed_v = (old.y_m - track.previous_obj.y_m) / observed_dt
                # The motion prior blends layer steering and recent object motion.
                # The residual allowance stays tied to object scale; within-object
                # wind spread is retained in the trace but does not inflate it.
                flow_u = 0.5 * (flow_u + observed_u)
                flow_v = 0.5 * (flow_v + observed_v)
        predicted_x = old.x_m + flow_u * dt
        predicted_y = old.y_m + flow_v * dt
        limit = uncertainty
        return predicted_x, predicted_y, self._advected_mask(old, dt, self._flow_u,
                                                               self._flow_v), limit

    def _seed(self, candidates: list[_Object], t: float, eligible) -> dict[str, Any]:
        pos = [o for o in eligible if o.sign > 0]
        neg = [o for o in eligible if o.sign < 0]
        if len(pos) != 1 or len(neg) != 1:
            self._seed_count = 0
            self._seed_pair = None
            self._seed_last_t = None
            return {"time_s": t, "status": "awaiting_unique_seed",
                    "candidate_count": {"positive": len(pos), "negative": len(neg)}}
        pair = (pos[0], neg[0])
        if self._seed_pair is not None:
            dt = t - float(self._seed_last_t)
            if dt <= 0 or dt > self.max_seed_gap_s:
                self._seed_count = 1
            else:
                pairs_ok = True
                for prev, curr in zip(self._seed_pair, pair):
                    limit = (prev.radius_m + curr.radius_m
                             + math.sqrt(self.dx ** 2 + self.dy ** 2))
                    pred_x, pred_y = prev.x_m + prev.flow_u * dt, prev.y_m + prev.flow_v * dt
                    if math.hypot(curr.x_m - pred_x, curr.y_m - pred_y) > limit:
                        pairs_ok = False
                        break
                self._seed_count = self._seed_count + 1 if pairs_ok else 1
        else:
            self._seed_count = 1
        self._seed_pair = pair
        self._seed_last_t = t
        if self._seed_count < self.seed_samples:
            return {"time_s": t, "status": "confirming_seed",
                    "seed_count": self._seed_count, "seed_required": self.seed_samples,
                    "positive": pair[0].public(), "negative": pair[1].public()}
        # R/L labels are assigned once from the source right-normal projection.
        right_normal = np.array([self.shear_uv[1], -self.shear_uv[0]])
        ordered = sorted(pair, key=lambda o: o.x_m * right_normal[0]
                         + o.y_m * right_normal[1], reverse=True)
        self.tracks = {
            "right": _Track("right", ordered[0].sign, ordered[0], t),
            "left": _Track("left", ordered[1].sign, ordered[1], t),
        }
        return self._result(t, "seeded", {
            "right": {**ordered[0].public(), "confidence": "seeded"},
            "left": {**ordered[1].public(), "confidence": "seeded"},
        }, {"seed_persistence": self._seed_count})

    def update(self, t_s: float, w5, zeta5, uh2d, flow_u, flow_v) -> dict[str, Any]:
        return self._update(t_s, w5, zeta5, uh2d, flow_u, flow_v, {})

    def _update(self, t_s: float, w5, zeta5, uh2d, flow_u, flow_v,
                forced_choices: dict[str, int]) -> dict[str, Any]:
        """Consume one saved/streamed frame and return a JSON-ready diagnostic."""
        t = float(t_s)
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        if self.trace and t <= self.trace[-1]["time_s"]:
            raise ValueError("frame times must be strictly increasing")
        objects = self._extract(w5, zeta5, uh2d, flow_u, flow_v)
        current_fu = np.asarray(flow_u, dtype=np.float64)
        current_fv = np.asarray(flow_v, dtype=np.float64)
        eligible = [o for o in objects if o.sign]
        if self._failed:
            return self._append({"time_s": t, "status": "failed_closed",
                                 "reason": "tracker already failed"})
        if not self.tracks:
            result = self._seed(objects, t, eligible)
            self._flow_u, self._flow_v = current_fu.copy(), current_fv.copy()
            if result.get("status") == "seeded":
                for name, track in self.tracks.items():
                    reliable = len(track.obj.rotation_cells) >= self.min_cells
                    result["members"][name]["rotation_support_confident"] = reliable
                    if not reliable:
                        result["members"][name]["rotation_confident"] = False
                        result["status"] = "weak_rotation"
            return self._append(result)

        if any(t - track.time_s > self.max_seed_gap_s for track in self.tracks.values()):
            return self._fail(t, "missing_frames", {"max_gap_s": self.max_seed_gap_s})

        # A component that intersects both advected footprints is a merge.
        predictions = {name: self._prediction(track, t)
                       for name, track in self.tracks.items()}
        for obj in objects:
            touched = [name for name, (_, _, pmask, _) in predictions.items()
                       if self._iou(pmask, obj.mask) > 0.0]
            if len(touched) == 2:
                return self._fail(t, "merged", {"component_id": obj.component_id,
                                                   "tracks": touched})

        assignments: dict[str, tuple[_Object, float, float, float, dict[str, Any]]] = {}
        for name, track in self.tracks.items():
            # The full updraft retains its original geometry, but its mixed
            # periphery is not a witness for the rotating core's identity.
            # Advect prior cells with both local signs matching that identity
            # and intersect current FULL components, so a true flip is visible.
            support_mask = self._advected_mask(
                track.obj, t - track.time_s, self._flow_u, self._flow_v,
                cells=track.obj.rotation_cells)
            support_count = int(np.count_nonzero(support_mask))
            # A sparse witness lowers confidence without proving identity
            # loss. It cannot exclude competing full-footprint continuations.
            prior_support_reliable = (
                len(track.obj.rotation_cells) >= self.min_cells
                and support_count >= self.min_cells)
            same = [o for o in objects if o.zeta_sign == track.sign
                    and o.uh_sign == track.sign]
            if not same:
                px, py, _, _ = predictions[name]
                dt = t - track.time_s
                invalid_nearby = [o for o in objects if o.sign == 0 and
                    math.hypot(o.x_m - px, o.y_m - py) <=
                    track.obj.radius_m + o.radius_m
                    + math.sqrt(self.dx ** 2 + self.dy ** 2)]
                if invalid_nearby:
                    return self._fail(t, "mixed_rotation", {"track": name,
                        "expected_sign": track.sign,
                        "candidate_ids": [o.component_id for o in invalid_nearby]})
                return self._fail(t, "lost_or_mixed", {"track": name,
                    "expected_sign": track.sign,
                    "candidates": [o.public() for o in objects]})
            px, py, pmask, _ = predictions[name]
            scored = []
            for obj in same:
                residual = math.hypot(obj.x_m - px, obj.y_m - py)
                limit = predictions[name][3] + obj.radius_m
                iou = self._iou(pmask, obj.mask)
                # Reject by a physical residual bound; rank remaining matches
                # by normalized residual plus advected-footprint disagreement.
                # A motion prior cannot by itself establish object lineage.
                # Even a sole nearby same-sign object must continue the
                # advected footprint; otherwise it may be a replacement cell.
                if residual <= limit and iou > 0.0:
                    score = residual / limit + (1.0 - iou)
                    scored.append((score, obj, residual, iou, limit))
            if not scored:
                return self._fail(t, "lost", {"track": name,
                    "prediction_m": [px, py], "candidate_ids": [o.component_id for o in same]})
            # Resolve geometry before rotation can exclude a continuation.
            # Use the same unique-overlap and one-cell residual margin as the
            # signed split matcher, including reversed or mixed children.
            invalid_scored = []
            for obj in objects:
                if obj.zeta_sign == track.sign and obj.uh_sign == track.sign:
                    continue
                residual = math.hypot(obj.x_m - px, obj.y_m - py)
                limit = predictions[name][3] + obj.radius_m
                iou = self._iou(pmask, obj.mask)
                if (residual <= limit and iou > 0.0
                        and (not prior_support_reliable
                             or np.any(support_mask & obj.mask))):
                    invalid_scored.append((residual / limit + (1.0 - iou),
                                           obj, residual, iou, limit))
            if invalid_scored:
                all_scored = sorted((*scored, *invalid_scored), key=lambda x: x[0])
                by_overlap = sorted(all_scored, key=lambda x: x[3], reverse=True)
                winner = all_scored[0][1]
                residual_advantage = min(x[2] for x in all_scored[1:]) - all_scored[0][2]
                clear_same_sign = (
                    winner.zeta_sign == track.sign and winner.uh_sign == track.sign
                    and by_overlap[0][1] is winner
                    and by_overlap[0][3] > by_overlap[1][3]
                    and residual_advantage > math.hypot(self.dx, self.dy))
                if not clear_same_sign:
                    return self._fail(t, "mixed_or_reversed_continuation", {
                        "track": name, "expected_sign": track.sign,
                        "candidate_ids": [x[1].component_id for x in invalid_scored],
                        "same_sign_candidate_ids": [x[1].component_id for x in scored],
                        "residual_advantage_m": residual_advantage,
                    })
            scored.sort(key=lambda x: x[0])
            if len(scored) > 1:
                # A split can retain the parent only when the score-best child
                # is also the uniquely dominant continuation of the advected
                # footprint. Centroid score alone cannot choose between siblings.
                by_overlap = sorted(scored, key=lambda x: x[3], reverse=True)
                best_iou = by_overlap[0][3]
                next_iou = by_overlap[1][3]
                residual_advantage = min(x[2] for x in scored[1:]) - scored[0][2]
                overlap_is_clear = (best_iou > next_iou and residual_advantage
                                    > math.hypot(self.dx, self.dy))
                score_best_is_overlap_best = by_overlap[0][1] is scored[0][1]
                forced_id = forced_choices.get(name)
                if forced_id is not None:
                    selected = [x for x in scored if x[1].component_id == forced_id]
                    if not selected:
                        return self._fail(t, "lost", {"track": name,
                            "forced_component_id": forced_id,
                            "candidate_ids": [x[1].component_id for x in scored]})
                    scored = selected
                elif not (overlap_is_clear and score_best_is_overlap_best):
                    return self._fail(t, "ambiguous", {"track": name,
                        "candidate_ids": [x[1].component_id for x in scored],
                        "scores": [x[0] for x in scored],
                        "overlaps": [x[3] for x in scored],
                        "residual_advantage_m": residual_advantage,
                        "reason": "centroid and footprint do not resolve a unique continuation"})
            score, obj, residual, iou, limit = scored[0]
            support_overlap = int(np.count_nonzero(support_mask & obj.mask))
            if prior_support_reliable and not support_overlap:
                return self._fail(t, "rotation_support_lost", {
                    "track": name, "component_id": obj.component_id,
                    "advected_support_cells": support_count})
            support_confident = (prior_support_reliable
                                 and len(obj.rotation_cells) >= self.min_cells)
            confidence = max(0.0, min(1.0, 0.5 * iou + 0.5 * (1.0 - residual / limit)))
            assignments[name] = (obj, score, iou, confidence, {
                "rotation_support_confident": support_confident,
                "advected_rotation_support_cells": support_count,
                "rotation_support_overlap_cells": support_overlap,
                "candidate_ids": [x[1].component_id for x in scored],
                "candidate_scores": [x[0] for x in scored],
                "best_score_gap": (scored[1][0] - scored[0][0]) if len(scored) > 1 else None,
                "clear_split_children": [x[1].component_id for x in scored
                                         if x[3] > 0.0] if len(scored) > 1 else [],
            })

        # The two fixed identities may not collapse onto the same component.
        if assignments["right"][0].component_id == assignments["left"][0].component_id:
            return self._fail(t, "merged", {"component_id": assignments["right"][0].component_id})

        members = {}
        diagnostics = {}
        for name, (obj, score, iou, confidence, match_detail) in assignments.items():
            track = self.tracks[name]
            track.previous_obj, track.previous_time_s = track.obj, track.time_s
            track.obj, track.time_s = obj, t
            members[name] = {**obj.public(), "identity_sign": track.sign,
                             "confidence": confidence,
                             "rotation_support_confident": match_detail["rotation_support_confident"],
                             "rotation_confident": bool(obj.sign
                                 and match_detail["rotation_support_confident"])}
            diagnostics[name] = {"match_score": score, "advected_iou": iou,
                                 **match_detail}
        self._flow_u, self._flow_v = current_fu.copy(), current_fv.copy()
        status = ("matched" if all(m["rotation_confident"] for m in members.values())
                  else "weak_rotation")
        return self._append(self._result(t, status, members, diagnostics))

    def _result(self, t: float, status: str, members, diagnostics) -> dict[str, Any]:
        separation = (members["left"]["y_m"] - members["right"]["y_m"]
                      if members["left"]["zeta_sign"] != 0 else float("nan"))
        return {"time_s": t, "status": status, "members": members,
                "separation_m": float(separation), "diagnostics": diagnostics}

    def _fail(self, t: float, reason: str, detail: dict[str, Any]) -> dict[str, Any]:
        self._failed = True
        return self._append({"time_s": t, "status": reason, "reason": reason,
                             "details": detail, "members": None,
                             "separation_m": None})

    def _append(self, result: dict[str, Any]) -> dict[str, Any]:
        self.trace.append(result)
        return result

    def trace_json(self, **kwargs) -> str:
        """Serialize all per-frame results with stable key ordering."""
        return json.dumps(self.trace, sort_keys=True, allow_nan=False, **kwargs)



class WK82StormTracker:
    """Track every admissible continuation when a split is unresolved.

    The single-path matcher still resolves cases where centroid residual and
    advected-footprint overlap identify the same child beyond one-cell raster
    uncertainty. Otherwise this wrapper forks one hypothesis per admissible
    child and carries all branches forward. A lost, merged, or sign-reversed
    branch fails the diagnostic closed; branches are never discarded to
    improve a reported separation.
    """

    _NORMAL = {"awaiting_unique_seed", "confirming_seed", "seeded", "matched",
               "weak_rotation"}

    def __init__(self, dx: float, dy: float, shear_uv, *, threshold: float = 15.0,
                 min_cells: int = 3, seed_samples: int = 3,
                 max_seed_gap_s: float = 180.0,
                 uh_coherence_min: float = 0.5,
                 zeta_dominance_min: float = 0.75,
                 max_hypotheses: int = 32):
        import copy
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be positive")
        self._copy = copy
        self._kwargs = dict(threshold=threshold, min_cells=min_cells,
                            seed_samples=seed_samples,
                            max_seed_gap_s=max_seed_gap_s,
                            uh_coherence_min=uh_coherence_min,
                            zeta_dominance_min=zeta_dominance_min)
        self.dx, self.dy = float(dx), float(dy)
        self.shear_uv = np.asarray(shear_uv, dtype=np.float64)
        self.max_hypotheses = int(max_hypotheses)
        self._paths: list[tuple[int, _SingleStormTracker]] = [(0, self._new_tracker())]
        self._next_path_id = 1
        self._last_time_s: float | None = None
        self._fatal = False
        self._fatal_detail: dict[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []

    def _new_tracker(self):
        return _SingleStormTracker(self.dx, self.dy, self.shear_uv, **self._kwargs)

    def _extract(self, *args, **kwargs):
        """Retain the private feature extractor used by CPU inspection tools."""
        return self._paths[0][1]._extract(*args, **kwargs)

    def _advance_path(self, path_id, prior, frame, events, budget):
        """Expand ambiguity recursively from an untouched pre-frame snapshot."""
        pending = [(prior, {})]
        outputs = []
        while pending:
            snapshot, forced = pending.pop()
            trial = self._copy.deepcopy(snapshot)
            result = trial._update(*frame, forced)
            if result.get("status") == "ambiguous":
                detail = result.get("details", {})
                track = detail.get("track")
                ids = list(detail.get("candidate_ids", ()))
                events.append({"path_id": path_id, **detail, "track": track})
                if not track or not ids:
                    outputs.append((path_id, trial, result))
                    continue
                if (budget[0] - 1 + len(pending) + len(outputs) + len(ids)
                        > self.max_hypotheses):
                    return [], {"reason": "hypothesis_cap", "limit": self.max_hypotheses,
                                "candidate_ids": ids, "track": track,
                                "path_id": path_id}
                for component_id in ids:
                    branch = self._copy.deepcopy(snapshot)
                    branch_choices = dict(forced)
                    branch_choices[track] = int(component_id)
                    pending.append((branch, branch_choices))
                continue
            result = {**result, "parent_path_id": path_id,
                      "branch_choices": dict(forced)}
            outputs.append((path_id, trial, result))
        budget[0] += max(0, len(outputs) - 1)
        return outputs, None

    def update(self, t_s: float, w5, zeta5, uh2d, flow_u, flow_v) -> dict[str, Any]:
        t = float(t_s)
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        if self._last_time_s is not None and t <= self._last_time_s:
            raise ValueError("frame times must be strictly increasing")
        frame = (t, w5, zeta5, uh2d, flow_u, flow_v)
        self._last_time_s = t
        if self._fatal:
            return self._append({"time_s": t, "status": "failed_closed",
                                 "reason": "a retained hypothesis failed",
                                 "details": self._fatal_detail,
                                 "members": None, "separation_m": None})

        expanded, branch_events, failures = [], [], []
        budget = [len(self._paths)]
        for path_id, prior in self._paths:
            outputs, cap_error = self._advance_path(path_id, prior, frame,
                                                     branch_events, budget)
            if cap_error:
                self._fatal = True
                self._fatal_detail = cap_error
                return self._append({"time_s": t, "status": "failed_closed",
                    "reason": "hypothesis_cap", "details": cap_error,
                    "branch_events": branch_events, "members": None,
                    "separation_m": None, "hypothesis_count": budget[0]})
            for out_id, tracker, result in outputs:
                if result.get("status") not in self._NORMAL:
                    failures.append({"path_id": out_id, "result": result})
                else:
                    if out_id == path_id and len(outputs) == 1:
                        new_id = path_id
                    else:
                        new_id = self._next_path_id
                        self._next_path_id += 1
                    expanded.append((new_id, tracker, result))
        if failures:
            self._fatal = True
            self._fatal_detail = {"reason": "hypothesis_failed", "failures": failures}
            rows = [self._hypothesis_row(pid, row) for pid, _, row in expanded]
            return self._append({"time_s": t, "status": "failed_closed",
                "reason": "hypothesis_failed", "details": self._fatal_detail,
                "branch_events": branch_events, "hypotheses": rows,
                "hypothesis_count": len(rows) + len(failures),
                "members": None, "separation_m": None})

        # No branch is pruned or deduplicated by its output geometry.
        if len(expanded) > self.max_hypotheses:
            self._fatal = True
            self._fatal_detail = {"reason": "hypothesis_cap",
                                  "limit": self.max_hypotheses,
                                  "count": len(expanded)}
            return self._append({"time_s": t, "status": "failed_closed",
                "reason": "hypothesis_cap", "details": self._fatal_detail,
                "branch_events": branch_events, "hypothesis_count": len(expanded),
                "members": None, "separation_m": None})
        self._paths = [(pid, tracker) for pid, tracker, _ in expanded]
        rows = [self._hypothesis_row(pid, result)
                for pid, _, result in expanded]
        if not rows:
            self._fatal = True
            self._fatal_detail = {"reason": "no_admissible_hypotheses"}
            return self._append({"time_s": t, "status": "failed_closed",
                "details": self._fatal_detail, "members": None,
                "separation_m": None, "hypotheses": []})
        separations = [row["separation_m"] for row in rows
                       if row.get("separation_m") is not None
                       and np.isfinite(row["separation_m"])]
        if len(rows) == 1:
            result = dict(rows[0])
            result.update(time_s=t, hypothesis_count=1,
                          hypotheses=rows, branch_events=branch_events)
            return self._append(result)
        all_confident = all(
            row.get("status") != "weak_rotation"
            and row.get("members") is not None
            and all(member.get("rotation_confident", False)
                    for member in row["members"].values()) for row in rows)
        return self._append({
            "time_s": t,
            "status": "bounded" if all_confident else "weak_rotation",
            "hypothesis_count": len(rows), "hypotheses": rows,
            "members": None, "separation_m": None,
            "separation_min_m": min(separations) if separations else None,
            "separation_max_m": max(separations) if separations else None,
            "branch_events": branch_events,
        })

    @staticmethod
    def _hypothesis_row(path_id: int, result: dict[str, Any]) -> dict[str, Any]:
        row = dict(result)
        row["path_id"] = path_id
        return row

    def _append(self, result):
        self.trace.append(result)
        return result

    def trace_json(self, **kwargs) -> str:
        return json.dumps(self.trace, sort_keys=True, allow_nan=False, **kwargs)
