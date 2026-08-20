"""Deterministic native-DOF MuJoCo IK candidates for strict bimanual search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

from .strict_bimanual_ik import IKCandidate


def _normalize_target_quaternion(value):
    quaternion = np.asarray(value, dtype=float)
    if quaternion.shape != (4,) or np.any(~np.isfinite(quaternion)):
        raise ValueError("target quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("target quaternion must be nonzero")
    return quaternion / norm


def _world_rotation_error(target_quaternion, current_quaternion, current_matrix=None):
    local = np.empty(3)
    mujoco.mju_subQuat(local, target_quaternion, current_quaternion)
    if current_matrix is None:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, current_quaternion)
        current_matrix = matrix.reshape(3, 3)
    return np.asarray(current_matrix).reshape(3, 3) @ local


def normalized_pose_residual(
        position_error_m, orientation_error_rad, *,
        position_tolerance_m=.001,
        orientation_tolerance_rad=np.deg2rad(1.5)):
    """Scale translation and rotation by their declared strict tolerances."""
    if position_tolerance_m <= 0 or orientation_tolerance_rad <= 0:
        raise ValueError("pose tolerances must be positive")
    position = np.atleast_1d(np.asarray(position_error_m, dtype=float))
    orientation = np.atleast_1d(np.asarray(
        orientation_error_rad, dtype=float))
    return np.r_[position / position_tolerance_m,
                 orientation / orientation_tolerance_rad]


def stratified_joint_seeds(lower, upper):
    """Deterministically cover PiperX shoulder, elbow and wrist branches."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("joint bounds must be matching one-dimensional arrays")
    if len(lower) != 6:
        raise ValueError("PiperX stratified seeds require six joints")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("stratified joint bounds must be finite")
    if np.any(upper <= lower):
        raise ValueError("each upper joint bound must exceed its lower bound")
    midpoint = (lower + upper) / 2.0
    span = upper - lower
    seeds = [midpoint]
    wrist_branches = ((.2, .2), (.2, .8), (.8, .2), (.8, .8))
    for shoulder_fraction in (.2, .8):
        for elbow_fraction in (.2, .8):
            for wrist4_fraction, wrist5_fraction in wrist_branches:
                seed = midpoint.copy()
                seed[0] = lower[0] + shoulder_fraction * span[0]
                seed[2] = lower[2] + elbow_fraction * span[2]
                seed[3] = lower[3] + wrist4_fraction * span[3]
                seed[4] = lower[4] + wrist5_fraction * span[4]
                seeds.append(seed)
    return tuple(seeds)


def piperx_wrist_risk(q, *, critical_angle_rad=np.deg2rad(89.0)):
    """Barrier-like score for PiperX J4/J5 near their physical hard stops."""
    values = np.asarray(q, dtype=float)
    if values.shape != (6,):
        raise ValueError("PiperX wrist risk requires six joints")
    if critical_angle_rad <= 0:
        raise ValueError("critical wrist angle must be positive")
    ratio = np.abs(values[3:5]) / critical_angle_rad
    remaining = np.maximum(1.0 - ratio, .005)
    return float(np.sum(ratio ** 8 / remaining))


@dataclass(frozen=True)
class CandidateGeneratorConfig:
    position_tolerance_m: float = .001
    orientation_tolerance_rad: float = np.deg2rad(1.5)
    damping: float = .04
    step_scale: float = .7
    max_iterations: int = 240
    dedup_rad: float = 1e-3
    maximum_candidates: int = 8
    global_seed_count: int = 12
    orientation_weight: float = 1.0
    constrained_fallback_enabled: bool = True
    constrained_fallback_seed_count: int = 12
    constrained_fallback_max_iterations: int = 180
    reference_max_iterations: int = 40
    stratified_seed_enabled: bool = False
    bounded_optimizer_enabled: bool = False
    wrist_risk_enabled: bool = False
    stratified_refresh_interval: int = 20
    rolling_early_stop_candidates: int = 2


class MuJoCoCandidateGenerator:
    """Callable adapter matching ``strict_bimanual_ik.CandidateGenerator``."""

    def __init__(self, model, data, contract, *, name_map, config=CandidateGeneratorConfig()):
        self.model, self.config = model, config
        self.scratch = mujoco.MjData(model)
        self.names = name_map
        self._rolling: dict[str, list[np.ndarray]] = {side: [] for side in name_map}
        self._seed_calls: dict[str, int] = {side: 0 for side in name_map}
        self._last_used_stratified: dict[str, bool] = {
            side: False for side in name_map}
        self._ids = {}
        for side, names in name_map.items():
            joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names["joints"]]
            if any(i < 0 for i in joints): raise ValueError(f"unknown {side} joint")
            qids = np.asarray([model.jnt_qposadr[i] for i in joints], dtype=int)
            dids = np.asarray([model.jnt_dofadr[i] for i in joints], dtype=int)
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, names.get("site", f"{side}_tcp"))
            self._ids[side] = (qids, dids, site)
        self._base_qpos = np.asarray(data.qpos).copy()

    def reset(self) -> None:
        """Forget path-dependent seeds before an independent planning run."""
        self._rolling = {side: [] for side in self.names}
        self._seed_calls = {side: 0 for side in self.names}
        self._last_used_stratified = {side: False for side in self.names}

    def _joint_limits(self, side):
        _, dids, _ = self._ids[side]
        joint_ids = self.model.dof_jntid[dids]
        ranges = np.asarray(self.model.jnt_range[joint_ids], dtype=float)
        limited = self.model.jnt_limited[joint_ids].astype(bool)
        lower = np.where(limited, ranges[:, 0], -np.inf)
        upper = np.where(limited, ranges[:, 1], np.inf)
        return joint_ids, ranges, limited, lower, upper

    def _seeds(self, side, *, force_stratified=False):
        qids, _, _ = self._ids[side]
        _, _, limited, lower, upper = self._joint_limits(side)
        lo = np.where(limited, lower, -np.pi)
        hi = np.where(limited, upper, np.pi)
        finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        mid = np.where(finite, (lo + hi) / 2, 0.)
        span = np.where(finite, hi - lo, 2 * np.pi)
        seeds = self._rolling[side] + [self._base_qpos[qids], mid]
        refresh_interval = int(self.config.stratified_refresh_interval)
        if refresh_interval < 1:
            raise ValueError("stratified_refresh_interval must be positive")
        use_stratified = (
            self.config.stratified_seed_enabled and
            (force_stratified or not self._rolling[side] or
             self._seed_calls[side] % refresh_interval == 0))
        self._seed_calls[side] += 1
        self._last_used_stratified[side] = use_stratified
        if use_stratified:
            seeds.extend(stratified_joint_seeds(lo, hi))
        count = int(self.config.global_seed_count)
        if count < 0:
            raise ValueError("global_seed_count may not be negative")
        # A fixed six-dimensional seed bank.  The former 3-bit sign pattern
        # repeated dimensions (J1/J4, J2/J5, J3/J6), leaving large IK basins
        # unreachable after a rolling seed was lost.
        side_offset = 0 if side == "left" else 104729
        rng = np.random.default_rng(20260811 + side_offset)
        for unit in rng.random((count, len(qids))):
            seeds.append(np.where(finite, lo + unit * span,
                                  (2.0 * unit - 1.0) * np.pi))
        unique = []
        for seed in seeds:
            value = np.asarray(seed, float).copy()
            if not any(np.allclose(value, old, rtol=0.0, atol=1e-12)
                       for old in unique):
                unique.append(value)
        return unique

    def _stratified_seeds(self, side):
        _, _, limited, lower, upper = self._joint_limits(side)
        lo = np.where(limited, lower, -np.pi)
        hi = np.where(limited, upper, np.pi)
        return stratified_joint_seeds(lo, hi)

    def _measure_candidate(self, side, target_p, target_q):
        qids, dids, site = self._ids[side]
        d = self.scratch
        actual_q = np.empty(4)
        mujoco.mju_mat2Quat(actual_q, d.site_xmat[site])
        rotation = _world_rotation_error(
            target_q, actual_q, d.site_xmat[site])
        pe = float(np.linalg.norm(target_p - d.site_xpos[site]))
        re = float(np.linalg.norm(rotation))
        q = d.qpos[qids].copy()
        tcp = np.r_[d.site_xpos[site].copy(), actual_q]
        _, ranges, limited, _, _ = self._joint_limits(side)
        margins = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
        limit_margin = (float(np.min(margins[limited]))
                        if np.any(limited) else float("inf"))
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jp, jr, site)
        jac = np.vstack((jp[:, dids], jr[:, dids]))
        singularity_margin = float(np.linalg.svd(
            jac, compute_uv=False)[-1])
        return q, tcp, pe, re, limit_margin, singularity_margin

    def _bounded_solve(self, side, seed, target_p, target_q):
        """Solve normalized 6D pose residuals inside hard joint bounds."""
        qids, dids, site = self._ids[side]
        _, _, _, lower, upper = self._joint_limits(side)
        initial = np.clip(np.asarray(seed, dtype=float), lower, upper)
        cache = {"q": None, "residual": None, "jacobian": None}

        def evaluate(q):
            q = np.asarray(q, dtype=float)
            if (cache["q"] is not None and
                    np.array_equal(q, cache["q"])):
                return cache["residual"], cache["jacobian"]
            d = self.scratch
            d.qpos[:] = self._base_qpos
            d.qpos[qids] = q
            mujoco.mj_forward(self.model, d)
            current_q = np.empty(4)
            mujoco.mju_mat2Quat(current_q, d.site_xmat[site])
            rotation = _world_rotation_error(
                target_q, current_q, d.site_xmat[site])
            residual = normalized_pose_residual(
                target_p - d.site_xpos[site],
                self.config.orientation_weight * rotation,
                position_tolerance_m=self.config.position_tolerance_m,
                orientation_tolerance_rad=self.config.orientation_tolerance_rad,
            )
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, d, jp, jr, site)
            jacobian = np.vstack((
                -jp[:, dids] / self.config.position_tolerance_m,
                (-self.config.orientation_weight * jr[:, dids] /
                 self.config.orientation_tolerance_rad),
            ))
            cache.update(q=q.copy(), residual=residual, jacobian=jacobian)
            return residual, jacobian

        def residual(q):
            return evaluate(q)[0]

        def jacobian(q):
            return evaluate(q)[1]

        result = least_squares(
            residual, initial, jac=jacobian,
            bounds=(lower, upper), method="trf",
            max_nfev=int(self.config.max_iterations),
            xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )
        self.scratch.qpos[:] = self._base_qpos
        self.scratch.qpos[qids] = result.x
        mujoco.mj_forward(self.model, self.scratch)
        measured = self._measure_candidate(side, target_p, target_q)
        if (measured[2] > self.config.position_tolerance_m or
                measured[3] > self.config.orientation_tolerance_rad):
            self._near_misses.append(measured)
            return None
        return measured

    def _solve(self, side, seed, target_p, target_q):
        qids, dids, site = self._ids[side]; d = self.scratch
        d.qpos[:] = self._base_qpos; d.qpos[qids] = seed; mujoco.mj_forward(self.model, d)
        jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
        for _ in range(self.config.max_iterations):
            current_q = np.empty(4); mujoco.mju_mat2Quat(current_q, d.site_xmat[site])
            rot = _world_rotation_error(target_q, current_q, d.site_xmat[site])
            err = np.r_[target_p - d.site_xpos[site],
                        self.config.orientation_weight * rot]
            if np.linalg.norm(err[:3]) <= self.config.position_tolerance_m and np.linalg.norm(err[3:]) <= self.config.orientation_tolerance_rad:
                break
            mujoco.mj_jacSite(self.model, d, jp, jr, site)
            jac = np.vstack((jp[:, dids],
                             self.config.orientation_weight * jr[:, dids]))
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + self.config.damping ** 2 * np.eye(6), err)
            dq = np.clip(dq, -.18, .18)
            d.qpos[qids] += self.config.step_scale * dq
            joint_ids = self.model.dof_jntid[dids]; ranges = self.model.jnt_range[joint_ids]
            limited = self.model.jnt_limited[joint_ids].astype(bool)
            d.qpos[qids[limited]] = np.clip(d.qpos[qids[limited]], ranges[limited, 0], ranges[limited, 1])
            mujoco.mj_forward(self.model, d)
        measured = self._measure_candidate(side, target_p, target_q)
        q, tcp, pe, re, limit_margin, singularity_margin = measured
        if pe > self.config.position_tolerance_m or re > self.config.orientation_tolerance_rad:
            self._near_misses.append(measured)
            return None
        return measured

    def _constrained_refine(self, side, seed, target_p, target_q):
        """Use declared tolerances as inequalities near a DLS boundary."""
        qids, dids, site = self._ids[side]
        joint_ids = self.model.dof_jntid[dids]
        ranges = self.model.jnt_range[joint_ids]
        limited = self.model.jnt_limited[joint_ids].astype(bool)
        bounds = [
            (float(lo), float(hi)) if is_limited else (None, None)
            for (lo, hi), is_limited in zip(ranges, limited)
        ]

        def pose_errors(q):
            d = self.scratch
            d.qpos[:] = self._base_qpos
            d.qpos[qids] = q
            mujoco.mj_forward(self.model, d)
            actual_q = np.empty(4)
            mujoco.mju_mat2Quat(actual_q, d.site_xmat[site])
            rotation = _world_rotation_error(
                target_q, actual_q, d.site_xmat[site])
            return (
                float(np.linalg.norm(target_p - d.site_xpos[site])),
                float(np.linalg.norm(rotation)),
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
                module=r"scipy\.optimize\._optimize",
            )
            result = minimize(
                lambda q: pose_errors(q)[0],
                np.asarray(seed, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": lambda q: (
                        self.config.orientation_tolerance_rad - pose_errors(q)[1]
                    ),
                },
                options={
                    "maxiter": int(
                        self.config.constrained_fallback_max_iterations),
                    "ftol": 1e-12,
                    "disp": False,
                },
            )
        q = np.asarray(result.x, dtype=float)
        pe, re = pose_errors(q)
        if (pe > self.config.position_tolerance_m or
                re > self.config.orientation_tolerance_rad):
            return None

        d = self.scratch
        actual_q = np.empty(4)
        mujoco.mju_mat2Quat(actual_q, d.site_xmat[site])
        tcp = np.r_[d.site_xpos[site].copy(), actual_q]
        margins = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
        limit_margin = (float(np.min(margins[limited]))
                        if np.any(limited) else float("inf"))
        jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jp, jr, site)
        jac = np.vstack((jp[:, dids], jr[:, dids]))
        singularity_margin = float(np.linalg.svd(jac, compute_uv=False)[-1])
        return q, tcp, pe, re, limit_margin, singularity_margin

    def generate_reference_candidate(
            self, side: str, target_position_m: np.ndarray,
            target_quaternion_wxyz: np.ndarray, *, reference_q: np.ndarray):
        """Find the least-motion joint state inside the strict pose box."""
        target_p = np.asarray(target_position_m, dtype=float)
        target_q = _normalize_target_quaternion(target_quaternion_wxyz)
        qids, dids, site = self._ids[side]
        reference = np.asarray(reference_q, dtype=float)
        if reference.shape != qids.shape or np.any(~np.isfinite(reference)):
            raise ValueError("reference_q must match the arm joint shape")
        joint_ids, ranges, limited, lower, upper = self._joint_limits(side)
        if np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("reference_q violates joint limits")
        periodic = ((self.model.jnt_type[joint_ids]
                     == mujoco.mjtJoint.mjJNT_HINGE) & ~limited)

        def delta(q):
            value = np.asarray(q, dtype=float) - reference
            value = value.copy()
            value[periodic] = (value[periodic] + np.pi) % (2*np.pi) - np.pi
            return value

        cache = {"q": None, "metrics": None}

        def pose_metrics(q):
            q = np.asarray(q, dtype=float)
            if cache["q"] is not None and np.array_equal(q, cache["q"]):
                return cache["metrics"]
            self.scratch.qpos[:] = self._base_qpos
            self.scratch.qpos[qids] = q
            mujoco.mj_forward(self.model, self.scratch)
            actual_q = np.empty(4)
            mujoco.mju_mat2Quat(actual_q, self.scratch.site_xmat[site])
            rotation = _world_rotation_error(
                target_q, actual_q, self.scratch.site_xmat[site])
            position_delta = target_p-self.scratch.site_xpos[site]
            pe = float(np.linalg.norm(position_delta))
            re = float(np.linalg.norm(rotation))
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.scratch, jp, jr, site)
            pe_jac = (-position_delta @ jp[:, dids] / pe
                      if pe > 1e-12 else np.zeros(len(dids)))
            re_jac = (-rotation @ jr[:, dids] / re
                      if re > 1e-12 else np.zeros(len(dids)))
            metrics = pe, re, pe_jac, re_jac
            cache.update(q=q.copy(), metrics=metrics)
            return metrics

        def pose_errors(q):
            return pose_metrics(q)[:2]

        bounds = [(float(lo), float(hi)) if is_limited else (None, None)
                  for lo, hi, is_limited in zip(lower, upper, limited)]
        contype = self.model.geom_contype.copy()
        conaffinity = self.model.geom_conaffinity.copy()
        try:
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0
            pe0, re0 = pose_errors(reference)
            if (pe0 <= self.config.position_tolerance_m and
                    re0 <= self.config.orientation_tolerance_rad):
                solution = reference.copy()
            else:
                result = minimize(
                    lambda q: float(delta(q) @ delta(q)), reference,
                    jac=lambda q: 2.0*delta(q),
                    method="SLSQP", bounds=bounds,
                    constraints=(
                        {"type": "ineq", "fun": lambda q:
                         self.config.position_tolerance_m-pose_metrics(q)[0],
                         "jac": lambda q: -pose_metrics(q)[2]},
                        {"type": "ineq", "fun": lambda q:
                         self.config.orientation_tolerance_rad-pose_metrics(q)[1],
                         "jac": lambda q: -pose_metrics(q)[3]},
                    ),
                    options={
                        "maxiter": int(
                            self.config.reference_max_iterations),
                        "ftol": 1e-12, "disp": False,
                    })
                solution = np.asarray(result.x, dtype=float)
            pose_errors(solution)
            measured = self._measure_candidate(side, target_p, target_q)
        finally:
            self.model.geom_contype[:] = contype
            self.model.geom_conaffinity[:] = conaffinity
        q, tcp, pe, re, margin, sigma = measured
        if (pe > self.config.position_tolerance_m or
                re > self.config.orientation_tolerance_rad):
            return None
        wrist = (piperx_wrist_risk(q)
                 if self.config.wrist_risk_enabled else 0.0)
        normalized = max(
            pe/self.config.position_tolerance_m,
            re/self.config.orientation_tolerance_rad)
        return IKCandidate(
            q=q, branch_index=-1, pose_cost=normalized+10.0*wrist,
            actual_tcp=tcp, position_error_m=pe,
            orientation_error_rad=re, joint_limit_margin_rad=margin,
            singularity_margin=sigma, wrist_risk=wrist)

    def generate_warm_start_candidate(
            self, side: str, target_position_m: np.ndarray,
            target_quaternion_wxyz: np.ndarray, *, reference_q: np.ndarray):
        """Run one DLS solve from the last commanded joint state only.

        Unlike ``generate_target``, this method never adds global or rolling
        seeds and never changes the rolling seed bank.  It is the online fast
        path required by the v3.1 protocol; multistart search remains reserved
        for frame-zero anchoring and explicit rescue scans.
        """
        if side not in self._ids:
            raise ValueError(f"unknown side: {side}")
        target_p = np.asarray(target_position_m, dtype=float)
        if target_p.shape != (3,) or np.any(~np.isfinite(target_p)):
            raise ValueError("target position must contain three finite values")
        target_q = _normalize_target_quaternion(target_quaternion_wxyz)
        qids, _, _ = self._ids[side]
        reference = np.asarray(reference_q, dtype=float)
        if reference.shape != qids.shape or np.any(~np.isfinite(reference)):
            raise ValueError("reference_q must match the arm joint shape")
        _, _, _, lower, upper = self._joint_limits(side)
        if np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("reference_q violates joint limits")
        self._near_misses = []
        contype = self.model.geom_contype.copy()
        conaffinity = self.model.geom_conaffinity.copy()
        try:
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0
            measured = self._solve(
                side, reference, target_p, target_q)
        finally:
            self.model.geom_contype[:] = contype
            self.model.geom_conaffinity[:] = conaffinity
        if measured is None:
            return None
        q, tcp, pe, re, margin, sigma = measured
        wrist = (piperx_wrist_risk(q)
                 if self.config.wrist_risk_enabled else 0.0)
        normalized = max(
            pe / self.config.position_tolerance_m,
            re / self.config.orientation_tolerance_rad,
        )
        return IKCandidate(
            q=q, branch_index=-1,
            pose_cost=normalized + 10.0 * wrist,
            actual_tcp=tcp, position_error_m=pe,
            orientation_error_rad=re,
            joint_limit_margin_rad=margin,
            singularity_margin=sigma, wrist_risk=wrist,
        )

    def __call__(self, model: Any, contract: Any, task: Any, row: int,
                 side: str):
        return self.generate_target(
            side,
            getattr(task, f"{side}_position_m")[row],
            getattr(task, f"{side}_quaternion_wxyz")[row],
        )

    def generate_target(self, side: str, target_position_m: np.ndarray,
                        target_quaternion_wxyz: np.ndarray, *,
                        force_stratified=False):
        """Generate candidates for one explicit target without copying a task."""
        if not 0.0 <= self.config.orientation_weight <= 1.0:
            raise ValueError("orientation_weight must be between zero and one")
        target_p = np.asarray(target_position_m, float)
        target_q = _normalize_target_quaternion(target_quaternion_wxyz)
        solved = []
        self._near_misses = []
        periodic = ((self.model.jnt_type[self.model.dof_jntid[self._ids[side][1]]]
                     == mujoco.mjtJoint.mjJNT_HINGE) &
                    ~self.model.jnt_limited[self.model.dof_jntid[self._ids[side][1]]].astype(bool))
        # Candidate generation is purely kinematic.  Contact generation during
        # every DLS iteration is expensive for mesh-heavy robot models and does
        # not change site kinematics.  The caller's collision adapter evaluates
        # the solved configurations and edges after these masks are restored.
        contype = self.model.geom_contype.copy()
        conaffinity = self.model.geom_conaffinity.copy()
        try:
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0
            primary_seeds = self._seeds(
                side, force_stratified=force_stratified)
            early_stop = int(self.config.rolling_early_stop_candidates)
            if early_stop < 1:
                raise ValueError(
                    "rolling_early_stop_candidates must be positive")
            for seed in primary_seeds:
                solve = (self._bounded_solve
                         if self.config.bounded_optimizer_enabled
                         else self._solve)
                item = solve(side, seed, target_p, target_q)
                if item is None:
                    continue
                equivalent = False
                for old in solved:
                    delta = item[0] - old[0]
                    delta[periodic] = (delta[periodic] + np.pi) % (2 * np.pi) - np.pi
                    if np.linalg.norm(delta) <= self.config.dedup_rad:
                        equivalent = True
                        break
                if equivalent:
                    continue
                solved.append(item)
                if (self.config.stratified_seed_enabled and
                        not self._last_used_stratified[side] and
                        len(solved) >= early_stop):
                    break
            # Rolling seeds are the fast path. If the target leaves all of
            # their basins, recover branch diversity immediately instead of
            # sacrificing one source frame until the next scheduled refresh.
            if (not solved and self.config.stratified_seed_enabled and
                    not self._last_used_stratified[side]):
                for seed in self._stratified_seeds(side):
                    item = solve(side, seed, target_p, target_q)
                    if item is None:
                        continue
                    if any(np.linalg.norm(item[0] - old[0]) <=
                           self.config.dedup_rad for old in solved):
                        continue
                    solved.append(item)
            if not solved and self.config.constrained_fallback_enabled:
                fallback_count = int(self.config.constrained_fallback_seed_count)
                fallback_iterations = int(
                    self.config.constrained_fallback_max_iterations)
                if fallback_count < 0:
                    raise ValueError(
                        "constrained_fallback_seed_count may not be negative")
                if fallback_iterations < 1:
                    raise ValueError(
                        "constrained_fallback_max_iterations must be positive")
                near_misses = sorted(
                    self._near_misses,
                    key=lambda item: (
                        max(item[2] / self.config.position_tolerance_m,
                            item[3] / self.config.orientation_tolerance_rad),
                        item[2] + item[3],
                        *item[0].tolist(),
                    ),
                )
                distinct_near_misses = []
                for near_miss in near_misses:
                    if all(np.linalg.norm(near_miss[0] - old[0]) >
                           self.config.dedup_rad
                           for old in distinct_near_misses):
                        distinct_near_misses.append(near_miss)
                for near_miss in distinct_near_misses[:fallback_count]:
                    item = self._constrained_refine(
                        side, near_miss[0], target_p, target_q)
                    if item is None:
                        continue
                    equivalent = False
                    for old in solved:
                        delta = item[0] - old[0]
                        delta[periodic] = (
                            (delta[periodic] + np.pi) % (2 * np.pi) - np.pi)
                        if np.linalg.norm(delta) <= self.config.dedup_rad:
                            equivalent = True
                            break
                    if not equivalent:
                        solved.append(item)
        finally:
            self.model.geom_contype[:] = contype
            self.model.geom_conaffinity[:] = conaffinity
        def order(item):
            q, _, pe, re, margin, sigma = item
            normalized = max(
                pe / self.config.position_tolerance_m,
                re / self.config.orientation_tolerance_rad,
            )
            wrist = (piperx_wrist_risk(q)
                     if self.config.wrist_risk_enabled else 0.0)
            return wrist, normalized, -sigma, -margin, *q.tolist()

        solved.sort(key=order)
        self._rolling[side] = [x[0].copy() for x in solved[:4]]
        maximum = int(self.config.maximum_candidates)
        if maximum < 1:
            raise ValueError("maximum_candidates must be positive")
        result = []
        for index, (q, tcp, pe, re, margin, sigma) in enumerate(
                solved[:maximum]):
            wrist = (piperx_wrist_risk(q)
                     if self.config.wrist_risk_enabled else 0.0)
            normalized = max(
                pe / self.config.position_tolerance_m,
                re / self.config.orientation_tolerance_rad,
            )
            result.append(IKCandidate(
                q=q, branch_index=index,
                pose_cost=normalized + 10.0 * wrist,
                actual_tcp=tcp, position_error_m=pe,
                orientation_error_rad=re,
                joint_limit_margin_rad=margin,
                singularity_margin=sigma,
                wrist_risk=wrist,
            ))
        return result
