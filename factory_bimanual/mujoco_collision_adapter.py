"""MuJoCo state and swept-pair collision callback for bimanual planning."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .bimanual_collision import CollisionClass, CollisionReport


@dataclass(frozen=True)
class ClearanceReport:
    minimum_m: float
    pair_distances_m: dict[str, float]
    margin_m: float
    valid: bool
    limiting_pair: str


class MuJoCoPairedCollisionChecker:
    def __init__(self, model, data, name_map, *, transition_steps=9,
                 clearance_margin_m=None, penetration_tolerance_m=1e-4):
        if transition_steps < 2: raise ValueError("transition_steps must be at least two")
        self.model, self.transition_steps = model, int(transition_steps)
        if (clearance_margin_m is not None and
                (not np.isfinite(clearance_margin_m) or
                 clearance_margin_m < 0.0)):
            raise ValueError(
                "clearance margin must be finite and nonnegative")
        self.clearance_margin_m = (None if clearance_margin_m is None
                                   else float(clearance_margin_m))
        if (not np.isfinite(penetration_tolerance_m) or
                penetration_tolerance_m < 0.0):
            raise ValueError(
                "penetration tolerance must be finite and nonnegative")
        self.penetration_tolerance_m = float(penetration_tolerance_m)
        self.data = mujoco.MjData(model)
        self.base_qpos = np.asarray(data.qpos).copy()
        self.qids = {}
        for side in ("left", "right"):
            jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in name_map[side]["joints"]]
            self.qids[side] = np.asarray([model.jnt_qposadr[j] for j in jids], int)
        # Geometry membership is structural and never changes with qpos.
        # Resolving every name on every pair/edge check dominated Mount search.
        self._clearance_family_cache = None

    def _matching_geoms(self, *needles):
        result = []
        for geom in range(self.model.ngeom):
            name = (mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").lower()
            if "collision" in name and any(needle in name for needle in needles):
                result.append(geom)
        return result

    def _clearance_families(self):
        if self._clearance_family_cache is not None:
            return self._clearance_family_cache
        side = lambda arm, *parts: self._matching_geoms(
            *(f"{arm}_{part}" for part in parts))
        self._clearance_family_cache = {
            "left_link4__right_gripper_base": ((
                side("left", "link4"), side("right", "gripper_base")),),
            "right_link4__left_gripper_base": ((
                side("right", "link4"), side("left", "gripper_base")),),
            "gripper_base__gripper_base": ((
                side("left", "gripper_base"),
                side("right", "gripper_base")),),
            "fingers__opposite_wrist": (
                (side("left", "gripper_link1", "gripper_link2"),
                 side("right", "link5", "link6")),
                (side("right", "gripper_link1", "gripper_link2"),
                 side("left", "link5", "link6"))),
            "elbow__opposite_base": (
                (side("left", "link3"),
                 side("right", "base_link", "mount_adapter")),
                (side("right", "link3"),
                 side("left", "base_link", "mount_adapter"))),
        }
        return self._clearance_family_cache

    def _set_state(self, left_q, right_q):
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.qids["left"]] = left_q
        self.data.qpos[self.qids["right"]] = right_q
        mujoco.mj_forward(self.model, self.data)

    def _minimum_geom_distance(self, first, second):
        minimum = np.inf
        segment = np.empty(6, dtype=float)
        for geom1 in first:
            for geom2 in second:
                if geom1 == geom2:
                    continue
                distance = mujoco.mj_geomDistance(
                    self.model, self.data, int(geom1), int(geom2),
                    10.0, segment)
                minimum = min(minimum, float(distance))
        return minimum

    def _clearance_at_current_state(self, margin_m):
        margin = float(margin_m)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("clearance margin must be finite and nonnegative")
        distances = {
            name: min(self._minimum_geom_distance(first, second)
                      for first, second in directional_pairs)
            for name, directional_pairs in self._clearance_families().items()
        }
        limiting = min(distances, key=distances.get)
        minimum = float(distances[limiting])
        return ClearanceReport(
            minimum, distances, margin, minimum >= margin, limiting)

    def clearance(self, left_q, right_q, *, margin_m=.015):
        """Measure named PiperX cross-arm gaps at one paired state."""
        self._set_state(left_q, right_q)
        return self._clearance_at_current_state(margin_m)

    def transition_clearance(self, previous, current, *, margin_m=.015):
        """Return the minimum named clearance along one paired swept edge."""
        reports = [self.clearance(
            (1.0 - alpha) * np.asarray(previous[0]) +
            alpha * np.asarray(current[0]),
            (1.0 - alpha) * np.asarray(previous[1]) +
            alpha * np.asarray(current[1]),
            margin_m=margin_m,
        ) for alpha in np.linspace(0.0, 1.0, self.transition_steps)]
        names = tuple(reports[0].pair_distances_m)
        distances = {name: min(report.pair_distances_m[name]
                               for report in reports)
                     for name in names}
        limiting = min(distances, key=distances.get)
        minimum = float(distances[limiting])
        margin = float(margin_m)
        return ClearanceReport(
            minimum, distances, margin, minimum >= margin, limiting)

    def _geom_side(self, geom):
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ""
        body = int(self.model.geom_bodyid[geom])
        while body:
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            name += " " + bname; body = int(self.model.body_parentid[body])
        if "left_" in name or "left_base" in name: return "left"
        if "right_" in name or "right_base" in name: return "right"
        return None

    def _geom_link_index(self, geom):
        """Return the prefixed robot link index owning ``geom``, if any."""
        body = int(self.model.geom_bodyid[int(geom)])
        while body:
            name = (mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body) or "")
            for side in ("left", "right"):
                prefix = f"{side}_link"
                if name.startswith(prefix):
                    suffix = name[len(prefix):].split("_")[0]
                    if suffix.isdigit():
                        return side, int(suffix)
            body = int(self.model.body_parentid[body])
        return None

    @staticmethod
    def _is_robot_base_geom_name(name):
        lower = str(name).lower()
        return any(
            lower.startswith(f"{side}_base")
            or lower.startswith(f"{side}_mount_adapter")
            for side in ("left", "right")
        )

    def _is_mount_attachment_contact(self, geom1, geom2):
        """Ignore only designed base/pedestal contacts for the same arm.

        Imported vendor collision meshes overlap their own pedestal around the
        bolted base.  Those contacts are permanent attachment geometry, not a
        motion-planning collision.  The I2RT asset likewise has a designed
        base/link1 overlap.  Contacts with later links or the opposite arm
        remain reportable.
        """
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ""
            for geom in (geom1, geom2)
        ]
        for side in ("left", "right"):
            finger_pair = {
                1 if name.startswith(f"{side}_gripper_link1") else
                2 if name.startswith(f"{side}_gripper_link2") else 0
                for name in names
            }
            if finger_pair == {1, 2}:
                return True
        for mount_index, other_index in ((0, 1), (1, 0)):
            name = names[mount_index]
            for side in ("left", "right"):
                if name == f"{side}_mount_adapter":
                    link = self._geom_link_index((geom1, geom2)[other_index])
                    return link is not None and link[0] == side and link[1] <= 2
                if name.startswith(f"{side}_base"):
                    link = self._geom_link_index((geom1, geom2)[other_index])
                    return link is not None and link[0] == side and link[1] == 1
        return False

    def _report(self):
        classes = set()
        for contact in self.data.contact:
            if float(contact.dist) >= -self.penetration_tolerance_m:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self._is_mount_attachment_contact(g1, g2):
                continue
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            s1, s2 = self._geom_side(g1), self._geom_side(g2)
            if "workbench" in (n1, n2): classes.add(CollisionClass.TABLE)
            elif s1 and s2 and s1 != s2: classes.add(CollisionClass.CROSS_ARM)
            elif s1 and s1 == s2: classes.add(CollisionClass.SELF)
            if (self._is_robot_base_geom_name(n1)
                    or self._is_robot_base_geom_name(n2)):
                classes.add(CollisionClass.BASE)
        order = list(CollisionClass)
        return CollisionReport(tuple(c for c in order if c in classes))

    def state(self, left_q, right_q):
        self._set_state(left_q, right_q)
        report = self._report()
        if self.clearance_margin_m is None:
            return report
        clearance = self._clearance_at_current_state(
            self.clearance_margin_m)
        if clearance.valid:
            return report
        classes = set(report.classes)
        classes.add(CollisionClass.CLEARANCE)
        return CollisionReport(tuple(
            item for item in CollisionClass if item in classes))

    def side_state(self, side, q):
        """Check collisions attributable to one arm without pairing branches."""
        if side not in self.qids:
            raise ValueError(f"unknown side: {side}")
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.qids[side]] = q
        mujoco.mj_forward(self.model, self.data)
        classes = set()
        for contact in self.data.contact:
            if float(contact.dist) >= -self.penetration_tolerance_m:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self._is_mount_attachment_contact(g1, g2):
                continue
            s1, s2 = self._geom_side(g1), self._geom_side(g2)
            if side not in (s1, s2):
                continue
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            if "workbench" in (n1, n2): classes.add(CollisionClass.TABLE)
            elif s1 == s2 == side: classes.add(CollisionClass.SELF)
            elif s1 and s2 and s1 != s2: classes.add(CollisionClass.CROSS_ARM)
            if (self._is_robot_base_geom_name(n1)
                    or self._is_robot_base_geom_name(n2)):
                classes.add(CollisionClass.BASE)
        return CollisionReport(tuple(c for c in CollisionClass if c in classes))

    def side_transition(self, side, previous, current):
        found = set()
        for alpha in np.linspace(0., 1., self.transition_steps):
            report = self.side_state(
                side, (1-alpha)*np.asarray(previous)+alpha*np.asarray(current))
            found.update(report.classes)
        return CollisionReport(tuple(c for c in CollisionClass if c in found))

    def transition(self, previous, current):
        found = set()
        for alpha in np.linspace(0., 1., self.transition_steps):
            report = self.state((1-alpha)*np.asarray(previous[0])+alpha*np.asarray(current[0]),
                                (1-alpha)*np.asarray(previous[1])+alpha*np.asarray(current[1]))
            found.update(report.classes)
        return CollisionReport(tuple(c for c in CollisionClass if c in found))
