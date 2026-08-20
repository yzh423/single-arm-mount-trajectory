"""Fair deterministic candidate generation for mount-orientation experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

from .mount_orientation import MOUNT_MODES, SUPPORTED_MOUNT_MODES
from .mount_constraints import MINIMUM_BIMANUAL_BASE_SEPARATION_M
from scripts.search_fold_box_piperx_paired_mount import deterministic_pair_mounts


MOUNT_FINGERPRINT_SCHEMA = "physical-installation-v3-pole-spacing-060"


@dataclass(frozen=True)
class OrientationSearchConfig:
    modes: tuple[str, ...] = MOUNT_MODES
    shared_z_values_m: tuple[float, ...] = (.81, .96, 1.11, 1.26, 1.41)
    maximum_candidates: int = 72

    def __post_init__(self):
        if self.maximum_candidates < len(self.shared_z_values_m):
            raise ValueError("maximum_candidates must cover every shared z value")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("mount modes must be unique")
        if any(mode not in SUPPORTED_MOUNT_MODES for mode in self.modes):
            raise ValueError("unsupported mount mode")


def generate_mounts(mode, task, config=OrientationSearchConfig()):
    if mode not in config.modes:
        raise ValueError(f"mode {mode!r} is outside this experiment")
    # Generate a broad feature-space pool first.  Filtering after an already
    # truncated shortlist can silently discard the safer, wider mounts.
    template_pool = deterministic_pair_mounts(
        task, maximum=max(
            4096 if mode == "horizontal_forward" else 512,
            config.maximum_candidates * 8),
        shared_base_z_m=.81)
    safe_templates = [mount for mount in template_pool
                      if math.dist(mount["xy"]["left"],
                                   mount["xy"]["right"])
                      >= MINIMUM_BIMANUAL_BASE_SEPARATION_M]
    if mode == "horizontal_forward":
        task_center_y = float(np.vstack((
            task.left_position_m, task.right_position_m))[:, 1].mean())
        safe_templates = [mount for mount in safe_templates
                          if np.mean((mount["xy"]["left"][1],
                                      mount["xy"]["right"][1]))
                          <= task_center_y - .15]
    xy_yaw_templates = evenly_spaced(safe_templates,
                                     config.maximum_candidates)
    if mode == "upright_table":
        installation_z = float(min(config.shared_z_values_m))
    elif mode == "inverted":
        installation_z = float(max(config.shared_z_values_m))
    else:
        all_z = [float(value) for side in ("left", "right")
                 for value in getattr(task, f"{side}_position_m")[:, 2]]
        installation_z = .5 * (min(all_z) + max(all_z))
    result = []
    for template in xy_yaw_templates:
        mount = json.loads(json.dumps(template))
        mount["shared_base_z_m"] = installation_z
        mount["mode"] = mode
        mount["base_z_m"] = {"left": installation_z,
                             "right": installation_z}
        result.append(mount)
    if len(result) != config.maximum_candidates:
        raise RuntimeError("candidate grid could not satisfy the common budget")
    return result


def mount_fingerprint(mount):
    payload = json.dumps(
        {"schema": MOUNT_FINGERPRINT_SCHEMA, "mount": mount},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evenly_spaced(values, maximum):
    values = list(values)
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[0]]
    indices = [round(i * (len(values) - 1) / (maximum - 1))
               for i in range(maximum)]
    return [values[index] for index in indices]
