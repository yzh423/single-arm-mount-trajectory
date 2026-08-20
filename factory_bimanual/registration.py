"""One rigid, shared left/right registration per factory task."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from .source_data import FactoryBimanualTask


@dataclass(frozen=True)
class RigidTaskRegistration:
    rotation_world_from_vr: np.ndarray
    translation_world_m: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_world_from_vr, dtype=float)
        translation = np.asarray(self.translation_world_m, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation and translation shapes must be (3,3) and (3,)")
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)):
            raise ValueError("rotation must be a proper orthonormal rotation")
        object.__setattr__(self, "rotation_world_from_vr", rotation.copy())
        object.__setattr__(self, "translation_world_m", translation.copy())

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_world_from_vr
        matrix[:3, 3] = self.translation_world_m
        return matrix

    def inverse(self) -> "RigidTaskRegistration":
        rotation = self.rotation_world_from_vr.T
        return RigidTaskRegistration(
            rotation, -rotation @ self.translation_world_m)


@dataclass(frozen=True)
class RegisteredBimanualTask(FactoryBimanualTask):
    registration: RigidTaskRegistration


def _transform_quaternion(quaternion_wxyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source_xyzw = np.asarray(quaternion_wxyz)[:, [1, 2, 3, 0]]
    source = Rotation.from_quat(source_xyzw).as_matrix()
    transformed = np.einsum("ij,njk->nik", rotation, source)
    output_xyzw = Rotation.from_matrix(transformed).as_quat()
    return output_xyzw[:, [3, 0, 1, 2]]


def register_task(
    task: FactoryBimanualTask, registration: RigidTaskRegistration
) -> RegisteredBimanualTask:
    rotation = registration.rotation_world_from_vr
    translation = registration.translation_world_m
    def position(values: np.ndarray) -> np.ndarray:
        return np.einsum("ij,nj->ni", rotation, values) + translation
    values = task.__dict__.copy()
    values.update(
        left_position_m=position(task.left_position_m),
        right_position_m=position(task.right_position_m),
        left_quaternion_wxyz=_transform_quaternion(task.left_quaternion_wxyz, rotation),
        right_quaternion_wxyz=_transform_quaternion(task.right_quaternion_wxyz, rotation),
        registration=registration,
    )
    return RegisteredBimanualTask(**values)
