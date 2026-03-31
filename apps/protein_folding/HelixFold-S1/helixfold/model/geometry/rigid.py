#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rigid3Array Transformation represented by a Rot3Array and a Vec3Array."""

import paddle
import dataclasses
from dataclasses import dataclass, fields
from helixfold.model.geometry import vec_rot


@dataclass
class Rigid3Array:
    """Rigid transformation, i.e. element of special euclidean group."""

    rotation: vec_rot.Rot3Array
    translation: vec_rot.Vec3Array

    def __matmul__(self, other):
        new_rotation = self.rotation @ other.rotation
        new_translation = self.apply_to_point(other.translation)
        return Rigid3Array(new_rotation, new_translation)

    def __getitem__(self, key):
        sliced = {}
        for field in fields(self):
            sliced[field.name] = getattr(self, field.name)[key]

        return dataclasses.replace(self, **sliced)

    def inverse(self):
        """Return new Rigid3Array corresponding to inverse transform"""
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply_to_point(-self.translation)
        return Rigid3Array(inv_rotation, inv_translation)

    def apply_to_point(self, point):
        """Apply Rigid3Array transform to point."""
        return self.rotation.apply_to_point(point) + self.translation

    def apply_inverse_to_point(self, point):
        """Apply inverse Rigid3Array transform to point."""
        new_point = point - self.translation
        return self.rotation.apply_inverse_to_point(new_point)

    def scale_translation(self, factor):
        """Scale translation in Rigid3Array by given factor."""
        return Rigid3Array(self.rotation, self.translation * factor)

    @classmethod
    def identity(cls, shape, dtype='float32'):
        return cls(vec_rot.Rot3Array.identity(shape, dtype),
                   vec_rot.Vec3Array.zeros(shape, dtype))

    def to_array(self):
        rot_array = self.rotation.to_array()
        vec_array = self.translation.to_array()
        return paddle.concat([rot_array, vec_array[..., None]], axis=-1)

    @classmethod
    def from_array(cls, array):
        rot = vec_rot.Rot3Array.from_array(array[..., :3])
        vec = vec_rot.Vec3Array.from_array(array[..., -1])
        return cls(rot, vec)

    def stop_rot_gradient(self):
        return Rigid3Array(self.rotation.stop_gradient(), self.translation)
