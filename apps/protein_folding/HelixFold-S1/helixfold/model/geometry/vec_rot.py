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

"""Vec3Array and Rot3Array classes."""


import dataclasses
from dataclasses import dataclass, fields
import paddle
import numpy as np


@dataclass
class Vec3Array:
    """Vector in 3-dim space implemented as arrays"""
    x: paddle.Tensor
    y: paddle.Tensor
    z: paddle.Tensor

    def __post_init__(self):
        if hasattr(self.x, 'dtype'):
            assert self.x.dtype == self.y.dtype
            assert self.x.dtype == self.z.dtype
            assert all([x == y for x, y in zip(self.x.shape, self.y.shape)])
            assert all([x == z for x, z in zip(self.x.shape, self.z.shape)])

    def __add__(self, other):
        return Vec3Array(self.x + other.x,
                         self.y + other.y,
                         self.z + other.z)

    def __sub__(self, other):
        return Vec3Array(self.x - other.x,
                         self.y - other.y,
                         self.z - other.z)

    def __mul__(self, scale):
        return Vec3Array(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale):
        return self * scale

    def __truediv__(self, norm):
        return Vec3Array(self.x / norm, self.y / norm, self.z / norm)

    def __neg__(self):
        return Vec3Array(-self.x, -self.y, -self.z)

    def __pos__(self):
        return Vec3Array(self.x, self.y, self.z)

    def __getitem__(self, key):
        sliced = {}
        for field in fields(self):
            sliced[field.name] = getattr(self, field.name)[key]

        return dataclasses.replace(self, **sliced)

    def cross(self, other):
        """Compute cross product between 'self' and 'other'."""
        new_x = self.y * other.z - self.z * other.y
        new_y = self.z * other.x - self.x * other.z
        new_z = self.x * other.y - self.y * other.x
        return Vec3Array(new_x, new_y, new_z)

    def dot(self, other):
        """Compute dot product between 'self' and 'other'."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self, eps=1e-4):
        """Compute Norm of Vec3Array, clipped to epsilon."""
        norm2 = self.dot(self)
        if eps:
            norm2 = paddle.maximum(paddle.ones_like(norm2) * eps ** 2, norm2)

        return paddle.sqrt(norm2)

    def normalized(self, eps=1e-4):
        """Return unit vector with optional clipping."""
        norm2 = self.dot(self)
        if eps:
            norm2 = paddle.maximum(paddle.ones_like(norm2) * eps ** 2, norm2)

        return self * paddle.rsqrt(norm2)

    @classmethod
    def zeros(cls, shape, dtype='float32'):
        return cls(paddle.zeros(shape, dtype), paddle.zeros(shape, dtype),
                   paddle.zeros(shape, dtype))

    def to_array(self):
        return paddle.stack([self.x, self.y, self.z], axis=-1)

    @classmethod
    def from_array(cls, array):
        return cls(*paddle.unstack(array, axis=-1))


@dataclass
class Rot3Array:
    xx: paddle.Tensor
    xy: paddle.Tensor
    xz: paddle.Tensor

    yx: paddle.Tensor
    yy: paddle.Tensor
    yz: paddle.Tensor

    zx: paddle.Tensor
    zy: paddle.Tensor
    zz: paddle.Tensor

    def inverse(self):
        """Returns inverse of Rot3Array."""
        return Rot3Array(self.xx, self.yx, self.zx,
                         self.xy, self.yy, self.zy,
                         self.xz, self.yz, self.zz)

    def apply_to_point(self, point):
        """Applies Rot3Array to point."""
        return Vec3Array(
            self.xx * point.x + self.xy * point.y + self.xz * point.z,
            self.yx * point.x + self.yy * point.y + self.yz * point.z,
            self.zx * point.x + self.zy * point.y + self.zz * point.z)

    def apply_inverse_to_point(self, point):
        """Applies inverse Rot3Array to point."""
        return self.inverse().apply_to_point(point)

    def __matmul__(self, other):
        """Composes two Rot3Arrays."""
        c0 = self.apply_to_point(Vec3Array(other.xx, other.yx, other.zx))
        c1 = self.apply_to_point(Vec3Array(other.xy, other.yy, other.zy))
        c2 = self.apply_to_point(Vec3Array(other.xz, other.yz, other.zz))
        return Rot3Array(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)

    def __getitem__(self, key):
        sliced = {}
        for field in fields(self):
            sliced[field.name] = getattr(self, field.name)[key]

        return dataclasses.replace(self, **sliced)

    @classmethod
    def identity(cls, shape, dtype='float32'):
        """Returns identity of given shape."""
        ones = paddle.ones(shape, dtype=dtype)
        zeros = paddle.zeros(shape, dtype=dtype)
        return cls(ones, zeros, zeros,
                   zeros, ones, zeros,
                   zeros, zeros, ones)

    @classmethod
    def from_two_vectors(cls, e0, e1):
        """Construct Rot3Array from two Vec3Array."""
        # Normalize to unit vector
        e0 = e0.normalized()

        # Make e1 unit and perpendicular to e0
        c = e1.dot(e0)
        e1 = (e1 - e0 * c).normalized()

        # Get e2 perpendicular to both e0 and e1
        e2 = e0.cross(e1)
        return cls(e0.x, e1.x, e2.x,
                   e0.y, e1.y, e2.y,
                   e0.z, e1.z, e2.z)

    @classmethod
    def from_quaternion(cls, w, x, y, z, normalize=True, epsilon=1e-6):
        if normalize:
            inv_norm = paddle.rsqrt(
                paddle.maximum(paddle.ones_like(w) * epsilon,
                               w ** 2 + x ** 2 + y ** 2 + z ** 2))
            w *= inv_norm
            x *= inv_norm
            y *= inv_norm
            z *= inv_norm

        xx = 1 - 2 * (y ** 2 + z ** 2)
        xy = 2 * (x * y - w * z)
        xz = 2 * (x * z + w * y)

        yx = 2 * (x * y + w * z)
        yy = 1 - 2 * (x ** 2 + z ** 2)
        yz = 2 * (y * z - w * x)

        zx = 2 * (x * z - w * y)
        zy = 2 * (y * z + w * x)
        zz = 1 - 2 * (x ** 2 + y ** 2)

        return cls(xx, xy, xz, yx, yy, yz, zx, zy, zz)

    def to_array(self):
        """Convert Rot3Array to array of shape [..., 3, 3]."""
        return paddle.stack(
            [paddle.stack([self.xx, self.xy, self.xz], axis=-1),
             paddle.stack([self.yx, self.yy, self.yz], axis=-1),
             paddle.stack([self.zx, self.zy, self.zz], axis=-1)],
            axis=-2)

    @classmethod
    def from_array(cls, array):
        """Construct Rot3Array Matrix from array of shape. [..., 3, 3]."""
        x, y, z = paddle.unstack(array, axis=-2)
        xx, xy, xz = paddle.unstack(x, axis=-1)
        yx, yy, yz = paddle.unstack(y, axis=-1)
        zx, zy, zz = paddle.unstack(z, axis=-1)
        return cls(xx, xy, xz, yx, yy, yz, zx, zy, zz)

    def stop_gradient(self):
        detached = {}
        for field in fields(self):
            detached[field.name] = getattr(self, field.name).detach()

        return dataclasses.replace(self, **detached)


if __name__ == '__main__':
    xyz = []
    for _ in range(3):
        i = np.random.random([5])
        xyz.append(paddle.to_tensor(i))

    v1 = Vec3Array(*xyz)
    import ipdb; ipdb.set_trace()
    v2 = Vec3Array(*xyz)
