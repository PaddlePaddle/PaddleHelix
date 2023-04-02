#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""Transformations for 3D coordinates.

This Module contains objects for representing Vectors (Vecs), Rotation Matrices
(Rots) and proper Rigid transformation (Rigids). These are represented as
named tuples with arrays for each entry, for example a set of
[N, M] points would be represented as a Vecs object with arrays of shape [N, M]
for x, y and z.

This is being done to improve readability by making it very clear what objects
are geometric objects rather than relying on comments and array shapes.
Another reason for this is to avoid using matrix
multiplication primitives like matmul or einsum, on modern accelerator hardware
these can end up on specialized cores such as tensor cores on GPU or the MXU on
cloud TPUs, this often involves lower computational precision which can be
problematic for coordinate geometry. Also these cores are typically optimized
for larger matrices than 3 dimensional, this code is written to avoid any
unintended use of these cores on both GPUs and TPUs.
"""
import paddle
import numpy as np
import collections
from typing import List
from alphafold_paddle.model import quat_affine

# Array of rigid 3D transformations, stored as array of rotations and
# array of translations.
Rigids = collections.namedtuple('Rigids', ['rot', 'trans'])

class Vecs:
    def __init__(self, *args):
        
        if len(args) == 1:
            if type(args[0]) in [list, tuple] and len(args[0]) == 3:
                self.translation = paddle.stack(args[0], axis=-1)
            elif len(args[0]) == 1:
                self.translation = args[0]
            elif args[0].shape[-1]==3:
                self.translation = args[0]
            else:
                raise ValueError('Invalid number of inputs')
        elif len(args) == 3:
            self.translation = paddle.stack(args, axis=-1)
        else:
            raise ValueError('Invalid number of inputs')

    def map(self, map_fn, *args):
        result = []
        for i in range(3):
            r = map_fn(self.translation[..., i], *args)
            result.append(r)

        if result[0].shape[-1] == 1:
            return Vecs(paddle.concat(result, axis=-1))
        else:
            return Vecs(paddle.stack(result, axis=-1))

    @property
    def shape(self):
        return self.translation.shape

    @property
    def x(self):
        return self.translation[..., 0]

    @property
    def y(self):
        return self.translation[..., 1]

    @property
    def z(self):
        return self.translation[..., 2]

    def __getitem__(self,index):
        return Vecs(self.translation[index])
    def __str__(self):
        return str(self.translation.shape)
    def __repr__(self):
        return str(self.translation.shape)

    def reshape(self,*argv):
        return self.translation.reshape(*argv)


class Rots:
    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
            if len(args) == 9:
                rots = paddle.stack(args, axis=-1)
                self.rotation = rots.reshape(rots.shape[:-1] + [3, 3])
            else:
                if args.shape[-1] == 3 and args.shape[-2] == 3:
                    self.rotation = args
                elif args.shape[-1] == 9:
                    self.rotation = args.reshape(args.shape[:-1] + [3, 3])
                else:
                    raise ValueError('Invalid shape of input')
        elif len(args) == 9:
            rots = paddle.stack(args, axis=-1)
            self.rotation = rots.reshape(rots.shape[:-1] + [3, 3])
        else:
            raise ValueError('Invalid number of inputs')

    def map(self, map_fn, *args):
        result_i = []
        for i in range(3):
            result_j = []
            for j in range(3):
                r = map_fn(self.rotation[..., i, j], *args)
                result_j.append(r)

            if result_j[0].shape[-1] == 1:
                result_i.append(paddle.concat(result_j, axis=-1))
            else:
                result_i.append(paddle.stack(result_j, axis=-1))

        return Rots(paddle.stack(result_i, axis=-2))

    @property
    def shape(self):
        return self.rotation.shape

    @property
    def xx(self):
        return self.rotation[..., 0, 0]

    @property
    def xy(self):
        return self.rotation[..., 0, 1]

    @property
    def xz(self):
        return self.rotation[..., 0, 2]

    @property
    def yx(self):
        return self.rotation[..., 1, 0]

    @property
    def yy(self):
        return self.rotation[..., 1, 1]

    @property
    def yz(self):
        return self.rotation[..., 1, 2]

    @property
    def zx(self):
        return self.rotation[..., 2, 0]

    @property
    def zy(self):
        return self.rotation[..., 2, 1]

    @property
    def zz(self):
        return self.rotation[..., 2, 2]

    def __getitem__(self,index):
        return Rots(self.rotation[index])
    def __str__(self):
        return str(self.rotation.shape)
    def __repr__(self):
        return str(self.rotation.shape)
    def reshape(self,*argv):
        return self.rotation.reshape(*argv)


def squared_difference(x, y):
    return paddle.square(x - y)


def invert_rigids(r: Rigids) -> Rigids:
    """Computes group inverse of rigid transformations 'r'."""
    inv_rots = invert_rots(r.rot)
    t = rots_mul_vecs(inv_rots, r.trans)
    inv_trans = Vecs(-t.x, -t.y, -t.z)
    return Rigids(inv_rots, inv_trans)


def invert_rots(m: Rots) -> Rots:
    """Computes inverse of rotations 'm'."""
    return Rots(m.xx, m.yx, m.zx,
                m.xy, m.yy, m.zy,
                m.xz, m.yz, m.zz)


def rigids_from_3_points_vecs(
    point_on_neg_x_axis: Vecs,
    origin: Vecs,
    point_on_xy_plane: Vecs,
) -> Rigids:
  """Create Rigids from 3 points.

  Jumper et al. (2021) Suppl. Alg. 21 "rigidFrom3Points"
  This creates a set of rigid transformations from 3 points by Gram Schmidt
  orthogonalization.

  Args:
    point_on_neg_x_axis: Vecs corresponding to points on the negative x axis
    origin: Origin of resulting rigid transformations
    point_on_xy_plane: Vecs corresponding to points in the xy plane
  Returns:
    Rigid transformations from global frame to local frames derived from
    the input points.
  """
  m = rots_from_two_vecs(
      e0_unnormalized=vecs_sub(origin, point_on_neg_x_axis),
      e1_unnormalized=vecs_sub(point_on_xy_plane, origin))

  return Rigids(rot=m, trans=origin)


def rigids_from_3_points(
    point_on_neg_x_axis: paddle.Tensor,
    origin: paddle.Tensor,
    point_on_xy_plane: paddle.Tensor,
    eps: float = 1e-8) -> Rigids:
    """Create Rigids from 3 points.

    Jumper et al. (2021) Suppl. Alg. 21 "rigidFrom3Points"
    This creates a set of rigid transformations from 3 points by Gram Schmidt
    orthogonalization.

    Argss:
        point_on_neg_x_axis: [*, 3] coordinates
        origin: [*, 3] coordinates
        point_on_xy_plane: [*, 3] coordinates
        eps: small regularizer added to squared norm before taking square root.
    Returns:
        Rigids corresponding to transformations from global frame
        to local frames derived from the input points.
    """
    point_on_neg_x_axis = paddle.unbind(point_on_neg_x_axis, axis=-1)
    origin = paddle.unbind(origin, axis=-1)
    point_on_xy_plane = paddle.unbind(point_on_xy_plane, axis=-1)

    e0 = [c1 - c2 for c1, c2 in zip(origin, point_on_neg_x_axis)]
    e1 = [c1 - c2 for c1, c2 in zip(point_on_xy_plane, origin)]

    norms = paddle.sqrt(paddle.square(e0[0]) +
                        paddle.square(e0[1]) +
                        paddle.square(e0[2]) + eps)
    e0 = [c / norms for c in e0]
    dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
    e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
    norms = paddle.sqrt(paddle.square(e1[0]) +
                        paddle.square(e1[1]) +
                        paddle.square(e1[2]) + eps)
    e1 = [c / norms for c in e1]
    e2 = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ]

    rots = paddle.stack([c for tup in zip(e0, e1, e2) for c in tup], axis=-1)

    return Rigids(Rots(rots), Vecs(origin))


def rigids_from_list(l: List[paddle.Tensor]) -> Rigids:
    """Converts flat list of arrays to rigid transformations."""
    assert len(l) == 12
    return Rigids(Rots(*(l[:9])), Vecs(*(l[9:])))


def rigids_from_quataffine(a: quat_affine.QuatAffine) -> Rigids:
    """Converts QuatAffine object to the corresponding Rigids object."""
    return Rigids(Rots(a.rotation),
                    Vecs(a.translation))


def rigids_from_tensor4x4(m: paddle.Tensor) -> Rigids:
    """Construct Rigids from an 4x4 array.

    Here the 4x4 is representing the transformation in homogeneous coordinates.

    Argss:
        m: [*, 4, 4] homogenous transformation tensor
    Returns:
        Rigids corresponding to transformations m
    """
    assert m.shape[-1] == 4
    assert m.shape[-2] == 4
    return Rigids(
      Rots(m[..., 0, 0], m[..., 0, 1], m[..., 0, 2],
           m[..., 1, 0], m[..., 1, 1], m[..., 1, 2],
           m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]),
      Vecs(m[..., 0, 3], m[..., 1, 3], m[..., 2, 3]))


def rigids_from_tensor_flat9(m: paddle.Tensor) -> Rigids:
    """Flat9 encoding: first two columns of rotation matrix + translation."""
    assert m.shape[-1] == 9
    e0 = Vecs(m[..., 0], m[..., 1], m[..., 2])
    e1 = Vecs(m[..., 3], m[..., 4], m[..., 5])
    trans = Vecs(m[..., 6], m[..., 7], m[..., 8])
    return Rigids(rot=rots_from_two_vecs(e0, e1),
                    trans=trans)


def rigids_from_tensor_flat12(
    m: paddle.Tensor  # shape (..., 12)
    ) -> Rigids:  # shape (...)
    """Flat12 encoding: rotation matrix (9 floats) + translation (3 floats)."""
    assert m.shape[-1] == 12
    return Rigids(Rots(m[..., :9]), Vecs(m[..., 9:]))


def rigids_mul_rigids(a: Rigids, b: Rigids) -> Rigids:
    """Group composition of Rigids 'a' and 'b'."""
    return Rigids(
        rots_mul_rots(a.rot, b.rot),
        vecs_add(a.trans, rots_mul_vecs(a.rot, b.trans)))


def rigids_mul_rots(r: Rigids, m: Rots) -> Rigids:
    """Compose rigid transformations 'r' with rotations 'm'."""
    return Rigids(rots_mul_rots(r.rot, m), r.trans)


def rigids_mul_vecs(r: Rigids, v: Vecs) -> Vecs:
    """Apply rigid transforms 'r' to points 'v'."""
    return vecs_add(rots_mul_vecs(r.rot, v), r.trans)


def rigids_to_list(r: Rigids) -> List[paddle.Tensor]:
    """Turn Rigids into flat list, inverse of 'rigids_from_list'."""
    return list(r.rot) + list(r.trans)


def rigids_to_quataffine(r: Rigids) -> quat_affine.QuatAffine:
    """Convert Rigids r into QuatAffine, inverse of 'rigids_from_quataffine'."""
    return quat_affine.QuatAffine(
        quaternion=None,
        rotation=r.rot.rotation,
        translation=r.trans.translation)


def rigids_to_tensor_flat9(
    r: Rigids) -> paddle.Tensor:  # shape (..., 9)
    """Flat9 encoding: first two columns of rotation matrix + translation."""
    return paddle.stack(
        [r.rot.xx, r.rot.yx, r.rot.zx, r.rot.xy, r.rot.yy, r.rot.zy]
        + list(r.trans), axis=-1)


def rigids_to_tensor_flat12(
    r: Rigids  # shape (...)
    ) -> paddle.Tensor:  # shape (..., 12)
    """Flat12 encoding: rotation matrix (9 floats) + translation (3 floats)."""
    
    return paddle.stack([r.rot.xx, r.rot.yx, r.rot.zx, r.rot.xy, r.rot.yy, r.rot.zy, r.rot.xz, r.rot.yz, r.rot.zz]
                        + [r.trans.x, r.trans.y, r.trans.z], axis=-1)


def rots_from_tensor3x3(
    m: paddle.Tensor,  # shape (..., 3, 3)
    ) -> Rots:  # shape (...)
    """Convert rotations represented as (3, 3) array to Rots."""
    assert m.shape[-1] == 3
    assert m.shape[-2] == 3
    return Rots(m[..., 0, 0], m[..., 0, 1], m[..., 0, 2],
                m[..., 1, 0], m[..., 1, 1], m[..., 1, 2],
                m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])


def rots_from_two_vecs(e0_unnormalized: Vecs, e1_unnormalized: Vecs) -> Rots:
    """Create rotation matrices from unnormalized vectors for the x and y-axes.

    This creates a rotation matrix from two vectors using Gram-Schmidt
    orthogonalization.

    Args:
        e0_unnormalized: vectors lying along x-axis of resulting rotation
        e1_unnormalized: vectors lying in xy-plane of resulting rotation
    Returns:
        Rotations resulting from Gram-Schmidt procedure.
    """
    # Normalize the unit vector for the x-axis, e0.
    e0 = vecs_robust_normalize(e0_unnormalized)

    # make e1 perpendicular to e0.
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = Vecs(e1_unnormalized.x - c * e0.x,
              e1_unnormalized.y - c * e0.y,
              e1_unnormalized.z - c * e0.z)
    e1 = vecs_robust_normalize(e1)

    # Compute e2 as cross product of e0 and e1.
    e2 = vecs_cross_vecs(e0, e1)

    return Rots(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)


def rots_mul_rots(a: Rots, b: Rots) -> Rots:
    """Composition of rotations 'a' and 'b'."""
    c0 = rots_mul_vecs(a, Vecs(b.xx, b.yx, b.zx))
    c1 = rots_mul_vecs(a, Vecs(b.xy, b.yy, b.zy))
    c2 = rots_mul_vecs(a, Vecs(b.xz, b.yz, b.zz))
    return Rots(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)


def rots_mul_vecs(m: Rots, v: Vecs) -> Vecs:
    """Apply rotations 'm' to vectors 'v'."""
    return Vecs(m.xx * v.x + m.xy * v.y + m.xz * v.z,
                m.yx * v.x + m.yy * v.y + m.yz * v.z,
                m.zx * v.x + m.zy * v.y + m.zz * v.z)


def vecs_add(v1: Vecs, v2: Vecs) -> Vecs:
    """Add two vectors 'v1' and 'v2'."""
    return Vecs(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)


def vecs_dot_vecs(v1: Vecs, v2: Vecs) -> paddle.Tensor:
    """Dot product of vectors 'v1' and 'v2'."""
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def vecs_cross_vecs(v1: Vecs, v2: Vecs) -> Vecs:
    """Cross product of vectors 'v1' and 'v2'."""
    return Vecs(v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x)


def vecs_from_tensor(x: paddle.Tensor  # shape (..., 3)
                    ) -> Vecs:  # shape (...)
    """Converts from tensor of shape (3,) to Vecs."""
    num_components = x.shape[-1]
    assert num_components == 3
    return Vecs(x[..., 0], x[..., 1], x[..., 2])


def vecs_robust_normalize(v: Vecs, epsilon: float = 1e-8) -> Vecs:
    """Normalizes vectors 'v'.

    Argss:
        v: vectors to be normalized.
        epsilon: small regularizer added to squared norm before taking square root.
    Returns:
        normalized vectors
    """
    norms = vecs_robust_norm(v, epsilon)
    return Vecs(v.x / norms, v.y / norms, v.z / norms)


def vecs_robust_norm(v: Vecs, epsilon: float = 1e-8) -> paddle.Tensor:
    """Computes norm of vectors 'v'.

    Args:
        v: vectors to be normalized.
        epsilon: small regularizer added to squared norm before taking square root.
    Returns:
        norm of 'v'
    """
    return paddle.sqrt(paddle.square(v.x) +
                       paddle.square(v.y) +
                       paddle.square(v.z) + epsilon)


def vecs_sub(v1: Vecs, v2: Vecs) -> Vecs:
    """Computes v1 - v2."""
    return Vecs(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)


def vecs_squared_distance(v1: Vecs, v2: Vecs) -> paddle.Tensor:
    """Computes squared euclidean difference between 'v1' and 'v2'."""
    return (squared_difference(v1.x, v2.x) +
            squared_difference(v1.y, v2.y) +
            squared_difference(v1.z, v2.z))


def vecs_to_tensor(v: Vecs  # shape (...)
                  ) -> paddle.Tensor:  # shape(..., 3)
    """Converts 'v' to tensor with shape 3, inverse of 'vecs_from_tensor'."""
    return paddle.stack([v.x, v.y, v.z], axis=-1)
