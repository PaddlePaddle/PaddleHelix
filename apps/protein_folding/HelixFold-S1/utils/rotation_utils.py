import random
import numpy as np
import copy
from scipy.spatial.transform import Rotation

predefined_rotation_arrays = np.array([
    [ # identity
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],
    [ # counterclockwise 90
        [0,0,1],
        [0,1.,0],
        [-1,0,0]
    ],  
    [
        [0,0,-1],
        [0,1.,0],
        [1,0,0]
    ],
    [
        [0,0,1],
        [0,-1.,0],
        [1,0,0]
    ]
], dtype=np.float32)

def rotate_random_predefined(all_atom_positions):
    """ rotate atom_positions with randomly chosen pre-defined rot array"""
    num_array = len(predefined_rotation_arrays)
    rand_index = random.randint(0, num_array-1)
    rot_array = predefined_rotation_arrays[rand_index]
    return rotate_all_atom_positions(all_atom_positions, rot_array)

def rotate_all_atom_positions(all_atom_positions, rot_array):
    """ rotate atom_positions with rot_array"""
    # all_atom_positions: [num_res, 37, 3]
    # rot_array: [3, 3]
    all_atom_positions = all_atom_positions.reshape((-1,3))  # [num_res*37, 3]
    all_atom_positions = all_atom_positions.T  # [3, num_res*37]
    rotated_atom_pos = rot_array @ all_atom_positions  # [3, num_res*37]
    return rotated_atom_pos.T.reshape(-1, 37, 3).astype('float32')

def get_rotation_mat(diffuser=None):
    """ randomly sample rotation array from diffuser or predifined. """
    # randomly sample from predifined rot arry
    rot_t = copy.deepcopy(predefined_rotation_arrays)
    num_array = len(rot_t)
    if not diffuser is None:
        # diffuse predified rot array
        rot_t, rot_score = diffuser._so3_diffuser.forward_marginal(
            Rotation.from_matrix(rot_t).as_rotvec(), t=np.random.uniform(0.01, 1.0)
            )
        rot_t = Rotation.from_rotvec(rot_t).as_matrix()
    rand_index = random.randint(0, num_array - 1)
    rot_array = rot_t[rand_index]
    return rot_array.astype('float32')

if __name__ == '__main__':
    import paddle
    from omegaconf import OmegaConf
    from utils.frame_diff.modules import Diffuser
    import numpy as np
    from scipy.spatial.transform import Rotation
    from utils.frame_diff.data import diffuse_utils
    conf = OmegaConf.load("utils/frame_diff/config/multimer.yaml")
    diffuser = Diffuser(conf.diffuser)

    # pdb_file = "exp-7kp8_B_E.pdb"
    pdb_file = "exp-7kp8_A_B_C_E.pdb"
    prot_obj = diffuse_utils.read_pdb(pdb_file)

    rot_array = get_rotation_mat(diffuser)
    rot_pos = rotate_all_atom_positions(prot_obj.atom_positions, rot_array)

    rot_port = diffuse_utils.update_atom_pos(rot_pos, prot_obj, pdb_file=f"7kp8_A_B_C_E-rot.pdb")

