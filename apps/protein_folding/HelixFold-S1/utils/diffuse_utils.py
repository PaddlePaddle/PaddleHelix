""" Diffusion utils. """
import paddle
from helixfold.model import all_atom
from helixfold.model import r3
from helixfold.data.data_utils import generate_backbone_affine, generate_torsion_angles
from helixfold.model import quat_affine, all_atom
from helixfold.relax.amber_minimize import make_atom14_positions
import gzip
from helixfold.common import protein as protein_utils
import numpy as np

def read_pdb(protein_struct_file):
    """ load prot_obj from pdb files. """
    if str(protein_struct_file).endswith(".gz"):
        with gzip.open(protein_struct_file, 'r') as f:
            pdb_file = f.read().decode('utf8')
    else:
        with open(protein_struct_file, 'r') as f:
            pdb_file = f.read()
    chain_id = None
    # {'aatype_index', 'all_atom_positions', 'all_atom_mask', 'resolution'}
    prot_obj = protein_utils.from_pdb_string(pdb_file, chain_id)
    return prot_obj


def update_atom_pos(atom_pos, prot_obj, pdb_file=None, seq_mask=None):
    """ Update atom_pos in prot_obj. Save new pdb if pdb_file is provided. """
    if seq_mask is None: 
        seq_mask = np.ones(len(atom_pos), dtype='bool')

    new_prot_obj = protein_utils.Protein(
      atom_positions=np.array(atom_pos)[seq_mask],
      atom_mask=np.array(prot_obj.atom_mask)[seq_mask],
      aatype=np.array(prot_obj.aatype)[seq_mask],
      residue_index=np.array(prot_obj.residue_index)[seq_mask],
      chain_index=prot_obj.chain_index[seq_mask],
      b_factors=np.array(prot_obj.b_factors)[seq_mask])
    if pdb_file is not None:
        pdb_str = protein_utils.to_pdb(new_prot_obj)
        open(pdb_file, 'w').write(pdb_str)
    return new_prot_obj


def trans_rots_angles_to_pos(trans, rots, angles, aatype,
            residx_atom37_to_atom14, atom37_atom_exists):
    """
    Assemble trans, rots, and angles to atom_pos. 
    refer to helixfold.model.folding MultiRigidSidechain
    trans: numpy array [num_res, 3]
    rots: numpy array  [num_res, 3, 3]
    angles: numpy array [num_res, 7, 2]
    aatype: numpy array [num_res]
    atom37_atom_exists: numpy array [num_res, 14]
    residx_atom37_to_atom14: numpy array [num_res, 37]

    returns:
        pos: (num_res, 37, 3)
    """
    
    rots = paddle.to_tensor(rots.astype(np.float32))[None, ...]  # [1, num_res, 3, 3]
    trans = paddle.to_tensor(trans.astype(np.float32))[None, ...]
    angles = paddle.to_tensor(angles.astype(np.float32))[None, ...]
    aatype = paddle.to_tensor(aatype.astype(np.int32))[None, ...]
    residx_atom37_to_atom14 = paddle.to_tensor(residx_atom37_to_atom14.astype(np.int32))[None, ...]
    atom37_atom_exists = paddle.to_tensor(atom37_atom_exists[None, ...].astype(np.float32))
    # merge rots and trans
    backbone_to_global = r3.Rigids(r3.Rots(rots), r3.Vecs(trans)) 
    # intergrate angles
    all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype, backbone_to_global, angles)
    pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
                aatype, all_frames_to_global)
    atom14_pred_positions = pred_positions.translation
    atom37_pred_positions = all_atom.atom14_to_atom37(
        atom14_pred_positions, {
            'residx_atom37_to_atom14': residx_atom37_to_atom14,
            'atom37_atom_exists': atom37_atom_exists
        })
    return atom37_pred_positions[0]


def pos_to_trans_rots_angles(pos, mask, aatype):
    """
    Disassemble atom_pos to trans, rots, and angles. 
    refer to helixfold.data.data_utils 
    pos: numpy array [num_res, 37, 3] res["structure_module"]["all_atom_positions"].numpy()
    mask: numpy array [num_res, 37] res["structure_module"]["all_atom_masks"].numpy() 
    aatype: numpy array [num_res] batch["aatype"][:,0,...].numpy() 

    returns:
        trans: (num_res, 3) paddle.tensor
        rots: (num_res, 3, 3) paddle.tensor
        affine_masks: (num_res,) paddle.tensor
        angles: (num_res, 7, 2) paddle.tensor
        angle_masks: (num_res, 7) paddle.tensor
    """
    # affine
    prot = {"aatype": aatype, "aatype_index": aatype,
            'all_atom_positions': pos, 'all_atom_mask': mask}
    prot = generate_backbone_affine(prot)
    prot = make_atom14_positions(prot)
    trans, rots, affine_masks = prot['backbone_affine_tensor_trans'], \
        prot['backbone_affine_tensor_rot'], prot['backbone_affine_mask']
    # prot = generate_torsion_angles(prot) returns (num_res, 4, 2) angles
    cpu = paddle.CPUPlace()
    torsion_angles_dict = all_atom.atom37_to_torsion_angles(
        aatype=paddle.to_tensor(prot['aatype_index'][None, None], place=cpu, stop_gradient=True),
        all_atom_pos=paddle.to_tensor(prot['all_atom_positions'][None, None], 'float32', place=cpu, stop_gradient=True),
        all_atom_mask=paddle.to_tensor(prot['all_atom_mask'][None, None], 'float32', place=cpu, stop_gradient=True),
        placeholder_for_undefined=True)
    torsion_angles_dict = {k: v.squeeze([0, 1]).numpy() for k, v in torsion_angles_dict.items()}
    angles, angle_masks = torsion_angles_dict['torsion_angles_sin_cos'], \
        torsion_angles_dict['torsion_angles_mask']
    return {"trans": trans, "rots": rots, "affine_masks": affine_masks, 
            "angles": angles, "angle_masks": angle_masks,
            "residx_atom37_to_atom14": prot["residx_atom37_to_atom14"], 
            "atom37_atom_exists": prot["atom37_atom_exists"]}


if __name__ == '__main__':
    from helixfold.data import mmcif_parsing, pipeline
    from helixfold.data import parsers
    from helixfold.data.data_utils import a3m_to_features, load_chain
    import gzip
    import pickle
    from utils.frame_diff.data import diffuse_utils

    # fasta_file = f"data_assembly/fasta/{protein}.fasta"
    # protein_struct_file = f"data_assembly/mmcif/{protein}.cif.gz"

    # seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
    # chain_ids = [i.split()[0].split('_')[1] for i in descs]

    # with gzip.open(protein_struct_file, 'r') as f: 
    #     mmcif_object = mmcif_parsing.parse(file_id=protein_struct_file,
    #                                         mmcif_string=f.read().decode('utf8')).mmcif_object

    # mmcif_chain_keys = mmcif_object.chain_to_seqres.keys()
    # valid_seq_chains = [(seq, cid) for seq, cid in zip(seqs, chain_ids) if cid in mmcif_chain_keys]
    # chain_ids = [cid for _, cid in valid_seq_chains]
    # seqs = [seq for seq, _ in valid_seq_chains]

    # protein_chain_d = load_chain(mmcif_object, chain_ids[0])
    # features_pkl = f"data_assembly/single_chain/{protein}_{chain_ids[0]}/features.pkl.gz"
    # with gzip.open(features_pkl, 'rb') as pkl:
    #     chain_feature = pickle.load(pkl)

    # mask = protein_chain_d["all_atom_mask"]
    # pos = protein_chain_d["all_atom_positions"]
    # aatype = chain_feature['aatype'].argmax(-1)

    pdb_file = "exp-7kp8_B_E.pdb"
    prot_obj = diffuse_utils.read_pdb(pdb_file)

    aatype = prot_obj.aatype
    pos = prot_obj.atom_positions
    mask = prot_obj.atom_mask

    trans, rots, affine_masks, angles, angle_masks, residx_atom37_to_atom14, atom37_atom_exists = \
        diffuse_utils.pos_to_trans_rots_angles(pos, mask, aatype)
    atom_pos = diffuse_utils.trans_rots_angles_to_pos(trans, rots, angles, aatype,
                                                      residx_atom37_to_atom14, atom37_atom_exists)

    print("cif pos", pos.shape, pos[0, :5, :2])
    print("output pos", atom_pos.shape, atom_pos[0, : 5, : 2])

    diffuse_utils.update_atom_pos(atom_pos, prot_obj, pdb_file="exp-7kp8_B_E-reverse.pdb")