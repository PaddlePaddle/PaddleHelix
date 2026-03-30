import gzip 
import os 
from helixfold.data import mmcif_parsing
from helixfold.data import parsers
from helixfold.common.protein import PDB_CHAIN_IDS

def traverse_until_valid_res(info, res_index) -> int:
    # NOTE: id of `info` starts from 0, while res_index
    # starts from 1.
    while res_index - 1 < len(info) and info[res_index - 1].is_missing:
        res_index += 1

    return res_index


def parse_fasta_chains(seqs_names):
    chains = []
    for n in seqs_names:
        ch = n.split()[0].split('_')[-1]
        chains.append(ch)

    return chains

def convert_to_atom_line(chain_id, atom, atom_index, res_index) -> str:
    record_type = 'ATOM'
    name = atom.name
    alt_loc = atom.altloc
    res_name_3 = atom.get_parent().get_resname()
    insertion_code = ''

    pos0 = float(atom.coord[0])
    pos1 = float(atom.coord[1])
    pos2 = float(atom.coord[2])

    occupancy = atom.occupancy
    b_factor = atom.bfactor
    element = atom.element

    charge = atom.get_charge()
    charge = '' if charge is None else charge

    atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                            f'{res_name_3:>3} {chain_id:>1}'
                            f'{res_index:>4}{insertion_code:>1}   '
                            f'{pos0:>8.3f}{pos1:>8.3f}{pos2:>8.3f}'
                            f'{occupancy:>6.2f} {b_factor:>6.2f}          '
                            f'{element:>2}{charge:>2}')

    return atom_line

def get_pdb_from_mmcif(exp_mmcif_file, exp_fasta_file, exp_pdb_file_from_mmcif, chain_ids=None):
    """
    Generate pdb file from mmcif 
    """
    if not os.path.exists(exp_pdb_file_from_mmcif):
        # Convert mmcif to pdb
        if exp_mmcif_file.endswith('.cif.gz'):
            with gzip.open(exp_mmcif_file, 'r') as f:
                cif_string = f.read().decode('utf8')
        else:
            cif_string = "".join(open(exp_mmcif_file, 'r').readlines())
        parse_result = mmcif_parsing.parse(file_id=exp_mmcif_file, mmcif_string=cif_string)
        mmcif_obj = parse_result.mmcif_object   

        # Parse
        with open(exp_fasta_file, 'r') as f:
            seqs, seqs_names = parsers.parse_fasta(f.read())

        if not chain_ids is None: # only save chain in chain_ids if specified
            # skip unnecessary chains
            chains_names, seqs_ = [], []
            for i, c in enumerate(parse_fasta_chains(seqs_names)):
                if c in chain_ids:
                    chains_names.append(c)
                    seqs_.append(seqs[i])
            seqs = seqs_
        else:
            chains_names = parse_fasta_chains(seqs_names)

        # Check
        chains_structures = dict()
        for i, chain in enumerate(mmcif_obj.structure.get_chains()):
            if chain.get_id() not in chains_names:
                continue

            i = chains_names.index(chain.get_id())
            assert seqs[i] == mmcif_obj.chain_to_seqres[chain.get_id()]
            chains_structures[chain.get_id()] = chain

        # Convert
        atom_lines = []
        atom_index = 1
        for i, ch in enumerate(chains_names):
            ch_ = PDB_CHAIN_IDS[i]
            res_index = 1
            info = mmcif_obj.seqres_to_structure[ch]

            res_index = traverse_until_valid_res(info, res_index)
            for res in chains_structures[ch].get_residues():
                for atom in res.get_atoms():
                    line = convert_to_atom_line(ch_, atom, atom_index, res_index)
                    atom_lines.append(line)
                    atom_index += 1

                res_index += 1
                res_index = traverse_until_valid_res(info, res_index)

                if res_index - 1 >= len(info):
                    break

            if ch == chains_names[-1]:
                atom_lines.append('END')
            else:
                atom_lines.append('TER')

        # Write
        with open(exp_pdb_file_from_mmcif, 'w') as f:
            for line in atom_lines:
                f.write(f'{line}\n')