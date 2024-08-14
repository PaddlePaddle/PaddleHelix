#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses the mmCIF file format."""

import collections
import dataclasses
import functools
import io
from typing import Any, Mapping, Optional, Sequence, Tuple, List
from Bio import PDB
import numpy as np
import time
import logging
from helixfold.common.residue_constants import crystallization_aids, ligand_exclusion_list

# Type aliases:
AtomName = str
ChainId = str
TypeChainID = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclasses.dataclass(frozen=True)
class Monomer:
  ''' tbd. '''
  id: str
  num: int


@dataclasses.dataclass(frozen=True)
class CovalentBond:
  ''' tbd. '''
  ptnr1_label_asym_id : str
  ptnr1_label_comp_id : str
  ptnr1_label_seq_id : str  ## may be non-integer, such . ?
  ptnr1_label_atom_id : str

  ptnr2_label_asym_id : str
  ptnr2_label_comp_id : str
  ptnr2_label_seq_id : str
  ptnr2_label_atom_id : str

  pdbx_dist_value : float
  pdbx_leaving_flag : str

  ptnr1_auth_asym_id : str
  ptnr1_auth_comp_id : str     
  ptnr1_auth_seq_id : str    ## use in branch-list, asym_id, seq_id 
  ptnr2_auth_asym_id : str  
  ptnr2_auth_comp_id : str   
  ptnr2_auth_seq_id : str ## use in branch-list, asym_id, seq_id 

  bond_type: str


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=False)
class AtomSite:
  ''' tbd. '''
  residue_name: str  # label comp_id
  author_chain_id: str 
  mmcif_chain_id: str  # label asym_id
  author_seq_num: str
  mmcif_seq_num: int  # label seq_id
  insertion_code: str
  hetatm_atom: str
  model_num: int

  entity_id: str
  pos_x: float
  pos_y: float
  pos_z: float
  occupancy: float
  label_alt_id: str
  label_atom_id: str # such as: CA, CB, OG
  label_type_symbol: str # such as H, N, c

  def __post_init__(self):
    ''' tbd. '''
    object.__setattr__(self, 'pos_x', float(self.pos_x))
    object.__setattr__(self, 'pos_y', float(self.pos_y))
    object.__setattr__(self, 'pos_z', float(self.pos_z))
    object.__setattr__(self, 'occupancy', float(self.occupancy))


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=False)
class ResidueAtPosition:
  ''' tbd. '''
  positions: List[List[float]]
  atom_ids: List[str]

  residue_name: str
  mmcif_chain_id: str
  mmcif_seq_num: str
  author_chain_id: str
  author_seq_num: str

  hetflag: str
  is_missing: bool

  def __post_init__(self):
    ''' tbd. '''
    assert len(self.positions) == len(self.atom_ids), (
      "ResidueAtPosition positons length must equal to atom_ids length.")


@dataclasses.dataclass(frozen=True)
class MmcifObject:
  """Representation of a parsed mmCIF file.

  Contains:
    file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
      files being processed.
    header: Biopython header.
    structure: Biopython structure.
    TypeChainID should be construct by following rules; concated by _ :   
    '''
        E.g. 'protein_A_BB'
        dtype: protein, dna, rna 
        mmcif_chain_id: mmcif chain_id
        author_chain_id: author chain_id
    '''
    chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
      {'A': 'ABCDEFG'}
    seqres_to_structure: Dict; for each chain_id contains a mapping between
      SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                        1: ResidueAtPosition,
                                                        ...}}
    raw_string: The raw string used to construct the MmcifObject.
  """
  file_id: str
  header: PdbHeader
  structure: PdbStructure
  mmcif_chain_to_typechain: Mapping[ChainId, TypeChainID]
  chain_to_seqres: Mapping[ChainId, SeqRes]
  seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
  covalent_bonds: Optional[List[CovalentBond]]
  raw_string: Any

  def __post_init__(self):
    ''' tbd. '''
    assert len(self.chain_to_seqres) == len(self.seqres_to_structure), (
        'mmCIF error: chain_to_seqres and seqres_to_structure have different '
        'lengths')


@dataclasses.dataclass(frozen=True)
class ParsingResult:
  """Returned by the parse function.

  Contains:
    mmcif_object: A MmcifObject, may be None if no chain could be successfully
      parsed.
    errors: A dict mapping (file_id, chain_id) to any exception generated.
  """
  mmcif_object: Optional[MmcifObject]
  errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
  """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(prefix: str,
                       parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
  """Extracts loop associated with a prefix from mmCIF data as a list.

  Reference for loop_ in mmCIF:
    http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

  Args:
    prefix: Prefix shared by each of the data items in the loop.
      e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
      _entity_poly_seq.mon_id. Should include the trailing period.
    parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
      parser.

  Returns:
    Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
  """
  cols = []
  data = []
  for key, value in parsed_info.items():
    if key.startswith(prefix):
      cols.append(key)
      data.append(value)

  assert all([len(xs) == len(data[0]) for xs in data]), (
      'mmCIF error: Not all loops are the same length: %s' % cols)

  return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(prefix: str,
                       index: str,
                       parsed_info: MmCIFDict,
                       ) -> Mapping[str, Mapping[str, str]]:
  """Extracts loop associated with a prefix from mmCIF data as a dictionary.

  Args:
    prefix: Prefix shared by each of the data items in the loop.
      e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
      _entity_poly_seq.mon_id. Should include the trailing period.
    index: Which item of loop data should serve as the key.
    parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
      parser.

  Returns:
    Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
    indexed by the index column.
  """
  entries = mmcif_loop_to_list(prefix, parsed_info)
  return {entry[index]: entry for entry in entries}


def _get_first_model(structure: PdbStructure) -> PdbStructure:
  """Returns the first model in a Biopython structure."""
  return next(structure.get_models())



def get_release_date(parsed_info: MmCIFDict) -> str:
  """Returns the oldest revision date."""
  revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
  return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
  """Returns a basic header containing method, release date and resolution."""
  header = {}

  experiments = mmcif_loop_to_list('_exptl.', parsed_info)
  header['structure_method'] = ','.join([
      experiment['_exptl.method'].lower() for experiment in experiments])

  # Note: The release_date here corresponds to the oldest revision. We prefer to
  # use this for dataset filtering over the deposition_date.
  if '_pdbx_audit_revision_history.revision_date' in parsed_info:
    header['release_date'] = get_release_date(parsed_info)
  else:
    # logging.warning('Could not determine release_date: %s',
    #                 parsed_info['_entry.id'])
    pass

  header['resolution'] = 0.00
  for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution',
                  '_reflns.d_resolution_high'):
    if res_key in parsed_info:
      try:
        raw_resolution = parsed_info[res_key][0]
        header['resolution'] = float(raw_resolution)
      except ValueError:
        logging.debug('Invalid resolution format: %s', parsed_info[res_key])

  return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
  """Returns list of atom sites; contains data not present in the structure."""
  return [AtomSite(*site) for site in zip(  # pylint:disable=g-complex-comprehension
      parsed_info['_atom_site.label_comp_id'],
      parsed_info['_atom_site.auth_asym_id'],
      parsed_info['_atom_site.label_asym_id'],
      parsed_info['_atom_site.auth_seq_id'],
      parsed_info['_atom_site.label_seq_id'],
      parsed_info['_atom_site.pdbx_PDB_ins_code'],
      parsed_info['_atom_site.group_PDB'],
      parsed_info['_atom_site.pdbx_PDB_model_num'],

      parsed_info['_atom_site.label_entity_id'],
      parsed_info['_atom_site.Cartn_x'],
      parsed_info['_atom_site.Cartn_y'],
      parsed_info['_atom_site.Cartn_z'],
      parsed_info['_atom_site.occupancy'],
      parsed_info['_atom_site.label_alt_id'],
      parsed_info['_atom_site.label_atom_id'],
      parsed_info['_atom_site.type_symbol'],
      )]


def _get_polymer_chains(
    *, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
  """Extracts polymer information for protein/dna/rna chains only.

  Args:
    parsed_info: _mmcif_dict produced by the Biopython parser.

  Returns:
    A dict mapping mmcif chain id to a list of Monomers.
  """
  # Get polymer information for each entity in the structure.
  entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)

  polymers = collections.defaultdict(list)
  for entity_poly_seq in entity_poly_seqs:
    polymers[entity_poly_seq['_entity_poly_seq.entity_id']].append(
        Monomer(id=entity_poly_seq['_entity_poly_seq.mon_id'],
                num=int(entity_poly_seq['_entity_poly_seq.num'])))

  # Get chemical compositions. Will allow us to identify which of these polymers
  # are proteins.
  chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

  # Get chains information for each entity. Necessary so that we can return a
  # dict keyed on chain id rather than entity.
  struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

  entity_to_mmcif_chains = collections.defaultdict(list)
  for struct_asym in struct_asyms:
    chain_id = struct_asym['_struct_asym.id']
    entity_id = struct_asym['_struct_asym.entity_id']
    entity_to_mmcif_chains[entity_id].append(chain_id)

  # Identify and return the valid protein chains.
  valid_chains = collections.defaultdict(dict)
  for entity_id, seq_info in polymers.items():
    chain_ids = entity_to_mmcif_chains[entity_id]

    if any(['peptide' in chem_comps[monomer.id]['_chem_comp.type'] for monomer in seq_info]):
      for chain_id in chain_ids:
        valid_chains['protein_' + chain_id] = seq_info
    if any(['DNA' in chem_comps[monomer.id]['_chem_comp.type'] for monomer in seq_info]):
      for chain_id in chain_ids:
        valid_chains['dna_' + chain_id] = seq_info
    if any(['RNA' in chem_comps[monomer.id]['_chem_comp.type'] for monomer in seq_info]):
      for chain_id in chain_ids:
        valid_chains['rna_' + chain_id] = seq_info

  appeared_chain_id_count = collections.defaultdict(int)
  chain_type_map = {}
  filter_valid_chains = {}
  for type_chainid, v in valid_chains.items():
    dtype, _chaid = type_chainid.rsplit('_')
    chain_type_map[_chaid] = dtype
    appeared_chain_id_count[_chaid] += 1

  for chainid, count in appeared_chain_id_count.items():
    if count > 1 and chain_type_map[chainid] in ['dna', 'rna']:
      # print('[WARNING] dna/rna hybrid, pass')
      continue
    dtype = chain_type_map[chainid]
    filter_valid_chains[f'{dtype}_' + chainid] = valid_chains[f'{dtype}_' + chainid]

  if filter_valid_chains:
    return filter_valid_chains
  else:
    return valid_chains


def _get_non_polymer_chains(
    *, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
  """Extracts non-polymer(ligand/ions) information, remove water, 

  Args:
    parsed_info: _mmcif_dict produced by the Biopython parser.

  Returns:
    A dict mapping mmcif chain id to a list of Monomer (non-polymer)
  """

  entity_ids = mmcif_loop_to_list('_entity.', parsed_info)
  ligand_ions_entity_ids = {}
  for entity_id in entity_ids:
    if entity_id['_entity.type'] not in ['polymer', 'water']:
      ligand_ions_entity_ids[entity_id['_entity.id']] = entity_id['_entity.type']

  # Get chains information for each entity. Necessary so that we can return a
  # dict keyed on chain id rather than entity.
  struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

  # ligand_ions entity id mapping to chain_id.
  entity_to_mmcif_chains = collections.defaultdict(list)
  for struct_asym in struct_asyms:
    chain_id = struct_asym['_struct_asym.id']
    entity_id = struct_asym['_struct_asym.entity_id']
    entity_to_mmcif_chains[entity_id].append(chain_id)

  # Identify and return the valid chains.
  valid_chains = {}
  for entity_id, entity_type in ligand_ions_entity_ids.items():
    valid_chains[entity_id] = entity_to_mmcif_chains[entity_id]

  return valid_chains


def _get_covalent_bond_info(
    *, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
  """Extracts covalent_bond information 

  Args:
    parsed_info: _mmcif_dict produced by the Biopython parser.
    ## NOTE: HF3 is only consider Token-level distance (_struct_conn.pdbx_dist_value： distance between two atoms)
    # use *_auth_*, ref: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.ptnr1_auth_asym_id.html

  Returns:
    A list of mmcif covalent_bond, support covalent, metal, disulf bond;
  """
  # Get polymer information for each entity in the structure.
  entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)

  polymers = collections.defaultdict(list)
  for entity_poly_seq in entity_poly_seqs:
    polymers[entity_poly_seq['_entity_poly_seq.entity_id']].append(
        Monomer(id=entity_poly_seq['_entity_poly_seq.mon_id'],
                num=int(entity_poly_seq['_entity_poly_seq.num'])))

  # Get chemical compositions. Will allow us to identify which of these polymers
  # are proteins.
  chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

  # Get chains information for each entity. Necessary so that we can return a
  # dict keyed on chain id rather than entity.
  struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

  entity_to_mmcif_chains = collections.defaultdict(list)
  for struct_asym in struct_asyms:
    chain_id = struct_asym['_struct_asym.id']
    entity_id = struct_asym['_struct_asym.entity_id']
    entity_to_mmcif_chains[entity_id].append(chain_id)

  filter_polymer_chains = set()
  for entity_id, seq_info in polymers.items():
    chain_ids = entity_to_mmcif_chains[entity_id]
    for chain_id in chain_ids:
      filter_polymer_chains.add(chain_id)


 # Get covalent_bond information for each entity/monomer in the structure.
  struct_conn_infos = mmcif_loop_to_list('_struct_conn.', parsed_info)
  covalent_bonds = []
  for struct_conn in struct_conn_infos:
    # # NOTE: now we only consider covalent bond
    # if struct_conn["_struct_conn.conn_type_id"] != "covale":
    #   continue

    ## filter covalent_bonds that are all in the polymer chains:
    if struct_conn['_struct_conn.ptnr1_label_asym_id'] in filter_polymer_chains and \
            struct_conn['_struct_conn.ptnr2_label_asym_id'] in filter_polymer_chains:
        continue

    if not _is_set(struct_conn["_struct_conn.pdbx_dist_value"]):
      struct_conn["_struct_conn.pdbx_dist_value"] = '99999'

    for i in [1, 2]:
      if struct_conn[f"_struct_conn.ptnr{i}_label_comp_id"] == 'MSE':
        struct_conn[f"_struct_conn.ptnr{i}_label_comp_id"] = 'MET'
      if struct_conn[f"_struct_conn.ptnr{i}_auth_comp_id"] == 'MSE':
        struct_conn[f"_struct_conn.ptnr{i}_auth_comp_id"] = 'MET'
      if struct_conn[f"_struct_conn.ptnr{i}_label_atom_id"] == 'SE':
        struct_conn[f"_struct_conn.ptnr{i}_label_atom_id"] = 'SD'

    covalent_bonds.append(
      CovalentBond(
          ptnr1_label_asym_id=struct_conn["_struct_conn.ptnr1_label_asym_id"],
          ptnr1_label_comp_id=struct_conn["_struct_conn.ptnr1_label_comp_id"],
          ptnr1_label_seq_id=struct_conn["_struct_conn.ptnr1_label_seq_id"],
          ptnr1_label_atom_id=struct_conn["_struct_conn.ptnr1_label_atom_id"],
          ptnr2_label_asym_id=struct_conn["_struct_conn.ptnr2_label_asym_id"],
          ptnr2_label_comp_id=struct_conn["_struct_conn.ptnr2_label_comp_id"],
          ptnr2_label_seq_id=struct_conn["_struct_conn.ptnr2_label_seq_id"],
          ptnr2_label_atom_id=struct_conn["_struct_conn.ptnr2_label_atom_id"],
          pdbx_dist_value=float(struct_conn["_struct_conn.pdbx_dist_value"]),
          pdbx_leaving_flag=struct_conn["_struct_conn.pdbx_leaving_atom_flag"],

          ## when meet the glycan, below informaction is used to identify the bond connection.
          ptnr1_auth_asym_id=struct_conn["_struct_conn.ptnr1_auth_asym_id"],
          ptnr1_auth_comp_id=struct_conn["_struct_conn.ptnr1_auth_comp_id"],
          ptnr1_auth_seq_id=struct_conn["_struct_conn.ptnr1_auth_seq_id"],
          ptnr2_auth_asym_id=struct_conn["_struct_conn.ptnr2_auth_asym_id"],
          ptnr2_auth_comp_id=struct_conn["_struct_conn.ptnr2_auth_comp_id"],
          ptnr2_auth_seq_id=struct_conn["_struct_conn.ptnr2_auth_seq_id"],

          bond_type=struct_conn["_struct_conn.conn_type_id"],
      )
    )
  return covalent_bonds



def _is_set(data: str) -> bool:
  """Returns False if data is a special mmCIF character indicating 'unset'."""
  return data not in ('.', '?')


def _residue_basic_clean(ResAtPos: ResidueAtPosition) -> ResidueAtPosition:
  ## Residue level; basic structure clean up, HF3:
  ## 1. MSE residues are converted to MET residues, waters are removed, 
  ## 2. arginine naming ambiguities are fixed (ensuring NH1 is always closer to CD than NH2).
  if ResAtPos.residue_name not in ['MSE', 'ARG']:
    return ResAtPos

  positions = ResAtPos.positions
  atom_ids = ResAtPos.atom_ids

  if ResAtPos.residue_name == 'MSE':
    _index_SE = -1
    try:
      _index_SE = atom_ids.index('SE')
    except:
      pass
    
    # MSE residues are converted to MET residues
    ResAtPos.residue_name = 'MET'
    if _index_SE != -1:
      atom_ids[_index_SE] = 'SD'
  elif ResAtPos.residue_name == 'ARG':
    cd_idx, nh1_idx, nh2_idx  = None, None, None
    for idx, atomid in enumerate(atom_ids):
      # Fix naming errors in arginine residues where NH2 is incorrectly
      # assigned to be closer to CD than NH1.
      if atomid == 'CD':
        cd_idx = idx
      elif atomid == 'NH1':
        nh1_idx = idx
      elif atomid == 'NH2':
        nh2_idx = idx

    if cd_idx is not None and nh1_idx is not None and nh2_idx is not None:
        if (np.linalg.norm(np.array(positions[nh1_idx]) - np.array(positions[cd_idx])) 
                > np.linalg.norm(np.array(positions[nh2_idx]) - np.array(positions[cd_idx]))):
          positions[nh1_idx], positions[nh2_idx] = positions[nh2_idx], positions[nh1_idx]

  ResAtPos.positions = positions
  ResAtPos.atom_ids = atom_ids

  return ResAtPos

def _flatten_position_infos(chain_position, type_chain_id, seqres_to_structure):
  ''' tbd. '''
  for seq_id, atom_objs in chain_position.items():
    positions_list = []
    atoms_ids_list = []
    resi_is_missing = False
    for atom_obj in atom_objs:
      if not atom_obj['is_missing']:
        positions_list.append([atom_obj['x'], atom_obj['y'], atom_obj['z']])
        atoms_ids_list.append(atom_obj['atom_id'])
      else:
        resi_is_missing = True
        continue

    if not resi_is_missing:
      seqres_to_structure[type_chain_id][seq_id] = ResidueAtPosition(
          positions=positions_list,
          atom_ids=atoms_ids_list,
          hetflag=atom_objs[0]['hetatm_atom'],
          residue_name=atom_objs[0]['residue_name'],
          is_missing=atom_objs[0]['is_missing'],
          author_chain_id=atom_objs[0]['author_chain_id'],
          mmcif_chain_id=atom_objs[0]['mmcif_chain_id'],
          author_seq_num=atom_objs[0]['author_seq_num'],
          mmcif_seq_num=atom_objs[0]['mmcif_seq_num'],
      )
    else:
      seqres_to_structure[type_chain_id][seq_id] = ResidueAtPosition(
          positions=positions_list,
          atom_ids=atoms_ids_list,
          hetflag='',
          residue_name=atom_objs[0]['residue_name'],
          is_missing=atom_objs[0]['is_missing'],
          author_chain_id='',
          mmcif_chain_id='',
          author_seq_num='',
          mmcif_seq_num='',
      )

    ## Residue, Basic Clean; 
    if type_chain_id.split('_')[0] == 'protein':
      seqres_to_structure[type_chain_id][seq_id] = \
                      _residue_basic_clean(seqres_to_structure[type_chain_id][seq_id])

  return seqres_to_structure


def _find_structure_position(valid_polymers, valid_nonpolymer, parsed_info, errors, file_id):
  """
    valid_polymers： {protein_<chain_id>: [Monomer]}, dna_<chain_id>: [Monomer]}, rna_<chain_id>: [Monomer]}}
    valid_nonpolymer： non-polymer/ligand: {entity_id: [chaind_id]}
  """

  # Determine the start number of each polymers-chain.
  seq_start_num = {dtype_chainid.split('_')[1]: min([monomer.num for monomer in seq])
                    for dtype_chainid, seq in valid_polymers.items()}

  ## Determine the dtype to chain_id mapping
  dtype_to_chainid = collections.defaultdict(set)
  chainid_to_dtype = {} 
  for dty_chain_id, _ in valid_polymers.items():
    dtype, chainid = dty_chain_id.split('_')
    dtype_to_chainid[dtype].add(chainid)
    if chainid in chainid_to_dtype:
      ## NOTE: dna-rna hybrid, we need to pass it.
      raise ValueError(f"dna/rna hybrid")
    chainid_to_dtype[chainid] = dtype
  for _, chain_ids in valid_nonpolymer.items():
    for chain_id_list in chain_ids.values():
      for chain_id in chain_id_list:
        dtype_to_chainid['ligand'].add(chain_id)
        if chain_id in chainid_to_dtype:
          ## NOTE: dna-rna hybrid, we need to pass it.
          raise ValueError(f"dna/rna hybrid")
        chainid_to_dtype[chain_id] = 'ligand'

  ## valid_nonpolymer_ccd is used to record the non-polymer residue name
  # NOTE: be careful about the glycan(E.g 1b3y) 
  valid_nonpolymer_ccd = collections.defaultdict(set) 

  # Determine the residue positions in the structure.
  # chain_to_structure_mappings: Dict mapping chain_id to a list of atom positions.
  ## Follow the dict format: {chain_id: {seq_idx: [atom_x, atom_y, atom_z]}}
  mmcif_chainid_to_author_chainid = {}
  chain_to_structure_mappings = collections.defaultdict(dict)
  all_pased_atom_list = _get_atom_site_list(parsed_info)
  current_idx = 0
  while current_idx < len(all_pased_atom_list):
    atom = all_pased_atom_list[current_idx]
    if atom.model_num != '1':
      # We only process the first model at the moment.
      current_idx += 1
      continue

    cur_mmcif_chain_id = atom.mmcif_chain_id

    if cur_mmcif_chain_id not in chainid_to_dtype:
      current_idx += 1
      continue

    if atom.label_type_symbol == 'H': # skip the hydrogen atoms
      current_idx += 1
      continue

    
    chain_dtype = chainid_to_dtype[cur_mmcif_chain_id]
    mmcif_chainid_to_author_chainid[atom.mmcif_chain_id] = atom.author_chain_id

    ## determine the chain_start_id
    if chain_dtype not in ['ligand', 'non_polymer']:
      ## record the chain seq id, after we will use to find missed residue position
      seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
    else:
      # seq_idx = 0  ## ligand/non-polymer we assert only has one residue
      # NOTE: be careful about the glycan(E.g 1b3y, 6NG3.cif), use `auth_seq_id`
      seq_idx = int(atom.author_seq_num) 

    current_chain = chain_to_structure_mappings.get(cur_mmcif_chain_id, {})
    current_res_atom_list = current_chain.get(seq_idx, [])

    atom_x, atom_y, atom_z = atom.pos_x, atom.pos_y, atom.pos_z
    atom_id = atom.label_atom_id # CB, ND2
    if _is_set(atom.label_alt_id):
      errors[(file_id, 'occupancy')] = f'occupancy is not 1.0'

      t_idx = current_idx
      _temp_atoms_occupancy = []
      while t_idx < len(all_pased_atom_list):
        _atom = all_pased_atom_list[t_idx]
        if _atom.label_atom_id == atom_id and _is_set(_atom.label_alt_id) \
                                              and _atom.mmcif_chain_id == cur_mmcif_chain_id:
          _temp_atoms_occupancy.append(_atom)
          t_idx += 1
        else:
          break
      current_idx = t_idx

      _temp_atoms_occupancy = sorted(_temp_atoms_occupancy, key=lambda x: x.occupancy, reverse=True)
      _atom_best = _temp_atoms_occupancy[0]
      atom_x, atom_y, atom_z = _atom_best.pos_x, _atom_best.pos_y, _atom_best.pos_z
    else:
      current_idx += 1

    current_res_atom_list.append({
      'residue_name': atom.residue_name, 
      'x': atom_x,
      'y': atom_y,
      'z': atom_z,
      'atom_id': atom_id,
      'atom_type': atom.label_type_symbol,
      'entity_id': atom.entity_id,
      'hetatm_atom': atom.hetatm_atom,
      'is_missing': False,

      "author_chain_id": atom.author_chain_id,
      "mmcif_chain_id": atom.mmcif_chain_id,
      "author_seq_num": atom.author_seq_num,
      "mmcif_seq_num":  atom.mmcif_seq_num,
    })
    current_chain[seq_idx] = current_res_atom_list
    chain_to_structure_mappings[cur_mmcif_chain_id] = current_chain

  filter_polymer_less4_resolved = set()  ## use to filter chain which has less than 4 resolved
  chain_to_seqres = {}
  # Add missing residue information to polymer-chains
  ## 1. To get the ccd seqs for each poly-chain, Mapping: TypeChainID: ccd seqs
  for dtype_chain_id, seq_info in valid_polymers.items():
    dtype, chain_id = dtype_chain_id.split('_')
    chain_position = chain_to_structure_mappings[chain_id]
    seq = []
    missed_count = 0
    for idx, monomer in enumerate(seq_info):
      if idx not in chain_position: # residue-level mapping
        missed_count += 1
        chain_position[idx] = [{
          'residue_name': "MET" if monomer.id == "MSE" and dtype == 'protein' else monomer.id,
          'is_missing': True, 
        }]
      seq_ccd = "MET" if monomer.id == "MSE" and dtype == 'protein' else monomer.id
      seq.append(f'({seq_ccd})')  ## only record the CCD seq; use ()

    if len(seq) - missed_count < 4:
      filter_polymer_less4_resolved.add(chain_id)
    seq = ''.join(seq)
    type_chain_id = f'{dtype}_{chain_id}_{mmcif_chainid_to_author_chainid[chain_id]}'
    chain_to_seqres[type_chain_id] = seq
  
  ### 1.1 to sorted the keys in chain_to_structure_mappings:
  for dtype_chain_id, seq_info in valid_polymers.items():
    dtype, chain_id = dtype_chain_id.split('_')
    current_chain_map = chain_to_structure_mappings[chain_id]

    _sorted_chain_map = {}
    for idx, monomer in enumerate(seq_info):
      _sorted_chain_map[idx] = current_chain_map[idx]
    chain_to_structure_mappings[chain_id] = _sorted_chain_map


  ## 2. To get the ligand seqs for each ligand-chain, Mapping: TypeChainID: ccd seqs
  for chain_id in dtype_to_chainid['ligand']:
    chain_position = chain_to_structure_mappings[chain_id]
    seq = []
    for idx in chain_position.keys():
      # we use the first atom to record the residue name (unique group by seq id)
      code = chain_position[idx][0]['residue_name'] 
      seq.append(f'({code})')
    seq = ''.join(seq)
    type_chain_id = f'ligand_{chain_id}_{mmcif_chainid_to_author_chainid[chain_id]}'
    chain_to_seqres[type_chain_id] = seq

  ## 3. To get seqres_to_structure. Mapping[TypeChainID, Mapping[int, ResidueAtPosition]]
  seqres_to_structure = collections.defaultdict(dict)
  for dtype_chain_id, seq_info in valid_polymers.items():
    dtype, chain_id = dtype_chain_id.split('_')
    chain_position = chain_to_structure_mappings[chain_id]
    type_chain_id = f'{dtype}_{chain_id}_{mmcif_chainid_to_author_chainid[chain_id]}'
    seqres_to_structure = _flatten_position_infos(chain_position, 
                                                  type_chain_id, seqres_to_structure)

  ## 4. To get ligand seqres_to_structure.
  for chain_id in dtype_to_chainid['ligand']:
    chain_position = chain_to_structure_mappings[chain_id]
    type_chain_id = f'ligand_{chain_id}_{mmcif_chainid_to_author_chainid[chain_id]}'
    seqres_to_structure = _flatten_position_infos(chain_position, 
                                                  type_chain_id, seqres_to_structure)


  ### Post-Process
  ## add mmcif_chain id to TypeChainID mapping;
  errors[(file_id, 'polymer_less4_resolved')] = ','.join(filter_polymer_less4_resolved)
  mmcif_chain_to_typechain = {}
  for type_cha in chain_to_seqres.keys():
    if type_cha.split('_')[1] in filter_polymer_less4_resolved:
      continue
    mmcif_chain_to_typechain[type_cha.split('_')[1]] = type_cha

  chain_to_seqres = {key.split('_')[1]: value for key, value in chain_to_seqres.items()
                                          if key.split('_')[1] not in filter_polymer_less4_resolved}
  seqres_to_structure = {key.split('_')[1]: value for key, value in seqres_to_structure.items()
                                          if key.split('_')[1] not in filter_polymer_less4_resolved}

  ## sorted the keys 
  mmcif_chain_to_typechain = {key: value for key, value 
                                      in sorted(mmcif_chain_to_typechain.items(), key=lambda x: x[0])}
  chain_to_seqres = {key: value for key, value 
                                      in sorted(chain_to_seqres.items(), key=lambda x: x[0])}
  seqres_to_structure = {key: value for key, value 
                                      in sorted(seqres_to_structure.items(), key=lambda x: x[0])}

  return {
    'chain_to_seqres': chain_to_seqres,
    'seqres_to_structure': seqres_to_structure,
    'mmcif_chain_to_typechain': mmcif_chain_to_typechain,
  }


def filter_protein_chains(seqres, chain_dtype):
  return {chain_id: residues for chain_id, residues in seqres.items() if chain_dtype.get(chain_id, "").startswith("protein")}


def filter_unk_chain(chain_to_seqres, chain_dtype,time_record,file_id):
  ##Remove polymer chains with all unknown residues 
  start_time =time.time()
  chains_to_remove = set()

  for chain_id, residues in chain_to_seqres.items():
    if chain_dtype.get(chain_id, "").startswith("protein"):
      residue_list = residues.strip('()').split(')(')
      all_unk = all(residue == "UNK" for residue in residue_list)
      if all_unk:
          chains_to_remove.add(chain_id)

  elapsed_time = time.time() - start_time
  logging.info(f"{file_id}: filter_unk_chain executed in {elapsed_time} seconds")
  time_record['filter_unk_chain'] = elapsed_time

  return chains_to_remove


def calculate_distance(coord1, coord2):
  pos1 = np.array(coord1)
  pos2 = np.array(coord2)
  return np.linalg.norm(pos1 - pos2)

def extract_atoms_from_chains(protein_chains):
  atoms_per_chain = {}
  for chain_id, residues in protein_chains.items():
    atoms = []
    for residue in residues.values():
      if not residue.is_missing:
        for atom_id, position in zip(residue.atom_ids, residue.positions):
          if atom_id == "CA":
            atoms.append(position)
    atoms_per_chain[chain_id] = atoms
  return atoms_per_chain

def select_chain_to_remove(clashing_score_1, clashing_score_2, chain_id_1, chain_id_2, ls_atoms_1, ls_atoms_2):
  if clashing_score_1 > clashing_score_2:
    return chain_id_1
  elif clashing_score_1 < clashing_score_2:
    return chain_id_2
  elif ls_atoms_1 > ls_atoms_2:
    return chain_id_2
  elif ls_atoms_1 < ls_atoms_2:
    return chain_id_1
  elif chain_id_1 < chain_id_2:
    return chain_id_1
  else:
    return chain_id_2

def filter_clash_chains(seqres, chain_dtype,time_record,file_id):
  ## Remove clashing chains
  start_time =time.time()
  protein_chains = filter_protein_chains(seqres, chain_dtype)
  atoms_per_chain = extract_atoms_from_chains(protein_chains)
  #print(atoms_per_chain)
  
  clashing_scores = {}
  chains_to_remove =set()
  atoms_per_chain = {k: v for k, v in atoms_per_chain.items() if v}
  chain_ids = list(atoms_per_chain.keys())

  for i, chain_id_1 in enumerate(chain_ids):
    for j, chain_id_2 in enumerate(chain_ids):
      if i >= j:
        continue
      atoms_1 = atoms_per_chain[chain_id_1]
      atoms_2 = atoms_per_chain[chain_id_2]
      clashing_count_1 = 0
      clashing_count_2 = 0

      for pos1 in atoms_1:
        for pos2 in atoms_2:
          if calculate_distance(pos1, pos2) < 1.7:
            clashing_count_1 += 1
            break

      for pos2 in atoms_2:
        for pos1 in atoms_1:
          if calculate_distance(pos1, pos2) < 1.7:
            clashing_count_2 += 1
            break

      ls_atoms_1=len(atoms_1)
      ls_atoms_2=len(atoms_2)

      clashing_score_1 = clashing_count_1 / len(atoms_1)
      clashing_score_2 = clashing_count_2 / len(atoms_2)

      if clashing_score_1 > 0.3  and clashing_score_2 > 0.3 :
        chains_to_remove.add(select_chain_to_remove(clashing_score_1, clashing_score_2,chain_id_1,chain_id_2,ls_atoms_1,ls_atoms_2))
      clashing_scores[f'{chain_id_1}_{chain_id_2}'] = [clashing_score_1, clashing_score_2]

  elapsed_time = time.time() - start_time
  
  logging.info(f"{file_id}: filter_clash_chains  executed in {elapsed_time} seconds")
  time_record['filter_clash_chains'] = elapsed_time

  return chains_to_remove,clashing_scores


def filter_ca_chain(seqres, chain_dtype,time_record,file_id):
  ## Remove protein chains with consecutive Cα atoms >10 Å
  start_time =time.time()
  protein_seqres = filter_protein_chains(seqres, chain_dtype)
  
  continuous_segments = {}
  consecutive_position ={}
  chains_to_remove = set()

  for chain_id, residues in protein_seqres.items():
    segments = []
    current_segment = []
    current_ca_positions = []

    for seq_num, residue in residues.items():
      if not residue.is_missing:
        current_segment.append(seq_num)
        for atom_id, position in zip(residue.atom_ids, residue.positions):
          if atom_id == "CA":
            current_ca_positions.append(position)
      else:
        if current_segment:
          segments.append({
              'sequence': current_segment,
              'ca_positions': current_ca_positions
          })
          current_segment = []
          current_ca_positions = []

    
    if current_segment:
      segments.append({
          'sequence': current_segment,
          'ca_positions': current_ca_positions
      })
    #print(chain_id,segments)
    for segment in segments:
      ca_positions = segment['ca_positions']
      for i in range(len(ca_positions) - 1):
        if calculate_distance(ca_positions[i], ca_positions[i + 1]) > 10.0:
          chains_to_remove.add(chain_id)
          consecutive_position[chain_id] =[ca_positions[i],ca_positions[i+1]]
          break
      if chain_id in chains_to_remove:
        break

    continuous_segments[chain_id] = segments
  
  elapsed_time = time.time() - start_time
  logging.info(f"{file_id}: filter_ca_chain executed in {elapsed_time} seconds")
  time_record['filter_ca_chain'] = elapsed_time

  return chains_to_remove , continuous_segments,consecutive_position


def filter_ligand_chain(chain_to_seqres, chain_dtype,filter_ligand_lst,time_record,file_id):

  ## Remove ligand chains ,which are Crystallization aids or exclusion ligands
  start_time =time.time()
  chains_to_remove = set()
  record_crystallization_protein_chain =set()
  for chain_id, residues in chain_to_seqres.items():
    if chain_dtype.get(chain_id, "").startswith("ligand"):
      residue_list = residues.strip('()').split(')(')
      for residue in residue_list:
        if residue in filter_ligand_lst:
          chains_to_remove.add(chain_id)
          break
    else:
      residue_list = residues.strip('()').split(')(')
      for residue in residue_list:
        if residue in filter_ligand_lst:
          record_crystallization_protein_chain.add(chain_id)
          break

  elapsed_time = time.time() - start_time
  logging.info(f"{file_id}: filter_ligand_chain executed in {elapsed_time} seconds")
  time_record['filter_ligand_chain'] = elapsed_time
  return chains_to_remove,record_crystallization_protein_chain


def filter(chain_to_seqres,chain_dtype,seqres,file_id,errors):

  filter_ligand_lst = list(set(crystallization_aids) | set(ligand_exclusion_list))

  time_record = {}
  unk_chains_to_remove = filter_unk_chain(chain_to_seqres, chain_dtype,time_record,file_id)
  clash_chains_to_remove,clashing_scores = filter_clash_chains(seqres, chain_dtype,time_record,file_id)
  ca_chains_to_remove, continuous_segments,consecutive_position = filter_ca_chain(seqres, chain_dtype,time_record,file_id)
  # clash_chains_to_remove,clashing_scores = set(), []
  # ca_chains_to_remove, continuous_segments,consecutive_position = set(), [], []
  crystal_exclusion_chains_to_remove,record_crystallization_protein_chain = filter_ligand_chain(chain_to_seqres, chain_dtype,filter_ligand_lst,time_record,file_id)

  # print(clashing_scores)
  #print(continuous_segments)
  #print(record_crystallization_protein_chain)
  chains_to_remove = unk_chains_to_remove | clash_chains_to_remove | ca_chains_to_remove | crystal_exclusion_chains_to_remove

  details = [
  '{},{} chains residue all unknown'.format(file_id,unk_chains_to_remove),
  '{},{} chains clash, clashing_scores {}'.format(file_id,clash_chains_to_remove, clashing_scores),
  '{},{} chains consecutive Ca atoms > 10A, consecutive_position {}'.format(file_id,ca_chains_to_remove, consecutive_position),
  '{},{} chains ligand is crystallization aids or exclusion ligands, {} protein chain have such ligand'.format(file_id,crystal_exclusion_chains_to_remove, record_crystallization_protein_chain)
  ]

  for detail in details:
    logging.info(detail)

  logging.info(f'{file_id}: {chains_to_remove} chains to be removed')
  errors[(file_id, 'filter_result')] = '{},{} chains to be removed'.format(file_id, chains_to_remove)
  #errors[(file_id,'detail')]= '\n'.join(details)
  if (file_id, 'detail') not in errors:
        errors[(file_id, 'details')] = []
  errors[(file_id,'details')].extend(details)

  errors[(file_id,'filter_time')] = time_record


  for chain_id in chains_to_remove:
    if chain_id in chain_to_seqres:
      del chain_to_seqres[chain_id]
    if chain_id in chain_dtype:
      del chain_dtype[chain_id]
    if chain_id in seqres:
      del seqres[chain_id]
  return chain_to_seqres, chain_dtype,seqres


@functools.lru_cache(16, typed=False)
def parse(*,
          file_id: str,
          mmcif_string: str,
          is_assembly_filter: bool = False,
          catch_all_errors: bool = True) -> ParsingResult:
  ''' tbd. '''
  t1 = time.time()
  errors = {}
  try:
    parser = PDB.MMCIFParser(QUIET=True)
    handle = io.StringIO(mmcif_string)
    full_structure = parser.get_structure('', handle)
    first_model_structure = _get_first_model(full_structure)
    # Extract the _mmcif_dict from the parser, which contains useful fields not
    # reflected in the Biopython structure.
    parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

    # Ensure all values are lists, even if singletons.
    for key, value in parsed_info.items():
      if not isinstance(value, list):
        parsed_info[key] = [value]

    header = _get_header(parsed_info)

    # Determine the polymers, nonpolymers, covalent-bond, and their start numbers according to the
    # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
    valid_polymers = _get_polymer_chains(parsed_info=parsed_info) # dtype: {chain id: a list of Monomers}
    valid_nonpolymer = _get_non_polymer_chains(parsed_info=parsed_info) # entity_id: list of chain id
    valid_covalent_bonds = _get_covalent_bond_info(parsed_info=parsed_info)
    
    if not valid_polymers and not valid_nonpolymer and not valid_covalent_bonds:
      return ParsingResult(
          None, {(file_id, ''): 'No entity found in this file.'})
    else:
      errors[(file_id, 'None')] =  ','.join([ 
            _n for _n, _obj in zip(['poly', 'nonpoly', 'bond'], 
                                    [valid_polymers, valid_nonpolymer, valid_covalent_bonds]) if not _obj])
    valid_nonpolymer = {'ligand': valid_nonpolymer} # dtype: {entity_id: list of chain id}

    structure_infos = _find_structure_position(valid_polymers, valid_nonpolymer, parsed_info, 
                                                                              errors, file_id)
    
    if is_assembly_filter:
      structure_infos['chain_to_seqres'],structure_infos['mmcif_chain_to_typechain'],structure_infos['seqres_to_structure']=filter(
        structure_infos['chain_to_seqres'],structure_infos['mmcif_chain_to_typechain'],structure_infos['seqres_to_structure'],file_id,errors)

    mmcif_object = MmcifObject(
          file_id=file_id,
          header=header,
          structure=first_model_structure,
          mmcif_chain_to_typechain=structure_infos['mmcif_chain_to_typechain'],
          chain_to_seqres=structure_infos['chain_to_seqres'],
          seqres_to_structure=structure_infos['seqres_to_structure'],
          covalent_bonds=valid_covalent_bonds,
          raw_string=parsed_info)
    
    errors[(file_id, f'hybrid', 'success')] = ''
    errors[(file_id, f'hybrid', 'time')] = time.time() - t1
    return ParsingResult(mmcif_object=mmcif_object, errors=errors)
  except Exception as e:  # pylint:disable=broad-except
    errors[(file_id, 'hybrid', 'ERROR')] = e
    errors[(file_id, f'hybrid', 'time')] = time.time() - t1
    # traceback.print_exc()
    if not catch_all_errors:
      raise
    return ParsingResult(mmcif_object=None, errors=errors)