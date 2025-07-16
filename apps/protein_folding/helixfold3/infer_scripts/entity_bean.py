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

"""Entity bean definitions for preprocessing.

This module contains the EntityBean class used to represent
converted entities during the preprocessing pipeline.
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=False)
class EntityBean:
    """Class bean representing a converted entity in the input JSON.
    
    private Attributes:
        _ERROR_MESSAGE_MAPPING: dict, error message mapping, 0 means SUCCESS.
        _error_message: str, error message
    public Attributes:
        dtype: Optional[str], entity type, one of ['protein', 'rna', 'dna', 'ligand']
        seqs: Optional[str], CCD sequence, linked by (), such as (GLY)(ALA)...
        msa_seqs: Optional[str], MSA sequence, one letter per residue, such as GLYALA...
        count: Optional[int], the number of the entity.
        extra_mol_infos: Optional[dict], extra molecular information, such as mol_id, mol_name, etc.
        raw_info: Optional[dict], raw information, such as CCD, SMILES, etc. from the original input.
        error_code: int, error code. Shouled be assigned by the constructor.
    """
    error_code: int
    _error_message: str = "SUCCESS"
    _ERROR_MESSAGE_MAPPING = {
        -1: "Others error",
        0: 'SUCCESS',
        1: 'Invalid ligand generate from SMILES: {0}',
        2: 'Invalid entity convert',
        3: 'Unknown error',
        4: "Invalid ligand, CCD {0} is not supported now",
        5: "Invalid modified residues (CCD) in the polymer chain",
        6: "Invalid modification index: {0}, expected 1-{1}",
        7: "Duplicate modification index: {0}",
        8: "Unsupported modification type: {0}",
        9: "Sequence length ({0}) too short, minimum length is 4",
        10: "Sequence length ({0}) exceeds maximum allowed ({1})",
        11: "Invalid {0} sequence characters: {1}",
        12: "Invalid ion CCD code: {0}.",
        13: "SMILES contains more than 100 heavy atoms: {0}",
        14: "Error validating SMILES: {0}, {1}",
        15: "Error validating R-SMILES: {0}",
        16: "Side chain modification error: {0}"
    }

    dtype: Optional[str] = None
    seqs: Optional[str] = None
    msa_seqs: Optional[str] = None
    count: Optional[int] = None
    extra_mol_infos: Optional[dict] = None
    raw_info: Optional[dict] = None

    def __post_init__(self):
        self._error_message = self._ERROR_MESSAGE_MAPPING.get(self.error_code, "Unknown error")

    @classmethod
    def create_with_kwargs(cls, error_code: int, **kwargs):
        """
        Create an EntityBean object with the given error code and additional keyword arguments.

        Args:
            error_code: int, error code
            **kwargs: Additional keyword arguments

        Returns:
            EntityBean: An EntityBean object with the given error code and additional keyword arguments
        """
        entity = cls(error_code)
        if error_code == 4:
            entity._error_message = entity._error_message.format(kwargs['CCD'])
        elif error_code == 1:
            entity._error_message = entity._error_message.format(kwargs['smiles'])
        elif error_code == 6:
            entity._error_message = entity._error_message.format(kwargs['index'], kwargs['max_index'])
        elif error_code == 7:
            entity._error_message = entity._error_message.format(kwargs['index'])
        elif error_code == 8:
            entity._error_message = entity._error_message.format(kwargs['mod_type'])
        elif error_code == 9:
            entity._error_message = entity._error_message.format(kwargs['length'])
        elif error_code == 10:
            entity._error_message = entity._error_message.format(kwargs['length'], kwargs['max_length'])
        elif error_code == 11:
            entity._error_message = entity._error_message.format(kwargs['type'], kwargs['sequence'])
        elif error_code == 12:
            entity._error_message = entity._error_message.format(kwargs['ccd'])
        elif error_code == 13:
            entity._error_message = entity._error_message.format(kwargs['smiles'])
        elif error_code == 14:
            entity._error_message = entity._error_message.format(kwargs['smiles'], kwargs['error'])
        elif error_code == 15:
            entity._error_message = entity._error_message.format(kwargs['error'])
        elif error_code == 16:
            entity._error_message = entity._error_message.format(kwargs['message'])
        return entity

    def set_error_message(self, message):
        self._error_message = message

    @property
    def error_message(self) -> str:
        return self._error_message 