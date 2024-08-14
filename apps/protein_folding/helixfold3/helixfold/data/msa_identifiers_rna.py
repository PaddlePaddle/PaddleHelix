

"""Utilities for extracting identifiers from RNA MSA sequence descriptions."""

from helixfold.data.msa_identifiers import Identifiers


def get_identifiers(description: str, species_identifer_df) -> Identifiers:
  """Computes extra MSA features from the description."""

  identifer = ''

  if species_identifer_df is None:
    return Identifiers(
          species_id=identifer)

  word_list = description.split()
  for i, word in enumerate(word_list):

    if word[0].isupper() and i+1 < len(word_list):
      name =  ' '.join(word_list[i:i+2])

      matching_rows = species_identifer_df.loc[species_identifer_df['Scientific name'] == name, 'Mnemonic']  
      if not matching_rows.empty:  
        identifer = matching_rows.iloc[0]

        break

  return Identifiers(
        species_id=identifer)


