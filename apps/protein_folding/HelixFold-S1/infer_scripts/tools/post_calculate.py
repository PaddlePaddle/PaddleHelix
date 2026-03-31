import numpy as np

def calculate_chain_pair_pae_matrix(pae_matrix, token_chain_ids, 
                                    mask_chain,  mask_value=np.nan, metric_type='mean'):
    """
    Compute a square matrix where each entry (i, j) is the minimum PAE between chains i and j.
    
    Args:
        pae_matrix (list or np.ndarray): A 2D numpy array where entry (i, j) is the PAE value between residues i and j. (N_token, N_token)
        token_chain_ids (list or np.ndarray): A list where each entry gives the chain ID of the corresponding residue. (N_token)
        mask_chain (np.ndarray): A boolean mask indicating which chain IDs are valid.
        mask_value (float): The value to use when masking invalid entries. Defaults to NaN.
        type (str): The type of aggregation to perform on the PAEs within each chain pair. Can be 'mean', 'max', or 'min'. Defaults to 'mean'.
    Returns:
        np.ndarray: A square matrix of shape (num_chains, num_chains), 
            where entry (i, j) is the minimum PAE between chains i and j.
    """
    def _func_cal(metric_type):
        assert metric_type in ['mean', 'max', 'min'], "Invalid metric_type provided"
        operations = {
            "mean": lambda x: np.mean(x),
            "max": lambda x: np.max(x),
            "min": lambda x: np.min(x),
        }
        return operations[metric_type]

    def _convert_list_to_np(object):
        if type(object) == list:
            return np.array(object)
        elif type(object) == np.ndarray:
            return object
        else:
            raise TypeError("Input must be either a list or a numpy array.")
    
    pae_matrix = _convert_list_to_np(pae_matrix)
    token_chain_ids = _convert_list_to_np(token_chain_ids)
    operation = _func_cal(metric_type)

    unique_chains = np.unique(token_chain_ids)
    num_chains = len(unique_chains)
    chain_pair_pae_min = np.full((num_chains, num_chains), np.nan) 

    for i, chain_i in enumerate(unique_chains):
        indices_i = np.where(token_chain_ids == chain_i)[0]

        for j, chain_j in enumerate(unique_chains):
            
            indices_j = np.where(token_chain_ids == chain_j)[0]            
            # Extract submatrix for tokens in chain_i (rows) and chain_j (columns)
            sub_matrix = pae_matrix[np.ix_(indices_i, indices_j)]
            min_pae = operation(sub_matrix)
            chain_pair_pae_min[i, j] = min_pae
    
    chain_pair_pae_min[~mask_chain, :] = mask_value
    return chain_pair_pae_min


def calculate_token_plddts(atom_plddts, token2atom):
    """
    Calculate per-token pLDDT values from per-atom pLDDT values.

    Args:
    atom_plddts : 
        Per-atom pLDDT values. Shape: [n_atoms].
    token2atom : 
        Mapping from atoms to tokens. Shape: [n_atoms].

    Returns:
        Per-token pLDDT values. Shape: [n_tokens].
    """
    n_tokens = np.max(token2atom) + 1
    token_plddts = np.zeros(n_tokens)
    sums = np.bincount(token2atom, weights=atom_plddts, minlength=n_tokens)
    counts = np.bincount(token2atom, minlength=n_tokens)
    token_plddts = sums / (counts + 1e-10)
    
    return token_plddts