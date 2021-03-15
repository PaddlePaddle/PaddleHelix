"""
Helper functions for OGBComp.ipynb.
Some functions come from different sources and so have different comment formats
"""

# The code below is copy and pasted from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/smiles2graph.py
from ogb.utils.features import (
    atom_to_feature_vector,
    bond_to_feature_vector,
)
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Callable, Generator, Iterable, Union
from ogb.graphproppred import GraphPropPredDataset
from collections import Sized
import multiprocessing
from multiprocessing import Pool


def mol_to_data_obj(smiles_string: str):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


# End code from ogb/blob/master/examples/graphproppred/mol/smiles2graph.py

# These codes are adapted from https://github.com/swansonk14/p_tqdm/blob/master/p_tqdm/p_tqdm.py
def parallel_apply(smi: Iterable[str], func=None):
    # Iterate over a list-like of strings, not s single string
    assert not isinstance(smi, str)
    assert func is not None

    generator = _parallel(func, smi)
    feat_list = list(generator)
    print("Catenating results")
    try:
        df = pd.concat(feat_list, axis=1).T
    except Exception:
        df = pd.Series(feat_list)
    df.index = smi
    return df


def _parallel(function: Callable, *iterables: Iterable, **kwargs: Any) -> Generator:
    """Returns a generator for a parallel map with a progress bar.

    Arguments:
        ordered(bool): True for an ordered map, false for an unordered map.
        function(Callable): The function to apply to each element of the given Iterables.
        iterables(Tuple[Iterable]): One or more Iterables containing the data to be mapped.

    Returns:
        A generator which will apply the function to each element of the given Iterables
        in parallel in order with a progress bar.
    """

    # Extract num_cpus
    num_cpus = kwargs.pop("num_cpus", None)

    # Determine num_cpus
    if num_cpus is None:
        num_cpus = multiprocessing.cpu_count()
    elif type(num_cpus) == float:
        num_cpus = int(round(num_cpus * multiprocessing.cpu_count()))

    # Determine length of tqdm (equal to length of shortest iterable)
    length = min(len(iterable) for iterable in iterables if isinstance(iterable, Sized))

    # Create parallel generator
    pool = Pool(num_cpus)

    for item in tqdm(pool.imap(function, *iterables), total=length, **kwargs):
        yield item


# End codes from swansonk14/p_tqdm/blob/master/p_tqdm/p_tqdm.py


def getmorganfingerprint2(x: Chem.rdchem.Mol, **kws):
    """
    Calculate morgan fingerprints from an rdkit molecule.
    """
    x = smiles2mol_maybe(x)
    return pd.Series(list(AllChem.GetMorganFingerprintAsBitVect(x, 2)))


def smiles2mol_maybe(smiles: Union[str, Chem.rdchem.Mol]):
    """
    Convert from smiles string to rdkit molecule, if necessary
    """
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        return mol
    elif not isinstance(smiles, Chem.rdchem.Mol):
        print(f"{type(smiles)}")
        raise Exception(f"Unrecognized input type: {type(smiles)}")
    else:
        return smiles


def spotcheck_order(smiles: pd.Series, dataset: GraphPropPredDataset, num_checks=5000):
    """
    Make sure performance is not due to differences in sort order
    btwn graphs and mapping file.
    """
    print("Spot checking order between mapping file and graphs...")
    idxs = np.arange(num_checks)

    for ii in tqdm(idxs):
        tg = dataset[ii][0]
        graph = mol_to_data_obj(smiles.iloc[ii])
        assert np.all(tg["node_feat"][:, :7] == graph["node_feat"][:, :7])


def process_y(y_train: pd.Series, max_mult=20, large_sampsize=50000):
    """
    Drop missing values, downsample the negative class
    if sample size is large and there is significant class imbalance
    """
    # Remove missing labels
    ytr = y_train.dropna()
    return ytr

    # The code below assumes the negative class is over-represented.
    assert ytr.mean() < 0.5

    # If there are too many negative samples, downsample
    if len(ytr) > large_sampsize:
        label_counts = ytr.value_counts()
        max_neg = max(label_counts.loc[1.0] * max_mult, large_sampsize)

        y_neg = ytr[ytr == 0.0]
        y_pos = ytr[ytr == 1.0]

        new_y = pd.concat(
            [y_neg.sample(frac=1.0, replace=False).iloc[:max_neg], y_pos]
        ).sample(frac=1.0, replace=False)
        return new_y
    else:
        return ytr


def probs_dict2df(all_probs: dict, task="ogbg-molpcba", dset="test", nreps=4):
    """
    Extract probabilities from the result dictionary and convert to a dataframe.
    """

    return [
        pd.DataFrame(
            {
                kk: vv
                for kk, vv in all_probs.items()
                if kk[0] == task and kk[-1] == dset and kk[2] == rep
            }
        )
        .T.reset_index(level=[0, 2, 3], drop=True)
        .T
        for rep in range(nreps)
    ]
