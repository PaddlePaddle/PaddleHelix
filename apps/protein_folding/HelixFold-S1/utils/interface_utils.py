import pickle
import random
import numpy as np
import json
import os
import glob
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

## TODO: Interface config should be loaded from the config file rather than hardcoded here.
## @liuyang: 2025-03-20 
GEN_INTERFACE_CONFIG = {
    "interface_source": "none", 
    "interface_type": "any_interface",
    "dist_thres": 5, 
    "interface_sample": {"top_n": 40, "min_m": 1, "max_m": 1, "seed": None},
    "interface_gen": {"gen_repeats":1, "interface_size": 1, "seed": 2, "temperature": 1.0,
                    "sampling_method": "in_repeats", "sampling_beam": 1000},
    "save_interface_prob": True
}

## TODO: Interface config should be loaded from the config file rather than hardcoded here.
## @liuyang: 2025-03-20 
INFERENCE_INTERFACE_CONFIG = {
    'use_interface_info': True,

    "interface_config":  {"dist_thres": 5, "top_n": 40, "min_m": 1, "max_m": 1, "seed": 1},
    "interface_type": "dynamic_sampling",
    "interface_indirect_thres": 5,
    "interface_weight_b": 0,
    "interface_weight_c": 0.4,
    "interface_conf_metric_name": "ranking_confidence",
    "conf_metric_accumulate_method": "max",
    "interface_sampling_method": "topn_cluster_Allnode",
    "topn": 50
}


class InterfaceInfo:
    def __init__(self, n_token, indexes=None, match_annotations=None, probability=None, precision=None, confidences=None):
        """
        Initialize the InterfaceInfo object.
        
        Parameters:
        - n_token: int, the total number of tokens in the protein complex conformation.
        - indexes: list of lists, each containing two ints representing a pair of token indexes across an interface (optional).
        - match_annotations: list of ints, each representing the match annotation for a node (optional).
        - probability: float, the posterior probability of the interface as predicted by the model (optional).
        - precision: float, the accuracy of the interface, i.e., the proportion of nodes with match_annotation=1 (optional).
        - confidences: dict, predicted confidence of the predicted structure with this interface, i.e., ranking_confidence, ptm, iptm, mean_plddt
        """
        self.n_token = n_token
        self.nodes = []  # List to store nodes, each node is a dict with 'index' and 'match_annotation'
        self.probability = probability
        self.confidences = confidences
        
        # If indexes are provided, initialize nodes with given or default match_annotations
        if indexes is not None:
            if match_annotations is None:
                match_annotations = [-1] * len(indexes)
            
            for idx, match in zip(indexes, match_annotations):
                self.add_node(idx, match)
                
            # If precision is not provided, calculate it based on match_annotations
            if precision is None:
                self.update_precision()
            else:
                self.precision = precision
        else:
            self.precision = -1  # Default precision to -1 if not provided and no indexes to calculate from

    def set_probability(self, probability):
        """
        Set the posterior probability of the interface.
        
        Parameters:
        - probability: float, the new value for the probability.
        """
        self.probability = probability
    
    def add_node(self, index, match_annotation):
        """
        Add a new node to the interface.
        
        Parameters:
        - index: list of two ints, representing a pair of token indexes across the interface.
        - match_annotation: int, the match annotation for this node.
        """
        index_sorted = sorted([int(index[0]), int(index[1])])
        self.nodes.append({'index': index_sorted, 'match_annotation': match_annotation})
        self.update_precision()  # Update precision after adding a new node

    def update_precision(self):
        """
        Update the precision of the interface based on the current nodes.
        """
        if self.nodes:
            total_nodes = sum(node['match_annotation'] >= 0 for node in self.nodes)
            matching_nodes = sum(node['match_annotation'] == 1 for node in self.nodes)
            # Calculate precision as the proportion of matching nodes
            self.precision = matching_nodes / total_nodes if total_nodes > 0 else -1
        else:
            self.precision = -1

    def interface_map(self):
        """
        Generate an interface map as a numpy array.
        
        Returns:
        - numpy.ndarray of shape (n_token, n_token) with 1s at positions corresponding to interface nodes and 0s elsewhere.
        """
        interface_map = np.zeros((self.n_token, self.n_token), dtype=int)
        for node in self.nodes:
            i, j = node['index']
            interface_map[i, j] = 1
            interface_map[j, i] = 1  # Ensure symmetry
        return interface_map

    def __json__(self):
        """
        Serialize the interface information to a JSON-compatible format.
        
        Returns:
        - dict, representing the interface information in a JSON-friendly structure.
        """
        return self.to_json()

    def to_json(self):
        """
        Serialize the interface information to a JSON string.
        
        Returns:
        - str, a JSON string representing the interface information.
        """
        interface_data = {
            'n_token': self.n_token,
            'nodes': self.nodes,
            'probability': self.probability,
            'precision': self.precision,
            'confidences': self.confidences
        }
        return json.dumps(interface_data, indent=4)

    def to_dict(self):
        """
        Serialize the interface information to a dictionary.
        
        Returns:
        - dict, representing the interface information in a Pythonic structure.
        """
        return {
            'n_token': self.n_token,
            'nodes': self.nodes,
            'probability': self.probability,
            'precision': self.precision,
            'confidences': self.confidences
        }

    @staticmethod
    def from_json(json_str):
        """
        Constructs an InterfaceInfo object from a JSON string.
        
        Parameters:
        json_str (str): A JSON string containing interface_info data.
        
        Returns:
        InterfaceInfo: An InterfaceInfo object constructed from the JSON string.
        
        Raises:
        ValueError: If the JSON string cannot be parsed or if required fields are missing or invalid.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unable to parse JSON string: {e}")
    
        return InterfaceInfo.from_dict(data)

    @staticmethod
    def from_dict(data):
        """
        Constructs an InterfaceInfo object from a dictionary.
        
        Parameters:
        data (dict): A dictionary containing interface_info data.
        
        Returns:
        InterfaceInfo: An InterfaceInfo object constructed from the dictionary.
        
        Raises:
        ValueError: If required fields are missing or invalid in the input dictionary.
        """
        # Extract required fields from the JSON data
        n_token = data.get('n_token')
        nodes = data.get('nodes', [])
        probability = data.get('probability')
        precision = data.get('precision')
        confidences = data.get('confidences', None)
    
        # Validate that n_token is present and is an integer
        if n_token is None or not isinstance(n_token, int):
            raise ValueError("JSON data is missing the 'n_token' field or the field is not an integer")
    
        # Instantiate an InterfaceInfo object
        interface_info = InterfaceInfo(n_token, probability=probability, precision=precision, confidences=confidences)
    
        # Add nodes
        for node in nodes:
            index = node.get('index')
            match_annotation = node.get('match_annotation')
            interface_info.add_node(index, match_annotation)
    
        return interface_info

    def get_interface_key(self):
        """
        Generate a unique key for the interface based on the node indexes.
        The order of nodes does not affect the key.
        
        Returns:
        - tuple of sorted tuples, each containing two ints representing a pair of token indexes.
        """
        # Sort node indexes and convert to tuple of tuples
        sorted_indexes = tuple(sorted(tuple(node['index']) for node in self.nodes))
        return sorted_indexes


def dump_interface_prob(results, batch_i, output_dir, base_name):
    """Dump the interface probability to a pickle file.
        
        Args:
            results: dict, the results from the inference.
            batch_i: int, the index of the batch.
            output_dir: str, the directory to save the pickle file.
            base_name: str, the base name of the pickle file.
    """

    if 'intermediate_interface_infos' not in results['interface_head'] \
        or len(results['interface_head']['intermediate_interface_infos']['pred_interface_prob_in_steps']) <= batch_i:
        return

    pred_interface_prob_repeats = results['interface_head']['intermediate_interface_infos']['pred_interface_prob_in_steps'][batch_i]
    if len(pred_interface_prob_repeats) == 0:
        return

    repeat_i = 0
    pred_interface_prob_step0 = pred_interface_prob_repeats[repeat_i][0] 
    print(f"[S1 Interface] pred_interface_prob_step0 shape: {pred_interface_prob_step0.shape}")

    # save pred_interface_prob_step0 to pkl file
    file_path = f'{output_dir}/{base_name}-interface-prob-step0.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(pred_interface_prob_step0, f)
    print(f"[S1 Interface] save interface prob to {file_path}")


def get_interface_sample_constraint_mask(interface_mask: np.ndarray, 
                                         chain_ids: np.ndarray,
                                         s1_sample_constraint: List[Dict[str, str]]) -> np.ndarray:
    """Get the interface sample constraint mask.

        Args:
            interface_mask: numpy.ndarray, (n_token, n_token)
                the interface mask indicating the interface nodes.
            chain_ids: numpy.ndarray, (n_token, )
                the chain ids of the protein complex conformation.
            s1_sample_constraint: list[dict], the sample constraint with the following format:
                [
                    {
                        'left_entity': str, the left entity id.  such as '1-1'
                        'right_entity': str, the right entity id.  such as '2-1'
                    },
                    ...
                ]
        Returns:
            interface_sample_constraint_mask: numpy.ndarray, (n_token, n_token)
                the interface sample constraint mask.
    """
    assert s1_sample_constraint is not None, \
        "[get_interface_sample_constraint_mask] s1_sample_constraint is None"
    assert interface_mask.shape[0] == interface_mask.shape[1], \
        "[get_interface_sample_constraint_mask] interface_mask is not square."
    assert chain_ids.shape[0] == interface_mask.shape[0] and chain_ids.shape[0] == interface_mask.shape[1], \
        "[get_interface_sample_constraint_mask] chain_ids shape is not consistent with interface_mask."
    
    if len(s1_sample_constraint) == 0:
        print(f"[S1 Interface]: No sample constraint found. Allow all interface.")
        return np.ones_like(interface_mask, dtype=np.float32)

    print(f"[S1 Interface]: Get interface sample constraint mask. {s1_sample_constraint}")

    constraint_mask = np.zeros_like(interface_mask, dtype=np.float32)
    for constraint in s1_sample_constraint:
        left_entity_id = constraint['left_entity']
        right_entity_id = constraint['right_entity']
        assert left_entity_id != right_entity_id, \
            f"[get_interface_sample_constraint_mask] left_entity_id {left_entity_id} " \
            f"should not be equal to right_entity_id {right_entity_id}"
        left_mask = chain_ids == left_entity_id
        right_mask = chain_ids == right_entity_id
        assert np.sum(left_mask) > 0 and np.sum(right_mask) > 0, \
            f"[get_interface_sample_constraint_mask] left_entity_id {left_entity_id} " \
            f"or right_entity_id {right_entity_id} not found in chain_ids."
        
        # Set mask for left->right and right->left (symmetric)
        constraint_mask[np.ix_(left_mask, right_mask)] = 1
        constraint_mask[np.ix_(right_mask, left_mask)] = 1

    return constraint_mask


def get_interface_prob_from_module1(interface_dir: str) -> np.ndarray:
    """Get the interface probability from the interface directory.
        
        Args:
            interface_dir: str, the directory to save the interface probability.
        Returns:
            interface_prob: numpy.ndarray, (n_token, n_token) 
                the interface probability generated by the interface prediction model (module 1).
        Raises:
            ValueError: if the interface probability file is not found.
    """

    if not os.path.exists(interface_dir):
        raise ValueError(f"interface_dir {interface_dir} not exists.")

    interface_prob_pkl_path = glob.glob(os.path.join(interface_dir, 'job-*', '*interface-prob-step0.pkl'))
    if len(interface_prob_pkl_path) == 0:
        raise ValueError(f"Not found interface prob file in {interface_dir}.")
    interface_prob_pkl_path = interface_prob_pkl_path[0]
    
    print(f'[S1 Interface]: loading interface prob from {interface_prob_pkl_path}')
    with open(interface_prob_pkl_path, 'rb') as pkl:
        interface_prob = pickle.load(pkl)

    return interface_prob


def get_interface_mask(feat, label, interface_type) -> np.ndarray:
    """Get the interface mask for the given interface type.

        Args:
            feat: dict, the feature dictionary.
            label: dict, the label dictionary.
            interface_type: str, the type of interface.
        Returns:
            interface_mask: numpy.ndarray, the interface mask.
    """
    from helixfold.data.utils import build_possible_interface_mask

    if interface_type == 'epitope_paratope':
        raise NotImplementedError('AbAg, epitope_paratope interface is not implemented')
    else: 
        interface_mask = build_possible_interface_mask(
            feat['asym_id'], 
            label['all_centra_token_indice_mask'], 
            binder_mask=None, 
            target_mask=None)

    return interface_mask


def get_interface_info(feat, label, interface_type, interface_mask) -> dict:
    """Get the interface information for the given interface type.
       
        Args:
            feat: dict, the feature dictionary.
            label: dict, the label dictionary.
            interface_type: str, the type of interface.
            interface_mask: numpy.ndarray, the interface mask.
        Returns:
            interface_info_dict: dict, the interface information.
        Raises:
            ValueError: if the interface type is not supported.
    """
    from helixfold.data.utils import build_interface_info_with_random_residues_module1

    if GEN_INTERFACE_CONFIG['interface_gen']['seed'] is not None:
        interface_gen_seed = GEN_INTERFACE_CONFIG['interface_gen']['seed']
    else:
        interface_gen_seed = random.randint(1, 10000)

    interface_gen_temperature = GEN_INTERFACE_CONFIG['interface_gen'].get('temperature', 1.0)
    interface_sampling_method = GEN_INTERFACE_CONFIG['interface_gen'].get('sampling_method', 'in_repeats')
    interface_sampling_beam = GEN_INTERFACE_CONFIG['interface_gen'].get('sampling_beam', 10)

    interface_info_dict = {
        'interface_source': GEN_INTERFACE_CONFIG['interface_source'],
        'interface_gen_size': GEN_INTERFACE_CONFIG['interface_gen']['interface_size'],
        'interface_gen_repeats': GEN_INTERFACE_CONFIG['interface_gen']['gen_repeats'],
        'interface_gen_seed': interface_gen_seed,
        'interface_gen_temperature': interface_gen_temperature,
        'interface_sampling_method': interface_sampling_method,
        'interface_sampling_beam': interface_sampling_beam
    }

    if interface_type in ['any_interface', 'epitope_paratope']:
        # random select m between min_m to max_m
        m = random.randint(GEN_INTERFACE_CONFIG['interface_sample']['min_m'], 
                        GEN_INTERFACE_CONFIG['interface_sample']['max_m'])
        seed = GEN_INTERFACE_CONFIG['interface_sample'].get('seed', None)
        ret_dict = build_interface_info_with_random_residues_module1(
                feat['asym_id'], 
                feat['entity_id'], 
                atom_pos=label['all_atom_pos'], 
                atom_mask=label['all_atom_pos_mask'],
                atom_to_token_mapping=feat['ref_token2atom_idx'],
                token_pos=label['all_atom_pos'][label['all_centra_token_indice']],
                token_pos_mask=label['all_centra_token_indice_mask'],
                top_n=GEN_INTERFACE_CONFIG['interface_sample']['top_n'],
                m=m,
                interface_mask=interface_mask,
                dist_thres=GEN_INTERFACE_CONFIG['dist_thres'],
                dist_type=GEN_INTERFACE_CONFIG.get('dist_type', 'heavy_atom'),
                seed=seed)
        interface_info_dict.update(ret_dict)
    elif interface_type in ['none', None]:
        n_token = len(feat['asym_id'])
        interface_info_dict.update({
            "interface_info_sample": np.zeros([n_token, n_token], dtype=np.float32),
            "interface_info_full": np.zeros([n_token, n_token], dtype=np.float32),
            "interface_info_mix": np.zeros([n_token, n_token], dtype=np.float32),
        })
    else:
        raise ValueError(interface_type)

    return interface_info_dict

   

# Example usage
if __name__ == "__main__":
    # Initialization
    n_token = 10
    indexes = [[0, 1], [4, 2], [2, 3]]
    match_annotations = [1, 0, -1]
    probability = 0.85

    interface_info = InterfaceInfo(n_token, indexes, match_annotations, probability)
    
    # Add a new node
    interface_info.add_node([3, 4], 1)
    
    # Print interface map
    print("Interface Map:\n", interface_info.interface_map())
    
    # Print JSON representation
    print("JSON String:\n", interface_info.to_json())

    # Get interface key
    interface_key = interface_info.get_interface_key()
    print("Interface Key:", interface_key)