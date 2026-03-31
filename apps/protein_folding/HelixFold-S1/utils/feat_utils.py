import os
from helixfold.data.pipeline_multimer import *
from helixfold.data import parsers
import pickle
import gzip
from helixfold.data import feature_processing

def get_feature_dict(prot_chains,feat_dir,protein_map_file):
    # protein, chain1, chain2 = protein_chains.split()[:3]
    # fasta_name = fasta_names[i]
    # fasta_name = f"{protein}_{chain1}{chain2}"
    # feat_dir = FLAGS.feat_dir
    id_mapping = {}
    # if os.path.exists(FLAGS.protein_map_file):
    if os.path.exists(protein_map_file):
        for line in open(protein_map_file):
            protein, protein_seqid = line.split()
            id_mapping[protein] = protein_seqid
    print(f"id_mapping_file: {protein_map_file} num mapped keys: {len(id_mapping.keys())}")

    protein_list_item = prot_chains.split()[:]
    targeted_chains_with_group_info = protein_list_item[1:]
    protein, targeted_chains = protein_list_item[0], [item.split("#")[0] for item in targeted_chains_with_group_info]
    grouped = [item.split("#")[0] for item in targeted_chains_with_group_info if item.endswith("#")]

    # fasta 
    fasta_file = os.path.join(feat_dir.replace("single_chain","fasta"), f"{protein}.fasta")
    seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
    chain_ids = [i.split()[0].split('_')[1] for i in descs]
    seqs_lens = [len(i) for i in seqs]
    chain_to_len = dict(zip(chain_ids, seqs_lens))
    chain_ids = [c for c in chain_ids if c in targeted_chains]
    chain_ids_w_group_info = [c+"#" if c in grouped else c for c in chain_ids]

    prediction_name = f"{protein}_{'_'.join(targeted_chains_with_group_info)}"
    all_chain_features = {}
    flag_missing_features = False
    for chain_id in chain_ids:
      feature_pkl_dir = protein+"_"+chain_id
      if feature_pkl_dir in id_mapping:
          feature_pkl_dir = id_mapping[feature_pkl_dir]
      chain_feat_path = os.path.join(feat_dir, feature_pkl_dir, 'features.pkl.gz')
      if not os.path.exists(chain_feat_path): chain_feat_path = os.path.join(feat_dir, feature_pkl_dir, 'features.pkl')
      if not os.path.exists(chain_feat_path):
        flag_missing_features = True
        print(f"{chain_feat_path} not exist {feature_pkl_dir} in mapping_file: {feature_pkl_dir in id_mapping}")
        break
      all_chain_features[chain_id] = pickle.load(gzip.open(chain_feat_path,"rb")) if chain_feat_path.endswith(".pkl.gz") else pickle.load(open(chain_feat_path,"rb")) 
    if flag_missing_features:
      print("features not complete, skipped")
      return None
      
    feat_dict = process_with_all_chain_features(all_chain_features)
    return prediction_name, feat_dict, chain_ids_w_group_info


def process_with_all_chain_features(
            all_chain_features: str) -> pipeline.FeatureDict:
  """convert all_chain_features to multimer features."""
  new_dict = {}

  # check if protein is homonmer with unique sequence
  input_seqs = set()
  for chain_id, chain_features in all_chain_features.items():
    input_seqs.add(str(chain_features["sequence"]))
  is_homomer_or_monomer = len(set(input_seqs)) == 1
  
  for chain_id, chain_features in all_chain_features.items():
    if is_homomer_or_monomer: 
      # delete keys with _all_seq if is_homomer_or_monomer
      key_list = list(chain_features.keys())
      for key in key_list:
        if str(key).endswith("_all_seq"): chain_features.pop(key)

    new_dict[chain_id] = convert_monomer_features(chain_features,
                                              chain_id=chain_id)
  new_dict = add_assembly_features(new_dict)

  np_example = feature_processing.pair_and_merge(
      all_chain_features=new_dict)

  # Pad MSA to avoid zero-sized extra_msa.
  np_example = pad_msa(np_example, 512)

  return np_example