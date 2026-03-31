#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AF3 featurization script."""

import os
import sys
import time
import json
import pickle
import logging
import pathlib
import argparse
import shutil
from typing import Dict

from helixfold.data import pipeline_multimer_parallel as pipeline_multimer
from helixfold.data import pipeline_parallel as pipeline
from helixfold.data import pipeline_rna_parallel as pipeline_rna
from helixfold.data import pipeline_rna_multimer
from helixfold.data import templates
from helixfold.data.tools import hmmsearch

from utils.misc import set_logging_level
from infer_scripts import feature_processing_aa, preprocess
from infer_scripts.colab_msa_to_feature import DataPipelineColabSearchProtein
from infer_scripts.validation import ref_structure_checker
from infer_scripts.validation.input_validation import validate_input_file
from infer_scripts.tools.utils import write_json


DEBUG = os.getenv("DEBUG", "0") == "1"
print(f">>> DEBUG: {DEBUG}")
logger = logging.getLogger(__file__)

MAX_TEMPLATE_HITS = 20

def get_msa_templates_pipeline(args) -> Dict:
    logger.info(f">>> use_colabfold_search: {args.use_colabfold_search}")
    use_precomputed_msas = True # FLAGS.use_precomputed_msas
    
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=args.hmmsearch_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        database_path=args.pdb_seqres_database_path)

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=args.obsolete_pdbs_path)

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        hhsearch_binary_path=args.hhsearch_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        # uniref30_database_path=args.uniref30_database_path,
        small_bfd_database_path=args.small_bfd_database_path ,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=args.use_small_bfd,
        use_precomputed_msas=use_precomputed_msas,
        use_msa_cache=args.use_msa_cache,
        use_oas_msa_cache=args.use_oas_msa_cache,
        use_online_msa_cache=args.use_online_msa_cache,
        msa_cache_dir=args.msa_cache_dir,
        split_dataset=args.split_dataset)

    prot_data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        uniprot_database_path=args.uniprot_database_path,
        use_precomputed_msas=use_precomputed_msas,
        split_dataset=args.split_dataset)
    if args.use_colabfold_search:
        template_mmcif_dir = os.path.join(args.colabfold_database_path, 'pdb', 'divided')
        pdb_seqres_database_path = os.path.join(args.colabfold_database_path, 'pdb', 'pdb_seqres.txt')
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,  #
            max_template_date=args.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=None)  #
    
        colabfold_prot_pipeline = DataPipelineColabSearchProtein(
            colabfold_database_path=args.colabfold_database_path,
            colabfold_search_binary_path=args.colabfold_search_binary_path,
            mmseqs_binary_path=args.mmseqs_binary_path,
            template_featurizer=template_featurizer, 
            pdb_seqres_database_path=pdb_seqres_database_path,
            species_identifier_path=args.species_identifer_map_path)
    else:
        colabfold_prot_pipeline=None
    rna_monomer_data_pipeline = pipeline_rna.RNADataPipeline(
      hmmer_binary_path=args.nhmmer_binary_path,
      rfam_database_path=args.rfam_database_path,
      rnacentral_database_path=args.rnacentral_database_path,
      nt_database_path=args.nt_database_path,     
      species_identifer_map_path=args.species_identifer_map_path,
      use_precomputed_msas=use_precomputed_msas)

    rna_data_pipeline = pipeline_rna_multimer.RNADataPipeline(
      monomer_data_pipeline=rna_monomer_data_pipeline)

    return {
        'protein': prot_data_pipeline,
        'rna': rna_data_pipeline,
        'colabfold_prot': colabfold_prot_pipeline,
    }


def run_features_pipeline(
        entities: list,
        json_path: str,
        output_dir_base: str,
        msa_templ_pipeline_dict: Dict[str, pipeline.DataPipeline],
        use_msa_templ_feats: bool,
        use_colabfold_search: bool,):

    timings = dict()
    timings['features'] = 0.

    output_dir_base = pathlib.Path(output_dir_base)
    msa_output_dir = output_dir_base.joinpath('msas')
    msa_output_dir.mkdir(parents=True, exist_ok=True)
    feature_dict = None
    features_pkl = output_dir_base.joinpath('final_features.pkl')
    if features_pkl.exists():
        logger.info('cached features.pkl is available. finished.')
        return
    else:
        t0 = time.time()
        feature_dict = feature_processing_aa.featurize_entities(
                    all_entities=entities,
                    json_path=json_path,
                    ccd_preprocessed_path=args.ccd_preprocessed_path,
                    msa_templ_data_pipeline_dict=msa_templ_pipeline_dict,
                    msa_output_dir=msa_output_dir,
                    use_msa_templ_feats=use_msa_templ_feats,
                    use_colabfold_search=use_colabfold_search,
                    kalign_binary_path=args.kalign_binary_path)
        timings['features'] = time.time() - t0

        with open(features_pkl, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

    ## only record the timings of preprocess.
    print(f"[FEAT] All features(include msa/template): {timings['features']}")
    logger.info('Final timings for featurization: %s', timings)
    with open(output_dir_base.joinpath('timings_featurization.json'), 'w') as f:
        f.write(json.dumps(timings, indent=4))
        
  
def main(args):
    set_logging_level("INFO")
    logger.info(f'[ARG] {args}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy input JSON file to the target location
    shutil.copy(args.json_path, os.path.join(args.output_dir, 'user_input.json'))
    
    # initialize job status
    job_status_path = os.path.join(args.output_dir, 'job_status.json')
    error_infos = {"status": "running"}
    error_infos['job_fail_reason'] = ""
    write_json(error_infos, job_status_path, cn=True)

    # 1. Validate input JSON against schema
    schema_path = os.path.join(os.path.dirname(__file__), "infer_scripts/validation/schema_hf3_input.json")
    logger.info(f'Validating input JSON against schema: {schema_path}')
    validate_input_file(args.json_path, args.output_dir, schema_path)
    
    # 2. Preprocess entities
    all_entities = preprocess.online_json_parser(args.json_path, args.output_dir)
    
    # 3. Reference structure check - this performs alignment check with kalign
    # kalign 校验速度很快，这里先做，快速校验快速失败；若成功再进入实际构建
    ref_structure_checker.ref_structure_check(all_entities, args.kalign_binary_path, args.json_path)

    # 4. run_features_pipeline
    use_small_bfd = args.preset == 'reduced_dbs'
    setattr(args, 'use_small_bfd', use_small_bfd)
    if use_small_bfd:
        assert args.small_bfd_database_path is not None
    else:
        assert args.bfd_database_path is not None
        assert args.uniclust30_database_path is not None
    
    if args.use_colabfold_search:
        assert args.colabfold_database_path is not None
        assert args.colabfold_search_binary_path is not None
        assert args.mmseqs_binary_path is not None

    t0 = time.time()
    logger.info('MSA/Template Pipelines in preparation...')
    if args.if_use_msa_templ_feats:
        msa_templ_data_pipeline_dict = get_msa_templates_pipeline(args)
    else:
        logger.debug('MSA/Template Pipelines are not used.')
        msa_templ_data_pipeline_dict = {}
    logger.info(f'MSA/Template Pipelines are ready, use: {time.time() - t0}')

    run_features_pipeline(entities=all_entities,
                        json_path=args.json_path,
                        output_dir_base=args.output_dir,
                        msa_templ_pipeline_dict=msa_templ_data_pipeline_dict,
                        use_msa_templ_feats=args.if_use_msa_templ_feats,
                        use_colabfold_search=args.use_colabfold_search,)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AF3 featurization pipeline')
    parser.add_argument('--json_path', type=str,
                        default=None, required=True,
                        help='Paths to json file, each containing '
                        'entity information including sequence, smiles or CCD, copies etc.')
    parser.add_argument('--output_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory that will store results.')


    parser.add_argument('--ccd_preprocessed_path', type=str,
                        default=None, required=True,
                        help='Path to CCD preprocessed files.')

    parser.add_argument('--use_colabfold_search', action='store_true', default=False,
                        help='Whether to run ColabFold search to get protein msa/template features.')
    parser.add_argument('--colabfold_database_path', type=str, default=None, 
                        help='Path to colabfold db.')
    parser.add_argument('--colabfold_search_binary_path', type=str,
                        default=shutil.which('colabfold_search'),
                        help='Path to colabfold_search executable.')
    parser.add_argument('--mmseqs_binary_path', type=str,
                        default=shutil.which('mmseqs'),
                        help='Path to mmseqs executable.')

    parser.add_argument('--jackhmmer_binary_path', type=str,
                        default='/usr/bin/jackhmmer',
                        help='Path to the JackHMMER executable.')
    parser.add_argument('--hhblits_binary_path', type=str,
                        default='/usr/bin/hhblits',
                        help='Path to the HHblits executable.')
    parser.add_argument('--hhsearch_binary_path', type=str,
                        default='/usr/bin/hhsearch',
                        help='Path to the HHsearch executable.')
    parser.add_argument('--kalign_binary_path', type=str,
                        default='/usr/bin/kalign',
                        help='Path to the Kalign executable.')
    parser.add_argument('--hmmsearch_binary_path', type=str,
                        default='/usr/bin/hmmsearch',
                        help='Path to the hmmsearch executable.')
    parser.add_argument('--hmmbuild_binary_path', type=str,
                        default='/usr/bin/hmmbuild',
                        help='Path to the hmmbuild executable.')

    # binary path of the tool for RNA MSA searching
    parser.add_argument('--nhmmer_binary_path', type=str,
                        default='/usr/bin/nhmmer',
                        help='Path to the nhmmer executable.')
    
    # parser.add_argument('--uniref30_database_path', type=str,
    #                     default=None, required=True,
    #                     help='Path to the Uniref30 database for use by HHblits.')
    parser.add_argument('--uniprot_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Uniprot database for use '
                        'by JackHMMER.')
    parser.add_argument('--pdb_seqres_database_path', type=str,
                        default=None, required=True,
                        help='Path to the PDB '
                        'seqres database for use by hmmsearch.')
    parser.add_argument('--uniref90_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Uniref90 database for use '
                        'by JackHMMER.')
    parser.add_argument('--mgnify_database_path', type=str,
                        default=None, required=True,
                        help='Path to the MGnify database for use by '
                        'JackHMMER.')
    parser.add_argument('--bfd_database_path', type=str, default=None,
                        help='Path to the BFD database for use by HHblits.')
    parser.add_argument('--small_bfd_database_path', type=str, default=None,
                        help='Path to the small version of BFD used '
                        'with the "reduced_dbs" preset.')
    parser.add_argument('--uniclust30_database_path', type=str, default=None,
                        help='Path to the Uniclust30 database for use '
                        'by HHblits.')
    # RNA MSA searching databases
    parser.add_argument('--rfam_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Rfam database for RNA MSA searching.')
    parser.add_argument('--rnacentral_database_path', type=str,
                        default=None, required=False,
                        help='Path to the RNACentral database for RNA MSA searching.')
    parser.add_argument('--nt_database_path', type=str,
                        default=None, required=False,
                        help='Path to the Nuclitide collections database for RNA MSA searching.')

    parser.add_argument('--template_mmcif_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory with template mmCIF '
                        'structures, each named <pdb_id>.cif')
    parser.add_argument('--max_template_date', type=str,
                        default=None, required=True,
                        help='Maximum template release date to consider. '
                        'Important if folding historical test sets.')
    parser.add_argument('--obsolete_pdbs_path', type=str,
                        default=None, required=True,
                        help='Path to file containing a mapping from '
                        'obsolete PDB IDs to the PDB IDs of their '
                        'replacements.')

    parser.add_argument('--species_identifer_map_path', type=str,
                        default=None, required=True,
                        help='Path to the species_identifer_map file for '
                        'adding species identifiers for RNA MSA')

    parser.add_argument('--preset',
                        default='full_dbs', required=False,
                        choices=['reduced_dbs', 'full_dbs', 'casp14'],
                        help='Choose preset model configuration - '
                        'no ensembling and smaller genetic database '
                        'config (reduced_dbs), no ensembling and full '
                        'genetic database config  (full_dbs) or full '
                        'genetic database config and 8 model ensemblings '
                        '(casp14).')
    parser.add_argument('--if_use_msa_templ_feats', default=True, action='store_false')

    ## msa cache options
    parser.add_argument('--use_msa_cache',
                        action='store_true', default=False)
    parser.add_argument('--use_oas_msa_cache',
                        action='store_true', default=False)
    parser.add_argument('--use_online_msa_cache',
                        action='store_true', default=False)
    parser.add_argument('--msa_cache_dir', type=str, default="/data/helixfold/msa_cache/",
                        help="MSA cache 的主路径，注意目前三级缓存都在此路径内。")
    ## dataset split options
    parser.add_argument('--split_dataset',
                        action='store_true', default=False)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        ## revised the task status
        json_status = os.path.join(args.output_dir, 'job_status.json')
        if not os.path.exists(json_status):
            status_dict = {}
            status_dict["status"] = 'failed'
            status_dict["job_fail_reason"] = str(e)
            with open(json_status, 'w') as f:
                json.dump(status_dict, f, indent=4)
        else:
            with open(json_status, 'r') as f:
                status_dict = json.load(f)
                status_dict["status"] = 'failed'
                if len(status_dict["job_fail_reason"]) > 0:
                    status_dict["job_fail_reason"] += f';{str(e)}'
                else:
                    status_dict["job_fail_reason"] = str(e)
            with open(json_status, 'w') as f:
                json.dump(status_dict, f, indent=4)
        
        sys.exit(1)
