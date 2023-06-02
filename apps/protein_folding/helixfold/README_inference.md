# HelixFold Inference

## Environment

To reproduce the results reported in this repo, specific environment settings are required as below. 

- python: 3.7
- cuda: 11.6
- cudnn: 8.4.0
- nccl: 2.14.3

## Installation

HelixFold depends on [PaddlePaddle](https://github.com/paddlepaddle/paddle).
Python dependencies available through `pip` is provided in `requirements.txt`. HelixFold also depends on `openmm==7.5.1` and `pdbfixer`, which are only available via `conda`. For producing multiple sequence alignments, `kalign`, the [HH-suite](https://github.com/soedinglab/hh-suite) and `jackhmmer` are also needed. The download scripts require `aria2c`.

We provide a script `setup_env` that setup a `conda` environment and installs all dependencies. You can change the name of the environment and CUDA version in `setup_env`. Locate to the directory of `helixfold` and run:
```bash
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
sh setup_env
conda activate helixfold # activate the conda environment
```
Note: If you have a different version of python3 and cuda, please refer to [here](https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html) for the compatible PaddlePaddle `dev` package.

In order to run scripts with DAP/BP/DP-DAP-BP mode, you also need to install `ppfleetx`. Please refer to [here](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/projects/protein_folding) for more details.
```bash
wget https://paddle-qa.bj.bcebos.com/PaddleFleetX/ppfleetx-0.0.0-py3-none-any.whl
python -m pip install ppfleetx-0.0.0-py3-none-any.whl      # install ppfleetx
```

## Usage

In order to run HelixFold, the genetic databases and model parameters are required.

You can use a script `scripts/download_all_data.sh`, which is the same as the original AlphaFold that can be used to download and set up all databases and model parameters:

*   Default:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR>
    ```

    will download the full databases. The total download size for the full databases is around 415 GB and the total size when unzipped is 2.2 TB.  

*   With `reduced_dbs`:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR> reduced_dbs
    ```

    will download a reduced version of the databases to be used with the
    `reduced_dbs` preset. The total download size for the reduced databases is around 190 GB and the total size when unzipped is around 530 GB. 

### Running HelixFold for Inference

To run inference on a sequence or multiple sequences using a set of DeepMind's pretrained parameters, run e.g.:

*   Inference on single GPU (DP):
    ```bash
    fasta_file="target.fasta"       # path to the target protein
    model_name="model_5"            # the alphafold model name
    DATA_DIR="data"                 # path to the databases
    OUTPUT_DIR="helixfold_output"   # path to save the outputs

    python run_helixfold.py \
      --fasta_paths=${fasta_file} \
      --data_dir=${DATA_DIR} \
      --bfd_database_path=${DATA_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
      --small_bfd_database_path=${DATA_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta \
      --uniclust30_database_path=${DATA_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
      --uniref90_database_path=${DATA_DIR}/uniref90/uniref90.fasta \
      --mgnify_database_path=${DATA_DIR}/mgnify/mgy_clusters_2018_12.fa \
      --pdb70_database_path=${DATA_DIR}/pdb70/pdb70 \
      --template_mmcif_dir=${DATA_DIR}/pdb_mmcif/mmcif_files \
      --obsolete_pdbs_path=${DATA_DIR}/pdb_mmcif/obsolete.dat \
      --max_template_date=2020-05-14 \
      --model_names=${model_name} \
      --output_dir=${OUTPUT_DIR} \
      --preset='reduced_dbs' \
      --jackhmmer_binary_path /opt/conda/envs/helixfold/bin/jackhmmer \
      --hhblits_binary_path /opt/conda/envs/helixfold/bin/hhblits \
      --hhsearch_binary_path /opt/conda/envs/helixfold/bin/hhsearch \
      --kalign_binary_path /opt/conda/envs/helixfold/bin/kalign \
      --random_seed=0
    ```

*   Inference on multiple GPUs (DAP):
    ```bash
    fasta_file="target.fasta"       # path to the target protein
    model_name="model_5"            # the alphafold model name
    DATA_DIR="data"                 # path to the databases
    OUTPUT_DIR="helixfold_output"   # path to save the outputs
    log_dir="demo_log"              # path to log file

    distributed_args="--run_mode=collective --log_dir=${log_dir}"
    python -m paddle.distributed.launch ${distributed_args} \
      --gpus="0,1,2,3,4,5,6,7" \
      run_helixfold.py \
      --distributed \
      --dap_degree 8 \
      --fasta_paths=${fasta_file} \
      --data_dir=${DATA_DIR} \
      --bfd_database_path=${DATA_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
      --small_bfd_database_path=${DATA_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta \
      --uniclust30_database_path=${DATA_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
      --uniref90_database_path=${DATA_DIR}/uniref90/uniref90.fasta \
      --mgnify_database_path=${DATA_DIR}/mgnify/mgy_clusters_2018_12.fa \
      --pdb70_database_path=${DATA_DIR}/pdb70/pdb70 \
      --template_mmcif_dir=${DATA_DIR}/pdb_mmcif/mmcif_files \
      --obsolete_pdbs_path=${DATA_DIR}/pdb_mmcif/obsolete.dat \
      --max_template_date=2020-05-14 \
      --model_names=${model_name} \
      --output_dir=${OUTPUT_DIR} \
      --preset='reduced_dbs' \
      --seed 2022 \
      --jackhmmer_binary_path /opt/conda/envs/helixfold/bin/jackhmmer \
      --hhblits_binary_path /opt/conda/envs/helixfold/bin/hhblits \
      --hhsearch_binary_path /opt/conda/envs/helixfold/bin/hhsearch \
      --kalign_binary_path /opt/conda/envs/helixfold/bin/kalign \
      --random_seed=0
    ```
You can use `python run_helixfold.py -h` to find the description of the arguments.

We retain the same outputs as AlphaFold. We copy the AlphaFold's descriptions here. 

The outputs will be in a subfolder of `output_dir`. They
include the computed MSAs, unrelaxed structures, relaxed structures, ranked
structures, raw model outputs, prediction metadata, and section timings. The
`output_dir` directory will have the following structure:

```
<target_name>/
    features.pkl
    ranked_{0,1,2,3,4}.pdb
    ranking_debug.json
    relaxed_model_{1,2,3,4,5}.pdb
    result_model_{1,2,3,4,5}.pkl
    timings.json
    unrelaxed_model_{1,2,3,4,5}.pdb
    msas/
        bfd_uniclust_hits.a3m
        mgnify_hits.sto
        uniref90_hits.sto
```

The contents of each output file are as follows:

*   `features.pkl` – A `pickle` file containing the input feature NumPy arrays
    used by the models to produce the structures.
*   `unrelaxed_model_*.pdb` – A PDB format text file containing the predicted
    structure, exactly as outputted by the model.
*   `relaxed_model_*.pdb` – A PDB format text file containing the predicted
    structure, after performing an Amber relaxation procedure on the unrelaxed
    structure prediction (see Jumper et al. 2021, Suppl. Methods 1.8.6 for
    details).
*   `ranked_*.pdb` – A PDB format text file containing the relaxed predicted
    structures, after reordering by model confidence. Here `ranked_0.pdb` should
    contain the prediction with the highest confidence, and `ranked_4.pdb` the
    prediction with the lowest confidence. To rank model confidence, we use
    predicted LDDT (pLDDT) scores (see Jumper et al. 2021, Suppl. Methods 1.9.6
    for details).
*   `ranking_debug.json` – A JSON format text file containing the pLDDT values
    used to perform the model ranking, and a mapping back to the original model
    names.
*   `timings.json` – A JSON format text file containing the times taken to run
    each section of the AlphaFold pipeline.
*   `msas/` - A directory containing the files describing the various genetic
    tool hits that were used to construct the input MSA.
*   `result_model_*.pkl` – A `pickle` file containing a nested dictionary of the
    various NumPy arrays directly produced by the model. In addition to the
    output of the structure module, this includes auxiliary outputs such as:

    *   Distograms (`distogram/logits` contains a NumPy array of shape [N_res,
        N_res, N_bins] and `distogram/bin_edges` contains the definition of the
        bins).
    *   Per-residue pLDDT scores (`plddt` contains a NumPy array of shape
        [N_res] with the range of possible values from `0` to `100`, where `100`
        means most confident). This can serve to identify sequence regions
        predicted with high confidence or as an overall per-target confidence
        score when averaged across residues.
    *   Present only if using pTM models: predicted TM-score (`ptm` field
        contains a scalar). As a predictor of a global superposition metric,
        this score is designed to also assess whether the model is confident in
        the overall domain packing.
    *   Present only if using pTM models: predicted pairwise aligned errors
        (`predicted_aligned_error` contains a NumPy array of shape [N_res,
        N_res] with the range of possible values from `0` to
        `max_predicted_aligned_error`, where `0` means most confident). This can
        serve for a visualisation of domain packing confidence within the
        structure.

The pLDDT confidence measure is stored in the B-factor field of the output PDB
files (although unlike a B-factor, higher pLDDT is better, so care must be taken
when using for tasks such as molecular replacement).

### Running HelixFold for CASP14 Demo

For convenience, we also provide a demo script `gpu_infer.sh` for some CASP14 proteins under folder `demo_data/casp14_demo`. To run them, you just need to execute following command:

```bash
sh gpu_infer.sh T1026
```

Note that such demo for T1026 and T1037 can work without downloading large MSA datasets, only model parameters are required.

### Running HelixFold for Ultra-Long Monomer Protein Demo

**Ultra-long monomer protein (around 6600 AA) prediction is supported now!** We provide a demo script `gpu_infer_long.sh` for predicting ultra-long monomer protein `F2ULY7` under folder `demo_data/demo_infer`. To run it, you just need to execute following command:

```bash
sh gpu_infer_long.sh
```

Note that predicting ultra-long monomer protein is only supported by model parameter `model_5` with setting `enable_low_memory` on 8 A100 80G GPUs currently.

## Copyright

HelixFold code is licensed under the Apache 2.0 License, which is same as AlphaFold. However, we use the AlphaFold parameters pretrained by DeepMind, which are made available for non-commercial use only under the terms of the CC BY-NC 4.0 license.
