# [HelixFold3](./Report_HelixFold3.pdf): An Implementation of [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) using [PaddlePaddle](https://github.com/paddlepaddle/paddle)

The AlphaFold series has transformed protein structure prediction with remarkable accuracy, often matching experimental methods. While [AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) are open-sourced, facilitating rapid and reliable predictions, [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) remains partially accessible through a limited [online server](https://alphafoldserver.com/about) and has not been open-sourced, restricting further development.
To address these challenges, the PaddleHelix team is developing [HelixFold3](./Report_HelixFold3.pdf), aiming to replicate AlphaFold3’s capabilities. Leveraging insights from previous models and extensive datasets, HelixFold3 achieves accuracy comparable to AlphaFold3 in predicting the structure of conventional ligands, nucleic acids, and proteins.

<p align="center">
<img src="images/ligands_posebusters_v1.png" align="left" height="60%" width="25.4%" />
<img src="images/NA_casp15.png" align="middle" height="60%" width="54.2%" />
<img src="images/proteins_heter_v2_success_rate.png" align="right" height="60%" width="20.4%" />
</p>


## HelixFold3 Inference

### Environment

To reproduce the results reported in this repo, specific environment settings are required as below.

- python: 3.7
- cuda: 12.0
- cudnn: 8.4.0
- nccl: 2.14.3

### Installation

HelixFold3 depends on [PaddlePaddle](https://github.com/paddlepaddle/paddle).
Python dependencies available through `pip` is provided in `requirements.txt`. 
HelixFold3 also depends on `openmm==7.5.1` and `pdbfixer`, which are only available via `conda`. 
For producing multiple sequence alignments, `kalign`, the [HH-suite](https://github.com/soedinglab/hh-suite) and `jackhmmer` are also needed. The download scripts require `aria2c`.

We provide a script `setup_env.sh` that setup a `conda` environment and installs all dependencies. The name of the environment and CUDA version can be modified in `setup_env.sh`. Locate to the directory of `helixfold` and run:

```bash
sh setup_env
# activate the conda environment
conda activate helixfold
# install paddlepaddle
wget https://paddle-wheel.bj.bcebos.com/2.5.1/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.5.1.post117-cp37-cp37m-linux_x86_64.whl
pip install paddlepaddle_gpu-2.5.1.post117-cp37-cp37m-linux_x86_64.whl
```
Note: If you have a different version of python3 and cuda, please refer to [here](https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html) for the compatible PaddlePaddle `dev` package.

### Usage

In order to run HelixFold3, the genetic databases and model parameters are required.

The parameters of HelixFold3 can be downloaded [here](FIXME), please place the downloaded checkpoint in ```./init_models/ ```directory.

The script `scripts/download_all_data.sh` can be used to download and set up all genetic databases with the following configs:

*   Default:

    ```bash
    scripts/download_all_data.sh <DATA_DIR>
    ```

    will download the full databases. The total download size for the full databases is around 415 GB and the total size when unzipped is 2.2 TB.  

*   With `reduced_dbs`:

    ```bash
    scripts/download_all_data.sh <DATA_DIR> reduced_dbs
    ```

    will download a reduced version of the databases to be used with the
    `reduced_dbs` preset. The total download size for the reduced databases is around 190 GB and the total size when unzipped is around 530 GB. 

#### Running HelixFold for Inference
To run inference on a sequence or multiple sequences using HelixFold3's pretrained parameters, run e.g.:
* Inference on single GPU:
```
sh run_infer.sh
```
The descriptions of the output files are as follows.
```
#!/bin/bash

PYTHON_BIN="PATH/TO/YOUR/PYTHON"
ENV_BIN="PATH/TO/YOUR/ENV"
MAXIT_BIN="PATH/TO/MAXIT/SRC"
DATA_DIR="PATH/TO/DATA"
export PATH="$MAXIT_BIN/hin:$PATH"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" inference.py \
    --maxit_binary "$MAXIT_BIN/bin/maxit" \
    --jackhmmer_binary_path "$ENV_BIN/jackhmmer" \
	--hhblits_binary_path "$ENV_BIN/hhblits" \
	--hhsearch_binary_path "$ENV_BIN/hhsearch" \
	--kalign_binary_path "$ENV_BIN/kalign" \
	--hmmsearch_binary_path "$ENV_BIN/hmmsearch" \
	--hmmbuild_binary_path "$ENV_BIN/hmmbuild" \
    --preset='reduced_dbs' \
    --bfd_database_path "$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
    --small_bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --uniclust30_database_path "$DATA_DIR/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
    --uniprot_database_path "$DATA_DIR/uniprot/uniprot.fasta" \
    --pdb_seqres_database_path "$DATA_DIR/pdb_seqres/pdb_seqres.txt" \
    --uniref90_database_path "$DATA_DIR/uniref90/uniref90.fasta" \
    --mgnify_database_path "$DATA_DIR/mgnify/mgy_clusters_2018_12.fa" \
    --template_mmcif_dir "$DATA_DIR/pdb_mmcif/mmcif_files" \
    --obsolete_pdbs_path "$DATA_DIR/pdb_mmcif/obsolete.dat" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz" \
    --rfam_database_path "$DATA_DIR/Rfam-14.9_rep_seq.fasta" \
    --max_template_date=2020-05-14 \
    --input_json data/demo_protein_ligand.json \
    --output_dir ./output \
    --model_name allatom_demo \
    --init_model ./init_models/checkpoints.pdparams \
    --infer_times 3 \
    --precision "bf16" \
    --no_msa_templ_feats # comment it to enable MSA searching
```
* replace MAXIT_SRC by your installed maxit's root path.
* replace DATA_DIR by your downloaded data path.
* replace ENV_BIN by your conda virtual environment or any environment wherehhblits, hmmsearch and other dependencies have installed.
* --preset set reduced_dbsto use small bfd or full_dbsto use full bfd.
* --*_database_pathpath to dataset you have downloaded.
* --input_jsoninput data in the form of JSON. Input pattern in ./data/demo_*.json for your reference.
* --outputmodel output path. The output will be in a folder named with the same as your --input_json under this path.
* --model_namemodel name in helixfold/model/config.py. Different model names specify different configurations. Mirro modification to configuration can be specified in CONFIG_DIFFS without change to full configuration in CONFIG_ALLATOM.
* --infer_timeThe number of inferences executed by model for single input.
* --no_msa_templ_feats Inference without MSA and template features.
Understanding Model Output
The outputs will be in a subfolder of output_dir. They
 include the computed MSAs, predict structures, ranked, and evaluation metric. For a rank-3-infer-2 task, assume your input json named demo_data.json, the
output_dir directory will have the following structure:
```
<output_dir>/
└── demo_data/
    ├── demo_data-pred-1-1/
    │   ├── all_results.json
    │   ├── predicted_structure.pdb
    │   └── predicted_structure.cif
    ├── demo_data-pred-1-2/
    ├── demo_data-pred-1-3/
    ├── demo_data-pred-2-1/
    ├── demo_data-pred-2-2/
    ├── demo_data-pred-2-3/
    |
    ├── demo_data-rank[1-6]/
    │   ├── predicted_structure.pdb
    │   └── predicted_structure.cif
    │   ├── all_results.json    
    |
    ├── features.pkl
    └── msas/
        ├── bfd_uniclust_hits.a3m
        ├── mgnify_hits.sto
        └── uniref90_hits.sto
```
The contents of each output file are as follows:
* features.pkl – A pickle file containing the input feature NumPy arrays
used by the models to produce the structures.
* msas/ - A directory containing the files describing the various genetic
tool hits that were used to construct the input MSA.

## Copyright

HelixFold3's code and parameters are available under the (FIXME CC) License for non-commercial use only, please check the details in (FIXME ./LICENSE) brefore using HelixFold3.

## Reference

[1]  Abramson, J et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature
630, 493–500. 10.1038/s41586-024-07487-w

[2] Jumper J, Evans R, Pritzel A, et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature
577 (7792), 583–589. 10.1038/s41586-021-03819-2.

## Citation

If you use the code or data in this repos, please cite:

```bibtex
@article{
  FIXME ./Report_HelixFold3.pdf 
}

@article{wang2022helixfold,
  title={HelixFold: An Efficient Implementation of AlphaFold2 using PaddlePaddle},
  author={Wang, Guoxia and Fang, Xiaomin and Wu, Zhihua and Liu, Yiqun and Xue, Yang and Xiang, Yingfei and Yu, Dianhai and Wang, Fan and Ma, Yanjun},
  journal={arXiv preprint arXiv:2207.05477},
  year={2022}
}
```
