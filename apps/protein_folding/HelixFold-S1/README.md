
## HelixFold-S1 Inference

### 🛠 Environment
Specific environment settings are required to reproduce the results reported in this repo,

* Python: 3.10
* CUDA: 12.0
* CuDNN: 8.4.0
* NCCL: 2.14.3
* Paddle: 3.1.0

Those settings are recommended as they are the same as we used in our A100 machines for all inference experiments. 

### 📦 Installation

HelixFold-S1 depends on [PaddlePaddle](https://github.com/paddlepaddle/paddle). Python dependencies available through `pip` 
is provided in `requirements.txt`. `kalign`, the [`HH-suite`](https://github.com/soedinglab/hh-suite) and `jackhmmer` are 
also needed to produce multiple sequence alignments. The download scripts require `aria2c`. 

```bash
conda create -n helixfold -c conda-forge python=3.10
conda activate helixfold

python3 -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
conda install -c bioconda aria2 hmmer==3.3.2 kalign2==2.04 hhsuite==3.3.0 -y
conda install -c bioconda anarci -y
conda install numpy=1.23.5 pandas=1.5.3 -y
python3 -m pip install -r requirements.txt
```

### 🎯 Usage

In order to run HelixFold-S1, the genetic databases and model parameters are required.

The parameters of HelixFold-S1 can be downloaded [here](https://paddlehelix.bd.bcebos.com/HelixFold-S1/helixfold_s1_params.tar.gz).
please place the downloaded checkpoint in ```./init_models/ ```directory.

The script `scripts/download_all_data.sh` can be used to download and set up all genetic databases with the following configs:

*   With `reduced_dbs`:

    ```bash
    scripts/download_all_data.sh ./data reduced_dbs
    ```

    will download a reduced version of the databases to be used with the `reduced_dbs` preset. The total download 
    size for the reduced databases is around 190 GB, and the total unzipped size is around 530 GB.

*   With `full_dbs`:

    NOTE: ***Support for full_dbs is not available yet and will be introduced in a future update.***

#### 🤔 Understanding Model Input

There are some demo input under `./data/` for your test and reference. Data input is in the form of JSON containing several entities such as `protein`, `ligand`, `dna`, `rna` and `ion`. Proteins and nucleic acids inputs are their sequence.

HelixFold-S1 supports input ligand as SMILES or CCD id, please refer to `./data/demo_6zcy_smiles.json` for more details about SMILES input. More flexible input will come in soon.

**Becareful**, HelixFold-S1 only support a least two chain in input json.

An example of input data is as follows:
```json
{
  "job_name": "7qg2",
  "recycle": 10,
  "ensemble": 30,
  "entities": [
    {
      "type": "protein",
      "sequence": "GENKSLEVSDTRFHSFSFYELKNVTNNFDERPISVGGNKMGEGGFGVVYKGYVNNTTVAVKKLAAMVDITTEELKQQFDQEIKVMAKCQHENLVELLGFSSDGDDLCLVYVYMPNGSLLDRLSCLDGTPPLSWHMRCKIAQGAANGINFLHENHHIHRDIKSANILLDEAFTAKISDFGLARASEKFAQTVMTSRIVGTTAYMAPEALRGEITPKSDIYSFGVVLLEIITGLPAVDEHREPQLLLDIKEEIEDEEKTIEDYIDKKMNDADSTSVEAMYSVASQCLHEKKNKRPDIKKVQQLLQEMTAS",
      "count": 1,
      "modification": [
        {
          "type": "sidechain_replace",
          "index": 193,
          "R_smiles": "CCOP(=O)(O)O",
          "R_connect_idx": 1
        }
      ]
    },
    {
      "type": "protein",
      "sequence": "GENKSLEVSDTRFHSFSF",
      "count": 1
    }
  ],
  "model_type": "HelixFold-S1"
}
```

**Input JSON fields:**
- `job_name` - A unique identifier for the inference job (required)
- `recycle` - Number of recycles during inference (default: 10)
- `ensemble` - Number of ensemble predictions (default: 30)
- `entities` - Array of molecular entities including:
  - `type` - Entity type: "protein", "ligand", "dna", "rna", or "ion"
  - `sequence` - Sequence for polymers (protein, dna, rna)
  - `count` - Number of copies
  - `ccd` - Chemical Component Dictionary ID for ligands
  - `smiles` - SMILES string for ligands (alternative to ccd)
  - `modification` - Optional modifications for polymers
- `model_type` - Model specification (default: "HelixFold-S1")
---  

The **`modification`** field is an optional parameter that specifies modified residues in a polymer sequence (protein, DNA, or RNA). It includes the following attributes:  

- **`index`** – The 1-based position of the residue to be modified.  
- **`ccd`** – The Chemical Component Dictionary (CCD) code of the modified residue. *(Currently, only modifications defined in the CCD database are supported.)*  
- **`type`** – The modification type. At present, only **`"residue_replace"`** is supported, but additional types will be introduced in future updates.  

Here is an example modification input:
```json
{
    "entities": [
        {
            "type": "dna",
            "sequence": "CCATTATAGC",
            "count": 1,
            "modification": [
                {"type": "residue_replace", "ccd": "5CM", "index": 2},
                {"type": "residue_replace", "ccd": "5CM", "index": 5}
            ]
        },
        {
            "type": "dna",
            "sequence": "GCTATAATGG",
            "count": 1
        }
    ]
}
```

#### 🚀 Running HelixFold-S1 for Inference
To run inference on a sequence or multiple sequences using HelixFold-S1's pretrained parameters, run e.g.:
* Inference on single GPU (change the settings in script BEFORE you run it)
```
sh run_inference.sh
```

The script is as follows,
```bash
#!/bin/bash

# ============== User Configuration ==============
inputjson=${1}
outpath=${2}
mkdir -p ${outpath}

# Python and Environment Configuration
PYTHON_BIN="PATH/TO/YOUR/PYTHON"
export LD_LIBRARY_PATH="PATH/TO/YOUR/ENV/lib:$LD_LIBRARY_PATH"
ENV_BIN="PATH/TO/YOUR/ENV/bin"

# Database Paths
DATA_DIR="PATH/TO/DATA"

# Model Configuration
CKPT_DIR="./init_models"

model_name_gen1="allatom_interface_gen_fixrepr_save_intermediate_interface"
init_model_gen1="${CKPT_DIR}/interface_gen100_merge-at-conditioning_fixrepr_atom5A_compress/step_604000.pdparams"

model_name_infer2="allatom_interface"
init_model_infer2="${CKPT_DIR}/interface004_any_interface0.7_lr2e-4_atom5A_compress/step_496000.pdparams"

# CUDA Device
export CUDA_VISIBLE_DEVICES=0

# ============== Run MSA Module ==============
echo ">>> Running MSA module"
${PYTHON_BIN} run_msa_module.py \
    --jackhmmer_binary_path "$ENV_BIN/jackhmmer" \
    --hhblits_binary_path "$ENV_BIN/hhblits" \
    --hhsearch_binary_path "$ENV_BIN/hhsearch" \
    --kalign_binary_path "$ENV_BIN/kalign" \
    --hmmsearch_binary_path "$ENV_BIN/hmmsearch" \
    --hmmbuild_binary_path "$ENV_BIN/hmmbuild" \
    --nhmmer_binary_path "$ENV_BIN/nhmmer" \
    --preset='reduced_dbs' \
    --small_bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --uniprot_database_path "$DATA_DIR/uniprot/uniprot.fasta" \
    --pdb_seqres_database_path "$DATA_DIR/pdb_seqres.txt" \
    --uniref90_database_path "$DATA_DIR/uniref90/uniref90.fasta" \
    --mgnify_database_path "$DATA_DIR/mgnify/mgy_clusters_2018_12.fa" \
    --template_mmcif_dir "$DATA_DIR/pdb_mmcif/mmcif_files" \
    --obsolete_pdbs_path "$DATA_DIR/pdb_mmcif/obsolete.dat" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz" \
    --rfam_database_path "$DATA_DIR/rna_msa_db/Rfam-14.9_rep_seq.fasta" \
    --species_identifer_map_path "$DATA_DIR/rna_msa_db/taxonomy_AND_taxonomies_with_1_uniprot_2024_07_14.tsv.gz" \
    --max_template_date=2021-09-30 \
    --json_path $inputjson \
    --output_dir ${outpath}

# ============== Run Gen Module 1 ==============
echo ">>> Running gen module 1"
${PYTHON_BIN} run_inference_module.py \
    --json_path $inputjson \
    --output_dir ${outpath} \
    --model_names "${model_name_gen1}" \
    --init_model "${init_model_gen1}" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz"

# ============== Run Inference Module 2 ==============
echo ">>> Running inference module 2"
${PYTHON_BIN} run_inference_S1_module2.py \
    --json_path $inputjson \
    --output_dir ${outpath} \
    --model_names "${model_name_infer2}" \
    --init_model "${init_model_infer2}" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz"

echo ">>> Inference completed! Output: ${outpath}"
```
The descriptions of the above script are as follows:
- `inputjson` - Path to input JSON file (first argument to script)
- `outpath` - Output directory path (second argument to script)
- Replace `PYTHON_BIN` with your python binary where `paddlepaddle-gpu` has been installed
- Replace `ENV_BIN` with your conda virtual environment or any environment where `hhblits`, `hmmsearch` and other dependencies have been installed
- Replace `DATA_DIR` with your downloaded data path
- `CKPT_DIR` - Directory containing model checkpoints (default: `./init_models/`)
- `CUDA_VISIBLE_DEVICES` - GPU device ID to use
- `--preset` - Set `'reduced_dbs'` to use small bfd or `'full_dbs'` to use full bfd
- `--*_database_path` - Path to datasets you have downloaded
- `--json_path` - Input data in the form of JSON. Input examples in `./data/*.json` for your reference
- `--output_dir` - Model output directory. Results will be written directly to this path
- `--model_names` - Model name in `./helixfold/model/config.py`. Different model names specify different configurations

### 🤔 Understanding Model Output

The outputs will be in the specified `output_dir`, including the computed MSAs, intermediate features,
predicted structures, and evaluation metrics. Assume your input JSON has `job_name` set to "demo",
the output directory will have the following structure:

```
<output_dir>/
├── final_features.pkl                    # Processed features from MSA module
├── user_input.json                       # Copy of the input JSON configuration
├── job_status.json                       # Job execution status
├── timings_featurization.json            # Timing information for feature processing
├── msas/                                 # Multiple sequence alignments for each entity
│   ├── protein_1-1/
│   │   ├── bfd
│   │   ├── mgnify
│   │   ├── uniref90
│   │   └── ...
│   └── protein_2-1/
│       └── ...
├── interface_infos/                      # Interface prediction information
│   ├── predicted_interface.png           # Visualization of predicted interface
│   ├── predicted_interface.json          # Interface coordinates and data
│   └── sample_infos.csv                  # Sampling probabilities per entity pair
├── module1/                              # Module 1 (gen) intermediate outputs
│   └── ...
├── module2/                              # Module 2 final ranked predictions
│   ├── job-demo-17-rank1/
│   │   ├── all_results.json              # Complete prediction metrics and results
│   │   ├── predicted_structure.cif       # Final predicted structure (mmCIF format)
│   │   ├── predicted_structure.cifABC    # Structure with chain ID info
│   │   ├── timings.json                  # Timing information for this prediction
│   │   └── chain_id_mapping.csv          # Mapping of chain IDs to entity indices
│   ├── job-demo-15-rank1/
│   ├── job-demo-28-rank2/
│   └── ...
└── previous_sampled_interface/           # Previously sampled interface conformations
    └── ...
```

The contents of each output directory/file are as follows:
- `msas/` - Multiple sequence alignment results for each entity, containing hits from various genetic databases (BFD, UniRef90, Mgnify, etc.)
- `interface_infos/` - Interface prediction results including:
  - `predicted_interface.png` - Visualization of predicted interface regions
  - `predicted_interface.json` - Interface coordinates and probability scores
  - `sample_infos.csv` - Sampling probabilities for each entity pair (e.g., protein-protein interfaces)
- `module1/` - Intermediate outputs from the generation module
- `module2/` - Final ranked predictions from the inference module:
  - `job-demo-{N}-rank{M}` - N-th sample with rank M
  - `rank1` represents the top-ranked prediction according to confidence metrics
  - Each contains the predicted structure in CIF format and comprehensive metrics in JSON
- `chain_id_mapping.csv` - Maps chain IDs in the structure to input entity indices for reference

### 📌 Resource Usage

We suggest a single GPU for inference has at least 32G available memory. The maximum number of tokens is around 
1200 for inference on a single A100-40G GPU with precision `bf16`. The length of inference input tokens on a 
single V100-32G with precision `fp32` is up to 1000. Inferring longer tokens or entities with larger atom numbers 
per token than normal protein residues like nucleic acids may cost more GPU memory.

For samples with larger tokens, you can reduce `model.global_config.subbatch_size` in `CONFIG_DIFFS` in `helixfold/model/config.py` to save more GPU memory but suffer from slower inference. `model.global_config.subbatch_size` is set as `96` by default. You can also
reduce the number of additional recycles by changing `model.num_recycle` in the same place.

**For Training:**
Model training was carried out on a cluster of 128 NVIDIA A100 GPUs and lasted for about 10 days in total.
