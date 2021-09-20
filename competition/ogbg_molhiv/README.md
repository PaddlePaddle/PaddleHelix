# Graph Property Prediction for Open Graph Benchmark (OGB) Molhiv Dataset

[The Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) is a collection of benchmark datasets, data loaders, and evaluators for graph machine learning. Here we complete the Graph Property Prediction task on molhiv dataset. Details can be found in our [paper](./Molecule_Representation_Learning_by_Leveraging_Chemical_Information.pdf).


## Results on ogbg-molhiv
Here, we demonstrate the following performance on the ogbg-molhiv dataset from Stanford Open Graph Benchmark (1.2.5)

| Model              |Test ROC-AUC    |Validation ROC-AUC  | Parameters    | Hardware |
| ------------------ |-------------------   | ----------------- | -------------- |----------|
|  Neural FingerPrints     | 0.8232 ± 0.0047 | 0.8331 ± 0.0054 | 2425102  | Tesla V100 (32GB) |


## Reproducing results
### Requirements

1. Create `conda` environment and install `rdkit` >= 2018.09.1 by:

    ```bash
    conda create -n ogbg_hiv python=3.6 --yes
    conda activate ogbg_hiv
    conda install -c conda-forge rdkit --yes
    ```

2. Install GPU or CPU version of `paddlepaddle` == 1.8.4:  - visit: https://www.paddlepaddle.org.cn/ for more options

    ```bash
    pip install paddlepaddle==1.8.4             # cpu or
    pip install paddlpaddle-gpu==1.8.4.post97   # gpu: cuda_10.2
    ```

3. Install `pgl` == 1.2.1:

    ```bash
    pip install pgl==1.2.1
    ```

4. Install `ogb` for evaluation:

    ```bash
    pip install ogb
    ```

### Training the model
To simply reproduce the results demonstrated above, run the following commands: 

```bash
python extract_fingerprint.py --dataset_name ogbg-molhiv

CUDA_VISIBLE_DEVICES=0 python main.py --config hiv_config.yaml
```
The learned model parameters will be saved in `./outputs/task_name/`, where the `task_name` is specified in `hiv_config.yaml`.

Then you can predict the learned morgan fingerprint vectors by running the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config hiv_config.yaml --infer_model ./outputs/task_name/model_name
```

The predicted morgan fingerprint will be saved in `./dataset/ogbg-molhiv/soft_mgf_feat.npy`.


To classify the property of moleculars by using the random forest classifier, run the following commands:

```bash
python random_forest.py --dataset_name ogbg-molhiv
```

### Detailed hyperparameters
All the hyperparameters can be found in the `hiv_config.yaml` file. 

