# ProteinSIGN

Adaption of the method on protein-binding affinity prediction proposed by Li <i>et al.</i>[[1]](#1) for protein function prediction. 

## Implementation Setup
* Python==3.7
* [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)==2.2.1
* [Pgl](https://pgl.readthedocs.io/en/stable/quick_start/instruction.html)==2.2.2
* scikit-learn==1.0.1
* tqdm==4.62.3 

## Dataset
The [Protein Data Bank (PDB)](https://www.rcsb.org/). Pre-processing and transformation of proteins into graphs can be found [here](../datasets_preprocess/PDB/). After preprocessing the data should be copied in the [./data](./data) folder. Dataset splits (i.e., test, validation, and test) as proposed by [[2]](#2) can be downloaded [here](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/protein_functions/data.zip) or from their [repository](https://github.com/flatironinstitute/DeepFRI/tree/master/preprocessing/data). They should also be copied to the folder [./data](./data) after extraction.


## Training
```
python train.py [params]   
```
Where <i>params</i> are keyword arguments. See [train.py](./train.py) for the list of arguments (with their default values).   

## Testing
```
python test.py --model_name <path-to-saved-model> --label_data_path <path-to-protein-with-their-labels> [more params]  
```
<i>model_name</i> and <i>label_data_path</i> are required arguments. More (optional) parameters can be added as well. See [test.py](./test.py) for a full list of expected arguments.  


## References
> <a id="1">[1]</a> 
Shuangli Li, Jingbo Zhou, Tong Xu, <i> et al.</i> [Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity](https://doi.org/10.1145/3447548.3467311). In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD '21). Association for Computing Machinery, New York, NY, USA, 975–985.  

> <a id="2">[2]</a> 
Gligorijević, V., Renfrew, P.D., Kosciolek, T. et al. [Structure-based protein function prediction using graph convolutional networks](https://doi.org/10.1038/s41467-021-23303-9). Nat Commun 12, 3168 (2021).
