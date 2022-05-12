## GeomGCL-Paddle
Source code for AAAI 2022 paper: "GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction".


### Dependencies
- python >= 3.7
- paddlepaddle >= 2.3.0
- pgl >= 2.2.3
- rdkit == 2020.03.3.0 (optional, only for preprocessing)

### Datasets
The molecule dataset can be downloaded [here](https://moleculenet.org/datasets-1).

### How to run
To pre-train the model based on geometric contrastive learning, you can run this command:
```
python train_gcl.py --cuda YOUR_DEVICE --dataset DATASET_NAME --num_dist 4 --cut_dist 5 --model_dir MODEL_PATH_TO_SAVE
```
To finetune the model:
```
python train_finetune.py --cuda YOUR_DEVICE --dataset DATASET_NAME --model_dir MODEL_PATH_TO_LOAD --task_dim TASK_NUMBER --num_dist 2 --num_angle 4 --cut_dist 4 --output_dir OUTPATH_PATH_TO_SAVE
```

### Citation
If you find our work is helpful in your research, please consider citing our paper:
```bibtex
@article{li2021geomgcl,
  title={GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction},
  author={Li, Shuangli and Zhou, Jingbo and Xu, Tong and Dou, Dejing and Xiong, Hui},
  journal={arXiv preprint arXiv:2109.11730},
  year={2021}
}
```
If you have any question, please contact Shuangli Li by email: lsl1997@mail.ustc.edu.cn.