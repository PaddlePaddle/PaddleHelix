## SIGN-Paddle

Source code for KDD 2021 paper: "Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity".

### Dependencies

- python >= 3.8
- paddlepaddle >= 2.1.0
- pgl >= 2.1.4
- openbabel == 3.1.1 (optional, only for preprocessing)

### Datasets
The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).
The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org).
You may need to use the [UCSF Chimera tool](https://www.cgl.ucsf.edu/chimera/) to convert the PDB-format files into MOL2-format files for feature extraction at first.

Alternatively, we also provided a [dropbox link](https://www.dropbox.com/sh/2uih3c6fq37qfli/AAD-LHXSWMLAuGWzcQLk5WI3a) for downloading PDBbind and CSAR-HiQ datasets.

The downloaded dataset should be preprocessed to obtain features and spatial coordinates:
```
python preprocess_pdbbind.py --data_path_core YOUR_DATASET_PATH --data_path_refined YOUR_DATASET_PATH --dataset_name pdbbind2016 --output_path YOUR_OUTPUT_PATH --cutoff 5
```
The parameter cutoff is the threshold of cutoff distance between atoms.

You can also use the processed data from [this link](https://www.dropbox.com/sh/68vc7j5cvqo4p39/AAB_96TpzJWXw6N0zxHdsppEa). Before training the model, please put the downloaded files into the directory (./data/).

### How to run
To train the model, you can run this command:
```
python train.py --cuda YOUR_DEVICE --model_dir MODEL_PATH_TO_SAVE --dataset pdbbind2016 --cut_dist 5 --num_angle 6
```
### Citation
If you find our work is helpful in your research, please consider citing our paper:
```bibtex
@inproceedings{li2021structure,
  title={Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity},
  author={Li, Shuangli and Zhou, Jingbo and Xu, Tong and Huang, Liang and Wang, Fan and Xiong, Haoyi and Huang, Weili and Dou, Dejing and Xiong, Hui},
  booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```