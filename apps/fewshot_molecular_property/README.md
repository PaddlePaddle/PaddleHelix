# PAR-NeurIPS21: PaddlePaddle Codes

<p align="center"><img src="PAR-thumbnail.png" alt="logo" width="600px" />

This is the PaddlePaddle implementation of ["Property-Aware Relation Networks (PAR) for Few-Shot Molecular Property Prediction"](https://papers.nips.cc/paper/2021/hash/91bc333f6967019ac47b49ca0f2fa757-Abstract.html) published in *NeurIPS 2021* as a *Spotlight* paper. 

Please cite our paper if you find it helpful. Thanks. 
```
@InProceedings{wang2021property,
  title={Property-Aware Relation Networks for Few-Shot Molecular Property Prediction},
  author={Wang, Yaqing and Abuduweili, Abulikemu and Yao, Quanming and Dou, Dejing},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2021},
}
```

## Environment  

We used the following packages for core development. We tested on `paddlepaddle 2.0.2`.

```
- paddlepaddle 2.0.2
- pgl 2.1.5
- paddlehelix 1.0.1
```

## Datasets 

Four datasets including Tox21, SIDER, MUV and ToxCast are provided. They are downloaded from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip).

## Experiments

To run the experiments, use the command (please check and tune the hyper-parameters in [parser.py](parser.py)):

```
python main.py
```

You can quickly check out how PAR operates on the Tox21 dataset by using the command:

```
bash script_train.sh
```

Using the folllowing scripts on different datasets and methods:

 ```
DATA=tox21 # training dataset
TDATA=tox21  # testing  dataset
setting=par
NS=10 # n-shot
NQ=16 # n-query
pretrain=1 # pretrain or training from scratch
seed=0 # random seed

nohup python -u main.py --epochs 1000 --eval_steps 10 --pretrained $pretrain \
--setting $setting --n-shot-train $NS  --n-shot-test $NS --n-query $NQ --dataset $DATA --test-dataset $TDATA --seed $seed \
> nohup_${DATA}${TDATA}-${setting}_s${NS}q${NQ} 2>&1 &
 ```

If you want to run other datasets, you only need to change `DATA` and `TDATA` in the above or in the [script_train.sh](script_train.sh). 
Please change `NS` and `NQ` for different n-shot and n-query settings.


## Contact
We welcome advices and feedbacks for PAR. Please feel free to open an issue or contact [Yaqing Wang](mailto:wangyaqing01@baidu.com).