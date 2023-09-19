# CamE - Paddle

This is the code of the paper **Multimodal Biological Knowledge Graph Completion via Triple Co-attention Mechanism** for ICDE 2023.


## Dependencies
- python=3.9.12
- paddlepaddle-gpu=2.5.1.post116
- h5py=3.9.0

## Datasets
The triples of DRKG come from [DRKG](https://github.com/gnn4dr/DRKG)

The KG and multimodal data (raw data and embedding) can be downloaded in the [Dropbox](https://www.dropbox.com/scl/fi/ahkro7h6l67o7a69hj07u/CamE-dropbox.zip?rlkey=2p869ew72lbosdf7imsve9e4h&dl=0)

## Training 
```
mkdir paddle_saved data
```
Put dataset `./drkg` in  `./data/` and run

```
python main.py --data data/drkg/drkg_all \
--strategy one_to_n \
--gpu 1 \
--lr 0.005 \
--neg_num 1000  \
--num_filt 128 \
--ker_sz 9 \
--threshold -0.5 \
--num_head 2 \
--interval 5. \
--batch 300 \
--test_batch 300 \
```

## Citation
If you find our work is helpful in your research, please consider citing our paper:

```
@inproceedings{xu2023multimodal,
  title={Multimodal Biological Knowledge Graph Completion via Triple Co-attention Mechanism},
  author={Xu, Derong and Zhou, Jingbo and Xu, Tong and Xia, Yuan and Liu, Ji and Chen, Enhong and Dou, Dejing},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={3928--3941},
  year={2023},
  organization={IEEE}
}
```
