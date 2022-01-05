DATA=tox21
TDATA=tox21
setting=par
NS=10
NQ=16
pretrain=1
gpu=2
seed=0


CUDA_VISIBLE_DEVICES=$gpu nohup python -u main.py --epochs 1000 --eval_steps 10 --pretrained $pretrain \
--setting $setting --n-shot-train $NS  --n-shot-test $NS --n-query $NQ --dataset $DATA --test-dataset $TDATA --seed $seed \
> nohup_${DATA}${TDATA}-${setting}_s${NS}q${NQ} 2>&1 &
