
## 1、曙光超算环境概述
操作系统：Linux version 3.10.0-957.el7.x86_64

CPU：海光CPU

GPU：海光DCU

调度器：Slurm

软件栈：ROCm框架


## 2、环境配置

### 配置PaddlePaddle Python的Conda环境
```
module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0
```

创建Python环境

```
conda create -p ~/conda-envs/paddle_20220413 python==3.7.10
```

激活Python环境

```
conda activate ~/conda-envs/paddle_20220413
```


### 安装PaddlePaddle

在激活的Conda Python环境中安装PaddlePaddle ROCm版本

```
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/PaddleFold/paddlepaddle_rocm-0.0.0-cp37-cp37m-linux_x86_64-0.whl
pip install ./paddlepaddle_rocm-0.0.0-cp37-cp37m-linux_x86_64-0.whl
```


## 3、PaddleFold

### 相关工具下载

下载PaddleFold需要的两个工具：lddt和tm-score

```
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/PaddleFold/lddt
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/PaddleFold/tm_score
mkdir tools && mv lddt tm_score tools
```

### 配置脚本

- 任务配置
  
在 dcu_train.sh 脚本中配置任务使用的资源量


- 任务脚本

在 train_dcu.sh 脚本中加载PaddlePaddle Python环境

例如：
```
module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0
source activate ~/conda-envs/paddle_20220413
```

### 提交训练任务
```
sbatch dcu_train.sh
```

### 查看训练日志

日志在log/目录下
