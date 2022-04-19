# 曙光超算平台部署

本文档参考自：https://zhuanlan.zhihu.com/p/366143771

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
conda create -p ~/conda-envs/paddle python==3.7.10
```

激活Python环境

```
conda activate ~/conda-envs/paddle
```


### 安装PaddlePaddle

在激活的Conda Python环境中安装PaddlePaddle ROCm版本

```
pip install --pre paddlepaddle-rocm -f https://www.paddlepaddle.org.cn/whl/rocm/develop.html
```

详细请参考PaddlePaddle官网按照教程(选择ROCm版本)：
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/guides/09_hardware_support/rocm_docs/paddle_install_cn.html

飞桨框架ROCm版安装说明:
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/09_hardware_support/rocm_docs/paddle_install_cn.html#anchor-0

# 3、PaddleFold

## 安装PaddleFold环境

## 配置脚本

- 任务配置
  
在 dcu_train.sh 脚本中配置任务使用的资源量


- 任务脚本

在 train_dcu.sh 脚本中加载PaddlePaddle Python环境

例如：
```
module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0
source activate ~/conda-envs/paddle
```

## 提交训练任务
```
sbatch dcu_train.sh
```

## 查看训练日志

日志在log/目录下
