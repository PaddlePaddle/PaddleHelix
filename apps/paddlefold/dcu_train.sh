#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -J paddlefold_dcu
#SBATCH -p normal
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 1000:00:00
#SBATCH --gres=dcu:4
#SBATCH --no-requeue

module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1

echo $SLURM_JOB_NODELIST

allhost=''
for i in `scontrol show hostnames $SLURM_JOB_NODELIST`
do
    ip=`net lookup ${i}`
    allhost=$allhost','${ip}
done
allhost=${allhost:1}
echo $allhost

rm -rf log*
rm -rf *_train_desc*
rm -rf details*

srun train_dcu.sh ${allhost}
