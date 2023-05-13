#!/bin/bash

TASKS=(
#dmc_humanoid_run
#dmc_walker_run
#dmc_cheetah_run
dmc_cheetah_run_sparse
dmc_walker_run_sparse
)
SEEDS=(0 1 2 3)
GPUS=(1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

i=0
for TASK in "${TASKS[@]}"; do
for SEED in "${SEEDS[@]}"; do
  i=$((i + 1))
  gpu=${GPUS[$((i % NUM_GPUS))]}

  logdir=logs/$TASK/mwm/$SEED
  mkdir -p $logdir

  CUDA_VISIBLE_DEVICES=$gpu XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda TF_XLA_FLAGS=--tf_xla_auto_jit=2 python mwm/train.py --logdir $logdir --configs dmc_vision --task $TASK --steps 252000 --mae.reward_pred True --mae.early_conv True --seed $SEED &> $logdir/log.txt &
done
done
