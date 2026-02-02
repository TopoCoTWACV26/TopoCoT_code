#!/usr/bin/env bash
set -x
export PYTHONPATH="./":$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # Commented out to allow using all available GPUs
export TORCH_DISTRIBUTED_DEBUG=DETAIL

GPUS=$1
PORT=${PORT:-3344}

timestamp=`date +"%y%m%d.%H%M%S"`
WORK_DIR="work_dirs/stream"

####自动
WORK_DIR1="$WORK_DIR/stage1"

CONFIG=projects/configs/topocot_olv2_stage1.py



python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/streamtrain.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR1} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log



