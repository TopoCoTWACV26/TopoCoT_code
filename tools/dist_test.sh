#!/usr/bin/env bash
set -x
export PYTHONPATH="./":$PYTHONPATH
timestamp=`date +"%y%m%d.%H%M%S"`
# export CUDA_VISIBLE_DEVICES=0,1,2,3
WORK_DIR=work_dirs/stream/stage3
CONFIG=projects/configs/topocot_olv2_stage3.py

CHECKPOINT=${WORK_DIR}/latest.pth

GPUS=$1
PORT=${PORT:-2345}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
