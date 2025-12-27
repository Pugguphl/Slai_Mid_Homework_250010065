#!/bin/bash

CONFIG_FILE="experiments/configs/rnn_base.yaml"
NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29501}

torchrun --nproc-per-node="$NUM_GPUS" --master_port="$MASTER_PORT" src/rnn/train.py --config "$CONFIG_FILE"
