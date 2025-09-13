#!/bin/sh

export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Training parameters
DATASET=wikitext-103
MODEL=opt-2.7b
EPOCHS=1
BATCH_SIZE=16
FREQ=3
RESUME=0
CPU_HOME=

deepspeed train.py \
  --ckpt_run \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --freq $FREQ \
  --resume $RESUME \
  --cpu_home $CPU_HOME \
  --ec_run
