#!/bin/bash

DATA_PATH="sample/abtest.json" 
OUTPUT_PATH="lora-Vicuna"
MODEL_PATH="decapoda-research/llama-7b-hf"
lora_checkpoint="./lora-Vicuna/checkpoint-8000"
from_data_beginning=False # False
# TEST_SIZE=200

# CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
python finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 3 \
--save_steps 3 \
--test_size 0 \
--resume_from_checkpoint $lora_checkpoint \
--ignore_data_skip $from_data_beginning
