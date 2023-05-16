DATA_PATH="../guanaco_belle_merge_v1.0/merge.json"
OUTPUT_PATH="outs"
MODEL_PATH="decapoda-research/llama-7b-hf"

torchrun --nproc_per_node=8 --master_port=29005 finetune_chat.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--log_steps 200 \
--save_steps 200 \
--test_size 200
