BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="./lora-Vicuna/checkpoint-8000" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=1 # 1: use local model, 0: use huggingface model
TYPE_WRITER=1 # whether output streamly

#if [[ USE_LOCAL -eq 1 ]]; then
#  cp sample/instruct/adapter_config.json $LORA_PATH
#fi
python generate.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL \
    --use_typewriter $TYPE_WRITER
