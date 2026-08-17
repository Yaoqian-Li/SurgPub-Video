#!/bin/bash
set -e

REPO_ROOT=${REPO_ROOT:-$(pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/dataset}
MODEL_ROOT=${MODEL_ROOT:-/path/to/models}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/surgpub}

VIDEO_DATA_PATH=${VIDEO_DATA_PATH:-annotations/bench/qa_train.json}
VIDEO_PATH=${VIDEO_PATH:-$DATA_ROOT}

PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-$MODEL_ROOT/TinyLLaVA-Video-Qwen2.5-3B-Group-16-512}
LLM_VERSION=${LLM_VERSION:-Qwen/Qwen2.5-3B}
VT_VERSION=${VT_VERSION:-google/siglip-so400m-patch14-384}
CN_VERSION=${CN_VERSION:-groupresampler}
TRAIN_RECIPE=${TRAIN_RECIPE:-common}
CONV_VERSION=${CONV_VERSION:-qwen2_base}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-3072}
NUM_FRAME=${NUM_FRAME:-16}
NUM_QUERY=${NUM_QUERY:-512}
NUM_EPOCHS=${NUM_EPOCHS:-5}
SAVE_STEPS=${SAVE_STEPS:-2500}
GPUS=${GPUS:-localhost:0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29501}

RUN_NAME=${RUN_NAME:surgpub}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-$OUTPUT_DIR/checkpoints/$RUN_NAME}
LOG_PATH=${LOG_PATH:-$OUTPUT_DIR/logs/train.log}

mkdir -p "$(dirname "$LOG_PATH")" "$TRAIN_OUTPUT_DIR"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

deepspeed --include "$GPUS" --master_port "$MASTER_PORT" tinyllava/train/train_coldstart.py > "$LOG_PATH" 2>&1 \
    --deepspeed ./scripts/zero3.json \
    --video_data_path "$VIDEO_DATA_PATH" \
    --video_folder "$VIDEO_PATH" \
    --is_multimodal True \
    --conv_version "$CONV_VERSION" \
    --model_name_or_path "$LLM_VERSION" \
    --vision_tower "$VT_VERSION" \
    --connector_type "$CN_VERSION" \
    --num_frames "$NUM_FRAME" \
    --num_queries "$NUM_QUERY" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe "$TRAIN_RECIPE" \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type constant \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name "$RUN_NAME"
