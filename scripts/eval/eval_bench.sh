#!/bin/bash
set -e

REPO_ROOT=${REPO_ROOT:-$(pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/dataset}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/e1}

VIDEO_PATH=${VIDEO_PATH:-$DATA_ROOT/surgpub}
EVAL_INPUT=${EVAL_INPUT:-annotations/bench/qa_test.csv}
CONV_VERSION=${CONV_VERSION:-qwen2_base}
NUM_FRAME=${NUM_FRAME:-16}

RUN_NAME=${RUN_NAME:-TinyLLaVA-Video-Coldstart-16-38k-e5}
MODEL_PATH=${MODEL_PATH:-$OUTPUT_DIR/checkpoints/$RUN_NAME}
EVAL_OUTPUT_PATH=${EVAL_OUTPUT_PATH:-$OUTPUT_DIR/eval/closed_vqa_full-e5.csv}

FRAME_DIR_NAME=${FRAME_DIR_NAME:-frames}
FRAME_NAME_FORMAT=${FRAME_NAME_FORMAT:-frame_{:05d}.png}

mkdir -p "$(dirname "$EVAL_OUTPUT_PATH")"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python -m tinyllava.eval.run_tiny_llava \
    --model-path "$MODEL_PATH" \
    --conv-mode "$CONV_VERSION" \
    --input_path "$EVAL_INPUT" \
    --video_folder "$VIDEO_PATH" \
    --frame_dir_name "$FRAME_DIR_NAME" \
    --frame_name_format "$FRAME_NAME_FORMAT" \
    --num_frame "$NUM_FRAME" \
    --output_path "$EVAL_OUTPUT_PATH"
