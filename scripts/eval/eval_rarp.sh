#!/bin/bash
set -e

REPO_ROOT=${REPO_ROOT:-$(pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/dataset}
RARP_ROOT=${RARP_ROOT:-$DATA_ROOT/RARP50}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/rarp}

VIDEO_PATH=${VIDEO_PATH:-$DATA_ROOT}
EVAL_INPUT=${EVAL_INPUT:-annotations/rarp/qa_test.json}
CONV_VERSION=${CONV_VERSION:-qwen2_base}
NUM_FRAME=${NUM_FRAME:-16}

RUN_NAME=${RUN_NAME:RUN_NAME}
MODEL_PATH=${MODEL_PATH:-$OUTPUT_DIR/checkpoints/$RUN_NAME}
EVAL_OUTPUT_PATH=${EVAL_OUTPUT_PATH:-$OUTPUT_DIR/eval/rarp_test.csv}

mkdir -p "$(dirname "$EVAL_OUTPUT_PATH")"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python -m tinyllava.eval.run_tiny_llava_rarp \
    --model-path "$MODEL_PATH" \
    --conv-mode "$CONV_VERSION" \
    --input_path "$EVAL_INPUT" \
    --video_folder "$VIDEO_PATH" \
    --num_frame "$NUM_FRAME" \
    --output_path "$EVAL_OUTPUT_PATH"
