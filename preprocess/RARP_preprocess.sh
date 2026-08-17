#!/bin/bash
set -e

RARP_ROOT=${RARP_ROOT:-dataset/RARP50}
SPLIT=${SPLIT:-trainset}
FPS=${FPS:-60}
SCALE=${SCALE:-960:540}
VIDEO_NAME=${VIDEO_NAME:-video_left.avi}
FRAME_DIR_NAME=${FRAME_DIR_NAME:-frame}
SKIP_EXISTING=${SKIP_EXISTING:-1}

split_dir="$RARP_ROOT/$SPLIT"

if [ ! -d "$split_dir" ]; then
    echo "Missing split directory: $split_dir" >&2
    exit 1
fi

for case_dir in "$split_dir"/*/ ; do
    video_path="$case_dir/$VIDEO_NAME"
    frame_dir="$case_dir/$FRAME_DIR_NAME"

    if [ ! -f "$video_path" ]; then
        echo "No $VIDEO_NAME found in: $case_dir"
        continue
    fi

    if [ "$SKIP_EXISTING" = "1" ] && [ -d "$frame_dir" ] && find "$frame_dir" -name "*.png" -print -quit | grep -q .; then
        echo "Skipping existing frames: $frame_dir"
        continue
    fi

    echo "Processing: $video_path"
    mkdir -p "$frame_dir"
    ffmpeg -hide_banner -loglevel error -y \
        -i "$video_path" \
        -vf "fps=$FPS,scale=$SCALE" \
        "$frame_dir/%09d.png"
done
