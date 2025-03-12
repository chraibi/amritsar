#!/bin/bash

# Exit on errors
set -e

# Define output files
OUTPUT_VIDEO="heatmap_animation.mp4"
FRAME_PATTERN="heatmap_frames/heatmap_%03d.png"

# Remove old video if it exists
if [ -f "$OUTPUT_VIDEO" ]; then
    echo "Removing old animation: $OUTPUT_VIDEO"
    rm "$OUTPUT_VIDEO"
fi

# Generate heatmap frames
echo "Running heatmap.py..."
python heatmap.py

# Check if frames exist before running ffmpeg
if ls heatmap_frames/heatmap_*.png 1> /dev/null 2>&1; then
    echo "Generating animation using ffmpeg..."
#    ffmpeg -framerate 5 -i "$FRAME_PATTERN" -c:v libx264 -r 30 -pix_fmt yuv420p "$OUTPUT_VIDEO"
    ffmpeg -framerate 5 -i "$FRAME_PATTERN" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -r 30 -pix_fmt yuv420p "$OUTPUT_VIDEO"
    open $OUTPUT_VIDEO
    echo "Animation created: $OUTPUT_VIDEO"
else
    echo "Error: No heatmap frames found. Check heatmap.py execution."
    exit 1
fi
