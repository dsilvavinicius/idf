#!/bin/bash
# Train an SDF using the ablation config with an absolute .ply mesh path.
#
# Usage:
#   ./train_sdf.sh <mesh_path> <output_dir> <output_filename> [run_name]
#
# Arguments:
#   mesh_path       - Absolute path to the .ply mesh file
#   output_dir      - Absolute path to the output directory
#   output_filename - Filename for the final detailed mesh (e.g. result.ply), saved at the root of output_dir
#   run_name        - (Optional) Name for this training run. Defaults to the mesh filename without extension.

set -e

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <mesh_path> <output_dir> <output_filename> [run_name]"
    echo ""
    echo "  mesh_path        Absolute path to a .ply mesh file"
    echo "  output_dir       Absolute path to the output directory"
    echo "  output_filename  Filename for the final detailed mesh (saved at root of output_dir)"
    echo "  run_name         Optional name for the training run"
    exit 1
fi

MESH_PATH="$1"
OUTPUT_DIR="$2"
OUTPUT_FILENAME="$3"

if [ ! -f "$MESH_PATH" ]; then
    echo "Error: File not found: $MESH_PATH"
    exit 1
fi

# Derive run name from filename if not provided
if [ -z "$4" ]; then
    RUN_NAME=$(basename "$MESH_PATH" .ply)
else
    RUN_NAME="$4"
fi

DIR=$(dirname "$0")

TRAIN_DIR="$OUTPUT_DIR/$RUN_NAME"

echo "Training SDF..."
echo "  Mesh:            $MESH_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Training dir:    $TRAIN_DIR"
echo "  Output filename: $OUTPUT_FILENAME"
echo "  Run name:        $RUN_NAME"
echo ""

python "$DIR/net/classes/runner.py" "$DIR/net/experiments/train_sdf_mesh.json" \
    --mesh_path "$MESH_PATH" \
    --output_dir "$TRAIN_DIR" \
    --name "$RUN_NAME"

# Copy the final detailed mesh to the root of output_dir with the user-specified name
FINAL_MESH=$(find "$TRAIN_DIR" -name "mesh_mesh_HighRes_final_*.ply" -type f | sort | tail -n 1)

if [ -n "$FINAL_MESH" ]; then
    cp "$FINAL_MESH" "$OUTPUT_DIR/$OUTPUT_FILENAME"
    echo ""
    echo "Final mesh copied to: $OUTPUT_DIR/$OUTPUT_FILENAME"
else
    echo ""
    echo "Warning: Could not find the final detailed mesh (mesh_mesh_HighRes_final_*.ply) in $TRAIN_DIR"
    exit 1
fi
