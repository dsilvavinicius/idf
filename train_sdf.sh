#!/bin/bash
# Train an SDF using the ablation config with an absolute .ply mesh path.
#
# Usage:
#   ./train_sdf.sh <mesh_path> <output_dir> [run_name]
#
# Arguments:
#   mesh_path   - Absolute path to the .ply mesh file
#   output_dir  - Absolute path to the output directory
#   run_name    - (Optional) Name for this training run. Defaults to the mesh filename without extension.

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <mesh_path> <output_dir> [run_name]"
    echo ""
    echo "  mesh_path   Absolute path to a .ply mesh file"
    echo "  output_dir  Absolute path to the output directory"
    echo "  run_name    Optional name for the training run"
    exit 1
fi

MESH_PATH="$1"
OUTPUT_DIR="$2"

if [ ! -f "$MESH_PATH" ]; then
    echo "Error: File not found: $MESH_PATH"
    exit 1
fi

# Derive run name from filename if not provided
if [ -z "$3" ]; then
    RUN_NAME=$(basename "$MESH_PATH" .ply)
else
    RUN_NAME="$3"
fi

DIR=$(dirname "$0")

echo "Training SDF..."
echo "  Mesh:       $MESH_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Run name:   $RUN_NAME"
echo ""

python "$DIR/net/classes/runner.py" "$DIR/net/experiments/train_sdf_mesh.json" \
    --mesh_path "$MESH_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --name "$RUN_NAME"
