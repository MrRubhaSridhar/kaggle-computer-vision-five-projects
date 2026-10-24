#!/usr/bin/env bash
# scripts/run_all_evaluations.sh
# Runs validation and evaluation across all five Kaggle Computer Vision projects.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

PYTHON_BIN="$(which python3 || which python)"
echo "============================================================"
echo "    Executing All Five Computer Vision Project Evaluations  "
echo "    Python: $PYTHON_BIN"
echo "============================================================"

# Project 1: Lane Detection
echo -e "\n[1/5] Evaluating Project 1: Road Lane Detection & Departure Warning..."
$PYTHON_BIN 01_lane_detection/scripts/evaluate.py \
    --dataset_dir 01_lane_detection/data/raw/dataset \
    --output_dir 01_lane_detection/results \
    --max_samples 103

# Project 2: Fruit Quality Inspection
echo -e "\n[2/5] Evaluating Project 2: Fruit Quality Inspection..."
$PYTHON_BIN 02_fruit_quality_inspection/scripts/evaluate.py \
    --dataset_root 02_fruit_quality_inspection/data/raw/dataset \
    --output_dir 02_fruit_quality_inspection/results \
    --test_limit 1000

# Project 3: Traffic Sign Recognition
echo -e "\n[3/5] Evaluating Project 3: Traffic Sign Recognition (GTSRB)..."
$PYTHON_BIN 03_traffic_sign_recognition/scripts/evaluate.py \
    --raw_dir 03_traffic_sign_recognition/data/raw \
    --output_dir 03_traffic_sign_recognition/results \
    --test_limit 2000

# Project 4: Robotic Object Detection
echo -e "\n[4/5] Evaluating Project 4: Real-Time Robotic Object Detection (YOLOv8n)..."
$PYTHON_BIN 04_robotic_object_detection/scripts/evaluate.py \
    --raw_dir 04_robotic_object_detection/data/raw \
    --output_dir 04_robotic_object_detection/results

# Project 5: Explainable AI
echo -e "\n[5/5] Evaluating Project 5: Explainable AI for Object Recognition (Grad-CAM)..."
$PYTHON_BIN 05_explainable_ai/scripts/evaluate.py \
    --raw_dir 05_explainable_ai/data/raw \
    --output_dir 05_explainable_ai/results \
    --test_limit 2000

$PYTHON_BIN 05_explainable_ai/scripts/explain.py \
    --raw_dir 05_explainable_ai/data/raw \
    --output_dir 05_explainable_ai/results

echo -e "\n============================================================"
echo "     ALL FIVE PROJECT EVALUATIONS COMPLETED SUCCESSFULLY!   "
echo "============================================================"
