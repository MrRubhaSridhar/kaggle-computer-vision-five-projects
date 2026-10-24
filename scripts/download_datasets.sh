#!/usr/bin/env bash
# scripts/download_datasets.sh
# Downloads and extracts the 5 Kaggle datasets into their respective project folders.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo "          Downloading Kaggle CV Datasets                    "
echo "============================================================"

PROJECT_NUM="${1:-all}"

download_p1() {
    echo -e "\n[+] Downloading Project 1: Road Lane Detection Dataset..."
    mkdir -p 01_lane_detection/data/raw
    kaggle datasets download -d princekhunt19/road-lane-segmentation-imgs-and-labels -p 01_lane_detection/data/raw --unzip
    echo "[✓] Project 1 dataset downloaded successfully."
}

download_p2() {
    echo -e "\n[+] Downloading Project 2: Fruit Quality Inspection Dataset..."
    mkdir -p 02_fruit_quality_inspection/data/raw
    kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p 02_fruit_quality_inspection/data/raw --unzip
    echo "[✓] Project 2 dataset downloaded successfully."
}

download_p3() {
    echo -e "\n[+] Downloading Project 3: German Traffic Sign Recognition Dataset..."
    mkdir -p 03_traffic_sign_recognition/data/raw
    kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p 03_traffic_sign_recognition/data/raw --unzip
    echo "[✓] Project 3 dataset downloaded successfully."
}

download_p4() {
    echo -e "\n[+] Downloading Project 4: Construction Site Safety Roboflow Dataset..."
    mkdir -p 04_robotic_object_detection/data/raw
    kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow -p 04_robotic_object_detection/data/raw --unzip
    echo "[✓] Project 4 dataset downloaded successfully."
}

download_p5() {
    echo -e "\n[+] Downloading Project 5: CIFAR-10 Dataset..."
    mkdir -p 05_explainable_ai/data/raw
    kaggle datasets download -d ekaakurniawan/the-cifar10-dataset -p 05_explainable_ai/data/raw --unzip
    echo "[✓] Project 5 dataset downloaded successfully."
}

case "$PROJECT_NUM" in
    1) download_p1 ;;
    2) download_p2 ;;
    3) download_p3 ;;
    4) download_p4 ;;
    5) download_p5 ;;
    all)
        download_p1
        download_p2
        download_p3
        download_p4
        download_p5
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|5|all]"
        exit 1
        ;;
esac

echo -e "\n============================================================"
echo "          Dataset Download Phase Completed!                 "
echo "============================================================"
