#!/bin/bash
# Download dataset, unzip, and clean it
# Usage: bash script/download_data.sh

# Download dataset
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/nnytj2v3n5-1.zip -P ./data
unzip ./data/nnytj2v3n5-1.zip -d ./data && 
mv "./data/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images" "./data/Segmented Medicinal Leaf Images" &&
rm -rf "./data/Medicinal Leaf Dataset" &&
python ./tools/clean_data.py
