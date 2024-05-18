#!/bin/bash

# Declare an array of site names
sites=(
  "garaz"
  "aha"
  "avmania"
  "doupe"
  "idnes"
  "lupa"
  "pravda"
  "sport"
  "aktualitysk"
  "blesk"
  "e15"
  "isport"
  "mobilmania"
  "seznamzpravy"
  "vtm"
  "auto"
  "connect"
  "lidovky"
  "novinky"
  "sme"
  "zive"
)

for site in "${sites[@]}"; do
  nohup ./miniforge3/bin/conda run -n knn python src/process_dataset.py --sites "$site" > "log-$(date +"%Y-%m-%d")-[$site]-v4-split.txt" 2>&1 &
done
