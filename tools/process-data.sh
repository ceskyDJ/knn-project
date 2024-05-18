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

# Loop through each site and run the command
for site in "${sites[@]}"; do
  nohup ./miniforge3/bin/conda run -n knn python src/process_dataset.py --sites "$site" & | tee "log-$(date +"%Y-%m-%d")-[$site]-v4-split.txt"
done
