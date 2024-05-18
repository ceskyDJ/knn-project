#!/bin/bash

# Declare an array of site names
sites=(
  "GARAZ"
  "AHA"
  "AVMANIA"
  "DOUPE"
  "IDNES"
  "LUPA"
  "PRAVDA"
  "SPORT"
  "AKTUALITYSK"
  "BLESK"
  "E15"
  "ISPORT"
  "MOBILMANIA"
  "SEZNAMZPRAVY"
  "VTM"
  "AUTO"
  "CONNECT"
  "LIDOVKY"
  "NOVINKY"
  "SME"
  "ZIVE"
)

# Loop through each site and run the command
for site in "${sites[@]}"; do
  nohup ./miniforge3/bin/conda run -n knn python src/process_dataset.py --sites "$site" & | tee "log-$(date +"%Y-%m-%d")-[$site]-v4-split.txt"
done
