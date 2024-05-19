#!/bin/bash

TRANSFORMERS_DATA=./transformers-cache
TRAINING_SCRIPT=./src/layoutlmv2-finetune-window.py

mkdir -p $TRANSFORMERS_DATA

HF_HOME=$TRANSFORMERS_DATA ./miniforge3/bin/conda run -n knn-gpu python $TRAINING_SCRIPT
