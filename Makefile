SHELL: /bin/bash

.PHONY: install pack train clean

# Install Miniforge
install:
	wget --output-document=miniforge-install.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
	chmod +x ./miniforge-install.sh
	./miniforge-install.sh -b -u -p "$${PWD}/miniforge3"
	rm ./miniforge-install.sh
	./miniforge3/bin/conda env create -f ./cpu-conda-env.yml
	./miniforge3/bin/conda env create -f ./gpu-conda-env.yml

train:
	mkdir -p ./transformers-cache
	@echo -n -`date +%s` > ./time-measurement.txt
	HF_HOME=./transformers-cache ./miniforge3/bin/conda run -n knn-gpu --no-capture-output python ./src/layoutlmv2-finetune-window.py
	@echo -n "+" >> ./time-measurement.txt
	@date +%s >> ./time-measurement.txt
	@echo -n "Training time: "
	@date -d@`cat ./time-measurement.txt | bc` -u +%H.%M:%S
	@rm ./time-measurement.txt

pack: clean
	tar -czvf knn-packed.tar.gz cpu-conda-env.yml gpu-conda-env.yml Makefile src/*

clean:
	rm -rf ./src/__pycache__
