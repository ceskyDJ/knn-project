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
	HF_HOME=./transformers-cache ./miniforge3/bin/conda run -n knn-gpu python ./src/layoutlmv2-finetune-window.py

pack: clean
	tar -czvf knn-packed.tar.gz cpu-conda-env.yml gpu-conda-env.yml Makefile src/*

clean:
	rm -rf ./src/__pycache__
