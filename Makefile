.PHONY: install pack

# Install Miniforge
install:
	wget --output-document=miniforge-install.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
	chmod +x ./miniforge-install.sh
	./miniforge-install.sh -b -u -p "$${PWD}/miniforge3"
	rm ./miniforge-install.sh
	./miniforge3/bin/conda env create -f ./conda-env.yml

pack:
	tar -czvf knn-packed.tar.gz conda-env.yml Makefile src/process_dataset.py src/preprocess_images.py
