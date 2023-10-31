# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST))

# Install exact Python and CUDA versions
env:
	conda env update --prune -f environment.yml

# Compile and install exact pip packages
install:
	pip install pip-tools==7.1.0 setuptools==68.0.0
	pip-compile requirements/req.in
	pip-sync requirements/req.txt
	pip install accelerate==0.24.1
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Bump versions of transitive dependencies
upgrade:
	pip install pip-tools==7.1.0 setuptools==68.0.0
	pip-compile --upgrade requirements/req.in
	pip-sync requirements/req.txt

# Setup
setup:
	pre-commit install
	export PYTHONPATH=.
	echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
	git clone https://huggingface.co/google/pix2struct-textcaps-base

# Lint
lint:
	pre-commit run --all-files

# Test experiment
run:
	python run.py
