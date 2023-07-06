# llm-object-search

## Installation

Create the conda environment:
```bash
conda_env_name=zsos # 'zero-shot object search'
conda create -n $conda_env_name python=3.9 -y &&
conda activate $conda_env_name &&

# Mamba is used for much, much faster installation.
conda install mamba -y -c conda-forge &&
mamba install \
  habitat-sim=0.2.4 headless pytorch==1.13.1 pytorch-cuda=11.6 \
  transformers \
  -c aihabitat -c pytorch -c huggingface \
  -c nvidia -c conda-forge -y
```

Then, follow the instructions in [readmes/installing_habitat.md](readmes/installing_habitat.md) to install Habitat and relevant datasets.

### TODO
1. Add instructions for installing `frontier_exploration`
2. Add instructions for installing FastChat
