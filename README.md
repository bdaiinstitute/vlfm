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
  habitat-sim=0.2.4 headless pytorch pytorch-cuda \
  transformers \
  -c aihabitat -c pytorch -c huggingface \
  -c nvidia -c conda-forge -y
```

Then, follow the instructions in [readmes/installing_habitat.md](readmes/installing_habitat.md) to install Habitat and relevant datasets.

### Installing GroundingDINO
To install GroundingDINO, you will need `CUDA_HOME` set as an environment variable. If you would like to install a certain version of CUDA that is compatible with the one used to compile your version of pytorch, and you are using conda, you can run the following commands to install CUDA and set `CUDA_HOME`:
```bash
# This example is specifically for CUDA 11.8
mamba install \
    cub \
    thrust \
    cuda-runtime \
    cudatoolkit=11.8 \
    cuda-nvcc==11.8.89 \
    -c "nvidia/label/cuda-11.8.0" \
    -c nvidia &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/include/*  ${CONDA_PREFIX}/include/ &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusparse/include/*  ${CONDA_PREFIX}/include/ &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cublas/include/*  ${CONDA_PREFIX}/include/ &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusolver/include/*  ${CONDA_PREFIX}/include/ &&
export CUDA_HOME=${CONDA_PREFIX}
```

### TODO
1. Add instructions for installing `frontier_exploration`
2. Add instructions for installing FastChat
