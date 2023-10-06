# Vision-Language Frontier Maps

## 1. Installation

Create the conda environment:
```bash
conda_env_name=vlfm
conda create -n $conda_env_name python=3.9 -y &&
conda activate $conda_env_name
```
If you are using habitat and are doing simulation experiments, install this repo into your env with the following:
```bash
pip install -e .[habitat]
```
If you are using the Spot robot, install this repo into your env with the following:
```bash
pip install -e .[reality]
```
Install all the dependencies:
```bash
git clone git@github.com:WongKinYiu/yolov7.git  # if using YOLOv7
git clone git@github.com:IDEA-Research/GroundingDINO.git
```
Follow the install directions for GroundingDINO. Nothing needs to be done for YOLOv7, but it needs to be cloned into the repo.

### Installing GroundingDINO (Only if using conda-installed CUDA)
Only attempt if the installation instructions in the GroundingDINO repo do not work.

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

## 2. Downloading the HM3D dataset
First, set the following variables during installation (don't need to put in .bashrc):
```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR=</path/to/vlfm/data>

# Link to the HM3D ObjectNav episodes dataset, listed here:
# https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets
# From the above page, locate the link to the HM3D ObjectNav dataset.
# Verify that it is the same as the next two lines.
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
```

### Clone and install habitat-lab, then download datasets
*Ensure that the correct conda environment is activated!!*
```bash
# Download HM3D 3D scans (scenes_dataset)
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR &&

# Download HM3D ObjectNav dataset episodes
wget $HM3D_OBJECTNAV &&
unzip objectnav_hm3d_v2.zip &&
mkdir -p $DATA_DIR/datasets/objectnav/hm3d  &&
mv objectnav_hm3d_v2 $DATA_DIR/datasets/objectnav/hm3d/v2 &&
rm objectnav_hm3d_v2.zip
```

## 3. Downloading weights for various models
The weights for MobileSAM, GroundingDINO, and PointNav must be saved to the `data/` directory. The weights can be downloaded from the following links:
- `mobile_sam.pt`:  https://github.com/ChaoningZhang/MobileSAM
- `groundingdino_swint_ogc.pth`: https://github.com/IDEA-Research/GroundingDINO
- `yolov7-e6e.pt`: https://github.com/WongKinYiu/yolov7
- `pointnav_weights.pth`:

## 4. Evaluation within Habitat
Run the following to evaluate on the HM3D dataset:
```bash
python -m vlfm.run
```
To evaluate on MP3D, run the following:
```bash
python -m vlfm.run habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz
```
