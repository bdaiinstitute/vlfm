# Installing Habitat and relevant datasets

First, export the following environment variables during installation (don't need to put in .bashrc):
```bash
export MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
export MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
export DATA_DIR=</path/to/llm-object-search/data>

# Link to the HM3D ObjectNav episodes dataset, listed here:
# https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets
# From the above page, locate the link to the HM3D ObjectNav dataset.
# Verify that it is the same as the next two lines.
export HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
export HM3D_POINTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/hm3d/v1/pointnav_hm3d_v1.zip

```

### Clone and install habitat-lab, then download datasets
*Ensure that the correct conda environment is activated!!*
```bash
git clone --branch v0.2.4 git@github.com:facebookresearch/habitat-lab.git &&
cd habitat-lab &&
pip install -e habitat-lab &&
pip install -e habitat-baselines &&
cd ..

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

# Download HM3D ObjectNav dataset episodes
wget $HM3D_POINTNAV &&
unzip pointnav_hm3d_v1.zip &&
mkdir -p $DATA_DIR/datasets/pointnav/hm3d  &&
mv pointnav_hm3d_v1 $DATA_DIR/datasets/pointnav/hm3d/v1 &&
rm pointnav_hm3d_v1.zip
```
