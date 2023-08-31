#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export OS_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

# We add a sleep of 30 seconds after each command to ensure that the user can see any errors that occur
# if they re-attach to the tmux session within 30 seconds of running this script.

export OS_PYTHON=${OS_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-zsos/vlm/classes.txt}
export START_PORT=${START_PORT:-1218}

# Create a detached tmux session
tmux new-session -d -s vlm_servers

# Split the window vertically
tmux split-window -v -t vlm_servers:0

# Split both panes horizontally
tmux split-window -h -t vlm_servers:0.0
tmux split-window -h -t vlm_servers:0.2

# Run commands in each pane
tmux send-keys -t vlm_servers:0.0 "${OS_PYTHON} -m zsos.vlm.grounding_dino --port ${START_PORT}1 ; sleep 30" C-m
tmux send-keys -t vlm_servers:0.1 "${OS_PYTHON} -m zsos.vlm.blip2itm --port ${START_PORT}2 ; sleep 30" C-m
tmux send-keys -t vlm_servers:0.2 "${OS_PYTHON} -m zsos.vlm.sam --port ${START_PORT}3 ; sleep 30" C-m
tmux send-keys -t vlm_servers:0.3 "${OS_PYTHON} -m zsos.vlm.yolov7 --port ${START_PORT}4 ; sleep 30" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session 'vlm_servers'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t vlm_servers"
