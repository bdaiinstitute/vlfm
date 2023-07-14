#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export OS_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

# We add a sleep of 30 seconds after each command to ensure that the user can see any errors that occur
# if they re-attach to the tmux session within 30 seconds of running this script.

# Create a detached tmux session
tmux new-session -d -s llm_server

# Split the window vertically and run ${OS_PYTHON} -m fastchat.serve.openai_api_server in the first pane
tmux split-window -v -t llm_server:0
tmux send-keys -t llm_server:0.0 "${OS_PYTHON} -m fastchat.serve.openai_api_server --host localhost --port 8000 ; sleep 30" C-m

# Split the second pane horizontally and run ${OS_PYTHON} -m fastchat.serve.model_worker in the new pane
tmux split-window -h -t llm_server:0
tmux send-keys -t llm_server:0.1 "${OS_PYTHON} -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 ; sleep 30" C-m

# Select the third pane and run ${OS_PYTHON} -m fastchat.serve.controller
tmux select-pane -t llm_server:0.2
tmux send-keys -t llm_server:0.2 "${OS_PYTHON} -m fastchat.serve.controller ; sleep 30" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session 'llm_server'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t llm_server"
