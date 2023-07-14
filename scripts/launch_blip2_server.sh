#!/bin/bash

# Name of the tmux session
SESSION_NAME="blip2_server"

# Command to run in the tmux session
COMMAND="python zsos/vlm/blip2.py"

# Check if the tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
else
    # Create a new detached tmux session with the specified name
    tmux new-session -d -s "$SESSION_NAME"

    # Run the command in the tmux session
    tmux send-keys -t "$SESSION_NAME" "$COMMAND" Enter

    echo "Tmux session '$SESSION_NAME' created and command started running."
fi
