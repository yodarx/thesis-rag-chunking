#!/bin/bash
set -e

# Usage: ./rsync_remote_run.sh <remote_user> <remote_host> <ssh_key_path>
# Example: ./rsync_remote_run.sh ubuntu 1.2.3.4 ~/.ssh/id_rsa

REMOTE_USER="$1"
REMOTE_HOST="$2"
SSH_KEY="$3"
PROJECT_DIR="$(pwd)"
REMOTE_PROJECT_DIR="~/thesis-rag-chunking-github"
RESULTS_DIR="indices" # or indices, as needed

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <remote_user> <remote_host> <ssh_key_path>"
  exit 1
fi

# Rsync project to remote
rsync -az --delete -e "ssh -i $SSH_KEY" "$PROJECT_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

# Run everything remotely
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PROJECT_DIR; bash startup_build_indices.sh"

# Rsync results back
rsync -az -e "ssh -i $SSH_KEY" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/$RESULTS_DIR/" "$PROJECT_DIR/$RESULTS_DIR/"

# Shutdown remote server
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "sudo shutdown -h now"
