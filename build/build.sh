#!/bin/bash -x
#
# Script setting up the environment in the base folder
#

if ! /usr/bin/git pull; then
    echo "Failed to git pull..."
    exit
fi

if ! [ -d ".env" ]; then
  virtualenv -p python3 ../.env
fi
source ../.env/bin/activate

pip install -r requirements.txt
