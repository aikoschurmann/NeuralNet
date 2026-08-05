#!/bin/bash

# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip3 install --upgrade pip

# Install requirements if the file exists
[ -f requirements.txt ] && pip3 install -r requirements.txt

