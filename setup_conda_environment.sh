#!/bin/bash

conda env create -f environment.lock.yaml --force
conda init "$(SHELL##*/"
conda activate mp_perception2
pip install -e .
