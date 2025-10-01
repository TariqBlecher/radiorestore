#!/bin/bash

module load pytorch
python -m venv deconv-env --system-site-packages
source deconv-env/bin/activate
pip install -e .