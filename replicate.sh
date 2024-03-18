#!/bin/bash

if ! command -v conda >/dev/null; then
    echo "You must have Conda installed (see README.md)!"
fi

if [ ! -d "data/spider/download" ]; then
    echo "You must manually download Spider (see README.md)!"
    exit
fi

if [ ! -d "data/openai_cache" ]; then
    echo "You must obtain data/openai_cache from us (see README.md)!"
    exit
fi

eval "$(conda shell.bash hook)"
conda env create -f environment-cuda.yml  # use environment.yml if you don't have CUDA
conda activate r2d2
export PYTHONPATH=${PYTHONPATH}:./

python scripts/lakes/download_openwikitable.py
python scripts/lakes/create_openwikitable_lakes.py

python scripts/lakes/create_spider_lakes.py

bash scripts/explore/explore.sh

bash scripts/slicing/slicing.sh

bash scripts/tqa/tqa.sh