#!/bin/bash

conda activate r2d2
export PYTHONPATH="${PYTHONPATH}:./"

for dataset in "openwikitable" "spider" ; do
  for partition in "train" "val" "test" ; do
    python scripts/explore/explore.py dataset=$dataset partition=$partition
    python scripts/explore/plot.py dataset=$dataset partition=$partition
  done
done
