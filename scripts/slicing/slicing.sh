#!/bin/bash

conda activate r2d2
export PYTHONPATH="${PYTHONPATH}:./"

slicer=no_slicer,rowwise_slicer,colwise_slicer,divisive_slicer
table_embedder=sbertparts_embedder

for dataset in "openwikitable" "spider" ; do
    if [ "$dataset" = "openwikitable" ]; then
        params="num_processes=1 silent=false"
        tiles_per_table=1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000
        partition="train"
    elif [ "$dataset" = "spider" ]; then
        params="skip_large_lakes=10000000"
        tiles_per_table=1,2,3,4,5,6,8,10,12,14,17,21,26,31,38,46,56,68,82,100
        partition="val"
    else
        params=""
    fi

    python scripts/slicing/run.py --multirun 'config_name="slicer=${slicer._target_}_TpT=${slicer.tiles_per_table}"' $params dataset=$dataset partition=$partition table_embedder=$table_embedder slicer=$slicer slicer.tiles_per_table=$tiles_per_table
    python scripts/slicing/eval.py dataset=$dataset partition=$partition
    python scripts/slicing/plot.py dataset=$dataset partition=$partition
done
