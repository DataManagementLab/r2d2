#!/bin/bash

conda activate r2d2
export PYTHONPATH="${PYTHONPATH}:./"

table_embedder="sbertparts_embedder"
tile_embedder="sbertparts_embedder"
query_embedder="sbertparts_embedder"
index="list_index"
merger="closure_merger"

use_cached_embedded_lakes=true
use_cached_sliced_lakes=true
use_cached_tile_embedded_lakes=true

for dataset in "openwikitable" "spider" ; do
    if [ "$dataset" = "openwikitable" ]; then
        params=""
        limit_ground_truths=200
        tiles_per_table=1000
        limit_tabular_context_tokens=1000
        partition="train"
    elif [ "$dataset" = "spider" ]; then
        params="skip_large_lakes=10000000"
        limit_ground_truths=20  # ... per lake
        tiles_per_table=50
        limit_tabular_context_tokens=200
        partition="val"
    else
        params=""
    fi

  merger="closure_merger"
  for slicer in "no_slicer" "rowwise_slicer" "colwise_slicer" "divisive_slicer" ; do
    python scripts/tqa/run.py --multirun 'config_name="embedder=${tile_embedder._target_}_slicer=${slicer._target_}_index=${index._target_}_merger=${merger._target_}_TpT=${slicer.tiles_per_table}_tok=${limit_tabular_context_tokens}"' $params dataset=$dataset partition=$partition limit_ground_truths=$limit_ground_truths table_embedder=$table_embedder slicer=$slicer slicer.tiles_per_table=$tiles_per_table tile_embedder=$tile_embedder query_embedder=$query_embedder index=$index merger=$merger limit_tabular_context_tokens=$limit_tabular_context_tokens use_cached_embedded_lakes=$use_cached_embedded_lakes use_cached_sliced_lakes=$use_cached_sliced_lakes use_cached_tile_embedded_lakes=$use_cached_tile_embedded_lakes "$@"
  done

  merger="no_merger"
  slicer="divisive_slicer"
  python scripts/tqa/run.py --multirun 'config_name="embedder=${tile_embedder._target_}_slicer=${slicer._target_}_index=${index._target_}_merger=${merger._target_}_TpT=${slicer.tiles_per_table}_tok=${limit_tabular_context_tokens}"' $params dataset=$dataset partition=$partition limit_ground_truths=$limit_ground_truths table_embedder=$table_embedder slicer=$slicer slicer.tiles_per_table=$tiles_per_table tile_embedder=$tile_embedder query_embedder=$query_embedder index=$index merger=$merger limit_tabular_context_tokens=$limit_tabular_context_tokens use_cached_embedded_lakes=$use_cached_embedded_lakes use_cached_sliced_lakes=$use_cached_sliced_lakes use_cached_tile_embedded_lakes=$use_cached_tile_embedded_lakes "$@"

  python scripts/tqa/eval.py dataset=$dataset partition=$partition
done
