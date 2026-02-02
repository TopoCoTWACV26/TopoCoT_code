#!/usr/bin/env bash
set -x
export PYTHONPATH="./":$PYTHONPATH

python tools/visualization/vis_per_frame.py projects/configs/streamlanesegnets_olv2_multiframe_stage2.py \
--data_path results/pos_predictions.pkl \
--out_dir vis_local/openlanev2/pred  \
--option vis-pred


###for gt
# python tools/visualization/vis_per_frame.py projects/configs/streamlanesegnets_olv2_multiframe_stage2.py \
# --data_path data/data_dict_subset_A_val_lanesegnet_gt_tracks.pkl \
# --out_dir vis_local/openlanev2/gt  \
# --option vis-gt