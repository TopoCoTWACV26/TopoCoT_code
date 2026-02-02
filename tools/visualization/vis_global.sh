#!/usr/bin/env bash
# set -x
export PYTHONPATH="./":$PYTHONPATH

python tools/visualization/vis_global.py projects/configs/streamlanesegnets_olv2_multiframe_stage2.py \
--data_path results/pos_predictions.pkl \
--out_dir vis_global/openlanev2/pred  \
--option vis-pred \
--per_frame_result 0

# python tools/visualization/vis_global.py projects/configs/streamlanesegnets_olv2_stage2_multiframe_vis.py \
# --data_path data/data_dict_subset_A_val_lanesegnet_gt_tracks.pkl \
# --out_dir vis_global/openlanev2/gt  \
# --option vis-gt \
# --per_frame_result 1


