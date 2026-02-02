#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Modified version to support LLM conversation data from train_conv directory            #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import os
import json
import glob
import random
from typing import Dict, Tuple, Any, List, Callable, Union
import numpy as np
import torch
import mmcv
import cv2
import os.path as osp
from shapely.geometry import LineString
from pyquaternion import Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from projects.lanesegnet.datasets.base_dataset import BaseCustom3DDataset
from projects.lanesegnet.datasets.openlanev2_subset_A_lanesegnet_dataset import OpenLaneV2_subset_A_LaneSegNet_Dataset
from ..core.lane.util import fix_pts_interpolate
from ..core.visualizer.lane_segment import draw_annotation_bev
import pickle
from collections import Counter
from collections import defaultdict
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

@DATASETS.register_module()
class OpenLaneV2_subset_A_LaneSegNet_LLM_Dataset(OpenLaneV2_subset_A_LaneSegNet_Dataset):
    """Enhanced dataset class that loads both image data and LLM conversation data.

    This dataset extends the original OpenLaneV2_subset_A_LaneSegNet_Dataset to also load
    LLM conversation data from the train_conv directory, which contains system prompts,
    user prompts, and answers in JSON format.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 train_conv_path='/data_test/home/lizhen/yym/TopoWMChange/data/train_conv',
                 queue_length=1,
                 filter_empty_te=False,
                 filter_map_change=False,
                 points_num=10,
                 split='train',
                 load_llm_data=False,
                 **kwargs):
        """Initialize the dataset with LLM data loading support.

        Args:
            data_root: Path to the main data directory containing train/ subdirectory
            ann_file: Path to the annotation file
            train_conv_path: Path to train_conv directory containing LLM data
            queue_length: Length of the image queue for temporal processing
            filter_empty_te: Whether to filter empty traffic elements
            filter_map_change: Whether to filter map changes
            points_num: Number of points per lane segment
            split: Dataset split ('train', 'val', 'test')
            load_llm_data: Whether to load LLM conversation data
            **kwargs: Additional arguments for parent class
        """
        # Store data_root and train_conv_path as instance variables
        self._data_root = data_root
        self.train_conv_path = train_conv_path
        self.load_llm_data = load_llm_data

        if self.load_llm_data:
            # Build mapping from timestamp to LLM conversation data
            self.llm_data_map = self._build_llm_data_map()
            print(f"Loaded {len(self.llm_data_map)} LLM conversation entries")
        else:
            self.llm_data_map = {}

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            queue_length=queue_length,
            filter_empty_te=filter_empty_te,
            filter_map_change=filter_map_change,
            points_num=points_num,
            split=split,
            **kwargs
        )

    def _build_llm_data_map(self):
        """Build a mapping from timestamps to LLM conversation data."""
        llm_data_map = {}

        # Find all directories in train_conv
        # train_conv_path can be either absolute or relative to data_root
        if os.path.isabs(self.train_conv_path):
            conv_root = self.train_conv_path
        else:
            # If relative, join with data_root
            conv_root = os.path.join(self.data_root, self.train_conv_path)

        if not os.path.exists(conv_root):
            # Handle the case where train_conv is a symlink
            if os.path.islink(conv_root):
                conv_root = os.path.realpath(conv_root)
            else:
                print(f"Warning: train_conv path {conv_root} does not exist")
                return llm_data_map

        # Iterate through all scene directories
        for scene_dir in sorted(os.listdir(conv_root)):
            scene_path = os.path.join(conv_root, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            # Iterate through all timestamp directories
            for timestamp_dir in sorted(os.listdir(scene_path)):
                timestamp_path = os.path.join(scene_path, timestamp_dir)
                if not os.path.isdir(timestamp_path):
                    continue

                # Look for bev_conv.json file
                bev_conv_path = os.path.join(timestamp_path, 'bev_conv.json')
                if os.path.exists(bev_conv_path):
                    try:
                        with open(bev_conv_path, 'r') as f:
                            conv_data = json.load(f)
                        if conv_data and isinstance(conv_data, list) and len(conv_data) > 0:
                            # Extract the conversation
                            entry = conv_data[0]  # Assuming single conversation per frame
                            llm_data_map[timestamp_dir] = {
                                'system': entry.get('system', ''),
                                'prompt': entry.get('prompt', ''),
                                'answer': entry.get('answer', '')
                            }
                    except Exception as e:
                        print(f"Error loading LLM data from {bev_conv_path}: {e}")
                        continue

        return llm_data_map

    def get_data_info(self, index):
        """Get data info according to the given index.

        This method extends the original implementation to include LLM conversation data.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data preprocessing pipelines.
        """
        # Get the original data info
        info = self.data_infos[index]
        input_dict = super().get_data_info(index)

        # Add LLM conversation data if available
        if self.load_llm_data and input_dict:
            timestamp = info['timestamp']
            if timestamp in self.llm_data_map:
                llm_data = self.llm_data_map[timestamp]
                input_dict['llm_data'] = llm_data
            else:
                # If no LLM data is found for this timestamp, add empty strings
                input_dict['llm_data'] = {
                    'system': '',
                    'prompt': '',
                    'answer': ''
                }

        return input_dict

    def get_llm_conv_data(self, index):
        """Get LLM conversation data for a given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: LLM conversation data or empty dict if not available.
        """
        info = self.data_infos[index]
        timestamp = info['timestamp']

        if self.load_llm_data and timestamp in self.llm_data_map:
            return self.llm_data_map[timestamp]
        else:
            return {
                'system': '',
                'prompt': '',
                'answer': ''
            }