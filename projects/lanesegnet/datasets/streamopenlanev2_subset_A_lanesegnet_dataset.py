#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import os
import random
import copy
from typing import Dict, Tuple, Any, List, Callable, Union
import numpy as np
import torch
import mmcv
import cv2
import os.path as osp
import shapely
from shapely.geometry import LineString
from pyquaternion import Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from projects.lanesegnet.datasets.base_dataset import BaseCustom3DDataset
from .openlanev2_evaluate_custom import lanesegnet_evaluate
from ..core.lane.util import fix_pts_interpolate
from ..core.visualizer.lane_segment import draw_annotation_bev
from projects.lanesegnet.datasets.visualization import draw_annotation_pv, assign_attribute, assign_topology
import pickle
from collections import Counter
from collections import defaultdict
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from tqdm import tqdm
@DATASETS.register_module()
# class StreamOpenLaneV2_subset_A_LaneSegNet_Dataset(Custom3DDataset): ###继承的是Custom3DDataset
class StreamOpenLaneV2_subset_A_LaneSegNet_Dataset(BaseCustom3DDataset): ###继承的是Custom3DDataset
    CAMS = ('ring_front_center', 'ring_front_left', 'ring_front_right',
            'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right')
    LANE_CLASSES = ('lane_segment', 'ped_crossing')
    TE_CLASSES = ('traffic_light', 'road_sign')
    TE_ATTR_CLASSES = ('unknown', 'red', 'green', 'yellow',
                       'go_straight', 'turn_left', 'turn_right',
                       'no_left_turn', 'no_right_turn', 'u_turn', 'no_u_turn',
                       'slight_left', 'slight_right')
    MAP_CHANGE_LOGS = [
        '75e8adad-50a6-3245-8726-5e612db3d165',
        '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
        'af170aac-8465-3d7b-82c5-64147e94af7d',
        '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 queue_length=1,
                 filter_empty_te=False,
                 filter_map_change=False,
                 points_num=10,
                 split='train',
                 **kwargs):
        self.filter_map_change = filter_map_change
        self.split = split
        super().__init__(data_root, ann_file, **kwargs)
        self.queue_length = queue_length
        self.filter_empty_te = filter_empty_te
        self.points_num = points_num
        self.LANE_CLASSES = self.CLASSES

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.seq_split_num == -1:
            self.flag = np.arange(len(self.samples))
            return
        
        res = []
        assert self.seq_split_num == 1

        all_log_ids = []
        for s in self.samples:
            if s['segment_id'] not in all_log_ids:
                all_log_ids.append(s['segment_id'])
        
        for idx in range(len(self.samples)):
            res.append(all_log_ids.index(self.samples[idx]['segment_id']))

        self.flag = np.array(res, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotation from a olv2 pkl file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: Annotation info from the json file.
        """
        data_infos = mmcv.load(ann_file, file_format='pkl')
       
        if isinstance(data_infos, dict):
            if self.filter_map_change and self.split == 'train':
           
                data_infos = [info for info in data_infos.values() if info['meta_data']['source_id'] not in self.MAP_CHANGE_LOGS]
     
            else:
                data_infos = list(data_infos.values())

        self.samples = data_infos
        group_all = set([data['segment_id'] for data in self.samples])
        group_all = list(group_all)
        group_all.sort()

        group_all_dict = {key: 0 for key in group_all}
        segment_counts = Counter(data['segment_id'] for data in self.samples)
        sorted_counts = dict(sorted(segment_counts.items()))
        segment_local_idx = defaultdict(int)

        for idx in range(len(self.samples)):
            self.samples[idx]['sample_idx'] = idx
            group_idx = self.samples[idx]['segment_id']
            self.samples[idx]['local_idx'] = segment_local_idx[group_idx]
        
            segment_local_idx[group_idx] += 1  # Increment for next sample in segment
            if group_all_dict[group_idx] == 0:
                self.samples[idx]['prev'] = -1
                group_all_dict[group_idx] = 1
            else:
                self.samples[idx]['prev'] = idx-1
            try:
                if self.samples[idx]['segment_id'] !=self.samples[idx+1]['segment_id']:
                    self.samples[idx]['next'] = -1
                else:
                    self.samples[idx]['next'] = idx+1
            except:
                self.samples[idx]['next'] = -1
       
        num_waypoint = 6
        for idx in range(len(self.samples)):
            current_sample = self.samples[idx]
            current_sample_r = current_sample['pose']['rotation']
            current_sample_t = current_sample['pose']['translation']
            waypoint_list = []
            
            number_segment = sorted_counts[current_sample['segment_id']]
            
            for future_idx in range(num_waypoint):
                
                if current_sample['next'] !=-1 and current_sample['next']==self.samples[idx+1]['sample_idx'] and current_sample['segment_id']==self.samples[idx+1]['segment_id']:
                    if idx+future_idx+1<len(self.samples):
                        future_sample = self.samples[idx+future_idx+1]
                        if  self.samples[idx+future_idx+1]['segment_id'] == current_sample['segment_id']:
                            future_sample_r = future_sample['pose']['rotation']
                            future_sample_t = future_sample['pose']['translation']

                            ###vis
                            # plot_vehicle_trajectory_with_frames(current_sample_r, current_sample_t, future_sample_r, future_sample_t, './vehicle_trajectory.png')
                            cur_g2l_rot = current_sample_r.T
                            cur_g2l_trans = -np.dot(cur_g2l_rot, current_sample_t[:, np.newaxis]).squeeze()
                            global_displacement = future_sample_t - current_sample_t
                            relative_translation = np.dot(cur_g2l_rot, global_displacement[:, np.newaxis]).squeeze()
                            waypoint_list.append(relative_translation)
                            
                            # future_sample_t = current_sample_t + np.dot(current_sample_r, relative_translation)
                        else:
                            break
                    else:
                        break

            future_way_point = np.array(waypoint_list)

            if current_sample['local_idx'] == number_segment - num_waypoint -1:
                fit_waypoints = fit_future_waypoints(future_way_point[:6,:])
                future_fit_t = (current_sample_r @ fit_waypoints.T).T + current_sample_t
                
                # future_fit_t = fit_future_waypoints_v2(future_way_point[:6,:])
            if len(future_way_point)==num_waypoint:

                if future_way_point[-1][0] >= 2:
                    command = np.array([1, 0, 0])  # Turn Right
                elif future_way_point[-1][0] <= -2:
                    command = np.array([0, 1, 0])  # Turn Left
                else:
                    command = np.array([0, 0, 1])  # Go Straight

                self.samples[idx]['annotation']['ego_fut_cmd'] = command

                self.samples[idx]['ego_fut_cmd'] = command
                future_way_point_offset = future_way_point[1:] - future_way_point[:-1]
                future_way_point[1:] = future_way_point_offset
                self.samples[idx]['annotation']['future_way_point'] = future_way_point[:,:2]
            else:
                fit_num = num_waypoint - len(future_way_point)
                fit_future_t = future_fit_t[:fit_num,:]
                global_displacement = fit_future_t - current_sample_t
                relative_translation = global_displacement @ cur_g2l_rot.T 
                if future_way_point is None or future_way_point.ndim == 1:
                    future_way_point = np.empty((0, 3))  # 初始化为空二维矩阵
                future_way_point = np.concatenate([future_way_point, relative_translation], axis=0)

                if future_way_point[-1][0] >= 2:
                    command = np.array([1, 0, 0])  # Turn Right
                elif future_way_point[-1][0] <= -2:
                    command = np.array([0, 1, 0])  # Turn Left
                else:
                    command = np.array([0, 0, 1])  # Go Straight
                self.samples[idx]['annotation']['ego_fut_cmd'] = command
                self.samples[idx]['ego_fut_cmd'] = command
                future_way_point_offset = future_way_point[1:] - future_way_point[:-1]
                future_way_point[1:] = future_way_point_offset
                self.samples[idx]['annotation']['future_way_point'] = future_way_point[:,:2]

        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """

        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['timestamp'],  # Keep as timestamp for temporal ordering
            timestamp=info['timestamp'],    # Also store as 'timestamp' for LLM data loading
            scene_token=info['segment_id']
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_name, cam_info in info['sensor'].items():
                image_path = cam_info['image_path']
                image_paths.append(os.path.join(self.data_root, image_path))

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['extrinsic']['rotation'])
                lidar2cam_t = cam_info['extrinsic']['translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = np.array(cam_info['intrinsic']['K'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_lane_labels_3d']) == 0:
                return None
            if self.filter_empty_te and len(annos['labels']) == 0:
                return None

        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(np.array(info['pose']['rotation']))
        can_bus[:3] = info['pose']['translation']
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus
        input_dict['lidar2global_rotation'] = np.array(info['pose']['rotation'])
        
        input_dict['lidar2global_translation'] = np.array(info['pose']['translation'])
        # NOTE: sample_idx is kept as timestamp (not array index) for:
        # 1. Temporal ordering comparison in prepare_train_data
        # 2. LLM conversation data loading path construction
        input_dict['all_idx'] = info['sample_idx']  # REMOVED: don't override with array index
        input_dict['prev'] = info['prev']
        input_dict['next'] = info['next']
        input_dict['ego_fut_cmd'] = info['ego_fut_cmd']

        # input_dict['lane_id'] = [data['id'] for data in info['annotation']['lane_segment']]
        input_dict['lane_id'] = None
        return input_dict

    def ped2lane_segment(self, points):
        assert points.shape[0] == 5
        dir_vector = points[1] - points[0]
        dir = np.rad2deg(np.arctan2(dir_vector[1], dir_vector[0]))

        if dir < -45 or dir > 135:
            left_boundary = points[[2, 3]]
            right_boundary = points[[1, 0]]
        else:
            left_boundary = points[[0, 1]]
            right_boundary = points[[3, 2]]
        
        centerline = LineString((left_boundary + right_boundary) / 2)
        left_boundary = LineString(left_boundary)
        right_boundary = LineString(right_boundary)

        return centerline, left_boundary, right_boundary

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information
        """

        info = self.data_infos[index]
        ann_info = info['annotation']

        gt_lanes = []
        gt_lane_labels_3d = []
        gt_lane_left_type = []
        gt_lane_right_type = []

        for idx, lane in enumerate(ann_info['lane_segment']):
            
            centerline = lane['centerline']
            LineString_lane = LineString(centerline)
            left_boundary = lane['left_laneline']
            LineString_left_boundary = LineString(left_boundary)
            right_boundary = lane['right_laneline']
            LineString_right_boundary = LineString(right_boundary)
            gt_lanes.append([LineString_lane, LineString_left_boundary, LineString_right_boundary])
            gt_lane_labels_3d.append(0)
            gt_lane_left_type.append(lane['left_laneline_type'])
            gt_lane_right_type.append(lane['right_laneline_type'])

        for area in ann_info['area']:
            if area['category'] == 1 and 'ped_crossing' in self.LANE_CLASSES:
                centerline, left_boundary, right_boundary = self.ped2lane_segment(area['points'])
                gt_lanes.append([centerline, left_boundary, right_boundary])
                gt_lane_labels_3d.append(1)
                gt_lane_left_type.append(0)
                gt_lane_right_type.append(0)

            elif area['category'] == 2 and 'road_boundary' in self.LANE_CLASSES:
                raise NotImplementedError

        topology_lsls = np.array(ann_info['topology_lsls'], dtype=np.float32)

        te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
        te_labels = np.array([sign['attribute'] for sign in ann_info['traffic_element']], dtype=np.int64)
        if len(te_bboxes) == 0:
            te_bboxes = np.zeros((0, 4), dtype=np.float32)
            te_labels = np.zeros((0, ), dtype=np.int64)

        topology_lste = np.array(ann_info['topology_lste'], dtype=np.float32)
        gt_future_waypoint = np.array(ann_info['future_way_point'], dtype=np.float32)
        gt_ego_fut_cmd = np.array(ann_info['ego_fut_cmd'], dtype=np.float32)
        annos = dict(
            gt_lanes_3d = gt_lanes,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = topology_lsls,
            bboxes = te_bboxes,
            labels = te_labels,
            gt_lane_lste_adj = topology_lste,
            gt_lane_left_type = gt_lane_left_type,
            gt_lane_right_type = gt_lane_right_type,
            gt_future_waypoint = gt_future_waypoint,
            gt_ego_fut_cmd = gt_ego_fut_cmd,
        )
        return annos

    def prepare_train_data(self, index):

        data_queue = []

        # temporal aug
        if isinstance(index, list):
            index = index[0]
        prev_indexs_list = list(range(index-self.queue_length, index))

        random.shuffle(prev_indexs_list)

        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        sample_idx = input_dict['sample_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        data_future_list = []

        if input_dict['next'] !=-1:
            input_dict_future = self.get_data_info(input_dict['next'])
            data_future = self.pipeline(input_dict_future)
            data_future_list.append(data_future)
        else:
            input_dict_future = self.get_data_info(0) ##没有未来就随便用0
            data_future = self.pipeline(input_dict_future)
            data_future_list.append(data_future)

        example['future_data'] = data_future_list
        if self.filter_empty_gt and \
                (example is None or len(example['gt_lane_labels_3d']._data) == 0):
            return None
        if self.filter_empty_te and \
                (example is None or len(example['gt_labels']._data) == 0):
            return None

        data_queue.insert(0, example)
        added_count = 0
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)

            if input_dict is None:
                return None
            if input_dict['sample_idx'] < sample_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                    (example is None or len(example['gt_lane_labels_3d']._data) == 0):
                    return None
                sample_idx = input_dict['sample_idx']
                data_queue.insert(0, copy.deepcopy(example))
                added_count += 1
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data

            # LLM data (llm_input_ids, llm_attention_mask, llm_labels) should already
            # be in metas_map[i] since they are collected via meta_keys in CustomCollect3D

            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def format_openlanev2_gt(self):
        gt_dict = {}
        for idx in range(len(self.data_infos)):
            info = copy.deepcopy(self.data_infos[idx])
            key = (self.split, info['segment_id'], str(info['timestamp']))
            areas = []
            for area in info['annotation']['area']:
                if area['category'] == 1:
                    points = area['points']
                    left_boundary = fix_pts_interpolate(points[[0, 1]], 10)
                    right_boundary = fix_pts_interpolate(points[[2, 3]], 10)
                    area['points'] = np.concatenate([left_boundary, right_boundary], axis=0)
                    areas.append(area)
            info['annotation']['area'] = areas
            gt_dict[key] = info
        return gt_dict

    def format_results(self, results, jsonfile_prefix=None):
        pred_dict = {}
        pred_dict['method'] = 'LaneSegNet'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'OpenDriveLab'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        all_pos_results = []
        for idx, result in tqdm(enumerate(results)):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_segment = [],
            )

            if result['lane_results'] is not None:
                lane_results = result['lane_results']
                pred_info['lane_segment'].append(lane_results)
                
            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict

    def evaluate(self, results, logger=None, show=False, out_dir=None, **kwargs):
        """Evaluation in Openlane-V2 subset_A dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            show (bool): Whether to visualize the results.
            out_dir (str): Path of directory to save the results.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        # if show:
        #     assert out_dir, 'Expect out_dir when show is set.'
        #     logger.info(f'Visualizing results at {out_dir}...')
        #     self.show(results, out_dir)
        #     logger.info(f'Visualize done.')

        # logger.info(f'Starting format results...')
        # gt_dict = self.format_openlanev2_gt()

        # print('len of results', len(results))
        # print('len of gt_dict', len(gt_dict))

        # print('formating')

        # pred_dict = self.format_results(results)


        # out_dir = "./results"
        # if out_dir is not None:
        #     if not os.path.exists(out_dir):
        #         os.makedirs(out_dir)
        #     mmcv.dump(pred_dict, os.path.join(out_dir, 'pred_results.pkl'))
        #     mmcv.dump(gt_dict, os.path.join(out_dir, 'gt_results.pkl'))

        # logger.info(f'Starting openlanev2 evaluate...')
        # metric_results = lanesegnet_evaluate(gt_dict, pred_dict)
        metric_results = None
        return metric_results

    def show(self, results, out_dir, score_thr=0.3, show_num=20, **kwargs):
        """Show the results.

        Args:
            results (list[dict]): Testing results of the dataset.
            out_dir (str): Path of directory to save the results.
            score_thr (float): The threshold of score.
            show_num (int): The number of images to be shown.
        """
        for idx, result in enumerate(results):

            info = self.data_infos[idx]

            pred_result = self.format_results([result])
            pred_result = list(pred_result['results'].values())[0]['predictions']
            pred_result = self._filter_by_confidence(pred_result, score_thr)

            pv_imgs = []
            for cam_name, cam_info in info['sensor'].items():
                image_path = os.path.join(self.data_root, cam_info['image_path'])
                image_pv = mmcv.imread(image_path)
                pv_imgs.append(image_pv)

            surround_img = self._render_surround_img(pv_imgs)
            output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/surround.jpg')
            mmcv.imwrite(surround_img, output_path)

            conn_img_gt = draw_annotation_bev(info['annotation'], with_centerline=True, with_laneline=True, with_area=True)
            conn_img_pred = draw_annotation_bev(pred_result, with_centerline=True, with_laneline=True, with_area=True)
            divider = np.ones((conn_img_gt.shape[0], 7, 3), dtype=np.uint8) * 128
            conn_img = np.concatenate([conn_img_gt, divider, conn_img_pred], axis=1)[..., ::-1]

            output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/bev.jpg')
            mmcv.imwrite(conn_img, output_path)

            pv_imgs = []
            for cam_name, cam_info in info['sensor'].items():
                image_path = os.path.join(self.data_root, cam_info['image_path'])
                image_pv = mmcv.imread(image_path, channel_order='rgb')
                image_pv = draw_annotation_pv(
                    cam_name,
                    image_pv,
                    pred_result,
                    cam_info['intrinsic'],
                    cam_info['extrinsic'],
                    False,
                    False,
                )
                pv_imgs.append(image_pv[..., ::-1])

            for cam_idx, image in enumerate(pv_imgs[:1]):
                output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/{self.CAMS[cam_idx]}.jpg')
                mmcv.imwrite(image, output_path)

    @staticmethod
    def _render_surround_img(images):
        all_image = []
        img_height = images[1].shape[0]

        for idx in [1, 0, 2, 5, 3, 4, 6]:
            if idx  == 0:
                all_image.append(images[idx][356:1906, :])
                all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))
            elif idx == 6 or idx == 2:
                all_image.append(images[idx])
            else:
                all_image.append(images[idx])
                all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))

        surround_img_upper = None
        surround_img_upper = np.concatenate(all_image[:5], 1)

        surround_img_down = None
        surround_img_down = np.concatenate(all_image[5:], 1)
        scale = surround_img_upper.shape[1] / surround_img_down.shape[1]
        surround_img_down = cv2.resize(surround_img_down, None, fx=scale, fy=scale)

        divider = np.full((25, surround_img_down.shape[1], 3), (255, 255, 255), dtype=np.uint8)

        surround_img = np.concatenate((surround_img_upper, divider, surround_img_down), 0)
        surround_img = cv2.resize(surround_img, None, fx=0.5, fy=0.5)

        return surround_img

    @staticmethod
    def _filter_by_confidence(annotations, threshold=0.3):
        annotations = annotations.copy()
        areas = annotations['area']
        ls_mask = []
        lane_segments = []
        for ls in annotations['lane_segment']:
            if ls['confidence'] > threshold:
                ls_mask.append(True)
                lane_segments.append(ls)
            else:
                ls_mask.append(False)
        ls_mask = np.array(ls_mask, dtype=bool)
        areas = [area for area in annotations['area'] if area['confidence'] > threshold]

        traffic_elements = annotations['traffic_element']
        te_mask = []
        tes = []
        for te in traffic_elements:
            if te['confidence'] > threshold:
                te_mask.append(True)
                tes.append(te)
            else:
                te_mask.append(False)
        te_mask = np.array(te_mask, dtype=bool)

        annotations['lane_segment'] = lane_segments
        annotations['area'] = areas
        annotations['traffic_element'] = tes
        annotations['topology_lsls'] = annotations['topology_lsls'][ls_mask][:, ls_mask] > 0.5
        annotations['topology_lste'] = annotations['topology_lste'][ls_mask][:, te_mask] > 0.5
        return annotations

def plot_vehicle_trajectory_with_frames(
    current_sample_r, current_sample_t,
    future_sample_r, future_sample_t,
    save_path="vehicle_trajectory.png"
):
    """
    改进版：自动根据坐标值调整绘图范围
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设置图形属性
    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_title('Vehicle Trajectory and Local Frames in Global Coordinates')
    ax.grid(True)
    ax.axis('equal')

    # 计算动态绘图范围（当前和未来位置的±5米范围）
    min_x = min(current_sample_t[0], future_sample_t[0]) - 5
    max_x = max(current_sample_t[0], future_sample_t[0]) + 5
    min_y = min(current_sample_t[1], future_sample_t[1]) - 5
    max_y = max(current_sample_t[1], future_sample_t[1]) + 5
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # 1. 绘制全局坐标系轴（从当前位置出发）
    ax.quiver(
        current_sample_t[0], current_sample_t[1], 
        1, 0, color='gray', linestyle='dashed', 
        scale=10, width=0.005, label='Global X'
    )
    ax.quiver(
        current_sample_t[0], current_sample_t[1],
        0, 1, color='gray', linestyle='dashed',
        scale=10, width=0.005, label='Global Y'
    )

    # 2. 绘制当前自车坐标系轴
    current_x_axis = current_sample_r[:2, 0]  # 自车X轴方向
    current_y_axis = current_sample_r[:2, 1]  # 自车Y轴方向
    
    ax.quiver(
        current_sample_t[0], current_sample_t[1],
        current_x_axis[0], current_x_axis[1],
        color='r', scale=10, width=0.005,
        label='Current Vehicle X (Forward)'
    )
    ax.quiver(
        current_sample_t[0], current_sample_t[1],
        current_y_axis[0], current_y_axis[1],
        color='b', scale=10, width=0.005,
        label='Current Vehicle Y (Left)'
    )
    ax.scatter(
        current_sample_t[0], current_sample_t[1], 
        c='k', s=100, label='Current Position'
    )

    # 3. 绘制未来位姿
    future_x_axis = future_sample_r[:2, 0]
    future_y_axis = future_sample_r[:2, 1]
    
    ax.quiver(
        future_sample_t[0], future_sample_t[1],
        future_x_axis[0], future_x_axis[1],
        color='m', scale=10, width=0.005,
        label='Future Vehicle X'
    )
    ax.quiver(
        future_sample_t[0], future_sample_t[1],
        future_y_axis[0], future_y_axis[1],
        color='c', scale=10, width=0.005,
        label='Future Vehicle Y'
    )
    ax.scatter(
        future_sample_t[0], future_sample_t[1], 
        c='m', s=100, label='Future Position'
    )

    # 4. 绘制行驶轨迹（蓝色箭头）
    ax.quiver(
        current_sample_t[0], current_sample_t[1],
        future_sample_t[0] - current_sample_t[0],
        future_sample_t[1] - current_sample_t[1],
        color='blue', scale=1, scale_units='xy', angles='xy',
        width=0.005, label='Trajectory'
    )

    # 添加图例并保存
    ax.legend(loc='upper right')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)

def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def convert_local_coords_to_global(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts local coordinates to global coordinates.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations stored in array of share [n_times, 2].
    """
    
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)

    return np.dot(transform, coordinates.T).T[:, :2] + np.atleast_2d(np.array(translation)[:2])

def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """

    yaw = angle_of_rotation(quaternion_yaw(rotation))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]

def fit_future_waypoints(waypoints, future_step = 6):
    deltas = np.diff(waypoints, axis=0)

    # 用最后一个差值来外推未来点
    last_delta = deltas[-1]

    # 插值/外推 6 个点
    num_extra = future_step
    last_point = waypoints[-1]
    extra_waypoints = np.array([last_point + (i+1) * last_delta for i in range(num_extra)])

    return extra_waypoints
def fit_future_waypoints_v2(waypoints, future_step = 6):

    t = np.arange(len(waypoints))  # 假设时间步长为 1
    # import pdb; pdb.set_trace()
    # 对 x, y, z 分别做样条插值
    cs_x = CubicSpline(t, waypoints[:, 0], extrapolate=True)
    cs_y = CubicSpline(t, waypoints[:, 1], extrapolate=True)
    cs_z = CubicSpline(t, waypoints[:, 2], extrapolate=True)

    # 生成新时间点（延后 6 步）
    t_new = np.arange(future_step, future_step+future_step)

    # 生成插值点
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)

    # 拼接
    extra_waypoints = np.stack((x_new, y_new, z_new), axis=-1)

    return extra_waypoints