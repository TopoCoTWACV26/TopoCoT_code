#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
from projects.lanesegnet.core.visualizer.lane_vis import draw_annotation_bev
import mmcv
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head,build_neck
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.plugin.models.utils.memory_buffer import StreamTensorMemory, StreamWMLatentMemory
from ...utils.builder import build_bev_constructor
import torch.nn.functional as F
import os
import time
from mmcv.utils import get_logger
from projects.lanesegnet.utils.builder import build_wm_bev_constructor
import torch.nn as nn
@DETECTORS.register_module()
class StreamLaneSegNet(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 lane_head=None,
                 bev_seg_head=None,
                wm_bev_constructor=None,
                 bev_h=None,
                 bev_w=None,
                 roi_size=None,
                 video_test_mode=False,
                 streaming_cfg=dict(),
                 use_llm=False,
                 **kwargs):

        super(StreamLaneSegNet, self).__init__(**kwargs)

        if bev_constructor is not None:
            self.bev_constructor = build_bev_constructor(bev_constructor)

        if lane_head is not None:
            lane_head.update(train_cfg=self.train_cfg.lane)
            self.pts_bbox_head = build_head(lane_head)
        else:
            self.pts_bbox_head = None

        if bev_seg_head is not None:
            self.bev_seg_head = build_head(bev_seg_head)
        else:
            self.bev_seg_head = None

        self.use_llm = use_llm

        self.fp16_enabled = False

        # BEV 
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size
        self.embed_dims = 256
        self.wm_bev_constructor = build_wm_bev_constructor(wm_bev_constructor)
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.action_aware_bev_encoder = nn.Sequential(
            nn.Linear(self.embed_dims + 6*2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.loss_plan_rec = nn.MSELoss(reduction='none')
        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
            self.stream = streaming_cfg['stream']
        else:
            self.streaming_bev = False
            self.stream = False
        if self.streaming_bev:
            self.stream_fusion_neck = build_neck(streaming_cfg['fusion_cfg'])

            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamWMLatentMemory(
                self.batch_size,
            )
            
            xmin, xmax = -roi_size[0]/2, roi_size[0]/2
            ymin, ymax = -roi_size[1]/2, roi_size[1]/2
            zmin, zmax = -roi_size[2]/2, roi_size[2]/2
            x = torch.linspace(xmin, xmax, bev_w)
            y = torch.linspace(ymax, ymin, bev_h)
            # z = torch.linspace(zmax, zmin, 4.0*100/51.2)

            y, x = torch.meshgrid(y, x)
            # z, y, x = torch.meshgrid(z, y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)

            self.register_buffer('plane', plane.double())
            self.stream_fusion_neck.init_weights()

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def wm_prediction(self, view_query_feat, pose_memory, img_metas): ##cur_waypoint # 1，3，2

        view_query_feat = view_query_feat.permute(1,2,0).unsqueeze(0)

        view_query_feat = view_query_feat.permute(0,3,1,2)
        view_query_feat = self.down_sample(view_query_feat)
        view_query_feat = view_query_feat.flatten(2).transpose(1,2)
        batch_size, num_tokens, num_channel = view_query_feat.shape #1 5000 256

        pos_encoding_list = []

        prev_e2g_trans = view_query_feat.new_tensor(pose_memory['lidar2global_translation'], dtype=torch.float64)
        prev_e2g_rot = view_query_feat.new_tensor(pose_memory['lidar2global_rotation'], dtype=torch.float64)
        curr_e2g_trans = view_query_feat.new_tensor(img_metas['lidar2global_translation'], dtype=torch.float64)
        curr_e2g_rot = view_query_feat.new_tensor(img_metas['lidar2global_rotation'], dtype=torch.float64)
        
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(view_query_feat.device)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(view_query_feat.device)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        pos_encoding = prev2curr_matrix.float()[:3].view(-1)
        pos_encoding = pos_encoding.view(1, -1).repeat(1, num_tokens, 1)
        pos_encoding_list.append(pos_encoding.squeeze(0))
          
        pos_encoding = torch.stack(pos_encoding_list)

        cur_view_query_feat_with_ego = torch.cat([view_query_feat, pos_encoding], dim=-1) 
        action_aware_latent = self.action_aware_bev_encoder(cur_view_query_feat_with_ego) #torch.Size([1, 20000, 256])
        
        action_aware_latent = action_aware_latent.permute(1,0,2)
        pose_memory = [pose_memory]
        wm_next_latent_downsample = self.wm_bev_constructor(action_aware_latent, pose_memory, prev_bev = None)
        wm_next_latent_downsample = wm_next_latent_downsample.transpose(1, 2).reshape(batch_size, num_channel, int(self.bev_h/2), int(self.bev_w/2))

        wm_next_latent = F.interpolate(wm_next_latent_downsample, size=( self.bev_h,  self.bev_w), mode='bilinear', align_corners=False)

        return wm_next_latent[0]

    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []
        wm_next_latent_list = []
        single_frame_feats_list = []
        memory = self.bev_memory.get(img_metas) ###取memory
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if self.stream:
                if is_first_frame:
                    ###自己和自己融合
                    new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                    fused_feats_list.append(new_feat)
                    wm_next_latent_list.append(new_feat)
                else:

                    wm_next_latent = self.wm_prediction(bev_memory[i], pose_memory[i], img_metas[i])
                    new_feat = self.stream_fusion_neck(wm_next_latent, curr_bev_feats[i])
                    fused_feats_list.append(new_feat)
                    wm_next_latent_list.append(wm_next_latent)

                wm_next_latent = torch.stack(wm_next_latent_list)
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                single_frame_feats_list.append(new_feat)
            else:

                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
                wm_next_latent = None
        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)
        if self.stream: ###一部分用于单帧训练一部分用于多帧，第一层用单帧，第二层和以后用多帧
            single_frame_feats = torch.stack(single_frame_feats_list, dim = 0)
 
            # fused_feats = torch.cat((single_frame_feats, fused_feats), dim = 0)

        return fused_feats, wm_next_latent, is_first_frame_list

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:

            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_constructor(img_feats, img_metas, prev_bev)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_lanes_3d=None,
                      gt_lane_labels_3d=None,
                      gt_lane_adj=None,
                      gt_lane_left_type=None,
                      gt_lane_right_type=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_instance_masks=None,
                      gt_bboxes_ignore=None,
                      future_data=None,
                      gt_future_waypoint=None,
                      gt_ego_fut_cmd=None,
                      ):

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]

        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
 
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        bev_feats = bev_feats.view(bev_feats.shape[0], self.bev_h, self.bev_w, 256)  #1 100 200 256还是1 200 100 256
        bev_feats = bev_feats.permute(0, 3, 1, 2) 

         #torch.Size([4, 256, 100, 200])
        bs = len(img_metas)

        if self.streaming_bev:
            self.bev_memory.train()
            bev_feats, wm_next_latent, is_first_frame_list = self.update_bev_feature(bev_feats, img_metas)

        # BEV semantic segmentation loss (Stage1 supervision)
        if self.bev_seg_head is not None and gt_instance_masks is not None:
            bev_feats_for_seg = bev_feats  # (B, C, H, W)
            seg_pred = self.bev_seg_head(bev_feats_for_seg)
            # Convert list to tensor if needed
            if isinstance(gt_instance_masks, list):
                seg_label = torch.stack(gt_instance_masks).unsqueeze(1).float().to(bev_feats.device)  # (B, 1, H, W)
            else:
                seg_label = gt_instance_masks.unsqueeze(1).float()  # (B, 1, H, W)
            seg_losses = self.bev_seg_head.losses(seg_pred, seg_label)
            losses = seg_losses
        else:
            losses = dict()

        # World model reconstruction loss (Stage1/Stage2/Stage3)
        # Compute loss_wmrec when stream=True and pts_bbox_head exists
        if self.stream  and wm_next_latent is not None:
            bev_feats_for_wmrec = bev_feats.permute(0, 2, 3, 1)
            bev_feats_for_wmrec = bev_feats_for_wmrec.flatten(start_dim=1, end_dim=2)

            wm_next_latent_flat = wm_next_latent.permute(0, 2, 3, 1)
            wm_next_latent_flat = wm_next_latent_flat.flatten(start_dim=1, end_dim=2)

            loss_rec = self.loss_plan_rec(wm_next_latent_flat, bev_feats_for_wmrec[:,:,:].detach())

            loss_rec_mean = loss_rec.mean(dim=(1,2))
            not_is_first_frame_list = [not x for x in is_first_frame_list]
            rec_weight = torch.tensor(not_is_first_frame_list, dtype = torch.float32, device = bev_feats_for_wmrec.device)
            loss_rec_weight = loss_rec_mean * rec_weight
            loss_rec_weight = loss_rec_weight.sum() / rec_weight.sum()

            if not any(not_is_first_frame_list):
                loss_rec_weight = wm_next_latent.new_zeros((1,))

            losses['lane_head.' + 'loss_wmrec'] = loss_rec_weight

        # LLM related losses (controlled by use_llm flag) - Stage2/Stage3 only
        if self.use_llm:
            bev_feats_llm = bev_feats.permute(0, 2, 3, 1)
            bev_feats_llm = bev_feats_llm.flatten(start_dim=1, end_dim=2)

            ####预训练

            #### +LLM

            if self.stream:
                chatloss = self.pts_bbox_head(img_feats, bev_feats_llm, img_metas, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type, return_loss=True)
            else:
                chatloss = self.pts_bbox_head(img_feats, bev_feats_llm, img_metas, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type, return_loss=True)

            losses['lane_head.chatloss'] = chatloss
            
            # 获取coord_loss用于监控（不参与梯度回传）
            coord_loss_dict = self.pts_bbox_head.loss(None, None, None, None, img_metas)
            if 'coord_loss' in coord_loss_dict and coord_loss_dict['coord_loss'] is not None:
                losses['lane_head.coord_loss'] = coord_loss_dict['coord_loss']

  

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        # t1= time.time()
        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # t2=time.time()
# 
        # print('fps',1/(t2-t1))
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None,  seq_info=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)
        scene_name, local_idx, seq_length  = seq_info[0]
        img_metas[0]['local_idx'] = local_idx
        first_frame = (local_idx == 0)
        
        bev_feats = self.bev_constructor(x, img_metas, prev_bev)

        bev_feats = bev_feats.view(bev_feats.shape[0], self.bev_h, self.bev_w, 256)  #1 100 200 256
        bev_feats = bev_feats.permute(0, 3, 1, 2) 
         #torch.Size([4, 256, 100, 200])

        if self.streaming_bev:
            self.bev_memory.eval()
            bev_feats, _, is_first_frame_list = self.update_bev_feature(bev_feats, img_metas)


            
        bev_feats = bev_feats.permute(0, 2, 3, 1)

        bev_feats = bev_feats.flatten(start_dim=1, end_dim=2)

        outs = self.pts_bbox_head(x, bev_feats, img_metas, return_loss=False)
      

        meta_results = img_metas


        return bev_feats, outs, meta_results

    def simple_test(self, img_metas, img=None, prev_bev=None, seq_info=None, rescale=False):
        """Test function without augmentaiton."""
  
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, lane_results, meta_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale, seq_info = seq_info)
        
        if lane_results is None:
            return new_prev_bev, results_list
        
        for result_dict,  meta in zip(results_list, meta_results):

            result_dict['bbox_results'] = None
            result_dict['lsls_results'] = None
            result_dict['lste_results'] = None
            result_dict['pos_results'] = None
            result_dict['meta_results'] = meta
            result_dict['sample_idx'] = img_metas[0]['all_idx']
            result_dict['all_idx'] = img_metas[0]['all_idx']
            
        result_dict['lane_results'] = lane_results['llm_generated_text']

        scene_token = img_metas[0]['scene_token']
        time_stamp = img_metas[0]['sample_idx']
        save_path = './work_dirs/test_output/' +scene_token + '/' + str(time_stamp) + '/'
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + 'llm_generated_text.txt', 'w', encoding='utf-8') as f:
            f.write(lane_results['llm_generated_text'])
 
        return new_prev_bev, results_list

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.streaming_bev:
            self.bev_memory.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        if self.streaming_bev:
            self.bev_memory.eval()
