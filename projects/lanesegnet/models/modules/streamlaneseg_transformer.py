#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from .lane_attention import LaneAttention

@TRANSFORMER.register_module()
class StreamLaneSegNetTransformer(BaseModule):

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 points_num=1,
                 pts_dim=3,
                 **kwargs):
        super(StreamLaneSegNetTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.points_num = points_num
        self.pts_dim = pts_dim
        self.fp16_enabled = False
        # self.init_layers()

    # def init_layers(self):
    #     self.reference_points = nn.Linear(self.embed_dims, self.pts_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, LaneAttention):
                m.init_weights()
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_embed,
                object_query_embed,
                reference_points,
                bev_h,
                bev_w,
                reg_branches=None,
                cls_branches=None,
                prop_query=None,
                prop_reference_points=None,
                prop_centerline_reference_points=None,
                is_first_frame_list=None,
                dn_query=None,
                dn_centerline_refer=None,
                dn_lane_refer=None,
                attn_masks=None,
                **kwargs):

        bs = mlvl_feats[0].size(0)

        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=2)
        query_pos = None

        # decoder
        if dn_query is not None:

            query = torch.cat((dn_query, query), 1).permute(1, 0, 2) # (num_q, bs, embed_dims)
            # reference_points = torch.cat((dn_refer, init_reference_out), 1)

        else:
            query = query.permute(1, 0, 2) # (num_q, bs, embed_dims)

        # query = query.permute(1, 0, 2) # (num_q, bs, embed_dims)
        if query_pos is not None:
            query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        init_states, inter_states, init_reference_out, init_inter_reference_points, inter_references, init_inter_lane_reference_points, inter_lane_reference = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            dn_centerline_reference_points = dn_centerline_refer,
            dn_lane_reference_points = dn_lane_refer,
            is_first_frame_list=is_first_frame_list,
            prop_query=prop_query,
            prop_reference_points=prop_reference_points,
            prop_centerline_reference_points=prop_centerline_reference_points,
            attn_masks=attn_masks,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return init_states, inter_states, init_reference_out, init_inter_reference_points, inter_references_out, init_inter_lane_reference_points, inter_lane_reference
