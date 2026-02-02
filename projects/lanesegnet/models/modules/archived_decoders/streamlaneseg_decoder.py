#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from mmdet.models.utils.transformer import inverse_sigmoid
import torch.nn as nn
import copy
import math
from mmcv.cnn import xavier_init
def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class StreamLaneSegNetDecoder(TransformerLayerSequence):

    def __init__(self, 
                 *args, 
                 return_intermediate=False, 
                 dn_query=0,
                 prop_add_stage=0,
                 pc_range=None, 
                 sample_idx=[0, 3, 6, 9], # num_ref_pts = len(sample_idx) * 2
                 pts_dim=3, 

                 **kwargs):
        super(StreamLaneSegNetDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.sample_idx = sample_idx
        self.pts_dim = pts_dim
        self.dn_query = dn_query
        self.prop_add_stage = prop_add_stage
        assert prop_add_stage >= 0  and prop_add_stage < 6
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        pt_pos_query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.pt_pos_query_projs = _get_clones(pt_pos_query_proj, self.num_layers)
        self.init_weights()

    def init_weights(self):

        for m in self.pt_pos_query_projs:
            xavier_init(m, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                *args,
                reference_points=None,
                dn_centerline_reference_points=None,
                dn_lane_reference_points=None,
                reg_branches=None,
                cls_branches=None,
                key_padding_mask=None,
                prop_query=None,
                prop_reference_points=None,
                prop_centerline_reference_points=None,
                is_first_frame_list=None,
                **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_lane_ref_points = []
        init_inter_reference_points = None
        init_inter_lane_reference_points = None
        num_queries, bs, embed_dims = query.shape
        value_bev = value[:, :bs, :]

        lane_ref_points = reference_points[:, :, self.sample_idx * 2, :] ##torch.Size([1, 200, 8, 3])

        init_reference_out = reference_points
        if self.dn_query:
            num_dn = num_queries - self.dn_query
        
        for lid, layer in enumerate(self.layers):

            if lid == self.prop_add_stage and prop_query is not None and prop_reference_points is not None:
                bs, topk, embed_dims = prop_query.shape

                output = output.permute(1, 0, 2) ##[200,1,256]->[1,200,256]
                
                with torch.no_grad():
                    tmp_scores, _ = cls_branches[lid](output).max(-1) # (bs, num_q)
                new_query = []
                new_refpts = []
                new_centerline_refpts = []

                for i in range(bs):
                    if is_first_frame_list[i]:
                        new_query.append(output[i])
                        new_refpts.append(lane_ref_points[i])
                        new_centerline_refpts.append(reference_points[i])

                    else:
                        _, valid_idx = torch.topk(tmp_scores[i], k=num_queries-topk, dim=-1) ###舍弃掉排名靠后的topk
                        new_query.append(torch.cat([prop_query[i], output[i][valid_idx]], dim=0))
                        new_refpts.append(torch.cat([prop_reference_points[i], lane_ref_points[i][valid_idx]], dim=0))
                        new_centerline_refpts.append(torch.cat([prop_centerline_reference_points[i],reference_points[i][valid_idx]], dim=0))

                lane_ref_points_stream_part = torch.stack(new_refpts) ##torch.Size([1, 200, 8, 3])
                output_stream_part = torch.stack(new_query).permute(1, 0, 2)
                reference_points_stream_part = torch.stack(new_centerline_refpts)

            if lid == self.prop_add_stage and prop_query is not None and prop_reference_points is not None:
                lane_ref_points = torch.cat((lane_ref_points, lane_ref_points_stream_part), dim = 0)
          
                output = torch.cat((output.permute(1, 0, 2), output_stream_part), dim = 1)
                reference_points = torch.cat((reference_points, reference_points_stream_part), dim = 0)

                init_inter_reference_points = reference_points
                init_inter_lane_reference_points = lane_ref_points
                value_bev = value

            lane_ref_points_reshape = lane_ref_points[..., :2] #torch.Size([1, 200, 8, 2])
            
            bs = lane_ref_points_reshape.shape[0]
            lane_ref_points_reshape = lane_ref_points_reshape.view(bs, -1, 2) #torch.Size([1, 14000, 2]
            query_sine_embed = gen_sineembed_for_position(lane_ref_points_reshape[..., :2])
            query_sine_embed = query_sine_embed.view(bs, -1, 8, 256) #torch.Size([1, 200, 8, 256])
            point_query_pos = self.pt_pos_query_projs[lid](query_sine_embed) #point_query_pos

            reference_points_input = lane_ref_points_reshape[..., :2].unsqueeze(2)
 
            output = layer(
                output,
                None,
                value_bev,
                *args,
                pt_query_pos=point_query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                query_key_padding_mask=None,
                **kwargs)
            
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                reg_center = reg_branches[0]
                reg_offset = reg_branches[1]

                tmp = reg_center[lid](output) #每一层的特征输入到回归层
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)

                assert reference_points.shape[-1] == self.pts_dim

                tmp = tmp + inverse_sigmoid(reference_points) #跟反sigmoid的采样点相加

                tmp = tmp.sigmoid()
                reference_points = tmp.detach() ##更新了reference_points。每一层加一个offset

                coord = tmp.clone()
                coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                if self.pts_dim == 3:
                    coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                centerline = coord.view(bs, num_query, -1).contiguous()

                offset = reg_offset[lid](output)
                left_laneline = centerline + offset
                right_laneline = centerline - offset
                left_laneline = left_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]
                right_laneline = right_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]

                lane_ref_points = torch.cat([left_laneline, right_laneline], axis=-2).contiguous().detach()

                lane_ref_points[..., 0] = (lane_ref_points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                lane_ref_points[..., 1] = (lane_ref_points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                if self.pts_dim == 3:
                    lane_ref_points[..., 2] = (lane_ref_points[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points.clone().detach())
                intermediate_lane_ref_points.append(lane_ref_points.clone().detach())

        if self.return_intermediate:
            return intermediate[0].unsqueeze(0), torch.stack(intermediate[1:]), init_reference_out, init_inter_reference_points, torch.stack(
                intermediate_reference_points[1:]), init_inter_lane_reference_points, torch.stack(
                intermediate_lane_ref_points[1:])

        return output, reference_points

