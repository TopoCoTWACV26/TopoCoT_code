#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class RelationshipHead(BaseModule):

    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=True,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25),
                 init_cfg=None):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)
        self.classifier = MLP(256, 256, 1, 3)
        self.loss_rel = build_loss(loss_rel)

    def forward_train(self, o1_feats, o1_assign_results, o2_feats, o2_assign_results, gt_adj):

        rel_pred = self.forward(o1_feats, o2_feats)
        losses = self.loss(rel_pred, gt_adj, o1_assign_results, o2_assign_results)
        return losses

    def forward_dn_train(self, o1_feats, o2_feats, gt_adj, dn_meta):

        #[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())] torch.Size([2, 82, 2]) torch.Size([132, 2])
        rel_pred = self.forward(o1_feats, o2_feats)
        losses = self.loss_dn(rel_pred, gt_adj, dn_meta)
        return losses
    
    def get_relationship(self, o1_feats, o2_feats):
        rel_pred = self.forward(o1_feats, o2_feats)
        rel_results = rel_pred.squeeze(-1).sigmoid()
        rel_results = [_ for _ in rel_results]
        return rel_results

    def forward(self, o1_feats, o2_feats):
        # feats: D, B, num_query, num_embedding
        o1_embeds = self.MLP_o1(o1_feats[-1]) #torch.Size([1, 200, 128])
        o2_embeds = self.MLP_o2(o2_feats[-1]) #torch.Size([1, 200, 128])
 
        num_query_o1 = o1_embeds.size(1) #200
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1) #torch.Size([1, 200, 200, 128])
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1) #torch.Size([1, 200, 200, 128])

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1) #torch.Size([1, 200, 200, 256])
        relationship_pred = self.classifier(relationship_tensor) #torch.Size([1, 200, 200, 1])
   
        return relationship_pred

    def loss(self, rel_preds, gt_adjs, o1_assign_results, o2_assign_results):
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results[-1]
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results[-1]
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        targets = []
        masked_rel_preds = []
        for i in range(B):
            gt_adj = gt_adjs[i]
            len_o1 = gt_adj.size(0)
            len_o2 = gt_adj.size(1)
            o1_pos_mask = o1_pos_assigned_gt_inds[i] < len_o1
            o2_pos_mask = o2_pos_assigned_gt_inds[i] < len_o2

            masked_rel_pred = rel_preds[i][o1_pos_inds[i]][:, o2_pos_inds[i]][o1_pos_mask][:, o2_pos_mask]
            masked_rel_preds.append(masked_rel_pred.view(-1, 1))

            target = gt_adj[o1_pos_assigned_gt_inds[i][o1_pos_mask]][:, o2_pos_assigned_gt_inds[i][o2_pos_mask]]
            targets.append(1 - target.view(-1).long())

        targets = torch.cat(targets, dim=0)
        rel_preds = torch.cat(masked_rel_preds, dim=0)
        if torch.numel(targets) == 0:
            return dict(loss_rel=rel_preds.sum() * 0)

        loss_rel = self.loss_rel(rel_preds, targets, avg_factor=targets.sum())

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_rel = torch.nan_to_num(loss_rel)

        return dict(loss_rel=loss_rel)

    def loss_dn(self, rel_preds, gt_adjs, dn_meta):
        B, num_query_o1, num_query_o2, _ = rel_preds.size()

        targets = []
        masked_rel_preds = []
        group = dn_meta['num_dn_group']
        num_dn_query = rel_preds.size(1)
        batch_indices = dn_meta['known_bid'].long()  # 批次索引 (size: [132])
        map_known_indice = dn_meta['map_known_indice'].long()  # 行索引 (size: [132])

        for i in range(B):
            batch_indice = torch.nonzero(batch_indices == i, as_tuple=True)
            pos_indice = map_known_indice[batch_indice]

            gt_lane_labels_3d = dn_meta['gt_lane_labels_3d'][i]
            gt_lane_labels_3d = gt_lane_labels_3d.repeat(group)
            non_ped_indice = torch.nonzero(gt_lane_labels_3d == 0, as_tuple=True)
            pos_indice = pos_indice[non_ped_indice]
            num_sample_group = int(len(pos_indice)/group)

            for group_idx in range(group):

                row_indices = pos_indice[num_sample_group*group_idx:num_sample_group*(group_idx+1)].unsqueeze(1).expand(-1, pos_indice[num_sample_group*group_idx:num_sample_group*(group_idx+1)].size(0))  # shape: [42, 42]
                col_indices = pos_indice[num_sample_group*group_idx:num_sample_group*(group_idx+1)].unsqueeze(0).expand(pos_indice[num_sample_group*group_idx:num_sample_group*(group_idx+1)].size(0), -1)  # shape: [42, 42]
                masked_rel_pred = rel_preds[i][row_indices,col_indices]
                masked_rel_preds.append(masked_rel_pred.view(-1, 1))

            gt_adj = gt_adjs[i]
            target = gt_adj
            targets.append((1 - target.view(-1).long()).repeat(group))

        targets = torch.cat(targets, dim=0)
        rel_preds = torch.cat(masked_rel_preds, dim=0)

        if torch.numel(targets) == 0:
            return dict(loss_rel=rel_preds.sum() * 0)

        loss_rel = self.loss_rel(rel_preds, targets, avg_factor=targets.sum())

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_rel = torch.nan_to_num(loss_rel)

        return dict(loss_dn_rel=loss_rel)

