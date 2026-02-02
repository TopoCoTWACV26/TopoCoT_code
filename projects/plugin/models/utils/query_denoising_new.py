# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox_xyxy_to_cxcywh
# from .utils import inverse_sigmoid
from mmdet.models.utils.transformer import inverse_sigmoid
import math
import numpy as np
import cv2
# from projects.lanesegnet.core.visualizer.lane_vis import draw_annotation_bev
import os
import mmcv
from shapely.geometry import LineString
def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)

def bbox_xyzxyz_to_cxcydzwhd(bbox):
    """Convert 3D bbox coordinates from (x1, y1, z1, x2, y2, z2) to (cx, cy, cz, w, h, d).

    Args:
        bbox (Tensor): Shape (n, 6) for bboxes.

    Returns:
        Tensor: Converted 3D bboxes.
    """
    x1, y1, z1, x2, y2, z2 = bbox.split((1, 1, 1, 1, 1, 1), dim=-1)
    bbox_new = [
        (x1 + x2) / 2,  # cx
        (y1 + y2) / 2,  # cy
        (z1 + z2) / 2,  # cz
        (x2 - x1),      # w
        (y2 - y1),      # h
        (z2 - z1),      # d
    ]
    return torch.cat(bbox_new, dim=-1)

def rotate_matrix(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.stack((np.stack((cos, -sin), 1), np.stack((sin, cos), 1)), 1)

class CdnQueryGenerator:
    def __init__(self,
                 hidden_dim=256,
                 num_classes=0,
                 num_queries=0,
                 noise_scale=dict(label=0.5, box=0.4, pt=0.0),
                 group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=None),
                 bev_h=200, bev_w=100,
                 pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                 voxel_size=[0.3, 0.3],
                 num_pts_per_vec=20,
                 rotate_range=0.0,
                 froze_class=None,
                 class_spesific=None,
                 noise_decay=False,
                 **kwargs):
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.label_noise_scale = noise_scale['label']
        self.box_noise_scale = noise_scale['box']
        self.pt_noise_scale = noise_scale['pt']
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, \
                'num_dn_queries should be set when using ' \
                'dynamic dn groups'
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using ' \
                'static dn groups'
            self.num_dn = group_cfg['num_groups']
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, \
            f'Expected the num in group_cfg to have type int. ' \
            f'Found {type(self.num_dn)} '
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.voxel_size = voxel_size
        self.num_pts_per_vec = num_pts_per_vec
        self.rotate_range = rotate_range
        self.froze_class = froze_class
        self.class_spesific = class_spesific
        self.noise_decay = noise_decay

    def get_num_groups(self, group_queries=None):
        """
        Args:
            group_queries (int): Number of dn queries in one group.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def __call__(self,
                 gt_bboxes,#####gt_bboxes [bs, num, 4] 4 is xyhw
                 gt_pts, #####gt_pts [bs, num, 20, 2] 4 is xyhw
                 gt_labels=None,  #####gt_labels [bs, num, 1]
                 label_enc=None, ###Embedding(3, 256) 3个类别 256 dim
                 prop_query_embedding=None, ##topk embeddings
                 noise_scale_list=None):
        """     

        Args:
            gt_bboxes (List[Tensor]): List of ground truth bboxes
                of the image, shape of each (num_gts, 4).
            gt_labels (List[Tensor]): List of ground truth labels
                of the image, shape of each (num_gts,), if None,

        Returns:
            TODO
        """
        if gt_labels is not None:
            assert len(gt_bboxes) == len(gt_labels), \
                f'the length of provided gt_labels ' \
                f'{len(gt_labels)} should be equal to' \
                f' that of gt_bboxes {len(gt_bboxes)}'

        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device

        # convert bbox
        gt_bboxes_list = []
        gt_pts_list = []
        loss_weight = []
        neglect_pos = []

        line_pos = []
        bound_pos = []
        ped_pos = []

        for label, bboxes, pts in zip(gt_labels, gt_bboxes, gt_pts):
            if self.froze_class is None:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3])).long())
            else:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3]) | (label!=self.froze_class)).long())  # 只计算某个类别的dn loss

            neglect_pos.append(((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3])).nonzero().squeeze(-1))

            pts_ = ((pts - bboxes[:, None, :2]) / (bboxes[:, None, 2:] - bboxes[:, None, :2])).clamp(min=0.0, max=1.0) #经过归一化的20个pts
            gt_pts_list.append(pts_)

            bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes) ##经过归一化的box
            gt_bboxes_list.append(bboxes_normalized)

            # 保存不同类别线的位置
            line_pos.append((label == 1).long())  ###判断label是line的位置就为1
            ped_pos.append((label == 0).long())
            bound_pos.append((label == 2).long())

        ###每一个batch有多少个线段，线段包括三种类别
        known = [torch.ones(b.shape[0]).int() for b in gt_bboxes]
        known_num = [sum(k) for k in known]

        num_groups = self.get_num_groups(int(max(known_num))) ##最大bs里线段有20个，60/20=3
        assert num_groups >= 1
        # print('num_groups',num_groups)
        unmask_bbox = unmask_label = torch.cat(known) ##将所有的cat在一起
        labels = torch.cat(gt_labels)
        boxes = torch.cat(gt_bboxes_list) ###[all box number, 4] xyhw?用box来表示
        # choice one: 
        pt = torch.cat(gt_pts_list) ### pt 是将所有gt pt聚合在一起

        batch_idx = torch.cat([torch.full_like(torch.ones(t.shape[0]).long(), i) for i, t in enumerate(gt_bboxes)]) ###每一个box属于哪个box id
        ##batch_idx = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        ##2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        known_indice = torch.nonzero(unmask_label + unmask_bbox) ##[48,1]内容是0~47
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(num_groups, 1).view(-1)### 4个bs 的 线段数量 x num_groups，复制num_groups次0~线段数量
        known_labels = labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_bid = batch_idx.repeat(num_groups, 1).view(-1) ###每一个box属于哪个box id复制num_groups次
        known_bboxs = boxes.repeat(num_groups, 1) ###[all box number * num_groups, 4]
        known_pts = pt.repeat(num_groups, 1, 1) ###[all box number * num_groups, ,20, 2] 20个点的x y
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale_list is not None:
            noise_scale_list = torch.cat(noise_scale_list).repeat(num_groups) ##noise_scale_list复制num_groups次

        # 
        if self.class_spesific is not None:
            line_pos = torch.cat(line_pos).repeat(num_groups) ###line的位置是1，复制num_groups次
            ped_pos = torch.cat(ped_pos).repeat(num_groups) ###ped_pos的位置是1，复制num_groups次
            bound_pos = torch.cat(bound_pos).repeat(num_groups) ###bound_pos的位置是1，复制num_groups次

        single_pad = int(max(known_num)) ###bs中最大instance的数量

        pad_size = int(single_pad * num_groups) ###bs中最大instance的数量 x num_groups
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2  ### x,y 减去 h w除以2 左下角坐标
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2 ### x,y 加上 h w除以2 右上角坐标

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2 ## h w 的一半
            diff[:, 2:] = known_bboxs[:, 2:] / 2 ## h w 的一半

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32) ###随机生成 0 1
            rand_sign = rand_sign * 2.0 - 1.0 ###随机生成 -1和 1
            rand_part = torch.rand_like(known_bboxs) ##随机为box xyhw生成randon值
            rand_part *= rand_sign ##随机为box xyhw生成加或者减randon值
            add = torch.mul(rand_sign, diff).to(device) ###加减 h w 的一半

            if self.class_spesific:
   
                noise = torch.mul(rand_part, diff).to(device)
                known_bbox_ += (noise*line_pos[:, None]*self.class_spesific[1] + noise*ped_pos[:, None]*self.class_spesific[0] + \
                                noise*bound_pos[:, None]*self.class_spesific[2])
            else:
                if self.noise_decay:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale * noise_scale_list[:, None]  ####对左下角和右上角的x,y进行关于h w一半*noise的变动，时序scale
                else:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            known_bbox_expand[:, :2] = \
                (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2 #左下角和右上角的平均，也就是中心坐标
            known_bbox_expand[:, 2:] = \
                known_bbox_[:, 2:] - known_bbox_[:, :2] #右上角减去左下角，得到的是宽高
        else:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

        if self.pt_noise_scale > 0: ##没用，对pts进行操作而不是对box进行操作
            rand_sign = (torch.rand_like(known_pts) * 2.0 - 1.0) / 20
            known_pts += rand_sign.to(device) * self.pt_noise_scale
            known_pts = known_pts.clamp(min=0.0, max=1.0)

        # Rotate
        if self.rotate_range > 0: ##没用 因为scale同时进行了缩放和旋转
            random_theta = (np.random.rand(known_bbox_.size(0)) * 2 - 1) * self.rotate_range * math.pi / 180
            R_matrix = rotate_matrix(random_theta)
            known_refers = (known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:] - known_bbox_expand[:, None, :2]).permute(0, 2, 1)
            known_refers = torch.bmm(torch.from_numpy(R_matrix).to(torch.float32).to(device), known_refers).permute(0, 2, 1)
            known_refers = known_refers + known_bbox_expand[:, None, :2]
        else:
            known_refers = known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:] ###左下角坐标 + 归一化pts x y坐标 * 宽高

        if self.label_noise_scale > 0: ###对label进行噪声
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes) ###变化的新标签
            known_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

        m = known_labels_expand.long().to(device)
        input_label_embed = label_enc(m) ## [每个bs中instance number的合 * group number,256]
        input_bbox_embed = known_bbox_expand ###加入过噪声的gt，box级别噪声
        padding_label = torch.zeros(pad_size, self.hidden_dim).to(device) #####[bs中最大instance的数量 x num_groups, 256]
        padding_bbox = torch.zeros(pad_size, 4).to(device) ###[bs中最大instance的数量 x num_groups, 4]
        padding_pts = torch.zeros(pad_size, self.num_pts_per_vec, 2).to(device) ###[bs中最大instance的数量 x num_groups, 20, 2]
        input_query_label = padding_label.repeat(batch_size, 1, 1) #torch.Size([bs, bs中最大instance的数量 x num_groups, 256])
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 4]
        input_query_pts = padding_pts.repeat(batch_size, 1, 1, 1)  ###[bs, bs中最大instance的数量 x num_groups, 20, 2]
        denoise_refers = padding_pts.repeat(batch_size, 1, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 20, 2]

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num): ##bs
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num]) ##每个bs 内的index 
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(num_groups)
            ]).long() ###根据一组的结果，再复制组别数，但是同一个bs里的index加上bs中最大instance数
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            #26, 27, 28, 29, 30, 31, 32, 33, 34, 20, 21, 22, 23, 24, 20, 21, 22, 23,
            #24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            #52, 53, 54, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 40, 41,
            #42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

        if len(known_bid):
            input_query_label[(known_bid.long(),map_known_indice)] = input_label_embed ##input_query_label[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 256] 放入噪声label embed
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 256])
            input_query_bbox[(known_bid.long(),map_known_indice)] = input_bbox_embed ##input_query_bbox[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 4] 放入噪声box
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 4])
            input_query_pts[(known_bid.long(),map_known_indice)] = known_pts##input_query_pts[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 20, 2] 放入噪声pts 好像没变化
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 4])
            denoise_refers[(known_bid.long(),map_known_indice)] = known_refers ##去归一化的噪声pts，经过了noise box处理 ##denoise_refers[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 20, 2] 放入噪声pts 好像没变化
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 4])

        if prop_query_embedding is not None:
            tgt_size = pad_size + self.num_queries + prop_query_embedding.size(1)
        else:
            tgt_size = pad_size + self.num_queries  ###[100+bs中最大instance的数量 x num_groups]
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0 ###[160,160] are all False
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True ##[60:,:60]

        # reconstruct cannot see each other
        for i in range(num_groups):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True ###[0:20,20:60] = True
            if i == num_groups - 1:
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True ###[40:60,:40] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True
        # import matplotlib.pyplot as plt
        # # 将PyTorch张量转换为numpy数组
        # attn_mask_np = attn_mask.cpu().numpy()

        # # 创建图像
        # plt.figure(figsize=(10, 10))

        # # 绘制二值图，true为黑色，false为白色
        # plt.imshow(attn_mask_np, cmap='gray')

        # # 设置x和y轴的刻度位置
        # plt.xticks(np.arange(0, 160, 10))
        # plt.yticks(np.arange(0, 160, 10))

        # # 保存图像
        # plt.savefig('./attn_mask.png')
        # import pdb; pdb.set_trace()
        dn_meta = {
            'pad_size': pad_size, ###bs中最大instance的数量 x num_groups
            'num_dn_group': num_groups,
            # 'post_dn': post_dn,
            'known_bid': known_bid.long(), ###num_groups x 指示每个instance属于哪个bs
            'map_known_indice': map_known_indice, ###根据一组的结果，再复制组别数，但是同一个bs里的index加上bs中最大instance数
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            #26, 27, 28, 29, 30, 31, 32, 33, 34, 20, 21, 22, 23, 24, 20, 21, 22, 23,
            #24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            #52, 53, 54, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 40, 41,
            #42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
            'loss_weight': loss_weight, ###bs x per_instance 
        }

        # 去掉完全的直线，分母为0
        for i, pos in enumerate(neglect_pos):
            if len(pos) != 0:
                for j in range(num_groups):
                    denoise_refers[i][pos+single_pad * j] = gt_pts[i][pos]
        # import pdb; pdb.set_trace()
        return input_query_label, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers

class LanednQueryGenerator_old:
    def __init__(self,
                 hidden_dim=256,
                 num_classes=0,
                 num_queries=0,
                 noise_scale=dict(label=0.5, box=0.4, pt=0.0),
                 group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=None),
                 bev_h=200, bev_w=100,
                 pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                 voxel_size=[0.3, 0.3],
                 num_pts_per_vec=20,
                 rotate_range=0.0,
                 froze_class=None,
                 class_spesific=None,
                 noise_decay=False,
                 **kwargs):
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.label_noise_scale = noise_scale['label']
        self.box_noise_scale = noise_scale['box']
        self.pt_noise_scale = noise_scale['pt']
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, \
                'num_dn_queries should be set when using ' \
                'dynamic dn groups'
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using ' \
                'static dn groups'
            self.num_dn = group_cfg['num_groups']
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, \
            f'Expected the num in group_cfg to have type int. ' \
            f'Found {type(self.num_dn)} '
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.voxel_size = voxel_size
        self.num_pts_per_vec = num_pts_per_vec
        self.rotate_range = rotate_range
        self.froze_class = froze_class
        self.class_spesific = class_spesific
        self.noise_decay = noise_decay

    def get_num_groups(self, group_queries=None):
        """
        Args:
            group_queries (int): Number of dn queries in one group.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def __call__(self,
                 gt_bboxes,#####gt_bboxes [bs, num, 4] 4 is xyhw
                 gt_pts, #####gt_pts [bs, num, 20, 2] 4 is xyhw
                 gt_labels=None,  #####gt_labels [bs, num, 1]
                 gt_lane_left_type = None,
                 gt_lane_right_type = None,
                 label_enc=None, ###Embedding(3, 256) 3个类别 256 dim
                 left_label_enc = None,
                 right_label_enc = None,
                 prop_query_embedding=None, ##topk embeddings
                 noise_scale_list=None):
        """     

        Args:
            gt_bboxes (List[Tensor]): List of ground truth bboxes
                of the image, shape of each (num_gts, 4).
            gt_labels (List[Tensor]): List of ground truth labels
                of the image, shape of each (num_gts,), if None,

        Returns:
            TODO
        """
        if gt_labels is not None:
            assert len(gt_bboxes) == len(gt_labels), \
                f'the length of provided gt_labels ' \
                f'{len(gt_labels)} should be equal to' \
                f' that of gt_bboxes {len(gt_bboxes)}'

        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device

        # convert bbox
        gt_bboxes_list = []
        gt_pts_list = []
        loss_weight = []
        neglect_pos = []

        line_pos = []
        bound_pos = []
        ped_pos = []

        for label, bboxes, pts in zip(gt_labels, gt_bboxes, gt_pts):

            if self.froze_class is None:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 3]) | (bboxes[:, 1]==bboxes[:, 4]) | (bboxes[:, 2]==bboxes[:, 5]) ).long())
            else:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 3]) | (bboxes[:, 1]==bboxes[:, 4]) | (bboxes[:, 2]==bboxes[:, 5]) | (label!=self.froze_class)).long())  # 只计算某个类别的dn loss

            neglect_pos.append(((bboxes[:, 0]==bboxes[:, 3]) | (bboxes[:, 1]==bboxes[:, 4]) | (bboxes[:, 2]==bboxes[:, 5]) ).nonzero().squeeze(-1))

            pts_ = ((pts - bboxes[:, None, :3]) / (bboxes[:, None, 3:] - bboxes[:, None, :3])).clamp(min=0.0, max=1.0) #经过归一化的20个pts
            gt_pts_list.append(pts_)

            bboxes_normalized = bbox_xyzxyz_to_cxcydzwhd(bboxes) ##经过归一化的box
            gt_bboxes_list.append(bboxes_normalized)

            # 保存不同类别线的位置
            line_pos.append((label == 0).long())  ###判断label是line的位置就为1
            ped_pos.append((label == 1).long())

        ###每一个batch有多少个线段，线段包括三种类别
        known = [torch.ones(b.shape[0]).int() for b in gt_bboxes]
        known_num = [sum(k) for k in known]

        num_groups = self.get_num_groups(int(max(known_num))) ##最大bs里线段有20个，60/20=3
        assert num_groups >= 1
        # print('num_groups',num_groups)
        unmask_bbox = unmask_label = torch.cat(known) ##将所有的cat在一起
        labels = torch.cat(gt_labels)
        left_type_labels = torch.cat(gt_lane_left_type) ###有三种
        right_type_labels = torch.cat(gt_lane_right_type)
        boxes = torch.cat(gt_bboxes_list) ###[all box number, 4] xyhw?用box来表示
        # choice one: 
        pt = torch.cat(gt_pts_list) ### pt 是将所有gt pt聚合在一起

        batch_idx = torch.cat([torch.full_like(torch.ones(t.shape[0]).long(), i) for i, t in enumerate(gt_bboxes)]) ###每一个box属于哪个box id
        ##batch_idx = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        ##2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        known_indice = torch.nonzero(unmask_label + unmask_bbox) ##[48,1]内容是0~47
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(num_groups, 1).view(-1)### 4个bs 的 线段数量 x num_groups，复制num_groups次0~线段数量
        known_labels = labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_left_type_labels = left_type_labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_right_type_labels = right_type_labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_bid = batch_idx.repeat(num_groups, 1).view(-1) ###每一个box属于哪个box id复制num_groups次
        known_bboxs = boxes.repeat(num_groups, 1) ###[all box number * num_groups, 4]
        known_pts = pt.repeat(num_groups, 1, 1) ###[all box number * num_groups, ,20, 2] 20个点的x y
        known_labels_expand = known_labels.clone()
        known_left_type_labels_expand = known_left_type_labels.clone()
        known_right_type_labels_expand = known_right_type_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale_list is not None:
            noise_scale_list = torch.cat(noise_scale_list).repeat(num_groups) ##noise_scale_list复制num_groups次

        # 
        if self.class_spesific is not None:
            line_pos = torch.cat(line_pos).repeat(num_groups) ###line的位置是1，复制num_groups次
            ped_pos = torch.cat(ped_pos).repeat(num_groups) ###ped_pos的位置是1，复制num_groups次

        single_pad = int(max(known_num)) ###bs中最大instance的数量

        pad_size = int(single_pad * num_groups) ###bs中最大instance的数量 x num_groups
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 3] = \
                known_bboxs[:, : 3] - known_bboxs[:, 3:] / 2  ### x,y 减去 h w除以2 左下角坐标
            known_bbox_[:, 3:] = \
                known_bboxs[:, :3] + known_bboxs[:, 3:] / 2 ### x,y 加上 h w除以2 右上角坐标

            diff = torch.zeros_like(known_bboxs)
            diff[:, :3] = known_bboxs[:, 3:] / 2 ## h w 的一半
            diff[:, 3:] = known_bboxs[:, 3:] / 2 ## h w 的一半

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32) ###随机生成 0 1
            rand_sign = rand_sign * 2.0 - 1.0 ###随机生成 -1和 1
            rand_part = torch.rand_like(known_bboxs) ##随机为box xyhw生成randon值
            rand_part *= rand_sign ##随机为box xyhw生成加或者减randon值
            add = torch.mul(rand_sign, diff).to(device) ###加减 h w 的一半

            if self.class_spesific:
   
                noise = torch.mul(rand_part, diff).to(device)
                known_bbox_ += (noise*line_pos[:, None]*self.class_spesific[1] + noise*ped_pos[:, None]*self.class_spesific[0] + \
                                noise*bound_pos[:, None]*self.class_spesific[2])
            else:
                if self.noise_decay:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale * noise_scale_list[:, None]  ####对左下角和右上角的x,y进行关于h w一半*noise的变动，时序scale
                else:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            known_bbox_expand[:, :3] = \
                (known_bbox_[:, :3] + known_bbox_[:, 3:]) / 2 #左下角和右上角的平均，也就是中心坐标
            known_bbox_expand[:, 3:] = \
                known_bbox_[:, 3:] - known_bbox_[:, :3] #右上角减去左下角，得到的是宽高
        else:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 3] = \
                known_bboxs[:, : 3] - known_bboxs[:, 3:] / 2
            known_bbox_[:, 3:] = \
                known_bboxs[:, :3] + known_bboxs[:, 3:] / 2

        if self.pt_noise_scale > 0: ##没用，对pts进行操作而不是对box进行操作
            rand_sign = (torch.rand_like(known_pts) * 2.0 - 1.0) / 20
            known_pts += rand_sign.to(device) * self.pt_noise_scale
            known_pts = known_pts.clamp(min=0.0, max=1.0)

        # Rotate
        if self.rotate_range > 0: ##没用 因为scale同时进行了缩放和旋转
            random_theta = (np.random.rand(known_bbox_.size(0)) * 2 - 1) * self.rotate_range * math.pi / 180
            R_matrix = rotate_matrix(random_theta)
            known_refers = (known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:] - known_bbox_expand[:, None, :2]).permute(0, 2, 1)
            known_refers = torch.bmm(torch.from_numpy(R_matrix).to(torch.float32).to(device), known_refers).permute(0, 2, 1)
            known_refers = known_refers + known_bbox_expand[:, None, :2]
        else:
            known_refers = known_bbox_[:, None, :3] + known_pts * known_bbox_expand[:, None, 3:] ###左下角坐标 + 归一化pts x y坐标 * 宽高

        if self.label_noise_scale > 0: ###对label进行噪声
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes) ###变化的新标签
            known_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

            p = torch.rand_like(known_left_type_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes+1) ###变化的新标签
            known_left_type_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

            p = torch.rand_like(known_right_type_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes+1) ###变化的新标签
            known_right_type_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

        known_labels_expand_long = known_labels_expand.long().to(device)
        input_label_embed = label_enc(known_labels_expand_long) ## [每个bs中instance number的合 * group number,256]

        known_left_type_labels_expand_long = known_left_type_labels_expand.long().to(device)
        input_left_type_label_embed = left_label_enc(known_left_type_labels_expand_long) ## [每个bs中instance number的合 * group number,256]

        known_right_type_labels_expand_long = known_right_type_labels_expand.long().to(device)
        input_right_type_label_embed = right_label_enc(known_right_type_labels_expand_long) ## [每个bs中instance number的合 * group number,256]

        input_bbox_embed = known_bbox_expand ###加入过噪声的gt，box级别噪声
        padding_label = torch.zeros(pad_size, self.hidden_dim//2).to(device) #####[bs中最大instance的数量 x num_groups, 256]
        padding_bbox = torch.zeros(pad_size, 6).to(device) ###[bs中最大instance的数量 x num_groups, 4]
        padding_pts = torch.zeros(pad_size, self.num_pts_per_vec*3, 3).to(device) ###[bs中最大instance的数量 x num_groups, 20, 2]
        input_query_label = padding_label.repeat(batch_size, 1, 1) #torch.Size([bs, bs中最大instance的数量 x num_groups, 256])
        input_query_left_type_label = padding_label.repeat(batch_size, 1, 1)
        input_query_right_type_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 4]
        input_query_pts = padding_pts.repeat(batch_size, 1, 1, 1)  ###[bs, bs中最大instance的数量 x num_groups, 20, 2]
        denoise_refers = padding_pts.repeat(batch_size, 1, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 20, 2]

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num): ##bs
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num]) ##每个bs 内的index 
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(num_groups)
            ]).long() ###根据一组的结果，再复制组别数，但是同一个bs里的index加上bs中最大instance数
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            #26, 27, 28, 29, 30, 31, 32, 33, 34, 20, 21, 22, 23, 24, 20, 21, 22, 23,
            #24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            #52, 53, 54, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 40, 41,
            #42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

        if len(known_bid):
            input_query_label[(known_bid.long(),map_known_indice)] = input_label_embed ##input_query_label[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 256] 放入噪声label embed
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 256])
            input_query_left_type_label[(known_bid.long(),map_known_indice)] = input_left_type_label_embed
            input_query_right_type_label[(known_bid.long(),map_known_indice)] = input_right_type_label_embed

            input_query_bbox[(known_bid.long(),map_known_indice)] = input_bbox_embed ##input_query_bbox[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 4] 放入噪声box
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 4])
            input_query_pts[(known_bid.long(),map_known_indice)] = known_pts##input_query_pts[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 20, 2] 放入噪声pts 好像没变化
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 4])
            denoise_refers[(known_bid.long(),map_known_indice)] = known_refers ##去归一化的噪声pts，经过了noise box处理 ##denoise_refers[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 20, 2] 放入噪声pts 好像没变化
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 4])

        if prop_query_embedding is not None:
            tgt_size = pad_size + self.num_queries + prop_query_embedding.size(1)
        else:
            tgt_size = pad_size + self.num_queries  ###[100+bs中最大instance的数量 x num_groups]
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0 ###[160,160] are all False
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True ##[60:,:60]

        # reconstruct cannot see each other
        for i in range(num_groups):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True ###[0:20,20:60] = True
            if i == num_groups - 1:
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True ###[40:60,:40] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True
        # import matplotlib.pyplot as plt
        # # 将PyTorch张量转换为numpy数组
        # attn_mask_np = attn_mask.cpu().numpy()

        # # 创建图像
        # plt.figure(figsize=(10, 10))

        # # 绘制二值图，true为黑色，false为白色
        # plt.imshow(attn_mask_np, cmap='gray')

        # # 设置x和y轴的刻度位置
        # plt.xticks(np.arange(0, 160, 10))
        # plt.yticks(np.arange(0, 160, 10))

        # # 保存图像
        # plt.savefig('./attn_mask.png')
        # import pdb; pdb.set_trace()
        dn_meta = {
            'pad_size': pad_size, ###bs中最大instance的数量 x num_groups
            'num_dn_group': num_groups,
            # 'post_dn': post_dn,
            'known_bid': known_bid.long(), ###num_groups x 指示每个instance属于哪个bs
            'map_known_indice': map_known_indice, ###根据一组的结果，再复制组别数，但是同一个bs里的index加上bs中最大instance数
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            #26, 27, 28, 29, 30, 31, 32, 33, 34, 20, 21, 22, 23, 24, 20, 21, 22, 23,
            #24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            #52, 53, 54, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 40, 41,
            #42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
            'loss_weight': loss_weight, ###bs x per_instance 
        }

        import pdb; pdb.set_trace()
        # 去掉完全的直线，分母为0
        for i, pos in enumerate(neglect_pos):
            if len(pos) != 0:
                for j in range(num_groups):
                    denoise_refers[i][pos+single_pad * j] = gt_pts[i][pos]
        # import pdb; pdb.set_trace()
        return input_query_label, input_query_left_type_label, input_query_right_type_label, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers

class LanednQueryGenerator:
    def __init__(self,
                 hidden_dim=256,
                 num_classes=0,
                 num_queries=0,
                 noise_scale=dict(label=0.5, box=0.4, pt=0.0),
                 group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=None),
                 bev_h=200, bev_w=100,
                 pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                 voxel_size=[0.3, 0.3],
                 num_pts_per_vec=20,
                 rotate_range=0.0,
                 froze_class=None,
                 class_spesific=None,
                 noise_decay=False,
                 map_size=None,
                 **kwargs):
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.label_noise_scale = noise_scale['label']
        self.box_noise_scale = noise_scale['box']
        self.pt_noise_scale = noise_scale['pt']
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, \
                'num_dn_queries should be set when using ' \
                'dynamic dn groups'
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using ' \
                'static dn groups'
            self.num_dn = group_cfg['num_groups']
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, \
            f'Expected the num in group_cfg to have type int. ' \
            f'Found {type(self.num_dn)} '
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.map_size = map_size
        self.voxel_size = voxel_size
        self.num_pts_per_vec = num_pts_per_vec
        self.rotate_range = rotate_range
        self.froze_class = froze_class
        self.class_spesific = class_spesific
        self.noise_decay = noise_decay

    def get_num_groups(self, group_queries=None):
        """
        Args:
            group_queries (int): Number of dn queries in one group.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_mask(self, gt_lanes_3d):
        device = gt_lanes_3d.device
        gt_lanes = gt_lanes_3d.cpu().numpy().reshape(-1, 3, 10, 3)
        gt_left_lines = gt_lanes[:, 1]
        gt_right_lines = gt_lanes[:, 2]

        origin = np.array([self.bev_w // 2, self.bev_h // 2])
        scale = np.array([self.bev_w / (self.map_size[2] - self.map_size[0]), self.bev_h / (self.map_size[3] - self.map_size[1])])

        inst_masks = []
        for idx, (left_line, right_line) in enumerate(zip(gt_left_lines, gt_right_lines)):

            segment_boundary = np.concatenate((left_line, right_line[::-1], left_line[0:1]), axis=0)
            mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)

            draw_coor = (segment_boundary[:, :2] * scale + origin).astype(np.int32)
            mask = cv2.fillPoly(mask, [draw_coor], 255)
            bitMask = (mask / 255)
            bitMask = torch.tensor(bitMask, dtype=torch.float32, device=device)
            inst_masks.append(bitMask)

        inst_masks = torch.stack(inst_masks).to(device).to(torch.float32)
        return inst_masks

    def __call__(self,
                 gt_bboxes,#####gt_bboxes [bs, num, 4] 4 is xyhw
                 gt_pts, #####gt_pts [bs, num, 20, 2] 4 is xyhw
                 gt_labels=None,  #####gt_labels [bs, num, 1]
                 gt_lane_left_type = None,
                 gt_lane_right_type = None,
                 label_enc=None, ###Embedding(3, 256) 3个类别 256 dim
                 left_label_enc = None,
                 right_label_enc = None,
                 prop_query_embedding=None, ##topk embeddings
                 origin=None,
                 roi_size=None,
                 noise_scale_list=None):
        """     

        Args:
            gt_bboxes (List[Tensor]): List of ground truth bboxes
                of the image, shape of each (num_gts, 4).
            gt_labels (List[Tensor]): List of ground truth labels
                of the image, shape of each (num_gts,), if None,

        Returns:
            TODO
        """
        if gt_labels is not None:
            assert len(gt_bboxes) == len(gt_labels), \
                f'the length of provided gt_labels ' \
                f'{len(gt_labels)} should be equal to' \
                f' that of gt_bboxes {len(gt_bboxes)}'

        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device

        # convert bbox
        gt_bboxes_list = []
        gt_pts_list = []
        loss_weight = []
        neglect_pos = []

        line_pos = []
        bound_pos = []
        ped_pos = []

        for label, bboxes, pts in zip(gt_labels, gt_bboxes, gt_pts):

            if self.froze_class is None:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 3]) | (bboxes[:, 1]==bboxes[:, 4]) | (bboxes[:, 2]==bboxes[:, 5]) ).long())
            else:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 3]) | (bboxes[:, 1]==bboxes[:, 4]) | (bboxes[:, 2]==bboxes[:, 5]) | (label!=self.froze_class)).long())  # 只计算某个类别的dn loss

            neglect_pos.append(((bboxes[:, 0]==bboxes[:, 3]) | (bboxes[:, 1]==bboxes[:, 4]) | (bboxes[:, 2]==bboxes[:, 5]) ).nonzero().squeeze(-1))

            pts_ = ((pts - bboxes[:, None, :3]) / (bboxes[:, None, 3:] - bboxes[:, None, :3])).clamp(min=0.0, max=1.0) #经过归一化的20个pts
            gt_pts_list.append(pts_)

            bboxes_normalized = bbox_xyzxyz_to_cxcydzwhd(bboxes) ##经过归一化的box
            gt_bboxes_list.append(bboxes_normalized)

            # 保存不同类别线的位置
            line_pos.append((label == 0).long())  ###判断label是line的位置就为1
            ped_pos.append((label == 1).long())

        ###每一个batch有多少个线段，线段包括三种类别
        known = [torch.ones(b.shape[0]).int() for b in gt_bboxes]
        known_num = [sum(k) for k in known]

        num_groups = self.get_num_groups(int(max(known_num))) ##最大bs里线段有20个，60/20=3
        assert num_groups >= 1

        unmask_bbox = unmask_label = torch.cat(known) ##将所有的cat在一起
        labels = torch.cat(gt_labels)
        left_type_labels = torch.cat(gt_lane_left_type) ###有三种
        right_type_labels = torch.cat(gt_lane_right_type)
        boxes = torch.cat(gt_bboxes_list) ###[all box number, 4] xyhw?用box来表示
        # choice one: 
        pt = torch.cat(gt_pts_list) ### pt 是将所有gt pt聚合在一起

        batch_idx = torch.cat([torch.full_like(torch.ones(t.shape[0]).long(), i) for i, t in enumerate(gt_bboxes)]) ###每一个box属于哪个box id
        ##batch_idx = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        ##2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        known_indice = torch.nonzero(unmask_label + unmask_bbox) ##[48,1]内容是0~47
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(num_groups, 1).view(-1)### 4个bs 的 线段数量 x num_groups，复制num_groups次0~线段数量
        known_labels = labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_left_type_labels = left_type_labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_right_type_labels = right_type_labels.repeat(num_groups, 1).view(-1) ###labels复制num_groups次label
        known_bid = batch_idx.repeat(num_groups, 1).view(-1) ###每一个box属于哪个box id复制num_groups次
        known_bboxs = boxes.repeat(num_groups, 1) ###[all box number * num_groups, 4]
        known_pts = pt.repeat(num_groups, 1, 1) ###[all box number * num_groups, ,20, 2] 20个点的x y
        known_labels_expand = known_labels.clone()
        known_left_type_labels_expand = known_left_type_labels.clone()
        known_right_type_labels_expand = known_right_type_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale_list is not None:
            noise_scale_list = torch.cat(noise_scale_list).repeat(num_groups) ##noise_scale_list复制num_groups次

        # 
        if self.class_spesific is not None:
            line_pos = torch.cat(line_pos).repeat(num_groups) ###line的位置是1，复制num_groups次
            ped_pos = torch.cat(ped_pos).repeat(num_groups) ###ped_pos的位置是1，复制num_groups次

        single_pad = int(max(known_num)) ###bs中最大instance的数量

        pad_size = int(single_pad * num_groups) ###bs中最大instance的数量 x num_groups
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 3] = \
                known_bboxs[:, : 3] - known_bboxs[:, 3:] / 2  ### x,y 减去 h w除以2 左下角坐标
            known_bbox_[:, 3:] = \
                known_bboxs[:, :3] + known_bboxs[:, 3:] / 2 ### x,y 加上 h w除以2 右上角坐标

            diff = torch.zeros_like(known_bboxs)
            diff[:, :3] = known_bboxs[:, 3:] / 2 ## h w 的一半
            diff[:, 3:] = known_bboxs[:, 3:] / 2 ## h w 的一半

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32) ###随机生成 0 1
            rand_sign = rand_sign * 2.0 - 1.0 ###随机生成 -1和 1
            rand_part = torch.rand_like(known_bboxs) ##随机为box xyhw生成randon值
            rand_part *= rand_sign ##随机为box xyhw生成加或者减randon值
            add = torch.mul(rand_sign, diff).to(device) ###加减 h w 的一半

            if self.class_spesific:
   
                noise = torch.mul(rand_part, diff).to(device)
                known_bbox_ += (noise*line_pos[:, None]*self.class_spesific[1] + noise*ped_pos[:, None]*self.class_spesific[0] + \
                                noise*bound_pos[:, None]*self.class_spesific[2])
            else:
                if self.noise_decay:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale * noise_scale_list[:, None]  ####对左下角和右上角的x,y进行关于h w一半*noise的变动，时序scale
                else:
      
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale

            # known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            known_bbox_expand[:, :3] = \
                (known_bbox_[:, :3] + known_bbox_[:, 3:]) / 2 #左下角和右上角的平均，也就是中心坐标
            known_bbox_expand[:, 3:] = \
                known_bbox_[:, 3:] - known_bbox_[:, :3] #右上角减去左下角，得到的是宽高
        else:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 3] = \
                known_bboxs[:, : 3] - known_bboxs[:, 3:] / 2
            known_bbox_[:, 3:] = \
                known_bboxs[:, :3] + known_bboxs[:, 3:] / 2

        if self.pt_noise_scale > 0: ##没用，对pts进行操作而不是对box进行操作
            rand_sign = (torch.rand_like(known_pts) * 2.0 - 1.0) / 20
            known_pts += rand_sign.to(device) * self.pt_noise_scale
            known_pts = known_pts.clamp(min=0.0, max=1.0)

        # Rotate
        if self.rotate_range > 0: ##没用 因为scale同时进行了缩放和旋转
            random_theta = (np.random.rand(known_bbox_.size(0)) * 2 - 1) * self.rotate_range * math.pi / 180
            R_matrix = rotate_matrix(random_theta)
            known_refers = (known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:] - known_bbox_expand[:, None, :2]).permute(0, 2, 1)
            known_refers = torch.bmm(torch.from_numpy(R_matrix).to(torch.float32).to(device), known_refers).permute(0, 2, 1)
            known_refers = known_refers + known_bbox_expand[:, None, :2]
        else:
            
            known_refers = known_bbox_[:, None, :3] + known_pts * known_bbox_expand[:, None, 3:] ###左下角坐标 + 归一化pts x y坐标 * 宽高

        ###known_pts known_refers
        # lane_pre = [lane[:10,:3].cpu().detach().numpy() for lane in known_refers[:gt_pts[0].shape[0],:,:]]
        # lane_target = [lane[:10,:3].cpu().detach().numpy() for lane in gt_pts[0]]

        # conn_img_pre= draw_annotation_bev(lane_pre, with_laneline=False, with_area=False)
        # conn_img_target = draw_annotation_bev(lane_target, with_laneline=False, with_area=False)
        # divider = np.ones((conn_img_pre.shape[0], 7, 3), dtype=np.uint8) * 128
        # conn_img = np.concatenate([conn_img_pre, divider, conn_img_target], axis=1)[..., ::-1]

        # import os
        # out_dir = f"./vis/streamvis_1/"
        # output_path = os.path.join(out_dir, 'noise.jpg')
        # mmcv.imwrite(conn_img, output_path)

        if self.label_noise_scale > 0: ###对label进行噪声
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes) ###变化的新标签
            known_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

            p = torch.rand_like(known_left_type_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes+1) ###变化的新标签
            known_left_type_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

            p = torch.rand_like(known_right_type_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1) ###选中一些样本进行标签变化
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes+1) ###变化的新标签
            known_right_type_labels_expand.scatter_(0, chosen_indice, new_label) ###变化的新标签进行替换

        known_labels_expand_long = known_labels_expand.long().to(device)
        input_label_embed = label_enc(known_labels_expand_long) ## [每个bs中instance number的合 * group number,256]

        known_left_type_labels_expand_long = known_left_type_labels_expand.long().to(device)
        input_left_type_label_embed = left_label_enc(known_left_type_labels_expand_long) ## [每个bs中instance number的合 * group number,256]

        known_right_type_labels_expand_long = known_right_type_labels_expand.long().to(device)
        input_right_type_label_embed = right_label_enc(known_right_type_labels_expand_long) ## [每个bs中instance number的合 * group number,256]

        input_bbox_embed = known_bbox_expand ###加入过噪声的gt，box级别噪声
        padding_label = torch.zeros(pad_size, self.hidden_dim//2).to(device) #####[bs中最大instance的数量 x num_groups, 256]
        padding_bbox = torch.zeros(pad_size, 6).to(device) ###[bs中最大instance的数量 x num_groups, 4]
        padding__centerline_pts = torch.zeros(pad_size, self.num_pts_per_vec, 3).to(device) ###[bs中最大instance的数量 x num_groups, 20, 2]
        padding__lane_pts = torch.zeros(pad_size, self.num_pts_per_vec*2, 3).to(device) ###[bs中最大instance的数量 x num_groups, 20, 2]
        padding__mask = torch.zeros(pad_size, 100, 200).to(device) ###[bs中最大instance的数量 x num_groups, 20, 2]
    
        input_query_label = padding_label.repeat(batch_size, 1, 1) #torch.Size([bs, bs中最大instance的数量 x num_groups, 256])
        input_query_left_type_label = padding_label.repeat(batch_size, 1, 1)
        input_query_right_type_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 4]
        input_query_mask = padding__mask.repeat(batch_size, 1, 1, 1)

        denoise_centerline_refers = padding__centerline_pts.repeat(batch_size, 1, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 20, 2]
        denoise_centerline_refers_pe =  padding__centerline_pts.repeat(batch_size, 1, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 20, 2]
        denoise_lane_refers = padding__lane_pts.repeat(batch_size, 1, 1, 1) ###[bs, bs中最大instance的数量 x num_groups, 20, 2]

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num): ##bs
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num]) ##每个bs 内的index 
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(num_groups)
            ]).long() ###根据一组的结果，再复制组别数，但是同一个bs里的index加上bs中最大instance数
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            #26, 27, 28, 29, 30, 31, 32, 33, 34, 20, 21, 22, 23, 24, 20, 21, 22, 23,
            #24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            #52, 53, 54, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 40, 41,
            #42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

        if len(known_bid):
            input_query_label[(known_bid.long(),map_known_indice)] = input_label_embed ##input_query_label[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 256] 放入噪声label embed
            #input_query_label : #torch.Size([bs, bs中最大instance的数量 x num_groups, 256])
            input_query_left_type_label[(known_bid.long(),map_known_indice)] = input_left_type_label_embed
            input_query_right_type_label[(known_bid.long(),map_known_indice)] = input_right_type_label_embed

            input_query_bbox[(known_bid.long(),map_known_indice)] = input_bbox_embed ##input_query_bbox[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 4] 放入噪声box
            # 

            known_centerline_refers_sigmod = (known_refers[:,:10,:3]- origin) / roi_size   ###应该归一化后sigmoid
            # known_centerline_refers = inverse_sigmoid(known_centerline_refers_sigmod) ###要不要反sigmoid，反的话有负数

            known_lane_refers = (known_refers[:,10:,:3]- origin) / roi_size

            denoise_centerline_refers[(known_bid.long(),map_known_indice)] = known_centerline_refers_sigmod ##去归一化的噪声pts，经过了noise box处理 ##denoise_refers[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 20, 2] 放入噪声pts 好像没变化
            denoise_centerline_refers_pe[(known_bid.long(),map_known_indice)] = known_centerline_refers_sigmod #known_centerline_refers
            denoise_lane_refers[(known_bid.long(),map_known_indice)] = known_lane_refers ##去归一化的噪声pts，经过了noise box处理 ##denoise_refers[(known_bid.long(),map_known_indice)] : [bs中最大instance的数量 x num_groups, 20, 2] 放入噪声pts 好像没变化

        if prop_query_embedding is not None:
            tgt_size = pad_size + self.num_queries + prop_query_embedding.size(1)
        else:
            tgt_size = pad_size + self.num_queries  ###[100+bs中最大instance的数量 x num_groups]
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0 ###[160,160] are all False
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True ##[60:,:60]

        # reconstruct cannot see each other
        for i in range(num_groups):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True ###[0:20,20:60] = True
            if i == num_groups - 1:
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True ###[40:60,:40] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True
                
        # import matplotlib.pyplot as plt
        # # 将PyTorch张量转换为numpy数组
        # attn_mask_np = attn_mask.cpu().numpy()

        # # 创建图像
        # plt.figure(figsize=(10, 10))

        # # 绘制二值图，true为黑色，false为白色
        # plt.imshow(attn_mask_np, cmap='gray')

        # # 设置x和y轴的刻度位置
        # plt.xticks(np.arange(0, len(attn_mask), 10))
        # plt.yticks(np.arange(0, len(attn_mask), 10))

        # # 保存图像
        # plt.savefig('./vis/streamvis_1/attn_mask.png')

        dn_meta = {
            'pad_size': pad_size, ###bs中最大instance的数量 x num_groups
            'num_dn_group': num_groups,
            # 'post_dn': post_dn,
            'known_bid': known_bid.long(), ###num_groups x 指示每个instance属于哪个bs
            'map_known_indice': map_known_indice, ###根据一组的结果，再复制组别数，但是同一个bs里的index加上bs中最大instance数
            #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,
            #3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
            #8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            #26, 27, 28, 29, 30, 31, 32, 33, 34, 20, 21, 22, 23, 24, 20, 21, 22, 23,
            #24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            #52, 53, 54, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 40, 41,
            #42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
            'loss_weight': loss_weight, ###bs x per_instance 
        }

        # 去掉完全的直线，分母为0
        for i, pos in enumerate(neglect_pos):
            if len(pos) != 0:
                for j in range(num_groups):
                    denoise_centerline_refers[i][pos+single_pad * j] = gt_pts[i][pos]
                    denoise_lane_refers[i][pos+single_pad * j] = gt_pts[i][pos]
                    denoise_centerline_refers_pe[i][pos+single_pad * j] = gt_pts[i][pos]
        # import pdb; pdb.set_trace()
        return input_query_label, input_query_left_type_label, input_query_right_type_label, input_query_bbox, input_query_mask, attn_mask, dn_meta, denoise_centerline_refers, denoise_centerline_refers_pe, denoise_lane_refers

    # def fix_pts_interpolate_pytorch(self, lane, n_points):
    #     """
    #     使用 PyTorch 实现的线性插值，将 lane 转换为 n_points 个点。
        
    #     Args:
    #         lane (torch.Tensor): 输入的形状为 [N, 2] 的 PyTorch 张量，表示一系列点。
    #         n_points (int): 插值后的点的数量。
        
    #     Returns:
    #         torch.Tensor: 形状为 [n_points, 2] 的 PyTorch 张量。
    #     """
    #     # 将输入转换为 NumPy 以便使用 shapely 计算

    #     lane_np = lane.cpu().numpy()
    #     lane_list = []
    #     for lane_singe in lane_np:
    #         import pdb; pdb.set_trace()
    #         ls = LineString(lane_singe)
            
    #         # 计算距离并进行插值
    #         distances = torch.linspace(0, ls.length, n_points)
    #         interpolated = torch.tensor(
    #             [ls.interpolate(distance.item()).coords[0] for distance in distances],
    #             dtype=lane.dtype,
    #             device=lane.device
    #         )
    #         import pdb; pdb.set_trace()
    #         lane_list.append(interpolated)
    #     return interpolated
