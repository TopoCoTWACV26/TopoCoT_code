#---------------------------------------------------------------------------------------#
# BEV Semantic Segmentation Head for Stage1 Supervision
#---------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmdet.models.losses import CrossEntropyLoss
from mmseg.models.decode_heads.fcn_head import FCNHead

@HEADS.register_module()
class BEVSegHead(FCNHead):
    """
    BEV Semantic Segmentation Head for supervising BEV features.

    This head takes BEV features as input and outputs a semantic segmentation map.
    It uses a simple MLP-like structure with convolutions for BEV supervision.

    Args:
        in_channels (int): Number of input channels from BEV features.
        channels (int): Number of intermediate channels.
        num_convs (int): Number of convolutions in the head.
        concat_input (bool): Whether to concatenate input with output.
        dropout_ratio (float): Dropout ratio.
        num_classes (int): Number of semantic classes (default: 2 for lane/background).
        norm_cfg (dict): Config for normalization layers.
        align_corners (bool): Align corners for interpolation.
        loss_decode (dict): Config for loss function.
    """

    def __init__(self, **kwargs):
        if 'train_cfg' in kwargs:
            del kwargs['train_cfg']
        if 'test_cfg' in kwargs:
            del kwargs['test_cfg']
        super(BEVSegHead, self).__init__(**kwargs)

    def forward(self, bev_feats):
        """
        Forward function for BEV semantic segmentation.

        Args:
            bev_feats (torch.Tensor): BEV features of shape (B, C, H, W)

        Returns:
            torch.Tensor: Semantic segmentation logits of shape (B, num_classes, H, W)
        """
        output = self.convs(bev_feats)
        if self.concat_input:
            output = self.conv_cat(torch.cat([bev_feats, output], dim=1))
        output = self.cls_seg(output)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """
        Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted segmentation logits (B, num_classes, H, W)
            seg_label (torch.Tensor): Ground truth segmentation labels (B, H, W)

        Returns:
            dict: Dictionary containing the loss
        """
        loss = dict()
        seg_label = seg_label.squeeze(1).long()  # (B, 1, H, W) -> (B, H, W)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            ignore_index=self.ignore_index
        )

        return loss

    @force_fp32(apply_to=('seg_logit',))
    def get_accuracy(self, seg_logit, seg_label):
        """Compute accuracy for semantic segmentation."""
        seg_label = seg_label.squeeze(1).long()
        seg_pred = seg_logit.argmax(dim=1)
        acc = (seg_pred == seg_label).float().mean()
        return acc
