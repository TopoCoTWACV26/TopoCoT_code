#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.dense_heads import AnchorFreeHead
import matplotlib.pyplot as plt
from projects.plugin.models.utils.memory_buffer import StreamTensorMemory, StreamListMemory
from projects.plugin.models.utils.query_update import LaneSegMotionMLP
from matplotlib.path import Path
from projects.plugin.models.utils.utils import gen_sineembed_for_position, SinePositionalEncoding, gen_3D_sineembed_for_position
from projects.plugin.models.utils.query_denoising_new import LanednQueryGenerator
from projects.plugin.models.utils.dn_memory_buffer import DNStreamTensorMemory
from projects.lanesegnet.core.visualizer.lane_vis import draw_annotation_bev
from mmcv.cnn import xavier_init

from torch.utils.checkpoint import checkpoint
@HEADS.register_module()
class StreamLaneSegHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 dn_iter=0,
                 dn_cls_num=3,
                 tolerant_noise=0.2,
                 num_points = 10,
                 noise_decay_scale=[0.7, 0.7, 0.7],
                 stream_dn=False,
                 chamfer_thresh=None,
                 roi_size=(60, 30),
                 num_query=200,
                 with_box_refine=False,
                 with_shared_param=None,
                 transformer=None,
                 bbox_coder=None,
                 num_reg_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_traj_modal=1,
                 pc_range=None,
                 pts_dim=3,
                 sync_cls_avg_factor=False,
                 num_lane_type_classes=3,
                 streaming_cfg=dict(),
                 dn_cfg=dict(),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=3.0),
                 loss_dice=dict(type='DiceLoss', loss_weight=3.0),
                 loss_lane_type=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                loss_reg=dict(
                    type='LinesL1Loss',
                    loss_weight=50.0,
                    beta=0.01,
                ),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
                     )),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 pred_mask=False,
                 use_llm_decoder=False,
                 llm_cfg=None,
                 llm_adapter_cfg=None,
                 chat_loss_weight=1.0,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        # NOTE: Only initialize loss_cls, loss_bbox, etc. for non-LLM mode
        # These losses are NOT used when use_llm_decoder=True (Stage2/Stage3)
        if not use_llm_decoder:
            self.loss_cls = build_loss(loss_cls)
            self.loss_bbox = build_loss(loss_bbox)
            self.loss_lane_type = build_loss(loss_lane_type)
            self.loss_mask = build_loss(loss_mask)
            self.loss_dice = build_loss(loss_dice)
            self.loss_reg = build_loss(loss_reg)
            self.pred_mask = pred_mask
            self.loss_mask_type = loss_mask['type']

            if self.loss_cls.use_sigmoid:
                self.cls_out_channels = num_classes
            else:
                self.cls_out_channels = num_classes + 1

            if loss_lane_type.use_sigmoid:
                self.cls_lane_type_out_channels = num_lane_type_classes
            else:
                self.cls_lane_type_out_channels = num_lane_type_classes + 1

        # LLM Decoder Configuration
        self.use_llm_decoder = use_llm_decoder
        self.llm_cfg = llm_cfg if llm_cfg is not None else dict()
        self.llm_adapter_cfg = llm_adapter_cfg if llm_adapter_cfg is not None else dict()
        self.chat_loss_weight = chat_loss_weight
        self._llm_chat_loss = None  # For storing chat loss from LLM decoder
        self._llm_coord_loss = None  # For storing coord loss from LLM decoder

        if self.use_llm_decoder:

            # Use LLM decoder - initialize LLM components
            from ..modules.llm_decoder import LLMDecoder
            from ..modules.llm_adapter import LLMAdapter

            # Set embed_dims from in_channels
            self.embed_dims = in_channels

            # Initialize LLM Adapter
            adapter_cfg = dict(
                in_channels=in_channels,
                out_channels=self.embed_dims,
                bev_h=bev_h,
                bev_w=bev_w,
                **self.llm_adapter_cfg
            )
            self.llm_adapter = LLMAdapter(**adapter_cfg)

            # Initialize LLM Decoder
            decoder_cfg = dict(
                llm_path=self.llm_cfg.get('llm_path', './pretrained/InternVL2-2B'),
                set_lora=self.llm_cfg.get('set_lora', True),
                is_pretraining=self.llm_cfg.get('is_pretraining', True),
                chat_cfg=self.llm_cfg.get('chat_config', dict()),
                input_dim=in_channels,
                attention_type=self.llm_cfg.get('attention_type', 'flash_attention_2'),
                img_length=self.llm_cfg.get('img_length', bev_h // 4),
                num_learnable_query=self.llm_cfg.get('num_learnable_query', 0),
                torch_dtype=torch.bfloat16 if self.llm_cfg.get('set_lora', True) else torch.float32,
                use_lora_r=self.llm_cfg.get('use_lora_r', 64),
                use_llm_lora_alpha=self.llm_cfg.get('use_llm_lora_alpha', 128),
            )
            self.llm_decoder = LLMDecoder(**decoder_cfg)
            # Get current device for LLM loading (important for Flash Attention 2.0)
            device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
            self.llm_decoder.create_llm(load_weight=self.llm_cfg.get('load_internvl_weight', True), device=device)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_cfg.get('llm_path', './InternVL2-2B'),
                trust_remote_code=True,
                use_fast=False
            )
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim

        self.with_box_refine = with_box_refine
        if with_shared_param is not None:
            self.with_shared_param = with_shared_param
        else:
            self.with_shared_param = not self.with_box_refine
        self.as_two_stage = False

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = pts_dim * 30
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, ] * self.code_size
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self.num_lane_type_classes = num_lane_type_classes

        ### stream
        self.iter = 0
        self.dn_iter = dn_iter
        self.stream_dn = stream_dn
        self.tolerant_noise = tolerant_noise
        self.noise_decay_scale = noise_decay_scale
        self.chamfer_thresh = chamfer_thresh
        self.dn_cfg = dn_cfg
        self.num_queries = num_query
        self.dn_cls_num = dn_cls_num
        self.num_points = num_points
        if streaming_cfg:
            self.streaming_query = streaming_cfg['streaming']

        else:
            self.streaming_query = False
        


        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2, -roi_size[1]/2, self.pc_range[2])
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))
        self.map_size = [-50, -25, 50, 25]

 

    
    @auto_fp16(apply_to=('mlvl_feats'))
    def forward_train(self, mlvl_feats, bev_feats, img_metas,gt_lanes_3d = None, gt_lane_labels_3d = None, gt_instance_masks = None, gt_lane_left_type = None, gt_lane_right_type = None, future_data = None, gt_ego_fut_cmd = None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_lanes_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 99].
            all_mask_preds (Tensor): Sigmoid outputs from the segmentation \
                head with normalized value in the range of [0,1].
                Shape []
        """

        bs = len(img_metas)
        dtype = mlvl_feats[0].dtype

        # Use LLM decoder or transformer decoder based on configuration
        if self.use_llm_decoder:
            # LLM Decoder path: BEV features -> LLM Adapter -> LLM -> text output
            # bev_feats is currently (B, H*W, C) format from detector's flatten
            # Need to reshape to (B, C, H, W) for LLM adapter
            bev_for_adapter = bev_feats[:bs].reshape(bs, self.bev_h, self.bev_w, self.embed_dims).permute(0, 3, 1, 2)

            # Pass through LLM adapter which handles downsampling and projection
            # NOTE: Hugging Face gradient checkpointing is now enabled in the LLM decoder itself
            # via llm.gradient_checkpointing_enable(), so we don't need manual checkpointing here
            adapter_output = self.llm_adapter(
                bev_features=bev_for_adapter,
                prompt_dict=None,  # Will be processed separately
                learn_query_input=None
            )

            # Get downsampled and projected BEV from adapter
            # adapter_output['projected_bev'] has shape (B, H*W, C) after downsampling
            # The adapter's bev_downsample module reduces (B, 256, 100, 200) to (B, 1024, 25, 50)
            # Then it's reshaped to (B, 1250, 256) after projection
            projected_bev = adapter_output['projected_bev']  # (B, 1250, 256)

            # Reshape back to (B, H, W, C) for LLM decoder
            # After adapter downsampling: H=25, W=50, C=embed_dims=256
            downsampled_h = self.bev_h // 4  # 25
            downsampled_w = self.bev_w // 4  # 50
            bev_for_llm = projected_bev.reshape(bs, downsampled_h, downsampled_w, self.embed_dims)

            # Prepare prompt_dict from conversation data (if available)
            # Try to get conversation data from img_metas
            prompt_dict = self._get_prompt_dict_from_img_metas(img_metas, bs, bev_feats.device)

            # Call LLM decoder with conversation data
            # NOTE: Hugging Face gradient checkpointing is enabled in the LLM itself

            llm_output = checkpoint(self.llm_decoder, bev_for_llm, prompt_dict, use_reentrant=False)

            # Store chat loss for later use
            self._llm_chat_loss = llm_output.get('chat_loss', None)
            # Store coord loss for monitoring
            self._llm_coord_loss = llm_output.get('coord_loss', None)

            # Store LLM outputs for parsing coordinates from text
            # The LLM generates text containing lane coordinates
            self._llm_logits = llm_output.get('logits', None)
            self._llm_text_emb = llm_output.get('text_emb', None)

        # Apply chat_loss_weight to the loss
        if self._llm_chat_loss is not None:
            return self._llm_chat_loss * self.chat_loss_weight
        return None

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward_test(self, mlvl_feats, bev_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_lanes_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 99].
            all_mask_preds (Tensor): Sigmoid outputs from the segmentation \
                head with normalized value in the range of [0,1].
                Shape []
        """

        bs = len(img_metas)
        dtype = mlvl_feats[0].dtype

        # Use LLM decoder or transformer decoder based on configuration
        if self.use_llm_decoder:
            # For LLM decoder, we need to ensure query_pos is extracted from object_query_embeds
            # This is needed for reference point initialization
            # Note: object_query_embeds shape is [bs, num_query, embed_dims * 2]
            # We split along dim=2 to get query_pos and content

            # LLM Decoder path: BEV features -> LLM Adapter -> LLM -> text output
            # bev_feats is currently (B, H*W, C) format from detector's flatten
            # Need to reshape to (B, C, H, W) for LLM adapter
            bev_for_adapter = bev_feats[:bs].reshape(bs, self.bev_h, self.bev_w, self.embed_dims).permute(0, 3, 1, 2)

            # Pass through LLM adapter which handles downsampling and projection
            # NOTE: Hugging Face gradient checkpointing is enabled in the LLM itself
            adapter_output = self.llm_adapter(
                bev_features=bev_for_adapter,
                prompt_dict=None,
                learn_query_input=None
            )

            # Get downsampled and projected BEV from adapter
            projected_bev = adapter_output['projected_bev']  # (B, 1250, 256)

            # Reshape back to (B, H, W, C) for LLM decoder
            downsampled_h = self.bev_h // 4  # 25
            downsampled_w = self.bev_w // 4  # 50
            bev_for_llm = projected_bev.reshape(bs, downsampled_h, downsampled_w, self.embed_dims)

            # Check if text generation is enabled (like HERMES)
            generate_text = self.llm_cfg.get('generate_text', False)
            # FIXED: Get max_new_tokens from chat_config (where it's actually set in config files)
            # Default to 512 if not specified in chat_config
            max_new_tokens = self.llm_cfg.get('chat_config', {}).get('max_new_tokens', 8192)

            # IMPORTANT: For test/inference, load the test prompt template
            # Load test_llm_prompt.json to get system and prompt
            import json
            import os

            test_prompt_path = './tools/test_llm_prompt.json'

            # Load and format the test prompt
            with open(test_prompt_path, 'r') as f:
                test_prompt_data = json.load(f)

            # Get system and prompt from the JSON file
            system = test_prompt_data[0]['system']
            prompt = test_prompt_data[0]['prompt']

            # Get img_context_token_id and create placeholder
            from projects.lanesegnet.models.modules.llm_decoder import IMG_CONTEXT_TOKEN
            img_context_token_id = self.llm_decoder.img_context_token_id
            num_bev_tokens = (self.bev_h // 4) * (self.bev_w // 4)  # 1250

            # Create <IMG_CONTEXT> placeholder tokens (same as FormatConversationForLLM)
            img_context_placeholder = ' '.join([IMG_CONTEXT_TOKEN] * num_bev_tokens)

            # Format: system + <IMG_CONTEXT> tokens + prompt (same as training format, but without answer)
            full_text = f"{system}\n{img_context_placeholder}\n{prompt}"

            # Get tokenizer and tokenize

            # Tokenize the full prompt (system + <IMG_CONTEXT> + prompt)
            encoded = self.tokenizer (
                full_text,
                return_tensors='pt',
                padding=False,
                truncation=False
            )

            # encoded['input_ids'] shape is [1, seq_len], squeeze to [seq_len]
            input_ids_for_gen = encoded['input_ids'].squeeze(0).unsqueeze(0).expand(bs, -1).to(device=bev_feats.device)
            attention_mask_for_gen = encoded['attention_mask'].squeeze(0).unsqueeze(0).expand(bs, -1).to(device=bev_feats.device)

            # IMPORTANT: For test/inference, we ALWAYS use generate to get text coordinates
            is_text_generation = True  # Force text generation in test mode

            if is_text_generation:

                generation_output = self.llm_decoder.generate(
                    bev=bev_for_llm,
                    input_ids=input_ids_for_gen,
                    attention_mask=attention_mask_for_gen,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                # Decode generation to text
                tokenizer = self.llm_decoder.tokenizer
                generated_text = tokenizer.batch_decode(
                    generation_output['generation_output'],
                    skip_special_tokens=True
                )[0]

                # Store the generated text in the output
                llm_output = {
                    'out_bev': bev_for_llm,  # Return BEV features for coordinate extraction
                    'generated_text': generated_text,
                    'generation_output': generation_output,
                }

                # Also store for external access
                self._llm_generated_text = generated_text

                # Debug breakpoint to inspect LLM output

            else:
                # Standard forward pass for coordinate extraction
                # Create prompt for lane segmentation task
                # IMPORTANT: For test/inference time, we need <IMG_CONTEXT> tokens to inject BEV features
                # The LLM decoder expects img_context_token_id (e.g., 92546) to know where to inject BEV tokens
                num_bev_tokens = (self.bev_h // 4) * (self.bev_w // 4)  # 1250 = 25 * 50

                # Get img_context_token_id from LLM decoder
                img_context_token_id = self.llm_decoder.img_context_token_id

                # Create input_ids filled with <IMG_CONTEXT> tokens for BEV injection
                # Shape: (bs, num_bev_tokens) - each position will receive one BEV token
                input_ids = bev_feats.new_full((bs, num_bev_tokens), img_context_token_id, dtype=torch.long)
                attention_mask = bev_feats.new_ones((bs, num_bev_tokens), dtype=torch.long)

                prompt_dict = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': None,
                }

                # Call LLM decoder - it will directly output text coordinates
                llm_output = self.llm_decoder(
                    bev=bev_for_llm,
                    text=prompt_dict,
                    device=bev_feats.device
                )

            # Store chat loss for later use
            self._llm_chat_loss = llm_output.get('chat_loss', None)
            # Store coord loss for monitoring
            self._llm_coord_loss = llm_output.get('coord_loss', None)

            # Store LLM outputs (for debugging, but don't keep in memory to avoid slowdown)
            # self._llm_logits = llm_output.get('logits', None)
            # self._llm_text_emb = llm_output.get('text_emb', None)

            # Convert LLM output to format expected by bbox_coder.decode
            # The output should contain all_cls_scores, all_lanes_preds, all_lanes_left_type, all_lanes_right_type
            outs = self._parse_llm_output_to_preds(llm_output, bs, bev_feats.device)

            # Include LLM generated text in output for JSON generation during testing
            outs['llm_generated_text'] = self._llm_generated_text
          
        return outs

    def _parse_llm_output_to_preds(self, llm_output, bs, device):
        """
        Parse LLM output to prediction format expected by bbox_coder.decode.

        This method extracts lane coordinates from LLM output and formats them as:
        - all_cls_scores: classification scores [1, bs, num_query, num_classes]
        - all_lanes_preds: lane coordinates [1, bs, num_query, code_size]
        - all_lanes_left_type: left lane type scores [1, bs, num_query, num_type_classes]
        - all_lanes_right_type: right lane type scores [1, bs, num_query, num_type_classes]

        Args:
            llm_output: Output from LLM decoder containing 'out_bev' and other data
            bs: Batch size
            device: Device to create tensors on

        Returns:
            dict: Formatted predictions with keys matching bbox_coder.decode expectations
        """
        import torch
        import torch.nn.functional as F

        # Get BEV features from LLM output
        out_bev = llm_output.get('out_bev')  # Shape: (B, H, W, C) where H=25, W=50, C=256

        if out_bev is None:
            # Fallback: return empty predictions
            return self._create_empty_predictions(bs, device)

        # Check if LLM generated text coordinates
        generated_text = llm_output.get('generated_text', None)

        if generated_text is not None:
            # Parse text coordinates from generated text
            return self._parse_text_coordinates(generated_text, bs, device)
        else:
            # Extract coordinates from BEV features using a decoding head
            return self._extract_coords_from_bev(out_bev, bs, device)

    def _create_empty_predictions(self, bs, device):
        """Create empty prediction dict when no valid output is available."""
        num_query = self.num_query
        num_classes = self.num_classes
        code_size = self.code_size
        num_type_classes = self.num_lane_type_classes

        all_cls_scores = torch.zeros((1, bs, num_query, num_classes), device=device)
        all_lanes_preds = torch.zeros((1, bs, num_query, code_size), device=device)
        all_lanes_left_type = torch.zeros((1, bs, num_query, num_type_classes), device=device)
        all_lanes_right_type = torch.zeros((1, bs, num_query, num_type_classes), device=device)

        return {
            'all_cls_scores': all_cls_scores,
            'all_lanes_preds': all_lanes_preds,
            'all_lanes_left_type': all_lanes_left_type,
            'all_lanes_right_type': all_lanes_right_type,
        }

    def _parse_text_coordinates(self, generated_text, bs, device):
        """
        Parse lane coordinates from LLM generated text.

        Expected text format:
        Lane 1: x1,y1,z1,x2,y2,z2,...; confidence: 0.95; type: centerline
        Lane 2: ...

        Args:
            generated_text: Text string generated by LLM
            bs: Batch size
            device: Device to create tensors on

        Returns:
            dict: Formatted predictions
        """
        import re
        import torch
        import numpy as np

        # Parse lane information from text
        # This is a placeholder - adjust based on your actual LLM output format
        lanes = []
        scores = []
        labels = []
        left_types = []
        right_types = []

        # Example parsing logic (adjust based on your LLM's output format)
        lines = generated_text.split('\n')
        for line in lines:
            if 'Lane' in line or 'lane' in line:
                # Extract coordinates using regex
                coords_match = re.search(r'\[([\d.,\s]+)\]', line)
                if coords_match:
                    coords_str = coords_match.group(1)
                    coords = [float(x) for x in coords_str.split(',')]
                    if len(coords) >= self.code_size:
                        lanes.append(coords[:self.code_size])

                        # Extract confidence
                        conf_match = re.search(r'confidence[:\s]+([\d.]+)', line, re.IGNORECASE)
                        score = float(conf_match.group(1)) if conf_match else 0.5
                        scores.append(score)

                        # Extract label
                        label_match = re.search(r'label[:\s]+(\d+)', line, re.IGNORECASE)
                        label = int(label_match.group(1)) if label_match else 0
                        labels.append(label)

                        # Extract types (placeholder)
                        left_types.append(0)
                        right_types.append(0)

        # Convert to tensors
        num_detected = len(lanes)
        num_query = self.num_query
        num_classes = self.num_classes
        num_type_classes = self.num_lane_type_classes

        # Pad or truncate to num_query
        if num_detected > 0:
            lanes_tensor = torch.tensor(lanes, dtype=torch.float32, device=device)
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            left_types_tensor = torch.tensor(left_types, dtype=torch.long, device=device)
            right_types_tensor = torch.tensor(right_types, dtype=torch.long, device=device)

            if num_detected < num_query:
                # Pad with zeros
                pad_size = num_query - num_detected
                lanes_tensor = F.pad(lanes_tensor, (0, 0, 0, pad_size))
                scores_tensor = F.pad(scores_tensor, (0, pad_size))
                labels_tensor = F.pad(labels_tensor, (0, pad_size))
                left_types_tensor = F.pad(left_types_tensor, (0, pad_size))
                right_types_tensor = F.pad(right_types_tensor, (0, pad_size))
            elif num_detected > num_query:
                # Truncate
                lanes_tensor = lanes_tensor[:num_query]
                scores_tensor = scores_tensor[:num_query]
                labels_tensor = labels_tensor[:num_query]
                left_types_tensor = left_types_tensor[:num_query]
                right_types_tensor = right_types_tensor[:num_query]

            # Create classification scores from labels
            all_cls_scores = torch.zeros((num_query, num_classes), device=device)
            all_cls_scores.scatter_(1, labels_tensor.unsqueeze(1), scores_tensor.unsqueeze(1))

            # Create type scores
            all_lanes_left_type = torch.zeros((num_query, num_type_classes), device=device)
            all_lanes_right_type = torch.zeros((num_query, num_type_classes), device=device)
            all_lanes_left_type.scatter_(1, left_types_tensor.unsqueeze(1), 1.0)
            all_lanes_right_type.scatter_(1, right_types_tensor.unsqueeze(1), 1.0)

            # Add batch dimension
            all_cls_scores = all_cls_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, num_query, num_classes)
            all_lanes_preds = lanes_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, num_query, code_size)
            all_lanes_left_type = all_lanes_left_type.unsqueeze(0).unsqueeze(0)
            all_lanes_right_type = all_lanes_right_type.unsqueeze(0).unsqueeze(0)

            # Expand to batch size
            all_cls_scores = all_cls_scores.expand(1, bs, -1, -1)
            all_lanes_preds = all_lanes_preds.expand(1, bs, -1, -1)
            all_lanes_left_type = all_lanes_left_type.expand(1, bs, -1, -1)
            all_lanes_right_type = all_lanes_right_type.expand(1, bs, -1, -1)
        else:
            # No lanes detected, return empty predictions
            return self._create_empty_predictions(bs, device)

        return {
            'all_cls_scores': all_cls_scores,
            'all_lanes_preds': all_lanes_preds,
            'all_lanes_left_type': all_lanes_left_type,
            'all_lanes_right_type': all_lanes_right_type,
        }

    def _extract_coords_from_bev(self, out_bev, bs, device):
        """
        Extract lane coordinates from BEV features using a simple prediction head.

        This method creates a simple MLP to predict lane coordinates directly from BEV features,
        similar to how traditional detection heads work.

        Args:
            out_bev: BEV features from LLM output, shape (B, H, W, C)
            bs: Batch size
            device: Device to create tensors on

        Returns:
            dict: Formatted predictions
        """
        import torch
        import torch.nn as nn

        # Get BEV dimensions
        B, H, W, C = out_bev.shape

        # Flatten BEV features
        bev_flat = out_bev.reshape(B, -1, C)  # (B, H*W, C)

        # Simple approach: use average pooling to get query features
        # In a real implementation, you would use a proper detection head here
        num_query = self.num_query

        # Create a simple linear projection if not already created
        if not hasattr(self, '_llm_coord_proj'):
            self._llm_coord_proj = nn.Linear(C, C).to(device)
            self._llm_cls_proj = nn.Linear(C, self.num_classes).to(device)
            self._llm_reg_proj = nn.Linear(C, self.code_size).to(device)
            self._llm_left_type_proj = nn.Linear(C, self.num_lane_type_classes).to(device)
            self._llm_right_type_proj = nn.Linear(C, self.num_lane_type_classes).to(device)

        # Project BEV features to query space
        query_features = self._llm_coord_proj(bev_flat)  # (B, H*W, C)

        # Sample random positions as queries (or use learned queries)
        # For simplicity, we'll use the first num_query BEV positions
        if H * W >= num_query:
            sampled_features = query_features[:, :num_query, :]  # (B, num_query, C)
        else:
            # Pad if BEV is smaller than num_query
            pad_size = num_query - (H * W)
            sampled_features = F.pad(query_features, (0, 0, 0, pad_size))

        # Predict classification, regression, and types
        cls_scores = self._llm_cls_proj(sampled_features)  # (B, num_query, num_classes)
        lanes_preds = self._llm_reg_proj(sampled_features)  # (B, num_query, code_size)
        left_type_scores = self._llm_left_type_proj(sampled_features)  # (B, num_query, num_type_classes)
        right_type_scores = self._llm_right_type_proj(sampled_features)  # (B, num_query, num_type_classes)

        # Add layer dimension to match expected format: (1, B, num_query, *)
        all_cls_scores = cls_scores.unsqueeze(0)  # (1, B, num_query, num_classes)
        all_lanes_preds = lanes_preds.unsqueeze(0)  # (1, B, num_query, code_size)
        all_lanes_left_type = left_type_scores.unsqueeze(0)  # (1, B, num_query, num_type_classes)
        all_lanes_right_type = right_type_scores.unsqueeze(0)  # (1, B, num_query, num_type_classes)

        return {
            'all_cls_scores': all_cls_scores,
            'all_lanes_preds': all_lanes_preds,
            'all_lanes_left_type': all_lanes_left_type,
            'all_lanes_right_type': all_lanes_right_type,
        }

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           masks_pred,
                           gt_labels,
                           gt_lanes,
                           gt_instance_masks,
                           gt_lanes_left_type,
                           gt_lanes_right_type):

        num_bboxes = lanes_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(lanes_pred, masks_pred, cls_score, gt_lanes, 
                                                gt_instance_masks, gt_labels)

        sampling_result = self.sampler.sample(assign_result, lanes_pred, gt_lanes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        labels_left_type = gt_lanes_left_type.new_full((num_bboxes,), self.num_lane_type_classes, dtype=torch.long)
        labels_left_type[pos_inds] = gt_lanes_left_type[sampling_result.pos_assigned_gt_inds].long()
                
        labels_right_type = gt_lanes_right_type.new_full((num_bboxes,), self.num_lane_type_classes, dtype=torch.long)
        labels_right_type[pos_inds] = gt_lanes_right_type[sampling_result.pos_assigned_gt_inds].long()
        
        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds] = 1.0

        # mask targets
        mask_targets = gt_instance_masks[pos_assigned_gt_inds]
        mask_weights = masks_pred.new_zeros((self.num_query, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, mask_weights, labels_left_type, labels_right_type,
                pos_inds, neg_inds, pos_assigned_gt_inds)

    def _get_prompt_dict_from_img_metas(self, img_metas, bs, device):
        """
        Extract conversation data from img_metas and format as prompt_dict for LLM.

        This method handles two scenarios:
        1. When data is loaded through the pipeline with LoadConversation and FormatConversationForLLM
        2. When data is loaded through the dataset class directly

        Args:
            img_metas: Image metadata from data pipeline (can be dict or list)
            bs: Batch size
            device: Device to create tensors on

        Returns:
            dict: Prompt dictionary with input_ids, attention_mask, labels, llm_generate_text flag
        """
        import torch
        from mmcv.parallel import DataContainer as DC

        def _unwrap_datacontainer(data):
            """Unwrap DataContainer to get the actual data."""
            if isinstance(data, DC):
                return data.data
            return data

        llm_input_ids = None
        llm_attention_mask = None
        llm_labels = None
        llm_generate_text = False  # Flag to trigger text generation (like HERMES)

        # Scenario 1: img_metas is a dict with frame indices as keys
        # if isinstance(img_metas, dict):
        #     # Get the first frame's metadata (current frame)
        #     first_frame_meta = img_metas.get(0, img_metas.get(list(img_metas.keys())[0], {}))

        #     # Check if conversation data exists in metadata
        #     # The pipeline should have added 'llm_input_ids', 'llm_attention_mask', 'llm_labels'
        #     llm_input_ids = _unwrap_datacontainer(first_frame_meta.get('llm_input_ids', None))
        #     llm_attention_mask = _unwrap_datacontainer(first_frame_meta.get('llm_attention_mask', None))
        #     llm_labels = _unwrap_datacontainer(first_frame_meta.get('llm_labels', None))
        #     llm_generate_text = _unwrap_datacontainer(first_frame_meta.get('llm_generate_text', False))
        #     import pdb; pdb.set_trace()
        # Scenario 2: img_metas is a list of metadata dicts
        if isinstance(img_metas, list) and len(img_metas) > 0:
            # Get the first item's metadata
            first_meta = img_metas[0]

            # Check for llm_generate_text flag first
            llm_generate_text = _unwrap_datacontainer(first_meta.get('llm_generate_text', False))
         
            # Check for llm_data (from dataset class)
            if 'llm_data' in first_meta and first_meta['llm_data'] is not None:
                # Format llm_data on-the-fly using tokenizer
                llm_data = first_meta['llm_data']
                system = llm_data.get('system', '')
                prompt = llm_data.get('prompt', '')
                answer = llm_data.get('answer', '')
              
                # Format: system message + user prompt + assistant response
                if system:
                    full_text = f"{system}\n{prompt}\n{answer}"
                else:
                    full_text = f"{prompt}\n{answer}"

                # Tokenize
                # try:
                # from transformers import AutoTokenizer
                # tokenizer = AutoTokenizer.from_pretrained(
                #     self.llm_cfg.get('llm_path', '/data_test/home/lizhen/yym/TopoWMChange/InternVL2-2B'),
                #     trust_remote_code=True,
                #     use_fast=False
                # )

                max_length = self.llm_cfg.get('chat_config', {}).get('max_new_tokens', 4096)
               
                encoded = self.tokenizer (
                    full_text,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                llm_input_ids = encoded['input_ids'].squeeze(0)
                llm_attention_mask = encoded['attention_mask'].squeeze(0)

                # Create labels for training
                llm_labels = llm_input_ids.clone()

                # Mask prompt tokens
                prompt_encoded = self.tokenizer (
                    prompt,
                    max_length=max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                prompt_length = prompt_encoded['input_ids'].shape[1]
                llm_labels[:prompt_length] = -100
            # except Exception:
            #     llm_input_ids = None
            else:
                # Try to get directly from metadata (unwrap DataContainer if needed)
                llm_input_ids = _unwrap_datacontainer(first_meta.get('llm_input_ids', None))
                llm_attention_mask = _unwrap_datacontainer(first_meta.get('llm_attention_mask', None))
                llm_labels = _unwrap_datacontainer(first_meta.get('llm_labels', None))

        if llm_input_ids is not None:
            # Use actual conversation data from pipeline
            # Convert to tensor if needed
            if not isinstance(llm_input_ids, torch.Tensor):
                llm_input_ids = torch.tensor(llm_input_ids, dtype=torch.long, device=device)
            if not isinstance(llm_attention_mask, torch.Tensor):
                llm_attention_mask = torch.tensor(llm_attention_mask, dtype=torch.long, device=device)
            if llm_labels is not None and not isinstance(llm_labels, torch.Tensor):
                llm_labels = torch.tensor(llm_labels, dtype=torch.long, device=device)

            # Expand to batch size if needed
            if llm_input_ids.dim() == 1:
                llm_input_ids = llm_input_ids.unsqueeze(0).expand(bs, -1)
                llm_attention_mask = llm_attention_mask.unsqueeze(0).expand(bs, -1)
                if llm_labels is not None:
                    llm_labels = llm_labels.unsqueeze(0).expand(bs, -1)
          
            prompt_dict = {
                'input_ids': llm_input_ids,
                'attention_mask': llm_attention_mask,
                'labels': llm_labels,
                'llm_generate_text': llm_generate_text,  # Include the flag for text generation
            }
        else:
            # Fallback: No conversation data available, create minimal input with <IMG_CONTEXT> tokens
            # This is required for BEV feature injection into the LLM
            num_bev_tokens = (self.bev_h // 4) * (self.bev_w // 4)  # 1250 = 25 * 50
            img_context_token_id = self.llm_decoder.img_context_token_id
       
            prompt_dict = {
                'input_ids': torch.full((bs, num_bev_tokens), img_context_token_id, dtype=torch.long, device=device),
                'attention_mask': torch.ones((bs, num_bev_tokens), dtype=torch.long, device=device),
                'labels': None,
                'llm_generate_text': False,  # No text generation without conversation data
            }

        return prompt_dict

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    masks_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_lanes_left_type_list,
                    gt_lanes_right_type_list):

        (labels_list, label_weights_list, lanes_targets_list,
         lanes_weights_list, masks_targets_list, masks_weights_list, labels_left_type_list, labels_right_type_list,
         pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, masks_preds_list,
            gt_labels_list, gt_lanes_list, gt_masks_list, gt_lanes_left_type_list, gt_lanes_right_type_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )

        return (labels_list, label_weights_list, lanes_targets_list,
                lanes_weights_list, masks_targets_list, masks_weights_list, labels_left_type_list, labels_right_type_list,
                num_total_pos, num_total_neg, assign_result)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """
        Loss method required by AnchorFreeHead base class.
        This implementation returns an empty dict since actual losses are computed
        in forward_train and aggregated in StreamLaneSegNet.forward_train.

        For LLM decoder mode (use_llm_decoder=True), chat_loss is returned from
        forward_train directly. For traditional transformer decoder mode, losses
        would be computed here.
        """
        # Return loss dict with coord_loss for monitoring
        # coord_loss已经是detached tensor，不参与梯度回传，但需要保持为tensor格式以便MMDetection处理
        loss_dict = dict()
        if self._llm_coord_loss is not None:
            # 保持为tensor格式（已经detach，不参与梯度回传）
            loss_dict['coord_loss'] = self._llm_coord_loss
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            result = [lanes, scores, labels]
            if 'left_type_scores' in preds:
                left_type_scores = preds['left_type_scores']
                left_type_labels = preds['left_type_labels']
                right_type_scores = preds['right_type_scores']
                right_type_labels = preds['right_type_labels']
                result.extend([left_type_scores, left_type_labels, right_type_scores, right_type_labels])
            ret_list.append(result)

        return ret_list

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)


