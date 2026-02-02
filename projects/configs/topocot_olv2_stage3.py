_base_ = []
custom_imports = dict(imports=['projects.bevformer', 'projects.lanesegnet', 'projects.plugin'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
roi_size = (102.4, 51.2, 4.0)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['lane_segment', 'ped_crossing']
class_nums = len(class_names)
num_queries = 200
#####stage3 iterbasedrunner
num_epochs = 12
all_train_sample_numer = 22349
batch_size = 1 #2
num_gpus = 8    #4
num_iters_per_epoch = all_train_sample_numer // (num_gpus * batch_size) ###870
# num_epochs_single_frame = num_epochs //2
num_epochs_single_frame = 0
total_iters = num_iters_per_epoch * num_epochs

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7
pts_dim = 3

# Use LLM-supporting dataset type
dataset_type = 'StreamOpenLaneV2_subset_A_LaneSegNet_LLM_Dataset'
data_root = 'data/'

para_method = 'fix_pts_interp'
method_para = dict(n_points=10)
code_size = 3 * method_para['n_points'] * pts_dim

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
),

_wm_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_*4,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
),

_num_levels_ = 4
bev_h_ = 100
bev_w_ = 200

model = dict(
    type='StreamLaneSegNet',
    bev_h=bev_h_,
    bev_w=bev_w_,
    roi_size=roi_size,
    use_llm=True,
    wm_bev_constructor=dict(
            type='WMBEVFormerConstructer',
            num_feature_levels=_num_levels_,
            num_cams=num_cams,
            embed_dims=_dim_,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            pc_range=point_cloud_range,
            bev_h=bev_h_//2,
            bev_w=bev_w_//2,
            rotate_center=[bev_h_//4, bev_w_//4],
            encoder=dict(
                type='WMBEVFormerEncoder',
                num_layers=2,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='WMBEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1)
                    ],
                    ffn_cfgs=_wm_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_),
        ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    bev_constructor=dict(
        type='BEVFormerConstructer',
        num_feature_levels=_num_levels_,
        num_cams=num_cams,
        embed_dims=_dim_,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        rotate_center=[bev_h_//2, bev_w_//2],
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        embed_dims=_dim_,
                        num_cams=num_cams,
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_)
                    )
                ],
                ffn_cfgs=_ffn_cfg_,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
    ),
    lane_head=dict(
        type='StreamLaneSegHead',
        dn_iter=0,
        tolerant_noise=0.2,
        noise_decay_scale=[0.7, 0.7, 0.7],
        roi_size=roi_size,
        num_classes=class_nums,
        num_points=method_para['n_points'],
        stream_dn=False, ##重新设计去噪
        num_lane_type_classes=3,
        in_channels=_dim_,
        num_query=num_queries,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        pts_dim=pts_dim,
        sync_cls_avg_factor=False,
        with_box_refine=True,
        code_size=code_size,
        num_traj_modal=3,
        code_weights= [1.0 for i in range(code_size)],
        dn_cfg=dict(  # CdnQueryGenerator
            hidden_dim=_dim_, ##256
            num_queries=num_queries,
            num_classes=class_nums,
            noise_scale=dict(label=0.5, box=0.2, pt=0.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=1, num_dn_queries=240), #60
            bev_h=bev_h_, bev_w=bev_w_,
            pc_range=point_cloud_range,
            voxel_size=[0.1, 0.1],
            num_pts_per_vec=method_para['n_points'],
            rotate_range=0.0,
            map_size=[-50, -25, 50, 25],
            noise_decay=False),###for stream
        streaming_cfg=dict(
            streaming=True,
            batch_size=batch_size,
            topk=int(num_queries*(1/3)),
            trans_loss_weight=0.3,
        ),
        # Transformer decoder removed - using LLM decoder instead
    
        bbox_coder=dict(type='LaneSegmentPseudoCoder'),
        # LLM Decoder Configuration - replaces transformer decoder
        use_llm_decoder=True,  # Enable LLM decoder
        llm_cfg=dict(
            llm_path='./InternVL2-2B',  # Path to pretrained LLM model
            set_lora=True,  # Use LoRA for efficient fine-tuning
            is_pretraining=True,  # Enable chat_loss to ensure LoRA parameters have gradients
            attention_type='flash_attention_2',  # Use Flash Attention 2 for efficiency
            img_length=(bev_h_ // 4) * (bev_w_ // 4),  # Downsampled BEV token sequence length (h/4 * w/4)
            num_learnable_query=0,  # Number of learnable queries
            load_internvl_weight=True,  # Load InternVL pretrained weights
            update_all_embeddings=True,  # Allow full embedding and lm_head update
            use_lora_r=64,  # LoRA rank (default: 128, reduced to 64)
            use_llm_lora_alpha=128,  # LoRA alpha (default: 256, reduced to 128, usually 2x rank)
            chat_config=dict(  # Chat generation config
                num_beams=1,
                max_new_tokens=8192,
                min_new_tokens=1,
                do_sample=False,
                temperature=0.,
            ),
        ),
        llm_adapter_cfg=dict(
            use_prompt_embedding=True,
            prompt_length=100,
            num_learnable_query=0,
            use_positional_encoding=True,
            projection_type='mlp',  # 'linear', 'mlp', or 'conv'
            dropout=0.1,
            use_bev_downsampling=True,  # Enable BEV downsampling from HERMES
        ),
        chat_loss_weight=1.0,  # Weight for chat loss
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.025),
        loss_lane_type=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.1),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=3.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=3.0),
        loss_reg=dict(type='L1Loss', loss_weight=0.025)),
    bev_seg_head=dict(
        type='BEVSegHead',
        in_channels=_dim_,
        channels=_dim_,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    streaming_cfg=dict(
        streaming_bev=True,
        stream=True,  ###允许判断first frame和融合
        batch_size=batch_size,
        fusion_cfg=dict(
            type='ConvGRU',
            out_channels=256,
        )),

    # model training and testing settings
    train_cfg=dict(
        lane=dict(
            assigner=dict(
                type='LaneSegmentHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.025),
                mask_cost=dict(type='CrossEntropyLossCost', weight=3.0, use_sigmoid=True),
                dice_cost=dict(type='DiceCost', weight=3.0, pred_act=False, eps=1.0),
                pc_range=point_cloud_range))))

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3DLaneSegment',
         with_lane_3d=True, with_lane_label_3d=True, with_lane_adj=True, with_lane_type=True,
         with_bbox=False, with_label=False, with_lane_lste_adj=False, with_future_waypoint=True, with_ego_fut_cmd = True),
    # Load LLM conversation data from train_conv
    dict(type='LoadConversation', conversation_dir='/data_test/home/lizhen/yym/TopoWMChange/data/train_conv_rdp', with_system=True),
    # Format conversation for LLM input (includes <IMG_CONTEXT> tokens for BEV feature injection)
    dict(type='FormatConversationForLLM',
         tokenizer_path='/data_test/home/lizhen/yym/TopoWMChange/InternVL2-2B',
         max_length=10500,
         with_system=True,
         num_bev_tokens=(bev_h_ // 4) * (bev_w_ // 4)),  # 1250 BEV tokens after downsampling (25*50)
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='GridMaskMultiViewImage'),
    dict(type='LaneSegmentParameterize3D', method=para_method, method_para=method_para),
    dict(type='GenerateLaneSegmentMask', points_num=method_para['n_points'], bev_h=bev_h_, bev_w=bev_w_),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=[
        'img', 'gt_lanes_3d', 'gt_lane_labels_3d', 'gt_lane_adj',
        'gt_instance_masks', 'gt_lane_left_type', 'gt_lane_right_type','gt_future_waypoint', 'gt_ego_fut_cmd'],
    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
               'depth2img', 'cam2img', 'pad_shape', 'crop_shape',
               'scale_factor', 'flip', 'pcd_horizontal_flip',
               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
               'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
               'transformation_3d_flow', 'scene_token',
               'can_bus', 'lidar2global_rotation','lidar2global_translation','prev',
               'next', 'timestamp',  # Add timestamp for LLM data loading
               'llm_input_ids', 'llm_attention_mask', 'llm_labels'))
]

# Test pipeline - no LLM conversation data needed (only training uses it)
# During test/evaluation, the model will create minimal input with <IMG_CONTEXT> tokens
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img'],
    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
               'depth2img', 'cam2img', 'pad_shape', 'crop_shape',
               'scale_factor', 'flip', 'pcd_horizontal_flip',
               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
               'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
               'transformation_3d_flow', 'scene_token',
               'can_bus', 'lidar2global_rotation','lidar2global_translation','prev',
               'next','lane_id', 'ego_fut_cmd', 'all_idx'))
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,  # Reduced from 4 to avoid OOM/bus error with temporal queue
    train=dict(
        type=dataset_type,
        data_root=data_root + 'Trainset/',
        ann_file=data_root + 'data_dict_subset_A_train_lanesegnet.pkl',
        train_conv_path='./data/train_conv_rdp',  # Absolute path to LLM conversation data
        queue_length=2,  # Temporal queue length: current + 2 previous frames (reduced from 3 to save memory while ensuring temporal context)
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        split='train',
        filter_map_change=True,
        points_num=method_para['n_points'],
        seq_split_num=1,
        load_llm_data=True,  # Enable LLM data loading
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root + 'Testset/',
        ann_file=data_root + 'data_dict_subset_A_val_lanesegnet.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        points_num=method_para['n_points'],
        seq_split_num=1,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root + 'Testset/',
        ann_file=data_root + 'data_dict_subset_A_val_lanesegnet.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        points_num=method_para['n_points'],
        seq_split_num=1,
        test_mode=True),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        # num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
        num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
        random_drop=0.0
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    # lr=4e-4,
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# HERMES stage2-2: Gradient accumulation with FP32 master weights
# - cumulative_iters=2: Accumulate gradients over 2 iterations before updating
# - Using GradientCumulativeOptimizerHook (not Fp16) to avoid GradScaler BF16 incompatibility
# - BF16 doesn't need loss scaling (unlike FP16)
optimizer_config = dict(
    type="GradientCumulativeOptimizerHook",  # Non-FP16 version to avoid GradScaler
    cumulative_iters=4,
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=1e-4,
    )

checkpoint_config = dict(
    interval=1000,
    max_keep_ckpts=3)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='StreamLaneSegNet', name='StreamLaneSegNet-BEVFormer-Stream')),
    ])

evaluation = dict(interval=num_epochs*num_iters_per_epoch)
runner = dict(
    type='IterBasedRunner',
    max_iters=total_iters)

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stream'
load_from = './work_dirs/stream/stage2/latest.pth'
#resume_from = './work_dirs/stream/stage3/iter_6000.pth'
#load_from = None
resume_from = None
workflow = [('train', 1)]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
find_unused_parameters=True #### 第一帧没用到一些fc 所以要设为true