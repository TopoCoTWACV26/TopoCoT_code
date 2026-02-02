#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# LLM Adapter Module - Adapts BEV features and prompts for LLM input                    #
#---------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
import numpy as np

@HEADS.register_module()
class LLMAdapter(nn.Module):
    """
    Adapter module to convert BEV features and prompts into LLM-compatible format.

    This module handles:
    1. BEV feature projection and tokenization
    2. Prompt embedding and formatting
    3. Multi-modal feature fusion

    Args:
        in_channels (int): Input BEV feature channels
        out_channels (int): Output feature channels (LLM dimension)
        bev_h (int): BEV height
        bev_w (int): BEV width
        use_prompt_embedding (bool): Whether to use learnable prompt embedding
        prompt_length (int): Length of prompt sequence
        num_learnable_query (int): Number of learnable queries
        use_positional_encoding (bool): Whether to add positional encoding
        projection_type (str): Type of projection ('linear', 'mlp', 'conv')
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=2048,
                 bev_h=100,
                 bev_w=200,
                 use_prompt_embedding=True,
                 prompt_length=100,
                 num_learnable_query=0,
                 use_positional_encoding=True,
                 projection_type='mlp',
                 dropout=0.1,
                 use_bev_downsampling=True):  # New parameter for BEV downsampling
        super(LLMAdapter, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_prompt_embedding = use_prompt_embedding
        self.prompt_length = prompt_length
        self.num_learnable_query = num_learnable_query
        self.use_positional_encoding = use_positional_encoding
        self.projection_type = projection_type
        self.use_bev_downsampling = use_bev_downsampling

        # BEV downsampling block (from HERMES)
        # Reduces BEV features from (w, h, c) to (w/4, h/4, c*4)
        # This reduces token count by 16x while preserving information
        if use_bev_downsampling:
            self.bev_downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            downsampled_channels = in_channels * 4
            self.downsampled_h = bev_h // 4
            self.downsampled_w = bev_w // 4
        else:
            downsampled_channels = in_channels
            self.downsampled_h = bev_h
            self.downsampled_w = bev_w

        # BEV feature projection
        if projection_type == 'linear':
            self.bev_proj = nn.Linear(downsampled_channels, out_channels)
        elif projection_type == 'mlp':
            self.bev_proj = nn.Sequential(
                nn.Linear(downsampled_channels, out_channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels * 2, out_channels),
            )
        elif projection_type == 'conv':
            self.bev_proj = nn.Sequential(
                nn.Conv2d(downsampled_channels, out_channels, 1, 1, 0),
                nn.Flatten(2),
            )
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

        # Learnable prompt embedding
        if use_prompt_embedding:
            self.prompt_embedding = nn.Embedding(prompt_length, out_channels)

        # Positional encoding for BEV tokens (use downsampled dimensions)
        if use_positional_encoding:
            self.pos_encoding = self._build_positional_encoding(
                self.downsampled_h, self.downsampled_w, out_channels
            )

        # Learnable query embedding
        if num_learnable_query > 0:
            self.learnable_query = nn.Parameter(
                torch.randn(num_learnable_query, out_channels) / (out_channels ** 0.5)
            )

        # Layer norm
        self.layer_norm = nn.LayerNorm(out_channels)

        # Initialize weights
        self.init_weights()

    def _build_positional_encoding(self, h, w, d):
        """Build positional encoding for BEV grid."""
        pe = torch.zeros(h * w, d)
        position = torch.arange(0, h * w, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform', bias=0.)
            elif isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @auto_fp16(apply_to=('bev_features',))
    def forward(self, bev_features, prompt_dict=None, learn_query_input=None):
        """
        Forward pass.

        Args:
            bev_features: BEV features with shape (B, C, H, W) or (B, H, W, C)
            prompt_dict: Dictionary containing prompt information with keys:
                - input_ids: Token IDs (B, seq_len)
                - attention_mask: Attention mask (B, seq_len)
                - labels: Target labels (B, seq_len)
            learn_query_input: Optional learnable query input (B, N, C)

        Returns:
            Dictionary containing:
                - projected_bev: Projected BEV features (B, H*W, out_channels)
                - prompt_dict: Processed prompt dictionary
                - learn_query: Learnable query embedding (B, num_learnable_query, out_channels)
                - img_length: Effective image/token length
        """
        # Handle BEV features input format
        if bev_features.dim() == 4:
            B, C, H, W = bev_features.shape

            # Apply BEV downsampling (from HERMES)
            # Reduces from (B, C, H, W) to (B, C*4, H/4, W/4)
            if self.use_bev_downsampling:
                bev_features = self.bev_downsample(bev_features)  # (B, C*4, H/4, W/4)
                C = bev_features.shape[1]  # Update to C*4
                H = bev_features.shape[2]  # Update to H/4
                W = bev_features.shape[3]  # Update to W/4

            # Convert to (B, H*W, C)
            if self.projection_type == 'conv':
                projected_bev = self.bev_proj(bev_features)  # (B, out_channels, H*W)
                projected_bev = projected_bev.permute(0, 2, 1)  # (B, H*W, out_channels)
            else:
                bev_features = bev_features.permute(0, 2, 3, 1)  # (B, H, W, C)
                projected_bev = bev_features.reshape(B, H * W, C)
                projected_bev = self.bev_proj(projected_bev)  # (B, H*W, out_channels)
        else:
            B, H, W, C = bev_features.shape
            projected_bev = bev_features.reshape(B, H * W, C)
            projected_bev = self.bev_proj(projected_bev)

        # Add positional encoding
        if self.use_positional_encoding:
            projected_bev = projected_bev + self.pos_encoding

        # Layer normalization
        projected_bev = self.layer_norm(projected_bev)

        # Process prompt dictionary
        processed_prompt = self._process_prompt(prompt_dict, B, device=bev_features.device)

        # Get learnable query
        learn_query = None
        if self.num_learnable_query > 0:
            if learn_query_input is not None:
                # Project input to output dimension
                if learn_query_input.shape[-1] != self.out_channels:
                    learn_query = F.linear(learn_query_input, self.bev_proj[-1].weight if isinstance(self.bev_proj, nn.Sequential) else self.bev_proj.weight)
                else:
                    learn_query = learn_query_input
            else:
                learn_query = self.learnable_query.unsqueeze(0).expand(B, -1, -1)

        # Calculate effective image/token length
        img_length = int(np.sqrt(H * W))

        return {
            'projected_bev': projected_bev,
            'prompt_dict': processed_prompt,
            'learn_query': learn_query,
            'img_length': img_length,
        }

    def _process_prompt(self, prompt_dict, batch_size, device):
        """
        Process prompt dictionary for LLM input.

        Args:
            prompt_dict: Input prompt dictionary
            batch_size: Batch size
            device: Device

        Returns:
            Processed prompt dictionary
        """
        if prompt_dict is None:
            # Create default prompt dictionary
            return {
                'input_ids': torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                'attention_mask': torch.ones(batch_size, 1, dtype=torch.long, device=device),
                'labels': None,
            }

        processed = {}
        for key, value in prompt_dict.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    processed[key] = value.to(device)
                else:
                    processed[key] = value
            else:
                processed[key] = None

        # Ensure labels exists for training
        if 'labels' not in processed or processed['labels'] is None:
            processed['labels'] = None

        return processed

@HEADS.register_module()
class StreamLLMAdapter(LLMAdapter):
    """
    Streaming version of LLM adapter with temporal support.

    Handles temporal BEV feature aggregation and memory fusion.
    """

    def __init__(self,
                 *args,
                 use_temporal_memory=True,
                 memory_length=4,
                 temporal_fusion_type='gru',
                 **kwargs):
        super(StreamLLMAdapter, self).__init__(*args, **kwargs)

        self.use_temporal_memory = use_temporal_memory
        self.memory_length = memory_length
        self.temporal_fusion_type = temporal_fusion_type

        if use_temporal_memory:
            if temporal_fusion_type == 'gru':
                self.temporal_fusion = nn.GRU(
                    input_size=self.out_channels,
                    hidden_size=self.out_channels,
                    num_layers=2,
                    batch_first=True
                )
            elif temporal_fusion_type == 'lstm':
                self.temporal_fusion = nn.LSTM(
                    input_size=self.out_channels,
                    hidden_size=self.out_channels,
                    num_layers=2,
                    batch_first=True
                )
            elif temporal_fusion_type == 'attention':
                self.temporal_attention = nn.MultiheadAttention(
                    embed_dim=self.out_channels,
                    num_heads=8,
                    batch_first=True
                )
            else:
                raise ValueError(f"Unknown temporal_fusion_type: {temporal_fusion_type}")

            # Memory buffer for temporal features
            self.register_buffer('memory_buffer', torch.zeros(memory_length, 1, self.out_channels))
            self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))

    @auto_fp16(apply_to=('bev_features',))
    def forward(self, bev_features, prompt_dict=None, learn_query_input=None, is_first_frame=False):
        """
        Forward pass with temporal support.

        Args:
            bev_features: Current frame BEV features
            prompt_dict: Prompt dictionary
            learn_query_input: Learnable query input
            is_first_frame: Whether this is the first frame

        Returns:
            Output dictionary with temporal fusion applied
        """
        # Get base adapter output
        output = super().forward(bev_features, prompt_dict, learn_query_input)

        projected_bev = output['projected_bev']
        B, N, C = projected_bev.shape

        # Apply temporal fusion if enabled and not first frame
        if self.use_temporal_memory and not is_first_frame:
            if self.temporal_fusion_type in ['gru', 'lstm']:
                fused_bev, _ = self.temporal_fusion(projected_bev)
            else:  # attention
                fused_bev, _ = self.temporal_attention(
                    projected_bev,
                    self.memory_buffer[:self.memory_length].repeat(1, B, 1),
                    self.memory_buffer[:self.memory_length].repeat(1, B, 1)
                )

            # Residual connection
            output['projected_bev'] = fused_bev + projected_bev

        # Update memory buffer
        if self.use_temporal_memory:
            with torch.no_grad():
                # Update memory with current frame's average feature
                avg_feature = projected_bev.mean(dim=1, keepdim=True)  # (B, 1, C)
                ptr = self.memory_ptr.item()
                self.memory_buffer[ptr] = avg_feature[0:1]  # Store first sample
                self.memory_ptr = (ptr + 1) % self.memory_length

        return output

@HEADS.register_module()
class BEVTokenizer(nn.Module):
    """
    Tokenize BEV features into a sequence of tokens for LLM processing.

    This module divides BEV features into patches and tokenizes them.
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=2048,
                 patch_size=2,
                 bev_h=100,
                 bev_w=200):
        super(BEVTokenizer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Calculate number of patches
        self.num_patches_h = bev_h // patch_size
        self.num_patches_w = bev_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding
        self.patch_embed = nn.Linear(
            in_channels * patch_size * patch_size,
            out_channels
        )

        # Class token (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_channels))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, out_channels)
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @auto_fp16(apply_to=('bev_features',))
    def forward(self, bev_features):
        """
        Forward pass.

        Args:
            bev_features: BEV features (B, C, H, W)

        Returns:
            Tokenized BEV features (B, num_patches + 1, out_channels)
        """
        B, C, H, W = bev_features.shape

        # Reshape into patches
        patches = bev_features.reshape(
            B, C,
            self.num_patches_h, self.patch_size,
            self.num_patches_w, self.patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(
            B, self.num_patches,
            C * self.patch_size * self.patch_size
        )

        # Embed patches
        patch_tokens = self.patch_embed(patches)

        # Concatenate class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        return tokens