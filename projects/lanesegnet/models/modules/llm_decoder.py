#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# LLM Decoder Module - Using LLM as decoder for lane segmentation                       #
#---------------------------------------------------------------------------------------#

import gc
import json
import logging
import math
import os
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from projects.lanesegnet.models.internvl_chat.internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
try:
    from transformers import AutoTokenizer
    from transformers import GenerationConfig
    has_transformers = True
except ImportError:
    has_transformers = False
    warnings.warn('transformers not installed. Please install transformers.')

try:
    from peft import LoraConfig, get_peft_model
    has_peft = True
except ImportError:
    has_peft = False
    warnings.warn('peft not installed. Please install peft for LoRA support.')

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Constants for tokens
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'

IGNORE_INDEX = -100

def _freeze_params(module):
    """Freeze parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False

def wrap_llm_lora(llm, r=64, lora_alpha=128, lora_dropout=0.05, target_modules=None):
    """Wrap LLM with LoRA."""
    if not has_peft:
        raise ImportError('peft is required for LoRA. Please install peft.')
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type='CAUSAL_LM'
    )
    llm = get_peft_model(llm, lora_config)
    llm.enable_input_require_grads()
    llm.print_trainable_parameters()
    

@HEADS.register_module()
class MLPV2(nn.Module):
    """MLP adapter for dimension transformation."""

    def __init__(self, input_channel, output_channel):
        super(MLPV2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            nn.GELU(),
            nn.Linear(output_channel, output_channel),
        )

    def forward(self, x):
        if not self.training:
            x_dtype = x.dtype
            weight_dtype = self.fc1[0].weight.dtype
            x = x.to(weight_dtype)
        x = self.fc1(x)
        if not self.training:
            x = x.to(x_dtype)
        return x

# @HEADS.register_module()
# class LLMDecoder(nn.Module):
#     """
#     LLM Decoder for lane segmentation.

#     This module uses a Large Language Model (LLM) as the decoder,
#     taking BEV features and prompts as input to generate lane predictions.

#     Args:
#         llm_path (str): Path to the pretrained LLM model
#         set_lora (bool): Whether to use LoRA for efficient fine-tuning
#         is_pretraining (bool): Whether to enable pretraining mode (chat_loss)
#         chat_cfg (dict): Configuration for chat generation
#         input_dim (int): Input dimension of BEV features
#         attention_type (str): Type of attention mechanism
#         img_length (int): Length of image token sequence
#         num_learnable_query (int): Number of learnable queries
#         torch_dtype (torch.dtype): Data type for the model
#     """

#     def __init__(self,
#                  llm_path,
#                  set_lora=True,
#                  is_pretraining=False,
#                  chat_cfg=None,
#                  input_dim=256,
#                  attention_type='flash_attention_2',
#                  img_length=50,
#                  num_learnable_query=0,
#                  torch_dtype=torch.bfloat16,
#                  use_lora_r=128,
#                  use_llm_lora_alpha=256):
#         super(LLMDecoder, self).__init__()

#         if not has_transformers:
#             raise ImportError('transformers is required. Please install transformers.')

#         self.tokenizer_path = llm_path
#         self.model_path = llm_path
#         self.img_length = img_length
#         self.chat_cfg = chat_cfg if chat_cfg is not None else dict()
#         self.input_dim = input_dim
#         self.attention_type = attention_type

#         self.set_lora = set_lora
#         self.is_pretraining = is_pretraining
#         self.num_learnable_query = num_learnable_query

#         if set_lora:
#             self.torch_dtype = torch_dtype
#         else:
#             self.torch_dtype = torch.float32

#         # Model name specific configurations
#         self.model_name = os.path.basename(llm_path) if '/' not in llm_path else llm_path.split('/')[-1]

#         if self.model_name.startswith("InternVL2-1B"):
#             self.template_name = "Hermes-2"
#             self.target_modules = [
#                 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
#                 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
#             ]
#             llm_channel = 896
#         elif self.model_name.startswith("InternVL2-2B"):
#             self.template_name = "internlm2-chat"
#             self.target_modules = [
#                 'attention.wqkv', 'attention.wo',
#                 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'
#             ]
#             llm_channel = 2048
#         elif self.model_name.startswith("InternVL2-4B"):
#             self.template_name = "phi3-chat"
#             self.target_modules = [
#                 'self_attn.o_proj', 'self_attn.qkv_proj',
#                 'mlp.gate_up_proj', 'mlp.down_proj'
#             ]
#             llm_channel = 3072
#         elif self.model_name.startswith("InternVL2-8B"):
#             self.template_name = "internlm2-chat"
#             self.target_modules = [
#                 'attention.wqkv', 'attention.wo',
#                 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'
#             ]
#             llm_channel = 4096
#         elif self.model_name.startswith("Qwen3-4B") or self.model_name.startswith("Qwen3") or self.model_name.startswith("Qwen"):
#             # Qwen3-4B configuration
#             self.template_name = "qwen-chat"
#             self.target_modules = [
#                 'self_attn.o_proj', 'self_attn.qkv_proj',
#                 'mlp.gate_up_proj', 'mlp.down_proj'
#             ]
#             llm_channel = 2048
#         else:
#             # Default configuration for unknown models
#             self.template_name = "qwen-chat"
#             self.target_modules = [
#                 'self_attn.o_proj', 'self_attn.qkv_proj',
#                 'mlp.gate_up_proj', 'mlp.down_proj'
#             ]
#             llm_channel = 2048

#         # MLP adapters for dimension transformation
#         self.in_mlp = MLPV2(input_dim, llm_channel)
#         self.out_mlp = MLPV2(llm_channel, input_dim)

#         # Learnable query embedding (optional)
#         if num_learnable_query > 0:
#             self.query_embedding = nn.Parameter(
#                 torch.randn(num_learnable_query, input_dim) / (input_dim ** 0.5)
#             )

#         self.use_lora_r = use_lora_r
#         self.use_llm_lora_alpha = use_llm_lora_alpha

#         # Initialize tokenizer and LLM model
#         self.tokenizer = None
#         self.llm = None
#         self.use_fallback = False  # Initialize use_fallback flag

#     def create_llm(self, load_weight=True, device=None):
#         """Initialize the tokenizer and LLM model with proper GPU placement."""
#         # Initialize tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.tokenizer_path,
#             add_eos_token=False,
#             trust_remote_code=True,
#             use_fast=False
#         )
#         self.tokenizer.tokenizer_path = self.tokenizer_path
#         self.tokenizer.model_max_length = 16000
       
#         # Add special tokens
#         # token_list = [
#         #     IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
#         #     QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
#         #     REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN
#         # ]
#         token_list = [
#             IMG_CONTEXT_TOKEN
#         ]
#         self.num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
#         self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
#         self.learnable_query_token_id = -10

#         # Determine device: use provided device or auto-detect
#         if device is None:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Try to load the LLM model
#         try:
#             from transformers import AutoModelForCausalLM, AutoConfig

#             config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
#             config._attn_implementation = self.attention_type  # Set Flash Attention

#             # Disable KV cache in config before loading model
#             # This is required for gradient checkpointing compatibility
#             config.use_cache = False
#             if hasattr(config, 'text_config'):
#                 config.text_config.use_cache = False

#             if load_weight:
#                 self.llm = AutoModelForCausalLM.from_pretrained(
#                     self.model_path,
#                     torch_dtype=self.torch_dtype,
#                     config=config,
#                     trust_remote_code=True,
#                     device_map=None  # Disable auto device mapping
#                 )
#             else:
#                 self.llm = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
#                 self.llm.to(self.torch_dtype)

#             # Critical: Move model to target device immediately after loading
#             self.llm = self.llm.to(device)
#             assert next(self.llm.parameters()).device.type == 'cuda', "模型未成功加载到GPU"

#             # Disable KV cache to save memory (not needed during training)
#             # IMPORTANT: KV cache must be disabled when using gradient checkpointing
#             # They are incompatible because gradient checkpointing recomputes forward pass
#             if hasattr(self.llm, 'config'):
#                 self.llm.config.use_cache = False
#             # For InternVL2-style models with nested language_model
#             if hasattr(self.llm, 'language_model') and hasattr(self.llm.language_model, 'config'):
#                 self.llm.language_model.config.use_cache = False
#             # For models with text_config attribute
#             if hasattr(self.llm, 'text_config'):
#                 self.llm.text_config.use_cache = False
#             logger.info("KV cache disabled for gradient checkpointing compatibility")

#             # Apply LoRA if specified
#             if self.set_lora:
#                 if self.is_pretraining:
#                     _freeze_params(self.llm)
#                     # Still apply LoRA during pretraining to ensure gradients flow through
#                     # This is needed for DDP to work properly (all trainable params must have gradients)
#                     self._set_lora(freeze_llm=True, use_llm_lora=self.use_lora_r)
#                 else:
#                     self._set_lora(freeze_llm=True, use_llm_lora=self.use_lora_r)

#                 # Enable gradient checkpointing AFTER LoRA is applied to avoid DDP conflicts
#                 # IMPORTANT: For DDP + LoRA compatibility, we ONLY set config.gradient_checkpointing = True
#                 # Do NOT call gradient_checkpointing_enable() as it uses reentrant checkpointing
#                 # Setting the config flag enables the newer non-reentrant checkpointing
#                 if hasattr(self.llm, 'config'):
#                     self.llm.config.gradient_checkpointing = True
#                 if hasattr(self.llm, 'language_model') and hasattr(self.llm.language_model, 'config'):
#                     self.llm.language_model.config.gradient_checkpointing = True

#                 logger.info("Gradient checkpointing enabled via config (non-reentrant, DDP-compatible)")
#             else:
#                 self._set_lora(freeze_llm=False, use_llm_lora=0)

#             # Mark that we're using the real LLM, not fallback
#             self.use_fallback = False
#             logger.info(f"LLM model loaded from {self.model_path} and moved to {device}")
#             import pdb; pdb.set_trace()
#         except Exception as e:
#             logger.warning(f"Failed to load LLM model: {e}")
#             # Create fallback decoder and ensure it's on GPU
#             self._create_fallback_decoder()
#             self.llm = self.llm.to(device)
#             assert next(self.llm.parameters()).device.type == 'cuda', "模型未成功加载到GPU"

#     def _create_fallback_decoder(self):
#         """Create a simple transformer decoder as fallback."""
#         logger.info("Creating fallback transformer decoder")
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=self.input_dim,
#             nhead=8,
#             dim_feedforward=self.input_dim * 4,
#             dropout=0.1,
#             activation='gelu',
#             batch_first=True
#         )
#         self.llm = nn.TransformerDecoder(decoder_layer, num_layers=6)
#         self.llm.to(self.torch_dtype)
#         self.use_fallback = True

#     def _set_lora(self, freeze_llm=True, unfreeze_lm_head=False, use_llm_lora=128):
#         """Set LoRA configuration for the LLM."""
#         if freeze_llm:
#             self.llm = self.llm.eval()
#             _freeze_params(self.llm)

#         if unfreeze_lm_head and hasattr(self.llm, 'lm_head'):
#             self.llm.lm_head.requires_grad = True

#         if use_llm_lora > 0 and has_peft:
#             wrap_llm_lora(
#                 self.llm,
#                 r=use_llm_lora,
#                 lora_alpha=2 * use_llm_lora,
#                 target_modules=self.target_modules
#             )

#     def forward(self,
#                 bev=None,
#                 text=None,
#                 learn_query=None,
#                 device=None,
#                 output_hidden_states=True,
#                 return_dict=True):
#         """
#         Forward pass of the LLM decoder.

#         Args:
#             bev: BEV features with shape (B, H, W, C)
#             text: Text prompt dictionary containing:
#                 - input_ids: Token IDs
#                 - attention_mask: Attention mask
#                 - labels: Target labels for training
#             learn_query: Optional learnable query embeddings
#             device: Device to run on
#             output_hidden_states: Whether to output hidden states
#             return_dict: Whether to return as dictionary

#         Returns:
#             Dictionary containing output features and loss
#         """
#         if device is None:
#             device = bev.device if bev is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         if bev is None:
#             # Default dummy BEV for debugging
#             bev = torch.rand(1, self.img_length, self.img_length, self.input_dim).to(
#                 torch.float32
#             ).to(device)

#         # Process text input
#         input_ids = None
#         attention_mask = None
#         labels = None
#         #import pdb; pdb.set_trace()
#         if text is not None:
#             if isinstance(text, dict):
#                 input_ids = text.get('input_ids', None)
#                 attention_mask = text.get('attention_mask', None)
#                 labels = text.get('labels', None)

#             if input_ids is not None:
#                 input_ids = input_ids.to(device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(device)
#             if labels is not None:
#                 labels = labels.to(device)

#         # Get input embeddings from tokenizer
#         if input_ids is not None and hasattr(self.llm, 'get_input_embeddings'):
#             input_embeds = self.llm.get_input_embeddings()(input_ids).clone()
#         else:
#             # Create dummy input embeddings if no text provided
#             B = bev.shape[0] if bev is not None else 1
#             input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
#             input_embeds = torch.zeros(B, 1, self.input_dim, device=device)

#         B, N, C = input_embeds.shape
#         _, H, W, _ = bev.shape
#         bev_num_tokens = H * W

#         input_embeds = input_embeds.reshape(B * N, C)
#         bev = bev.reshape(B, bev_num_tokens, -1)

#         # Transform BEV features to LLM dimension
#         bev_embeds = self.in_mlp(bev).to(self.torch_dtype)
#         input_ids = input_ids.reshape(B * N)

#         # Handle learnable queries
#         if self.num_learnable_query > 0 and learn_query is not None:
#             if learn_query.shape[-1] != C:
#                 learn_query = self.in_mlp(learn_query).to(self.torch_dtype)

#             query_ids = torch.zeros(
#                 self.num_learnable_query,
#                 device=input_ids.device,
#                 dtype=input_ids.dtype
#             ) + self.learnable_query_token_id

#             input_ids = torch.cat((input_ids, query_ids), dim=0)

#             selected_query = (input_ids == self.learnable_query_token_id)
#             for_query = torch.zeros(
#                 (self.num_learnable_query, input_embeds.size(1)),
#                 device=input_embeds.device,
#                 dtype=input_embeds.dtype
#             )
#             input_embeds = torch.cat((input_embeds, for_query), dim=0)

#             input_embeds[selected_query] = learn_query.to(input_embeds.dtype)

#         # Inject BEV features into input embeddings
#         selected = (input_ids == self.img_context_token_id)

#         input_embeds[selected] = input_embeds[selected] * 0.0 + bev_embeds.reshape(-1, C).to(
#             input_embeds.dtype
#         )
#         ignore_flag = False

#         input_embeds = input_embeds.reshape(B, -1, C).to(self.torch_dtype)

#         # Fix attention_mask and labels dimension mismatch
#         # After reshaping input_embeds, we need to adjust attention_mask and labels to match
#         actual_seq_len = input_embeds.shape[1]  # Actual sequence length after reshape
#         if attention_mask is not None:
#             # attention_mask might be (B, N) where N is original length
#             # Need to ensure it matches (B, actual_seq_len)
#             if attention_mask.shape[1] < actual_seq_len:
#                 # Pad with 1s (meaning "attend to this token")
#                 padding = torch.ones(
#                     attention_mask.shape[0],
#                     actual_seq_len - attention_mask.shape[1],
#                     device=attention_mask.device,
#                     dtype=attention_mask.dtype
#                 )
#                 attention_mask = torch.cat([attention_mask, padding], dim=1)
#             elif attention_mask.shape[1] > actual_seq_len:
#                 # Truncate
#                 attention_mask = attention_mask[:, :actual_seq_len]

#         if labels is not None:
#             # labels might be (B, N) where N is original length
#             # Need to ensure it matches (B, actual_seq_len)
#             if labels.shape[1] < actual_seq_len:
#                 # Pad with -100 (ignore index)
#                 padding = torch.full(
#                     (labels.shape[0], actual_seq_len - labels.shape[1]),
#                     -100,
#                     device=labels.device,
#                     dtype=labels.dtype
#                 )
#                 labels = torch.cat([labels, padding], dim=1)
#             elif labels.shape[1] > actual_seq_len:
#                 # Truncate
#                 labels = labels[:, :actual_seq_len]

#         # Run LLM forward pass
#         if self.use_fallback or not hasattr(self.llm, 'forward'):
#             # Use fallback transformer decoder
#             hidden_states = self.llm(
#                 tgt=input_embeds,
#                 memory=input_embeds
#             )
#             logits = None
#         else:
#             # Use actual LLM with device consistency check
#             try:
#                 # Ensure input matches model device
#                 model_device = next(self.llm.parameters()).device
#                 input_embeds = input_embeds.to(model_device)

#                 if attention_mask is not None:
#                     attention_mask = attention_mask.to(model_device)
#                 if labels is not None:
#                     labels = labels.to(model_device)

#                 # Check if the model has language_model attribute (InternVL2 structure)
#                 if hasattr(self.llm, 'language_model'):
#                     # InternVL2 structure: call the language_model directly
#                     outputs = self.llm.language_model(
#                         inputs_embeds=input_embeds,
#                         attention_mask=attention_mask,
#                         output_hidden_states=output_hidden_states,
#                         return_dict=return_dict,
#                     )
#                 elif hasattr(self.llm, 'model'):
#                     # Some wrapper models
#                     outputs = self.llm.model(
#                         inputs_embeds=input_embeds,
#                         attention_mask=attention_mask,
#                         output_hidden_states=output_hidden_states,
#                         return_dict=return_dict,
#                     )
#                 else:
#                     # Standard HuggingFace model
#                     outputs = self.llm(
#                         inputs_embeds=input_embeds,
#                         attention_mask=attention_mask,
#                         output_hidden_states=output_hidden_states,
#                         return_dict=return_dict,
#                     )

#                 hidden_states = outputs.hidden_states[-1] if output_hidden_states else None
#                 logits = outputs.logits if hasattr(outputs, 'logits') else None
#             except Exception as e:
#                 logger.warning(f"LLM forward failed: {e}, using fallback")
#                 import traceback
#                 logger.warning(f"Traceback: {traceback.format_exc()}")
#                 hidden_states = input_embeds
#                 logits = None

#         hidden_states = hidden_states.reshape(B, -1, C)

#         # Extract BEV and text outputs
#         # selected has shape [N_total], where N_total is the total number of tokens across all batches
#         # Since we reshaped hidden_states to (B, -1, C), we need to reshape selected to match
#         try:
#             # Try to use the original selected mask with advanced indexing
#             # Reshape selected to (B, N) to match hidden_states
#             seq_len = hidden_states.shape[1]
#             selected_reshaped = selected[:B * seq_len].reshape(B, seq_len)

#             # Now we can index properly
#             hidden_bev_list = []
#             text_emb_list = []
#             for i in range(B):
#                 hidden_bev_list.append(hidden_states[i, selected_reshaped[i], :])
#                 text_emb_list.append(hidden_states[i, ~selected_reshaped[i], :])

#             hidden_bev = torch.cat(hidden_bev_list, dim=0).to(torch.float32)
#             text_emb = torch.cat(text_emb_list, dim=0).to(torch.float32)
#         except Exception as e:
#             # If that fails, process batch by batch with properly sliced masks
#             bev_tokens = []
#             text_tokens = []

#             seq_len = hidden_states.shape[1]
#             # Slice and reshape selected to match each batch's sequence length
#             for i in range(B):
#                 # Get the portion of selected mask that corresponds to this batch
#                 batch_selected = selected[i * seq_len:(i + 1) * seq_len]
#                 bev_tokens.append(hidden_states[i, batch_selected, :])
#                 text_tokens.append(hidden_states[i, ~batch_selected, :])

#             hidden_bev = torch.cat(bev_tokens, dim=0).to(torch.float32)
#             text_emb = torch.cat(text_tokens, dim=0).to(torch.float32)

#         # Handle learnable query output
#         if self.num_learnable_query > 0 and learn_query is not None:
#             try:
#                 # Reshape selected_query to match batch dimension
#                 seq_len = hidden_states.shape[1]
#                 # selected_query is a 1D mask, need to reshape it for each batch
#                 selected_query_reshaped = selected_query[:seq_len].unsqueeze(0).expand(B, seq_len)

#                 # Now we can index properly
#                 hidden_query = hidden_states[:, selected_query_reshaped, :].to(torch.float32)
#                 if logits is not None:
#                     logits = logits[:, ~selected_query_reshaped, :]
#             except Exception as e:
#                 # Fallback to batch processing
#                 query_tokens = []
#                 seq_len = hidden_states.shape[1]
#                 for i in range(B):
#                     # Use only the relevant portion of selected_query for this batch
#                     batch_selected_query = selected_query[:seq_len]
#                     query_tokens.append(hidden_states[i, batch_selected_query])

#                 hidden_query = torch.cat(query_tokens, dim=0).to(torch.float32)

#                 # Remove query tokens from text embeddings
#                 text_emb = text_emb[:-selected_query.sum(), :]

#                 if logits is not None:
#                     # For logits, we need to reshape and remove query tokens
#                     logits = logits.reshape(B, -1, logits.shape[-1])
#                     new_logits = []
#                     for i in range(B):
#                         batch_selected_query = selected_query[:seq_len]
#                         new_logits.append(logits[i, ~batch_selected_query])
#                     logits = torch.cat(new_logits, dim=0)

#             # Apply MLP to hidden_query
#             hidden_query = self.out_mlp(hidden_query)
#         else:
#             hidden_query = None

#         # Compute chat loss if training
#         loss = None
#         if labels is not None and self.is_pretraining and logits is not None:
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, logits.size(-1))
#             shift_labels = shift_labels.view(-1)
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#             if ignore_flag:
#                 loss = loss * 0.0

#         # Transform output back to BEV dimension
#         output_bev = self.out_mlp(hidden_bev).reshape(B, H, W, -1)

#         return {
#             'out_bev': output_bev,
#             'logits': logits,
#             'chat_loss': loss,
#             'out_query': hidden_query,
#             'text_emb': text_emb,
#         }

#     def generate(
#         self,
#         bev=None,
#         input_ids=None,
#         attention_mask=None,
#         max_new_tokens=512,
#         do_sample=False,
#         temperature=0.7,
#         top_p=0.9,
#         eos_token_id=None,
#         pad_token_id=None,
#         **kwargs
#     ):
#         """
#         Generate text from BEV features using the LLM (similar to HERMES).

#         Args:
#             bev: BEV features [B, H, W, C]
#             input_ids: Input token IDs [B, seq_len]
#             attention_mask: Attention mask [B, seq_len]
#             max_new_tokens: Maximum number of tokens to generate
#             do_sample: Whether to use sampling
#             temperature: Sampling temperature
#             top_p: Nucleus sampling parameter
#             eos_token_id: End-of-sequence token ID
#             pad_token_id: Padding token ID
#             **kwargs: Additional arguments for generation

#         Returns:
#             dict: Generated token IDs and metadata
#         """
#         from transformers import GenerationConfig
#         import torch

#         device = next(self.parameters()).device

#         # Process BEV features and prepare inputs
#         text = None
#         if input_ids is not None and hasattr(self.llm, 'get_input_embeddings'):
#             input_embeds = self.llm.get_input_embeddings()(input_ids).clone()
#         else:
#             # Create dummy input embeddings if no text provided
#             B = bev.shape[0] if bev is not None else 1
#             input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
#             input_embeds = torch.zeros(B, 1, self.input_dim, device=device)

#         B, N, C = input_embeds.shape
#         _, H, W, _ = bev.shape
#         bev_num_tokens = H * W

#         input_embeds = input_embeds.reshape(B * N, C)
#         bev = bev.reshape(B, bev_num_tokens, -1)

#         # Transform BEV features to LLM dimension
#         bev_embeds = self.in_mlp(bev).to(self.torch_dtype)
#         input_ids = input_ids.reshape(B * N)

#         # Inject BEV features into input embeddings
#         selected = (input_ids == self.img_context_token_id)
#         input_embeds[selected] = input_embeds[selected] * 0.0 + bev_embeds.reshape(-1, C).to(
#             input_embeds.dtype
#         )

#         input_embeds = input_embeds.reshape(B, -1, C).to(self.torch_dtype)

#         # Prepare generation config
#         if eos_token_id is None and hasattr(self._tokenizer, 'eos_token_id'):
#             eos_token_id = self._tokenizer.eos_token_id
#         if pad_token_id is None and hasattr(self._tokenizer, 'pad_token_id'):
#             pad_token_id = self._tokenizer.pad_token_id

#         generation_config = GenerationConfig(
#             max_new_tokens=max_new_tokens,
#             do_sample=do_sample,
#             temperature=temperature,
#             top_p=top_p,
#             eos_token_id=eos_token_id,
#             pad_token_id=pad_token_id,
#         )

#         # Generate tokens using the LLM
#         with torch.no_grad():
#             # Check model structure
#             if hasattr(self.llm, 'language_model'):
#                 # InternVL2 structure
#                 model = self.llm.language_model
#             elif hasattr(self.llm, 'model'):
#                 model = self.llm.model
#             else:
#                 model = self.llm

#             # Prepare generation inputs
#             generation_outputs = model.generate(
#                 inputs_embeds=input_embeds,
#                 attention_mask=attention_mask,
#                 generation_config=generation_config,
#                 **kwargs
#             )

#         return {
#             'generation_output': generation_outputs,
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#         }

@HEADS.register_module()
class LLMDecoder(nn.Module):
    """
    LLM Decoder for lane segmentation.

    This module uses a Large Language Model (LLM) as the decoder,
    taking BEV features and prompts as input to generate lane predictions.

    Args:
        llm_path (str): Path to the pretrained LLM model
        set_lora (bool): Whether to use LoRA for efficient fine-tuning
        is_pretraining (bool): Whether to enable pretraining mode (chat_loss)
        chat_cfg (dict): Configuration for chat generation
        input_dim (int): Input dimension of BEV features
        attention_type (str): Type of attention mechanism
        img_length (int): Length of image token sequence
        num_learnable_query (int): Number of learnable queries
        torch_dtype (torch.dtype): Data type for the model
    """

    def __init__(self,
                 llm_path,
                 set_lora=True,
                 is_pretraining=False,
                 chat_cfg=None,
                 input_dim=256,
                 attention_type='flash_attention_2',
                 img_length=50,
                 num_learnable_query=0,
                 torch_dtype=torch.bfloat16,
                 use_lora_r=128,
                 use_llm_lora_alpha=256,
                 load_weight=True):
        super(LLMDecoder, self).__init__()

        if not has_transformers:
            raise ImportError('transformers is required. Please install transformers.')

        self.tokenizer_path = llm_path
        self.model_path = llm_path

        self.load_weight = load_weight
        self.img_length = img_length
        self.chat_cfg = chat_cfg if chat_cfg is not None else dict()
        self.input_dim = input_dim
        self.attention_type = attention_type

        self.lora = set_lora
        self.is_pretraining = is_pretraining
        self.num_learnable_query = num_learnable_query

        if set_lora:
            self.torch_dtype = torch.bfloat16
        else:
            # FlashAttention only supports fp16 and bf16, not float32
            # Use bfloat16 even when not using LoRA
            self.torch_dtype = torch.bfloat16

        # Model name specific configurations
        self.model_name = os.path.basename(llm_path) if '/' not in llm_path else llm_path.split('/')[-1]

        if self.model_name.startswith("InternVL2-1B"):
            self.template_name = "Hermes-2"
            self.target_modules = [
                'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
            ]
            llm_channel = 896
        elif self.model_name.startswith("InternVL2-2B"):
            self.template_name = "internlm2-chat"
            self.target_modules = [
                'attention.wqkv', 'attention.wo',
                'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'
            ]
            llm_channel = 2048
        elif self.model_name.startswith("InternVL2-4B"):
            self.template_name = "phi3-chat"
            self.target_modules = [
                'self_attn.o_proj', 'self_attn.qkv_proj',
                'mlp.gate_up_proj', 'mlp.down_proj'
            ]
            llm_channel = 3072
        elif self.model_name.startswith("InternVL2-8B"):
            self.template_name = "internlm2-chat"
            self.target_modules = [
                'attention.wqkv', 'attention.wo',
                'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'
            ]
            llm_channel = 4096
        elif self.model_name.startswith("Qwen3-4B") or self.model_name.startswith("Qwen3") or self.model_name.startswith("Qwen"):
            # Qwen3-4B configuration
            self.template_name = "qwen-chat"
            self.target_modules = [
                'self_attn.o_proj', 'self_attn.qkv_proj',
                'mlp.gate_up_proj', 'mlp.down_proj'
            ]
            llm_channel = 2048
        else:
            # Default configuration for unknown models
            self.template_name = "qwen-chat"
            self.target_modules = [
                'self_attn.o_proj', 'self_attn.qkv_proj',
                'mlp.gate_up_proj', 'mlp.down_proj'
            ]
            llm_channel = 2048

        # MLP adapters for dimension transformation
        self.in_mlp = MLPV2(input_dim, llm_channel)
        self.out_mlp = MLPV2(llm_channel, input_dim)

        # Learnable query embedding (optional)
        if num_learnable_query > 0:
            self.query_embedding = nn.Parameter(
                torch.randn(num_learnable_query, input_dim) / (input_dim ** 0.5)
            )

        self.use_lora_r = use_lora_r
        self.use_llm_lora_alpha = use_llm_lora_alpha

        # Initialize tokenizer and LLM model
        self.tokenizer = None
        self.llm = None
        self.use_fallback = False  # Initialize use_fallback flag

    def build_coord_tokens(self,  x_cap=500, y_cap=1000):
        xs = [f"<X_{i}>" for i in range(x_cap + 1)]
        ys = [f"<Y_{i}>" for i in range(y_cap + 1)]
        return xs + ys



    def create_llm(self, load_weight=True, device=None):
        """Initialize the tokenizer and LLM model with proper GPU placement."""
        # ---------------------------
        # 1) Initialize tokenizer
        # ---------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.tokenizer_path = self.tokenizer_path
        self.tokenizer.model_max_length = 10100

        # ---------------------------
        # 2) Add special tokens (tokenizer side)
        # ---------------------------
        token_list = [IMG_CONTEXT_TOKEN]
        self.num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


        self.learnable_query_token_id = -10

        print("tokenizer vocab:", len(self.tokenizer))

        # ---------------------------
        # 3) Load InternVL2 (LLM inside)
        # ---------------------------
        print('load pretrained llm:', self.model_path)
        config = InternVLChatConfig.from_pretrained(self.model_path)

        config.vision_config.drop_path_rate = 0.
        config.template = self.template_name
        config.select_layer = -1
        config.dynamic_image_size = True
        config.use_thumbnail = True
        config.ps_version = 'v2'
        config.min_dynamic_patch = 1
        config.max_dylisnamic_patch = 6

        if self.model_name == "InternVL2-1B":
            config.llm_config._attn_implementation = self.attention_type
        config.llm_config.attn_implementation = self.attention_type

        config.llm_config.gradient_checkpointing = True
        
        if self.load_weight:
            model = InternVLChatModel.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype, config=config
            )
        else:
            model = InternVLChatModel._from_config(config).to(self.torch_dtype)

        model.language_model.config.use_cache = False
        self.llm = model.language_model

        # ---------------------------
        # 4) Resize embeddings to match tokenizer
        #    + record base vocab size for "方案C"
        # ---------------------------
        old_vocab_size = self.llm.get_input_embeddings().weight.size(0)

        # ---------------------------
        # 5) Apply freeze / LoRA
        # ---------------------------
        if self.is_pretraining:
            _freeze_params(self.llm)
            if self.lora:
                warnings.warn("Should not use lora during pretraining!")
                self.set_lora(model, use_llm_lora=self.use_lora_r)
        else:
            if self.lora:
                self.set_lora(model, use_llm_lora=self.use_lora_r)
            else:
                self.set_lora(model, freeze_llm=False, freeze_mlp=False, use_llm_lora=0)

        del model
        torch.cuda.empty_cache()

    def set_lora(
        self,
        freeze_llm=True,
        unfreeze_lm_head=False,
        use_llm_lora=64,
        freeze_mlp=True,
    ):
        """
        方案C：不影响原有生成能力
        - 冻结 LLM 全部参数（或大部分）
        - LoRA 正常训练（可选）
        - 只允许“新增 tokens 对应的 embedding 行”和“lm_head 对应行”更新
        （旧 vocab 的行梯度全部置零）
        """
        if freeze_llm:
            self.llm = self.llm.eval()
            _freeze_params(self.llm)

        # if unfreeze_lm_head:
        #     # 注意：这里只是放开 requires_grad，下面仍然会用 hook 把旧行梯度清零
        #     self.llm.lm_head.requires_grad = True

        if use_llm_lora:
            wrap_llm_lora(
                self.llm,
                r=use_llm_lora,
                lora_alpha=2 * use_llm_lora,
                target_modules=self.target_modules
            )
      


    def _set_lora(self, freeze_llm=True, unfreeze_lm_head=False, use_llm_lora=128):
        """Set LoRA configuration for the LLM."""
        if freeze_llm:
            self.llm = self.llm.eval()
            _freeze_params(self.llm)

        if unfreeze_lm_head and hasattr(self.llm, 'lm_head'):
            self.llm.lm_head.requires_grad = True

        if use_llm_lora > 0 and has_peft:
            wrap_llm_lora(
                self.llm,
                r=use_llm_lora,
                lora_alpha=2 * use_llm_lora,
                target_modules=self.target_modules
            )

    def forward(self,
                bev=None,
                text=None,
                learn_query=None,
                device=None,
                output_hidden_states=True,
                return_dict=True):
        """
        Forward pass of the LLM decoder.

        Args:
            bev: BEV features with shape (B, H, W, C)
            text: Text prompt dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Target labels for training
            learn_query: Optional learnable query embeddings
            device: Device to run on
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return as dictionary

        Returns:
            Dictionary containing output features and loss
        """
        if device is None:
            device = bev.device if bev is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if bev is None:
            # Default dummy BEV for debugging
            raise ValueError("BEV features cannot be None")

        # Process text input
        input_ids = None
        attention_mask = None
        labels = None
        #import pdb; pdb.set_trace()
        if text is not None:
            if isinstance(text, dict):
                input_ids = text.get('input_ids', None)
                attention_mask = text.get('attention_mask', None)
                labels = text.get('labels', None)

            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)

        # Get input embeddings from tokenizer
        if input_ids is not None and hasattr(self.llm, 'get_input_embeddings'):
            input_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        else:
            # Create dummy input embeddings if no text provided
            B = bev.shape[0] if bev is not None else 1
            input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
            input_embeds = torch.zeros(B, 1, self.input_dim, device=device)

        B, N, C = input_embeds.shape
        _, H, W, _ = bev.shape
        bev_num_tokens = H * W

        input_embeds = input_embeds.reshape(B * N, C)
        bev = bev.reshape(B, bev_num_tokens, -1)

        # Transform BEV features to LLM dimension
        bev_embeds = self.in_mlp(bev).to(self.torch_dtype)
        input_ids = input_ids.reshape(B * N)

        # Handle learnable queries
        if self.num_learnable_query > 0 and learn_query is not None:
            if learn_query.shape[-1] != C:
                learn_query = self.in_mlp(learn_query).to(self.torch_dtype)

            query_ids = torch.zeros(
                self.num_learnable_query,
                device=input_ids.device,
                dtype=input_ids.dtype
            ) + self.learnable_query_token_id

            input_ids = torch.cat((input_ids, query_ids), dim=0)

            selected_query = (input_ids == self.learnable_query_token_id)
            for_query = torch.zeros(
                (self.num_learnable_query, input_embeds.size(1)),
                device=input_embeds.device,
                dtype=input_embeds.dtype
            )
            input_embeds = torch.cat((input_embeds, for_query), dim=0)

            input_embeds[selected_query] = learn_query.to(input_embeds.dtype)

        # Inject BEV features into input embeddings
        selected = (input_ids == self.img_context_token_id)

        input_embeds[selected] = input_embeds[selected] * 0.0 + bev_embeds.reshape(-1, C).to(
            input_embeds.dtype
        )
        ignore_flag = False

        input_embeds = input_embeds.reshape(B, -1, C).to(self.torch_dtype)

        # Fix attention_mask and labels dimension mismatch
        # After reshaping input_embeds, we need to adjust attention_mask and labels to match
        actual_seq_len = input_embeds.shape[1]  # Actual sequence length after reshape
        if attention_mask is not None:
            # attention_mask might be (B, N) where N is original length
            # Need to ensure it matches (B, actual_seq_len)
            if attention_mask.shape[1] < actual_seq_len:
                # Pad with 1s (meaning "attend to this token")
                padding = torch.ones(
                    attention_mask.shape[0],
                    actual_seq_len - attention_mask.shape[1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([attention_mask, padding], dim=1)
            elif attention_mask.shape[1] > actual_seq_len:
                # Truncate
                attention_mask = attention_mask[:, :actual_seq_len]

        if labels is not None:
            # labels might be (B, N) where N is original length
            # Need to ensure it matches (B, actual_seq_len)
            if labels.shape[1] < actual_seq_len:
                # Pad with -100 (ignore index)
                padding = torch.full(
                    (labels.shape[0], actual_seq_len - labels.shape[1]),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype
                )
                labels = torch.cat([labels, padding], dim=1)
            elif labels.shape[1] > actual_seq_len:
                # Truncate
                labels = labels[:, :actual_seq_len]
  
        # Run LLM forward pass
        if self.use_fallback or not hasattr(self.llm, 'forward'):
            # Use fallback transformer decoder
            hidden_states = self.llm(
                tgt=input_embeds,
                memory=input_embeds
            )
            logits = None
        else:
            # Use actual LLM with device consistency check
            try:
                # Ensure input matches model device
                model_device = next(self.llm.parameters()).device
                input_embeds = input_embeds.to(model_device)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(model_device)
                if labels is not None:
                    labels = labels.to(model_device)

                # Check if the model has language_model attribute (InternVL2 structure)
                # if hasattr(self.llm, 'language_model'):
                #     # InternVL2 structure: call the language_model directly
                #     outputs = self.llm.language_model(
                #         inputs_embeds=input_embeds,
                #         attention_mask=attention_mask,
                #         output_hidden_states=output_hidden_states,
                #         return_dict=return_dict,
                #     )
                # elif hasattr(self.llm, 'model'):
                #     # Some wrapper models
                #     outputs = self.llm.model(
                #         inputs_embeds=input_embeds,
                #         attention_mask=attention_mask,
                #         output_hidden_states=output_hidden_states,
                #         return_dict=return_dict,
                #     )
                # else:
                    # Standard HuggingFace model

                outputs = self.llm(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                hidden_states = outputs.hidden_states[-1] if output_hidden_states else None
                logits = outputs.logits if hasattr(outputs, 'logits') else None
                # Log successful LLM forward
                if self.training and logits is not None:
                    logger.info(f"✓ LLM forward successful - using FlashAttention2 (dtype: {input_embeds.dtype}, logits shape: {logits.shape})")
            except Exception as e:
                logger.warning(f"✗ LLM forward failed: {e}, using fallback (dtype: {input_embeds.dtype})")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
                hidden_states = input_embeds
                logits = None

        hidden_states = hidden_states.reshape(B, -1, C)

        # Extract BEV and text outputs
        # selected has shape [N_total], where N_total is the total number of tokens across all batches
        # Since we reshaped hidden_states to (B, -1, C), we need to reshape selected to match
        try:
            # Try to use the original selected mask with advanced indexing
            # Reshape selected to (B, N) to match hidden_states
            seq_len = hidden_states.shape[1]
            selected_reshaped = selected[:B * seq_len].reshape(B, seq_len)

            # Now we can index properly
            hidden_bev_list = []
            text_emb_list = []
            for i in range(B):
                hidden_bev_list.append(hidden_states[i, selected_reshaped[i], :])
                text_emb_list.append(hidden_states[i, ~selected_reshaped[i], :])

            hidden_bev = torch.cat(hidden_bev_list, dim=0).to(torch.float32)
            text_emb = torch.cat(text_emb_list, dim=0).to(torch.float32)
        except Exception as e:
            # If that fails, process batch by batch with properly sliced masks
            bev_tokens = []
            text_tokens = []

            seq_len = hidden_states.shape[1]
            # Slice and reshape selected to match each batch's sequence length
            for i in range(B):
                # Get the portion of selected mask that corresponds to this batch
                batch_selected = selected[i * seq_len:(i + 1) * seq_len]
                bev_tokens.append(hidden_states[i, batch_selected, :])
                text_tokens.append(hidden_states[i, ~batch_selected, :])

            hidden_bev = torch.cat(bev_tokens, dim=0).to(torch.float32)
            text_emb = torch.cat(text_tokens, dim=0).to(torch.float32)

        # Handle learnable query output
        if self.num_learnable_query > 0 and learn_query is not None:
            try:
                # Reshape selected_query to match batch dimension
                seq_len = hidden_states.shape[1]
                # selected_query is a 1D mask, need to reshape it for each batch
                selected_query_reshaped = selected_query[:seq_len].unsqueeze(0).expand(B, seq_len)

                # Now we can index properly
                hidden_query = hidden_states[:, selected_query_reshaped, :].to(torch.float32)
                if logits is not None:
                    logits = logits[:, ~selected_query_reshaped, :]
            except Exception as e:
                # Fallback to batch processing
                query_tokens = []
                seq_len = hidden_states.shape[1]
                for i in range(B):
                    # Use only the relevant portion of selected_query for this batch
                    batch_selected_query = selected_query[:seq_len]
                    query_tokens.append(hidden_states[i, batch_selected_query])

                hidden_query = torch.cat(query_tokens, dim=0).to(torch.float32)

                # Remove query tokens from text embeddings
                text_emb = text_emb[:-selected_query.sum(), :]

                if logits is not None:
                    # For logits, we need to reshape and remove query tokens
                    logits = logits.reshape(B, -1, logits.shape[-1])
                    new_logits = []
                    for i in range(B):
                        batch_selected_query = selected_query[:seq_len]
                        new_logits.append(logits[i, ~batch_selected_query])
                    logits = torch.cat(new_logits, dim=0)

            # Apply MLP to hidden_query
            hidden_query = self.out_mlp(hidden_query)
        else:
            hidden_query = None

        # Compute chat loss if training
        loss = None
        if labels is not None and self.is_pretraining and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if ignore_flag:
                loss = loss * 0.0

        # Transform output back to BEV dimension
        output_bev = self.out_mlp(hidden_bev).reshape(B, H, W, -1)

        return {
            'out_bev': output_bev,
            'logits': logits,
            'chat_loss': loss,
            'out_query': hidden_query,
            'text_emb': text_emb,
        }

    def generate(
        self,
        bev=None,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=None,
        pad_token_id=None,
        **kwargs
    ):
        """
        Generate text from BEV features using the LLM (similar to HERMES).

        Args:
            bev: BEV features [B, H, W, C]
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            **kwargs: Additional arguments for generation

        Returns:
            dict: Generated token IDs and metadata
        """
        from transformers import GenerationConfig
        import torch

        device = next(self.parameters()).device

        # Process BEV features and prepare inputs
        text = None
        if input_ids is not None and hasattr(self.llm, 'get_input_embeddings'):
            input_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        else:
            # Create dummy input embeddings if no text provided
            B = bev.shape[0] if bev is not None else 1
            input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
            input_embeds = torch.zeros(B, 1, self.input_dim, device=device)

        B, N, C = input_embeds.shape
        _, H, W, _ = bev.shape
        bev_num_tokens = H * W

        input_embeds = input_embeds.reshape(B * N, C)
        bev = bev.reshape(B, bev_num_tokens, -1)

        # Transform BEV features to LLM dimension
        bev_embeds = self.in_mlp(bev).to(self.torch_dtype)
        input_ids = input_ids.reshape(B * N)

        # Inject BEV features into input embeddings
        selected = (input_ids == self.img_context_token_id)
        input_embeds[selected] = input_embeds[selected] * 0.0 + bev_embeds.reshape(-1, C).to(
            input_embeds.dtype
        )

        input_embeds = input_embeds.reshape(B, -1, C).to(self.torch_dtype)

        # Prepare generation config
        if eos_token_id is None and hasattr(self.tokenizer, 'eos_token_id'):
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None and hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        # Generate tokens using the LLM
        with torch.no_grad():
            # self.llm is already the language model (InternLM2ForCausalLM)
            # No need to access .language_model attribute
            model = self.llm

            # Prepare generation inputs
            generation_outputs = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs
            )

        return {
            'generation_output': generation_outputs,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

@HEADS.register_module()
class StreamLLMDecoder(LLMDecoder):
    """
    Streaming version of LLM decoder for temporal lane segmentation.
    """

    def __init__(self, *args, use_temporal_fusion=False, **kwargs):
        super(StreamLLMDecoder, self).__init__(*args, **kwargs)
        self.use_temporal_fusion = use_temporal_fusion

        if use_temporal_fusion:
            # Temporal fusion module
            self.temporal_fusion = nn.GRU(
                input_size=kwargs.get('input_dim', 256),
                hidden_size=kwargs.get('input_dim', 256),
                num_layers=2,
                batch_first=True
            )

    def forward(self, bev=None, text=None, prev_hidden=None,** kwargs):
        """
        Forward pass with temporal support.

        Args:
            bev: Current frame BEV features
            text: Text prompt
            prev_hidden: Previous hidden state for temporal fusion
            **kwargs: Other arguments
        """
        # Apply temporal fusion if enabled
        if self.use_temporal_fusion and prev_hidden is not None:
            B, H, W, C = bev.shape
            bev_flat = bev.view(B, -1, C)
            bev_fused, new_hidden = self.temporal_fusion(bev_flat, prev_hidden)
            bev = bev_fused.view(B, H, W, C)
        else:
            new_hidden = None

        # Call parent forward
        output = super().forward(bev=bev, text=text,** kwargs)

        # Add hidden state to output
        if self.use_temporal_fusion:
            output['hidden_state'] = new_hidden

        return output