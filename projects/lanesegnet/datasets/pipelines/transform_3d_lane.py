#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString

@PIPELINES.register_module()
class LaneSegmentParameterize3D(object):

    def __init__(self, method, method_para):
        method_list = ['fix_pts_interp']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):

        lanes = results['gt_lanes_3d']
        para_lanes = getattr(self, self.method)(lanes, **self.method_para)
        results['gt_lanes_3d'] = para_lanes
        return results

    def fix_pts_interp(self, input_data, n_points=11):

        lane_list = []
        for lane in input_data:
            ls = lane[1]
            distances = np.linspace(0, ls.length, n_points)
            left_line = np.array([ls.interpolate(distance).coords[0] for distance in distances])

            ls = lane[2]
            distances = np.linspace(0, ls.length, n_points)
            right_line = np.array([ls.interpolate(distance).coords[0] for distance in distances])

            centerline = (left_line + right_line) / 2.0

            line = np.concatenate([centerline.flatten(), left_line.flatten(), right_line.flatten()])
            lane_list.append(line)

        return np.array(lane_list, dtype=np.float32)

# @PIPELINES.register_module()
# class GenerateLaneSegmentMask(object):
#     """Generate mask ground truth for segmentation head
#     Args:
#         results (dict): Result dict from loading pipeline.
#     Returns:
#         dict: Instance mask gt is added into result dict.
#     """
#     def __init__(self, points_num, map_size=[-50, -25, 50, 25], bev_h=100, bev_w=200) -> None:
#         self.points_num = points_num
#         self.map_size = map_size  # [min_x, min_y, max_x, max_y]
#         self.bev_h = bev_h
#         self.bev_w = bev_w

#     def __call__(self,results):
#         results = self._generate_lanesegment_instance_mask(results)
#         return results

#     def _generate_lanesegment_instance_mask(self, results):
#         gt_lanes = np.array(results['gt_lanes_3d']).reshape(-1, 3, self.points_num, 3)
#         gt_left_lines = gt_lanes[:, 1]
#         gt_right_lines = gt_lanes[:, 2]

#         origin = np.array([self.bev_w // 2, self.bev_h // 2])
#         scale = np.array([self.bev_w / (self.map_size[2] - self.map_size[0]), self.bev_h / (self.map_size[3] - self.map_size[1])])

#         inst_masks = []
#         for idx, (left_line, right_line) in enumerate(zip(gt_left_lines, gt_right_lines)):

#             segment_boundary = np.concatenate((left_line, right_line[::-1], left_line[0:1]), axis=0)
#             mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)

#             draw_coor = (segment_boundary[:, :2] * scale + origin).astype(np.int32)
#             mask = cv2.fillPoly(mask, [draw_coor], 255)
#             bitMask = (mask / 255)
#             inst_masks.append(bitMask)

#         results['gt_instance_masks'] = inst_masks

#         return results

@PIPELINES.register_module()
class GenerateLaneSegmentMask(object):
    """Generate a single semantic BEV mask with lane segments.

    Background = 0, lane area = 1.
    """

    def __init__(self, points_num, map_size=[-50, -25, 50, 25], bev_h=100, bev_w=200) -> None:
        self.points_num = points_num
        self.map_size = map_size  # [min_x, min_y, max_x, max_y]
        self.bev_h = bev_h
        self.bev_w = bev_w

    def __call__(self, results):
        return self._generate_lanesegment_semantic_mask(results)

    def _generate_lanesegment_semantic_mask(self, results):
        gt_lanes = np.array(results['gt_lanes_3d']).reshape(-1, 3, self.points_num, 3)
        gt_left_lines = gt_lanes[:, 1]
        gt_right_lines = gt_lanes[:, 2]

        origin = np.array([self.bev_w // 2, self.bev_h // 2])
        scale = np.array([
            self.bev_w / (self.map_size[2] - self.map_size[0]),
            self.bev_h / (self.map_size[3] - self.map_size[1])
        ])

        # Start with background zeros; fill lane regions with 1.
        semantic_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)

        for left_line, right_line in zip(gt_left_lines, gt_right_lines):
            segment_boundary = np.concatenate((left_line, right_line[::-1], left_line[0:1]), axis=0)
            draw_coor = (segment_boundary[:, :2] * scale + origin).astype(np.int32)
            semantic_mask = cv2.fillPoly(semantic_mask, [draw_coor], 1)

        results['gt_instance_masks'] = semantic_mask
        return results

@PIPELINES.register_module()
class LoadConversation(object):
    """
    Load conversation data from bev_conv.json for LLM training.

    Args:
        conversation_dir (str): Root directory of conversation data (e.g., 'data/train_conv')
        with_system (bool): Whether to include system message. Default: True
    """

    def __init__(self, conversation_dir='/data_test/home/lizhen/yym/TopoWMChange/data/train_conv', with_system=True):
        self.conversation_dir = conversation_dir
        self.with_system = with_system
      
    def __call__(self, results):
        """
        Load conversation data and add to results dict.

        Args:
            results (dict): Result dict containing 'timestamp' and 'scene_token'

        Returns:
            dict: Updated result dict with conversation data
        """
        import json
        import os.path as osp

        # Get timestamp and scene_token from results
        # IMPORTANT: Use 'timestamp' key (actual timestamp like '3159734xxxxxxxxxxx')
        # NOT 'sample_idx' which is just the array index (0, 1, 2, ...)
        timestamp = results.get('timestamp')
        scene_token = results.get('scene_token')

        if timestamp is None or scene_token is None:
            # No conversation data available, return results as-is
            results['conversation'] = None
            return results

        # Construct path to bev_conv.json
        # Path format: conversation_dir/scene_token/timestamp/bev_conv.json
        conv_path = osp.join(self.conversation_dir, scene_token, str(timestamp), 'bev_conv.json')
    
        if not osp.exists(conv_path):
            # Conversation file not found
            results['conversation'] = None
            print(f"[DEBUG LoadConversation] Conversation file not found: {conv_path}")
            return results

        try:
            with open(conv_path, 'r') as f:
                conv_data = json.load(f)

            # conv_data is a list of conversations, take the first one
            if isinstance(conv_data, list) and len(conv_data) > 0:
                conv_item = conv_data[0]
            else:
                results['conversation'] = None
                return results
       
            # Extract system, prompt, and answer
            system = conv_item.get('system', '')
            prompt = conv_item.get('prompt', '')
            answer = conv_item.get('answer', '')

            # Store conversation data
            conversation = {
                'system': system if self.with_system else '',
                'prompt': prompt,
                'answer': answer
            }
            results['conversation'] = conversation

        except Exception as e:
            print(f"Warning: Failed to load conversation from {conv_path}: {e}")
            results['conversation'] = None

        return results

def build_coord_tokens(x_cap=500, y_cap=1000):
    xs = [f"<X_{i}>" for i in range(x_cap + 1)]
    ys = [f"<Y_{i}>" for i in range(y_cap + 1)]
    return xs + ys
@PIPELINES.register_module()
class FormatConversationForLLM(object):
    """
    Format conversation data for LLM input using tokenizer.

    Args:
        tokenizer_path (str): Path to the LLM tokenizer
        max_length (int): Maximum sequence length. Default: 2048
        with_system (bool): Whether to include system message. Default: True
        num_bev_tokens (int): Number of <IMG_CONTEXT> tokens to insert for BEV features. Default: 1250
        generate_text (bool): Whether to generate text during inference (like HERMES). Default: False
        max_new_tokens (int): Maximum number of tokens to generate during inference. Default: 512
    """

    def __init__(self, tokenizer_path='data/InternVL2-2B', max_length=2048, with_system=True, num_bev_tokens=1250,
                 generate_text=False, max_new_tokens=512):
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.with_system = with_system
        self.num_bev_tokens = num_bev_tokens  # Number of BEV tokens (H*W after downsampling)
        self.generate_text = generate_text  # Whether to generate text output like HERMES
        self.max_new_tokens = max_new_tokens

        # Lazy import of tokenizer (will be loaded when needed)
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                use_fast=False
            )
            self._tokenizer.tokenizer_path = self.tokenizer_path
            self._tokenizer.model_max_length = self.max_length

            # Add special tokens if not already added
            from projects.lanesegnet.models.modules.llm_decoder import IMG_CONTEXT_TOKEN
            special_tokens = [IMG_CONTEXT_TOKEN]
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            num_added = self._tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                print(f"Added {num_added} special tokens: {special_tokens}")
            # coord_tokens = build_coord_tokens(x_cap=500, y_cap=1000)
            # special_tokens_dict = {"additional_special_tokens": coord_tokens}
            # num_added = self._tokenizer.add_special_tokens(special_tokens_dict)

            # print("added:", num_added, "new vocab:", len(self._tokenizer))
      
        return self._tokenizer

    def _decode_generation(self, generation_output, input_ids):
        """
        Decode generation output to text (similar to HERMES).

        Args:
            generation_output: Generated token IDs from LLM
            input_ids: Original input token IDs

        Returns:
            str: Decoded text response
        """
        tokenizer = self._get_tokenizer()

        # Decode the generated tokens
        # Only decode the newly generated tokens (excluding input)
        if generation_output.shape[1] > input_ids.shape[1]:
            new_tokens = generation_output[:, input_ids.shape[1]:]
            response_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        else:
            # Fallback: decode everything
            response_text = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

        # Clean up response (remove common artifacts)
        # Remove template separators if present
        for sep_token in ['<|im_end|>', '\n\n', '<|im_start|>']:
            if sep_token in response_text:
                response_text = response_text.split(sep_token)[0].strip()

        return response_text

    def __call__(self, results):
        """
        Format conversation into input_ids, attention_mask, and labels.
        Also inserts <IMG_CONTEXT> tokens as placeholders for BEV feature injection.

        Args:
            results (dict): Result dict containing 'conversation' or 'llm_data'

        Returns:
            dict: Updated result dict with formatted LLM inputs
        """
        # Support both 'conversation' and 'llm_data' keys
        conversation = results.get('conversation')
        # print(f"[DEBUG FormatConversation] conversation key exists: {conversation is not None}")
        if conversation is None:
            conversation = results.get('llm_data')
            # print(f"[DEBUG FormatConversation] llm_data key exists: {conversation is not None}")

        if conversation is None:
            # No conversation data, create minimal input with <IMG_CONTEXT> tokens
            import torch
            tokenizer = self._get_tokenizer()
            from projects.lanesegnet.models.modules.llm_decoder import IMG_CONTEXT_TOKEN

            # Create input with only <IMG_CONTEXT> tokens
            img_context_tokens = [IMG_CONTEXT_TOKEN] * self.num_bev_tokens
            encoded = tokenizer(
                ' '.join(img_context_tokens),  # Space-separated tokens
                max_length=self.max_length,
                truncation=False,  # Don't truncate BEV tokens
                padding='max_length',
                return_tensors='pt'
            )

            results['llm_input_ids'] = encoded['input_ids'].squeeze(0)
            results['llm_attention_mask'] = encoded['attention_mask'].squeeze(0)
            results['llm_labels'] = torch.full((self.max_length,), -100, dtype=torch.long)  # All masked for inference
            return results

        try:
            tokenizer = self._get_tokenizer()
            from projects.lanesegnet.models.modules.llm_decoder import IMG_CONTEXT_TOKEN

            # Build the prompt text based on template
            # InternVL uses template: system + <IMG_CONTEXT> tokens + prompt + answer
            system = conversation.get('system', '')
            prompt = conversation.get('prompt', '')
            answer = conversation.get('answer', '')

            # Create <IMG_CONTEXT> placeholder tokens
            img_context_placeholder = ' '.join([IMG_CONTEXT_TOKEN] * self.num_bev_tokens)

            # Check if this is inference mode (no answer provided)
            is_inference = (not answer or answer.strip() == '') and self.generate_text

            if is_inference:
                # For inference: format input WITHOUT answer (like HERMES)
                if self.with_system and system:
                    prompt_text = f"{system}\n{img_context_placeholder}\n{prompt}"
                else:
                    prompt_text = f"{img_context_placeholder}\n{prompt}"

                # Tokenize the prompt only
                encoded = tokenizer(
                    prompt_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].squeeze(0)
                attention_mask = encoded['attention_mask'].squeeze(0)

                # Store for later generation during model forward pass
                results['llm_input_ids'] = input_ids
                results['llm_attention_mask'] = attention_mask
                results['llm_labels'] = torch.full((self.max_length,), -100, dtype=torch.long)
                results['llm_prompt_text'] = prompt_text  # Store for generation
                results['llm_generate_text'] = True  # Flag to trigger generation

                return results

            # Training mode: format with answer
            # Format: system message + <IMG_CONTEXT> tokens + user prompt + assistant response
            if self.with_system and system:
                # Include system message
                full_text = f"{system}\n{img_context_placeholder}\n{prompt}\n{answer}"
            else:
                # Only use prompt and answer with BEV tokens
                full_text = f"{img_context_placeholder}\n{prompt}\n{answer}"

            # Tokenize
            encoded = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)  # (L,)
            attention_mask = encoded['attention_mask'].squeeze(0)  # (L,)

            # Create labels for training (shift input_ids, mask prompt tokens)
            labels = input_ids.clone()

            # Find where answer starts (approximately after system, BEV tokens, and prompt)
            # IMPORTANT: Include system message in the mask calculation to avoid training on it
            if self.with_system and system:
                # Include system + BEV + prompt
                prompt_text = f"{system}\n{img_context_placeholder}\n{prompt}"
            else:
                # Only include BEV + prompt
                prompt_text = f"{img_context_placeholder}\n{prompt}"

            prompt_encoded = tokenizer(
                prompt_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            prompt_length = prompt_encoded['input_ids'].shape[1]

            # Mask system, BEV, and prompt tokens (set to -100 so they're ignored in loss)
            labels[:prompt_length] = -100

            # IMPORTANT: If answer is empty, we still need valid labels to ensure gradients flow
            # Unmask the last token to create a minimal loss (this token will be trained)
            if not answer or answer.strip() == '':
                # Find the last non-padding token (where attention_mask is 1)
                valid_length = attention_mask.sum().item()
                if valid_length > prompt_length:
                    # Unmask the last valid token to ensure gradients flow through LoRA
                    labels[valid_length - 1] = input_ids[valid_length - 1].item()

            # Store in results
            results['llm_input_ids'] = input_ids
            results['llm_attention_mask'] = attention_mask
            results['llm_labels'] = labels
           
        except Exception as e:
            print(f"Warning: Failed to format conversation: {e}")
            import traceback
            traceback.print_exc()
            # Create minimal input with <IMG_CONTEXT> tokens on error
            import torch
            tokenizer = self._get_tokenizer()
            from projects.lanesegnet.models.modules.llm_decoder import IMG_CONTEXT_TOKEN

            img_context_tokens = [IMG_CONTEXT_TOKEN] * self.num_bev_tokens
            encoded = tokenizer(
                ' '.join(img_context_tokens),
                max_length=self.max_length,
                truncation=False,
                padding='max_length',
                return_tensors='pt'
            )

            results['llm_input_ids'] = encoded['input_ids'].squeeze(0)
            results['llm_attention_mask'] = encoded['attention_mask'].squeeze(0)
            results['llm_labels'] = torch.full((self.max_length,), -100, dtype=torch.long)

        return results
