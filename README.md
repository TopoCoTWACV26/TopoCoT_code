# TopoCoT

TopoCoT (Topology Chain-of-Thought) is a multi-stage training pipeline for lane topology prediction using LLM-based decoders. This project combines BEVFormer backbone with InternVL2 LLM decoder to achieve state-of-the-art performance on lane segment topology reasoning tasks.

**Challenge Homepage**: [TopoCoT @ WACV 2026](https://topocotwacv26.github.io/)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Stage 1: BEVFormer Pre-training](#stage-1-bevformer-pre-training)
  - [Stage 2: Adapter Training](#stage-2-adapter-training)
  - [Stage 3: LoRA Fine-tuning](#stage-3-lora-fine-tuning)
- [Testing and Evaluation](#testing-and-evaluation)
- [Configuration](#configuration)
- [Pre-trained Weights](#pre-trained-weights)
- [Project Structure](#project-structure)
- [Challenge](#challenge)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [TODO / Future Work](#todo--future-work)

## Features

- **Three-stage Training Pipeline**: Progressive training from BEV feature extraction to LLM fine-tuning
- **BEVFormer Backbone**: Leverages temporal and spatial attention for robust BEV feature learning
- **InternVL2 LLM Decoder**: Uses large language model for structured topology prediction
- **RDP Simplification**: Reduces token count by simplifying lane segments using Ramer-Douglas-Peucker algorithm
- **Temporal Streaming**: Supports temporal context with queue-based frame processing
- **CoT Support**: Chain-of-Thought reasoning capability (can be enabled)

## Installation

### 1. Create Conda Environment

```bash
conda create -n topocot python==3.9
conda activate topocot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download InternVL2 Model

```bash
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir InternVL2-2B
```

**Note**: The project includes OpenLane-V2 as an editable dependency (see `requirements.txt` line 139).

## Data Preparation

### 1. Download Dataset

Download the TopoCoT dataset from [ModelScope](https://modelscope.cn/datasets/yimingyang23/TopoCoT).

### 2. Build Training Conversations

Run the conversion script to generate training conversation data:

```bash
python convert_train_conv_rdp.py
```

This script:
- Loads lane annotations from the dataset
- Applies RDP simplification (epsilon=3.0) to reduce token count by simplifying lane segments
- Generates conversation format with system prompt, instruction, and JSON answer
- Outputs to `./data/train_conv_rdp/{segment_id}/{timestamp}/bev_conv.json`

**Note**: The repository includes a basic implementation without CoT (Chain-of-Thought). To enable CoT training, modify `convert_train_conv_rdp.py` to uncomment and use the CoT-related code sections.

### 3. Update Configuration Paths

Before training, update the paths in the config files:
- `projects/configs/topocot_olv2_stage1.py`: Update data paths
- `projects/configs/topocot_olv2_stage3.py`: Update `train_conv_path` to point to your `train_conv_rdp` directory

## Training

The training process consists of three stages. Each stage builds upon the previous one.

### Stage 1: BEVFormer Pre-training

**Purpose**: Pre-train BEVFormer for map segmentation

```bash
bash ./tools/dist_train.sh 8
```

- **Config**: `projects/configs/topocot_olv2_stage1.py`
- **Output**: `work_dirs/stream/stage1/`
- **Key Features**: 
  - BEV feature extraction with temporal attention
  - Lane segment detection and segmentation
  - No LLM components (pure perception)

### Stage 2: Adapter Training

**Purpose**: Align image encoder with LLM

```bash
bash ./tools/dist_train_stage2.sh 8
```

- **Config**: `projects/configs/topocot_olv2_stage2.py`
- **Loads**: Stage 1 checkpoint automatically
- **Output**: `work_dirs/stream/stage2/`
- **Key Features**:
  - Introduces LLM adapter components
  - Aligns BEV features with LLM token space
  - Trains adapter while keeping BEV backbone mostly frozen

### Stage 3: LoRA Fine-tuning

**Purpose**: Fine-tune LLM with LoRA for topology prediction

```bash
bash ./tools/dist_train_stage3.sh 8
```

- **Config**: `projects/configs/topocot_olv2_stage3.py`
- **Loads**: Stage 2 checkpoint automatically
- **Output**: `work_dirs/stream/stage3/`
- **Key Settings**:
  - LoRA rank: 64
  - LoRA alpha: 128
  - Max sequence length: 10500 tokens
  - Flash Attention 2 enabled

**Note**: The number `8` in the commands indicates the number of GPUs. Adjust according to your hardware.

## Testing and Evaluation

### 1. Run Inference

```bash
bash ./tools/dist_test.sh 8
```

- **Checkpoint**: Uses `work_dirs/stream/stage3/latest.pth` by default
- **Output**: Predictions saved to `./work_dirs/test_output/`

### 2. Convert Results to JSON

Convert the prediction results to JSON format for evaluation:

```bash
python ./tools/evaluate/format_eval_results2json.py
```

### 3. Upload to Evaluation Server

Upload the converted JSON results to the evaluation server:
- **Server**: https://huggingface.co/spaces/zhanchao019/test_server_wacv

### 4. Visualization

Visualize the prediction results:

```bash
python ./tools/visualize_test.py
```

## Configuration

### Memory Management

The `max_length` parameter in config files controls the maximum sequence length for training:

- **Default**: 10500 tokens (in `topocot_olv2_stage3.py`)
- **Trade-off**: Lower values save GPU memory but may truncate longer sequences, potentially affecting training quality
- **Location**: Set in `FormatConversationForLLM` pipeline step

### Token Distribution Analysis

Before training, check token distribution in your training data:

```bash
python tools/datadebug.py
```

**Important**: Update the following paths in `tools/datadebug.py`:
- `TRAIN_CONV_PATH`: Path to your `train_conv_rdp` directory
- `TOKENIZER_PATH`: Path to your `InternVL2-2B` directory

The script will:
- Check JSON format validity
- Analyze token count distribution
- Identify sequences exceeding `max_length`
- Generate a token distribution plot: `token_count_distribution_rdp.png`

## Pre-trained Weights

Pre-trained model weights are available at:
- **ModelScope**: https://modelscope.cn/models/yimingyang23/TopoCoT_weight

To use pre-trained weights, update the `load_from` parameter in your config file:

```python
load_from = './path/to/pretrained/checkpoint.pth'
```

## Project Structure

```
TopoCoT_code/
├── projects/              # Model implementations
│   ├── bevformer/        # BEVFormer backbone
│   ├── lanesegnet/       # Lane segmentation network
│   ├── plugin/           # Plugin modules
│   └── configs/          # Configuration files
├── tools/                # Training and evaluation scripts
│   ├── dist_train.sh     # Stage 1 training script
│   ├── dist_train_stage2.sh  # Stage 2 training script
│   ├── dist_train_stage3.sh  # Stage 3 training script
│   ├── dist_test.sh      # Testing script
│   ├── evaluate/         # Evaluation tools
│   └── visualize_test.py # Visualization script
├── data/                 # Dataset and generated data
│   ├── Trainset/         # Training data
│   ├── Testset/          # Test data
│   └── train_conv_rdp/   # Generated conversation data
├── OpenLane-V2-master/   # OpenLane-V2 dependency
├── convert_train_conv_rdp.py  # Data conversion script
└── requirements.txt      # Python dependencies
```

## Challenge

TopoCoT is associated with the **1st WACV 2026 Workshop on Robust and Generalized Lane Topology Understanding and HD Map Generation through CoT Design**.

- **Challenge Homepage**: [https://topocotwacv26.github.io/](https://topocotwacv26.github.io/)
- **Workshop Date**: March 7, 2026

The workshop provides a platform for industry experts and academics to exchange ideas about road understanding CoT (Chain-of-Thought) and its applications in autonomous driving. It includes regular and demo paper presentations, invited talks, and hosts a challenge based on open-source real-world CoT lane topology reasoning datasets.

## Acknowledgments

We acknowledge all the open-source contributors for the following projects that made this work possible:

- **[OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)**: The topology reasoning benchmark for unified 3D HD mapping
- **[BEVFormer](https://github.com/fundamentalvision/BEVFormer)**: Bird's-eye-view transformer for autonomous driving perception
- **[LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)**: Map learning with lane segment perception for autonomous driving

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2023openlane,
    title={Openlane-v2: A topology reasoning benchmark for unified 3d hd mapping},
    author={Wang, Huijie and Li, Tianyu and Li, Yang and Chen, Li and Sima, Chonghao and Liu, Zhenbo and Wang, Bangjun and Jia, Peijin and Wang, Yuting and Jiang, Shengyin and others},
    journal={Advances in Neural Information Processing Systems},
    volume={36},
    pages={18873--18884},
    year={2023}
}

@article{li2023lanesegnet,
    title={Lanesegnet: Map learning with lane segment perception for autonomous driving},
    author={Li, Tianyu and Jia, Peijin and Wang, Bangjun and Chen, Li and Jiang, Kun and Yan, Junchi and Li, Hongyang},
    journal={arXiv preprint arXiv:2312.16108},
    year={2023}
}

@article{yang2025topo2seq,
    title={Topo2Seq: Enhanced Topology Reasoning via Topology Sequence Learning},
    author={Yang, Yiming and Luo, Yueru and He, Bingkun and Li, Erlong and Cao, Zhipeng and Zheng, Chao and Mei, Shuqi and Li, Zhen},
    journal={AAAI 2025},
    year={2025}
}

@article{yang2025topostreamer,
    title={TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving},
    author={Yang, Yiming and Luo, Yueru and He, Bingkun and Lin, Hongbin and Fu, Suzhong and Zheng, Chao and Cao, Zhipeng and Li, Erlong and Yan, Chao and Cui, Shuguang and others},
    journal={arXiv preprint arXiv:2507.00709},
    year={2025}
}

@article{yang2025fastopowm,
    title={FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models},
    author={Yang, Yiming and Lin, Hongbin and Luo, Yueru and Fu, Suzhong and Zheng, Chao and Yan, Xinrui and Mei, Shuqi and Tang, Kun and Cui, Shuguang and Li, Zhen},
    journal={arXiv preprint arXiv:2507.23325},
    year={2025}
}

@article{luo2025reltopo,
    title={RelTopo: Enhancing Relational Modeling for Driving Scene Topology Reasoning},
    author={Luo, Yueru and Zhou, Changqing and Yang, Yiming and Li, Erlong and Zheng, Chao and Mei, Shuqi and Cui, Shuguang and Li, Zhen},
    journal={arXiv preprint arXiv:2506.13553},
    year={2025}
}
```

## TODO / Future Work

- [ ] Support for higher versions of torch and transformers
- [ ] Support for more models (e.g., Qwen3VL)
