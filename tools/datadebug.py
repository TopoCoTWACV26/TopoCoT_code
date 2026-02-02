#!/usr/bin/env python3
"""
Data Debugging Script for train_conv Directory

This script scans all scenes and frames in /data_test/home/lizhen/yym/TopoWMChange/data/train_conv
to identify problematic data that will cause issues during training.

Problems checked:
1. Invalid JSON format or missing required fields (system, prompt, answer)
2. Excessive token length after adding BEV features (exceeds max_length=8192)

Usage:
    cd /data_test/home/lizhen/yym/TopoWMChange
    python tools/datadebug.py

Output:
    - Console: Progress information and summary
    - Log file: datadebug.log in the same directory as this script
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

class TeeOutput:
    """Redirect output to both console and log file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# Configuration
TRAIN_CONV_PATH = 'PATH_TO_CONVERSATION_DATA'
MAX_LENGTH = 10000  # Maximum sequence length
NUM_BEV_TOKENS = 1250  # Number of <IMG_CONTEXT> tokens (100*200//16 = 1250)
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
TOKENIZER_PATH = './InternVL2-2B'


class DataDebugger:
    """Debug and validate LLM conversation data."""

    def __init__(self, train_conv_path: str, tokenizer_path: str):
        self.train_conv_path = Path(train_conv_path)
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None

        # Statistics
        self.total_scenes = 0
        self.total_frames = 0
        self.invalid_format_frames = []
        self.too_long_frames = []
        self.scene_frame_counts = defaultdict(int)
        self.token_counts_all = []  # token counts for all valid frames
        self.token_count_failures = []  # frames where tokenization failed

    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer from {self.tokenizer_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Add special token
            special_tokens = [IMG_CONTEXT_TOKEN]
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            self._tokenizer.add_special_tokens(special_tokens_dict)
            print("Tokenizer loaded successfully.")

            
        return self._tokenizer

    def check_format(self, conv_data: List[Dict], scene_id: str, timestamp: str) -> Tuple[bool, str]:
        """
        Check if conversation data has valid format.

        Args:
            conv_data: List of conversation dictionaries
            scene_id: Scene identifier
            timestamp: Frame timestamp

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if data is a list and non-empty
        if not isinstance(conv_data, list):
            return False, f"Not a list, got type: {type(conv_data)}"

        if len(conv_data) == 0:
            return False, "Empty list"

        # Check first conversation entry
        entry = conv_data[0]

        # Check required fields
        required_fields = ['system', 'prompt', 'answer']
        missing_fields = []
        empty_fields = []

        for field in required_fields:
            if field not in entry:
                missing_fields.append(field)
            elif not entry[field] or not isinstance(entry[field], str):
                empty_fields.append(f"{field} (empty or not string)")

        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"

        if empty_fields:
            return False, f"Empty/invalid fields: {', '.join(empty_fields)}"

        return True, ""

    def check_length(self, conv_data: List[Dict], scene_id: str, timestamp: str) -> Tuple[bool, int, int]:
        """
        Check if tokenized conversation exceeds max_length.

        Args:
            conv_data: List of conversation dictionaries
            scene_id: Scene identifier
            timestamp: Frame timestamp

        Returns:
            Tuple of (is_within_limit, token_count, truncated_tokens)
        """
        try:
            tokenizer = self._get_tokenizer()
            entry = conv_data[0]

            system = entry.get('system', '')
            prompt = entry.get('prompt', '')
            answer = entry.get('answer', '')

            # Create full text with BEV placeholder tokens (same as training pipeline)
            img_context_placeholder = ' '.join([IMG_CONTEXT_TOKEN] * NUM_BEV_TOKENS)
            # full_text = f"{system}\n{img_context_placeholder}\n{prompt}\n{answer}"
            full_text = f"{system}\n{img_context_placeholder}\n{prompt}\n{answer}"

            # Tokenize
            encoded = tokenizer(
                full_text,
                max_length=MAX_LENGTH,
                truncation=False,  # Don't truncate to get actual length
                padding=False,
                return_tensors='pt'
            )

            token_count = encoded['input_ids'].shape[1]
            truncated_tokens = max(0, token_count - MAX_LENGTH)
            is_within_limit = token_count <= MAX_LENGTH

            return is_within_limit, token_count, truncated_tokens

        except Exception as e:
            return False, 0, 0

    def scan_directory(self):
        """Scan all scenes and frames in train_conv directory."""
        if not self.train_conv_path.exists():
            print(f"Error: Directory {self.train_conv_path} does not exist!")
            return

        print(f"\nScanning directory: {self.train_conv_path}")
        print("=" * 80)

        # First pass: count total frames for progress bar
        total_frames_est = 0
        for scene_dir in sorted(self.train_conv_path.iterdir()):
            if scene_dir.is_dir():
                for timestamp_dir in sorted(scene_dir.iterdir()):
                    if timestamp_dir.is_dir():
                        total_frames_est += 1

        print(f"Estimated total frames to scan: {total_frames_est}")
        print("-" * 80)

        # Iterate through all scene directories
        frame_count = 0
        progress_interval = max(1, total_frames_est // 100)  # Report progress every 1%

        for scene_dir in sorted(self.train_conv_path.iterdir()):
            if not scene_dir.is_dir():
                continue

            scene_id = scene_dir.name
            self.total_scenes += 1

            # Iterate through all timestamp directories
            for timestamp_dir in sorted(scene_dir.iterdir()):
                if not timestamp_dir.is_dir():
                    continue

                timestamp = timestamp_dir.name
                self.total_frames += 1
                self.scene_frame_counts[scene_id] += 1
                frame_count += 1

                # Progress reporting
                if frame_count % progress_interval == 0 or frame_count == total_frames_est:
                    progress = frame_count / total_frames_est * 100
                    print(f"Progress: {frame_count}/{total_frames_est} ({progress:.1f}%) - "
                          f"Errors: {len(self.invalid_format_frames)}, "
                          f"Too long: {len(self.too_long_frames)}")

                bev_conv_path = timestamp_dir / 'bev_conv.json'

                # Skip if file doesn't exist
                if not bev_conv_path.exists():
                    self.invalid_format_frames.append({
                        'scene': scene_id,
                        'timestamp': timestamp,
                        'error': 'File not found'
                    })
                    continue

                # Load and validate JSON
                try:
                    with open(bev_conv_path, 'r', encoding='utf-8') as f:
                        conv_data = json.load(f)
                except json.JSONDecodeError as e:
                    self.invalid_format_frames.append({
                        'scene': scene_id,
                        'timestamp': timestamp,
                        'error': f'JSON decode error: {str(e)}'
                    })
                    continue
                except Exception as e:
                    self.invalid_format_frames.append({
                        'scene': scene_id,
                        'timestamp': timestamp,
                        'error': f'Read error: {str(e)}'
                    })
                    continue

                # Check format
                is_valid, error_msg = self.check_format(conv_data, scene_id, timestamp)
                if not is_valid:
                    self.invalid_format_frames.append({
                        'scene': scene_id,
                        'timestamp': timestamp,
                        'error': error_msg
                    })
                    continue

                # Check length
                is_within_limit, token_count, truncated = self.check_length(
                    conv_data, scene_id, timestamp
                )

                if token_count <= 0:
                    self.token_count_failures.append({
                        'scene': scene_id,
                        'timestamp': timestamp,
                        'error': 'Tokenization failed'
                    })
                    continue

                self.token_counts_all.append(token_count)

                if not is_within_limit:
                    self.too_long_frames.append({
                        'scene': scene_id,
                        'timestamp': timestamp,
                        'token_count': token_count,
                        'truncated_tokens': truncated,
                        'excess_ratio': truncated / token_count * 100
                    })

        print("-" * 80)
        print("Scan complete!")

    def print_summary(self):
        """Print summary of scan results."""
        print("\n" + "=" * 80)
        print("SCAN SUMMARY")
        print("=" * 80)
        print(f"Total scenes:     {self.total_scenes}")
        print(f"Total frames:     {self.total_frames}")
        print(f"Valid frames:     {self.total_frames - len(self.invalid_format_frames)}")
        print(f"Invalid format:   {len(self.invalid_format_frames)}")
        print(f"Too long:         {len(self.too_long_frames)}")
        print(f"Tokenized frames: {len(self.token_counts_all)}")
        print(f"Token failures:   {len(self.token_count_failures)}")
        print("=" * 80)

        if self.token_counts_all:
            counts = self.token_counts_all
            counts_sorted = sorted(counts)
            p50 = counts_sorted[len(counts_sorted) // 2]
            p95 = counts_sorted[int(len(counts_sorted) * 0.95) - 1]
            print("Token length stats (all tokenized frames):")
            print(f"  min={min(counts)}, p50={p50}, p95={p95}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}")

    def print_invalid_frames(self):
        """Print details of frames with invalid format."""
        if not self.invalid_format_frames:
            print("\nNo format errors found.")
            return

        print("\n" + "=" * 80)
        print(f"INVALID FORMAT DATA ({len(self.invalid_format_frames)} frames)")
        print("=" * 80)
        print(f"{'Scene':<10} {'Timestamp':<25} {'Error'}")
        print("-" * 80)

        for frame in self.invalid_format_frames:
            print(f"{frame['scene']:<10} {frame['timestamp']:<25} {frame['error']}")

    def print_too_long_frames(self):
        """Print details of frames that are too long."""
        if not self.too_long_frames:
            print("\nNo length issues found.")
            return

        print("\n" + "=" * 80)
        print(f"TOO LONG DATA ({len(self.too_long_frames)} frames)")
        print("=" * 80)
        print(f"{'Scene':<10} {'Timestamp':<25} {'Tokens':<8} {'Excess':<10} {'Truncated'}")
        print("-" * 80)

        # Sort by token count (descending)
        sorted_frames = sorted(self.too_long_frames, key=lambda x: x['token_count'], reverse=True)

        for frame in sorted_frames:
            print(f"{frame['scene']:<10} {frame['timestamp']:<25} "
                  f"{frame['token_count']:<8} {frame['excess_ratio']:<10.2f}% "
                  f"{frame['truncated_tokens']} tokens")

        # Print statistics
        token_counts = [f['token_count'] for f in sorted_frames]
        if token_counts:
            print("\n" + "-" * 80)
            print(f"Max tokens:    {max(token_counts)}")
            print(f"Min tokens:    {min(token_counts)}")
            print(f"Avg tokens:    {sum(token_counts) / len(token_counts):.1f}")

    def save_token_distribution_plot(self, output_path: Path, bins: int = 60):
        """Save token count distribution histogram to output_path."""
        if not self.token_counts_all:
            print("\nNo token counts collected; skip plotting.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print("\nWARNING: matplotlib is not available; cannot save token distribution plot.")
            print(f"Reason: {e}")
            return

        counts = self.token_counts_all

        plt.figure(figsize=(10, 5))
        plt.hist(counts, bins=bins, color='#4C78A8', edgecolor='black', alpha=0.85)
        plt.title('Token Count Distribution')
        plt.xlabel('Token count')
        plt.ylabel('Number of frames')
        plt.grid(axis='y', linestyle='--', alpha=0.35)

        mean_val = sum(counts) / len(counts)
        plt.axvline(mean_val, color='#F58518', linestyle='--', linewidth=2, label=f'mean={mean_val:.1f}')
        plt.axvline(MAX_LENGTH, color='#E45756', linestyle='-', linewidth=2, label=f'max_length={MAX_LENGTH}')
        plt.legend()
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"\nSaved token distribution plot to: {output_path}")

    def print_scene_statistics(self):
        """Print frame count per scene."""
        print("\n" + "=" * 80)
        print("FRAME COUNT PER SCENE")
        print("=" * 80)

        # Sort by scene ID
        sorted_scenes = sorted(self.scene_frame_counts.items())

        print(f"{'Scene':<10} {'Frame Count'}")
        print("-" * 80)
        for scene_id, count in sorted_scenes:
            print(f"{scene_id:<10} {count}")

def main():
    """Main execution function."""
    # Setup log file in the same directory as this script
    script_dir = Path(__file__).parent
    log_file = script_dir / 'datadebug.log'

    # Redirect output to both console and log file
    tee = TeeOutput(log_file)
    sys.stdout = tee

    print("=" * 80)
    print("LLM Conversation Data Debugger")
    print(f"Log file: {log_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize debugger
    debugger = DataDebugger(
        train_conv_path=TRAIN_CONV_PATH,
        tokenizer_path=TOKENIZER_PATH
    )

    # Scan directory
    debugger.scan_directory()

    # Save token distribution plot
    plot_path = script_dir / 'token_count_distribution_rdp.png'
    debugger.save_token_distribution_plot(plot_path)

    # Print results
    debugger.print_summary()
    debugger.print_invalid_frames()
    debugger.print_too_long_frames()
    debugger.print_scene_statistics()

    # Return exit code
    has_errors = (
        len(debugger.invalid_format_frames) > 0
        or len(debugger.too_long_frames) > 0
        or len(debugger.token_count_failures) > 0
    )
    if has_errors:
        print("\n" + "!" * 80)
        print("WARNING: Found problematic data that may cause training issues!")
        print("!" * 80)
        retcode = 1
    else:
        print("\n" + "=" * 80)
        print("SUCCESS: All data is valid!")
        print("=" * 80)
        retcode = 0

    # Close log file and restore stdout
    tee.close()
    sys.stdout = tee.stdout

    return retcode

if __name__ == '__main__':
    sys.exit(main())
