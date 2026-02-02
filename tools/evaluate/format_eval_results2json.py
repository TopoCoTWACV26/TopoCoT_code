import json
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe  # Optional: add stroke to text for clarity
from matplotlib.lines import Line2D
import cv2
import os
import mmcv
import imageio.v2 as imageio
import copy
import pickle
import sys
from shapely.geometry import LineString
from pathlib import Path
import sys, os




def opencv_to_bev(
    uv_or_uvz,
    W=500, H=1000, Z=40,
    x_min=-50, x_max=50,
    y_min=-25, y_max=25,
    z_min=-2.3, z_max=17
):
    """
    Convert OpenCV (u, v[, z]) coordinates to BEV coordinates.
    - Input (N, 2): (u, v)      → Output (x, y)
    - Input (N, 3): (u, v, z)   → Output (x, y, z)
    """
    pts = np.asarray(uv_or_uvz, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(
            f"Expected shape (N, 2) or (N, 3), got {pts.shape}"
        )

    u = pts[:, 0]
    v = pts[:, 1]

    # -------- u, v → x, y --------
    # y direction (note: y is negated in bev_to_opencv)
    y = u / W * (y_max - y_min) + y_min
    y = -y

    # x direction
    x = x_max - v / H * (x_max - x_min)

    # -------- Handle z if present --------
    if pts.shape[1] == 3:
        z_pix = pts[:, 2]
        z = z_pix / Z * (z_max - z_min) + z_min
        return np.stack([x, y, z], axis=1)
    else:
        return np.stack([x, y], axis=1)


def clean_and_validate_points(points):
    """
    Clean and validate point data, ensuring each point is in valid [x, y] format.
    Filter out invalid points (single element, incorrect format, etc.).
    
    Args:
        points: List of points, may be [[x1, y1], [x2, y2], ...] or [[x1, y1], [x2], ...] etc.
        
    Returns:
        np.ndarray: Cleaned point array with shape (N, 2), returns None if all points are invalid
    """
    if not points or len(points) == 0:
        return None
    
    cleaned_points = []
    for pt in points:
        # Convert to list for processing
        if isinstance(pt, (list, tuple, np.ndarray)):
            pt_list = list(pt) if not isinstance(pt, np.ndarray) else pt.tolist()
        else:
            continue
        
        # Check point validity
        if len(pt_list) >= 2:
            # Has at least two elements, take first two as x, y
            try:
                x, y = float(pt_list[0]), float(pt_list[1])
                cleaned_points.append([x, y])
            except (ValueError, TypeError):
                # Cannot convert to float, skip this point
                continue
        elif len(pt_list) == 1:
            # Only one element, likely format error, skip
            continue
        else:
            # Empty point, skip
            continue
    
    if len(cleaned_points) == 0:
        return None
    
    return np.array(cleaned_points, dtype=np.float32)

def fix_pts_interpolate(lane, n_points):
    """
    Interpolate lane points to ensure n_points points.
    If there is only one point or line segment length is 0, return that point repeated n_points times.
    """
    lane = np.asarray(lane)
    
    # Handle case with only one point or empty points
    if len(lane) == 0:
        raise ValueError("Empty lane points")
    
    if len(lane) == 1:
        # Only one point, return that point repeated n_points times
        single_point = lane[0]
        return np.tile(single_point, (n_points, 1))
    
    # Multiple points case, use LineString interpolation
    ls = LineString(lane)
    
    # If line segment length is 0 (all points coincide), also return first point repeated n_points times
    if ls.length < 1e-6:
        return np.tile(lane[0], (n_points, 1))
    
    distances = np.linspace(0, ls.length, n_points)
    lane_interpolated = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane_interpolated


def fix_json_string(text):
    """
    Attempt to fix JSON string, handling common format errors:
    - Missing { at the beginning
    - Missing } at the end
    - Entire JSON wrapped in quotes (starts and ends with ")
    - Other bracket mismatch cases
    """
    if not text or not isinstance(text, str):
        return text
    
    original_text = text  # Save original text
    text = text.strip()
    if not text:
        return original_text
    
    # First try direct parsing
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, RecursionError):
        pass
    
    # Case 1: If starts with ", the entire JSON might be wrapped in quotes
    if text.startswith('"'):
        # If starts and ends with ", try removing outer quotes
        if text.endswith('"'):
            unquoted = text[1:-1]
            try:
                json.loads(unquoted)
                return unquoted
            except (json.JSONDecodeError, RecursionError):
                pass
        
        # If starts with " but second character is {, try removing first quote
        if len(text) > 1 and text[1] == '{':
            unquoted = text[1:]
            try:
                json.loads(unquoted)
                return unquoted
            except (json.JSONDecodeError, RecursionError):
                pass
        
        # If starts with " but second character is not {, try adding { before quote
        if len(text) > 1 and text[1] != '{':
            try:
                fixed = '{' + text
                json.loads(fixed)
                return fixed
            except (json.JSONDecodeError, RecursionError):
                pass
    
    # Case 3: Check first character
    if not text.startswith('{'):
        # If first character is not {, try adding it
        if text.startswith('"') or text.startswith('['):
            # If starts with " or [, may need to add {
            text = '{' + text
        elif text[0] in '([<':
            # If starts with other brackets, replace with {
            text = '{' + text[1:]
        else:
            # Other cases, directly add {
            text = '{' + text
    
    # Case 4: Check last character
    if not text.endswith('}'):
        # If last character is not }, try adding it
        if text.endswith('"') or text.endswith(']'):
            # If ends with " or ], may need to add }
            text = text + '}'
        elif text[-1] in ')]>':
            # If ends with other brackets, replace with }
            text = text[:-1] + '}'
        else:
            # Other cases, directly add }
            text = text + '}'
    
    # Try parsing again, catch all possible exceptions (including recursion errors)
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, RecursionError, Exception):
        # If still fails (including recursion errors), return original text (let caller handle)
        return original_text

def collect_llm_results_from_test_output(test_output_dir):
    """
    Collect all llm_generated_text.txt files from test_output directory and aggregate into eval_result_examples.json format.
    
    Args:
        test_output_dir (str): Path to test_output directory
        
    Returns:
        dict: Result dictionary in the same format as eval_result_examples.json
    """
    result = {
        "detailed_results": {
            "predictions": []
        }
    }
    
    predictions = []
    sample_idx = 1
    
    # Traverse test_output directory
    test_output_path = Path(test_output_dir)
    if not test_output_path.exists():
        print(f"⚠️ Directory does not exist: {test_output_dir}")
        return result
    
    # Find all llm_generated_text.txt files
    # Path format: test_output/{segment_id}/{timestamp}/llm_generated_text.txt
    llm_files = list(test_output_path.glob("*/*/llm_generated_text.txt"))
    
    if not llm_files:
        print(f"⚠️ No llm_generated_text.txt files found in: {test_output_dir}")
        return result
    
    print(f"📂 Found {len(llm_files)} llm_generated_text.txt files")
    
    # Statistics
    stats = {
        "total_files": 0,
        "original_parse_success": 0,  # Original JSON parse success
        "fixed_parse_success": 0,     # Parse success after fixing
        "parse_failed": 0              # Final parse failure
    }
    
    for llm_file in sorted(llm_files):
        # Extract segment_id and timestamp from path
        # Path format: test_output/{segment_id}/{timestamp}/llm_generated_text.txt
        parts = llm_file.parts
        if len(parts) < 3:
            print(f"⚠️ Skipping invalid path: {llm_file}")
            continue
        
        segment_id = parts[-3]  # segment_id is the third-to-last part
        timestamp = parts[-2]   # timestamp is the second-to-last part
        
        # Read file content
        try:
            with open(llm_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"⚠️ Failed to read file {llm_file}: {e}")
            continue
        
        if not content:
            print(f"⚠️ File is empty: {llm_file}")
            continue
        
        stats["total_files"] += 1
        
        # First try to parse original content directly
        original_parse_success = False
        try:
            json.loads(content)
            original_parse_success = True
            stats["original_parse_success"] += 1
            answer_json_str = content
        except (json.JSONDecodeError, RecursionError):
            # Original parse failed, try fixing
            try:
                fixed_content = fix_json_string(content)
                
                # Check if fixed content differs from original
                is_fixed = (fixed_content != content)
                
                # Verify JSON is valid
                try:
                    json.loads(fixed_content)
                    stats["fixed_parse_success"] += 1
                    answer_json_str = fixed_content
                    # if is_fixed:
                    #     print(f"✅ JSON fixed successfully (segment_id={segment_id}, timestamp={timestamp})")
                except (json.JSONDecodeError, RecursionError) as e:
                    stats["parse_failed"] += 1
                    # print(f"⚠️ JSON parse failed (segment_id={segment_id}, timestamp={timestamp}): {type(e).__name__}: {str(e)[:100]}")
                    # print(f"   First 100 chars of original content: {content[:100]}")
                    # Still failed after fixing, save "error" marker
                    answer_json_str = "error"
            except (RecursionError, Exception) as e:
                stats["parse_failed"] += 1
                # print(f"⚠️ Error during JSON fixing (segment_id={segment_id}, timestamp={timestamp}): {type(e).__name__}: {str(e)[:100]}")
                # print(f"   First 100 chars of original content: {content[:100]}")
                # Error during fixing process, save "error" marker
                answer_json_str = "error"
        
        # Build prediction entry
        prediction_entry = {
            "sample_idx": sample_idx,
            "segment_id": segment_id,
            "timestamp": timestamp,
            "result": {
                "answer_json_str": answer_json_str,
                "cot": None  # llm_generated_text.txt has no cot
            }
        }
        
        predictions.append(prediction_entry)
        sample_idx += 1

    result["detailed_results"]["predictions"] = predictions
    
    # Output statistics
    print(f"\n📊 JSON parsing statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Original parse success: {stats['original_parse_success']} frames")
    print(f"   Fixed parse success: {stats['fixed_parse_success']} frames")
    print(f"   Parse failed: {stats['parse_failed']} frames")
    print(f"✅ Successfully collected {len(predictions)} prediction results")
    
    return result



split = 'val'
points_num = 10

# Collect llm_generated_text.txt files from test_output and aggregate
test_output_dir = "/data_test/home/lizhen/yym/TopoWMChange/work_dirs/test_output/"


aggregated_results = collect_llm_results_from_test_output(test_output_dir)


output_json_path = "./tools/evaluate/llm_results_aggregated.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
print(f"Saved to: {output_json_path}")




