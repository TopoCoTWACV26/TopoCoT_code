import json
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import cv2
import os
import mmcv
import pickle
import copy
from shapely.geometry import LineString

# ============== Configuration ==============
COLOR_DICT = {  # RGB [0, 1]
    'centerline': np.array([243, 90, 2]) / 255,
    'laneline': np.array([0, 32, 127]) / 255,
    'ped_crossing': np.array([55, 126, 71]) / 255,
    'road_boundary': np.array([220, 30, 0]) / 255,
}

LINE_PARAM = {
    0: {'color': COLOR_DICT['laneline'], 'alpha': 0.3, 'linestyle': ':'},       # none
    1: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': 'solid'},  # solid
    2: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': '--'},     # dashed
    'ped_crossing': {'color': COLOR_DICT['ped_crossing'], 'alpha': 1, 'linestyle': 'solid'},
    'road_boundary': {'color': COLOR_DICT['road_boundary'], 'alpha': 1, 'linestyle': 'solid'}
}

BOUNDARY_MAP = {
    "invisible": 0,
    "solid": 1,
    "dashed": 2,
}

split = 'val'
points_num = 10

# ============== Coordinate Conversion ==============
def opencv_to_bev(
    uv_or_uvz,
    W=500, H=1000, Z=40,
    x_min=-50, x_max=50,
    y_min=-25, y_max=25,
    z_min=-2.3, z_max=17
):
    """OpenCV (u, v[, z]) -> BEV coordinates"""
    pts = np.asarray(uv_or_uvz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"Expected shape (N, 2) or (N, 3), got {pts.shape}")

    u = pts[:, 0]
    v = pts[:, 1]

    y = u / W * (y_max - y_min) + y_min
    y = -y
    x = x_max - v / H * (x_max - x_min)

    if pts.shape[1] == 3:
        z_pix = pts[:, 2]
        z = z_pix / Z * (z_max - z_min) + z_min
        return np.stack([x, y, z], axis=1)
    else:
        return np.stack([x, y], axis=1)

def bev_to_opencv(
    xy_or_xyz,
    W=500, H=1000, Z=40,
    x_min=-50, x_max=50,
    y_min=-25, y_max=25,
    z_min=-2.3, z_max=17
):
    """
    BEV 坐标 -> OpenCV (u, v[, z])
    - 输入 (N, 2): (x, y)      -> 输出 (u, v)
    - 输入 (N, 3): (x, y, z)   -> 输出 (u, v, z)
    """
    pts = np.asarray(xy_or_xyz, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"Expected shape (N, 2) or (N, 3), got {pts.shape}")

    x = pts[:, 0]
    y = pts[:, 1]

    # BEV -> u, v (反向变换)
    y_bev = -y
    u = (y_bev - y_min) / (y_max - y_min) * W

    v = (x_max - x) / (x_max - x_min) * H

    if pts.shape[1] == 3:
        z = pts[:, 2]
        z_pix = (z - z_min) / (z_max - z_min) * Z
        return np.stack([u, v, z_pix], axis=1)
    else:
        return np.stack([u, v], axis=1)

def compute_left_right_boundaries_2d(centerline: np.ndarray, offset: float):
    """Compute left and right boundaries from centerline"""
    cl = np.asarray(centerline, dtype=np.float32)
    if cl.ndim != 2 or cl.shape[0] < 1 or cl.shape[1] != 2:
        raise ValueError(f"centerline must be (N,2), got {cl.shape}")

    if np.max(np.linalg.norm(cl - cl[0], axis=1)) < 1e-6:
        left_boundary = cl + np.array([0.0, offset], dtype=np.float32)
        right_boundary = cl - np.array([0.0, offset], dtype=np.float32)
        return left_boundary, right_boundary

    whole_direction = cl[-1] - cl[0]
    whole_direction /= (np.linalg.norm(whole_direction) + 1e-8)

    whole_orth = np.array([-whole_direction[1], whole_direction[0]], dtype=np.float32)
    if whole_orth[1] < 0:
        whole_orth = -whole_orth
    whole_orth /= (np.linalg.norm(whole_orth) + 1e-8)

    left_boundary = []
    right_boundary = []
    last_orth = whole_orth
    n = cl.shape[0]

    for i in range(n - 1):
        seg = cl[i + 1] - cl[i]
        seg_norm = np.linalg.norm(seg)
        if seg_norm < 1e-8:
            orth = last_orth
        else:
            d = seg / seg_norm
            orth = np.array([-d[1], d[0]], dtype=np.float32)
            if np.dot(orth, whole_orth) < 0:
                orth = -orth
            orth /= (np.linalg.norm(orth) + 1e-8)

        last_orth = orth
        left_boundary.append(cl[i] + orth * offset)
        right_boundary.append(cl[i] - orth * offset)

    left_boundary.append(cl[-1] + last_orth * offset)
    right_boundary.append(cl[-1] - last_orth * offset)

    return np.asarray(left_boundary), np.asarray(right_boundary)

def fix_pts_interpolate(lane, n_points):
    """Interpolate lane points to fixed number"""
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane

# ============== Load Annotations ==============
def load_annotations(ann_file):
    """Load annotation from a pkl file"""
    with open(ann_file, "rb") as f:
        data_infos = pickle.load(f)
    if isinstance(data_infos, dict):
        data_infos = list(data_infos.values())
    return data_infos

def lanes_to_annotation(lanes):
    """
    Convert lanes data to annotation format for visualization
    Uses raw image coordinates (u, v) directly
    """
    lane_segment = []
    areas = []

    for lane in lanes:
        cat = lane.get("category", "")
        lane_id = lane.get("id", "")
        # Support both 'point_2d' and 'point' keys
        pts = lane.get("point_2d") or lane.get("point", [])
        left_b = lane.get("left_boundary", "invisible")
        right_b = lane.get("right_boundary", "invisible")

        if not pts:
            continue

        # Check if this is a pedestrian crossing based on ID (PED0, PED1, etc.)
        if lane_id.startswith("PED"):
            # For PED: Treat as a simple lane segment (like LANE)
            # Draw PED as centerline only, just like prediction lanes
            lane_segment.append({
                "centerline": pts,
                "left_laneline": pts,
                "right_laneline": pts,
                "left_laneline_type": 1,  # solid line for visibility
                "right_laneline_type": 1,
                "confidence": 1,
                "is_ped": True,  # Mark as PED for coloring
            })
        else:
            # All non-PED items go to lane_segment (LANE0, LANE1, etc.)
            lane_segment.append({
                "centerline": pts,
                "left_laneline": pts,
                "right_laneline": pts,
                "left_laneline_type": BOUNDARY_MAP.get(left_b, 0),
                "right_laneline_type": BOUNDARY_MAP.get(right_b, 0),
                "confidence": 1,
                "is_ped": False,
            })

    return {
        "lane_segment": lane_segment,
        "area": areas,  # Empty for predictions, GT uses this
    }

def format_openlanev2_gt(data_infos):
    """Format GT data from pkl - convert BEV to image coordinates for visualization"""
    gt_dict = {}
    for idx in range(len(data_infos)):
        info = copy.deepcopy(data_infos[idx])
        key = (split, info['segment_id'], str(info['timestamp']))

        # Convert lane_segment centerline from BEV (x, y) to image coordinates (u, v)
        for lane_segment in info['annotation']['lane_segment']:
            centerline = lane_segment['centerline']
            if centerline.shape[1] == 3:
                centerline = centerline[:, :2]
            # Convert BEV to OpenCV coordinates for visualization
            lane_segment['centerline'] = bev_to_opencv(centerline)

            # Also convert left and right lanelines if they exist
            if 'left_laneline' in lane_segment:
                left_line = lane_segment['left_laneline']
                if left_line.shape[1] == 3:
                    left_line = left_line[:, :2]
                lane_segment['left_laneline'] = bev_to_opencv(left_line)

            if 'right_laneline' in lane_segment:
                right_line = lane_segment['right_laneline']
                if right_line.shape[1] == 3:
                    right_line = right_line[:, :2]
                lane_segment['right_laneline'] = bev_to_opencv(right_line)

        # Process ped_crossing areas - convert BEV to image coordinates
        areas = []
        for area in info['annotation']['area']:
            if area['category'] == 1:
                points = area['points']
                if points.shape[0] == 5:
                    dir_vector = points[1] - points[0]
                    direction = np.rad2deg(np.arctan2(dir_vector[1], dir_vector[0]))

                    if direction < -45 or direction > 135:
                        left_boundary = points[[2, 3]]
                        right_boundary = points[[1, 0]]
                    else:
                        left_boundary = points[[0, 1]]
                        right_boundary = points[[3, 2]]

                    left_boundary = fix_pts_interpolate(left_boundary, 10)
                    right_boundary = fix_pts_interpolate(right_boundary, 10)
                    centerline = (left_boundary + right_boundary) / 2

                    # Convert BEV to image coordinates
                    area['points'] = bev_to_opencv(centerline)
                    areas.append(area)
        info['annotation']['area'] = areas
        gt_dict[key] = info

    return gt_dict

# ============== Format Prediction Results ==============
# ============== Drawing Functions ==============
def _draw_line_align_coord(ax, line, label=None, color=None):
    """Draw a line on the axis"""
    points = np.asarray(line['points'])
    config = LINE_PARAM.get(line.get('linetype', 0), LINE_PARAM[0])
    if color is not None:
        config['color'] = color
    ax.plot(points[:, 0], points[:, 1], linewidth=2, zorder=1, label=label, **config)

def _draw_centerline_align_coord(ax, lane_centerline, label=None, iid=None, all_num=None, multi_color=False):
    """Draw centerline with optional color and label"""
    points = np.asarray(lane_centerline['points'])
    # Check if this is a PED crossing (for prediction data)
    is_ped = lane_centerline.get('is_ped', False)

    texts = []

    if multi_color:
        # For PED, use PED color, otherwise use colormap
        if is_ped:
            lane_name = f"<PED{iid}>"
            color = COLOR_DICT['ped_crossing']
        else:
            lane_name = f"{iid}"
            cmap = plt.get_cmap('tab20')
            color = cmap(iid / max(1, all_num - 1))

        ax.plot(points[:, 0], points[:, 1], color=color, alpha=1.0, linewidth=2, zorder=2, label=label)
        ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=10, zorder=3)
        ax.annotate('', xy=(points[-1, 0], points[-1, 1]),
                    xytext=(points[-2, 0], points[-2, 1]),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color=color), zorder=3)

        if len(points) > 1:
            mid = len(points) // 2
            x, y = points[mid, 0], points[mid, 1]
            x += np.random.uniform(-5, 5)
            y += np.random.uniform(-5, 5)

            txt = ax.text(x, y, lane_name,
                        fontsize=8, color=color,
                        ha='center', va='center', fontweight='bold',
                        zorder=10, clip_on=False,
                        path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
            texts.append(txt)
        return texts
    else:
        # Non-multi-color mode
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=1.0, linewidth=2, zorder=2, label=label)
        ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=10, zorder=2)
        ax.annotate('', xy=(points[-1, 0], points[-1, 1]),
                    xytext=(points[-2, 0], points[-2, 1]),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color=color), zorder=2)
        return None

def _draw_lane_segment_align_coord(ax, lane_segment, with_centerline, with_laneline, iid=None, all_num=None, multi_color=False):
    """Draw lane segment"""
    texts = None

    if with_centerline:
        texts = _draw_centerline_align_coord(
            ax, {'points': lane_segment['centerline']}, iid=iid, all_num=all_num, multi_color=multi_color
        )

    if with_laneline:
        _draw_line_align_coord(ax, {'points': lane_segment['left_laneline'], 'linetype': lane_segment['left_laneline_type']})
        _draw_line_align_coord(ax, {'points': lane_segment['right_laneline'], 'linetype': lane_segment['right_laneline_type']})

    return texts

def _draw_area_align_coord(ax, area, point_like=False, iid=None, all_num=None, multi_color=False):
    """Draw area (pedestrian crossing)"""
    texts = None
    if area['category'] == 1:
        if multi_color:
            texts = []
            lane_name = f"<PED{iid}>"
            cmap = plt.get_cmap('tab20')
            color = cmap(iid / max(1, all_num - 1))
            _draw_line_align_coord(ax, {'points': area['points'], 'linetype': 'ped_crossing'}, color=color)
            points = np.asarray(area['points'])
            ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=6, zorder=2)

            if len(points) > 1:
                mid = len(points) // 2
                x, y = points[mid, 0], points[mid, 1]
                x += np.random.uniform(-0.5, 0.5)
                y += np.random.uniform(-0.5, 0.5)

                txt = ax.text(x, y, lane_name,
                            fontsize=12, color=color,
                            ha='center', va='center', fontweight='bold',
                            zorder=10, clip_on=False,
                            path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
                texts.append(txt)
        else:
            _draw_line_align_coord(ax, {'points': area['points'], 'linetype': 'ped_crossing'})
            color = COLOR_DICT['ped_crossing']
            points = np.asarray(area['points'])
            ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=1, zorder=2)

    return texts

def draw_annotation_bev_align_coord(annotation, with_centerline=True, with_laneline=True, with_area=True, with_car=True, multi_color=False):
    """Draw annotation in BEV coordinate system"""
    fig, ax = plt.figure(figsize=(5, 10), dpi=100), plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 500])
    ax.invert_yaxis()
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Draw ego car
    if with_car:
        rect_width = 20
        rect_height = 40
        rect_x = 250 - rect_width / 2
        rect_y = 500 - rect_height / 2

        rect = plt.Rectangle(
            (rect_x, rect_y),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            zorder=3,
            label='Ego vehicle'
        )
        ax.add_patch(rect)

    # Draw lane segments
    all_texts = []
    for iid, lane_segment in enumerate(annotation.get('lane_segment', [])):
        texts = _draw_lane_segment_align_coord(
            ax, lane_segment, with_centerline, with_laneline, iid, len(annotation.get('lane_segment', [])), multi_color
        )
        if texts:
            all_texts.extend(texts)

    # Draw areas
    if with_area:
        for iid, area in enumerate(annotation.get('area', [])):
            texts = _draw_area_align_coord(
                ax, area, point_like=True, iid=iid, all_num=len(annotation.get('lane_segment', [])), multi_color=multi_color
            )
            if texts:
                all_texts.extend(texts)

    # Adjust text labels
    if multi_color and all_texts:
        objects = list(ax.lines) + list(ax.collections)
        adjust_text(
            all_texts, ax=ax,
            only_move={'text': 'xy'},
            expand_text=(1.8, 1.8),
            expand_points=(1.4, 1.4),
            expand_objects=(1.4, 1.4),
            force_text=(2.5, 2.5),
            force_points=(0.8, 0.8),
            force_objects=(1.0, 1.0),
            lim=1000,
            precision=0.001,
            add_objects=objects,
            arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3')
        )

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def put_label(img, text):
    """Add text label to image"""
    img = img.copy()
    cv2.putText(
        img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img

def put_ego_info(img, ego_data):
    """Add ego information text to image (multiple lines)"""
    img = img.copy()

    # Extract ego information
    current_lane = ego_data.get('current_lane', 'N/A')
    drive_direction = ego_data.get('drive_direction', 'N/A')
    downstream_lanes = ego_data.get('downstream_lanes', [])
    downstream_directions = ego_data.get('downstream_directions', [])

    # Format downstream lanes info
    downstream_info = ''
    if downstream_lanes:
        for lane, direction in zip(downstream_lanes, downstream_directions):
            downstream_info += f'{lane}: {direction}\n'
    else:
        downstream_info = 'N/A'

    # Build multi-line text
    text_lines = [
        f'ego:',
        f'  current_lane: {current_lane}',
        f'  drive_direction: {drive_direction}',
        f'  downstream_lanes: {downstream_info}',
    ]

    # Text configuration
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # Black
    line_height = 18

    # Starting position
    x, y = 10, 55

    # Draw each line
    for line in text_lines:
        if line.endswith('\n'):
            line = line[:-1]  # Remove newline for cv2
        if ':' in line and not line.startswith('ego:'):
            # Handle downstream lanes with multiple entries
            if 'downstream_lanes:' in line:
                cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                y += line_height
            else:
                # Individual downstream lane entries
                cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                y += line_height
        else:
            cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height

    return img

def render_lane_and_ped(annotation, with_car=True):
    """Render lane and ped images separately"""
    # Check if this is prediction data (has is_ped flag) or GT data (has separate area)
    has_ped_in_lane = any(seg.get('is_ped', False) for seg in annotation.get('lane_segment', []))
    has_separate_area = len(annotation.get('area', [])) > 0

    if has_ped_in_lane and not has_separate_area:
        # Prediction format: PEDs are in lane_segment with is_ped=True
        # Split into lane-only and ped-only
        lane_only_annotation = {
            'lane_segment': [seg for seg in annotation.get('lane_segment', []) if not seg.get('is_ped', False)],
        }
        ped_only_annotation = {
            'lane_segment': [seg for seg in annotation.get('lane_segment', []) if seg.get('is_ped', False)],
        }

        lane_img = draw_annotation_bev_align_coord(
            lane_only_annotation,
            with_centerline=True,
            with_laneline=False,
            with_area=False,
            with_car=with_car,
            multi_color=True,
        )[..., ::-1]  # RGB -> BGR

        ped_img = draw_annotation_bev_align_coord(
            ped_only_annotation,
            with_centerline=True,
            with_laneline=False,
            with_area=False,
            with_car=with_car,
            multi_color=True,
        )[..., ::-1]  # RGB -> BGR
    else:
        # GT format: lanes in lane_segment, PEDs in separate area
        lane_img = draw_annotation_bev_align_coord(
            annotation,
            with_centerline=True,
            with_laneline=False,
            with_area=False,
            with_car=with_car,
            multi_color=True,
        )[..., ::-1]  # RGB -> BGR

        ped_img = draw_annotation_bev_align_coord(
            annotation,
            with_centerline=False,
            with_laneline=False,
            with_area=True,
            with_car=with_car,
            multi_color=True,
        )[..., ::-1]  # RGB -> BGR

    return lane_img, ped_img

# ============== JSON Fix Function ==============
def fix_json_string(text):
    """
    尝试修复 JSON 字符串，处理常见的格式错误：
    - 第一个字符缺少 {
    - 最后一个字符缺少 }
    - 整个 JSON 被包在字符串中（以 " 开头和结尾）
    - 其他括号不匹配的情况
    """
    if not text or not isinstance(text, str):
        return text

    original_text = text  # 保存原始文本
    text = text.strip()
    if not text:
        return original_text

    # 首先尝试直接解析
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, RecursionError):
        pass

    # 情况1: 如果以 " 开头，可能是整个 JSON 被包在字符串中
    if text.startswith('"'):
        # 如果以 " 开头和结尾，尝试去掉外层的引号
        if text.endswith('"'):
            unquoted = text[1:-1]
            try:
                json.loads(unquoted)
                return unquoted
            except (json.JSONDecodeError, RecursionError):
                pass

        # 如果以 " 开头但第二个字符是 {，尝试去掉第一个引号
        if len(text) > 1 and text[1] == '{':
            unquoted = text[1:]
            try:
                json.loads(unquoted)
                return unquoted
            except (json.JSONDecodeError, RecursionError):
                pass

        # 如果以 " 开头但第二个字符不是 {，尝试在引号前添加 {
        if len(text) > 1 and text[1] != '{':
            try:
                fixed = '{' + text
                json.loads(fixed)
                return fixed
            except (json.JSONDecodeError, RecursionError):
                pass

    # 情况3: 检查第一个字符
    if not text.startswith('{'):
        # 如果第一个字符不是 {，尝试添加
        if text.startswith('"') or text.startswith('['):
            # 如果以 " 或 [ 开头，可能需要添加 {
            text = '{' + text
        elif text[0] in '([<':
            # 如果以其他括号开头，替换为 {
            text = '{' + text[1:]
        else:
            # 其他情况，直接添加 {
            text = '{' + text

    # 情况4: 检查最后一个字符
    if not text.endswith('}'):
        # 如果最后一个字符不是 }，尝试添加
        if text.endswith('"') or text.endswith(']'):
            # 如果以 " 或 ] 结尾，可能需要添加 }
            text = text + '}'
        elif text[-1] in ')]>':
            # 如果以其他括号结尾，替换为 }
            text = text[:-1] + '}'
        else:
            # 其他情况，直接添加 }
            text = text + '}'

    # 再次尝试解析，捕获所有可能的异常（包括递归错误）
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, RecursionError, Exception):
        # 如果还是失败（包括递归错误），返回原始文本（让调用者处理）
        return original_text

# ============== Main Visualization ==============
def visualize_single_file(txt_file_path, segment_id, timestamp, output_path):
    """
    Visualize a single llm_generated_text.txt file (prediction only)

    Args:
        txt_file_path: Path to the llm_generated_text.txt file
        segment_id: Segment ID (first level folder name)
        timestamp: Timestamp (second level folder name)
        output_path: Path to save the visualization image
    """
    try:
        # Read the file content
        with open(txt_file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Fix JSON string if needed
        content = fix_json_string(content)

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[Invalid JSON] segment_id={segment_id}, timestamp={timestamp}: {e}")
            return False

        # Extract lanes data from the JSON structure
        # The content is directly the lane data with lanes, topology, ego keys
        if 'lanes' not in data:
            print(f"[Invalid Format] segment_id={segment_id}, timestamp={timestamp}: Missing 'lanes' key")
            return False

        lane_results = data

        # Convert prediction lanes to annotation format
        pred_lanes = []
        for lane_single in lane_results.get('lanes', []):
            point_data = lane_single.get('point_2d') or lane_single.get('point', [])
            if point_data:
                pred_lanes.append({
                    'id': lane_single.get('id', ''),
                    'category': lane_single.get('category', 'centerline'),
                    'point': point_data,
                    'left_boundary': lane_single.get('left_boundary', 'invisible'),
                    'right_boundary': lane_single.get('right_boundary', 'invisible'),
                })

        if not pred_lanes:
            print(f"[No Lanes] segment_id={segment_id}, timestamp={timestamp}: No valid lane data found")
            return False

        pred_annotation = lanes_to_annotation(pred_lanes)

        # Render Pred only
        pred_lane_img, pred_ped_img = render_lane_and_ped(pred_annotation, with_car=True)

        # Add labels
        pred_lane_img = put_label(pred_lane_img, f"Pred-Lane ({segment_id})")
        pred_ped_img = put_label(pred_ped_img, f"Pred-Ped ({segment_id})")

        # Add ego information to pred images (if available)
        ego_data = lane_results.get('ego', {})
        if ego_data:
            pred_lane_img = put_ego_info(pred_lane_img, ego_data)
            pred_ped_img = put_ego_info(pred_ped_img, ego_data)

        # Ensure same size
        h, w = pred_lane_img.shape[:2]

        def _resize(img):
            return cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

        pred_ped_img = _resize(pred_ped_img)

        # Stitch 2x1 grid (vertical: Pred-Lane on top, Pred-Ped on bottom)
        grid = np.concatenate([pred_lane_img, pred_ped_img], axis=0)

        # Save
        mmcv.imwrite(grid, output_path)
        return True

    except Exception as e:
        print(f"[Error] segment_id={segment_id}, timestamp={timestamp}: {e}")
        return False

def visualize_test_output(test_output_dir):
    """
    Traverse test_output directory and visualize all llm_generated_text.txt files (prediction only)

    Args:
        test_output_dir: Path to test_output directory (e.g., /data_test/home/lizhen/yym/TopoWMChange/work_dirs/test_output)
    """
    test_output_dir = os.path.abspath(test_output_dir)

    if not os.path.exists(test_output_dir):
        print(f"Error: Directory does not exist: {test_output_dir}")
        return

    print(f"Scanning directory: {test_output_dir}")

    success_count = 0
    fail_count = 0
    total_count = 0

    # Iterate through segment_id folders (first level)
    for segment_id in os.listdir(test_output_dir):
        segment_path = os.path.join(test_output_dir, segment_id)

        # Skip if not a directory
        if not os.path.isdir(segment_path):
            continue

        # Iterate through timestamp folders (second level)
        for timestamp in os.listdir(segment_path):
            timestamp_path = os.path.join(segment_path, timestamp)

            # Skip if not a directory
            if not os.path.isdir(timestamp_path):
                continue

            # Look for llm_generated_text.txt
            txt_file_path = os.path.join(timestamp_path, "llm_generated_text.txt")

            if not os.path.exists(txt_file_path):
                continue

            total_count += 1

            # Output image path (same directory as the txt file)
            output_path = os.path.join(timestamp_path, "visualization1.jpg")

            # Visualize
            if visualize_single_file(txt_file_path, segment_id, timestamp, output_path):
                print(f"[{total_count}] Saved: {output_path}")
                success_count += 1
            else:
                fail_count += 1
   
    print(f"\n=== Summary ===")
    print(f"Total processed: {total_count}")
    print(f"Success: {success_count}")
    print(f"Failed (invalid format): {fail_count}")

# ============== Run ==============
if __name__ == "__main__":
    test_output_dir = "/data_test/home/lizhen/yym/TopoWMChange/work_dirs/test_output"

    visualize_test_output(test_output_dir)
