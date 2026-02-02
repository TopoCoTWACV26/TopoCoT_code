import json
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe  # 可选：给文字加描边更清晰
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

from evaluate_custom_modified import lanesegnet_evaluate

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
BEV_RANGE = [-50, 50, -25, 25]

BOUNDARY_MAP = {
    "invisible": 0,
    "solid": 1,
    "dashed": 2,
}


scene_id = ['10145', '10125', '10104', '10018', '10019', '10090', '10141', '10016', '10091', '10116', '10127', '10051', '10066', '10078', '10102', '10029', '10069', '10087', '10043', '10083', '10138', '10134', '10026', '10011', '10079', '10130', '10093', '10117', '10131', '10132', '10006', '10118', '10084', '10049', '10147', '10114', '10062', '10082', '10033', '10148', '10144', '10017', '10124', '10094', '10120', '10068', '10143', '10031', '10008', '10115']

# 是否只计算select场景的指标
USE_SELECT_SCENES = False  # 设置为True则只计算scene_id列表中的场景，False则计算所有场景

def opencv_to_bev(
    uv_or_uvz,
    W=500, H=1000, Z=40,
    x_min=-50, x_max=50,
    y_min=-25, y_max=25,
    z_min=-2.3, z_max=17
):
    """
    OpenCV (u, v[, z]) → BEV 坐标
    - 输入 (N, 2): (u, v)      → 输出 (x, y)
    - 输入 (N, 3): (u, v, z)   → 输出 (x, y, z)
    """
    pts = np.asarray(uv_or_uvz, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(
            f"Expected shape (N, 2) or (N, 3), got {pts.shape}"
        )

    u = pts[:, 0]
    v = pts[:, 1]

    # -------- u, v → x, y --------
    # y 方向（注意 bev_to_opencv 中 y 被取了负号）
    y = u / W * (y_max - y_min) + y_min
    y = -y

    # x 方向
    x = x_max - v / H * (x_max - x_min)

    # -------- 是否处理 z --------
    if pts.shape[1] == 3:
        z_pix = pts[:, 2]
        z = z_pix / Z * (z_max - z_min) + z_min
        return np.stack([x, y, z], axis=1)
    else:
        return np.stack([x, y], axis=1)

def compute_left_right_boundaries_2d(centerline: np.ndarray, offset: float):
    cl = np.asarray(centerline, dtype=np.float32)
    if cl.ndim != 2 or cl.shape[0] < 1 or cl.shape[1] != 2:
        raise ValueError(f"centerline must be (N,2), got {cl.shape}")

    # ===== 0. 退化情况：所有点都一样 =====
    if np.max(np.linalg.norm(cl - cl[0], axis=1)) < 1e-6:
        # 👉 直接用固定方向偏移（y 轴）
        left_boundary  = cl + np.array([0.0,  offset], dtype=np.float32)
        right_boundary = cl - np.array([0.0,  offset], dtype=np.float32)
        return left_boundary, right_boundary

    # ===== 1. 正常情况：整体方向 =====
    whole_direction = cl[-1] - cl[0]
    whole_direction /= (np.linalg.norm(whole_direction) + 1e-8)

    # 左法向
    whole_orth = np.array([-whole_direction[1], whole_direction[0]], dtype=np.float32)
    if whole_orth[1] < 0:
        whole_orth = -whole_orth
    whole_orth /= (np.linalg.norm(whole_orth) + 1e-8)

    left_boundary = []
    right_boundary = []
    last_orth = whole_orth
    n = cl.shape[0]

    # ===== 2. 逐段 =====
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

    # ===== 3. 末尾点 =====
    left_boundary.append(cl[-1] + last_orth * offset)
    right_boundary.append(cl[-1] - last_orth * offset)

    return np.asarray(left_boundary), np.asarray(right_boundary)

def clean_and_validate_points(points):
    """
    清理和验证点数据，确保每个点都是有效的 [x, y] 格式。
    过滤掉无效的点（只有一个元素、格式不正确等）。
    
    Args:
        points: 点列表，可能是 [[x1, y1], [x2, y2], ...] 或 [[x1, y1], [x2], ...] 等格式
        
    Returns:
        np.ndarray: 清理后的点数组，形状为 (N, 2)，如果所有点都无效则返回 None
    """
    if not points or len(points) == 0:
        return None
    
    cleaned_points = []
    for pt in points:
        # 转换为列表以便处理
        if isinstance(pt, (list, tuple, np.ndarray)):
            pt_list = list(pt) if not isinstance(pt, np.ndarray) else pt.tolist()
        else:
            continue
        
        # 检查点的有效性
        if len(pt_list) >= 2:
            # 有至少两个元素，取前两个作为 x, y
            try:
                x, y = float(pt_list[0]), float(pt_list[1])
                cleaned_points.append([x, y])
            except (ValueError, TypeError):
                # 无法转换为浮点数，跳过这个点
                continue
        elif len(pt_list) == 1:
            # 只有一个元素，可能是格式错误，跳过
            continue
        else:
            # 空点，跳过
            continue
    
    if len(cleaned_points) == 0:
        return None
    
    return np.array(cleaned_points, dtype=np.float32)

def fix_pts_interpolate(lane, n_points):
    """
    对车道线点进行插值，确保有 n_points 个点。
    如果只有一个点或线段长度为0，返回该点重复 n_points 次。
    """
    lane = np.asarray(lane)
    
    # 处理只有一个点或空点的情况
    if len(lane) == 0:
        raise ValueError("Empty lane points")
    
    if len(lane) == 1:
        # 只有一个点，返回该点重复 n_points 次
        single_point = lane[0]
        return np.tile(single_point, (n_points, 1))
    
    # 多个点的情况，使用 LineString 插值
    ls = LineString(lane)
    
    # 如果线段长度为0（所有点重合），也返回第一个点重复 n_points 次
    if ls.length < 1e-6:
        return np.tile(lane[0], (n_points, 1))
    
    distances = np.linspace(0, ls.length, n_points)
    lane_interpolated = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane_interpolated

def lanes_to_annotation(lanes):
    """
    把 eval_results 里的 lanes(list) 转成 draw_annotation_bev_align_coord 用的 annotation 结构
    - centerline: 直接用 point_2d
    - ped_cross: 塞到 area 里，category = 1, points=[point_2d]
    """
    lane_segment = []
    areas = []

    for lane in lanes:
        cat = lane.get("category", "")
        pts = lane.get("point_2d", [])
        left_b = lane.get("left_boundary", "invisible")
        right_b = lane.get("right_boundary", "invisible")

        if not pts:
            continue

        if cat == "centerline":
            lane_segment.append(
                {
                    "centerline": pts,
                    # 这里用同一条线代替左右边界（只是为了可视化，严格几何你可以之后再细化）
                    "left_laneline": pts,
                    "right_laneline": pts,
                    "left_laneline_type": BOUNDARY_MAP.get(left_b, 0),
                    "right_laneline_type": BOUNDARY_MAP.get(right_b, 0),
                    "confidence": 1,
                }
            )
        elif cat == "ped_cross":
            areas.append(
                {
                    "category": 1,    # 你的 _draw_area_align_coord 里用 category == 1 判断行人过街
                    "points": [pts],  # 保持和你那边一致：area['points'][0]
                    "confidence": 1,
                }
            )
        else:
            # 其他类别先忽略
            continue

    return {
        "lane_segment": lane_segment,
        "area": areas,
    }
def _draw_line_align_coord(ax, line, label=None, multi_color_laneline = False, color = None):

    if multi_color_laneline:

        points = np.asarray(line['points'])

        config = LINE_PARAM[line['linetype']]
        config['color'] = color
        ax.plot(points[:, 0], points[:, 1], linewidth=2, zorder=1, label=label, **config)
    else:
        points = np.asarray(line['points'])

        config = LINE_PARAM[line['linetype']]
        ax.plot(points[:, 0], points[:, 1], linewidth=2, zorder=1, label=label, **config)

def _draw_centerline_align_coord(ax, lane_centerline, label=None, iid = None, all_num = None, multi_color_centerline = False):
    points = np.asarray(lane_centerline['points'])

    color = COLOR_DICT['centerline']
    texts = []
    if multi_color_centerline:
        
        lane_name=f"{iid}"
        cmap = plt.get_cmap('tab20') 
        color = cmap(iid / max(1, all_num-1))
        # draw line
        try:
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=1.0, linewidth=2, zorder=2, label=label)
        except Exception as e:
            print(f"⚠️ Error plotting centerline: {e}")
            pass
        # draw start and end vertex
        ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=10, zorder=3)
        # draw arrow

        ax.annotate('', xy=(points[-1, 0], points[-1, 1]),
                    xytext=(points[-2, 0], points[-2, 1]),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color=color), zorder=3)
        # ax.text(points[4,1], points[4,0], lane_name, fontsize=8, color=color,
        #          ha='center', va='center', fontweight='bold', zorder=10, clip_on=False)
        mid = len(points)//2
        x, y = points[mid,0], points[mid,1]
        t = points[min(mid+1,len(points)-1)] - points[max(mid-1,0)]
        nx, ny = -t[0], t[1]
        nrm = np.hypot(nx,ny)+1e-6
        # x += 2.0*nx/nrm; y += 2.0*ny/nrm          # 法向偏移 2 个单位
        x += np.random.uniform(-5,5)          # 轻微随机初始化
        y += np.random.uniform(-5,5)

        txt = ax.text(x, y, lane_name,
                    fontsize=8, color=color,  # 统一黑字可读性更好
                    ha='center', va='center', fontweight='bold',
                    zorder=10, clip_on=False, path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])  # 白描边
        texts.append(txt)  # 收集
        return texts
    else:
        # draw line
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=1.0, linewidth=2, zorder=2, label=label)
        # draw start and end vertex
        ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=10, zorder=2)
        # draw arrow

        ax.annotate('', xy=(points[-1, 0], points[-1, 1]),
                    xytext=(points[-2, 0], points[-2, 1]),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color=color), zorder=2)
        return None

def _draw_lane_segment_alig_coord(ax, lane_segment, with_centerline, with_laneline, iid=None, all_num=None, multi_color_centerline = False, multi_color_laneline=False):
    texts = None

    if with_centerline:
        texts = _draw_centerline_align_coord(ax, {'points': lane_segment['centerline']}, label='Centerline', iid= iid, all_num=all_num, multi_color_centerline= multi_color_centerline)
 
    if with_laneline:
        line_type = {0:'Invisible boundary line', 1:'Solid boundary line', 2:'Dashed boundary line'}
        if multi_color_laneline:
            texts = []
            lane_name=f"{iid}"
            cmap = plt.get_cmap('tab20') 
            color = cmap(iid / max(1, all_num-1))

            _draw_line_align_coord(ax, {'points': lane_segment['left_laneline'], 'linetype': lane_segment['left_laneline_type']}, label = line_type[lane_segment['left_laneline_type']], multi_color_laneline = multi_color_laneline, color = color)
            _draw_line_align_coord(ax, {'points': lane_segment['right_laneline'], 'linetype': lane_segment['right_laneline_type']}, label = line_type[lane_segment['right_laneline_type']], multi_color_laneline = multi_color_laneline, color = color)
            
            left_pts = np.asarray(lane_segment['left_laneline'])
   
            right_pts = np.asarray(lane_segment['right_laneline'])

            middle_pts = (left_pts+right_pts)/2 
        
            mid = len(middle_pts)//2
            x, y = middle_pts[mid,0], middle_pts[mid,1]
            t = middle_pts[min(mid+1,len(middle_pts)-1)] - middle_pts[max(mid-1,0)]
            nx, ny = -t[0], t[1]
            nrm = np.hypot(nx,ny)+1e-6
            # x += 2.0*nx/nrm; y += 2.0*ny/nrm          # 法向偏移 2 个单位
            x += np.random.uniform(5,5)          # 轻微随机初始化
            y += np.random.uniform(5,5)

            txt = ax.text(x, y, lane_name,
                        fontsize=8, color=color,  # 统一黑字可读性更好
                        ha='center', va='center', fontweight='bold',
                        zorder=10, clip_on=False, path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])  # 白描边
            texts.append(txt)  # 收集

        else:
            _draw_line_align_coord(ax, {'points': lane_segment['left_laneline'], 'linetype': lane_segment['left_laneline_type']}, label = line_type[lane_segment['left_laneline_type']], multi_color_laneline = multi_color_laneline)
            _draw_line_align_coord(ax, {'points': lane_segment['right_laneline'], 'linetype': lane_segment['right_laneline_type']}, label = line_type[lane_segment['right_laneline_type']], multi_color_laneline = multi_color_laneline)
    return texts

def _draw_area_align_coord(ax, area, point_like=False, multi_color_area=False, iid=None, all_num=None):
    texts = None
    if point_like == True:
        if area['category'] == 1:  # ped crossing with lane segment style.

            if multi_color_area:
                texts = []
                lane_name=f"<PED{iid}>"
                cmap = plt.get_cmap('tab20') 
                color = cmap(iid / max(1, all_num-1))
                _draw_line_align_coord(ax, {'points': area['points'][0], 'linetype': 'ped_crossing'}, label='Pedestrian crossing', multi_color_laneline = multi_color_area, color = color)
                # color = COLOR_DICT['ped_crossing']
                points =  np.asarray(area['points'][0])

                ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=6, zorder=2)

                mid = len(points)//2
                x, y = points[mid,0], points[mid,1]
                t = points[min(mid+1,len(points)-1)] - points[max(mid-1,0)]
                nx, ny = -t[0], t[1]
                nrm = np.hypot(nx,ny)+1e-6
                # x += 2.0*nx/nrm; y += 2.0*ny/nrm          # 法向偏移 2 个单位
                x += np.random.uniform(-0.5,0.5)          # 轻微随机初始化
                y += np.random.uniform(-0.5,0.5)

                txt = ax.text(x, y, lane_name,
                            fontsize=12, color=color,  # 统一黑字可读性更好
                            ha='center', va='center', fontweight='bold',
                            zorder=10, clip_on=False, path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])  # 白描边
                texts.append(txt)  # 收集

            else:
                _draw_line_align_coord(ax, {'points': area['points'][0], 'linetype': 'ped_crossing'}, label='Pedestrian crossing')
                color = COLOR_DICT['ped_crossing']
                points = np.asarray(area['points'][0])
 
                ax.scatter(points[[0, -1], 0], points[[0, -1], 1], color=color, s=1, zorder=2)

    else:
        if area['category'] == 1:  # ped crossing with lane segment style.
            _draw_line_align_coord(ax, {'points': area['points'][0], 'linetype': 'ped_crossing'})

    return texts

def draw_annotation_bev_align_coord(annotation, with_centerline=True, with_laneline=True, with_area=True, with_car=False, point_like=False, with_nav=False, multi_color_centerline = False, multi_color_laneline = False, multi_color_area = False, rexy=False):

    fig, ax = plt.figure(figsize=(5, 10), dpi=100), plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 500])

    ax.invert_yaxis()
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for iid, lane_segment in enumerate(annotation['lane_segment']):

        texts = _draw_lane_segment_alig_coord(ax, lane_segment, with_centerline, with_laneline, iid, len(annotation['lane_segment']), multi_color_centerline, multi_color_laneline)
        objects = []
    for line in ax.lines:
        objects.append(line)
    for coll in ax.collections:
        objects.append(coll)

    if multi_color_centerline:
    
        adjust_text(
        texts, ax=ax,
        only_move={'text': 'xy'},        # 允许文本在 x/y 两方向移动
        expand_text=(1.8, 1.8),         # 文本-文本最小间距放大
        expand_points=(1.4, 1.4),       # 若有散点也能避
        expand_objects=(1.4, 1.4),      # 文本-线/面 间距放大
        force_text=(2.5, 2.5),          # 推开力度更大
        force_points=(0.8, 0.8),
        force_objects=(1.0, 1.0),
        lim=1000,   
        precision=0.001, 
        add_objects=objects,           
        arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3')  # 需要时给挪远的标签拉细线
        )
        
    if with_area:
        for iid, area in enumerate(annotation['area']):
  
            texts = _draw_area_align_coord(ax, area, point_like, multi_color_area, iid, len(annotation['lane_segment']))
        
        objects = []
        for line in ax.lines:
            objects.append(line)
        for coll in ax.collections:
            objects.append(coll)

        if multi_color_area:
            adjust_text(
            texts, ax=ax,
            only_move={'text': 'xy'},        # 允许文本在 x/y 两方向移动
            expand_text=(2.0, 2.0),          # 文本间最小间隔放大系数
            force_text=(1.0, 1.0), # 推开力度（越大越分散，迭代更慢）
            lim=500,   
            precision=0.01, 
            add_objects=objects,           
            arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3')  # 需要时给挪远的标签拉细线
            )

    if with_car:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        # 绘制红色边框的矩形（您图片中的主要图案）
        # 定义矩形参数

        rect_width = 20   # 矩形宽度
        rect_height = 40  # 矩形高度
        line_length = 20 # 顶部竖线长度
        
        # 计算矩形左下角坐标（使其居中）
        rect_x =  250-rect_width / 2
        rect_y = 500-rect_height / 2
        
        # 绘制红色边框矩形（透明填充）
        rect = plt.Rectangle(
            (rect_x, rect_y), 
            rect_width, 
            rect_height,
            linewidth=2,        # 边框线宽
            edgecolor='red',      # 红色边框
            facecolor='none',
            zorder=3, label = 'Ego vehicle'     # 透明填充

        )
       
        ax.add_patch(rect)

    fig.canvas.draw()  # 关键：渲染一次，才能获取 legend 的位置

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def render_lane_and_ped(annotation, with_car=True):
    """
    对一个 annotation 生成两张图：
    1. 只画 lane（centerline）
    2. 只画 ped crossing（area, category == 1）
    返回 (lane_img, ped_img)，都是 BGR 格式
    """

    # Lane 图
    lane_img = draw_annotation_bev_align_coord(
        annotation,
        with_centerline=True,      # 画 centerline
        with_laneline=False,       # 如果你也想看左右车道线，可改 True
        with_area=False,
        with_car=with_car,
        point_like=False,
        with_nav=True,
        multi_color_centerline=True,
        multi_color_laneline=False,
        multi_color_area=False,
    )[..., ::-1]  # RGB -> BGR，方便用 mmcv / cv2 保存

    # Ped 图
    ped_img = draw_annotation_bev_align_coord(
        annotation,
        with_centerline=False,
        with_laneline=False,
        with_area=True,            # 只画 area 里的行人横道
        with_car=with_car,
        point_like=True,           # 你原来 ped_cross 是 point_like=True 的那种画法
        with_nav=True,
        multi_color_centerline=False,
        multi_color_laneline=False,
        multi_color_area=True,
    )[..., ::-1]

    return lane_img, ped_img

def put_label(img, text):
    """
    在图左上角写一个文本标签，比如 'GT-Lane' / 'Pred-Ped'
    """
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

def visualize_one_result_entry(result_entry,
                               out_dir,
                               segment_id="seg",
                               timestamp="ts"):
    """
    result_entry: eval_results['detailed_results']['predictions'][k]['result']
        里面包含 prediction / ground_truth 两个字段

    out_dir: 输出根目录
    """

    pred_obj = result_entry.get("prediction", {})
    gt_obj = result_entry.get("ground_truth", {})

    # 1⃣️ 取 lanes 并转成 annotation 格式
    pred_lanes = pred_obj.get("lanes", [])
    gt_lanes = gt_obj.get("lanes", [])

    pred_annotation = lanes_to_annotation(pred_lanes)
    gt_annotation = lanes_to_annotation(gt_lanes)

    # 2⃣️ 分别画 GT / Pred 的 lane 和 ped
    gt_lane_img, gt_ped_img = render_lane_and_ped(gt_annotation, with_car=True)
    pred_lane_img, pred_ped_img = render_lane_and_ped(pred_annotation, with_car=True)

    # 3⃣️ 加标签方便肉眼看
    gt_lane_labeled = put_label(gt_lane_img, "GT-Lane")
    pred_lane_labeled = put_label(pred_lane_img, "Pred-Lane")
    gt_ped_labeled = put_label(gt_ped_img, "GT-Ped")
    pred_ped_labeled = put_label(pred_ped_img, "Pred-Ped")

    # 4⃣️ 保证尺寸一致（draw_annotation_bev_align_coord 参数一样时，理论上尺寸一致）
    # 如果你后面改了 figsize/dpi 不同，可以在这里统一 resize
    h, w = gt_lane_labeled.shape[:2]
    def _ensure_size(img):
        if img.shape[0] != h or img.shape[1] != w:
            return cv2.resize(img, (w, h))
        return img

    pred_lane_labeled = _ensure_size(pred_lane_labeled)
    gt_ped_labeled = _ensure_size(gt_ped_labeled)
    pred_ped_labeled = _ensure_size(pred_ped_labeled)

    # 5⃣️ 2×2 拼图：
    # [ GT-Lane   |  Pred-Lane ]
    # [ GT-Ped    |  Pred-Ped  ]
    row_lane = np.concatenate([gt_lane_labeled, pred_lane_labeled], axis=1)
    row_ped = np.concatenate([gt_ped_labeled,  pred_ped_labeled],  axis=1)
    grid = np.concatenate([row_lane, row_ped], axis=0)

    # 6⃣️ 保存
    save_dir = os.path.join(out_dir, str(segment_id))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(timestamp)+"_lane_ped_gt_pred_grid.jpg")
    mmcv.imwrite(grid, save_path)
    print(f"✅ Saved GT/Pred lane+ped comparison to: {save_path}")

def visualize_all_results(eval_json_path, out_dir):
    """
    批量可视化 eval_results.json 里的所有条目。
    每个 sample 输出四张拼成的 2×2 图:
        [ GT-Lane | Pred-Lane ]
        [ GT-Ped  | Pred-Ped  ]

    并且对每个 scene(segment_id) 下面的所有 timestamp 帧，按时间顺序保存成一个 gif。
    """

    print(f"📂 Loading eval results from: {eval_json_path}")
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data["detailed_results"]["predictions"]
    print(f"🔢 Found {len(predictions)} samples to visualize.")

    # 用来记录：每个 scene 下有哪些帧（timestamp, jpg_path）
    scene_frames = {}

    for item in predictions:
        idx = item.get("sample_idx")
        segment_id = item.get("segment_id", f"sample_{idx}")
        timestamp = item.get("timestamp", "0")
        result_entry = item.get("result", {})

        if not result_entry:
            print(f"⚠️ Sample {idx} has empty result, skip.")
            continue

        pred_obj = result_entry.get("prediction", {})
        gt_obj = result_entry.get("ground_truth", {})

        if not pred_obj or not gt_obj:
            print(f"⚠️ Sample {idx}: missing pred/gt, skip.")
            continue

        # --- Convert lanes → annotation format ---
        pred_annotation = lanes_to_annotation(pred_obj.get("lanes", []))
        gt_annotation = lanes_to_annotation(gt_obj.get("lanes", []))

        # --- Render lane + ped ---
        gt_lane_img, gt_ped_img = render_lane_and_ped(gt_annotation)
        pred_lane_img, pred_ped_img = render_lane_and_ped(pred_annotation)

        # --- Add labels ---
        gt_lane_img = put_label(gt_lane_img, f"GT-Lane ({idx})")
        pred_lane_img = put_label(pred_lane_img, f"Pred-Lane ({idx})")
        gt_ped_img = put_label(gt_ped_img, f"GT-Ped ({idx})")
        pred_ped_img = put_label(pred_ped_img, f"Pred-Ped ({idx})")

        # --- Ensure same size ---
        h, w = gt_lane_img.shape[:2]
        def _resize(img):
            return cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

        pred_lane_img = _resize(pred_lane_img)
        gt_ped_img = _resize(gt_ped_img)
        pred_ped_img = _resize(pred_ped_img)

        # --- Stitch into 2×2 grid ---
        row1 = np.concatenate([gt_lane_img, pred_lane_img], axis=1)
        row2 = np.concatenate([gt_ped_img, pred_ped_img], axis=1)
        grid = np.concatenate([row1, row2], axis=0)

        # --- Save single frame jpg ---
        save_dir = os.path.join(out_dir, str(segment_id))
        os.makedirs(save_dir, exist_ok=True)
        frame_name = str(timestamp) + "_lane_ped_gt_pred_grid.jpg"
        save_path = os.path.join(save_dir, frame_name)

        mmcv.imwrite(grid, save_path)
        print(f"✅ Saved sample {idx} → {save_path}")

        # --- 记录到对应 scene 的帧列表里 ---
        scene_frames.setdefault(segment_id, []).append((timestamp, save_path))

    print("📸 All JPG frames saved, now making GIFs per scene...")

    # --- 为每个 scene 生成 gif ---
    for scene_id, frames in scene_frames.items():
        # 按 timestamp 排序（尽量按时间顺序播放）
        def _ts_key(t):
            ts = str(t[0])
            try:
                return int(ts)
            except ValueError:
                return ts  # 如果不是纯数字，就按字符串排序

        frames_sorted = sorted(frames, key=_ts_key)

        images = []
        for ts, path in frames_sorted:
            img = imageio.imread(path)
            images.append(img)

        if not images:
            print(f"⚠️ Scene {scene_id} has no images, skip GIF.")
            continue

        gif_path = os.path.join(out_dir, str(scene_id), f"{scene_id}_lane_ped_gt_pred.gif")

        # 🔸 控制速度的两个参数：
        ORI_FRAME_DURATION = 0.5   # 每个“子帧”的时间（秒）
        REPEAT_PER_FRAME = 2       # 每一张原图重复多少次

        # 1) 展开帧：比如 10 张 → 10 * 4 = 40 帧
        expanded_frames = []
        durations = []
        for img in images:
            for _ in range(REPEAT_PER_FRAME):
                expanded_frames.append(img)
                durations.append(ORI_FRAME_DURATION)

        # 2) 保存 gif
        imageio.mimsave(
            gif_path,
            expanded_frames,
            duration=durations,   # 可以是 list，和帧数一样长
            loop=0,               # 0 = 无限循环
        )
        print(f"🎞 Saved GIF for scene {scene_id}: {gif_path}")

    print("🎉 All samples visualized & GIFs generated!")

def test_all_results(eval_json_path, out_dir, use_select_scenes=False, select_scenes=None):
    """
    批量可视化 eval_results.json 里的所有条目。
    每个 sample 输出四张拼成的 2×2 图:
        [ GT-Lane | Pred-Lane ]
        [ GT-Ped  | Pred-Ped  ]

    并且对每个 scene(segment_id) 下面的所有 timestamp 帧，按时间顺序保存成一个 gif。
    
    Args:
        eval_json_path: 评估结果JSON文件路径
        out_dir: 输出目录
        use_select_scenes: 是否只计算select场景的指标
        select_scenes: 要计算的场景ID列表（当use_select_scenes=True时生效）
    """

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data["detailed_results"]["predictions"]
    print(f"🔢 Found {len(predictions)} samples to evaluate.")

    # 用来记录：每个 scene 下有哪些帧（timestamp, jpg_path）
    scene_frames = {}

    data_infos = load_annotations('/data_test/home/lizhen/yym/TopoWMChange/data/data_dict_subset_A_val_lanesegnet.pkl')
    
    # 根据flag决定是否过滤场景
    scenes_to_use = select_scenes if use_select_scenes else None
    if use_select_scenes and select_scenes:
        print(f"🎯 只计算以下 {len(select_scenes)} 个场景的指标: {select_scenes}")
    else:
        print(f"🌍 计算所有场景的指标")
    
    gt_dict = format_openlanev2_gt(data_infos, select_scenes=scenes_to_use)
    pred_dict = format_results(data, data_infos, select_scenes=scenes_to_use)
    # gt_dict = mmcv.load("./formated_results/gt_results.pkl", file_format="pkl")
    # out_dir = "./formated_results"
    # if out_dir is not None:
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #     mmcv.dump(pred_dict, os.path.join(out_dir, 'pred_results.pkl'))
    #     mmcv.dump(gt_dict, os.path.join(out_dir, 'gt_results.pkl'))
    metric_results = lanesegnet_evaluate(gt_dict, pred_dict)
    
    # 计算select场景的平均指标
    if use_select_scenes and select_scenes and 'per_scene' in metric_results:
        select_scenes_set = set(str(s) for s in select_scenes)
        scene_metrics_list = []
        
        for scene_id, scene_metrics in metric_results['per_scene'].items():
            if scene_id in select_scenes_set:
                scene_metrics_list.append(scene_metrics)
        
        if scene_metrics_list:
            avg_f1_ls = np.mean([m['F1_ls'] for m in scene_metrics_list])
            avg_f1_ped = np.mean([m['F1_ped'] for m in scene_metrics_list])
            avg_top_lsls = np.mean([m['TOP_lsls'] for m in scene_metrics_list])
            avg_mf1 = (avg_f1_ls + avg_f1_ped) / 2
            total_frames = sum([m['num_frames'] for m in scene_metrics_list])
            
            print(f"\n{'='*60}")
            print(f"📊 Select场景平均指标 ({len(scene_metrics_list)} 个场景, {total_frames} 帧):")
            print(f"{'='*60}")
            print(f"   F1_ls (车道线):     {avg_f1_ls:.4f}")
            print(f"   F1_ped (行人横道):  {avg_f1_ped:.4f}")
            print(f"   TOP_lsls (拓扑):    {avg_top_lsls:.4f}")
            print(f"   mF1 (平均):         {avg_mf1:.4f}")
            print(f"{'='*60}")
            
            # 保存平均指标到metric_results中
            metric_results['select_scenes_avg'] = {
                'F1_ls': float(avg_f1_ls),
                'F1_ped': float(avg_f1_ped),
                'TOP_lsls': float(avg_top_lsls),
                'mF1': float(avg_mf1),
                'num_scenes': len(scene_metrics_list),
                'num_frames': total_frames,
                'scene_ids': select_scenes
            }
    
    return metric_results

def format_results( results, data_infos, jsonfile_prefix=None, select_scenes=None):
    pred_dict = {}
    pred_dict['method'] = 'TopoCoT'
    pred_dict['authors'] = []
    pred_dict['e-mail'] = 'dummy'
    pred_dict['institution / company'] = 'CUHKSZ'
    pred_dict['country / region'] = 'CN'
    pred_dict['results'] = {}

    # 统计信息
    stats = {
        'total_samples': 0,
        'error_marked': 0,           # answer_json_str == "error"
        'json_parse_failed': 0,      # JSON 解析失败
        'not_dict': 0,               # 不是字典类型
        'missing_lanes': 0,          # 缺少 'lanes' 键
        'missing_topology': 0,       # 缺少 'topology' 键
        'success': 0,                # 成功处理
        'filtered_by_scene': 0       # 被场景过滤掉的样本
    }
    
    # 如果指定了select_scenes，转换为set以便快速查找
    select_scenes_set = set(select_scenes) if select_scenes else None
    
    for idx, result in enumerate(results['detailed_results']['predictions']):
        stats['total_samples'] += 1
        info = data_infos[idx]
        
        # 场景过滤：如果指定了select_scenes，只处理这些场景
        if select_scenes_set is not None:
            if str(info['segment_id']) not in select_scenes_set:
                stats['filtered_by_scene'] += 1
                continue
        
        key = (split, info['segment_id'], str(info['timestamp']))

        pred_info = dict(
            lane_segment = [],
            area = [],
            traffic_element = [],
            topology_lsls = None,
            topology_lste = None
        )

        lane_results = None
        lanes = []
        lane_ids = []
        
        if result['result']['answer_json_str'] is not None:
            lane_results = result['result']['answer_json_str']
            
            # 检查是否是错误标记
            if lane_results == "error":
                stats['error_marked'] += 1
                # print(f"[WARN] Sample {idx}: answer_json_str is 'error', skip.")
                continue
            
            try:
                lane_results = json.loads(lane_results)
            except json.JSONDecodeError as e:
                # 👉 整个 scene 作废
                stats['json_parse_failed'] += 1
                # print(f"[WARN] JSON parse failed at sample {idx}: {e}")
                # pred_dict['results'][key] = dict(predictions=pred_info)

                continue
            
            # 检查必需的属性：lanes 和 topology
            if not isinstance(lane_results, dict):
                stats['not_dict'] += 1
                # print(f"[WARN] Sample {idx}: lane_results is not a dict, skip.")
                continue
            
            if 'lanes' not in lane_results:
                stats['missing_lanes'] += 1
                # print(f"[WARN] Sample {idx}: lane_results missing 'lanes' key, skip.", lane_results.keys())
                continue
            
            if 'topology' not in lane_results:
                stats['missing_topology'] += 1
                # print(f"[WARN] Sample {idx}: lane_results missing 'topology' key, skip.", lane_results.keys())
                continue

            lanes = []
            labels = []

            lane_ids = []
           
            for lane_single in lane_results['lanes']:
                # 清理和验证点数据
                cleaned_points = clean_and_validate_points(lane_single.get('point', []))
                
                # 如果点数据无效，跳过这条车道线
                if cleaned_points is None or len(cleaned_points) == 0:
                    lane_id = lane_single.get('id', 'unknown')
                    print(f"⚠️ Sample {idx}: lane {lane_id} has invalid points, skip.")
                    continue
                
                lanes.append(cleaned_points)
                
                # 根据 id 判断是 LANE 还是 PED
                lane_id = lane_single.get('id', lane_single.get(' id', ''))
                if isinstance(lane_id, str):
                    if lane_id.startswith('LANE'):
                        labels.append(0)
                    elif lane_id.startswith('PED'):
                        labels.append(1)
                    else:
                        print(f"⚠️ Sample {idx}: unknown lane id format: {lane_id}")
                        labels.append(0)
                else:
                    # 如果 id 不是字符串，默认当作 centerline
                    labels.append(0)

                lane_ids.append(lane_single.get('id', lane_single.get(' id', f'lane_{len(lane_ids)}')))
            # 检查是否有有效的车道线
            if len(lanes) == 0:
                print(f"⚠️ Sample {idx}: no valid lanes after cleaning, skip.")
                continue
            
            # lanes 保持为列表，因为每个车道线的点数可能不同
            labels = np.array(labels, dtype=np.int64 )

            pred_area_index = []
            for pred_idx, (lane, label) in enumerate(zip(lanes, labels)):
                if label == 0:
                    # lane 已经是清理后的 numpy 数组
                    points = np.asarray(lane, dtype=np.float32)
                    
                    # 再次验证点的有效性
                    if len(points) == 0 or points.shape[1] != 2:
                        print(f"⚠️ Sample {idx}: lane {pred_idx} has invalid points after cleaning, skip.")
                        continue

                    pred_lane_segment = {}
                    pred_lane_segment['id'] = 20000 + pred_idx
                    pred_lane_segment['centerline'] = opencv_to_bev(fix_pts_interpolate(points, 10))
                    left_lane, right_lane = pred_lane_segment['centerline'], pred_lane_segment['centerline']
                    pred_lane_segment['left_laneline'] = left_lane
                    pred_lane_segment['right_laneline'] = right_lane

                    pred_lane_segment['confidence'] = 1.0
                    pred_info['lane_segment'].append(pred_lane_segment)
                    
                elif label == 1:
                    # lane 已经是清理后的 numpy 数组
                    points = np.asarray(lane, dtype=np.float32)
                    
                    # 再次验证点的有效性
                    if len(points) == 0 or points.shape[1] != 2:
                        print(f"⚠️ Sample {idx}: ped {pred_idx} has invalid points after cleaning, skip.")
                        continue
                    
                    pred_ped = {}
                    pred_ped['id'] = 20000 + pred_idx
                    pred_points = opencv_to_bev(fix_pts_interpolate(points, 10))

                    pred_ped['points'] = pred_points
                    pred_ped['category'] = label
                    pred_ped['confidence'] = 1.0
                    pred_info['area'].append(pred_ped)
                    pred_area_index.append(pred_idx)
 
                elif label == 2:
                    raise NotImplementedError

            # 处理topology（在if块内）
            if lane_results is not None and lane_results.get('topology') is not None:
                topology = lane_results['topology']

                lane2idx = {lane: i for i, lane in enumerate(lane_ids)}
                idx2lane = {i: lane for lane, i in lane2idx.items()}
                N = len(lanes)
                adj = np.zeros((N, N), dtype=np.float32)

                for path in topology:
                    for u, v in zip(path[:-1], path[1:]):
                        try:
                            adj[lane2idx[u], lane2idx[v]] = 1.0
                        except:
                            pass
                
                pred_info['topology_lsls'] = adj
            else:
                N = len(lanes) if lanes else 0
                adj = np.zeros((N, N), dtype=np.float32)
                pred_info['topology_lsls'] = adj

        pred_dict['results'][key] = dict(predictions=pred_info)
        stats['success'] += 1

    # 输出统计信息
    print(f"\n📊 结果读取统计:")
    print(f"   总样本数: {stats['total_samples']}")
    if select_scenes_set is not None:
        print(f"   场景过滤: {stats['filtered_by_scene']} 个样本被过滤（不在select_scenes列表中）")
    print(f"   成功处理: {stats['success']} 个")
    print(f"   无法读取: {stats['total_samples'] - stats['success'] - stats['filtered_by_scene']} 个")
    print(f"      - 错误标记 (error): {stats['error_marked']} 个")
    print(f"      - JSON 解析失败: {stats['json_parse_failed']} 个")
    print(f"      - 不是字典类型: {stats['not_dict']} 个")
    print(f"      - 缺少 'lanes' 键: {stats['missing_lanes']} 个")
    print(f"      - 缺少 'topology' 键: {stats['missing_topology']} 个")
    valid_samples = stats['total_samples'] - stats['filtered_by_scene']
    print(f"   成功率: {stats['success']/max(1, valid_samples)*100:.2f}%")

    return pred_dict

def format_openlanev2_gt(data_infos, select_scenes=None):
    gt_dict = {}
    # 如果指定了select_scenes，转换为set以便快速查找
    select_scenes_set = set(select_scenes) if select_scenes else None
    
    for idx in range(len(data_infos)):
        info = copy.deepcopy(data_infos[idx])
        
        # 场景过滤：如果指定了select_scenes，只处理这些场景
        if select_scenes_set is not None:
            if str(info['segment_id']) not in select_scenes_set:
                continue
        
        key = (split, info['segment_id'], str(info['timestamp']))
        areas = []
        for lane_segment in info['annotation']['lane_segment']:
            lane_segment['centerline'] = lane_segment['centerline'][:, :2]

        for area in info['annotation']['area']:
            if area['category'] == 1:
                points = area['points']
                # left_boundary = fix_pts_interpolate(points[[0, 1]], 10)
                # right_boundary = fix_pts_interpolate(points[[2, 3]], 10)
                # area['points'] = np.concatenate([left_boundary, right_boundary], axis=0)
                # import pdb; pdb.set_trace()
                assert points.shape[0] == 5
                dir_vector = points[1] - points[0]
                dir = np.rad2deg(np.arctan2(dir_vector[1], dir_vector[0]))

                if dir < -45 or dir > 135:
                    left_boundary = points[[2, 3]]
                    right_boundary = points[[1, 0]]
                else:
                    left_boundary = points[[0, 1]]
                    right_boundary = points[[3, 2]]
                left_boundary = fix_pts_interpolate(left_boundary, 10)
                right_boundary = fix_pts_interpolate(right_boundary, 10)
                centerline = (left_boundary + right_boundary)/2
                centerline = centerline[:,:2]
            
                area['points'] = centerline
                areas.append(area)
        info['annotation']['area'] = areas
        gt_dict[key] = info
     
    return gt_dict

def load_annotations(ann_file):
    """Load annotation from a olv2 pkl file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: Annotation info from the json file.
    """
    with open(ann_file, "rb") as f:
        data_infos = pickle.load(f)
    if isinstance(data_infos, dict):
        if  split == 'train':
            data_infos = [info for info in data_infos.values() if info['meta_data']['source_id'] not in MAP_CHANGE_LOGS]
        else:
            data_infos = list(data_infos.values())
    return data_infos

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

def collect_llm_results_from_test_output(test_output_dir):
    """
    从 test_output 目录收集所有 llm_generated_text.txt 文件，汇总成 eval_result_examples.json 格式。
    
    Args:
        test_output_dir (str): test_output 目录路径
        
    Returns:
        dict: 格式与 eval_result_examples.json 相同的结果字典
    """
    result = {
        "detailed_results": {
            "predictions": []
        }
    }
    
    predictions = []
    sample_idx = 1
    
    # 遍历 test_output 目录
    test_output_path = Path(test_output_dir)
    if not test_output_path.exists():
        print(f"⚠️ 目录不存在: {test_output_dir}")
        return result
    
    # 查找所有 llm_generated_text.txt 文件
    # 路径格式: test_output/{segment_id}/{timestamp}/llm_generated_text.txt
    llm_files = list(test_output_path.glob("*/*/llm_generated_text.txt"))
    
    if not llm_files:
        print(f"⚠️ 未找到任何 llm_generated_text.txt 文件在: {test_output_dir}")
        return result
    
    print(f"📂 找到 {len(llm_files)} 个 llm_generated_text.txt 文件")
    
    # 统计信息
    stats = {
        "total_files": 0,
        "original_parse_success": 0,  # 原始 JSON 解析成功
        "fixed_parse_success": 0,     # 修复后解析成功
        "parse_failed": 0              # 最终解析失败
    }
    
    for llm_file in sorted(llm_files):
        # 从路径中提取 segment_id 和 timestamp
        # 路径格式: test_output/{segment_id}/{timestamp}/llm_generated_text.txt
        parts = llm_file.parts
        if len(parts) < 3:
            print(f"⚠️ 跳过无效路径: {llm_file}")
            continue
        
        segment_id = parts[-3]  # segment_id 是倒数第三部分
        timestamp = parts[-2]   # timestamp 是倒数第二部分
        
        # 读取文件内容
        try:
            with open(llm_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"⚠️ 读取文件失败 {llm_file}: {e}")
            continue
        
        if not content:
            print(f"⚠️ 文件为空: {llm_file}")
            continue
        
        stats["total_files"] += 1
        
        # 首先尝试直接解析原始内容
        original_parse_success = False
        try:
            json.loads(content)
            original_parse_success = True
            stats["original_parse_success"] += 1
            answer_json_str = content
        except (json.JSONDecodeError, RecursionError):
            # 原始解析失败，尝试修复
            try:
                fixed_content = fix_json_string(content)
                
                # 检查修复后的内容是否与原始内容不同
                is_fixed = (fixed_content != content)
                
                # 验证 JSON 是否有效
                try:
                    json.loads(fixed_content)
                    stats["fixed_parse_success"] += 1
                    answer_json_str = fixed_content
                    # if is_fixed:
                    #     print(f"✅ JSON 修复成功 (segment_id={segment_id}, timestamp={timestamp})")
                except (json.JSONDecodeError, RecursionError) as e:
                    stats["parse_failed"] += 1
                    # print(f"⚠️ JSON 解析失败 (segment_id={segment_id}, timestamp={timestamp}): {type(e).__name__}: {str(e)[:100]}")
                    # print(f"   原始内容前100字符: {content[:100]}")
                    # 修复后仍然解析失败，保存 "error" 标记
                    answer_json_str = "error"
            except (RecursionError, Exception) as e:
                stats["parse_failed"] += 1
                # print(f"⚠️ JSON 修复过程出错 (segment_id={segment_id}, timestamp={timestamp}): {type(e).__name__}: {str(e)[:100]}")
                # print(f"   原始内容前100字符: {content[:100]}")
                # 修复过程出错，保存 "error" 标记
                answer_json_str = "error"
        
        # 构建预测条目
        prediction_entry = {
            "sample_idx": sample_idx,
            "segment_id": segment_id,
            "timestamp": timestamp,
            "result": {
                "answer_json_str": answer_json_str,
                "cot": None  # llm_generated_text.txt 没有 cot
            }
        }
        
        predictions.append(prediction_entry)
        sample_idx += 1

    result["detailed_results"]["predictions"] = predictions
    
    # 输出统计信息
    print(f"\n📊 JSON 解析统计:")
    print(f"   总文件数: {stats['total_files']}")
    print(f"   原始解析成功: {stats['original_parse_success']} 帧")
    print(f"   修复后解析成功: {stats['fixed_parse_success']} 帧")
    print(f"   解析失败: {stats['parse_failed']} 帧")
    print(f"✅ 成功收集 {len(predictions)} 个预测结果")
    
    return result

CAMS = ('ring_front_center', 'ring_front_left', 'ring_front_right',
        'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right')
LANE_CLASSES = ('lane_segment', 'ped_crossing')
TE_CLASSES = ('traffic_light', 'road_sign')
TE_ATTR_CLASSES = ('unknown', 'red', 'green', 'yellow',
                    'go_straight', 'turn_left', 'turn_right',
                    'no_left_turn', 'no_right_turn', 'u_turn', 'no_u_turn',
                    'slight_left', 'slight_right')
MAP_CHANGE_LOGS = [
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
]

split = 'val'
points_num = 10

# 从 test_output 收集 llm_generated_text.txt 并汇总
test_output_dir = "./work_dirs/test_output"

#####没生成的话，要运行一次这个
aggregated_results = collect_llm_results_from_test_output(test_output_dir)

# # 保存汇总结果为 JSON 文件
output_json_path = "./tools/evaluate/llm_results_aggregated.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
print(f"✅ 汇总结果已保存到: {output_json_path}")

# 更新 json_path 指向新生成的文件
json_path = output_json_path

test_all_results(
    eval_json_path=json_path,
    out_dir="vis_results",
    use_select_scenes=USE_SELECT_SCENES,
    select_scenes=scene_id if USE_SELECT_SCENES else None
)

print("✅ 所有处理完成！")

# visualize_all_results(
#     eval_json_path="./saves/qwen3vl-2b/lora/topocot_sft/eval_results_20251216_114806qwenvl2b.json",
#     out_dir="vis_results"
# )
