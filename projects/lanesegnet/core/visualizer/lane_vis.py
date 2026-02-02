#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
COLOR_DICT = {  # RGB [0, 1]
    'centerline': np.array([243, 90, 2]) / 255,
    'laneline': np.array([0, 32, 127]) / 255,
    'ped_crossing': np.array([255, 192, 0]) / 255,
    'road_boundary': np.array([220, 30, 0]) / 255,
    'pop_lane': np.array([0, 0, 255]) / 255,
}
LINE_PARAM = {
    0: {'color': COLOR_DICT['laneline'], 'alpha': 0.3, 'linestyle': ':'},       # none
    1: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': 'solid'},  # solid
    2: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': '--'},     # dashed
    'ped_crossing': {'color': COLOR_DICT['ped_crossing'], 'alpha': 1, 'linestyle': 'solid'},
    'road_boundary': {'color': COLOR_DICT['road_boundary'], 'alpha': 1, 'linestyle': 'solid'},
    'pop_lane': {'color': COLOR_DICT['pop_lane'], 'alpha': 1, 'linestyle': 'solid'}
}
BEV_RANGE = [-50, 50, -25, 25]

def _draw_centerline(ax, idx, lane_centerline, pop_mask= None):
    points = np.asarray(lane_centerline['points'])
    color = COLOR_DICT['centerline']
    color_pop = COLOR_DICT['pop_lane']
    # draw line
    if pop_mask:
        ax.plot(points[:, 1], points[:, 0], color=color_pop, alpha=1.0, linewidth=0.6)
    else:
        ax.plot(points[:, 1], points[:, 0], color=color, alpha=1.0, linewidth=0.6)
    # draw start and end vertex
 
    ax.scatter(points[[0, -1], 1], points[[0, -1], 0], color=color, s=1)
    ax.text(points[[0, -1], 1][0],points[[0, -1], 0][0],s=str(idx),  fontsize=12)
    # draw arrow
    ax.annotate('', xy=(points[-1, 1], points[-1, 0]),
                xytext=(points[-2, 1], points[-2, 0]),
                arrowprops=dict(arrowstyle='->', lw=0.6, color=color))

def _draw_line(ax, line):
    points = np.asarray(line['points'])
    config = LINE_PARAM[line['linetype']]
    ax.plot(points[:, 1], points[:, 0], linewidth=0.6, **config)

def _draw_lane_segment(ax, lane_segment, idx, with_centerline, with_laneline = False, pop_mask = None):
    if with_centerline:
        _draw_centerline(ax, idx, {'points': lane_segment}, pop_mask=pop_mask)
    if with_laneline:
        _draw_line(ax, {'points': lane_segment['left_laneline'], 'linetype': lane_segment['left_laneline_type']})
        _draw_line(ax, {'points': lane_segment['right_laneline'], 'linetype': lane_segment['right_laneline_type']})

def _draw_area(ax, area):
    if area['category'] == 1:  # ped crossing with lane segment style.
        _draw_line(ax, {'points': area['points'], 'linetype': 'ped_crossing'})
    elif area['category'] == 2:  # road boundary
        _draw_line(ax, {'points': area['points'], 'linetype': 'road_boundary'})

def draw_annotation_bev(annotation, with_centerline=True, with_laneline=True, with_area=True, pop_mask = None):

    fig, ax = plt.figure(figsize=(2, 4), dpi=200), plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim([BEV_RANGE[0], BEV_RANGE[1]])
    ax.set_xlim([BEV_RANGE[2], BEV_RANGE[3]])
    ax.invert_xaxis()
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.2)

    for idx, lane_segment in enumerate(annotation):
        if pop_mask is not None:
            _draw_lane_segment(ax, lane_segment, idx, with_centerline, with_laneline, pop_mask = pop_mask[idx])
        else:
            _draw_lane_segment(ax, lane_segment, idx, with_centerline, with_laneline)
    if with_area:
        for area in annotation['area']:
            _draw_area(ax, area)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def vis_from_nodelist_mul3( nodelist, img_meta, path, aux_name):
    Map_size = [(-50, 50), (-50, 50)]
    Map_resolution = 0.5
    image = np.zeros([600, 600, 3])
    point_color_map = {"start": (0, 0, 255), 'fork': (0, 255, 0), "continue": (0, 255, 255), "merge": (255, 0, 0)}

    for idx, node in enumerate(nodelist):
        if node['sque_type'] == 'start':
            cv2.circle(image, np.array(node['coord'])*3, 1, color=point_color_map['start'], thickness=2)
            text_position = (np.array(nodelist[idx]['coord'][0])*3 , np.array(nodelist[idx]['coord'][1])*3 )
            cv2.putText(image, str(idx), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif node['sque_type'] == 'continue':
            cv2.circle(image, np.array(node['coord'])*3, 1, color=point_color_map['continue'], thickness=2)
            
            # cv2.polylines(image, [subgraphs_points_in_between_nodes[(node.sque_index-1, node.sque_index)]], False, color=point_color_map['continue'], thickness=1)
            cv2.arrowedLine(image, np.array(nodelist[idx - 1]['coord'])*3, np.array(node['coord'])*3, color=point_color_map['continue'],
                            thickness=1, tipLength=0.1)
            text_position = (np.array(nodelist[idx]['coord'][0])*3 , np.array(nodelist[idx]['coord'][1])*3 )
            cv2.putText(image, str(idx), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif node['sque_type'] == 'fork':
            if node['fork_from'] > idx or node['fork_from'] < 0:
                continue
            fork_node = next((fork_node for fork_node in nodelist if fork_node['sque_index'] == node["fork_from"]), None)
            cv2.circle(image, np.array(node['coord'])*3, 1, color=point_color_map['fork'], thickness=2)
            cv2.arrowedLine(image, np.array(fork_node['coord'])*3, np.array(node['coord'])*3,
                            color=point_color_map['fork'],
                            thickness=1, tipLength=0.1)
            text_position = (np.array(nodelist[idx]['coord'][0])*3 , np.array(nodelist[idx]['coord'][1])*3 )
            cv2.putText(image, str(idx), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif node['sque_type'] == 'merge':
            if node['merge_with'] > idx or node['merge_with'] < 0:
                continue
            merge_node = next((merge_node for merge_node in nodelist if merge_node['sque_index'] == node["merge_with"]), None)
            if merge_node == None:
                continue
            cv2.circle(image, np.array(node['coord'])*3, 1, color=point_color_map['merge'], thickness=2)

            cv2.arrowedLine(image, np.array(node['coord'])*3, np.array(merge_node['coord'])*3, 
                            color=point_color_map['merge'], thickness=1, tipLength=0.1)
            
            text_position = (np.array(nodelist[idx]['coord'][0])*3 , np.array(nodelist[idx]['coord'][1])*3 )
            cv2.putText(image, str(idx), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # print('img_meta',img_meta)
    try:
        name = img_meta['img_filename'][0].split('/')[-1].split('.jpg')[0]
    except:
        name = img_meta['filename'][0].split('/')[-1].split('.jpg')[0]
    save_dir = f"./vis/{path}/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cv2.imwrite(os.path.join(save_dir, f"{name}_{aux_name}.png"), image)
    print('filename',f"{name}_{aux_name}.png")
    print('all file',img_meta['filename'])
    # print('os.path.join(save_dir, f"{name}_{aux_name}.png")')

def _render_surround_img(images):
    all_image = []
    img_height = images[1].shape[0]

    for idx in [1, 0, 2, 5, 3, 4, 6]:
        if idx  == 0:
            all_image.append(images[idx][356:1906, :])
            all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))
        elif idx == 6 or idx == 2:
            all_image.append(images[idx])
        else:
            all_image.append(images[idx])
            all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))

    surround_img_upper = None
    surround_img_upper = np.concatenate(all_image[:5], 1)

    surround_img_down = None
    surround_img_down = np.concatenate(all_image[5:], 1)
    scale = surround_img_upper.shape[1] / surround_img_down.shape[1]
    surround_img_down = cv2.resize(surround_img_down, None, fx=scale, fy=scale)

    divider = np.full((25, surround_img_down.shape[1], 3), (255, 255, 255), dtype=np.uint8)

    surround_img = np.concatenate((surround_img_upper, divider, surround_img_down), 0)
    surround_img = cv2.resize(surround_img, None, fx=0.5, fy=0.5)

    return surround_img
    