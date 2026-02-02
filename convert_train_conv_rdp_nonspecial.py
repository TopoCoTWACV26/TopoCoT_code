import os
import json
import pickle
from tqdm import tqdm
import re
import pdb
from collections import Counter
from textwrap import dedent
import numpy as np
from rdp import rdp as rdp_simplify
def clean_action(text: str):
    """清洗动作：去 markdown、去空格、去 the、统一小写"""
    if text is None:
        return ""

    t = text.strip()
    t = t.strip("*_`")                   # 去 markdown
    t = t.lower()

    # 🔥 自动修正常见生成错误，例如带 "the"
    t = t.replace(" to the right", " to right")
    t = t.replace(" to the left", " to left")

    # 再处理可能出现的多余空格
    t = re.sub(r"\s+", " ", t).strip()

    return t

import re

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
        bev =  np.stack([x, y, z], axis=1)
    else:
        bev =  np.stack([x, y], axis=1)
    return np.round(bev, 1)
def _match_valid_prefix(action_cleaned: str, valid_options_lower):
    """
    在一个可能带解释的长字符串中，找“前缀是合法动作”的情况：
    例如：
        "move forward — the vehicle ..."  -> "move forward"
        "turn right: the ego ..."         -> "turn right"
    """
    if not action_cleaned:
        return None

    # 先把常见分隔符前的部分单独拿出来，增加成功率
    # 比如 'move forward — xxx' -> 'move forward'
    #      'turn left: xxx'     -> 'turn left'
    for sep in ["—", "-", ":", "；", ";", "，", ","]:
        if sep in action_cleaned:
            action_cleaned = action_cleaned.split(sep, 1)[0].strip()
            break

    # 再按长度从长到短，做前缀匹配
    for opt in sorted(valid_options_lower, key=len, reverse=True):
        if action_cleaned.startswith(opt):
            return opt

    # 如果前缀没匹配上，再直接试一下“完全等于”
    if action_cleaned in valid_options_lower:
        return action_cleaned

    return None

def extract_actions_from_cot(cot: str, segment_id_index: str, timestamp_index: str):
    """
    从 CoT 字符串中提取 Lateral / Longitudinal action。
    如果解析失败或不在允许集合中，则触发断点。
    """

    # 先把所有 * 去掉，避免 markdown 影响匹配
    cot_clean = cot.replace("*", "")

    lat_match = re.search(
        r"Lateral\s+Actions?\s*:\s*([^\n\r]+)",
        cot_clean,
        flags=re.IGNORECASE,
    )

    lon_match = re.search(
        r"Longitudinal\s+Actions?\s*:\s*([^\n\r]+)",
        cot_clean,
        flags=re.IGNORECASE,
    )

    if not lat_match or not lon_match:
        print("❌ 未找到 Lateral/Longitudinal action 行")
        print("===== COT DUMP START =====")
        print(cot_clean)
        print("segment_id_index:", segment_id_index)
        print("timestamp_index:", timestamp_index)
        print("===== COT DUMP END =====")
        pdb.set_trace()  # 卡在这里检查
        raise ValueError("Cannot find action lines in CoT.")

    # 原始字符串（可能带解释）
    raw_lateral = lat_match.group(1).strip()
    raw_longitudinal = lon_match.group(1).strip()

    # 先做通用清洗（小写、去 the 等）
    lateral_cleaned = clean_action(raw_lateral)
    longitudinal_cleaned = clean_action(raw_longitudinal)

    # 再从中抽取“合法动作前缀”
    lateral_action = _match_valid_prefix(lateral_cleaned, LATERAL_OPTIONS_LOWER)
    longitudinal_action = _match_valid_prefix(longitudinal_cleaned, LONGITUDINAL_OPTIONS_LOWER)

    # 校验是否在允许集合中
    if lateral_action is None:
        print("❌ 未识别到合法的 lateral action:")
        print(f"  raw     = '{raw_lateral}'")
        print(f"  cleaned = '{lateral_cleaned}'")
        print("允许值为:", LATERAL_OPTIONS)
        print("segment_id_index:", segment_id_index)
        print("timestamp_index:", timestamp_index)
        pdb.set_trace()
        raise ValueError(f"Invalid lateral action (no valid prefix): {raw_lateral}")

    if longitudinal_action is None:
        print("❌ 未识别到合法的 longitudinal action:")
        print(f"  raw     = '{raw_longitudinal}'")
        print(f"  cleaned = '{longitudinal_cleaned}'")
        print("允许值为:", LONGITUDINAL_OPTIONS)
        print("segment_id_index:", segment_id_index)
        print("timestamp_index:", timestamp_index)
        pdb.set_trace()
        raise ValueError(f"Invalid longitudinal action (no valid prefix): {raw_longitudinal}")

    # 此时返回的是“归一化后的小写合法动作”，例如 "move forward", "turn left"
    return lateral_action, longitudinal_action

def load_annotations(ann_file):
    """Load annotation from a olv2 pkl file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: Annotation info from the json file.
    """
    with open(ann_file, 'rb') as f:
        data_infos = pickle.load(f)
    if isinstance(data_infos, dict):
        if True and split == 'train':
            data_infos = [
                info for info in data_infos.values()
                if info['meta_data']['source_id'] not in MAP_CHANGE_LOGS
            ]
        else:
            data_infos = list(data_infos.values())
    return data_infos



if __name__ == '__main__':

    CAMS = (
        'ring_front_center', 'ring_front_left', 'ring_front_right',
        'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right'
    )
    LANE_CLASSES = ('lane_segment', 'ped_crossing')
    MAP_CHANGE_LOGS = [
        '75e8adad-50a6-3245-8726-5e612db3d165',
        '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
        'af170aac-8465-3d7b-82c5-64147e94af7d',
        '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    ]
    data_root = '/data_test/home/lizhen/yym/TopoStreamer_vlm/'
    ann_file = data_root + 'data_dict_subset_A_train_lanesegnet_reasoningv3.pkl'
    split = 'train'
    # 允许的动作集合
    LATERAL_OPTIONS = {
        "move forward",
        "turn left",
        "change lane to left",
        "turn right",
        "change lane to right",
    }

    LONGITUDINAL_OPTIONS = {
        "stop",
        "deceleration to zero",
        "maintain constant speed",
        "deceleration",
        "acceleration",
    }

    LATERAL_OPTIONS_LOWER = {x.lower() for x in LATERAL_OPTIONS}
    LONGITUDINAL_OPTIONS_LOWER = {x.lower() for x in LONGITUDINAL_OPTIONS}

    data_infos = load_annotations(ann_file)

    # lateral_counter = Counter()
    # longitudinal_counter = Counter()
    # pair_counter = Counter()   # (lateral, longitudinal) 组合的统计，可选
    sharegpt_samples = []

    special_tokens = False
    for idx in tqdm(range(len(data_infos))):
        segment_id_index = str(data_infos[idx]['segment_id'])
        timestamp_index = str(data_infos[idx]['timestamp'])

        json_path = f"./data/Trainset/{segment_id_index}/{timestamp_index}/"
        with open(json_path + "lane_with_drive.json", "r", encoding="utf-8") as f:
            lane_data = json.load(f)
        for lane in lane_data['lanes']:
            if 'coords_2d' in lane:
                lane['point'] = lane.pop('coords_2d')
                lane['id'] = lane.pop('center_id')
                # lane['id'] = lane['id'].replace('LANE', 'L').replace('PED', 'P')
                
                lane_coord = lane['point']

            
                lane_coord =  lane_coord
       
                lane_coord = rdp_simplify(lane_coord, epsilon=3.0)
                lane_coord = np.array(lane_coord)
              
                lane_coord = [
                [int(x),  int(y)]
                for x, y in lane_coord[:, :2]

                ]       
                lane['point'] = lane_coord
                
                for k in ['category', 'left_boundary', 'right_boundary', 'offset']:
                    lane.pop(k, None)

        navigation_information = lane_data['navigation']
        del lane_data['future_waypoints']
        
        del lane_data['navigation']

        ### to do cot
        # with open(json_path + "TopoCot_without_thinking.json", "r", encoding="utf-8") as f:
        #     cot_data = json.load(f)
        #     cot = cot_data['answers']['all_pure_answer']

        # lateral_action, longitudinal_action = extract_actions_from_cot(cot, segment_id_index, timestamp_index)

        # lane_data['ego']['lateral_action'] = lateral_action
        # lane_data['ego']['longitudinal_action'] = longitudinal_action


 



        system_prompt = dedent("""
        - You are a traffic engineer and autonomous driving perception analyst.
        - The map spans a longitudinal range of 0 to 1000 decimeters and a lateral range of 0 to 500 decimeters.

        The coordinate system is defined as follows:
        - The origin [0, 0] is located at the top left corner of the map.
        - The ego vehicle is always positioned at the center point [250, 500].
        - The x-axis decreases toward the left side of the ego vehicle and increases toward the right side of the ego vehicle.
        - The y-axis decreases toward the front of the ego vehicle and increases toward the rear of the ego vehicle.

        • x < 250 ⇒ LEFT the ego vehicle; x > 250 ⇒ RIGHT the ego vehicle.
        • y < 500 ⇒ IN FRONT OF the ego vehicle; y > 500 ⇒ BEHIND of the ego vehicle.
        """).strip()

        Instruction_prompt = dedent(f"""
        Carefully predict the map information and lane topology in JSON format.""").strip()

        assistant_answer = (
            f"{json.dumps(lane_data, ensure_ascii=False)}"
        )

        sample = {
                    
                    "system": system_prompt,
                    "prompt": Instruction_prompt,
                    "answer": assistant_answer,
                },

        dir_path = f'./data/train_conv_rdp/{segment_id_index}/{timestamp_index}'
        os.makedirs(dir_path, exist_ok=True)

        out_path = os.path.join(dir_path, "bev_conv.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        import pdb; pdb.set_trace()
