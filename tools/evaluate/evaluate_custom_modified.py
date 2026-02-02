# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# evaluate.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-V2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from tqdm import tqdm

from typing import Optional
from openlanev2.lanesegment.evaluation.distance import pairwise, area_distance, lane_segment_distance, lane_segment_distance_c, traffic_element_distance
from openlanev2.centerline.evaluation.distance import chamfer_distance, frechet_distance
from openlanev2.lanesegment.io import io

THRESHOLDS_AREA = [0.5, 1.0, 1.5]
THRESHOLDS_LANESEG = [1.0, 2.0, 3.0]
THRESHOLDS_TE = [0.75]
THRESHOLD_RELATIONSHIP_CONFIDENCE = 0.5

def _micro_edge_f1(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    tp = int(np.logical_and(gt_bool, pred_bool).sum())
    fp = int(np.logical_and(~gt_bool, pred_bool).sum())
    fn = int(np.logical_and(gt_bool, ~pred_bool).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2*p*r/(p+r)) if (p+r) > 0 else 0.0

def _topology_f1_lsls_with_unmatched_penalty(gts_topo: np.ndarray,
                                            preds_topo_unmatched: np.ndarray,
                                            idx_match_gt: np.ndarray) -> float:
    """
    gts_topo: (Ngt, Ngt) GT adjacency (0/1 or any numeric; nonzero treated as edge)
    preds_topo_unmatched: (Npred, Npred) pred adjacency (nonzero treated as edge)
    idx_match_gt: (Npred,) pred->gt mapping, NaN means unmatched
    """
    gt_bool = (gts_topo != 0)
    pred_bool_full = (preds_topo_unmatched != 0)

    Ngt = gts_topo.shape[0]
    Npred = preds_topo_unmatched.shape[0]

    # matched pred indices and their gt targets
    matched_pred = np.where(~np.isnan(idx_match_gt))[0]
    if matched_pred.size == 0:
        # no matched nodes: all GT edges are FN; all pred edges are FP
        fp_extra = int(pred_bool_full.sum())
        fn_all = int(gt_bool.sum())
        # f1 from tp=0
        return _micro_edge_f1(gt_bool, np.zeros_like(gt_bool, dtype=bool)) if fp_extra == 0 else 0.0

    gt_indices = idx_match_gt[matched_pred].astype(int)
    pred_indices = matched_pred.astype(int)

    # 1) project pred topology onto full GT node set (unmatched GT nodes => remain 0 => FN)
    pred_on_gt = np.zeros((Ngt, Ngt), dtype=bool)
    pred_on_gt[np.ix_(gt_indices, gt_indices)] = pred_bool_full[np.ix_(pred_indices, pred_indices)]

    # 2) count FP contributed by unmatched pred nodes (nodes that have no gt counterpart)
    unmatched_pred = np.where(np.isnan(idx_match_gt))[0].astype(int)
    fp_extra = 0
    if unmatched_pred.size > 0:
        # any edge incident to an unmatched pred node has no place in GT => FP
        fp_extra += int(pred_bool_full[np.ix_(unmatched_pred, np.arange(Npred))].sum())
        fp_extra += int(pred_bool_full[np.ix_(np.arange(Npred), unmatched_pred)].sum())
        # remove double-counted edges within unmatched-unmatched block
        fp_extra -= int(pred_bool_full[np.ix_(unmatched_pred, unmatched_pred)].sum())

        # plus edges inside unmatched-unmatched block (still FP)
        fp_extra += int(pred_bool_full[np.ix_(unmatched_pred, unmatched_pred)].sum())

    # micro F1 on GT-sized matrix first
    tp = int(np.logical_and(gt_bool, pred_on_gt).sum())
    fp = int(np.logical_and(~gt_bool, pred_on_gt).sum()) + fp_extra
    fn = int(np.logical_and(gt_bool, ~pred_on_gt).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2*p*r/(p+r)) if (p+r) > 0 else 0.0

def _greedy_match(distance_matrix: np.ndarray, distance_threshold: float):
    """One-to-one greedy matching by ascending distance (confidence-free)."""
    if distance_matrix.size == 0:
        return []
    G, P = distance_matrix.shape
    pairs = []
    for gi in range(G):
        for pi in range(P):
            d = distance_matrix[gi, pi]
            if np.isfinite(d) and d < distance_threshold:
                pairs.append((float(d), gi, pi))
    pairs.sort(key=lambda x: x[0])

    matched_g = set()
    matched_p = set()
    matches = []
    for _, gi, pi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        matches.append((gi, pi))
    return matches

def _precision_recall_f1(tp: int, fp: int, fn: int):
    """Return (precision, recall, f1). If both GT and Pred are empty => (1,1,1)."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)

def _match_and_inject(gts, preds, distance_matrix: np.ndarray, distance_threshold: float, *,
                      preds_token_dict: Optional[dict] = None,
                      object_type: Optional[str] = None):
    """Compute greedy matching and (optionally) inject idx_match_gt into preds[token]."""
    num_gts = len(gts)
    num_preds = len(preds)

    if num_gts == 0 and num_preds == 0:
        idx_match_gt = np.asarray([], dtype=float)
        if preds_token_dict is not None and object_type is not None:
            preds_token_dict[f'{object_type}_{distance_threshold}_idx_match_gt'] = idx_match_gt
        return 0, 0, 0, idx_match_gt

    if num_preds == 0:
        idx_match_gt = np.asarray([], dtype=float)
        if preds_token_dict is not None and object_type is not None:
            preds_token_dict[f'{object_type}_{distance_threshold}_idx_match_gt'] = idx_match_gt
        return 0, 0, num_gts, idx_match_gt

    # Default: all unmatched
    idx_match_gt = np.ones((num_preds,), dtype=float) * np.nan

    if num_gts == 0:
        if preds_token_dict is not None and object_type is not None:
            preds_token_dict[f'{object_type}_{distance_threshold}_idx_match_gt'] = idx_match_gt
        return 0, num_preds, 0, idx_match_gt

    matches = _greedy_match(distance_matrix, distance_threshold)
    for gi, pi in matches:
        idx_match_gt[pi] = gi

    tp = len(matches)
    fp = num_preds - tp
    fn = num_gts - tp

    if preds_token_dict is not None and object_type is not None:
        preds_token_dict[f'{object_type}_{distance_threshold}_idx_match_gt'] = idx_match_gt

    return tp, fp, fn, idx_match_gt

def _F1_over_threshold(gts, preds, distance_matrices, distance_threshold: float, object_type: str,
                       filter, inject: bool, gt_num : int):
    """Confidence-free evaluation: greedy match => Precision/Recall/F1, aggregated over all samples."""
    total_tp = total_fp = total_fn = 0

    for token in gts.keys():
        gt_list = [gt for gt in gts[token][object_type] if filter(gt)]
        pred_list = [pred for pred in preds[token][object_type] if filter(pred)]

        dm = distance_matrices[token].copy()
        gt_mask = np.array([filter(gt) for gt in gts[token][object_type]], dtype=bool)
        pred_mask = np.array([filter(pr) for pr in preds[token][object_type]], dtype=bool)

        dm = dm[gt_mask, :][:, pred_mask] if (dm.size > 0) else dm

        tp, fp, fn, _ = _match_and_inject(
            gts=gt_list,
            preds=pred_list,
            distance_matrix=dm,
            distance_threshold=distance_threshold,
            preds_token_dict=(preds[token] if inject else None),
            object_type=object_type if inject else None,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    _, _, f1 = _precision_recall_f1(total_tp, total_fp, total_fn)

    f1 = f1*len(gts) / gt_num

    return np.float32(f1)

def _mF1_over_threshold(gts, preds, distance_matrices, distance_thresholds, object_type, gt_num, filter, inject):
    """Return mean F1 over multiple distance thresholds (kept name compatibility if needed)."""
    f1s = np.asarray([
        _F1_over_threshold(
            gts=gts,
            preds=preds,
            distance_matrices=distance_matrices,
            distance_threshold=thr,
            object_type=object_type,
            filter=filter,
            inject=inject,
            gt_num=gt_num,
        )
        for thr in distance_thresholds
    ], dtype=np.float32)
    return f1s

def _edge_f1(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    """Binary edge F1 for two same-shaped boolean matrices."""
    assert gt_bool.shape == pred_bool.shape
    tp = int(np.logical_and(gt_bool, pred_bool).sum())
    fp = int(np.logical_and(~gt_bool, pred_bool).sum())
    fn = int(np.logical_and(gt_bool, ~pred_bool).sum())
    _, _, f1 = _precision_recall_f1(tp, fp, fn)
    return float(f1)

def _topology_f1_directed(gts: np.ndarray, preds: np.ndarray) -> float:
    """Confidence-free directed-graph F1, averaged over rows and cols."""
    assert gts.shape == preds.shape and gts.ndim == 2 and gts.shape[0] == gts.shape[1]
    # Treat as binary edges: any non-zero entry is an edge.
    gt_bool = (gts != 0)
    pred_bool = (preds != 0)

    f1s = []
    # outgoing
    for i in range(gt_bool.shape[0]):
        f1s.append(_edge_f1(gt_bool[i:i+1, :], pred_bool[i:i+1, :]))
    # incoming
    for j in range(gt_bool.shape[1]):
        f1s.append(_edge_f1(gt_bool[:, j:j+1], pred_bool[:, j:j+1]))
    return float(np.mean(f1s)) if f1s else 0.0

def _topology_f1_undirected(gts: np.ndarray, preds: np.ndarray) -> float:
    """Confidence-free bipartite/undirected-like adjacency F1, averaged over both axes."""
    assert gts.shape == preds.shape and gts.ndim == 2
    gt_bool = (gts != 0)
    pred_bool = (preds != 0)

    f1s = []
    for i in range(gt_bool.shape[0]):
        f1s.append(_edge_f1(gt_bool[i:i+1, :], pred_bool[i:i+1, :]))
    for j in range(gt_bool.shape[1]):
        f1s.append(_edge_f1(gt_bool[:, j:j+1], pred_bool[:, j:j+1]))
    return float(np.mean(f1s)) if f1s else 0.0

def _mF1_topology_lsls(gts, preds, distance_thresholds):
    """Topology F1 among lane segments; does NOT use any confidence threshold."""
    acc = []
    for distance_threshold in distance_thresholds:
        for token in gts.keys():
            preds_topology_lsls_unmatched = preds[token]['topology_lsls']

            idx_match_gt = preds[token].get(f'lane_segment_{distance_threshold}_idx_match_gt', None)
            if idx_match_gt is None:
                continue
            gt_pred = {m: i for i, m in enumerate(idx_match_gt) if not np.isnan(m)}

            gts_topology_lsls = gts[token]['topology_lsls']
            if 0 in gts_topology_lsls.shape or len(gt_pred) == 0:
                continue

            gt_indices = np.array(list(gt_pred.keys())).astype(int)
            pred_indices = np.array(list(gt_pred.values())).astype(int)

            preds_topology_lsls = np.zeros_like(gts_topology_lsls, dtype=gts_topology_lsls.dtype)
            xs = gt_indices[:, None].repeat(len(gt_indices), 1)
            ys = gt_indices[None, :].repeat(len(gt_indices), 0)
            preds_topology_lsls[xs, ys] = preds_topology_lsls_unmatched[pred_indices][:, pred_indices]

            acc.append(_topology_f1_directed(gts_topology_lsls, preds_topology_lsls))

    return np.float32(np.mean(acc)) if len(acc) > 0 else np.float32(0)

def _mF1_topology_lste(gts, preds, distance_thresholds):
    """Topology F1 between lane segments and traffic elements; confidence-free."""
    acc = []
    for distance_threshold_lane_segment in distance_thresholds['lane_segment']:
        for distance_threshold_traffic_element in distance_thresholds['traffic_element']:
            for token in gts.keys():
                preds_topology_lste_unmatched = preds[token]['topology_lste']

                idx_match_gt_lane_segment = preds[token].get(
                    f'lane_segment_{distance_threshold_lane_segment}_idx_match_gt', None
                )
                idx_match_gt_traffic_element = preds[token].get(
                    f'traffic_element_{distance_threshold_traffic_element}_idx_match_gt', None
                )
                if idx_match_gt_lane_segment is None or idx_match_gt_traffic_element is None:
                    continue

                gt_pred_lane_segment = {m: i for i, m in enumerate(idx_match_gt_lane_segment) if not np.isnan(m)}
                gt_pred_traffic_element = {m: i for i, m in enumerate(idx_match_gt_traffic_element) if not np.isnan(m)}

                gts_topology_lste = gts[token]['topology_lste']
                if 0 in gts_topology_lste.shape or len(gt_pred_lane_segment) == 0 or len(gt_pred_traffic_element) == 0:
                    continue

                gt_indices_ls = np.array(list(gt_pred_lane_segment.keys())).astype(int)
                pred_indices_ls = np.array(list(gt_pred_lane_segment.values())).astype(int)
                gt_indices_te = np.array(list(gt_pred_traffic_element.keys())).astype(int)
                pred_indices_te = np.array(list(gt_pred_traffic_element.values())).astype(int)

                preds_topology_lste = np.zeros_like(gts_topology_lste, dtype=gts_topology_lste.dtype)
                xs = gt_indices_ls[:, None].repeat(len(gt_indices_te), 1)
                ys = gt_indices_te[None, :].repeat(len(gt_indices_ls), 0)
                preds_topology_lste[xs, ys] = preds_topology_lste_unmatched[pred_indices_ls][:, pred_indices_te]

                acc.append(_topology_f1_undirected(gts_topology_lste, preds_topology_lste))

    return np.float32(np.mean(acc)) if len(acc) > 0 else np.float32(0)

def lane_segment_distance_c(gt: dict, pred: dict) -> float:
    """Fast distance for masking (confidence-free): Chamfer distance of centerlines."""
    gt_centerline = gt['centerline']
    pred_centerline = pred['centerline']
    return chamfer_distance(gt_centerline, pred_centerline)

def lane_segment_distance_f(gt: dict, pred: dict) -> float:
    r"""
    Calculate distance between lane segments.

    Parameters
    ----------
    gt : dict
    pred : dict

    Returns
    -------
    float
    
    """
    gt_centerline = gt['centerline']
    pred_centerline = pred['centerline']
    return frechet_distance(gt_centerline, pred_centerline)

def _mAP_type_lsls(gts, preds, distance_thresholds):
    r"""
    Calculate mAP on topology among lane segments.

    Parameters
    ----------
    gts : dict
        Dict storing ground truth for all samples.
    preds : dict
        Dict storing predictions for all samples.
    distance_thresholds : list
        Distance thresholds.

    Returns
    -------
    float
        mAP over all samples abd distance thresholds.

    """
    acc_type = []
    for distance_threshold in distance_thresholds:
        for token in gts.keys():

            idx_match_gt = preds[token][f'lane_segment_{distance_threshold}_idx_match_gt']
            gt_pred = {m: i for i, m in enumerate(idx_match_gt) if not np.isnan(m)} ##哪些匹配上了

            gt_indices = np.array(list(gt_pred.keys())).astype(int)
            pred_indices = np.array(list(gt_pred.values())).astype(int)

            gt_left_type = [lanesegment['left_laneline_type'] for lanesegment in gts[token]['lane_segment']]
            gt_right_type = [lanesegment['right_laneline_type'] for lanesegment in gts[token]['lane_segment']]

            pred_left_type = [lanesegment['left_laneline_type'] for lanesegment in preds[token]['lane_segment']]
            pred_right_type = [lanesegment['right_laneline_type'] for lanesegment in preds[token]['lane_segment']]

            left_type_matching = np.zeros(len(gt_left_type))
            right_type_matching = np.zeros(len(gt_left_type))

            for idx, gt_indice in enumerate(gt_indices):
                if gt_left_type[gt_indice] == pred_left_type[pred_indices[idx]]:
                    left_type_matching[gt_indice] = 1.0
                if gt_right_type[gt_indice] == pred_right_type[pred_indices[idx]]:
                    right_type_matching[gt_indice] = 1.0

            acc_type.append((np.mean(left_type_matching)+np.mean(right_type_matching))/2)

    if len(acc_type) == 0:
        return np.float32(0)

    return np.hstack(acc_type).mean()

def _scene_id_from_token(token):
    """Extract scene_id (segment_id) from token."""
    if isinstance(token, tuple) and len(token) >= 2:
        return str(token[1])  # segment_id is the second element
    if isinstance(token, str):
        s = token.strip()
        if s.startswith('(') and ',' in s:
            try:
                return s.split(',')[1].strip().strip("'\"")
            except Exception:
                pass
        parts = s.split('/')
        if len(parts) >= 2:
            return parts[1]
    raise ValueError(f'Unsupported token format: {token}')

def _timestamp_from_token(token):
    """Extract timestamp from token."""
    if isinstance(token, tuple) and len(token) >= 3:
        return str(token[2])  # timestamp is the third element
    if isinstance(token, str):
        s = token.strip()
        if s.startswith('(') and ',' in s:
            try:
                parts = s.split(',')
                if len(parts) >= 3:
                    return parts[2].strip().strip(")'\"")
            except Exception:
                pass
    return None

def _group_tokens_by_scene(tokens):
    """Group tokens by scene_id."""
    scenes = {}
    for t in tokens:
        sid = _scene_id_from_token(t)
        scenes.setdefault(sid, []).append(t)
    return scenes

def _F1_per_frame(gts, preds, distance_matrices, distance_threshold: float, object_type: str,
                   filter, gt_num: int):
    """Calculate F1 for each frame individually."""
    frame_metrics = {}
    
    for token in gts.keys():
        gt_list = [gt for gt in gts[token][object_type] if filter(gt)]
        pred_list = [pred for pred in preds[token][object_type] if filter(pred)]
        
        dm = distance_matrices[token].copy()
        gt_mask = np.array([filter(gt) for gt in gts[token][object_type]], dtype=bool)
        pred_mask = np.array([filter(pr) for pr in preds[token][object_type]], dtype=bool)
        
        dm = dm[gt_mask, :][:, pred_mask] if (dm.size > 0) else dm
        
        tp, fp, fn, _ = _match_and_inject(
            gts=gt_list,
            preds=pred_list,
            distance_matrix=dm,
            distance_threshold=distance_threshold,
            preds_token_dict=None,
            object_type=None,
        )
        
        _, _, f1 = _precision_recall_f1(tp, fp, fn)
        frame_metrics[token] = {
            'F1': float(f1),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }
    
    return frame_metrics

def _TOP_lsls_with_unmatched_penalty(gts, preds, distance_thresholds, gt_num):
    acc = []
    for distance_threshold in distance_thresholds:
        for token in gts.keys():
            gts_topology_lsls = gts[token]['topology_lsls']
            if 0 in gts_topology_lsls.shape:
                continue

            preds_topology_lsls_unmatched = preds[token]['topology_lsls']
            idx_match_gt = preds[token][f'lane_segment_{distance_threshold}_idx_match_gt']

            acc.append(_topology_f1_lsls_with_unmatched_penalty(
                gts_topo=gts_topology_lsls,
                preds_topo_unmatched=preds_topology_lsls_unmatched,
                idx_match_gt=idx_match_gt,
            ))
    target_len = 3 * gt_num
    if len(acc) < target_len:
        acc.extend([0.0] * (target_len - len(acc)))

    return np.float32(np.mean(acc)) if len(acc) > 0 else np.float32(0)

def lanesegnet_evaluate(ground_truth, predictions, verbose=True):

    if isinstance(ground_truth, str):
        ground_truth = io.pickle_load(ground_truth)

    if predictions is None:
        preds = {}
        print('\nDummy evaluation on ground truth.\n')
    else:
        if isinstance(predictions, str):
            predictions = io.pickle_load(predictions)
        predictions = predictions['results']

    gts = {}
    preds = {}
    gt_num = len(ground_truth)

    for token in predictions.keys():  ##ground_truth
        gts[token] = ground_truth[token]['annotation']

        preds[token] = predictions[token]['predictions']

    assert set(gts.keys()) == set(preds.keys()), '#frame differs'

    """
        calculate distances between gts and preds    
    """

    distance_matrices = {
        'laneseg': {},
        'area': {},
    }

    for token in tqdm(gts.keys(), desc='calculating distances:', ncols=80, disable=not verbose):

        mask = pairwise(
            [gt for gt in gts[token]['lane_segment']],
            [pred for pred in preds[token]['lane_segment']],
            lane_segment_distance_c,
            relax=True,
        ) < THRESHOLDS_LANESEG[-1]
      
        distance_matrices['laneseg'][token] = pairwise(
            [gt for gt in gts[token]['lane_segment']],
            [pred for pred in preds[token]['lane_segment']],
            lane_segment_distance_f,
            mask=mask,
            relax=True,
        )

        distance_matrices['area'][token] = pairwise(
            [gt for gt in gts[token]['area']],
            [pred for pred in preds[token]['area']],
            area_distance,
        )
  
    """
        evaluate
    """

    metrics = {
        'mF1': 0
    }
    
    metrics['F1_ls'] = _mF1_over_threshold(
        gts=gts, 
        preds=preds, 
        distance_matrices=distance_matrices['laneseg'], 
        distance_thresholds=THRESHOLDS_LANESEG,
        object_type='lane_segment',
        gt_num = gt_num,
        filter=lambda _: True,
        inject=True, # save tp for eval on graph
    )

    metrics['F1_ls'] = metrics['F1_ls'].mean()

    metrics['F1_ped'] = _mF1_over_threshold(
        gts=gts, 
        preds=preds, 
        distance_matrices=distance_matrices['area'], 
        distance_thresholds=THRESHOLDS_AREA, 
        object_type='area',
        gt_num = gt_num,
        filter=lambda x: x['category'] == 1,
        inject=False,
    )

    metrics['F1_ped'] = metrics['F1_ped'].mean()
    metrics['TOP_lsls'] = _TOP_lsls_with_unmatched_penalty(gts, preds, THRESHOLDS_LANESEG, gt_num)
    # metrics['mAP_type'] =   _mAP_type_lsls(gts, preds, THRESHOLDS_LANESEG)
    metrics['mF1'] = (metrics['F1_ls'] + metrics['F1_ped']) / 2
    print('metrics',metrics)

    # Calculate per-scene and per-frame metrics
    metrics['per_scene'] = {}
    metrics['per_frame'] = {}
    
    scene_groups = _group_tokens_by_scene(gts.keys())
    
    # Calculate per-scene metrics
    for scene_id, scene_tokens in scene_groups.items():
        gts_scene = {t: gts[t] for t in scene_tokens}
        preds_scene = {t: preds[t] for t in scene_tokens}
        dist_scene = {
            'laneseg': {t: distance_matrices['laneseg'][t] for t in scene_tokens},
            'area': {t: distance_matrices['area'][t] for t in scene_tokens},
        }
        
        # Calculate F1 for lane segments
        f1_ls_scene = _mF1_over_threshold(
            gts=gts_scene,
            preds=preds_scene,
            distance_matrices=dist_scene['laneseg'],
            distance_thresholds=THRESHOLDS_LANESEG,
            object_type='lane_segment',
            gt_num=len(scene_tokens),
            filter=lambda _: True,
            inject=True,
        ).mean()
        
        # Calculate F1 for pedestrian crossings
        f1_ped_scene = _mF1_over_threshold(
            gts=gts_scene,
            preds=preds_scene,
            distance_matrices=dist_scene['area'],
            distance_thresholds=THRESHOLDS_AREA,
            object_type='area',
            gt_num=len(scene_tokens),
            filter=lambda x: x['category'] == 1,
            inject=False,
        ).mean()
        
        # Calculate topology F1
        top_lsls_scene = _TOP_lsls_with_unmatched_penalty(
            gts_scene, preds_scene, THRESHOLDS_LANESEG, len(scene_tokens)
        )
        
        metrics['per_scene'][scene_id] = {
            'F1_ls': float(f1_ls_scene),
            'F1_ped': float(f1_ped_scene),
            'TOP_lsls': float(top_lsls_scene),
            'num_frames': len(scene_tokens),
        }
    
    # Calculate per-frame metrics (average over all thresholds)
    for token in gts.keys():
        scene_id = _scene_id_from_token(token)
        timestamp = _timestamp_from_token(token)
        
        # Calculate F1 for each threshold and average
        f1_ls_list = []
        f1_ped_list = []
        precision_ls_list = []
        recall_ls_list = []
        precision_ped_list = []
        recall_ped_list = []
        tp_ls_list = []
        fp_ls_list = []
        fn_ls_list = []
        tp_ped_list = []
        fp_ped_list = []
        fn_ped_list = []
        
        for threshold_ls in THRESHOLDS_LANESEG:
            frame_f1_ls = _F1_per_frame(
                gts={token: gts[token]},
                preds={token: preds[token]},
                distance_matrices={token: distance_matrices['laneseg'][token]},
                distance_threshold=threshold_ls,
                object_type='lane_segment',
                filter=lambda _: True,
                gt_num=1,
            )
            fm = frame_f1_ls[token]
            f1_ls_list.append(fm['F1'])
            precision_ls_list.append(fm['precision'])
            recall_ls_list.append(fm['recall'])
            tp_ls_list.append(fm['tp'])
            fp_ls_list.append(fm['fp'])
            fn_ls_list.append(fm['fn'])
        
        for threshold_area in THRESHOLDS_AREA:
            frame_f1_ped = _F1_per_frame(
                gts={token: gts[token]},
                preds={token: preds[token]},
                distance_matrices={token: distance_matrices['area'][token]},
                distance_threshold=threshold_area,
                object_type='area',
                filter=lambda x: x['category'] == 1,
                gt_num=1,
            )
            fm = frame_f1_ped[token]
            f1_ped_list.append(fm['F1'])
            precision_ped_list.append(fm['precision'])
            recall_ped_list.append(fm['recall'])
            tp_ped_list.append(fm['tp'])
            fp_ped_list.append(fm['fp'])
            fn_ped_list.append(fm['fn'])
        
        # Average over thresholds
        metrics['per_frame'][token] = {
            'scene_id': scene_id,
            'timestamp': timestamp,
            'F1_ls': np.mean(f1_ls_list),
            'F1_ped': np.mean(f1_ped_list),
            'precision_ls': np.mean(precision_ls_list),
            'recall_ls': np.mean(recall_ls_list),
            'precision_ped': np.mean(precision_ped_list),
            'recall_ped': np.mean(recall_ped_list),
            'tp_ls': int(np.mean(tp_ls_list)),
            'fp_ls': int(np.mean(fp_ls_list)),
            'fn_ls': int(np.mean(fn_ls_list)),
            'tp_ped': int(np.mean(tp_ped_list)),
            'fp_ped': int(np.mean(fp_ped_list)),
            'fn_ped': int(np.mean(fn_ped_list)),
        }
    
    # Sort scenes by performance
    sorted_scenes = sorted(
        metrics['per_scene'].items(),
        key=lambda x: (x[1]['F1_ls'] + x[1]['F1_ped']) / 2
    )
    
    # Find best frames (top 10)
    frames_list = []
    for token, frame_metrics in metrics['per_frame'].items():
        f1_ls = frame_metrics['F1_ls']
        f1_ped = frame_metrics['F1_ped']
        f1_overall = (f1_ls + f1_ped) / 2
        frames_list.append((token, frame_metrics, f1_ls, f1_ped, f1_overall))
    
    # Sort by different metrics
    best_frames_ls = sorted(frames_list, key=lambda x: x[2], reverse=True)[:10]
    best_frames_ped = sorted(frames_list, key=lambda x: x[3], reverse=True)[:10]
    best_frames_overall = sorted(frames_list, key=lambda x: x[4], reverse=True)[:10]
    
    if verbose:
        # print('\n📊 Overall metrics results:', metrics)
        print(f'\n📈 Per-scene metrics (showing top 5 and bottom 5):')
        print('Top 5 scenes:')
        for scene_id, scene_metrics in sorted_scenes[-5:]:
            print(f'  Scene {scene_id}: F1_ls={scene_metrics["F1_ls"]:.4f}, '
                  f'F1_ped={scene_metrics["F1_ped"]:.4f}, '
                  f'TOP_lsls={scene_metrics["TOP_lsls"]:.4f}, '
                  f'frames={scene_metrics["num_frames"]}')
        print('Bottom 5 scenes:')
        for scene_id, scene_metrics in sorted_scenes[:5]:
            print(f'  Scene {scene_id}: F1_ls={scene_metrics["F1_ls"]:.4f}, '
                  f'F1_ped={scene_metrics["F1_ped"]:.4f}, '
                  f'TOP_lsls={scene_metrics["TOP_lsls"]:.4f}, '
                  f'frames={scene_metrics["num_frames"]}')
        
        print(f'\n🏆 Top 10 best frames by F1_ls:')
        for rank, (token, fm, f1_ls, f1_ped, f1_overall) in enumerate(best_frames_ls, 1):
            print(f'  {rank}. Scene {fm["scene_id"]}, Timestamp {fm["timestamp"]}')
            print(f'     F1_ls={f1_ls:.4f}, Precision={fm["precision_ls"]:.4f}, Recall={fm["recall_ls"]:.4f}')
            print(f'     TP={fm["tp_ls"]}, FP={fm["fp_ls"]}, FN={fm["fn_ls"]}')
        
        print(f'\n🏆 Top 10 best frames by F1_ped:')
        for rank, (token, fm, f1_ls, f1_ped, f1_overall) in enumerate(best_frames_ped, 1):
            print(f'  {rank}. Scene {fm["scene_id"]}, Timestamp {fm["timestamp"]}')
            print(f'     F1_ped={f1_ped:.4f}, Precision={fm["precision_ped"]:.4f}, Recall={fm["recall_ped"]:.4f}')
            print(f'     TP={fm["tp_ped"]}, FP={fm["fp_ped"]}, FN={fm["fn_ped"]}')
        
        print(f'\n🏆 Top 10 best frames by overall F1 (avg):')
        for rank, (token, fm, f1_ls, f1_ped, f1_overall) in enumerate(best_frames_overall, 1):
            print(f'  {rank}. Scene {fm["scene_id"]}, Timestamp {fm["timestamp"]}')
            print(f'     F1_ls={f1_ls:.4f}, F1_ped={f1_ped:.4f}, Avg F1={f1_overall:.4f}')
            print(f'     Precision_ls={fm["precision_ls"]:.4f}, Recall_ls={fm["recall_ls"]:.4f}')
            print(f'     Precision_ped={fm["precision_ped"]:.4f}, Recall_ped={fm["recall_ped"]:.4f}')
        import pdb; pdb.set_trace()
    return metrics
