#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
from tqdm import tqdm
from openlanev2.lanesegment.io import io
from openlanev2.lanesegment.evaluation.distance import (pairwise, area_distance,
                                                        lane_segment_distance)
from openlanev2.centerline.evaluation.distance import chamfer_distance, frechet_distance

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
                      preds_token_dict: dict = None,
                      object_type: str = None):
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
                       filter, inject: bool, gt_num: int):
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

    f1 = f1 * len(gts) / gt_num

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

def lane_segment_distance_c(gt: dict, pred: dict) -> float:
    """Fast distance for masking (confidence-free): Chamfer distance of centerlines."""
    gt_centerline = gt['centerline']
    pred_centerline = pred['centerline']
    return chamfer_distance(gt_centerline, pred_centerline)

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

    for token in predictions.keys():
        gts[token] = ground_truth[token]['annotation']
        if predictions is None:
            preds[token] = gts[token]
            for i, _ in enumerate(preds[token]['lane_segment']):
                preds[token]['lane_segment'][i]['confidence'] = np.float32(1)
            for i, _ in enumerate(preds[token]['area']):
                preds[token]['area'][i]['confidence'] = np.float32(1)
            for i, _ in enumerate(preds[token]['traffic_element']):
                preds[token]['traffic_element'][i]['confidence'] = np.float32(1)
        else:
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
            lane_segment_distance,
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
        gt_num=gt_num,
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
        gt_num=gt_num,
        filter=lambda x: x['category'] == 1,
        inject=False,
    )

    metrics['F1_ped'] = metrics['F1_ped'].mean()
    metrics['TOP_lsls'] = _TOP_lsls_with_unmatched_penalty(gts, preds, THRESHOLDS_LANESEG, gt_num)

    print('metrics results:', metrics)
    return metrics
