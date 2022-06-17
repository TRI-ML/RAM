from __future__ import absolute_import

import _init_paths
import numpy as np
import math
import argparse
from typing import Dict

from utils.lacater_eval import ResultsAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--exp', type=str, help='Experiment name')   
    parser.add_argument('--dataset_path', type=str, default='../data/la_cater/test_data', help='Path to the evaluation dataset')  
    parser.add_argument('--max_vids', type=int, default=0)

    return parser.parse_args()


def cal_map(x: np.ndarray):
    return x.sum() / x.shape[0]

if __name__ == '__main__':
    args = parse_args()

    max_vids = None
    if args.max_vids > 0:
        max_vids = args.max_vids

    analyzer: ResultsAnalyzer = ResultsAnalyzer.init_from_files('../exp/tracking/%s/results/' % args.exp, '%s/labels/' % args.dataset_path, [0.5,0.1], max_vids=max_vids)

    containment_mask: Dict[str, np.ndarray] = analyzer.get_frames_mask('%s/containment_and_occlusions/containment_annotations.txt' % args.dataset_path )
    containment_static_mask: Dict[str, np.ndarray] = analyzer.get_frames_mask('%s/containment_and_occlusions/containment_only_static_annotations.txt' % args.dataset_path)
    containment_with_move_mask: Dict[str, np.ndarray] = analyzer.get_frames_mask('%s/containment_and_occlusions/containment_with_move_annotations.txt' % args.dataset_path)
    visibility_gt_0_mask = analyzer.get_frames_mask('%s/containment_and_occlusions/visibility_rate_gt_0.txt' % args.dataset_path)

    for metric, agg_func in zip(["iou", "map"], [np.mean, cal_map]):
        analyzer.compute_aggregated_metric(aggregations_name="overall", aggregation_function=agg_func, metric=metric)

        analyzer.compute_aggregated_metric_masking_frames("contained", agg_func, containment_mask, metric=metric)
        analyzer.compute_aggregated_metric_masking_frames("static_contained", agg_func, containment_static_mask, metric=metric)
        analyzer.compute_aggregated_metric_masking_frames("contained_with_move", agg_func, containment_with_move_mask, metric=metric)

        analyzer.compute_aggregated_metric_masking_frames("visibility_gt_0", agg_func, visibility_gt_0_mask, metric=metric)
        not_visible_at_all_mask = {video: ~mask for video, mask in visibility_gt_0_mask.items()}
        not_visible_not_contained_mask = {video: np.logical_and(not_visible_mask, ~containment_mask[video]) for video, not_visible_mask, in not_visible_at_all_mask.items()}
        analyzer.compute_aggregated_metric_masking_frames("full_occlusion", agg_func, not_visible_not_contained_mask, metric=metric)

    for metric in analyzer.videos_metrics.keys():
        values = analyzer.videos_metrics[metric]
        metric_agg = 0
        count = 0
        for video in values.keys():
            if math.isnan(values[video]):
                continue

            score = values[video]
            metric_agg += score
            count += 1
        
        print('%s: %f' % (metric, metric_agg / count))
