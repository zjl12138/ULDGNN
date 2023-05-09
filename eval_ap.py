import numpy as np
import json
from lib.evaluators import eval_map
import torch
det_results = torch.load("./det_results.pkl")
annotations = torch.load("./gt_annotation.pkl")
mean_ap, eval_results = eval_map(
    det_results, annotations, iou_thr = 0.50 , use_legacy_coordinate=True)
print(mean_ap) 
''' 
import numpy as np

from lib.evaluators import eval_map, tpfp_default, tpfp_imagenet

det_bboxes = np.array([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])
gt_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])
gt_ignore = np.array([[5, 5, 10, 20], [6, 10, 10, 19]])


def test_tpfp_imagenet():

    result = tpfp_imagenet(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=True)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()

    result = tpfp_imagenet(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=False)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()


def test_tpfp_default():

    result = tpfp_default(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=True)

    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()
    result = tpfp_default(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=False)

    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()


def test_eval_map():

    # 2 image and 2 classes
    det_results = [[det_bboxes, det_bboxes], [det_bboxes, det_bboxes]]

    labels = np.array([0, 1, 1])
    labels_ignore = np.array([0, 1])
    gt_info = {
        'bboxes': gt_bboxes,
        'bboxes_ignore': gt_ignore,
        'labels': labels,
        'labels_ignore': labels_ignore
    }
    
    annotations = [gt_info, gt_info]
    print(det_results, annotations)
    mean_ap, eval_results = eval_map(
        det_results, annotations, use_legacy_coordinate=True)
    assert 0.291 < mean_ap < 0.293
    eval_map(det_results, annotations, use_legacy_coordinate=False)
    assert 0.291 < mean_ap < 0.293

test_eval_map()
'''