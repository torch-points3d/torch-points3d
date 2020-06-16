""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import numpy as np
from multiprocessing import Pool

from torch_points3d.utils.box_utils import box3d_iou
from torch_points3d.datasets.object_detection.box_data import BoxData


def voc_ap(recall, precision):
    """ ap = voc_ap(recall, precision)
    Compute PASCAL VOC AP given precision and recall.
    recall and precision contain one element per detected instance,
    ordered by certainty (most certain element first)
    (see here for an explanation https://github.com/rafaelpadilla/Object-Detection-Metrics)
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_det_cls(pred, gt, ovthresh=0.25):
    """ Generic functions to compute precision/recall for object detection
    for a single class. For each detected box (starting with the most confident),
    find the box with the highest overlap in the ground truth  and mark that one as "detected".
    The same box being detected multiple times counts as false positive.

    Input:
        pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        ovthresh: scalar, iou threshold
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "detected": det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "detected": []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    num_images = len(image_ids)
    tp = np.zeros(num_images)
    fp = np.zeros(num_images)
    for d in range(num_images):
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = box3d_iou(bb, BBGT[j, ...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R["detected"][jmax]:
                tp[d] = 1.0
                R["detected"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return rec, prec, ap


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh)
    return (rec, prec, ap)


def eval_detection(pred_all, gt_all, ovthresh=0.25, processes=4):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [BoxData]}
            gt_all: map of {img_id: [BoxData]}
            ovthresh: scalar, iou threshold
            processes: number of threads to use
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for box in pred_all[img_id]:
            if box.classname not in pred:
                pred[box.classname] = {}
            if img_id not in pred[box.classname]:
                pred[box.classname][img_id] = []
            if box.classname not in gt:
                gt[box.classname] = {}
            if img_id not in gt[box.classname]:
                gt[box.classname][img_id] = []
            pred[box.classname][img_id].append((box.corners3d, box.score))
    for img_id in gt_all.keys():
        for box in gt_all[img_id]:
            if box.classname not in gt:
                gt[box.classname] = {}
            if img_id not in gt[box.classname]:
                gt[box.classname][img_id] = []
            gt[box.classname][img_id].append(box.corners3d)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=processes)
    ret_values = p.map(
        eval_det_cls_wrapper,
        [(pred[classname], gt[classname], ovthresh) for classname in gt.keys() if classname in pred],
    )
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0

    return rec, prec, ap
