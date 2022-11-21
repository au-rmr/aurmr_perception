# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
# import pickle
# from collections import OrderedDict
import pycocotools.mask as mask_util
# import torch
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from tabulate import tabulate

# import detectron2.utils.comm as comm
# from detectron2.config import CfgNode
# from detectron2.data import MetadataCatalog
# from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco, instances_to_coco_json
from detectron2.evaluation.evaluator import DatasetEvaluator
# from detectron2.evaluation.fast_eval_api import COCOeval_opt
# from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
# from detectron2.utils.logger import create_small_table
from pycocotools.cocoeval import COCOeval

from scipy.optimize import linear_sum_assignment

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval
from pycocotools import mask as maskUtils
from detectron2.structures import Boxes, BoxMode, pairwise_iou

from torchvision.ops import masks_to_boxes
# implement InstanceSegEvaluator DatasetEvaluator for instance segmetnat
class FrameInstanceSegEvaluator(DatasetEvaluator):
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.result_table = None
        self.COCOeval = COCOeval

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self.predictions = []
        self.ground_truth = []

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for sequence_input, sequence_predict in zip(inputs, outputs):
            print(f"sequence {sequence_input[0]['image_id']}")
            for frame_idx, (frame_input, frame_predict) in enumerate(zip(sequence_input, sequence_predict)):
                image_unique_id = frame_input['image_id']*1000 + frame_idx
                self.process_single_frame(frame_input, frame_predict, image_unique_id)
                # print(frame_input, frame_predict)

    def process_single_frame(self, frame_input, frame_predict, image_unique_id):
        """
        Process a single frame of input and output.
        """
        # compute iou between each dt and gt region
        # dt = instances_to_coco_json(frame_predict['instances'], image_unique_id)
        # gt = instances_to_coco_json(frame_input['instances'], image_unique_id)
        # g = [g['segmentation'] for g in gt]
        # d = [d['segmentation'] for d in dt]
        # print(frame_predict['instances'].pred_masks.shape)
        # print(frame_input['instances'].gt_masks.shape)
        d = [
            mask_util.encode(np.array(mask.cpu().numpy()[:, :, None], order="F", dtype="uint8"))[0]
            for mask in frame_predict['instances'].pred_masks
        ]
        g = [
            mask_util.encode(np.array(mask.cpu().numpy()[:, :, None], order="F", dtype="uint8"))[0]
            for mask in frame_input['instances'].gt_masks
        ]
        iscrowd = [0 for o in g]
        ious = maskUtils.iou(d,g,iscrowd) # d x g
        gt_indices, dt_indices = linear_sum_assignment(-ious.T)
        result_per_frame = []
        for gt_idx, dt_idx in zip(gt_indices, dt_indices):
            result = {'mask': frame_predict['']}
            frame_predict['instances'].pred_masks[dt_idx] = 
        return ious

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass

def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    img_ids=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_results = copy.deepcopy(coco_results)
    # When evaluating mask AP, if the results contain bbox, cocoapi will
    # use the box area as the area of the instance, instead of the mask area.
    # This leads to a different definition of small/medium/large.
    # We remove the bbox field to let mask AP use mask area.
    for c in coco_results:
        c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = YTVOSeval(coco_gt, coco_dt)
    # For COCO, the default max_dets_per_image is [1, 10, 100].
    max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    if instances.has('pred_boxes'):
        boxes = instances.pred_boxes.tensor.numpy()
    elif instances.has('gt_boxes'):
        boxes = instances.gt_boxes.tensor.numpy()
    elif instances.has('pred_masks'):
        masks_to_boxes(instances['instances']['pred_masks'])
    elif instances.has('gt_masks'):
        masks_to_boxes(instances['instances']['gt_masks'])
    else:
        raise ValueError("Instances must have boxes or masks!")
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    if instances.has('scores'):
        scores = instances.scores.tolist()
    else:
        scores = None    
    if instances.has('pred_classes'):
        classes = instances.pred_classes.tolist()
    else:
        classes = instances.gt_classes.tolist()

    has_mask = instances.has("pred_masks") or instances.has("gt_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        masks = instances.gt_masks if instances.has("gt_masks") else instances.pred_masks
        rles = [
            mask_util.encode(np.array(mask.cpu().numpy()[:, :, None], order="F", dtype="uint8"))[0]
            for mask in masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
        }

        result["score"] = scores[k] if scores is not None else 1.0
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results