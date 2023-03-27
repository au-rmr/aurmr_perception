# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
import numpy as np

from visualizer import TrackVisualizer
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, test_type="coco"):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        assert not parallel, NotImplemented
        self.test_type = test_type
        if test_type == 'coco':
            self.predictor = FramePredictor(cfg)
        elif test_type == 'ytvis':
            self.predictor = VideoPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        raise NotImplementedError("Use `run_on_sequence` instead.")
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output
    def run_on_sequence(self, sequence, labels=None, separate_anno=False):
        if self.test_type=='coco':
            return self.run_on_sequence_coco(sequence)
        elif self.test_type=='ytvis':
            return self.run_on_sequence_ytvis(sequence, labels=labels, separate_anno=separate_anno)
        else:
            NotImplementedError("Test type not supported")
            
    def run_on_sequence_coco(self, images):
        """
        Visualizes predictions on a sequence of images.
        Args:
            imgs (list[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
        Yields:
            ndarray: BGR visualizations of each image.
        """
        vis_output = []
        predictions = self.predictor(images)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        for frame_idx, image in enumerate(images):
            image = image[:, :, ::-1]
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
            instances = predictions[0][frame_idx]["instances"].to(self.cpu_device)
            vis_output.append(visualizer.draw_instance_predictions(predictions=instances))

        return predictions, vis_output
    
    def calc_IoU(self, pred, label):
        intersection = torch.sum(pred*label)
        union = torch.sum((pred + label - pred*label))
        # if (label==1).sum()==0:
        #     return -1
        # if union == 0:
        #     return 0

        return intersection, union

    def run_on_sequence_ytvis(self, frames, labels=None, separate_anno=False):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            labels: masks of shape (num_frames, H, W), or (num_frames, num_objects, H, W) if separate_anno is True
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(frames)
        image_size = predictions["image_size"]
        pred_scores = predictions["pred_scores"]
        pred_labels = predictions["pred_labels"]
        pred_masks = predictions["pred_masks"]

        # Calculate the max sequence IoU between the predicted masks and the ground truth masks
        if labels is not None:
            if not separate_anno:
                intersections = np.zeros((len(pred_labels), len(np.unique(labels))-2, len(frames)))
                unions = np.zeros((len(pred_labels), len(np.unique(labels))-2, len(frames)))
                for i in range(len(pred_labels)):
                    for j in range(1, len(np.unique(labels))-1):
                        for f in range(len(frames)):
                            intersections[i, j-1, f] = torch.sum(pred_masks[i][f]*(labels[f] == j))
                            unions[i, j-1, f] = torch.sum((pred_masks[i][f] + (labels[f] == j) - pred_masks[i][f]*(labels[f] == j)))
                max_seq_iou = np.nanmax(np.sum(intersections, axis=2)/np.sum(unions, axis=2), axis=1)
                max_seq_iou_idx = np.nanargmax(np.sum(intersections, axis=2)/np.sum(unions, axis=2), axis=1)
                iou = np.nan_to_num(intersections/unions)
                max_frame_iou = np.max(iou, axis=1)
                max_frame_iou_idx = np.argmax(iou, axis=1)
            else:
                intersections = np.zeros((len(pred_labels), labels.shape[1], len(frames)))
                unions = np.zeros((len(pred_labels), labels.shape[1], len(frames)))
                for i in range(len(pred_labels)):
                    for j in range(labels.shape[1]):
                        for f in range(len(frames)):
                            intersections[i, j, f] = torch.sum(pred_masks[i][f]*(labels[f][j]))
                            unions[i, j, f] = torch.sum((pred_masks[i][f] + (labels[f][j]) - pred_masks[i][f]*(labels[f][j])))
                max_seq_iou = np.nanmax(np.sum(intersections, axis=2)/np.sum(unions, axis=2), axis=1)
                max_seq_iou_idx = np.nanargmax(np.sum(intersections, axis=2)/np.sum(unions, axis=2), axis=1)
                iou = np.nan_to_num(intersections/unions)
                max_frame_iou = np.max(iou, axis=1)
                max_frame_iou_idx = np.argmax(iou, axis=1)
        frame_masks = list(zip(*pred_masks))
        total_vis_output = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
            ins = Instances(image_size)
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

            if labels is not None:
                vis_output = visualizer.draw_instance_predictions(predictions=ins, max_frame_iou=max_frame_iou[:,frame_idx], max_seq_iou=max_seq_iou,
                                                                 max_frame_iou_id=max_frame_iou_idx[:,frame_idx], max_seq_iou_id=max_seq_iou_idx)
            else:
                vis_output = visualizer.draw_instance_predictions(predictions=ins)
            total_vis_output.append(vis_output)

        return predictions, total_vis_output
    
    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        # raise NotImplementedError("Use `run_on_sequence` instead.")
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """
    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        raise NotImplementedError("Use `FramePredictor` instead.")
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


class FramePredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_sequence):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # batched_sequence = []
            # for frame in original_sequence:
            #     # Apply pre-processing to image.
            #     if self.input_format == "RGB":
            #         # whether the model expects BGR inputs or RGB
            #         frame = frame[:, :, ::-1]
            #     # Apply pre-processing to image.
            #     height, width = frame.shape[:2]
            #     image = self.aug.get_transform(frame).apply_image(frame)
            #     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            #     inputs = {"image": image, "height": height, "width": width}
            #     batched_sequence.append(inputs)
            # predictions = self.model([batched_sequence])
            # batched_sequence = []
            images = []
            height, width = original_sequence[0].shape[:2]
            for frame in original_sequence:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    frame = frame[:, :, ::-1]
                # Apply pre-processing to image.
                height, width = frame.shape[:2]
                image = self.aug.get_transform(frame).apply_image(frame)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                images.append(image)
            sequence = {"image": images, "height": height, "width": width}
            predictions = self.model([sequence])
            return predictions


class VideoPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __call__(self, frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_frames = []
            for original_image in frames:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_frames.append(image)

            inputs = {"image": input_frames, "height": height, "width": width}
            predictions = self.model([inputs])
            return predictions