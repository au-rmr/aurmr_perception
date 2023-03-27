# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
# fmt: on

import tempfile
import time
import warnings
import json

import cv2
import numpy as np
import tqdm
import h5py
import pycocotools.mask as mask_util

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from predictor import VisualizationDemo

from mask2former_video.data_frame import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--sequence",
        nargs="+",
        help="A list of space separated input sequence folder; "
        "or a single glob pattern such as 'directory/*'",
    )
    parser.add_argument(
        "--annotation",
         help="A json file that contains the mask",
    )
    parser.add_argument(
        "--frame_name",
        nargs="+",
        help="A list of space separated input frame name in sequence folder; "
        "or a single glob pattern such as '*'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--test_type",
        help="test as coco or ytvis"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--h5_path",
        help="A h5 file that contains sequences that is used for visualization"
    )
    parser.add_argument(
        "--h5_test_num",
        type=int,
        help="The number of sequences in an h5 file that is used for visualization"
    )
    parser.add_argument("--include_label", action="store_true", help="Include label as part of the input")
    parser.add_argument("--BGR_input", action="store_true", help="Input is RGB")
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, test_type=args.test_type)

    if args.sequence:
        if len(args.sequence) == 1:
            args.sequence = glob.glob(os.path.expanduser(args.sequence[0]))
            assert args.sequence, "The sequence path(s) was not found"
        if args.annotation:
            with open(args.annotation, "r") as f:
                annotations = json.load(f)
        annot_idx = 0
        for seq_idx, sequence_path in tqdm.tqdm(enumerate(sorted(list(args.sequence))), disable=not args.output):
            if len(args.frame_name) == 1:
                sequence_frame = glob.glob(os.path.join(sequence_path, args.frame_name[0]))
                if not sequence_frame:
                    continue
            else:
                sequence_frame = []
                for frame_name in args.frame_name:
                    if os.path.exists(os.path.join(sequence_path, frame_name)):
                        sequence_frame.append(os.path.join(sequence_path, frame_name))
                    else:
                        x = glob.glob(os.path.join(sequence_path, frame_name))
                        # assert len(x) == 1, "The frame name(s) was not found"
                        sequence_frame.extend(x)
            
            sequence_imgs = []
            sequence_frame.sort()
            for frame_idx, frame_path in enumerate(sequence_frame):
                print(frame_path)
                # use PIL, to be consistent with evaluation
                if not args.BGR_input:
                    img = read_image(frame_path, format="BGR")
                else:
                    img = read_image(frame_path, format="RGB")
                sequence_imgs.append(img)
            
            if not len(sequence_imgs):
                continue

            if args.annotation:
                seq_name = args.sequence[0].split("/")[-2]
                name_idx_map = {"scene_01" : 0, "scene_02" : 16, "scene_03" : 28}
                sequence_labels = [list() for _ in range(len(sequence_imgs))]
                while annot_idx < len(annotations['annotations']) and \
                    annotations['annotations'][annot_idx]['video_id'] <= seq_idx+name_idx_map[seq_name]:
                    if annotations['annotations'][annot_idx]['video_id'] != seq_idx+name_idx_map[seq_name]:
                        annot_idx += 1
                        continue
                    for frame_idx in range(len(annotations['annotations'][annot_idx]['segmentations'])):
                        if annotations['annotations'][annot_idx]['segmentations'][frame_idx] is not None:
                            sequence_labels[frame_idx].append(mask_util.decode(annotations['annotations'][annot_idx]['segmentations'][frame_idx]))
                        else:
                            sequence_labels[frame_idx].append(np.zeros(sequence_imgs[frame_idx].shape[:2]))
                    annot_idx += 1
                sequence_labels = np.array(sequence_labels)
                print(sequence_labels.shape)
            
            start_time = time.time()
            if args.annotation:
                predictions, visualized_output = demo.run_on_sequence(sequence_imgs, sequence_labels, separate_anno=True)
            else:
                predictions, visualized_output = demo.run_on_sequence(sequence_imgs)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    sequence_path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            for frame_idx, frame_path in enumerate(sequence_frame):
                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, "_".join(os.path.normpath(frame_path).split(os.sep)[-2:]))
                    else:
                        # assert len(args.input) == 1, "Please specify a directory with args.output"
                        os.makedirs(args.output, exist_ok=True)
                        out_filename = os.path.join(args.output, "_".join(os.path.normpath(frame_path).split(os.sep)[-2:]))
                    visualized_output[frame_idx].save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, visualized_output[frame_idx].get_image()[:, :, ::-1])
                # if cv2.waitKey(0) == 27:
                #     break

    elif args.h5_path:
        f = h5py.File(args.h5_path, 'r')
        imgs = f['data']
        masks = f['mask']
        for idx in tqdm.tqdm(range(args.h5_test_num)):
            start_time = time.time()
            predictions, visualized_output = demo.run_on_sequence(imgs[idx][..., :3][..., ::-1], masks[idx])
            logger.info(
                "{}: {} in {:.2f}s".format(
                    idx,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            
            for frame_idx in range(imgs[idx].shape[0]):
                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, "seq_" + str(idx) + '_' + "frame_" + str(frame_idx) + '.png')
                    else:
                        # assert len(args.input) == 1, "Please specify a directory with args.output"
                        os.makedirs(args.output, exist_ok=True)
                        out_filename = os.path.join(args.output, "seq_" + str(idx) + '_' + "frame_" + str(frame_idx) + '.png')
                    visualized_output[frame_idx].save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, visualized_output[frame_idx].get_image()[:, :, ::-1])

    elif args.webcam:
        raise NotImplementedError("Input is not supported yet.")
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        raise NotImplementedError("Input is not supported yet.")
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
