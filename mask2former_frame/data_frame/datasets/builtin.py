# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .amazon import (
    register_amazon_video_instances,
    register_amazon_image_instances,
    register_amazon_syn_video_instances,
    register_amazon_videoperframe_instances,
    register_yuxiangds_instances,
    register_realtabletop_instances,
    _get_amazon_2022_instances_meta,
)

_PREDEFINED_SPLITS_AMAZON_2022_VIDEO = {
    "amazon_video_real_v1_test": ("amazon_syn/annotated_real_v1_resized/images",
                            "amazon_syn/annotated_real_v1_resized/video_instances.json"),
}

_PREDEFINED_SPLITS_AMAZON_2022_IMAGE = {
    "amazon_image_real_v1_test": ("amazon_syn/annotated_real_v1_resized/images",
                            "amazon_syn/annotated_real_v1_resized/image_instances.json"),
}

_PREDEFINED_SPLITS_AMAZON_2022_VIDEOPERFRAME = {
    "amazon_videoperframe_real_v1_test": ("amazon_syn/annotated_real_v1_resized/images",
                            "amazon_syn/annotated_real_v1_resized/video_instances.json"),
}

_PREDEFINED_SPLITS_AMAZON_SYN_V2_VIDEO = {
    "amazon_video_syn_v2_train": ("amazon_syn/bin_syn/train_shard_000000.h5",
                            "amazon_syn/bin_syn/train_shard_000000_coco.json"),
    "amazon_video_syn_v2_val": ("amazon_syn/bin_syn/test_shard_000000.h5",
                            "amazon_syn/bin_syn/test_shard_000000_coco.json"),
}
_PREDEFINED_SPLITS_NEWBIN_SYN = {
    "newbin_syn_train": ("amazon_syn/bin_syn/train_shard_1k.h5",
                            "amazon_syn/bin_syn/train_shard_1k_coco.json"),
    "newbin_syn_val": ("amazon_syn/bin_syn/test_shard_200.h5",
                            "amazon_syn/bin_syn/test_shard_200_coco.json"),
}
_PREDEFINED_SPLITS_TABLETOP_SYN = {
    "tabletop_syn_train": ("amazon_syn/tabletop_syn/train_shard_000000.h5",
                            "amazon_syn/tabletop_syn/train_shard_000000_coco.json"),
    "tabletop_syn_val": ("amazon_syn/tabletop_syn/test_shard_000000.h5",
                            "amazon_syn/tabletop_syn/test_shard_000000_coco.json"),
    "tabletop_syn_mini_val": ("amazon_syn/tabletop_syn/mini_test.h5",
                            "amazon_syn/tabletop_syn/mini_test_coco.json"),
}

_PREDEFINED_SPLITS_YuXiangDS = {
    "YuXiangDS_train": ("self-supervised-segmentation/filtered_train",
                            "self-supervised-segmentation/filtered_train/video_instances.json"),
    "YuXiangDS_test": ("self-supervised-segmentation/filtered_test",
                            "self-supervised-segmentation/filtered_test/video_instances.json"),
}

_PREDEFINED_SPLITS_REALTABLETOP = {
    "tabletop_real_val": ("amazon_syn/tabletop_real/images",
                            "amazon_syn/tabletop_real/video_instances.json"),
    "tabletop_real_val_oneframe": ("singleframe_camera/images",
                            "singleframe_camera/video_instances.json"),
}

def register_all_amazon_2022_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_AMAZON_2022_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_amazon_video_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_amazon_2022_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_AMAZON_2022_IMAGE.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_amazon_image_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_amazon_2022_videoperframe(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_AMAZON_2022_VIDEOPERFRAME.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_amazon_videoperframe_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_amazon_v2_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_AMAZON_SYN_V2_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_amazon_syn_video_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_newbin_syn(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_NEWBIN_SYN.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_amazon_syn_video_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_tabletop_syn(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TABLETOP_SYN.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_amazon_syn_video_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_yuxiangds(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YuXiangDS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_yuxiangds_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_realtabletop(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REALTABLETOP.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_realtabletop_instances(
            key,
            _get_amazon_2022_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_amazon_2022_video(_root)
    register_all_amazon_2022_image(_root)
    register_all_amazon_v2_video(_root)
    register_all_amazon_2022_videoperframe(_root)
    register_all_newbin_syn(_root)
    register_all_tabletop_syn(_root)
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_yuxiangds(_root)
    register_all_realtabletop(_root)