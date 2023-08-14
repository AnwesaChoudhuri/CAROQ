import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ytvis_2022_instances_meta,
    _get_ovis_instances_meta,
)

from .cityscapes_vps import _get_vps_instances_meta, register_vps_instances
from .mots import _get_mots_instances_meta, register_mots_instances


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": (
        "Youtube_vis_2019/train/JPEGImages",
        "Youtube_vis_2019/train.json",
    ),
    "ytvis_2019_val": (
        "Youtube_vis_2019/valid/JPEGImages",
        "Youtube_vis_2019/valid.json",
    ),
    "ytvis_2019_test": (
        "Youtube_vis_2019/test/JPEGImages",
        "Youtube_vis_2019/test.json",
    ),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": (
        "Youtube_vis_2021/train/JPEGImages",
        "Youtube_vis_2021/train.json",
    ),
    "ytvis_2021_val": (
        "Youtube_vis_2021/valid/JPEGImages",
        "Youtube_vis_2021/valid.json",
    ),
    "ytvis_2021_test": (
        "Youtube_vis_2021/test/JPEGImages",
        "Youtube_vis_2021/test.json",
    ),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("OVIS/train/JPEGImages", "OVIS/train.json"),
    "ovis_val": ("OVIS/valid/JPEGImages", "OVIS/valid.json"),
    "ovis_test": ("OVIS/test/JPEGImages", "OVIS/test.json"),
}


# ==== Predefined splits for CITYSCAPES VPS ===========
_PREDEFINED_SPLITS_CITYSCAPES_VPS = {
    "cityscapes_vps_train": (
        "Cityscapes/cityscapes_vps/train/img",
        "Cityscapes/cityscapes_vps/panoptic_vps_train.json",
    ),
    "cityscapes_vps_val": (
        "Cityscapes/cityscapes_vps/val/img",
        "Cityscapes/cityscapes_vps/panoptic_vps_val.json",
    ),
    "cityscapes_vps_test": (
        "Cityscapes/cityscapes_vps/test/img",
        "Cityscapes/cityscapes_vps/panoptic_vps_test.json",
    ),
}

# ==== Predefined splits for KITTI MOTS ===========
_PREDEFINED_SPLITS_KITTIMOTS = {
    "kittimots_train": ("KITTI_MOTS/train/images",
                         "KITTI_MOTS/train_full.json"),
    "kittimots_val": ("KITTI_MOTS/val/images",
                         "KITTI_MOTS/val_full.json"),
    "kittimots_test": ("KITTI_MOTS/test/images",
                         ""),
}

# ==== Predefined splits for MOTS 2020 ===========
_PREDEFINED_SPLITS_MOTS2020 = {
    "mots2020_train": ("../../data/MOTS2020/train/images",
                         "../../data/MOTS2020/train_full.json"),
    "mots2020_val": ("../../data/MOTS2020/train/images",
                         "../../data/MOTS2020/train_full.json"),
    "mots2020_test": ("../../MOTS2020/test/images",
                         ""),
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


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_vps(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_CITYSCAPES_VPS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vps_instances(
            key,
            _get_vps_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_kittimots(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_KITTIMOTS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mots_instances(
            key,
            _get_mots_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_mots2020(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_MOTS2020.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mots_instances(
            key,
            _get_mots_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "../data")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
    register_all_vps(_root)
    register_all_kittimots(_root)
    register_all_mots2020(_root)
