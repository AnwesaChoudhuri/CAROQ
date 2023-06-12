

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ytvis_2022_instances_meta,
    _get_ovis_instances_meta
)

from .cityscapes_vps import  _get_vps_instances_meta, register_vps_instances

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

# ==== Predefined splits for YTVIS 2022 ===========
_PREDEFINED_SPLITS_YTVIS_2022 = {
    "ytvis_2022_val": ("ytvis_2022/valid/JPEGImages",
                       "ytvis_2022/valid.json"),
    "ytvis_2022_test": ("ytvis_2022/test/JPEGImages",
                       "ytvis_2022/test.json"),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train/JPEGImages",
                         "ovis/train.json"),
    "ovis_val": ("ovis/valid/JPEGImages",
                       "ovis/valid.json"),
    "ovis_test": ("ovis/test/JPEGImages",
                        "ovis/test.json"),
}



# ==== Predefined splits for CITYSCAPES VPS ===========
_PREDEFINED_SPLITS_CITYSCAPES_VPS = {
    "cityscapes_vps_train": ("../data/Cityscapes/cityscapes_vps/train/img",
                         "../data/Cityscapes/cityscapes_vps/panoptic_vps_train.json"),
    "cityscapes_vps_val": ("../data/Cityscapes/cityscapes_vps/val/img",
                         "../data/Cityscapes/cityscapes_vps/panoptic_vps_val.json"),
    "cityscapes_vps_test": ("../data/Cityscapes/cityscapes_vps/test/img",
                         "../data/Cityscapes/cityscapes_vps/panoptic_vps_test.json"),
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

def register_all_ytvis_2022(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2022.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2022_instances_meta(),
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



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ytvis_2022(_root)
    register_all_ovis(_root)
    register_all_vps(_root)
