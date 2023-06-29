import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
import pdb
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

# from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES as CITYSCAPES_VPS_CATEGORIES

logger = logging.getLogger(__name__)

__all__ = ["load_vps_json", "register_vps_instances"]

CITYSCAPES_VPS_CATEGORIES = [
    {
        "id": 0,
        "name": "road",
        "supercategory": "flat",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 0,
        "ori_id": 7,
        "color": [128, 64, 128],
    },
    {
        "id": 1,
        "name": "sidewalk",
        "supercategory": "flat",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 1,
        "ori_id": 8,
        "color": [244, 35, 232],
    },
    {
        "id": 2,
        "name": "building",
        "supercategory": "construction",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 2,
        "ori_id": 11,
        "color": [70, 70, 70],
    },
    {
        "id": 3,
        "name": "wall",
        "supercategory": "construction",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 3,
        "ori_id": 12,
        "color": [102, 102, 156],
    },
    {
        "id": 4,
        "name": "fence",
        "supercategory": "construction",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 4,
        "ori_id": 13,
        "color": [190, 153, 153],
    },
    {
        "id": 5,
        "name": "pole",
        "supercategory": "object",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 5,
        "ori_id": 17,
        "color": [153, 153, 153],
    },
    {
        "id": 6,
        "name": "traffic light",
        "supercategory": "object",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 6,
        "ori_id": 19,
        "color": [250, 170, 30],
    },
    {
        "id": 7,
        "name": "traffic sign",
        "supercategory": "object",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 7,
        "ori_id": 21,
        "color": [220, 220, 0],
    },
    {
        "id": 8,
        "name": "vegitation",
        "supercategory": "nature",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 8,
        "ori_id": 21,
        "color": [107, 142, 35],
    },
    {
        "id": 9,
        "name": "terrain",
        "supercategory": "nature",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 9,
        "ori_id": 22,
        "color": [152, 251, 152],
    },
    {
        "id": 10,
        "name": "sky",
        "supercategory": "sky",
        "isthing": 0,
        "instance_eval": 0,
        "trainid": 10,
        "ori_id": 23,
        "color": [70, 130, 180],
    },
    {
        "id": 11,
        "name": "person",
        "supercategory": "human",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 11,
        "ori_id": 24,
        "color": [220, 20, 60],
    },
    {
        "id": 12,
        "name": "rider",
        "supercategory": "human",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 12,
        "ori_id": 25,
        "color": [255, 0, 0],
    },
    {
        "id": 13,
        "name": "car",
        "supercategory": "vehicle",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 13,
        "ori_id": 26,
        "color": [0, 0, 142],
    },
    {
        "id": 14,
        "name": "truck",
        "supercategory": "vehicle",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 14,
        "ori_id": 27,
        "color": [0, 0, 70],
    },
    {
        "id": 15,
        "name": "bus",
        "supercategory": "vehicle",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 15,
        "ori_id": 28,
        "color": [0, 60, 100],
    },
    {
        "id": 16,
        "name": "train",
        "supercategory": "vehicle",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 16,
        "ori_id": 31,
        "color": [0, 80, 100],
    },
    {
        "id": 17,
        "name": "motorcycle",
        "supercategory": "vehicle",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 17,
        "ori_id": 32,
        "color": [0, 0, 230],
    },
    {
        "id": 18,
        "name": "bicycle",
        "supercategory": "vehicle",
        "isthing": 1,
        "instance_eval": 1,
        "trainid": 18,
        "ori_id": 33,
        "color": [119, 11, 32],
    },
]


def _get_vps_instances_meta():

    thing_ids = [k["id"] for k in CITYSCAPES_VPS_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_VPS_CATEGORIES]
    thing_classes = [k["name"] for k in CITYSCAPES_VPS_CATEGORIES]

    assert len(thing_ids) == 19
    # # Mapping from the incontiguous category id to an id in [0, 24]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }

    return ret


def load_vps_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from .ytvis_api.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)

    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info(
        "Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [
            os.path.join(image_root, vid_dict["file_names"][i])
            for i in range(vid_dict["length"])
        ]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_vps_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_vps_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="vps", **metadata
    )
