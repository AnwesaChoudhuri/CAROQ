import sys
import numpy as np
import cv2
from PIL import Image
import json
import pdb
import torch
import os
from functools import partial
import time
from multiprocessing import Pool
import pycocotools.mask as cocomask
from detectron2.data import detection_utils as utils
from panopticapi.utils import rgb2id

mode="train"

input_dir="../data/Cityscapes/cityscapes_vps/"+mode+"/panoptic_video/"
json_file="../data/Cityscapes/cityscapes_vps/panoptic_gt_"+mode+"_city_vps.json"
output_file="../data/Cityscapes/cityscapes_vps/panoptic_vps_"+mode+".json"

#json_file="../data/Cityscapes/cityscapes_vps/panoptic_im_train_city_vps.json"

json_data=json.load(open(json_file))
output_data={}
output_data["categories"]=json_data["categories"]
output_data["videos"]=[]
output_data["annotations"]=[]
videos=np.unique([a["file_name"][:a["file_name"].find("_")] for a in json_data["images"]])

# cityscapes_dataset=build_vps_dataset()
# abc=cityscapes_dataset.dataset.prepare_train_img(0)

for idx, v in enumerate(videos):
    print(idx)
    files=[i for i in json_data["images"] if i["file_name"].startswith(v)]
    anno_files=[i for i in json_data["annotations"] if i["file_name"].startswith(v)]
    temp={}
    temp["id"]=idx
    temp["file_names"]=[i["file_name"] for i in files]
    temp["image_ids"]=[i["file_name"] for i in files]
    temp["length"]=len(files)
    temp["height"]=files[0]["height"]
    temp["width"]=files[0]["width"]

    unique_ids=np.unique([j["id"] for k in anno_files for j in k["segments_info"]])
    unique_objects=[{"image_id":[], "file_name":[],"segmentations":[],"areas":[], "bboxes":[]} for u in unique_ids]
    for annos in anno_files:
        pan_seg_gt = utils.read_image(input_dir+annos["file_name"], "RGB")
        pan_seg_gt = rgb2id(pan_seg_gt)
        for u_idx, u in enumerate(unique_ids):

            unique_objects[u_idx]["image_id"].append(annos["image_id"])
            unique_objects[u_idx]["height"]=files[0]["height"]
            unique_objects[u_idx]["width"]=files[0]["width"]
            unique_objects[u_idx]["length"]=len(files)

            unique_objects[u_idx]["file_name"].append(annos["file_name"])
            seg=cocomask.encode(np.asfortranarray(pan_seg_gt==u))
            unique_objects[u_idx]["segmentations"].append({"size":seg["size"], "counts":seg["counts"].decode(encoding='UTF-8')})
            unique_objects[u_idx]["bboxes"].append(cocomask.toBbox(seg).tolist())

            if u in [k["id"] for k in annos["segments_info"]]:

                obj=[k for k in annos["segments_info"] if k["id"]==u][0]
                unique_objects[u_idx]["areas"].append(obj["area"])
                unique_objects[u_idx]["category_id"]=obj["category_id"]
                unique_objects[u_idx]["iscrowd"]=obj["iscrowd"]
                unique_objects[u_idx]["id"]=obj["id"]
                unique_objects[u_idx]["video_id"]=idx


            else:
                unique_objects[u_idx]["areas"].append(0)

    output_data["annotations"]=output_data["annotations"]+unique_objects
    output_data["videos"].append(temp)

json.dump(output_data, open(output_file, 'w'))
