import pycocotools.mask as cocomask
import pdb
import json
import cv2
import os
import os.path
import shutil

import sys
sys.path.append("../")

import external.mots_tools.mots_common.io as io1


DATA_DIR="../../data/KITTI_MOTS/"
SEQMAP_DIR="../external/mots_tools/mots_eval/"


def load_seqmap(seqmap_filename):
    print("Loading seqmap...")
    seqmap = []
    max_frames = {}
    with open(seqmap_filename, "r") as fh:
        for i, l in enumerate(fh):
            fields = l.split(" ")
            seq = "%04d" % int(fields[0])
            seqmap.append(seq)
            max_frames[seq] = int(fields[3])
    return seqmap, max_frames

def get_mots_dict(data_dir):

    all_dirs = sorted(os.listdir(os.path.join(data_dir, "images")))

    idx = 0
    record = {}
    record["categories"] = [
        {"id": 0, "name": "car"},
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "ambiguous"},
    ]
    record["videos"] = []
    record["annotations"] = []
    for dir1 in all_dirs:
        print(dir1)
        image_names = sorted(os.listdir(os.path.join(data_dir, "images", dir1)))
        if os.path.exists(os.path.join(data_dir, "instances_txt")):
            objects_per_frame = io1.load_txt(
                os.path.join(data_dir, "instances_txt", dir1 + ".txt")
            )
        else:
            objects_per_frame = []

        file_names = []
        image_ids = []
        unique_track_ids = list(
            set(
                [
                    ob.track_id
                    for k in objects_per_frame.keys()
                    for ob in objects_per_frame[k]
                    if k in [int(img[:-4]) for img in image_names]
                ]
            )
        )

        labels = {
            k: {
                "video_id": idx,
                "vid": dir1,
                "id": id + len(record["annotations"]),
                "image_ids": [],
                "file_names": [],
                "bboxes": [],
                "areas": [],
                "segmentations": [],
            }
            for id, k in enumerate(unique_track_ids)
        }

        for i in range(0, len(image_names)):

            image_name = image_names[i]
            image_id = str(idx) + "_" + image_names[i]
            image_path = os.path.join(
                os.path.join(data_dir, "images", dir1, image_name)
            )
            np_img = cv2.imread(image_path)

            file_names.append(dir1 + "/" + image_name)
            image_ids.append(image_id)
            height = np_img.shape[0]
            width = np_img.shape[1]
            del np_img
            for obj_id in unique_track_ids:
                labels[obj_id]["image_ids"].append(image_id)
                labels[obj_id]["file_names"].append(dir1 + "/" + image_name)
                labels[obj_id]["height"] = height
                labels[obj_id]["width"] = width

                if (
                    objects_per_frame != []
                    and int(image_name[:-4]) in list(objects_per_frame.keys())
                    and obj_id
                    in [k.track_id for k in objects_per_frame[int(image_name[:-4])]]
                ):
                    objects = [
                        k
                        for k in objects_per_frame[int(image_name[:-4])]
                        if k.track_id == obj_id
                    ]
                    if len(objects) == 1:
                        obj = objects[0]
                        labels[obj_id]["areas"].append(int(cocomask.area(obj.mask)))
                        labels[obj_id]["segmentations"].append(
                            {
                                "size": obj.mask["size"],
                                "counts": obj.mask["counts"].decode(encoding="UTF-8"),
                            }
                        )
                        labels[obj_id]["bboxes"].append(
                            [
                                cocomask.toBbox(obj.mask)[0],
                                cocomask.toBbox(obj.mask)[1],
                                cocomask.toBbox(obj.mask)[0]
                                + cocomask.toBbox(obj.mask)[2],
                                cocomask.toBbox(obj.mask)[1]
                                + cocomask.toBbox(obj.mask)[3],
                            ]
                        )

                        labels[obj_id]["category_id"] = (
                            obj.class_id - 1 if obj.class_id in [1, 2] else 2
                        )  # car, ped, ignore
                        if labels[obj_id]["category_id"] != 2:
                            labels[obj_id]["iscrowd"] = 0
                        else:
                            labels[obj_id]["iscrowd"] = 1
                    elif len(objects) > 1:  # for other ambiguous categories
                        mask = np.zeros(
                            (labels[obj_id]["height"], labels[obj_id]["width"]),
                            dtype=bool,
                        )
                        for obj in objects:
                            mask = np.logical_or(mask, cocomask.decode(obj.mask) > 0)
                        mask = cocomask.encode(np.asfortranarray(mask * 1))

                        labels[obj_id]["areas"].append(int(cocomask.area(mask)))
                        labels[obj_id]["segmentations"].append(
                            {
                                "size": mask["size"],
                                "counts": mask["counts"].decode(encoding="UTF-8"),
                            }
                        )
                        labels[obj_id]["bboxes"].append(cocomask.toBbox(mask).tolist())
                        labels[obj_id]["category_id"] = 2
                        labels[obj_id]["iscrowd"] = 1
                else:
                    # mask=cocomask.encode(np.asfortranarray(np.zeros((labels[obj_id]["height"],labels[obj_id]["width"]), dtype=np.uint8)))
                    labels[obj_id]["areas"].append(None),
                    labels[obj_id]["segmentations"].append(None)
                    labels[obj_id]["bboxes"].append(None)

        for k in labels:
            labels[k]["length"] = len(image_ids)
        video = {
            "id": idx,
            "file_names": file_names,
            "image_ids": image_ids,
            "length": len(image_ids),
            "height": height,
            "width": width,
        }
        record["annotations"] = record["annotations"] + [labels[k] for k in labels]
        record["videos"].append(video)
        idx += 1

    return record


def get_all_mots_dict2(data_dir=".", save_file="trial.json"):

    mots_dict = get_mots_dict(data_dir)
    json.dump(mots_dict, open(save_file, "w"))

    return


if __name__ == "__main__":

    image_dir=DATA_DIR+"data_tracking_image_2/training/image_02"
    instances_dir=DATA_DIR+"instances_txt"
    for mode in ["train", "val"]:
        seq_map, max_frame=load_seqmap(SEQMAP_DIR+mode+".seqmap")
        os.makedirs(os.path.join(DATA_DIR,mode), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR,mode,"images"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR,mode,"instances_txt"), exist_ok=True)
        for seq in seq_map:
            if not os.path.exists(os.path.join(DATA_DIR,mode,"images", seq)):
                shutil.copytree(os.path.join(image_dir, seq), os.path.join(DATA_DIR,mode,"images", seq))
            if not os.path.exists(os.path.join(DATA_DIR,mode, "instances_txt", seq)+".txt"):
                shutil.copy(os.path.join(instances_dir, seq)+".txt", os.path.join(DATA_DIR,mode, "instances_txt", seq)+".txt")


        get_all_mots_dict2(
            data_dir=DATA_DIR+mode,
            save_file=DATA_DIR+mode+"_full.json",
        )


