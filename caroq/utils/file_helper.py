import numpy as np
from collections import namedtuple
import pycocotools.mask as cocomask
import os

import torch
import cv2

import pdb


def export_tracking_result_in_kitti_format(tag, tracks, add_masks, model_str, out_folder="", start_time_at_1=False):
    if out_folder == "":
        out_folder = "forwarded/" + model_str + "/tracking_data"
    os.makedirs(out_folder, exist_ok=True)
    out_filename = out_folder + "/" + tag + ".txt"
    with open(out_filename, "w") as f:
        start = 1 if start_time_at_1 else 0
        for t, tracks_t in enumerate(tracks, start):  # TODO this works?
            for track in tracks_t:
                if add_masks:
                    # MOTS methods
                    if track.class_== 0:
                        print(track.t, track.track_id, track.class_+1,
                              *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)
                    if track.class_ == 1:
                       print(track.t, track.track_id + 500, track.class_+1,
                             *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)
                else:
                    # MOT methods
                    print(str(track.t+1) + "," + str(track.track_id)+","+str(track.box[0])+","+str(track.box[1])+","+str(track.box[2]-track.box[0])+","+str(track.box[3]-track.box[1])+","+str(track.score)+",-1,-1,-1", file=f)

                #print(t,track.track_id,track.box[0],track.box[1],track.box[2]-track.box[0], track.box[3]-track.box[1], track.score, -1, -1, -1, file=f)
