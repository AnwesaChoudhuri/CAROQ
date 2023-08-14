import torch
#import munkres
import numpy as np
import matplotlib.pyplot as plt
import io
import pdb
import time
import cv2
#import networkx as nx

import math
import pycocotools.mask as cocomask
import scipy.optimize as opt
from collections import namedtuple

def get_carry_forward(dets):

    dets_carry_ids=list(set(dets.track_ids[0])-set(dets.track_ids[1]))

    carry_forward=[]
    if dets_carry_ids!=[]:
        carry_forward_pos=[dets.track_ids[0].index(m) for m in dets_carry_ids]
        for k in carry_forward_pos:
            print("frame_diff:", dets.t1+1-dets.ts[0][k])
            if dets.t1+1-dets.ts[0][k]<=dets.keep_alive:
                carry_forward.append({"boxes": dets.boxes[0][k],
                                      "reids": dets.reids[0][k],
                                      "scores": dets.scores[0][k],
                                      "masks": dets.masks[0][k],
                                      "ts": dets.ts[0][k],
                                      "original_track_ids":dets.original_track_ids[0][k],
                                      "classes": dets.classes[0][k]})

    print("carry:", [forward["original_track_ids"] for forward in carry_forward], ", carry from frame: ",  [forward["ts"] for forward in carry_forward])

    return carry_forward


def get_track_elements(detections, args):
    TrackElement = namedtuple("TrackElement", ["t", "box", "track_id", "class_", "mask", "score"])

    all_tracks = []
    for t, (boxes_t, scores_t, classes_t, masks_t, track_ids_t) in enumerate(
            zip(detections.boxes, detections.scores, detections.classes, detections.masks, detections.track_ids)):
        tracks = []
        if args.use_given_detections:
            add_ = 0
        else:
            add_ = 1
        for i, (box, score, class_, mask, track_id) in enumerate(
                zip(boxes_t, scores_t, classes_t, masks_t, track_ids_t)):

            if args.mots:
                tracks.append(TrackElement(t=t, box=box.cpu().numpy(), mask=cocomask.encode(
                                               np.asfortranarray((mask > 0.5).cpu().numpy().astype(np.uint8))),
                                           # mask=cocomask.encode(np.asfortranarray((mask > 0.5).cpu().numpy())),
                                           class_=class_.item() + add_,
                                           track_id=track_id, score=score.item()))
            else:
                tracks.append(TrackElement(t=t, box=box.cpu().numpy(),
                                           mask=mask,
                                           class_=class_.item() + add_,
                                           track_id=track_id, score=score.item()))
        all_tracks.append(tracks)

    return all_tracks


def make_mask_from_box(boxes, image_shape, mots=True):
    masks=[]
    for t, boxes_t in enumerate(boxes):
        masks_t=[]
        # if boxes_t != []:
        #     m = torch.zeros(len(boxes_t), image_shape[0], image_shape[1])
        #     b = torch.stack(boxes_t).int()
        #     x = [list(range(b[i, 1], b[i, 3])) for i in range(len(b))]
        #     y = [list(range(b[i, 0], b[i, 2])) for i in range(len(b))]
        #     x_all= [[xi for xi in x[k] for yi in y[k]] for k in range(len(b))]
        #     y_all= [[yi for xi in x[k] for yi in y[k]] for k in range(len(b))]
        #     m[:,x_all, y_all]=1

        for i, b in enumerate(boxes_t): #b: xyxy
            m=torch.zeros(image_shape)
            m[int(b[1]):int(b[3]), int(b[0]):int(b[2])]=1
            masks_t.append(m)
        if masks_t!=[]:
            if mots:
                masks.append(torch.stack(masks_t))
            else:
                masks.append(cocomask.encode(np.asfortranarray(np.array(torch.stack(masks_t).permute(1, 2, 0), dtype=np.uint8))))
        else:
            if mots:
                masks.append(torch.tensor(masks_t))
            else:
                masks.append(cocomask.encode(np.asfortranarray(np.array(torch.zeros(image_shape[0], image_shape[1],0), dtype=np.uint8))))

    return masks

def make_mask_from_box_discriminative(boxes, images, image_shape, mots=True):
    masks=[]
    mask_model = MRCNN_FPN()
    mask_model.model.eval()
    for t, (boxes_t, img_path) in enumerate(zip(boxes, images)):
        print("masks", t)
        image=cv2.imread(img_path[0])
        masks_t=mask_model.predict_masks(torch.stack(boxes_t), torch.tensor(image).permute(2,0,1).unsqueeze(0).float()/255)
        masks.append(masks_t)
    return masks



def remove_detections(boxes, scores, classes, masks, reids, mots=True):
    # remove detections with very small area
    boxes_=[]
    scores_=[]
    classes_=[]
    masks_=[]
    reids_=[]

    for t, (boxes_t, scores_t, classes_t, masks_t, reids_t) in enumerate(zip(boxes, scores, classes, masks, reids)):
        boxes_t_=[]
        scores_t_=[]
        classes_t_=[]
        masks_t_=[]
        reids_t_=[]
        for i, (box, score, class_, mask, reid) in enumerate(zip(boxes_t, scores_t, classes_t, masks_t, reids_t)):
            if mots:
                criterion=mask.sum()>10
            else:
                criterion=(int(box[3])-int(box[1]))*(int(box[2])-int(box[0])) > 1

            if criterion:
                boxes_t_.append(box)
                scores_t_.append(score)
                masks_t_.append(mask)
                reids_t_.append(reid)
                classes_t_.append(class_)
        boxes_.append(boxes_t_)
        scores_.append(scores_t_)
        masks_.append(masks_t_)
        reids_.append(reids_t_)
        classes_.append(classes_t_)


    return boxes_, scores_, reids_, classes_, masks_


def remove_overlap(masks, scores, train=True):

    for t, (masks_t, scores_t) in enumerate(zip(masks, scores)):

        if len(masks_t)>=2:
            # scores_t.shape: [n]
            # masks_t.shape: [n, h, w]
            sorted_scores, sorted_idx = torch.sort(torch.tensor(scores_t))
            canvas=torch.zeros_like(masks_t[0])
            for i, (score, idx) in enumerate(zip(sorted_scores, sorted_idx)):
                masks_t[idx]=masks_t[idx]*(1-canvas)
                canvas=((canvas+masks_t[idx])>0).type(torch.uint8)

    if not train:
        return [[m for m in mt] for mt in masks]

    return masks

def rearrange_gt(gt_seq, predictions_mots, det_class_ids=[2,0], gt_class_ids=[1,2]):

    for i in range(0,len(predictions_mots["boxes"])):
        gt_seq_temp=[]
        det_masks_temp=[]
        det_boxes_temp = []
        det_scores_temp=[]
        det_classes_temp = []
        det_reids_temp = []


        del_ids=list(set(predictions_mots["classes"][i].tolist())-set(det_class_ids))
        for del_id in del_ids:
            ids=np.where(np.array(predictions_mots["classes"][i].cpu())==del_id)[0]
            temp_pred_classes = [v for i2, v in enumerate(predictions_mots["classes"][i]) if i2 not in ids]
            predictions_mots["boxes"][i] = [predictions_mots["boxes"][i][i2] for i2, _ in enumerate(predictions_mots["classes"][i]) if i2 not in ids]
            predictions_mots["scores"][i] = [predictions_mots["scores"][i][i2] for i2, _ in enumerate(predictions_mots["classes"][i]) if i2 not in ids]
            predictions_mots["reids"][i] = [predictions_mots["reids"][i][i2] for i2, _ in enumerate(predictions_mots["classes"][i]) if i2 not in ids]
            predictions_mots["masks"][i] = [predictions_mots["masks"][i][i2] for i2, _ in enumerate(predictions_mots["classes"][i]) if i2 not in ids]
            predictions_mots["classes"][i]= temp_pred_classes
        # delete other classes except car(id:1) and ped(id:2) from gt. This is essentially the ignore class 10
        gt_seq[i]=[i2 for i2 in gt_seq[i] if int(i2["track_id"].item()/1000) in gt_class_ids] #according to trackrcnn format


        # match gt with predictions and rearrange
        if len(predictions_mots["masks"][i])>0 and len(gt_seq[i])>0:
            mask_matrix=cocomask.iou(
                [cocomask.encode(np.asfortranarray((j > 0.5).cpu().numpy().astype(np.uint8))) for j in predictions_mots["masks"][i]],
                [cocomask.encode(np.asfortranarray(j["segmentation"][0].cpu().numpy().astype(np.uint8))) for j in gt_seq[i]],
                [False] * len(gt_seq[i]))
            cost_matrix = munkres.make_cost_matrix(mask_matrix)
            indexes = np.array(munkres_obj.compute(cost_matrix))

            for j in range(0,len(predictions_mots["masks"][i])):
                if j in indexes[:,0]:
                    j_pos=np.where(indexes[:,0]==j)

                    if mask_matrix[j][indexes[j_pos[0][0],1]]>0.01:

                        #gt_seq[i][indexes[j_pos[0][0], 1]]["segmentation"]=predictions_mots["masks"][i][indexes[j_pos[0][0], 0]]
                    #if gt_seq[i][indexes[j_pos[0][0], 1]]["segmentation"]
                        gt_seq[i][indexes[j_pos[0][0], 1]]["segmentation"] = predictions_mots["masks"][i][indexes[j_pos[0][0], 0]]
                        gt_seq_temp.append(gt_seq[i][indexes[j_pos[0][0],1]])
                        det_masks_temp.append(predictions_mots["masks"][i][indexes[j_pos[0][0], 0]])
                        det_scores_temp.append(predictions_mots["scores"][i][indexes[j_pos[0][0], 0]])
                        det_boxes_temp.append(predictions_mots["boxes"][i][indexes[j_pos[0][0], 0]])
                        det_classes_temp.append(predictions_mots["classes"][i][indexes[j_pos[0][0], 0]])
                        det_reids_temp.append(predictions_mots["reids"][i][indexes[j_pos[0][0], 0]])

                # else:
                #     gt_seq_temp.append(SegmentedObject([],[],-1))
        # elif len(det_masks[i])>0:
        #     for j in range(0, len(det_masks[i])):
        #         gt_seq_temp.append(SegmentedObject([],[],-1))

        gt_seq[i]=gt_seq_temp
        predictions_mots["masks"][i]=det_masks_temp
        predictions_mots["scores"][i] = det_scores_temp
        predictions_mots["boxes"][i] = det_boxes_temp
        predictions_mots["classes"][i] = det_classes_temp
        predictions_mots["reids"][i] = det_reids_temp

    return gt_seq, predictions_mots


def rearrange_gt_mot(gt_seq, predictions_mots):

    for i in range(0,len(predictions_mots["boxes"])):
        gt_seq_temp=[]
        det_masks_temp=[]
        det_boxes_temp = []
        det_scores_temp=[]
        det_classes_temp = []
        det_reids_temp = []


        # match gt with predictions and rearrange
        if len(predictions_mots["masks"][i])>0 and len(gt_seq[i])>0:
            mask_matrix=cocomask.iou(
                [cocomask.encode(np.asfortranarray((j > 0.5).cpu().numpy().astype(np.uint8))) for j in predictions_mots["masks"][i]],
                [cocomask.encode(np.asfortranarray(j["segmentation"][0].cpu().numpy().astype(np.uint8))) for j in gt_seq[i]],
                [False] * len(gt_seq[i]))
            cost_matrix = munkres.make_cost_matrix(mask_matrix)
            indexes = np.array(munkres_obj.compute(cost_matrix))

            for j in range(0,len(predictions_mots["masks"][i])):
                if j in indexes[:,0]:
                    j_pos=np.where(indexes[:,0]==j)

                    if mask_matrix[j][indexes[j_pos[0][0],1]]>0.01:

                        #gt_seq[i][indexes[j_pos[0][0], 1]]["segmentation"]=predictions_mots["masks"][i][indexes[j_pos[0][0], 0]]
                    #if gt_seq[i][indexes[j_pos[0][0], 1]]["segmentation"]
                        gt_seq[i][indexes[j_pos[0][0], 1]]["segmentation"] = predictions_mots["masks"][i][indexes[j_pos[0][0], 0]]
                        gt_seq_temp.append(gt_seq[i][indexes[j_pos[0][0],1]])
                        det_masks_temp.append(predictions_mots["masks"][i][indexes[j_pos[0][0], 0]])
                        det_scores_temp.append(predictions_mots["scores"][i][indexes[j_pos[0][0], 0]])
                        det_boxes_temp.append(predictions_mots["boxes"][i][indexes[j_pos[0][0], 0]])
                        det_classes_temp.append(predictions_mots["classes"][i][indexes[j_pos[0][0], 0]])
                        det_reids_temp.append(predictions_mots["reids"][i][indexes[j_pos[0][0], 0]])

                # else:
                #     gt_seq_temp.append(SegmentedObject([],[],-1))
        # elif len(det_masks[i])>0:
        #     for j in range(0, len(det_masks[i])):
        #         gt_seq_temp.append(SegmentedObject([],[],-1))

        gt_seq[i]=gt_seq_temp
        predictions_mots["masks"][i]=det_masks_temp
        predictions_mots["scores"][i] = det_scores_temp
        predictions_mots["boxes"][i] = det_boxes_temp
        predictions_mots["classes"][i] = det_classes_temp
        predictions_mots["reids"][i] = det_reids_temp

    return gt_seq, predictions_mots


def create_assignment_labels(labels, predictions_mots,det_class_ids=[2,0], mots=True):

    if mots:
        tracking_labels, predictions_mots_new = rearrange_gt(labels, predictions_mots,det_class_ids=det_class_ids)
    else: # mot
        tracking_labels, predictions_mots_new = rearrange_gt_mot(labels, predictions_mots)

    return tracking_labels, predictions_mots_new


def make_disjoint(tracks, strategy):
    TrackElement = namedtuple("TrackElement", ["t", "box", "track_id", "class_", "mask", "score"])
    def get_max_y(obj):
        _, y, _, h = cocomask.toBbox(obj.mask)
        return y + h

    for frame, objects in enumerate(tracks):
        if len(objects) == 0:
            continue
        if strategy == "y_pos":
            objects_sorted = sorted(objects, key=lambda x: get_max_y(x), reverse=True)
        elif strategy == "score":
            objects_sorted = sorted(objects, key=lambda x: x.score, reverse=True)
        else:
            assert False, "Unknown mask_disjoint_strategy"
        objects_disjoint = [objects_sorted[0]]
        used_pixels = objects_sorted[0].mask
        for obj in objects_sorted[1:]:
            new_mask = obj.mask
            if cocomask.area(cocomask.merge([used_pixels, obj.mask], intersect=True)) > 0.0:
                obj_mask_decoded = cocomask.decode(obj.mask)
                used_pixels_decoded = cocomask.decode(used_pixels)
                obj_mask_decoded[np.where(used_pixels_decoded > 0)] = 0
                new_mask = cocomask.encode(obj_mask_decoded)
            used_pixels = cocomask.merge([used_pixels, obj.mask], intersect=False)
            objects_disjoint.append(TrackElement(t=obj.t, box=obj.box, track_id=obj.track_id, class_=obj.class_, score=obj.score,
                                                 mask=new_mask))
        tracks[frame] = objects_disjoint
    return tracks


def apply_mask(image, mask, clr,bb=None, alpha=0.5):
    m2=np.stack([np.array(mask.cpu()), np.array(mask.cpu() ), np.array(mask.cpu())], axis=2)*alpha
    image = image*(1-m2)+(m2*clr)
    if bb is not None:
        bb=np.array(bb.detach().cpu()).astype(np.uint16)
        image=cv2.rectangle(image, (bb[0],bb[1]),(bb[2],bb[3]), clr, thickness=3)
    return image

def check_with_leaf(node,leaves, appthresh=0.95, leaf_dist=0):
  # node: dictionary {"name":str(t)+"_"+str(id), "box": data["boxes"][t][id], "reid": data["reids"][t][id]}
  # leaves: list of leaf_nodes

  leaf_name=-1
  node_id=[]
  min_dist=0.01
  min_app =appthresh#0.95#0.816
  lf_id=0
  temp_dist = cf.find_cost_matrix_bb_dist(torch.stack([node["box"]]), torch.stack([lf_node["box"] for lf_node in leaves]))
  if leaf_dist==0:
      temp_app = cf.find_cost_matrix_app([node["reid"]], [lf_node["reid"] for lf_node in leaves])
      temp_app1 = cf.find_cost_matrix_app([node["reid"]], [lf_node["reid1"] for lf_node in leaves])
      temp_app2 = cf.find_cost_matrix_app([node["reid"]], [lf_node["reid2"] for lf_node in leaves])
      if "reid1" in node.keys():
          temp_n1_app = cf.find_cost_matrix_app([node["reid1"]], [lf_node["reid"] for lf_node in leaves])
          temp_n1_app1 = cf.find_cost_matrix_app([node["reid1"]], [lf_node["reid1"] for lf_node in leaves])
          temp_n1_app2 = cf.find_cost_matrix_app([node["reid1"]], [lf_node["reid2"] for lf_node in leaves])

      min_app=min(temp_app.min().item(), min_app)
      #print(node["name"], [l["name"] for l in leaves])
      #print(temp_dist, temp_app)
      #min_dist = min(temp_dist.min().item(), min_dist)


      if  min_app in temp_app:#min_dist in temp_dist and
          #pos_dist=torch.where(temp_dist==min_dist)[1].item()
          pos_app = torch.where(temp_app == min_app)[1].item()
          #if pos_app==pos_dist:

          leaf_name=leaves[pos_app]["name"]
          node_id=pos_app
      else:
          pos_app = torch.argmin(temp_app).item()
          if temp_app1[0][pos_app]<=min_app and temp_app1[0][pos_app]<=min_app:
              leaf_name = leaves[pos_app]["name"]
              node_id = pos_app

      if node_id == [] and "reid1" in node.keys():
          pos_n1_app = torch.argmin(temp_n1_app).item()
          if temp_n1_app[0][pos_n1_app].item() <= min_app or (
                  temp_n1_app1[0][pos_n1_app].item() <= min_app and temp_n1_app2[0][pos_n1_app].item() <= min_app):
              leaf_name = leaves[pos_n1_app]["name"]
              node_id = pos_n1_app
            #print("Inside")

      return leaf_name,node_id
  else:
      min_dist = min(temp_dist.min().item(), min_dist)
      if min_dist in temp_dist:
          pos_dist = torch.where(temp_dist == min_dist)[1].item()
          leaf_name = leaves[pos_dist]["name"]
          node_id = pos_dist
      return leaf_name, node_id

def assign_ids(data,G, y, leaf=False,K=15, second_order=True, train=False,keep_alive=45,appthresh=0.95, leaf_dist=0):
    start = 0
    all_tracks = []
    for t, det_t in enumerate(data["boxes"]):
        tracks = []

        if start == 0:
            track_ids = np.array(range(1, len(det_t) + 1)).astype(np.int64)
            leaf_nodes = []
            if leaf:
                for id in range(0, len(det_t)):
                    leaf_nodes.append({"name":str(t)+"_"+str(id), "box": data["boxes"][t][id], "reid": data["reids"][t][id]})
            track_ids_prev = np.array([])
            track_counter = len(det_t) + 1
            start = 1
        elif t > 0 and t < len(data["boxes"]):
            assignment= np.stack(np.where(G.nodes[str(t - 1) + "_" + str(y[t-1])]["node_assignment"].cpu()), axis=1)
            track_ids_prev2 = track_ids_prev
            track_ids_prev = track_ids
            track_ids = np.zeros((len(det_t)), dtype=np.int64)
            new_id_list = []
            for id in range(0, len(det_t)):
                new_id = 1

                if id in assignment[:, 1] and len(track_ids_prev) > 0:
                    pos = np.where(assignment[:, 1] == id)
                    if train:
                        condition=True
                    else:
                        condition=G.nodes[str(t - 1) + "_" + str(y[t - 1])]["cost_matrix_iou"][assignment[pos, 0][0][0]][
                            assignment[pos, 1][0][0]] < 1
                    if condition:
                        track_ids[id] = int(track_ids_prev[assignment[pos, 0]])
                        if leaf:
                            leaf_nodes.append({"name":str(t)+"_"+str(id), "box": data["boxes"][t][id], "reid": data["reids"][t][id]})
                            if str(t - 1) + "_" + str(assignment[pos, 0][0][0]) in [l["name"] for l in leaf_nodes]:
                                # leaf_nodes.remove(track_ids_prev[assignment[pos,0]])
                                leaf_nodes.remove({"name":str(t - 1) + "_" + str(assignment[pos, 0][0][0]),"box": data["boxes"][t-1][assignment[pos, 0][0][0]], "reid": data["reids"][t-1][assignment[pos, 0][0][0]]})
                        new_id = 0

                elif t > 1 and K> 1 and second_order:
                    if len(data["boxes"][t - 2]) > len(data["boxes"][t - 1]) and len(data["boxes"][t - 1]) < len(det_t):
                        cost_matrix1_iou=G.nodes[str(t - 1) + "_" + str(y[t - 1])]["cost_matrix_iou_2nd_order"]
                        find_extras = np.where(cost_matrix1_iou[:,id] < 1)[0]
                        if len(find_extras) > 0 and id in find_extras:
                            arg_extra = find_extras[
                                np.argmin(cost_matrix1_iou[np.where(cost_matrix1_iou[:,id] < 1)[0], id])]
                            if track_ids_prev2[arg_extra] not in track_ids and track_ids_prev2[
                                arg_extra] not in track_ids_prev:
                                track_ids[id] = track_ids_prev2[arg_extra]

                                leaf_nodes.append({"name":str(t) + "_" + str(id), "box": data["boxes"][t][id], "reid": data["reids"][t][id]})
                                if str(t - 2) + "_" + str(arg_extra) in [l_n["name"] for l_n in leaf_nodes]:
                                    leaf_nodes.remove({"name":str(t - 2) + "_" + str(arg_extra), "box": data["boxes"][t-2][arg_extra], "reid": data["reids"][t-2][arg_extra]})
                                    # leaf_nodes.remove(track_ids_prev2[arg_extra]) #remove from leaf nodes because edge is drawn from it now

                                new_id = 0

                if new_id == 1:
                    new_id_list.append(id)
                    new_id = 0

            leaf_nodes_prev = []
            if leaf:
                for nodes in leaf_nodes:
                    if int(nodes["name"][:nodes["name"].find("_")]) != t:
                        leaf_nodes_prev.append(nodes)


            for id in new_id_list:
                # print(t, id)

                new_name = -1

                if leaf and leaf_nodes_prev!=[]:
                    #print(t)

                    for lf in leaf_nodes_prev:
                        lf["reid1"]=torch.zeros_like(lf["reid"])+5
                        lf["reid2"] = torch.zeros_like(lf["reid"]) + 5
                        lf_t=int(lf["name"][:lf["name"].find("_")])
                        search_id=all_tracks[lf_t][int(lf["name"][lf["name"].find("_")+1:])]
                        lf_count = 0
                        for t_ in range(lf_t-1, 0, -1):
                            if search_id in all_tracks[t_]:
                                if lf_count==0:
                                    lf["reid1"]=data["reids"][t_][all_tracks[t_].index(search_id)]
                                    lf_count+=1
                                elif lf_count == 1:
                                    lf["reid2"] = data["reids"][t_][all_tracks[t_].index(search_id)]
                                    lf_count += 1
                                else:
                                    break

                    curr_node = {"name": str(t) + "_" + str(id), "box": data["boxes"][t][id],
                                 "reid": data["reids"][t][id]}
                    if t < len(data["boxes"]) - 1:
                        asgn_next = np.stack(np.where(G.nodes[str(t) + "_" + str(y[t])]["node_assignment"].cpu()),
                                             axis=1)
                        if id in asgn_next[:, 0]:
                            next_pos = np.where(asgn_next[:, 0] == id)[0][0]
                            curr_node["reid1"] = data["reids"][t + 1][asgn_next[next_pos, 1]]

                    new_name, node_id = check_with_leaf(curr_node,
                                                         leaf_nodes_prev, appthresh=appthresh, leaf_dist=leaf_dist)
                    for lf in leaf_nodes_prev:
                        lf.pop("reid1",None)
                        lf.pop("reid2", None)

                if new_name == -1:
                    new_name = int(track_counter)
                    track_counter += 1
                    track_ids[id] = int(new_name)

                else:

                    if new_name in [i_n["name"] for i_n in leaf_nodes_prev]:
                        t1=int(new_name[:new_name.find("_")])
                        id1=int(new_name[new_name.find("_")+1:])
                        leaf_nodes.remove({"name":new_name, "box": data["boxes"][t1][id1], "reid": data["reids"][t1][id1]})
                        leaf_nodes_prev.remove({"name":new_name, "box": data["boxes"][t1][id1], "reid": data["reids"][t1][id1]})
                        track_ids[id] = all_tracks[t1][id1]

                if leaf:
                    leaf_nodes.append({"name":str(t)+"_"+str(id), "box": data["boxes"][t][id], "reid": data["reids"][t][id]})



        all_tracks.append(list(track_ids))
        if t > keep_alive:
            for nodes in leaf_nodes:
                if int(nodes["name"][:nodes["name"].find("_")]) <= t - keep_alive:
                    leaf_nodes.remove(nodes)



    return all_tracks

def assign_ids_new(data,G, y, leaf=False,K=15, second_order=True, train=False,keep_alive=45,appthresh=0.95, leaf_dist=0, mots=True):
    start = 0
    all_tracks = []
    for t, det_t in enumerate(data["boxes"]):
        tracks = []

        if start == 0:
            track_ids = np.array(range(1, len(det_t) + 1)).astype(np.int64)
            leaf_nodes = []
            track_ids_prev = np.array([])
            track_counter = len(det_t) + 1
            start = 1
        elif t > 0 and t < len(data["boxes"]):
            assignment= np.stack(np.where(G.nodes[str(t - 1) + "_" + str(y[t-1])]["node_assignment"].cpu()), axis=1)
            track_ids_prev2 = track_ids_prev
            track_ids_prev = track_ids
            track_ids = np.zeros((len(det_t)), dtype=np.int64)
            new_id_list = []
            for id in range(0, len(det_t)):
                new_id = 1
                if id in assignment[:, 1] and len(track_ids_prev) > 0:
                    pos = np.where(assignment[:, 1] == id)
                    if train:
                        condition=True
                    else:
                        condition=G.nodes[str(t - 1) + "_" + str(y[t - 1])]["cost_matrix_iou"][assignment[pos, 0][0][0]][
                            assignment[pos, 1][0][0]] < 1
                    if condition:
                        track_ids[id] = int(track_ids_prev[assignment[pos, 0]])
                        new_id = 0

                elif t > 1 and K> 1 and second_order:
                    if len(data["boxes"][t - 2]) > len(data["boxes"][t - 1]) and len(data["boxes"][t - 1]) < len(det_t):
                        cost_matrix1_iou=G.nodes[str(t - 1) + "_" + str(y[t - 1])]["cost_matrix_iou_2nd_order"]
                        find_extras = np.where(cost_matrix1_iou[:,id] < 1)[0]
                        if len(find_extras) > 0 and id in find_extras:
                            arg_extra = find_extras[
                                np.argmin(cost_matrix1_iou[np.where(cost_matrix1_iou[:,id] < 1)[0], id])]
                            if track_ids_prev2[arg_extra] not in track_ids and track_ids_prev2[
                                arg_extra] not in track_ids_prev:
                                track_ids[id] = track_ids_prev2[arg_extra]

                                leaf_nodes.append({"name":str(t) + "_" + str(id), "box": data["boxes"][t][id], "reid": data["reids"][t][id]})
                                if str(t - 2) + "_" + str(arg_extra) in [l_n["name"] for l_n in leaf_nodes]:
                                    leaf_nodes.remove({"name":str(t - 2) + "_" + str(arg_extra), "box": data["boxes"][t-2][arg_extra], "reid": data["reids"][t-2][arg_extra]})
                                    # leaf_nodes.remove(track_ids_prev2[arg_extra]) #remove from leaf nodes because edge is drawn from it now

                                new_id = 0

                if new_id == 1:
                    new_id_list.append(id)
                    new_id = 0


            for id in new_id_list:
                # print(t, id)

                new_name = -1


                if new_name == -1:
                    new_name = int(track_counter)
                    track_counter += 1
                    track_ids[id] = int(new_name)


        all_tracks.append(list(track_ids))

    tracklets=get_tracklets(all_tracks, data["reids"], data["boxes"])

    reids = [torch.stack([tr["reids"] for tr in tracks]) for tracks in tracklets]
    boxes = [torch.stack([tr["box"] for tr in tracks]) for tracks in tracklets]
    #avg_reids = torch.stack([torch.sum(rd, dim=0) for rd in reids])
    avg_reids_begin=torch.stack([torch.mean(rd[:5], dim=0) for rd in reids])
    avg_reids_end = torch.stack([torch.mean(rd[-5:], dim=0) for rd in reids])
    ts = [[tr["t"] for tr in tracks] for tracks in tracklets]
    track_ids_all = [[tr["track_id"] for tr in tracks] for tracks in tracklets]

    #dist_matrix = torch.cdist(avg_reids.unsqueeze(0),avg_reids.unsqueeze(1)).squeeze(2)# np.zeros((len(tracklets), len(tracklets)))
    reid_matrix = 1-torch.mm(avg_reids_end, avg_reids_begin.permute(1, 0))

    #dist_matrix = torch.cdist(avg_reids_end.unsqueeze(0),avg_reids_begin.unsqueeze(1)).squeeze(2)
    #cost_matrix = munkres.make_cost_matrix(dist_matrix)
    for i in range(len(reid_matrix)):
        #these are the beginnings

        reid_matrix[i,i]=100000 # preventing these values
        reid_matrix[i,np.where(ts[i][0]<=np.array([t[-1] for t in ts]))]=100000 # preventing these values
        reid_matrix[i, np.where(ts[i][0] > (np.array([t[-1] for t in ts])+keep_alive))] = 100000  # preventing these values

    #cost_matrix
    indexes_temp = opt.linear_sum_assignment(reid_matrix.cpu())
    indexes=[]
    for pairs in np.array(indexes_temp).transpose():
        if reid_matrix[pairs[0], pairs[1]]<0.9:
            if not mots:
                b1 = boxes[pairs[1]][-1]
                b2 = boxes[pairs[0]][0]
                condition1=abs(b1[2]/b1[3] -b2[2]/b2[3]) <0.2 #aspect_ratio

                box_diff=b2-b1
                tdiff = ts[pairs[0]][0] - ts[pairs[1]][-1]
                condition2=((box_diff[0]/tdiff < 40) and (box_diff[1]/tdiff < 40) and (box_diff[2]/tdiff < 40) and (box_diff[3]/tdiff < 40))
                condition= condition1 and condition2
            else:
                condition =True
            if condition:
                indexes.append(pairs)
                print(track_ids_all[pairs[1]][0],ts[pairs[1]][-1], track_ids_all[pairs[0]][0],ts[pairs[0]][0], reid_matrix[pairs[0], pairs[1]])

    groups=get_groups(np.array(indexes))
    print(groups)
    all_tracks_new=all_tracks.copy()
    for g in groups:
        dominant_track=tracklets[g[0]][0]["track_id"]
        all_tracks_new=[[tr if tr not in [tri["track_id"] for i in g[1:] for tri in tracklets[i]]
                         else dominant_track for tr in tracks_t] for tracks_t in all_tracks_new]

    return all_tracks_new, all_tracks

def get_groups(indexes):
    import networkx as nx
    groups=[]
    G=nx.Graph()
    for pair in indexes:
        G.add_edge(pair[0],pair[1])

    groups=list(nx.connected_components(G))

    return [sorted(list(g)) for g in groups]

def get_tracklets(all_tracks, reids, boxes):

    unique_tracks=np.unique(np.array([altr for alltr in all_tracks for altr in alltr]))
    tracklets=[[] for unq in unique_tracks]

    for t, (tracks_t, reids_t, boxes_t) in enumerate(zip(all_tracks, reids, boxes)):
        for i, (tracks_i, reids_i, box_i) in enumerate(zip(tracks_t, reids_t, boxes_t)):
            tracklets[tracks_i-1].append({"t":t, "track_id":tracks_i, "reids": reids_i, "box":box_i})
    return tracklets

def get_y_temp_gt(labels_1, labels_2, assignments, device=0):
    #find the gt assignment matrix

    lb1=np.array([i["track_id"].tolist()[0] for i in labels_1]).reshape(-1,1)
    lb2 = np.array([i["track_id"].tolist()[0] for i in labels_2]).reshape(1,-1)
    for i, a in enumerate(assignments):
        if len(torch.where(torch.tensor((lb1 == lb2) * 1).cuda(device) != a)[0])==0:
            return i, []
    return -1, torch.tensor((lb1 == lb2) * 1).cuda(device)


#munkres_obj = munkres.Munkres()
def hung_2nd_order(assignment_matrix_2nd_order, cost_matrix_2nd_order):
    cm_temp = []
    idx=[]
    keep = torch.where(assignment_matrix_2nd_order.sum(0) == 0)[0]
    for i in range(0,len(assignment_matrix_2nd_order)):

        row=assignment_matrix_2nd_order[i]

        if row.sum()==0:
            cm_temp.append(cost_matrix_2nd_order[i][keep])
            idx.append(i)

    mx = max(torch.stack(cm_temp).shape[0], torch.stack(cm_temp).shape[1])
    cm_hung = np.zeros((mx, mx)) + 100
    cm_hung[:torch.stack(cm_temp).shape[0], :torch.stack(cm_temp).shape[1]] = torch.stack(cm_temp).detach().cpu().numpy()
    asgn = munkres_obj.compute(cm_hung)
    indexes=[]
    for pair in asgn:
        if pair[0] in list(range(torch.stack(cm_temp).shape[0])) and pair[1] in list(range(torch.stack(cm_temp).shape[1])):
            indexes.append(pair)


    assignment_temp = torch.zeros(torch.stack(cm_temp).shape).cuda(cost_matrix_2nd_order.device)
    assignment_temp[np.stack(indexes).transpose()] = 1

    for id in range(len(idx)):
        assignment_matrix_2nd_order[idx[id],keep]=assignment_temp[id]


    # for pos in range(len(idx)):
    #     cm_pair=idx[pos], keep[pos]
    #     hung_pair=indexes[pos]
    #     assignment_matrix_2nd_order[cm_pair[0]+hung_pair[0],cm_pair[1]+hung_pair[1]]=1


    aux_cost=(1-assignment_matrix_2nd_order.sum(0)).sum()*100

    return assignment_matrix_2nd_order, aux_cost

def find_gt_cost_with_gt_detections(detections_gt, lmbda, size):
    L=0
    L_grad_lmbda=torch.zeros(lmbda.shape)
    y_GT=[]
    for i in range(0,len(detections_gt)-1):
        det=detections_gt[i]
        det=detections_gt[i]
        det_next=detections_gt[i+1]
        cost_matrix=torch.zeros((len(det), len(det_next)))
        cost_grad_lmbda_matrix = torch.zeros((len(det), len(det_next),2))
        for i1, dt in enumerate(det):
            for i2, dt_next in enumerate(det_next):
                cost_matrix[i1,i2], cost_grad_lmbda_matrix[i1,i2,:], _ = cf.find_cost_simple(dt, dt_next, lmbda, size=size)

        k = min(15,math.factorial(min(cost_matrix.shape[0], cost_matrix.shape[1])))

        phi_1, assignment = k_best_costs(k, cost_matrix.detach().numpy())
        y_GT.append(np.where(np.array([sum(abs(i[:, 0] - i[:, 1])) == 0 for i in assignment]))[0][0])
        L = L + phi_1[y_GT[i]]
        L_grad_lmbda = L_grad_lmbda + torch.sum(torch.stack([cost_grad_lmbda_matrix[j[0],j[1]] for j in assignment[y_GT[i]]]), axis=0)
    return L, L_grad_lmbda, y_GT


def get_optical_flow(args, seq, names):

    if args.mots:
        dirflow1 = args.optical_flow_path + "/flow_skip0/dir1/"
    else:
        dirflow1 =  "/"+args.optical_flow_method+"/flow_skip0/"

    #############load optical_flow

    optical_flow_skip0 = []

    print("Appending Optflow Images...")

    for im,name in enumerate(names[1:]):
        if args.mots:
            fl_1 = -np.load(dirflow1 + seq + "/" + name[0][:-4] + ".npy")
        else:
            fl_1 = -np.load(args.optical_flow_path + seq.replace("SDP", "FRCNN").replace("DPM", "FRCNN") + dirflow1 + name[0][:-4] + ".npy")
        optical_flow_skip0.append(fl_1)  # @(fl_1 - fl_2) / 2)

    optical_flow_skip1 = []
    if args.mots:
        dirflow1 = args.optical_flow_path + "/flow_skip1/dir1/"
    else:
        dirflow1 = "/"+args.optical_flow_method+"/flow_skip1/"

    print("Appending Optflow 2 Images...")
    for im, name in enumerate(names[2:]):
        if args.mots:
            fl_1 = -np.load(dirflow1 + seq + "/" + name[0][:-4] + ".npy")
        else:
            fl_1 = -np.load(args.optical_flow_path + seq.replace("SDP", "FRCNN").replace("DPM", "FRCNN") + dirflow1 + name[0][:-4] + ".npy")

        optical_flow_skip1.append(fl_1)

    return optical_flow_skip0, optical_flow_skip1



def get_optical_flow_locations(args, seq, names):

    if args.mots:
        dirflow1 = args.optical_flow_path + "/flow_skip0/dir1/"
    else:
        dirflow1 =  "/"+args.optical_flow_method+"/flow_skip0/"

    #############load optical_flow

    optical_flow_skip0 = []

    print("Appending Optflow Images...")

    for im,name in enumerate(names[1:]):
        if args.mots:
            fl_1 = dirflow1 + seq + "/" + name[0][:-4] + ".npy"
        else:

            fl_1 = args.optical_flow_path + seq.replace("SDP", "FRCNN").replace("DPM", "FRCNN") + dirflow1 + name[0][:-4] + ".npy"
        optical_flow_skip0.append(fl_1)  # @(fl_1 - fl_2) / 2)

    optical_flow_skip1 = []
    if args.mots:
        dirflow1 = args.optical_flow_path + "/flow_skip1/dir1/"
    else:
        dirflow1 = "/"+args.optical_flow_method+"/flow_skip1/"

    print("Appending Optflow 2 Images...")
    for im, name in enumerate(names[2:]):
        if args.mots:
            fl_1 = dirflow1 + seq + "/" + name[0][:-4] + ".npy"
        else:
            fl_1 =args.optical_flow_path + seq.replace("SDP", "FRCNN").replace("DPM", "FRCNN") + dirflow1 + name[0][:-4] + ".npy"

        optical_flow_skip1.append(fl_1)

    return optical_flow_skip0, optical_flow_skip1
