from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import munkres
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
import numpy as np
import cv2
from detectron2.data import MetadataCatalog
from torchvision.utils import save_image, draw_segmentation_masks
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
import pycocotools.mask as cocomask

from caroq.utils import file_helper, mots_helper
import pdb
import time
from caroq_video.data_video.datasets.cityscapes_vps import CITYSCAPES_VPS_CATEGORIES
import json
from collections import namedtuple

TrackElement = namedtuple(
    "TrackElement", ["t", "box", "track_id", "class_", "mask", "score"]
)
import os
from panopticapi.utils import id2rgb, rgb2id
from detectron2.utils.visualizer import Visualizer
from PIL import Image

torch.manual_seed(1)
np.random.seed(1)

munkres_obj = munkres.Munkres()


class Meta:
    def __init__(self, thing_classes, stuff_classes):
        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        time_gap: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        outfolder,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.time_gap = time_gap
        self.num_frames = (
            time_gap * 2
        )  # for the dataset it's double for the tracking part

        self.outfolder = outfolder

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,  # /(cfg.MODEL.MASK_FORMER.TIME_GAP*2),
            cost_dice=dice_weight,  # /(cfg.MODEL.MASK_FORMER.TIME_GAP*2),
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS
            * cfg.MODEL.MASK_FORMER.TIME_GAP
            * 2,
            time_gap=cfg.MODEL.MASK_FORMER.TIME_GAP,
            matcher_config=cfg.MODEL.MASK_FORMER.MATCHER_CONFIG,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS
            * cfg.MODEL.MASK_FORMER.TIME_GAP,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            time_gap=cfg.MODEL.MASK_FORMER.TIME_GAP,
            ignore_value=cfg.INPUT.IGNORE_VALUE,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "time_gap": cfg.MODEL.MASK_FORMER.TIME_GAP,
            "outfolder": cfg.OUTPUT_DIR + "/" + cfg.TEST.OUTPUT_DIR + "/Instances_txt/",
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, vid=0):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            images = [x["image"].to(self.device).squeeze(0) for x in batched_inputs]

            images = [
                (x - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)
                for x in images
            ]
            images = ImageList.from_tensors(images, self.size_divisibility)

            c, h, w = images.tensor.shape[-3:]

            features = self.backbone(images.tensor.reshape(-1, c, h, w))
            outputs, _ = self.sem_seg_head(
                features, training=True
            )  # training= true for track decoder (pairs during training)
            targets = self.prepare_targets(batched_inputs, images)

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:

            images = [
                x["image"].squeeze(0) for x in batched_inputs
            ]  # sequence can be huge when evaluating. Don't put on cuda

            # only supports inference on batch size 1

            images = [
                (x - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)
                for x in images
            ]
            images = ImageList.from_tensors(images, self.size_divisibility)

            c, h, w = images.tensor.shape[-3:]

            # batch size is 1
            time_gap = self.time_gap

            all_tracks = []
            vid_len = images.tensor.shape[1]
            ct = 0
            if (
                self.metadata.name.find("vis") > -1
                or self.metadata.name.find("vps") > -1
            ):
                mask_cls_results_all = []
                mask_pred_results_all = []

            if images.tensor.shape[1] % time_gap == 0:
                imgs = images.tensor.reshape(-1, time_gap, c, h, w)

            else:
                images_pad = (
                    images.tensor[:, -1]
                    .repeat((time_gap - images.tensor.shape[1] % time_gap), 1, 1, 1)
                    .unsqueeze(0)
                )
                imgs = torch.cat([images.tensor, images_pad], dim=1)
                imgs = imgs.reshape(-1, time_gap, c, h, w)

            previous_tracks = []
            max_id = 0

            previous_keep = []
            time = 0
            prev_output = []
            annotations_all = []

            if (
                self.metadata.name.find("vis") == -1
                and self.metadata.name.find("vps") == -1
            ):
                names = [k.split("/")[-1] for k in batched_inputs[0]["file_names"]]
                vid_name = batched_inputs[0]["file_names"][0].split("/")[-2]

            for img in imgs:

                current_height = img.shape[
                    2
                ]  # batched_inputs[0]["image"].shape[-2:][0]
                current_width = img.shape[3]  # batched_inputs[0]["image"].shape[-2:][1]

                features = self.backbone(img.to(self.device))

                outputs, prev_output = self.sem_seg_head(
                    features, training=False, prev_output=prev_output
                )

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

                del outputs

                if self.metadata.name.find("vis") > -1:
                    mask_cls_results_all.append(mask_cls_results[0].cpu())
                    mask_pred_results_all.append(
                        mask_pred_results.permute(1, 0, 2, 3).cpu()
                    )

                elif self.metadata.name.find("vps") > -1:

                    # upsample to current frame size
                    mask_pred_results = F.interpolate(
                        mask_pred_results,
                        size=(current_height, current_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                    # remove the padding
                    mask_pred_results = mask_pred_results[
                        :,
                        :,
                        : batched_inputs[0]["image"].shape[-2:][0],
                        : batched_inputs[0]["image"].shape[-2:][1],
                    ]

                    # upsample to original frame size
                    mask_pred_results = F.interpolate(
                        mask_pred_results,
                        size=(batched_inputs[0]["height"], batched_inputs[0]["width"]),
                        mode="bilinear",
                        align_corners=False,
                    )

                    for i, mask_pred_result in enumerate(mask_pred_results):

                        mask_cls_result = mask_cls_results[int(i / self.time_gap)]

                        if ct == vid_len:
                            break

                        save_name = batched_inputs[0]["file_names"][ct]
                        img_name = save_name[::-1][: save_name[::-1].find("/")][::-1]
                        save_file_name = img_name[
                            : img_name.find(
                                img_name[::-1][: img_name[::-1].find("_")][::-1]
                            )
                            - 1
                        ]

                        output_folder = self.outfolder.replace(
                            "Instances_txt", "panoptic_pred"
                        ).replace("./", "")
                        if not os.path.exists(output_folder):
                            os.mkdir(output_folder)
                        if not os.path.exists(output_folder + "/pan_pred"):
                            os.mkdir(output_folder + "/pan_pred")
                        annotations = {}
                        annotations["file_name"] = (
                            save_file_name + ".png"
                        )  # output_folder+"/pan_pred/"+
                        annotations["image_id"] = save_file_name

                        panoptic_seg, segments_info = self.panoptic_inference(
                            mask_cls_result,
                            mask_pred_result,
                            save_file=output_folder
                            + "/pan_pred/"
                            + save_file_name
                            + ".png",
                        )

                        annotations["segments_info"] = segments_info
                        annotations_all.append(annotations)
                        ct += 1

                else:

                    # upsample masks
                    mask_pred_results = F.interpolate(
                        mask_pred_results,
                        size=(current_height, current_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                    for i, mask_pred_result in enumerate(mask_pred_results):

                        mask_cls_result = mask_cls_results[int(i / self.time_gap)]
                        if self.sem_seg_postprocess_before_inference:
                            mask_pred_result = sem_seg_postprocess(
                                mask_pred_result,
                                (current_height, current_width),
                                batched_inputs[0]["height"],
                                batched_inputs[0]["width"],
                            )

                        # # remove the padding
                        # mask_pred_result = mask_pred_result[
                        #     :,
                        #     : batched_inputs[0]["height"],
                        #     : batched_inputs[0]["width"],
                        # ]

                        if i == self.time_gap:
                            previous_tracks = tracks

                        if ct == vid_len:
                            break

                        tracks, max_id, previous_keep = self.inference(
                            mask_cls_result,
                            mask_pred_result,
                            t=int(names[ct][:-4]),
                            previous_tracks=previous_tracks,
                            max_id=max_id,
                            previous_keep=previous_keep,
                            inference_type=batched_inputs[0]["dataset"],
                        )
                        all_tracks.append(tracks)

                        ct += 1

            if self.metadata.name.find("vis") > -1:
                unpadded_size = images.image_sizes[0]
                padded_size = (images.tensor.shape[-2], images.tensor.shape[-1])

                return self.inference_video(
                    torch.stack(mask_cls_results_all),
                    torch.stack(mask_pred_results_all),
                    unpadded_size,
                    padded_size,
                    batched_inputs[0]["height"],
                    batched_inputs[0]["width"],
                    batched_inputs[0]["length"],
                )

            elif self.metadata.name.find("mots") > -1:
                hyp_tracks = mots_helper.make_disjoint(all_tracks, "score")
                file_helper.export_tracking_result_in_kitti_format(
                    vid_name,
                    hyp_tracks,
                    True,
                    "",
                    out_folder=self.outfolder,
                )
                return 0
            elif self.metadata.name.find("vps") > -1:

                json.dump(
                    annotations_all,
                    open(
                        output_folder
                        + "/pred_"
                        + str(batched_inputs[0]["video_id"])
                        + ".json",
                        "w",
                    ),
                )
                return

    def prepare_targets(self, targets, images):

        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(
                mask_shape, dtype=torch.bool, device=self.device
            )

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[
                    :, f_i, :h, :w
                ] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append(
                {"labels": gt_classes_per_video, "ids": gt_ids_per_video}
            )
            gt_masks_per_video = gt_masks_per_video[
                valid_idx
            ].float()  # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

            return gt_instances

    # inference for different tasks

    def panoptic_inference(self, mask_cls, mask_pred, save_file="tmp.png"):

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        indices = torch.where(keep)[0]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):

                current_segment_id = indices[k].item() + 1

                pred_class = cur_classes[k].item()

                mapping = {}
                indx = 0
                for k_ in CITYSCAPES_VPS_CATEGORIES:
                    if k_["isthing"] == True:
                        mapping[indx] = k_["id"]
                        indx += 1

                isthing = pred_class in mapping.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "area": int(mask.sum()),
                        }
                    )

            cv2.imwrite(save_file, panoptic_seg.cpu().numpy().astype(np.uint8))

            return panoptic_seg, segments_info

    def inference_video(
        self,
        pred_cls,
        pred_masks,
        unpadded_size,
        padded_size,
        output_height,
        output_width,
        length,
    ):

        if len(pred_cls) > 0:
            scores_tmp = F.softmax(pred_cls, dim=-1)[:, :, :-1]
            nonzero_masks = (pred_masks > 0).sum(-1).sum(-1).sum(-1) > 0
            scores_tmp[nonzero_masks == 0] = 0.0
            normalizing_factor = (nonzero_masks).sum(
                0
            )  # to scale the scores; if all T*H*W masks are non-zero, this is total no of num_frames
            normalizing_factor[normalizing_factor == 0] = pred_masks.shape[0]
            scores = scores_tmp.sum(0) / normalizing_factor.unsqueeze(1)

            pred_masks = pred_masks.permute(1, 0, 2, 3, 4).flatten(1, 2)
            # print(length, pred_masks.shape[1])
            pred_masks = F.interpolate(
                pred_masks, size=padded_size, mode="bilinear", align_corners=False,
            )
            labels = (
                torch.arange(self.sem_seg_head.num_classes, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_queries, 1)
                .flatten(0, 1)
            )

            # # mine: keep predictions with confidence greater than self.object_mask_threshold
            # scores_per_image = scores.flatten(0,1)
            # topk_indices=torch.where(scores_per_image>self.object_mask_threshold)[0]
            # labels_per_image = labels[topk_indices]
            # scores_per_image=scores_per_image[topk_indices]
            # topk_indices = topk_indices // self.sem_seg_head.num_classes
            # pred_masks = pred_masks[topk_indices]

            # original: keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :length, : unpadded_size[0], : unpadded_size[1]]
            pred_masks = F.interpolate(
                pred_masks,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )

            masks = pred_masks > 0.0

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    def ytvis_inference(self, mask_pred_results, mask_cls_results, objs, time, vid_len):
        if time[0] < vid_len:

            scores_ = F.softmax(mask_cls_results, dim=-1)[:, :-1]
            scores, labels = scores_.max(-1)
            mask_pred_results = mask_pred_results  # .sigmoid()

            keep = scores > 0.01  # self.object_mask_threshold)

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_results[keep]
            cur_mask_cls = mask_cls_results[keep]

            for k_itr, k in enumerate(torch.where(keep)[0]):
                if time[1] < vid_len:
                    objs[k]["mask"][time[0] : time[1]] = (cur_masks[k_itr] > 0).float()
                else:
                    shape0 = objs[k]["mask"][time[0] : time[1]].shape[0]
                    objs[k]["mask"][time[0] : time[1]] = (
                        cur_masks[k_itr][:shape0] > 0
                    ).float()
                objs[k]["score"].append(cur_scores[k_itr])
                objs[k]["class"].append(cur_classes[k_itr])
        return objs

    def inference_video_2(self, objs, height, width):

        unique_objs = [obj for obj in objs if obj["score"] != []]

        scores = [(sum(obj["score"]) / len(obj["score"])).item() for obj in unique_objs]
        classes = [torch.tensor(obj["class"]).mode()[0].item() for obj in unique_objs]
        masks = [obj["mask"] for obj in unique_objs]

        if len(scores) > 10:
            scores, indices = torch.tensor(scores).topk(10)

            scores = scores.tolist()
            classes = torch.tensor(classes)[indices].tolist()
            masks = [m for idx, m in enumerate(masks) if idx in indices]

        # print(scores)

        video_output = {
            "image_size": (height, width),
            "pred_scores": scores,
            "pred_labels": classes,
            "pred_masks": masks,
        }

        return video_output

    def inference(
        self,
        mask_cls,
        mask_pred,
        t=0,
        previous_tracks=[],
        max_id=0,
        previous_keep=[],
        inference_type="mots",
    ):
        # need to change
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        new_keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        if previous_tracks == []:
            previous_keep = new_keep.clone()

        keep = torch.logical_or(new_keep, previous_keep)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        ids = torch.where(keep)[0].tolist()

        h, w = cur_masks.shape[-2:]
        segments_info = []

        current_segment_id = 0

        # for im in range(len(cur_masks)):
        #     cv2.imwrite("masks/mask"+str(im)+"_label"+str(cur_classes[im].item())+"_score"+str(cur_scores[im].item())+".png",cur_masks[im].cpu().numpy()*255)

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            max_id = max(ids + [max_id])
            return segments_info, max_id, keep
        else:
            # take argmax
            # cur_mask_ids = cur_prob_masks.argmax(0)
            # stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                # isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask = cur_masks[k].cpu().numpy()
                area = (cur_masks[k] >= 0.5).sum().item()

                if area > 0:  # and cur_scores[k]>self.object_mask_threshold:

                    if inference_type == "mots":

                        encoded_mask = cocomask.encode(
                            np.asfortranarray((mask > 0.5).astype(np.uint8))
                        )
                        if int(pred_class) == 0:
                            cls_id = 1  # car
                        elif int(pred_class) == 1:
                            cls_id = 2  # person
                        else:
                            cls_id = 10
                        if cls_id in (1, 2):
                            segments_info.append(
                                TrackElement(
                                    t=t,
                                    box=cocomask.toBbox(encoded_mask),
                                    track_id=ids[current_segment_id],
                                    mask=encoded_mask,
                                    class_=cls_id,
                                    score=cur_scores[k].item(),
                                )
                            )

                    elif inference_type in ("sailvos", "ytvis"):

                        encoded_mask = cocomask.encode(
                            np.asfortranarray((mask > 0.5).astype(np.uint8))
                        )
                        segments_info.append(
                            TrackElement(
                                t=t,
                                box=cocomask.toBbox(encoded_mask),
                                track_id=ids[current_segment_id],
                                mask=encoded_mask,
                                class_=pred_class,
                                score=cur_scores[k].item(),
                            )
                        )

                current_segment_id += 1
        max_id = max(ids + [max_id])

        return segments_info, max_id, keep
