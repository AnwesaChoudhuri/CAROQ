
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import pycocotools.mask as cocomask
import numpy as np

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
import pdb
import cv2
from mask_former import file_helper, mots_helper

TrackElement = namedtuple("TrackElement", ["t", "box", "track_id", "class_", "mask", "score"])

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
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
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
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.panoptic_on = panoptic_on
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
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
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, vid=0, outfolder="new/Instances_txt/"):
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


        images = [x["image"].to(self.device).squeeze(0) for x in batched_inputs]

        abc = [x["name"] for x in batched_inputs]
        names= [a for b in abc for a in b]

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)


        c, h, w =images.tensor.shape[-3:]
        time_gap=5

        if self.training:
            features = self.backbone(images.tensor.reshape(-1, c, h,w))
            outputs = self.sem_seg_head(features)
            torch.cuda.empty_cache()
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"] for x in batched_inputs]
                #print("prepare target")
                targets = self.prepare_targets(gt_instances, images)

                 #targets = [x["instances"] for x in batched_inputs]
            else:
                targets = None

            # bipartite matching-based

            #print("loss calculate")
            losses = self.criterion(outputs, targets, len(batched_inputs))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses

        else:
            all_tracks=[]
            ct=0
            imgs=images.tensor.reshape(-1, time_gap, c, h,w)
            for img in imgs:
                features = self.backbone(img)
                outputs = self.sem_seg_head(features)
                #features, outputs= self.get_split_inference(imgs)

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )


                #processed_results = []
                image_size=batched_inputs[0]["image"].shape[-2:]

                height = image_size[0]
                width = image_size[1]


                for i, mask_pred_result in enumerate(mask_pred_results):

                    mask_cls_result=mask_cls_results[int(i/5)]

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = sem_seg_postprocess(mask_pred_result, image_size, height, width)

                    tracks = self.mots_inference(mask_cls_result, mask_pred_result, t=int(names[ct][:-4]))
                    all_tracks.append(tracks)
                    ct+=1



            hyp_tracks = mots_helper.make_disjoint(all_tracks, "score")
            file_helper.export_tracking_result_in_kitti_format("0002", hyp_tracks, True, "",
                                                   out_folder=outfolder)

            return all_tracks


    def get_split_inference(self, imgs):

        outputs={}
        features={}
        for it, im in enumerate(imgs):
            tmp_features = self.backbone(im)
            for k in tmp_features.keys():
                if it==0:
                    features[k]=tmp_features[k]
                else:
                    features[k]=torch.cat([features[k], tmp_features[k]])

            tmp_outputs = self.sem_seg_head(tmp_features)
            for k in tmp_outputs.keys():
                if k!="aux_outputs":
                    if it==0:
                        outputs[k]=tmp_outputs[k]
                    else:
                        outputs[k]=torch.cat([outputs[k], tmp_outputs[k]])

        return features, outputs

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image_chunk in targets:
            target_ids=list(np.unique([tg["track_id"] for tgt in targets_per_image_chunk for tg in tgt]))
            gt_masks_per_chunk=[]
            lbls_per_chunk=[]
            for t_id in target_ids:
                gt_masks_unique=[]
                for tgt in targets_per_image_chunk:
                    gtm=torch.tensor([tg["segmentation"] for tg in tgt if tg["track_id"]==t_id]).squeeze(0).to(images.device)
                    if len(gtm)==0:
                        gtm=torch.zeros(h, w).to(images.device)
                    else:
                        lbls=torch.tensor([tg["category_id"] for tg in tgt if tg["track_id"]==t_id])
                    gt_masks_unique.append(gtm)
                gt_masks_per_chunk.append(torch.stack(gt_masks_unique))
                lbls_per_chunk.append(lbls)



            gt_masks=torch.stack(gt_masks_per_chunk)
            labels=torch.stack(lbls_per_chunk).squeeze(-1)
            # pad gt
            #gt_masks = targets_per_image.gt_masks

            # gt_masks = torch.stack([t["segmentation"].squeeze(0) for t in targets_per_image])
            # padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=images.device)
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": labels.to(images.device),
                    "masks": gt_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def mots_inference(self, mask_cls, mask_pred, t=0):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        #panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        for im in range(len(cur_masks)):
            cv2.imwrite("masks/mask"+str(im)+"_label"+str(cur_classes[im].item())+"_score"+str(cur_scores[im].item())+".png",cur_masks[im].cpu().numpy()*255)


        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return segments_info
        else:
            # take argmax
            #cur_mask_ids = cur_prob_masks.argmax(0)
            #stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                #isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask = cur_masks[k].cpu().numpy()
                area = (cur_masks[k] >= 0.5).sum().item()
                current_segment_id += 1


                if area > 0 :

                    encoded_mask=cocomask.encode(np.asfortranarray((mask > 0.5).astype(np.uint8)))
                    if int(pred_class)==0:
                        cls_id=1 #car
                    elif int(pred_class)==1:
                        cls_id=2 #person
                    else:
                        cls_id=10
                    if cls_id in (1,2):
                        segments_info.append(
                            TrackElement(
                                t=t,
                                box=cocomask.toBbox(encoded_mask),
                                track_id= current_segment_id,
                                mask=encoded_mask,
                                class_= cls_id,
                                score=cur_scores[k].item()
                            )
                        )

        return segments_info
