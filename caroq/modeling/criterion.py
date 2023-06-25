"""
Loss Criterion.
"""
import logging
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from random import choice
import pdb
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
import time

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, time_gap=5, ignore_value=10):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.time_gap=time_gap
        self.ignore_value=ignore_value

    def loss_labels(self, outputs, targets, indices, num_masks, keep_indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs

        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t_["labels"][J] for t_, (_, J) in zip(targets, indices)])
        target_classes_o[target_classes_o==self.num_classes]=-100 # for mots/vps ambiguous category

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        src_logits=src_logits.reshape(src_logits.shape[0]*src_logits.shape[1],-1)
        target_classes=target_classes.reshape(target_classes.shape[0]*target_classes.shape[1])

        loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight, ignore_index=-100)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, keep_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        bs_t, d1, d2, d3=src_masks.shape
        bs=int(bs_t/self.time_gap)
        src_masks=src_masks.reshape(bs, self.time_gap, d1, d2, d3).permute(0,2,1,3,4)

        src_masks = src_masks[src_idx]




        masks = [k for t_ in targets for k in t_["masks"]]#[t_["masks"][k] for t_ in targets for k in range(len(t_["masks"])) if t_["labels"][k]!=ignore_label] #
        #keep=[k.item() nt in (self.ignore_value, -100) for t_ in targets for k in t_["labels"]]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)

        tmp=[0]#+[len([k for k in t["labels"]]) for t in targets]
        for t in targets:
            tmp.append(len([k for k in t["labels"]])+tmp[-1])

        lngth=torch.tensor([tmp[i] for i,t in enumerate(targets) for r in t["labels"]])
        target_masks = target_masks[tgt_idx[1]+lngth]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W



        #src_masks = src_masks[keep_indices[src_idx], None]
        src_masks = src_masks[:, None]
        src_masks=src_masks.flatten(2,3)
        #target_masks = target_masks[keep_indices[src_idx], None]
        target_masks = target_masks[:, None]
        target_masks=target_masks.flatten(2,3)

        # keep_nonzero_target_masks=target_masks.sum(-1).sum(-1).sum(-1)!=0
        # target_masks=target_masks[keep_nonzero_target_masks]
        # src_masks=src_masks[keep_nonzero_target_masks]




        #l_mask=0.
        #l_dice=0.

        #for id in range(src_masks.shape[2]):


        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),# src_masks[:,:,id].float(),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
        point_labels = point_sample(target_masks,point_coords,align_corners=False,).squeeze(1)#target_masks[:,:,id].float()

        point_logits = point_sample(src_masks.float(),point_coords,align_corners=False,).squeeze(1)#src_masks[:,:,id].float()

        #l_mask+=sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)

        #l_dice+=dice_loss_jit(point_logits, point_labels, num_masks)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):

        keep_indices=[]#self.remove_ignore_indices(outputs, targets)
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, keep_indices)


    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        #outputs_without_aux1, outputs_without_aux2=self.modify_outputs(outputs_without_aux)



        # Retrieve the matching between the outputs of the last layer and the targets


        targets_new=self.split_targets(targets)

        indices=self.new_matcher2(outputs_without_aux, targets, targets_new)
        #indices=self.matcher(outputs_without_aux, targets)

        #indices = self.new_matcher(outputs_without_aux1, outputs_without_aux2, targets_new) #repeats indices from self.matcher

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t_["labels"]) for t_ in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs["pred_masks"].device #device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        #print("others: ", time.time()-t1)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets_new.copy(), indices.copy(), num_masks))


        # scores1, labels1 = F.softmax(outputs["pred_logits"], dim=-1).max(-1)
        # keep1 = labels1.ne(2) & (scores1 > 0.8)
        #print(labels1[keep1])



        # tmp_masks=outputs["pred_masks"][torch.stack([keep1, keep1]).permute(1,0,2).flatten(0,1)]
        # if tmp_masks.shape[0]>0:
        #     save_image(tmp_masks.unsqueeze(1), fp= "match/src_"+str(time.time())+".png", padding=1, pad_value=1)



        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):

                #aux_outputs1, aux_outputs2=self.modify_outputs(aux_outputs)
                indices=self.new_matcher2(aux_outputs, targets, targets_new)
                #indices=self.matcher(aux_outputs, targets)

                #indices = self.new_matcher(aux_outputs1, aux_outputs2, targets_new)

                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets_new.copy(), indices.copy(), num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
            #print("aux: ", time.time()-t1)

        return losses

    def new_matcher2(self, outputs, targets, targets_new):

        ind1 = self.matcher(outputs, targets, targets_new)

        indices=[]
        for i1 in ind1:

            indices.append(i1)
            indices.append(i1)

        return indices

    def new_matcher(self, outputs1, outputs2, targets):
        ind1 = self.matcher(outputs1, targets[::2])
        ind2 = self.matcher(outputs2, targets[1::2])
        #in_1= [list(set(i1[1].tolist())-set(i2[1].tolist())) for (i1,i2) in zip(ind1, ind2)]

        indices=[]
        for (i1,i2) in zip(ind1, ind2):

            indices.append(i1)
            indices.append(i1)
            # in_2= torch.tensor(list(set(i2[1].tolist())-set(i1[1].tolist())))
            #
            # if len(in_2)>0:
            #     extra_in_2=(i2[0][torch.where(i2[1]==in_2)[0]], in_2)
            #     indices.append((torch.cat([i1[0], extra_in_2[0]]), torch.cat([i1[1], extra_in_2[1]])))
            # else:
            #     indices.append(i1)
            # print(i1)
            # print(i2)
            # print((torch.cat([i1[0], extra_in_2[0]]), torch.cat([i1[1], extra_in_2[1]])))

        return indices

    def modify_outputs(self, outputs):

        bs_t, c, h,w= outputs["pred_masks"].shape
        outputs1={}
        outputs2={}
        outputs1["pred_masks"]=outputs["pred_masks"].reshape(outputs["pred_logits"].shape[0], -1, c, h, w)[::2].flatten(0,1)
        outputs2["pred_masks"]=outputs["pred_masks"].reshape(outputs["pred_logits"].shape[0], -1, c, h, w)[1::2].flatten(0,1)

        outputs1["pred_logits"]=outputs["pred_logits"][::2]
        outputs2["pred_logits"]=outputs["pred_logits"][1::2]


        return outputs1, outputs2


    def split_targets(self,targets):
        targets_new=[]

        for target in targets:
            masks=torch.split(target["masks"], self.time_gap, dim=1)
            for msks in masks:
                lbls=target["labels"].clone()
                lbls[msks.sum(-1).sum(-1).sum(-1)==0]=-100#self.num_classes#
                targets_new.append({"labels": lbls, "masks": msks})


        return targets_new

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def remove_ignore_indices(self, outputs, targets):

        bs_t, d1, d2, d3=outputs["pred_masks"].shape
        bs=int(bs_t/self.time_gap)
        ops=[op.permute(1,0,2,3) for op in outputs["pred_masks"].reshape(bs,self.time_gap,d1,d2,d3)]

        ignore_label_indices=[[k.item() in (self.ignore_value, -100) for k in t_["labels"]] for t_ in targets]

        ignore_masks=[F.interpolate(t_["masks"][ignore_label], [d2, d3]) if True in ignore_label else torch.zeros(1,self.time_gap,d2,d3).to(outputs["pred_masks"]) for t_, ignore_label in zip(targets,ignore_label_indices)]


        #normalized_product=[(ignore_masks[ct]*(ops[ct]>0.5).float()).sum(1).sum(1).sum(1)/((ops[ct]>0.5).float().sum(1).sum(1).sum(1)+0.000000001) for ct in range(bs)]

        #keep_indices=[pr<0.5 for pr in normalized_product] # keep indices with low match with ignore regions

        keep_indices=[]
        for ct in range(len(targets)):
            n_prod=[(igm*(ops[ct]>0.5).float()).sum(1).sum(1).sum(1)/((ops[ct]>0.5).float().sum(1).sum(1).sum(1)+0.000000001) for igm in ignore_masks[ct]]
            keep_indices.append((torch.stack(n_prod)<0.5).sum(0).bool())



        return torch.stack(keep_indices)
