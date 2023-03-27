# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


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


class SetCriterionSequence(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, reid_frame_loss_type, 
                 reid_seq_loss_type, reid_seq_tangled_loss, detach_frame_from_seq,
                 reid_seq_loss_over, reid_softmax_temperature):
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
        self.reid_frame_loss_type = reid_frame_loss_type
        self.reid_seq_loss_type = reid_seq_loss_type
        self.reid_seq_tangled_loss = reid_seq_tangled_loss
        self.detach_frame_from_seq = detach_frame_from_seq
        self.reid_seq_loss_over = reid_seq_loss_over
        self.reid_softmax_temperature = reid_softmax_temperature

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits_seq"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["matched_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce_seq": loss_ce}
        return losses
    
    def loss_reid_seq(self, outputs, targets, indices, num_masks):
        matched_reid = outputs['matched_reid'] # BxTxGxC
        if self.detach_frame_from_seq:
            matched_reid = [x.detach() for x in matched_reid]
        pred_center = outputs['pred_reid_seq'] # BxGxC
        matched_reid_valid = outputs['matched_reid_valid'] # BxTxG
        loss_reid_seq = 0
        count = 0
        batch_size = len(matched_reid)
        for b in range(batch_size):
            sQ_indices, fQ_indices = indices[b]
            # similarity = torch.einsum("tgc,qc->tgq", 
            #                           matched_reid[b][:, group_indices],
            #                           pred_center[b][query_indices])    
            seq_logits = torch.max(outputs['pred_logits_seq'][b].softmax(dim=-1)[:, :-1], dim=1)[0] # sQ
            if self.reid_seq_loss_type == "sigmoid":
                similarity = (matched_reid[b][:, fQ_indices]@pred_center[b][sQ_indices].t()).permute(0,2,1) # TxfQxC @ TxCxsQ -> TxfQxsQ -> TxsQxfQ
                if self.reid_seq_tangled_loss:
                    similarity = similarity*seq_logits[sQ_indices].unsqueeze(0).unsqueeze(2) # TxsQxfQ
                similarity_valid = matched_reid_valid[b][:, fQ_indices].unsqueeze(1) # Tx1xfQ
                similarity_label = torch.zeros_like(similarity) # TxsQxfQ
                for i in range(similarity.shape[1]):
                    similarity_label[:, i, i] = 1
                loss = F.binary_cross_entropy_with_logits(similarity, similarity_label, reduction="none")
                per_query_loss = (loss*similarity_valid).sum(dim=0)/(similarity_valid.sum(dim=0)+1e-6) # sQxfQ
                loss_reid_seq += per_query_loss.sum()/per_query_loss.shape[0]
            elif self.reid_seq_loss_type == 'softmax' or self.reid_seq_loss_type == 'symmetric_softmax':
                # TODO: make matched_reid into all reid
                similarity = (matched_reid[b]@pred_center[b].t()).permute(0,2,1) # TxfQxC @ CxsQ -> TxfQxsQ -> TxsQxfQ
                similarity_label = torch.zeros_like(similarity[0]) # sQxfQ
                similarity_label[sQ_indices, fQ_indices] = 1
                similarity_valid = matched_reid_valid[b] # TxfQ
                over_frame, _, over_seq, _ = self.reid_seq_loss_over.split('_')
                if over_seq == 'matched':
                    similarity = similarity[:, sQ_indices, :] # TxsQxfQ
                    similarity_label = similarity_label[sQ_indices, :] # sQxfQ
                    # similarity_valid = similarity_valid[:, sQ_indices, :] # Tx1xsQ
                if over_frame == 'matched':
                    similarity = similarity[:, :, fQ_indices] # TxsQxfQ
                    similarity_label = similarity_label[:, fQ_indices] # sQxfQ
                    similarity_valid = similarity_valid[:, fQ_indices] # TxfQ
                max_value_0, fQ_label = similarity_label.max(dim=0)
                max_value_1, sQ_label = similarity_label.max(dim=1)
                fQ_label[max_value_0==0] = -1
                sQ_label[max_value_1==0] = -1
                fQ_label = fQ_label.to(similarity.device).unsqueeze(0).repeat(similarity.shape[0], 1) # TxfQ
                sQ_label = sQ_label.to(similarity.device).unsqueeze(0).repeat(similarity.shape[0], 1) # TxsQ
                # similarity = (matched_reid[b][:, fQ_indices]@pred_center[b].t()).permute(0,2,1) # T,fQ,C @ T,C,sQ -> T,fQ,sQ -> T,sQ',fQ
                if self.reid_seq_tangled_loss:
                    similarity = similarity*seq_logits.unsqueeze(0).unsqueeze(2)
                # similarity_label = sQ_indices.unsqueeze(0).unsqueeze(0).repeat(similarity.shape[0],1,1) # Tx1xfQ range in [0, sQ)
                loss = F.cross_entropy(similarity/self.reid_softmax_temperature, fQ_label, reduction="none", ignore_index=-1) # TxfQ
                if self.reid_seq_loss_type == 'symmetric_softmax':
                    loss += F.cross_entropy(similarity.permute([0,2,1])/self.reid_softmax_temperature, sQ_label, reduction="none", ignore_index=-1) # TxsQ
                per_query_loss = (loss*similarity_valid).sum(dim=0)
                loss_reid_seq += per_query_loss.mean()
            else:
                raise KeyError("Unknown reid_seq_loss_type: {}".format(self.reid_seq_loss_type))
            count += 1
        loss_reid_seq /= count
        return {'loss_reid_seq': loss_reid_seq}
    
    def loss_reid_frame(self, outputs, targets, indices, num_masks):
        matched_reid = outputs['matched_reid'] # BxTxGxC
        matched_reid_valid = outputs['matched_reid_valid'] # BxTxG
        loss_reid_frame = 0
        count = 0 
        batch_size = len(matched_reid)
        for b in range(batch_size):
            # sQ_indices, fQ_indices = indices[b]
            for f in range(matched_reid[b].shape[0]-1):
                # select the queries that matched_reid_valid is 1
                prev_valid = matched_reid_valid[b][f]
                cur_valid = matched_reid_valid[b][f+1]
                similarity = torch.einsum("gc,Gc->gG", 
                                          matched_reid[b][f, prev_valid==1],
                                          matched_reid[b][f, cur_valid==1])
                if self.reid_frame_loss_type == 'sigmoid':
                    similarity_label = torch.zeros([prev_valid.shape[0], cur_valid.shape[0]], dtype=torch.float, device=similarity.device)
                    diag_value = matched_reid_valid[b][f]*matched_reid_valid[b][f+1] # G
                    similarity_valid = torch.ones([prev_valid.shape[0], cur_valid.shape[0]], dtype=torch.float, device=similarity.device)
                    for i, v in enumerate(diag_value):
                        similarity_label[i, i] = v
                        similarity_valid[i, i] = v
                    similarity_label = similarity_label[prev_valid==1, :][:, cur_valid==1]
                    similarity_valid = similarity_valid[prev_valid==1, :][:, cur_valid==1]
                    loss = F.binary_cross_entropy_with_logits(similarity, similarity_label, reduction="none")
                    cur_loss = (loss*similarity_valid).mean()
                elif self.reid_frame_loss_type == 'softmax':
                    label_forward = torch.cumsum(matched_reid_valid[b][f+1], dim=0)[prev_valid==1].to(torch.long)-1
                    label_backward = torch.cumsum(matched_reid_valid[b][f], dim=0)[cur_valid==1].to(torch.long)-1
                    # set invalid in label_backward to be -1
                    label_forward_valid = (cur_valid*prev_valid)[prev_valid==1]
                    label_backward_valid = (cur_valid*prev_valid)[cur_valid==1]
                    label_forward[label_forward_valid==0] = -1
                    label_backward[label_backward_valid==0] = -1
                    loss_forward = F.cross_entropy(similarity, label_forward, reduction="mean", ignore_index=-1)
                    loss_backward = F.cross_entropy(similarity.t(), label_backward, reduction="mean", ignore_index=-1)
                    cur_loss = loss_forward+loss_backward
                else:
                    raise KeyError
                loss_reid_frame += cur_loss
                count += 1
        loss_reid_frame /= count
        return {'loss_reid_frame': loss_reid_frame}
                
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
        loss_map = {
            "labels_seq": self.loss_labels,
            'reid_seq': self.loss_reid_seq,
            'reid_frame': self.loss_reid_frame,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        sequence_indices = self.matcher(outputs_without_aux, targets)
        
        indices = sequence_indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices

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
