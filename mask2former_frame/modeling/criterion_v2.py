# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# implement groupping loss
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

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


class SetCriterionV2(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, reid_loss_over,
                 reid_loss_type, softmax_temp, matcher_assoc, contrastive_alpha, contrastive_sigma):
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
        self.matcher_assoc = matcher_assoc
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
        self.reid_loss_over = reid_loss_over
        self.reid_loss_type = reid_loss_type
        self.softmax_temp = softmax_temp
        
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_sigma = contrastive_sigma

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses
    
    def loss_reid_kernel(self, pred_reid, pred_match, ref_reid, ref_match):
        loss_reid = 0
        loss_inv_reid = 0
        if '_' in self.reid_loss_type:
            loss_type = self.reid_loss_type.split('_')[0]
            metric = self.reid_loss_type.split('_')[1]
        else:
            loss_type = self.reid_loss_type
        if loss_type == 'contrastive':
            if metric=='cosdist':
                pred_reid_flatten = torch.cat(pred_reid, dim=0)
                ref_reid_flatten = torch.cat(ref_reid, dim=0)
                pred_match_flatten = torch.cat(pred_match, dim=0).float().detach()
                ref_match_flatten = torch.cat(ref_match, dim=0).float().detach()
                
                reid_label = pred_match_flatten @ ref_match_flatten.T
                pos_map = (reid_label>0).detach()
                neg_map = (reid_label==0).detach()
                cosine_sim = F.cosine_similarity(pred_reid_flatten.unsqueeze(2), ref_reid_flatten.T.unsqueeze(0), dim=1)
                d = 0.5*(1-cosine_sim)
                K = max(pred_match_flatten.shape[-1], 2)
                alpha=self.contrastive_alpha
                sigma=self.contrastive_sigma
                pos_map = (((d-alpha)>0)*pos_map).float().detach()
                num_pos = max(pos_map.sum().item(), 1)
                intra_loss = 1/K*(1/num_pos*(pos_map*d*d))
                inter_loss = 2/(K*(K-1))*torch.pow(torch.clamp(sigma-d, min=0), 2)*neg_map
                loss_reid += intra_loss.sum()
                loss_inv_reid += inter_loss.sum()
            elif metric == 'cossim':
                raise NotImplementedError
        elif loss_type == 'triplet':
            raise NotImplementedError
        elif loss_type == 'sigmoid':
            raise NotImplementedError
        elif loss_type == 'softmax':
            temperature = self.softmax_temp
            num_frames = len(pred_reid)
            for frame_idx in range(num_frames):
                cur_reid = pred_reid[frame_idx]
                other_reid = torch.cat([ref_reid[i] for i in range(num_frames) if i != frame_idx], dim=0)
                cur_match = pred_match[frame_idx].float().detach()
                other_match = torch.cat([ref_match[i] for i in range(num_frames) if i != frame_idx], dim=0).float().detach()
                
                similarity_map = torch.einsum("nc,mc->nm", other_reid, cur_reid)
                match_map = other_match @ cur_match.T
                match_value, match_label = torch.max(match_map, dim=1)
                match_label[match_value==0] = -100
                loss_reid += F.cross_entropy(similarity_map/temperature, match_label)

                self_similarity_map = torch.einsum("nc,mc->nm", cur_reid, cur_reid)
                match_map = cur_match @ cur_match.T
                match_value, match_label = torch.max(match_map, dim=1)
                match_label[match_value==0] = -100
                loss_inv_reid += F.cross_entropy(self_similarity_map/temperature, match_label)
        return loss_reid, loss_inv_reid
        
    def loss_assoc(self, outputs, targets, indices_frame, ref_reid=None, ref_match=None):
        l_dict = {}
        matching_table_frame, table_class = self.create_matching_table(outputs, targets, indices_frame)
        num_frames = len(outputs)
        batch_size, num_queries = outputs[0]['pred_masks'].shape[:2]
        # matching_table_frame: batch_size, num_frame x num_queries x num_gt
        # table_class: batch_size, num_gt
        if 'assoc' in self.losses:
            # pred_reid_assoc: batch_size, 1 x num_queries x channel
            # pred_class_assoc: batch_size, 1 x num_queries x num_classes+1)
            # ref_reid: batch_size, num_frame(+1) x num_queries x channel
            # ref_match: batch_size, num_frame(+1) x num_queries x num_gt
            pred_reid_assoc = [x for x in outputs[0]['pred_reid_seq']]
            pred_class_assoc = [x for x in outputs[0]['pred_logits_seq']]
            if ref_reid is None:
                ref_reid_frame = [torch.stack([outputs[f]['pred_reid'][b] for f in range(num_frames)]) for b in range(batch_size)]
                ref_match_frame = matching_table_frame
            else:
                ref_reid_frame = [x[:-1] for x in ref_reid]
                ref_match_frame = [x[:-1] for x in ref_match]
            indices_assoc = self.matcher_assoc(pred_reid_assoc, pred_class_assoc, ref_reid_frame, ref_match_frame, table_class)
            matching_table_assoc = [torch.zeros_like(x) for x in matching_table_frame]
            for b in range(len(matching_table_assoc)):
                query_idx, gt_idx = indices_assoc[b]
                matching_table_assoc[b][0, query_idx, gt_idx] = 1
                matching_table_assoc[b] = matching_table_assoc[b][0]
            # pred_reid_frame_last, matching_table_frame_last = self.collect_reid(outputs, targets, indices_frame, matching_table_frame)
            pred_reid, matching_table = self.collect_reid(outputs, targets, indices_frame, matching_table_frame, indices_assoc, matching_table_assoc)
            if ref_reid is None:
                ref_reid = pred_reid 
                ref_match = matching_table
            loss_assoc, loss_inv_assoc = self.loss_reid_assoc(pred_reid, matching_table, ref_reid, ref_match)
            
            loss_ce_assoc = self.loss_labels_assoc(pred_class_assoc, indices_assoc, table_class)
            l_dict.update({'loss_reid_assoc': loss_assoc, 'loss_inv_reid_assoc': loss_inv_assoc, 'loss_ce_assoc': loss_ce_assoc})
            
        elif 'reid' in self.losses:
            pred_reid_frame_last, matching_table_frame_last = self.collect_reid(outputs, targets, indices_frame, matching_table_frame)
            pred_reid = pred_reid_frame_last
            matching_table = matching_table_frame_last
            if ref_reid is None:
                ref_reid = pred_reid
                ref_match = matching_table
            loss_reid, loss_inv_reid = self.loss_reid_assoc(pred_reid, matching_table, ref_reid, ref_match)
            l_dict.update({'loss_reid': loss_reid, 'loss_inv_reid': loss_inv_reid})
            
        return l_dict, pred_reid, matching_table
    
    def loss_reid_assoc(self, pred_reid_batch, pred_match_batch, ref_reid_batch, ref_match_batch):
        """
            pred_reid_batch: batch_size, num_frames(+1) x num_queries x channel
            pred_match_batch: batch_size, num_frames(+1) x num_queries x num_gt 
            ref_
        """
        loss_reid = 0
        loss_inv_reid = 0
        bs = len(pred_reid_batch)
        for b in range(bs):
            pred_reid = pred_reid_batch[b]
            pred_match = pred_match_batch[b]
            ref_reid = ref_reid_batch[b]
            ref_match = ref_match_batch[b]
            r, ir = self.loss_reid_kernel(pred_reid, pred_match, ref_reid, ref_match)
            loss_reid += r
            loss_inv_reid += ir
        return loss_reid / bs, loss_inv_reid / bs
    
    def loss_labels_assoc(self, pred_logits_batch, indices_batch, table_class_batch):    
        # pred_logits: batch_size, num_queries x num_classes
        loss_ce = 0
        for pred_logits, indices, table_class in zip(pred_logits_batch, indices_batch, table_class_batch):
            query_indices, gt_indices = indices
            target_classes = torch.full(
                pred_logits.shape[:1], self.num_classes, dtype=torch.int64, device=pred_logits.device
            )
            target_classes[query_indices] = torch.tensor(table_class, device=target_classes.device)[gt_indices]

            loss_ce += F.cross_entropy(pred_logits, target_classes, self.empty_weight)            
        return loss_ce / len(pred_logits_batch)
        
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
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)
    
    def check_lists_equal(list1, list2):
        n = min(len(list1), len(list2))
        for i in range(n):
            if list1[i] != list2[i]:
                return False
        return True

    def create_matching_table(self, outputs, targets, indices):
        # return a matching table for each batch
        # return batch_size, num_frames x num_queries x max_num_gt
        batch_size, num_queries, _, _ = outputs[0]['pred_masks'].shape
        num_frames = len(outputs)
        # create matching table
        matching_table = []
        table_class = []
        for b in range(batch_size):
            t = [targets[f][b] for f in range(num_frames)]
            all_gt_uid = torch.unique(torch.cat([x['unique_indices'] for x in t])).tolist()
            all_gt_uid.sort()
            max_num_gt = len(all_gt_uid)
            cur_matching_table = torch.zeros((num_frames, num_queries, max_num_gt), dtype=torch.bool, device=outputs[0]['pred_masks'].device)
            for f in range(num_frames):
                # # check consistency
                # for idx, uid in enumerate(t[f]['unique_indices']):
                #     if idx not in mapping_table:
                #         mapping_table[idx] = uid
                #     else:
                #         assert mapping_table[idx] == uid, "unique indices are not consistent"
                        
                query_indices, gt_idx = indices[f][b]
                uid_cur_frame = t[f]['unique_indices'][gt_idx]
                uid_indices = torch.tensor([all_gt_uid.index(x) for x in uid_cur_frame], dtype=query_indices.dtype, device=query_indices.device)
                cur_matching_table[f, query_indices, uid_indices] = True
            matching_table.append(cur_matching_table)
            
            cur_table_class = []
            all_class = torch.cat([x['labels'] for x in t]).tolist()
            all_gt = torch.cat([x['unique_indices'] for x in t]).tolist()
            for x in all_gt_uid:
                cur_table_class.append(all_class[all_gt.index(x)])
            table_class.append(cur_table_class)
                
        return matching_table, table_class
    
    def collect_reid(self, outputs, targets, indices_frame=None, matching_table_frame=None, indices_assoc=None, matching_table_assoc=None):
        num_frames = len(outputs)
        pred_reid_batch = []
        pred_match_batch = []
        for b, t in enumerate(targets[0]):
            pred_reid = []
            pred_match = []
            if indices_frame is not None:
                pred_reid_frame = [outputs[f]['pred_reid'][b] for f in range(num_frames)]
                pred_match_frame = matching_table_frame[b]
                if self.reid_loss_over == "matched":
                    pred_reid_frame = [pred_reid_frame[frame_idx][query_idx] for frame_idx, (query_idx, gt_idx) in enumerate([x[b] for x in indices_frame])]
                    pred_match_frame = [pred_match_frame[frame_idx][query_idx] for frame_idx, (query_idx, gt_idx) in enumerate([x[b] for x in indices_frame])]
                else:
                    pred_reid_frame = [x for x in pred_reid_frame]
                    pred_match_frame = [x for x in pred_match_frame]
                # pred_reid_frame: num_frame, num_queries x channel
                # pred_match_frame: num_frame, num_queries x num_gt
                pred_reid=pred_reid+pred_reid_frame
                pred_match=pred_match+pred_match_frame

            if indices_assoc is not None:
                # assoc are stored in frame 0
                pred_reid_seq = outputs[0]['pred_reid_seq'][b]
                pred_match_seq = matching_table_assoc[b]
                if self.reid_loss_over == "matched":
                    query_idx, gt_idx = indices_assoc[b]
                    pred_reid_seq = pred_reid_seq[query_idx]
                    pred_match_seq = pred_match_seq[query_idx]
                # pred_reid_seq: num_queries x channel
                # pred_match_seq: num_queries x num_gt
                pred_reid.append(pred_reid_seq)
                pred_match.append(pred_match_seq)
            
            # pred_reid_batch: batch_size, num_frame(+1 if assoc), num_matched x channel
            # pred_match_batch: batch_size, num_frame(+1 if assoc), num_queries x num_gt
            pred_reid_batch.append(pred_reid)
            pred_match_batch.append(pred_match)
            
        return pred_reid_batch, pred_match_batch
    
    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        batch_size = len(targets[0])
        num_frames = len(targets)
        # Retrieve the matching between the outputs of the last layer and the targets
        indices_frame = []
        losses = {}
        for frame_idx in range(num_frames):
            outputs_cur_frame = outputs[frame_idx]
            targets_cur_frame = targets[frame_idx]
            outputs_without_aux = {k: v for k, v in outputs_cur_frame.items() if k != "aux_outputs"}
            indices = self.matcher(outputs_without_aux, targets_cur_frame)
            indices_frame.append(indices)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_masks = sum(len(t["labels"]) for t in targets_cur_frame)
            num_masks = torch.as_tensor(
                [num_masks], dtype=torch.float, device=next(iter(outputs_without_aux.values())).device
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                if loss == 'reid' or loss == 'assoc':
                    continue
                l_dict = self.get_loss(loss, outputs_without_aux, targets_cur_frame, indices, num_masks)
                l_dict = {f"frame_{frame_idx}_" + k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'reid' in self.losses or 'assoc' in self.losses:
                l_dict, ref_reid, ref_match = self.loss_assoc(outputs, targets, indices_frame)
                losses.update(l_dict)
                
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs[0]:
            num_aux = len(outputs[0]["aux_outputs"])
            for i in range(num_aux):
                indices_frame = []
                outputs_current_layer = [outputs[f]["aux_outputs"][i] for f in range(num_frames)]
                for frame_idx in range(num_frames):
                    aux_outputs = outputs_current_layer[frame_idx]
                    indices = self.matcher(aux_outputs, targets[frame_idx])
                    indices_frame.append(indices)
                    for loss in self.losses:
                        if loss == 'reid' or loss == 'assoc':
                            continue
                        l_dict = self.get_loss(loss, aux_outputs, targets[frame_idx], indices, num_masks)
                        l_dict = {f"frame_{frame_idx}_" + k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
                if 'reid' in self.losses or 'assoc' in self.losses:
                    l_dict, _, _ = self.loss_assoc(outputs_current_layer, targets, indices_frame, ref_reid, ref_match)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                
        return losses

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
