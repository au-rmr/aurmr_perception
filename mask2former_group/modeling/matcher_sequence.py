# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample

def batch_reid_loss(matched_reid: torch.Tensor, pred_center: torch.Tensor, matched_reid_valid: torch.Tensor,
                    reid_seq_loss_type: str):
    """
    Args:
        matched_reid: A float tensor of arbitrary shape.
                The predictions for each example.
                TxGx256
        pred_center: A float tensor with the same shape as inputs. 
                Store the predicted reid grouping center
                Qx256
        matched_reid_valid: A float tensor indicating whether the matched reid is valid
                TxG
    Returns:
        per_query_loss: A float tensor of shape GxQ
    """
    similarity = (matched_reid@pred_center.t()).permute(0,2,1) # TxsQxfQ
    # similarity = torch.einsum("qc,tgc->tqg", pred_center, matched_reid)    
    if reid_seq_loss_type == 'sigmoid':
        similarity_valid = matched_reid_valid.unsqueeze(1) # Tx1xG
        similarity_label = torch.ones_like(similarity)
        loss = F.binary_cross_entropy_with_logits(similarity, similarity_label, reduction="none")
        per_query_loss = (loss*similarity_valid).sum(dim=0)/(similarity_valid.sum(dim=0)+1e-6)
    elif reid_seq_loss_type == 'softmax':
        similarity_valid = matched_reid_valid.unsqueeze(1) # Tx1xG
        # softmax along Q
        loss_forward = -F.log_softmax(similarity, dim=1) 
        # softmax along G, but some G are invalid
        # exp = F.exp(similarity, dim=2)*similarity_valid 
        # loss_backward = -F.log(exp/(exp.sum(dim=2, keepdim=True)+1e-6))
        loss_backward = 0
        loss = (loss_forward+loss_backward)*similarity_valid
        per_query_loss = loss.sum(dim=0)/(similarity_valid.sum(dim=0)+1e-6)
    else:
        raise NotImplementedError
    
    return per_query_loss

batch_reid_loss_jit = torch.jit.script(
    batch_reid_loss
)  # type: torch.jit.ScriptModule

class HungarianMatcherSequence(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, 
                 cost_mask: float = 1, 
                 cost_dice: float = 1,
                 cost_reid_seq: float = 1, 
                 cost_inv_reid: float = 0,
                 num_points: int = 0,
                 reid_seq_loss_type = 'sigmoid'):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_reid = cost_reid_seq
        self.cost_inv_reid = cost_inv_reid

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_reid_seq != 0, "all costs cant be 0"

        self.num_points = num_points
        self.reid_seq_loss_type = reid_seq_loss_type

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits_seq"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits_seq"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]['matched_labels']

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            with autocast(enabled=False):
                if self.cost_reid>0:
                    cost_reid = batch_reid_loss(outputs["matched_reid"][b], 
                                                outputs["pred_reid_seq"][b],
                                                outputs['matched_reid_valid'][b],
                                                self.reid_seq_loss_type)
                                                            
                else:
                    cost_reid = 0
            # Final cost matrix
            C = ( self.cost_class * cost_class
                + self.cost_reid * cost_reid
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class_sequence: {}".format(self.cost_class),
            "cost_reid_sequence: {}".format(self.cost_reid),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
