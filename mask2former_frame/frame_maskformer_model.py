# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import numpy as np
from itertools import combinations

from scipy.optimize import linear_sum_assignment
import heapq
from .modeling.reid.reid_encoder_head import reid_encoder_head

@META_ARCH_REGISTRY.register()
class FrameMaskFormer(nn.Module):
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
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        # multi-frame
        num_frames,
        memory_forward_startegy,
        # matching
        reid_loss_over,
        # reid_fg_ratio,
        # reid_loss_on,
        reid_encoder_head,
        forward_as,
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
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.num_frames = num_frames
        self.memory_forward_strategy = memory_forward_startegy
        # self.reid_fg_ratio = reid_fg_ratio
        self.reid_loss_over = reid_loss_over

        self.semantic_on = False
        self.panoptic_on = False
        self.instance_on = True
        self.test_topk_per_image = test_topk_per_image
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.reid_encoder_head = reid_encoder_head 
        self.forward_as = forward_as
        
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
        reid_weight = cfg.MODEL.MASK_FORMER.REID_WEIGHT
        reid_coeff = cfg.MODEL.MASK_FORMER.REID_COEFF
        inv_reid_weight = cfg.MODEL.MASK_FORMER.INV_REID_WEIGHT
        inv_reid_coeff = cfg.MODEL.MASK_FORMER.INV_REID_COEFF
        
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_reid=reid_coeff,
            cost_inv_reid=inv_reid_coeff,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, 
                       "loss_mask": mask_weight, 
                       "loss_dice": dice_weight,
                       "loss_reid": reid_weight,
                       "loss_inv_reid": inv_reid_weight}

        # weight_dict['loss_reid_pos'] = cfg.MODEL.MASK_FORMER.REID_POSITIVE_WEIGHT
        # weight_dict['loss_reid_neg'] = cfg.MODEL.MASK_FORMER.REID_NEGATIVE_WEIGHT

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "reid"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        
        reid_head = reid_encoder_head(cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                                      cfg.MODEL.MASK_FORMER.REID_DIM,
                                      cfg.MODEL.REID,
                                      cfg.MODEL.SEM_SEG_HEAD)
        
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME == 'SequenceMultiScaleMaskedTransformerDecoder':
            forward_as = 'sequence'
        else:
            forward_as = 'frame'
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # additional args
            "reid_loss_over": cfg.MODEL.MASK_FORMER.REID_LOSS_OVER,
            # "reid_fg_ratio": cfg.MODEL.MASK_FORMER.REID_FG_RATIO,
            'memory_forward_startegy': cfg.MODEL.MASK_FORMER.MEMORY_FORWARD_STRATEGY,
            "reid_encoder_head": reid_head,
            "forward_as": forward_as,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_one_sample(self, input_sequences):
        import matplotlib.pyplot as plt
        for input_sequence in input_sequences:
            for frame_idx, sample_inputs in enumerate(input_sequence):
                plt_idx = frame_idx*5+1
                plt.subplot(2,5,plt_idx)
                rgb = sample_inputs['image'].numpy().transpose([1, 2, 0])/255
                plt.imshow(rgb)
                plt.axis('off')
                # plt.subplot(2,3,2)
                # depth = sample_inputs['xyz'][:,:,-1].numpy()
                # plt.imshow(depth)
                if 'instances' in sample_inputs:
                    for idx in range(min(len(sample_inputs['instances']), 4)):
                        plt.subplot(2,5,plt_idx+1+idx)
                        plt.imshow(sample_inputs['instances'][idx].gt_masks[0])
                        plt.axis('off')
            plt.show()

    def prepare_for_next_frame(self, output_from_last_frame):
        """
        Args:
            output_from_last_frame (dict): a dict contains the outputs from last frame.
            For now, each dict contains:
                * "pred_logits": Tensor, classification prediction in (BATCH_SIZE, NUM_QUERY, NUM_CLASS) format.
                * "pred_masks": Tensor, instance segmentation prediction in (BATCH_SIZE, NUM_QUERY, H/4, W/4) format.
                * "meta_embedding": Tensor, meta embedding in (BATCH_SIZE, NUM_QUERY, 256) format.
        Outputs:
            batched_inputs (list[dict]): a list of dicts, each dict contains the inputs for one image.
            For now, each dict contains:
                * "image": Tensor, image in (C, H, W) format.
                * "instances": per-region ground truth
                * "memory_from_last_frame": Tensor, top-K meta embedding from last frame in (BATCH_SIZE, K, 256) format.

        """
        # if memory_forward_startegy is start with top-k, we need to update the memory
        if self.memory_forward_strategy.startswith("top-k_latest_frame") or self.memory_forward_strategy.startswith("top-k_all_frames"):
            topk = int(self.memory_forward_strategy.split(":")[-1])
            if topk == '-1':
                topk = meta_embedding.shape[1]
            # select top-K meta embedding from last frame with scores from pred_logits
            pred_logits = output_from_last_frame["pred_logits"][..., 1:].max(-1)[0]
            meta_embedding = output_from_last_frame["meta_embedding"]
            _, topk_indices = torch.topk(pred_logits, topk, dim=1)
            topk_meta_embedding = torch.gather(meta_embedding, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 256))
            return topk_meta_embedding
        elif self.memory_forward_strategy == "none":
            return None
        else:
            raise NotImplementedError

    def forward_as_frame(self, batched_sequences_in_frames):
        batched_outputs = []
        all_frame_memory = []
        batched_images = []
        batched_targets = []
        for frame_idx in range(self.num_frames):
            batch_cur_frame = batched_sequences_in_frames[frame_idx]
            images = [x["image"].to(self.device) for x in batch_cur_frame]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            
            batched_images.append(images)
            if self.training:
                batched_targets.append([b["instances"].to(self.device) for b in batched_sequences_in_frames[frame_idx]])
            if len(batched_outputs)>0:
                all_frame_memory.append(self.prepare_for_next_frame(batched_outputs[-1]))
            if all_frame_memory:
                if self.memory_forward_strategy.startswith("top-k_latest_frame"):
                    frame_memory = all_frame_memory[-1]
                elif self.memory_forward_strategy.startswith("top-k_all_frames"):
                    frame_memory = torch.cat(all_frame_memory, dim=1)
                elif self.memory_forward_strategy == "none":
                    frame_memory = None
                else:
                    raise NotImplementedError
            else:   
                frame_memory = None
            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features, frame_memory)
            batched_outputs.append(outputs)
        self.reid_encoder_head(batched_outputs, self.sem_seg_head.predictor.query_embed.weight.data)
        return batched_outputs, batched_images, batched_targets
    
    def forward_as_sequence(self, batched_sequences_in_frames):
        batched_outputs = []
        batched_images = []
        batched_targets = []
        for frame_idx in range(self.num_frames):
            batch_cur_frame = batched_sequences_in_frames[frame_idx]
            images = [x["image"].to(self.device) for x in batch_cur_frame]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            
            batched_images.append(images)
            if self.training:
                batched_targets.append([b["instances"].to(self.device) for b in batched_sequences_in_frames[frame_idx]])
        
        # (TxN)xCxHxW
        images = torch.cat([x.tensor for x in batched_images], dim=0)
        features = self.backbone(images)
        
        batched_outputs = self.sem_seg_head(features)
        return batched_outputs, batched_images, batched_targets
    
    def forward(self, batched_sequences):
        """
        Args:
            batched_sequences (list[list[dict]]): a list of sequences, each sequence is a list of dicts.
            Each dict contains the inputs for one frame.
            For now, each dict contains:
                * "image": Tensor, image in (C, H, W) format.
                * "instances": per-region ground truth
        """
        # self.visualize_one_sample(batched_sequences)

        batched_sequences_in_frames = []
        for frame_idx in range(self.num_frames):
            if self.num_frames == 1:
                # test coco
                batched_inputs_cur_frame = [x for x in batched_sequences]
            elif type(batched_sequences[0]) == dict:
                # test ytvis
                batched_inputs_cur_frame = [{'image': v['image'][frame_idx], 
                                             'height': v['height'], 
                                             'width': v['width']} for v in batched_sequences]
            else:
                # TODO change to ytvis format
                batched_inputs_cur_frame = [x[frame_idx] for x in batched_sequences]
            batched_sequences_in_frames.append(batched_inputs_cur_frame)
        
        if self.forward_as == 'frame':
            batched_outputs, batched_images, batched_targets = self.forward_as_frame(batched_sequences_in_frames)
        else:
            batched_outputs, batched_images, batched_targets = self.forward_as_sequence(batched_sequences_in_frames)
        return self.post_process(batched_images, batched_sequences_in_frames, batched_targets, batched_outputs)
    
    def post_process(self, batched_images, batched_sequence_in_frames, batched_targets, batched_outputs):
        # breakpoint()
        if self.training:
            all_frame_losses = {}
            # Batch_size X 20 x num_frames
            # matched_query_in_each_frame = -torch.ones(len(batched_targets[0]), 20, len(batched_targets), dtype=torch.int32)
            history = [{} for i in range(len(batched_targets[0]))]
            for frame_id in range(self.num_frames):
                images = batched_images[frame_id]
                outputs = batched_outputs[frame_id]
                
                # mask classification target
                if len(batched_targets)>0:
                    gt_instances = batched_targets[frame_id]
                    targets = self.prepare_targets(gt_instances, images, history)
                else:
                    targets = None

                # bipartite matching-based loss
                losses, indices = self.criterion(outputs, targets)
                if frame_id+1<self.num_frames:
                    for b, t in enumerate(batched_targets[frame_id+1]):
                        # t['track_reid'] = batched_outputs[frame_id]["reid_embed"][b]
                        track_id_cur_frame = t.track_id.tolist()
                        track_id_prev_frame = batched_targets[frame_id][b].track_id.tolist()
                        gt_matched_history_indices = []
                        for track_id in track_id_cur_frame:
                            if track_id in track_id_prev_frame:
                                track_index_in_prev_frame = track_id_prev_frame.index(track_id)
                                src_idx = indices[b][0].tolist()
                                tgt_idx = indices[b][1].tolist()
                                query_index_in_prev_frame = src_idx[tgt_idx.index(track_index_in_prev_frame)]
                                gt_matched_history_indices.append(query_index_in_prev_frame)
                            else:
                                gt_matched_history_indices.append(-1)
                        if self.reid_loss_over == 'all':
                            history[b]["track_reid"] = batched_outputs[frame_id]["pred_reid"][b]
                            history[b]['gt_matched_track_indices'] = gt_matched_history_indices
                        elif self.reid_loss_over == 'matched':
                            track_id = []
                            gt_matched_track_indices = []
                            for i in gt_matched_history_indices:
                                if i!=-1:
                                    track_id.append(batched_outputs[frame_id]["pred_reid"][b][i])
                                    gt_matched_track_indices.append(len(track_id)-1)
                                else:
                                    gt_matched_track_indices.append(-1)
                            history[b]["track_reid"] = torch.stack(track_id, dim=0)
                            history[b]['gt_matched_track_indices'] = torch.as_tensor(gt_matched_track_indices)
                        
                    
                # if self.reid_loss_on:
                #     # fill the matching matrix
                #     self.fill_matching_matrix(indices, matched_query_in_each_frame, frame_id)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                
                for k in list(losses.keys()):
                    all_frame_losses[f"frame_{frame_id}_{k}"] = losses.pop(k)
            
            # if self.reid_loss_on:
            #     # generate contrastive loss for all frames
            #     reid_positive_loss, reid_negative_loss = self.reid_loss(matched_query_in_each_frame, batched_outputs)
            #     all_frame_losses["loss_reid_pos"] = self.criterion.weight_dict["loss_reid_pos"] * reid_positive_loss
            #     all_frame_losses["loss_reid_neg"] = self.criterion.weight_dict["loss_reid_neg"] * reid_negative_loss

            return all_frame_losses
        else:
            # test
            batch_size = len(batched_images[0])
            all_frame_processed_results = [[] for _ in range(batch_size)]
            for frame_id in range(self.num_frames):
                images = batched_images[frame_id]
                outputs = batched_outputs[frame_id]
                batch_cur_frame = batched_sequence_in_frames[frame_id]
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                reid_embed_results = outputs["pred_reid"]
                # upsample mask
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                for batch_idx, mask_cls_result, mask_pred_result, reid_embed_result, input_image, input_per_image, image_size in zip(
                    range(batch_size), mask_cls_results, mask_pred_results, reid_embed_results, batched_images, batch_cur_frame, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_result = {}

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, reid_embed_result)
                        processed_result["instances"] = instance_r

                    all_frame_processed_results[batch_idx].append(processed_result)

            if self.num_frames==1:
                all_frame_processed_results = [x[0] for x in all_frame_processed_results]
                return all_frame_processed_results
            else:
                instances = self.postprocess_video_output(all_frame_processed_results)
                return self.convert_to_ytvis_format(instances)

    def postprocess_video_output(self, all_frame_processed_results):
        """
        :return
        outputs["pred_scores"]: value
        outputs["pred_classes"]: value
        outputs["pred_masks"]: list of 2d array
        """
        assert len(all_frame_processed_results)==1, "Only support single video inference"
        # topk_sequence = heapq()
        for batch_idx, batch_results in enumerate(all_frame_processed_results):
            instances_in_history = []
            for frame_idx, frame_results in enumerate(batch_results):
                instances_cur_frame = [{'pred_masks': frame_results['instances'].pred_masks[i].cpu(),
                                        'scores': frame_results['instances'].scores[i].item(),
                                        'reid_embed': frame_results['instances'].reid_embed[i].cpu(),
                                        'pred_classes': frame_results['instances'].pred_classes[i].cpu(),
                                        'frame_index': frame_idx,
                                        'idx_in_img': i}
                                       for i in range(len(frame_results['instances']))]
                if len(instances_cur_frame)==0:
                    continue 
                
                if len(instances_in_history)==0 and len(instances_cur_frame)>0:
                    instances_cur_frame = [{k: [v] for k, v in instances.items()} for instances in instances_cur_frame]
                    instances_in_history = instances_cur_frame
                    continue
                
                method = 'hungarian'
                if method == 'hungarian':
                    # find the best match for each instance in the current frame
                    reid_embed_in_history = torch.stack([x['reid_embed'][-1] for x in instances_in_history])
                    reid_embed_in_cur_frame = torch.stack([x['reid_embed'] for x in instances_cur_frame])
                    similarity_matrix = torch.einsum("nc,mc->nm", reid_embed_in_history, reid_embed_in_cur_frame)
                    # similarity_matrix = F.log_softmax(similarity_matrix, dim=1)
                    # similarity_matrix = F.softmax(reid_embed_in_history.unsqueeze(2), reid_embed_in_cur_frame.T.unsqueeze(0), dim=1)
                    indices_in_prev_frame, indices_in_cur_frame = linear_sum_assignment(-similarity_matrix)
                    
                    matched_instance_indices = []
                    for instance_idx_in_prev_frame, instance in enumerate(instances_in_history):
                        if instance_idx_in_prev_frame in indices_in_prev_frame:
                            # update the instance
                            instance_idx_in_cur_frame = indices_in_cur_frame[indices_in_prev_frame==instance_idx_in_prev_frame][0]
                            matched_instance_indices.append(instance_idx_in_cur_frame)
                            instance_to_update = instances_cur_frame[instance_idx_in_cur_frame]
                            for k in instances_in_history[instance_idx_in_prev_frame].keys():
                                instances_in_history[instance_idx_in_prev_frame][k].append(instance_to_update[k])
                        
                    # add a new instance
                    for instance_idx, instance in enumerate(instances_cur_frame):
                        if instance_idx not in matched_instance_indices:
                            new_instance_this_frame = {k: [v] for k, v in instance.items()}
                            instances_in_history.append(new_instance_this_frame)
                elif method == 'beam_search':
                    reid_embed_in_history = torch.stack([x['reid_embed'][-1] for x in instances_in_history])
                    reid_embed_in_cur_frame = torch.stack([x['reid_embed'] for x in instances_cur_frame])
                    similarity_matrix = torch.einsum("nc,mc->nm", reid_embed_in_history, reid_embed_in_cur_frame)
                    # similarity_matrix = F.cosine_similarity(reid_embed_in_history.unsqueeze(2), reid_embed_in_cur_frame.T.unsqueeze(0), dim=1)
                    similarity_matrix = torch.softmax(similarity_matrix, dim=1)
                    new_history = []
                    for instance_idx_in_prev_frame, instance_sequence in enumerate(instances_in_history):
                        for instance_idx, instance_to_update in enumerate(instances_cur_frame):
                            instance_to_update = {k: v for k, v in instance_to_update.items()}
                            instance_to_update['scores'] = instance_to_update['scores']*instance_sequence['scores'][-1]*similarity_matrix[instance_idx_in_prev_frame][instance_idx].item()
                            
                            instance_history_copy = {k: [t for t in v] for k,v in instance_sequence.items()}
                            for k in instance_history_copy.keys():
                                instance_history_copy[k].append(instance_to_update[k])
                            new_history.append(instance_history_copy)
                    instances_in_history = new_history
                    # add a new instance
                    for instance_idx, instance in enumerate(instances_cur_frame):
                        new_instance_this_frame = {k: [v] for k, v in instance.items()}
                        instances_in_history.append(new_instance_this_frame)
        if method == 'beam_search':
            instances_in_history = self.use_prior_in_amazon_task(instances_in_history)
                                    
        return instances_in_history
    
    def use_prior_in_amazon_task(self, instances):
        max_life_length = self.num_frames+1
        num_object = max(x['idx_in_img'][-1] for x in instances)+1
        max_score = np.zeros([num_object, max_life_length])
        max_score_idx = np.zeros([num_object, max_life_length], dtype=np.int32)
        for instance_idx, instance in enumerate(instances):
            object_idx = instance['idx_in_img'][-1]
            object_life_length = len(instance['idx_in_img'])
            score = instance['scores'][-1]
            print(f"{instance_idx}, {instance['idx_in_img']}, {instance['scores'][-1]}, {max_score[object_idx, object_life_length]}")
            if score>max_score[object_idx, object_life_length]:
                max_score[object_idx, object_life_length] = score
                max_score_idx[object_idx, object_life_length] = instance_idx
        for i in range(num_object):
            for j in range(max_life_length):
                print(f"{i}, {j}, {max_score[i, j]}, {max_score_idx[i, j]}, {instances[max_score_idx[i, j]]['idx_in_img']}")
        valid_score = max_score[:, 2:-1] # remove sequence with length 0, 1 and max_life_length
        object_idx, life_time = linear_sum_assignment(-valid_score)
        instance_indices = [max_score_idx[object_idx[i], life_time[i]+2] for i in range(len(object_idx))]
        return [instances[t] for t in instance_indices]
        
    def convert_to_ytvis_format(self, instances):
        """
        :param results: list of dict
        :return: list of dict
        """
        results_ytvis = {"pred_scores": [], "pred_labels": [], "pred_masks": [], "image_size": instances[0]['pred_masks'][0].shape}
        empty_mask = torch.zeros_like(instances[0]['pred_masks'][0])
        for instance_idx, instance in enumerate(instances):
            score = instance['scores'][-1]
            pred_label = instance['pred_classes'][-1].item()
            pred_masks = [empty_mask] * self.num_frames
            for appearance_idx, frame_idx in enumerate(instance['frame_index']):
                pred_masks[frame_idx] = instance['pred_masks'][appearance_idx]
            results_ytvis["pred_scores"].append(score)
            results_ytvis["pred_labels"].append(pred_label)
            results_ytvis["pred_masks"].append(pred_masks)
        return results_ytvis
    
    def prepare_targets(self, targets, images, history):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for img_idx, targets_per_image in enumerate(targets):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            new_t = {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            for k, v in history[img_idx].items():
                if type(v) != torch.tensor:
                    new_t[k] = torch.as_tensor(v, device=gt_masks.device)
                else:
                    new_t[k] = v
            new_targets.append(new_t)
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
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
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, reid_embed=None):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        reid_embed = reid_embed[topk_indices]
        # breakpoint()
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
            reid_embed = reid_embed[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.reid_embed = reid_embed

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        above_thresh_indices = torch.where(result.scores>0.6)[0]
        for k in result._fields.keys():
            result._fields[k] = result._fields[k][above_thresh_indices]
        return result
