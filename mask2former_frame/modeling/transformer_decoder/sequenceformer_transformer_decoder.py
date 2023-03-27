# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from .frame_maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias=True):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if bias==False:
            self.layers[-1] = nn.Linear(h[-1], output_dim, bias=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class SequenceMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
                if "transformer_self_attention_layers" in k and self.self_init_frame_attn:
                    newk = k.replace("transformer_self_attention_layers", "transformer_frame_attention_layers")
                    state_dict[newk] = state_dict[k]

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        reid_dim: int, 
        reid_layer_type: str,
        enforce_input_project: bool,
        #---
        encoder_layer_structure,
        frame_attn_pos_type,
        batch_size,
        num_gpus,
        reid_layer_bias,
        normalize_before_reid,
        use_layer_attnmask,
        mask_use_different_feature,
        ignore_first_output,
        ignore_first_reid,
        reid_norm,
        frame_selfmask,
        dropout,
        frame_dropout,
        feat_dropout,
        self_init_frame_attn,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_frame_attention_layers = nn.ModuleList()
        self.transformer_video_attention_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            if 'self' in encoder_layer_structure:
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )
            if 'cross' in encoder_layer_structure:
                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )

            if 'ffn' in encoder_layer_structure:
                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )
            
            if 'frame' in encoder_layer_structure:
                self.transformer_frame_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=frame_dropout,
                        normalize_before=pre_norm,
                    )
                )
            
            if 'video' in encoder_layer_structure:
                self.transformer_video_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )                
                )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        if reid_layer_type == "mlp":
            self.reid_embed = MLP(hidden_dim, hidden_dim, reid_dim, 3, bias=reid_layer_bias)
        elif reid_layer_type == "linear":
            self.reid_embed = nn.Linear(hidden_dim,  reid_dim, bias=reid_layer_bias)
        else:
            raise KeyError
        self.encoder_layer_structure = encoder_layer_structure
        self.frame_attn_pos_type = frame_attn_pos_type
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.normalize_before_reid = normalize_before_reid
        self.use_layer_attnmask = use_layer_attnmask
        self.mask_use_different_feature = mask_use_different_feature
        self.ignore_first_output = ignore_first_output
        self.ignore_first_reid = ignore_first_reid
        self.reid_norm = reid_norm
        self.frame_selfmask = frame_selfmask
        self.feat_dropout = feat_dropout
        self.self_init_frame_attn = self_init_frame_attn

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["reid_dim"] = cfg.MODEL.MASK_FORMER.REID_DIM
        ret["reid_layer_type"] = cfg.MODEL.REID.ENCODER_TYPE
        ret["reid_layer_bias"] = cfg.MODEL.REID.ENCODER_LAST_LAYER_BIAS
        ret["encoder_layer_structure"] = cfg.MODEL.REID.ENCODER_LAYER_STRUCTURE
        ret["frame_attn_pos_type"] = cfg.MODEL.REID.ENCODER_POS_TYPE
        ret["batch_size"] = cfg.SOLVER.IMS_PER_BATCH
        ret["num_gpus"] = cfg.SOLVER.NUM_GPUS
        ret["normalize_before_reid"] = cfg.MODEL.REID.ENCODER_NORMALIZE_BEFORE
        ret["use_layer_attnmask"] = cfg.MODEL.MASK_FORMER.USE_LAYER_ATTNMASK
        ret["mask_use_different_feature"] = cfg.MODEL.MASK_FORMER.MASK_USE_DIFFERENT_FEATURE
        ret["ignore_first_output"] = cfg.MODEL.MASK_FORMER.IGNORE_FIRST_OUTPUT
        ret["ignore_first_reid"] = cfg.MODEL.MASK_FORMER.IGNORE_FIRST_REID
        ret["reid_norm"] = cfg.MODEL.REID.REID_NORMALIZE
        ret["frame_selfmask"] = cfg.MODEL.MASK_FORMER.FRAME_SELFMASK
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["frame_dropout"] = cfg.MODEL.MASK_FORMER.FRAME_DROPOUT
        ret["feat_dropout"] = cfg.MODEL.MASK_FORMER.FEAT_DROPOUT
        ret["self_init_frame_attn"] = cfg.MODEL.MASK_FORMER.SELF_INIT_FRAME_ATTN


        return ret

    def forward(self, x, mask_features, frame_memory=None, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        feat_dropout_mask = []
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
            
            if self.feat_dropout > 0:
                cur_pos = []
                cur_src = []
                hw, n, c = pos[-1].shape
                cur_feat_dropout_mask = []
                for j in range(n):
                    perm = torch.randperm(hw)
                    keep = perm[:int(hw * (1 - self.feat_dropout))].sort()[0]
                    cur_pos.append(pos[-1][:, j][keep])
                    cur_src.append(src[-1][:, j][keep])
                    cur_feat_dropout_mask.append(keep)
                    
                pos[-1] = torch.stack(cur_pos, dim=1)
                src[-1] = torch.stack(cur_src, dim=1)
                feat_dropout_mask.append(cur_feat_dropout_mask)

        _, TB, feat_dim = src[0].shape
        batch_size = self.batch_size // self.num_gpus if self.training else 1
        num_frames = TB // batch_size
        num_queries = self.num_queries
        mask_h = mask_features.shape[-2]
        mask_w = mask_features.shape[-1]

        # Qx(TxN)xC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, TB, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, TB, 1)

        predictions_class = []
        predictions_mask = []
        predictions_reid = []
        # meta_embedding = []
        
        # prediction heads on learnable query features
        outputs_class, outputs_mask, output_reid, attn_mask = self.forward_prediction_heads(output, None, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class.reshape(num_frames, batch_size, num_queries, -1))
        predictions_mask.append(outputs_mask.reshape(num_frames, batch_size, num_queries, mask_h, mask_w))
        predictions_reid.append(output_reid.reshape(num_frames, batch_size, num_queries, -1))
        if self.feat_dropout > 0:
            attn_mask = attn_mask.reshape([TB, self.num_heads, num_queries, size_list[0][0]*size_list[0][1]])
            attn_mask = torch.stack([x[:, :, i] for x, i in zip(attn_mask, feat_dropout_mask[0])], dim=0).flatten(0, 1)
        # meta_embedding.append(self.decoder_norm(output).transpose(0, 1))
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False                
            for layer_type in self.encoder_layer_structure:
                if layer_type == "cross":
                    # attention between queries and image features
                    output = self.transformer_cross_attention_layers[i](
                        output, src[level_index],
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=pos[level_index], query_pos=query_embed
                    )
                elif layer_type == "self":
                    # self attention
                    output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed
                    )
                elif layer_type == "ffn":
                    # FFN
                    output = self.transformer_ffn_layers[i](
                        output
                    )
                elif layer_type == "frame":
                    # full attention with queries from other frames
                    # Qx(TxN)xC -> (QxT)xNxC
                    output_before_frame = output
                    output = output.reshape(num_queries*num_frames, batch_size, feat_dim)
                    if self.frame_attn_pos_type == "learned":
                        frame_pos = query_embed.reshape(num_queries*num_frames, batch_size, feat_dim)
                    else:
                        raise KeyError
                    if self.frame_selfmask:
                        only_self_mask = [torch.ones((100,100)) for i in range(num_frames)]
                        only_self_mask = (torch.block_diag(*only_self_mask) == 1).cuda()
                        output = self.transformer_frame_attention_layers[i](
                            output, tgt_mask=only_self_mask,
                            tgt_key_padding_mask=None,
                            query_pos=frame_pos
                        )
                    else:
                        output = self.transformer_frame_attention_layers[i](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=frame_pos
                        )
                    output = output.reshape(num_queries, num_frames*batch_size, feat_dim)
                elif layer_type == "video":
                    output = output.reshape(num_queries*num_frames, batch_size, feat_dim)
                    output = self.transformer_video_attention_layers[i](
                        output, src[level_index].reshape(src[level_index].shape[0]*num_frames, batch_size, feat_dim),
                        memory_mask=None,
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=pos[level_index].reshape(src[level_index].shape[0]*num_frames, batch_size, feat_dim), 
                        query_pos=query_embed.reshape(num_queries*num_frames, batch_size, feat_dim)
                    )
                    output = output.reshape(num_queries, num_frames*batch_size, feat_dim)
                else:
                    raise KeyError

            if self.encoder_layer_structure[-1] == "frame" and self.mask_use_different_feature:
                output_mask = output_before_frame
            else:
                output_mask = None
            outputs_class, outputs_mask, outputs_reid, attn_mask = \
                            self.forward_prediction_heads(output, output_mask, \
                                mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class.reshape(num_frames, batch_size, num_queries, -1))
            predictions_mask.append(outputs_mask.reshape(num_frames, batch_size, num_queries, mask_h, mask_w))
            predictions_reid.append(outputs_reid.reshape(num_frames, batch_size, num_queries, -1))
            if self.feat_dropout > 0:
                hw = size_list[(i + 1) % self.num_feature_levels]
                attn_mask = attn_mask.reshape([TB, self.num_heads, num_queries, hw[0]*hw[1]])
                attn_mask = torch.stack([x[:, :, i] for x, i in zip(attn_mask, feat_dropout_mask[(i + 1) % self.num_feature_levels])], dim=0).flatten(0, 1)
            # meta_embedding.append(self.decoder_norm(output).transpose(0, 1))

        assert len(predictions_class) == self.num_layers + 1
        if self.ignore_first_output:
            predictions_class = predictions_class[1:]
            predictions_mask = predictions_mask[1:]
            predictions_reid = predictions_reid[1:]
        elif self.ignore_first_reid:
            predictions_reid[0] = [None for _ in predictions_reid[0]]
        out = []
        output_layer = -1
        if output_layer != -1:
            # warn the user with flashing text
            logging.warning("\033[91m" + "WARNING: output_layer is not -1, this is just for debugging" + "\033[0m")
        for frame_idx in range(num_frames):
            out.append({
                'pred_logits': predictions_class[output_layer][frame_idx],
                'pred_masks': predictions_mask[output_layer][frame_idx],
                'pred_reid': predictions_reid[output_layer][frame_idx],
                'aux_outputs': self._set_aux_loss(
                    [x[frame_idx] for x in predictions_class] if self.mask_classification else None, 
                    [x[frame_idx] for x in predictions_mask],
                    [x[frame_idx] for x in predictions_reid],
                ),
                # 'meta_embedding': meta_embedding[-1],
                'mask_feature': mask_features,
            })

        return out

    def forward_prediction_heads(self, output, output_mask, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        if output_mask is not None:
            decoder_output_mask = self.decoder_norm(output_mask)
            decoder_output_mask = decoder_output_mask.transpose(0, 1)
            mask_embed = self.mask_embed(decoder_output_mask)
        else:
            mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        if self.normalize_before_reid:
            outputs_reid = self.reid_embed(decoder_output)
        else:
            outputs_reid = self.reid_embed(output.transpose(0,1))
        if self.reid_norm:
            outputs_reid = F.normalize(outputs_reid, p=2, dim=-1)
        if self.use_layer_attnmask:
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # breakpoint()
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
        else:
            attn_mask = torch.zeros((outputs_mask.shape[0]*self.num_heads, outputs_mask.shape[1], 
                attn_mask_target_size[0]*attn_mask_target_size[1]), dtype=torch.bool, device=outputs_mask.device)

        return outputs_class, outputs_mask, outputs_reid, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_reid):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_reid": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_reid[:-1])
            ]
        else:
            return [{"pred_masks": b, "pred_reid": c} for b, c in zip(outputs_seg_masks[:-1], meta_embedding[:-1])]
