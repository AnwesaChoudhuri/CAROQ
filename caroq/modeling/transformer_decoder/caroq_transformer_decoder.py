import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
import pdb
import time
from .position_encoding import PositionEmbeddingSine
from .position_encoding_3d import PositionEmbeddingSine3D
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from torchvision.utils import save_image
import numpy as np
import cv2

from .relative_attention import RelativeMultiHeadAttention


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        relative_pos=True,
        P1=100,
        P2=100,
    ):
        super().__init__()
        self.relative_pos = relative_pos
        if self.relative_pos:
            self.self_attn = RelativeMultiHeadAttention(
                nhead, d_model, dropout=dropout, P1=P1, P2=P2
            )

        else:
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

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_embeddings=None,
        rel_scale=0,
    ):

        if self.relative_pos:
            tgt2 = self.self_attn(
                query=tgt,
                key=tgt,
                value=tgt,
                mask=None,
                query_pos=query_pos,
                key_pos=query_pos,
                pos_embeddings=pos_embeddings,
                rel_scale=rel_scale,
            )
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_embeddings=None,
        rel_scale=0,
    ):
        tgt2 = self.norm(tgt)
        if self.relative_pos:
            tgt2 = self.self_attn(
                query=tgt2,
                key=tgt2,
                value=tgt2,
                mask=None,
                query_pos=query_pos,
                key_pos=query_pos,
                pos_embeddings=pos_embeddings,
                rel_scale=rel_scale,
            )
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(
                q,
                k,
                value=tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_embeddings=None,
        rel_scale=0,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                tgt_mask,
                tgt_key_padding_mask,
                query_pos,
                pos_embeddings=pos_embeddings,
                rel_scale=rel_scale,
            )
        return self.forward_post(
            tgt,
            tgt_mask,
            tgt_key_padding_mask,
            query_pos,
            pos_embeddings=pos_embeddings,
            rel_scale=rel_scale,
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        relative_pos=True,
        P1=1024,
        P2=100,
    ):
        super().__init__()
        self.relative_pos = relative_pos

        if self.relative_pos:
            self.multihead_attn = RelativeMultiHeadAttention(
                nhead, d_model, dropout=dropout, P1=P1, P2=P2
            )
        else:
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

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_embeddings=None,
        rel_scale=0,
    ):
        if self.relative_pos:

            tgt2 = self.multihead_attn(
                query=tgt,
                key=memory,
                value=memory,
                mask=memory_mask,
                query_pos=query_pos,
                key_pos=pos,
                pos_embeddings=pos_embeddings,
                rel_scale=rel_scale,
            )
        else:
            q = self.with_pos_embed(tgt, query_pos)
            k = self.with_pos_embed(memory, pos)

            tgt2 = self.multihead_attn(
                query=q,
                key=k,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_embeddings=None,
        rel_scale=0,
    ):
        tgt2 = self.norm(tgt)
        if self.relative_pos:
            tgt2 = self.multihead_attn(
                query=tgt,
                key=memory,
                value=memory,
                mask=memory_mask,
                query_pos=query_pos,
                key_pos=pos,
                pos_embeddings=pos_embeddings,
                rel_scale=rel_scale,
            )

        else:
            q = self.with_pos_embed(tgt2, query_pos)
            k = self.with_pos_embed(memory, pos)
            tgt2 = self.multihead_attn(
                query=q,
                key=k,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_embeddings=None,
        rel_scale=0,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                pos_embeddings=pos_embeddings,
                rel_scale=rel_scale,
            )
        return self.forward_post(
            tgt,
            memory,
            memory_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            pos_embeddings=pos_embeddings,
            rel_scale=rel_scale,
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
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
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
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
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

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
        time_gap=5,
        pe_layer=".",
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        relative_pos: bool,
        relative_pos_power: int,
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
        self.relative_pos = relative_pos

        if self.relative_pos:
            P = 2 ** relative_pos_power
            self.key_pos_embeddings = None  # nn.Embedding(P * 2, hidden_dim)
            self.query_pos_embeddings = (
                None  # nn.Embedding(num_queries * 2, hidden_dim)
            )
            self.key_second_order_scale = torch.tensor(
                0.0
            )  # nn.Parameter(torch.tensor(0.), requires_grad=True)
            self.query_second_order_scale = torch.tensor(
                0.0
            )  # nn.Parameter(torch.tensor(0.), requires_grad=True)
            # nn.init.xavier_normal_(self.key_pos_embeddings)
            # self.Wr=nn.Linear(hidden_dim, hidden_dim)
        else:
            self.key_pos_embeddings = None
            self.query_pos_embeddings = None
            self.key_second_order_scale = torch.tensor(
                0.0
            )  # nn.Parameter(torch.tensor(0.), requires_grad=True)
            self.query_second_order_scale = torch.tensor(
                0.0
            )  # nn.Parameter(torch.tensor(0.), requires_grad=True)

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = get_pe_layer(pe_layer, N_steps)

        self.pe_layer_from_config = pe_layer
        self.num_feature_levels = 3

        # define Transformer decoder here
        self.time_gap = time_gap
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    relative_pos=self.relative_pos,
                    P1=num_queries,
                    P2=num_queries,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    relative_pos=self.relative_pos,
                    P1=2
                    ** (
                        relative_pos_power
                        - (self.num_feature_levels - (i % self.num_feature_levels) - 1)
                        * 2
                    ),
                    P2=num_queries,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.relative_pos = relative_pos
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        if self.relative_pos:
            self.query_embed = None
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)

        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        if pe_layer == "2D":
            self.mask_embed = MLPmsk(hidden_dim, hidden_dim, mask_dim, 3, self.time_gap)
        elif pe_layer == "3D":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["time_gap"] = cfg.MODEL.MASK_FORMER.TIME_GAP
        ret["pe_layer"] = cfg.MODEL.MASK_FORMER.POSITIONAL_ENCODING

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["relative_pos"] = cfg.MODEL.MASK_FORMER.RELATIVE_POS
        ret["relative_pos_power"] = cfg.MODEL.MASK_FORMER.RELATIVE_POS_POWER

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

        return ret

    def forward(
        self,
        x,
        mask_features,
        mask=None,
        prev_output=[],
        training=True,
        start=0,
        memory=[],
    ):

        # x is a list of multi-scale feature
        if training:
            bs = int(mask_features.shape[0] / self.time_gap)
            start = (
                torch.cat(
                    [
                        torch.tensor([0, self.time_gap], device=x[0].device)
                        for k in range(int(bs / 2))
                    ]
                )
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
            )
        else:
            bs = 1
            x_copy = x.copy()
            mask_features_copy = mask_features.clone()
            if memory != []:
                x = [torch.cat([m, xi], dim=0) for xi, m in zip(memory[:-1], x)]
                mask_features = torch.cat([memory[-1], mask_features], dim=0)

        if self.pe_layer_from_config == "3D":
            _, c_m, h_m, w_m = mask_features.shape
            mask_features = mask_features.view(bs, -1, c_m, h_m, w_m)

        assert len(x) == self.num_feature_levels
        del mask  # doesn't affect performance

        pos, src, size_list = self.get_pos_src_sizes(
            x, start, bs, memory=memory[:-1], training=training
        )

        if self.pe_layer_from_config == "3D":
            # QxNxC
            if self.query_embed == None:
                query_embed = None
            else:
                query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        elif self.pe_layer_from_config == "2D":
            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(
                1, bs * self.time_gap, 1
            )
            output = self.query_feat.weight.unsqueeze(1).repeat(
                1, bs * self.time_gap, 1
            )

        if self.training:
            memory = []
        else:
            memory = x_copy + [mask_features_copy]

        predictions_class, predictions_mask, output_return = self.forward_decoder(
            output,
            mask_features,
            size_list,
            src,
            pos,
            query_embed,
            training,
            prev_output=prev_output,
            memory=memory[:-1],
        )

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }

        return out, output_return

    def get_pos_src_sizes(self, x, start, bs, memory=[], training=True):
        src = []
        pos = []
        size_list = []
        if training:
            time_gap = self.time_gap
        else:  # during eval
            time_gap = x[0].shape[0]
        x_new = x

        for i in range(self.num_feature_levels):
            size_list.append(x_new[i].shape[-2:])
            if self.pe_layer_from_config == "3D":

                pos.append(
                    self.pe_layer(
                        x_new[i].view(
                            bs, time_gap, -1, size_list[-1][0], size_list[-1][1]
                        ),
                        mask=None,
                        start=start,
                    ).flatten(3)
                )
                src.append(
                    self.input_proj[i](x_new[i]).flatten(2)
                    + self.level_embed.weight[i][None, :, None]
                )

                # NTxCxHW => NxTxCxHW => (TxHW)xNxC
                _, c, hw = src[-1].shape
                pos[-1] = (
                    pos[-1].view(bs, time_gap, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
                )
                src[-1] = (
                    src[-1].view(bs, time_gap, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
                )

            elif self.pe_layer_from_config == "2D":
                pos.append(self.pe_layer(x_new[i], None).flatten(2))
                src.append(
                    self.input_proj[i](x_new[i]).flatten(2)
                    + self.level_embed.weight[i][None, :, None]
                )

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

        return pos, src, size_list

    def forward_decoder_level(self, output, src, pos, query_embed, i, attn_mask):
        level_index = i % self.num_feature_levels
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        # attention: cross-attention first
        output = self.transformer_cross_attention_layers[i](
            output,
            src[level_index],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos[level_index],
            query_pos=query_embed,
            pos_embeddings=self.key_pos_embeddings,
            rel_scale=self.key_second_order_scale,
        )

        output = self.transformer_self_attention_layers[i](
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed,
            pos_embeddings=self.query_pos_embeddings,
            rel_scale=self.query_second_order_scale,
        )

        # # FFN
        output = self.transformer_ffn_layers[i](output)

        return output

    def forward_decoder(
        self,
        output,
        mask_features,
        size_list,
        src,
        pos,
        query_embed,
        training,
        prev_output=[],
        memory=[],
    ):
        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], training=training
        )
        predictions_class.append(outputs_class)

        if self.pe_layer_from_config == "3D":
            bs, q, t, h, w = outputs_mask.shape
            outputs_mask = outputs_mask.permute(0, 2, 1, 3, 4).reshape(bs * t, q, h, w)

        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            current_size = size_list[(i) % self.num_feature_levels]

            output = self.forward_decoder_level(
                output, src, pos, query_embed, i, attn_mask
            )

            if i == 2:
                output = self.propagate(output, training, prev_output=prev_output)

            if i < self.num_layers - 1:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                    training=training,
                )
            else:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                    training=training,
                )

            predictions_class.append(outputs_class)

            if self.pe_layer_from_config == "3D":
                bs, q, t, h, w = outputs_mask.shape
                # bs,q,t,h,w => bs,t,q,h,w => bs*t,q,h,w
                outputs_mask = outputs_mask.permute(0, 2, 1, 3, 4).flatten(0, 1)

            predictions_mask.append(outputs_mask)

        return predictions_class, predictions_mask, output

    def propagate(self, output, training, prev_output=[]):

        if training:
            if self.pe_layer_from_config == "3D":
                t_indices = torch.tensor(range(0, output.shape[1]))[1::2]
                t_indices_prev = torch.tensor(range(0, output.shape[1]))[0::2]

            elif self.pe_layer_from_config == "2D":
                t_indices = (
                    torch.tensor(range(0, output.shape[1]))
                    .reshape(-1, self.time_gap)[1::2]
                    .flatten()
                )
                t_indices_prev = (
                    torch.tensor(range(0, output.shape[1]))
                    .reshape(-1, self.time_gap)[0::2]
                    .flatten()
                )

            prev_output = output[:, t_indices_prev]
            new_output = output[:, t_indices]
            output[:, t_indices] = prev_output

        elif prev_output != []:
            output = prev_output

        return output

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, training=True
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        bs_t, d1, d2 = decoder_output.shape
        if training:
            bs = int(bs_t / self.time_gap)
            time_gap = self.time_gap
        else:
            bs = 1

        if self.pe_layer_from_config == "3D":
            outputs_class = self.class_embed(decoder_output)
        elif self.pe_layer_from_config == "2D":
            outputs_class = self.class_embed(
                decoder_output.reshape(bs, self.time_gap, d1, d2)
                .permute(0, 2, 1, 3)
                .reshape(bs, d1, d2 * self.time_gap)
            )
        mask_embed = self.mask_embed(decoder_output)

        if self.pe_layer_from_config == "3D":
            outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
            b, q, _, _, _ = outputs_mask.shape
            # NOTE: prediction is of higher-resolution
            # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]

            attn_mask = (
                F.interpolate(
                    outputs_mask.flatten(0, 1),
                    size=attn_mask_target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .detach()
                .view(b, q, -1, attn_mask_target_size[0], attn_mask_target_size[1])
                .sigmoid()
                < 0.5
            )
            attn_mask2 = attn_mask.clone()

        elif self.pe_layer_from_config == "2D":
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.

        if self.relative_pos:
            # for RelativeMultiHeadAttention
            attn_mask = attn_mask.flatten(2)
        else:
            attn_mask = (
                attn_mask.flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
            )

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


def get_pe_layer(pe_layer, N_steps):
    return PositionEmbeddingSine3D(N_steps, normalize=True)
