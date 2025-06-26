import math

import torch
from diffusers.models.attention_processor import AttnProcessor
from torch import einsum


def _linear1(x, w, b):
    from torch.nn.functional import conv1d
    return conv1d(x.transpose(-2, -1), w.unsqueeze(-1), b).transpose(-2, -1)


class LegacySelfAttnProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states = None, attention_mask = None,
                 temb = None, *args, **kwargs):
        assert encoder_hidden_states is None
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.flatten(2, 3).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        num_query = attn.to_q.weight.size(0)
        num_key = attn.to_k.weight.size(0)
        num_value = attn.to_v.weight.size(0)

        w_qkv = torch.concat((attn.to_q.weight, attn.to_k.weight, attn.to_v.weight))
        b_qkv = torch.concat((attn.to_q.bias, attn.to_k.bias, attn.to_v.bias)) if attn.to_q.bias is not None else None
        query, key, value = _linear1(hidden_states, w_qkv, b_qkv).split_with_sizes((num_query, num_key, num_value), dim=-1)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        num_group = attn.group_norm.num_groups
        scale = 1 / math.sqrt(math.sqrt(num_group))
        weight = torch.einsum("btc,bsc->bts", query * scale, key * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        hidden_states = torch.einsum("bts,bsc->btc", weight, value)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = _linear1(hidden_states, attn.to_out[0].weight, attn.to_out[0].bias)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = residual + hidden_states # keep output stride same to input.

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _linear2(x, w, b):
    from torch.nn.functional import conv2d
    return conv2d(x, w[..., None, None], b)


class LegacyVqvaeAttnProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states = None, attention_mask = None,
                 temb = None, *args, **kwargs):
        assert encoder_hidden_states is None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        assert input_ndim == 4
        # if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        # hidden_states = hidden_states.flatten(2, 3).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)

        query = _linear2(hidden_states, attn.to_q.weight, attn.to_q.bias).flatten(2, 3)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = _linear2(hidden_states, attn.to_k.weight, attn.to_k.bias).flatten(2, 3)
        value = _linear2(hidden_states, attn.to_v.weight, attn.to_v.bias).flatten(2, 3)

        # query = attn.head_to_batch_dim(query)
        # key = attn.head_to_batch_dim(key)
        # value = attn.head_to_batch_dim(value)

        weight = torch.bmm(query.transpose(1, 2), key)
        weight = weight * attn.scale
        weight = torch.softmax(weight, dim=2)
        hidden_states = torch.bmm(value, weight.transpose(1, 2))
        
        hidden_states = hidden_states.reshape(batch_size, channel, height, width)
        # hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = _linear2(hidden_states, attn.to_out[0].weight, attn.to_out[0].bias)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
            # hidden_states = hidden_states.reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = residual + hidden_states # keep output stride same to input.

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LegacyCrossAttnProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states = None, attention_mask = None,
                 temb = None, *args, **kwargs):
        residual = hidden_states

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        weight = einsum("bid,bjd->bij", query, key)
        
        del query
        del key
        
        weight = weight * attn.scale
        weight = weight.softmax(dim=-1)
        hidden_states = einsum("bij,bjd->bid", weight, value)
        
        del value

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = residual + hidden_states # keep output stride same to input.

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
