import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim = None, 
                       vdim = None, dropout = 0.0, bias = True,
                       self_attention = False, q_noise = 0.0,
                       qn_block_size = 8, batch_first = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else self.embed_dim
        self.vdim = vdim if vdim is not None else self.embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.batch_first = batch_first
        
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert( self.head_dim * num_heads == self.embed_dim )
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention

        assert( self.self_attention )

        self.k_proj = quant_noise(
            nn.linear(self.kdim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.v_proj = quant_noise(
            nn.linear(self.vdim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.q_proj = quant_noise(
            nn.linear(self.embed_dim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.linear(self.embed_dim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.reset_parameters()
        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def forward(self, query, key, value, attn_bias, key_padding_mask, need_weights, 
                      attn_mask, before_softmax, need_head_weights):
        
        if need_head_weights:
            need_weights = True

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = k.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if v is not None:
            v = v.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.reshape(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights