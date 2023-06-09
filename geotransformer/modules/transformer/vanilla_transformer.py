r"""Vanilla Transformer without positional embeddings.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput
import geotransformer.modules.transformer.utils_epn.anchors as L
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),'../e2pn','vgtk') )
import vgtk.so3conv as sptk
import vgtk.functional as fr

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores

class MultiHeadAttentionEQ(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, attn_mode=None, alternative_impl=False, kanchor=4):
        """
        The equivariant attention has four steps. 
            1. calculate the local attention matrix (per-point-pair, per-anchor-pair inner products);
            2. calculate the global rotation-wise (r_soft or r_best) or anchor-wise (a_soft or a_best) attention matrix, shared across the whole point cloud;
            3. use the global attention to weight (a_soft or r_soft) or select the best dim in (a_best or r_best) local attention;
            4. calculate the attended feature by multiplying the attention matrix with value feature. 

        attn_mode: when calculating global attention, how to attend query anchors to key anchors
            'a_soft': each anchor in query attends to each anchor in key;
            'a_best': each anchor in query finds the best-matchng anchor in key;
            'r_soft': anchors in key are permuted according to each rotation, each rotation is attended by query;
            'r_best': find the best-matching rotation (permutation of anchors) in key and attend by query;
            None: do not calculate global attention. Skip step 2 and 3 above. 
        attn_r_positive: when calculating global attention, how to make non-negative attention value before normalization, needed if attn_mode is soft
            'sq': square on the attention value;
            'abs': take absolute on the attention value;
            'relu': take relu on the attention value;
            'sigmoid': take sigmoid on the attention value;
            None: do not make the attention value non-negative.
        attn_r_summ: when calculating global attention, how to pool over all points
            'mean': average over all point pairs; 
            'double_norm': double normalization on rows and columns of point pair matrix, pick the topk matching pairs and take the average;
        attn_on_sub: if attn_mode == 'r_soft' or 'r_best', optional to use a subset of anchors to determine the best rotation or weight of each rotation
            True: only attend first two anchors of the query point cloud with key point cloud; 
            False: use all anchors of the query point cloud for attention.
        attn_r_multihead: use the same or different global attention for different attention heads.
            True: attention on different heads separately; 
            False: attention averaged over different heads.
        alternative_impl: different implementations of gathering which are equivalent
            True: use torch.gather, might consume more memory;
            False: use flat indexing.
        """
        super(MultiHeadAttentionEQ, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        # self.quotient_factor = quotient_factor

        self.attn_on_sub = False
        self.attn_r_summ = 'mean'
        self.attn_r_multihead = False
        self.attn_r_positive = 'sq' # 'sq', 'abs', 'relu', 'sigmoid', None
        self.num_correspondences = 256
        # self.attn_r_soft = False
        # self.attn_ra_soft = False
        self.attn_mode = attn_mode   # 'r_best' 'r_soft' 'a_best' 'a_soft'
        self.alternative_impl = alternative_impl
        self.kanchor = kanchor

        # # obselete (only used in cross_anchor_attn_ra)
        # self.attn_r_soft = False
        # self.attn_ra_soft = False

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

        self.init_anchors()
    def init_anchors(self):
        if self.kanchor == 12:
            # anchors = L.get_anchorsV()
            vs, v_adjs, v_level2s, v_opps, vRs = L.get_icosahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 60*3*3
            trace_idx_ori, trace_idx_rot = L.get_relativeV_index()  # 60*12, 60*12 (ra)
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
        elif self.kanchor == 6:
            vs, v_adjs, vRs, ecs, face_normals = L.get_octahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 12*3*3
            trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
        elif self.kanchor == 4:
            vs, v_adjs, vRs, ecs, face_normals = sptk.get_tetrahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 12*3*3
            trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
            """
            trace_idx_ori
            [[0 1 2 3]                                                                                            
            [0 2 3 1]                                                                                             
            [0 3 1 2]                                                                                             
            [1 2 0 3]
            [1 0 3 2]
            [1 3 2 0]
            [2 3 0 1]
            [2 0 1 3]
            [2 1 3 0]
            [3 1 0 2]
            [3 0 2 1]
            [3 2 1 0]]
            trace_idx_rot
            [[0 1 2 3]
            [0 3 1 2]
            [0 2 3 1]
            [2 0 1 3]
            [1 0 3 2]
            [3 0 2 1]
            [2 3 0 1]
            [1 2 0 3]
            [3 1 0 2]
            [2 1 3 0]
            [1 3 2 0]
            [3 2 1 0]]
            """
        elif self.kanchor == 3:            
            self.anchors = nn.Parameter(torch.tensor(L.get_anchors(self.kanchor), dtype=torch.float32), requires_grad=False)
            trace_idx_ori, trace_idx_rot = fr.get_relativeR_index(self.anchors)  
            trace_idx_rot = trace_idx_rot.transpose(0,1)
            # trace_idx_ori
            # [[0 1 2]
            # [1 2 0]
            # [2 0 1]]
            # trace_idx_rot
            # [[0 2 1]
            # [1 0 2]
            # [2 1 0]]
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
        else:
            raise NotImplementedError(f'self.kanchor={self.kanchor} not implemented')
        return

    def cross_anchor_attn_aa(self, q, k):
        '''q: bahnc, k: behmc -> 
        attention_scores: baehnm / brahnm / bahnm
        where: b: batch size; 
        a and e: number of anchors; 
        h: number of heads in the multi-head attention;
        n and m: number of points;
        return: local attention matrix and global attention matrix.
        '''
        b = q.shape[0]
        n = q.shape[-2]
        m = k.shape[-2]
        ####################
        ### local attention
        ####################
        if self.attn_on_sub:
            assert self.attn_mode == 'r_soft' or self.attn_mode == 'r_best', self.attn_mode
            attention_scores_ae = torch.einsum('bahnc,behmc->baehnm', q[:,[0,self.adj0]], k) \
                                    / self.d_model_per_head ** 0.5
        else:
            attention_scores_ae = torch.einsum('bahnc,behmc->baehnm', q, k) \
                                / self.d_model_per_head ** 0.5
        
        if self.attn_mode is None:
            return attention_scores_ae, None, None

        attention_scores_ae_raw = attention_scores_ae   # local attention
        # attention_scores_ae = torch.sigmoid(attention_scores_ae)   # 0-1

        ####################
        ### global attention
        ####################
        if not self.attn_r_multihead:
            ### average over attention heads
            attention_scores_ae = attention_scores_ae.mean(3) # baenm
        ### make the attention value non-negative for rotation/anchor normalization
        if self.attn_r_positive == 'sq':
            attention_scores_ae = attention_scores_ae**2
        elif self.attn_r_positive == 'abs':
            attention_scores_ae = torch.abs(attention_scores_ae)
        elif self.attn_r_positive == 'relu':
            attention_scores_ae = F.relu(attention_scores_ae)
        elif self.attn_r_positive == 'sigmoid':
            attention_scores_ae = F.sigmoid(attention_scores_ae)

        ### pool over all points
        if self.attn_r_summ == 'mean':
            attn_ae = attention_scores_ae.mean([-2,-1])   # bae(h)
        elif self.attn_r_summ == 'double_norm':
            ref_matching_scores = attention_scores_ae / attention_scores_ae.sum(dim=-1, keepdim=True)
            src_matching_scores = attention_scores_ae / attention_scores_ae.sum(dim=-2, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
            num_correspondences = min(self.num_correspondences, matching_scores.numel())
            corr_scores, corr_indices = matching_scores.flatten(-2).topk(k=num_correspondences, largest=True)    # bae(h)k
            attn_ae = corr_scores.mean(-1)          # bae(h)
        else:
            raise NotImplementedError(f'attn_r_summ ={self.attn_r_summ} not recognized')
        
        if self.attn_mode == 'a_soft':
            ### calculate the global anchor attention weight
            assert self.attn_r_positive is not None, "Normalization should be conducted on non-negative weights."
            ### normalize over anchors in key 
            attn_ae = attn_ae / attn_ae.sum(2, keepdim=True)    # bae(h)
            attn_ae = attn_ae[..., None, None]
            if not self.attn_r_multihead:
                attn_ae = attn_ae.unsqueeze(3)    # baehnm
            attention_scores = attention_scores_ae_raw   # baehnm

            return attention_scores, attn_ae
            ### attention_scores: local attention matrix; attn_ae: global anchor attention matrix
        elif self.attn_mode == 'a_best':
            ### pick the optimal anchor in key for each anchor in query
            attn_a_max, attn_a_max_idx = attn_ae.max(dim=2, keepdim=True) # ba1(h)
            if self.alternative_impl:
                if self.attn_r_multihead:
                    attn_a_max_idx = attn_a_max_idx[..., None, None].expand(-1, -1, -1, -1, n, m)  # baehnm
                else:
                    attn_a_max_idx = attn_a_max_idx[..., None, None, None].expand(-1, -1, -1, self.num_heads, n, m)  # baehnm
                attention_scores = torch.gather(attention_scores_ae_raw, 2, attn_a_max_idx) # ba1hnm
                attention_scores = attention_scores.squeeze(2)  # bahnm
            else:
                if self.attn_r_multihead:
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1,1) # b,1,1,1
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1,1) # b,1,1,1
                    attn_ae_idx = lin_idx_b * self.na + lin_idx_a
                    attn_ae_idx = attn_ae_idx * self.na + attn_a_max_idx
                    lin_idx_h = torch.arange(self.num_heads, device=q.device).reshape(1,1,1,-1) # b,1,1,1
                    attn_ae_idx = attn_ae_idx * self.num_heads + lin_idx_h  # bae(=1)h
                    attention_scores = attention_scores_ae_raw.flatten(0,3)[attn_ae_idx].squeeze(2) # baehnm -> bahnm
                else:
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1) # b,1,1
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1) # b,1,1
                    attn_ae_idx = lin_idx_b * self.na + lin_idx_a
                    attn_ae_idx = attn_ae_idx * self.na + attn_a_max_idx  # bae(=1)
                    attention_scores = attention_scores_ae_raw.flatten(0,2)[attn_ae_idx].squeeze(2) # baehnm -> bahnm

            return attention_scores, attn_a_max_idx
            ### attention_scores: local attention matrix; attn_ae: global best matching anchor indices
        elif self.attn_mode == 'r_soft': 
            ### calculate the global rotation attention weight
            assert self.attn_r_positive is not None, "Normalization should be conducted on non-negative weights."
            if self.alternative_impl:
                if self.attn_on_sub:
                    trace_idx_ori = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
                else:
                    trace_idx_ori = self.trace_idx_ori
                attn_are = attn_ae[:,:,trace_idx_ori] # bae(h) -> bare(h)
                if self.attn_r_multihead:
                    attn_r = torch.einsum('barah -> brh', attn_are)
                    ### normalize over rotations in key 
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r.unsqueeze(2)[..., None, None]   # brahnm
                else:
                    attn_r = torch.einsum('bara -> br', attn_are)
                    ### normalize over rotations in key 
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r[..., None, None, None, None]    # brahnm

                # print('attn_r', attn_r.squeeze())
                k_permute = k[:, self.trace_idx_ori]    # behmc -> brahmc
                attention_scores = torch.einsum('bahnc, brahmc -> brahnm', q, k_permute) \
                                    / self.d_model_per_head ** 0.5
                attn_r_new = attention_scores.mean((2,3,4,5))   # br
                # attn_r_new = torch.sigmoid(attention_scores).mean((2,3,4,5))   # br
                attn_r_new = attn_r_new / attn_r_new.sum(1, keepdim=True)
                # print('attn_r_new', attn_r_new)
                # print('---------------')

            else:
                if self.attn_on_sub:
                    trace_idx_ori2 = self.trace_idx_ori[:, [0,self.adj0]]   # 60*2
                    trace_idx_ori2 = trace_idx_ori2.transpose(0,1)[None]    # 1*2*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(2, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * 2 + lin_idx_a
                    idx = idx * self.na + trace_idx_ori2
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r.unsqueeze(2)[..., None, None]   # bra(h)nm
                    if not self.attn_r_multihead:
                        attn_r = attn_r.unsqueeze(3)    # brahnm

                    attention_scores_ae = torch.einsum('bahnc,behmc->baehnm', q, k) \
                                        / self.d_model_per_head ** 0.5
                    trace_idx_ori = self.trace_idx_ori.transpose(0,1)[None]  # 1*12*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * self.na + lin_idx_a
                    idx = idx * self.na + trace_idx_ori     # bar
                    attention_scores = attention_scores_ae.flatten(0,2)[idx]    # barhnm
                    attention_scores = attention_scores.transpose(1,2)          # brahnm
                else:
                    # attn_are = attn_ae[:,:,self.trace_idx_ori] # bare(h)
                    trace_idx_ori = self.trace_idx_ori.transpose(0,1)[None]  # 1*12*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * self.na + lin_idx_a
                    idx = idx * self.na + trace_idx_ori     # bar
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r.unsqueeze(2)[..., None, None]   # bra(h)nm
                    if not self.attn_r_multihead:
                        attn_r = attn_r.unsqueeze(3)    # brahnm
                    
                    attention_scores = attention_scores_ae_raw.flatten(0,2)[idx]   # baehnm -> barhnm
                    attention_scores = attention_scores.transpose(1,2)             # brahnm

            return attention_scores, [attn_r, attn_ae]
            ### attention_scores: local attention matrix; attn_ae: global rotation attention matrix
        elif self.attn_mode == 'r_best':
            ### pick the optimal rotation in key and permute key anchors accordingly
            if self.alternative_impl:
                if self.attn_on_sub:
                    trace_idx_ori = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
                else:
                    trace_idx_ori = self.trace_idx_ori
                attn_are = attn_ae[:,:,trace_idx_ori] # bae(h) -> bare(h)
                if self.attn_r_multihead:
                    attn_r = torch.einsum('barah -> brh', attn_are)
                else:
                    attn_r = torch.einsum('bara -> br', attn_are)
                attn_r_max, attn_r_max_idx = attn_r.max(dim=1) # b(h)
                trace_idx_ori = self.trace_idx_ori[attn_r_max_idx]  # ra -> b(h)a
                if self.attn_r_multihead:
                    trace_idx_ori = trace_idx_ori.transpose(1,2)[..., None, None].expand_as(k)    # bahmc
                else:
                    trace_idx_ori = trace_idx_ori[..., None, None, None].expand_as(k)        # bahmc
                k_permute = torch.gather(k, 1, trace_idx_ori)
                attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                        / self.d_model_per_head ** 0.5
            else:
                if self.attn_on_sub:
                    trace_idx_ori2 = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
                    # attn_are = attn_ae[:,:,trace_idx_ori2] # bae(h) -> bare(h)
                    trace_idx_ori2 = trace_idx_ori2.transpose(0,1)[None]  # 1*2*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(2, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * 2 + lin_idx_a
                    idx = idx * self.na + trace_idx_ori2
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                else:
                    trace_idx_ori = self.trace_idx_ori.transpose(0,1)[None]  # 1*12*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * self.na + lin_idx_a
                    idx = idx * self.na + trace_idx_ori     # bar
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                # attn_r = attn_are.mean((1,3))   # br(h)
                attn_r_max, attn_r_max_idx = attn_r.max(dim=1) # b(h)
                trace_idx_ori = self.trace_idx_ori[attn_r_max_idx]  # ra -> b(h)a
                if self.attn_r_multihead:
                    trace_idx_ori = trace_idx_ori.transpose(1,2)    #bah
                    lin_idx_b = torch.arange(b).reshape(-1,1,1) # b,1,1
                    trace_idx_ori = lin_idx_b * self.na + trace_idx_ori   # b,a,h
                    lin_idx_h = torch.arange(self.num_heads).reshape(1,1,-1)
                    trace_idx_ori = trace_idx_ori * self.num_heads + lin_idx_h
                    k_permute = k.flatten(0,2)[trace_idx_ori] #bahmc
                    attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                            / self.d_model_per_head ** 0.5
                else:
                    lin_idx_b = torch.arange(b, device=trace_idx_ori.device).reshape(-1,1) # b,1
                    trace_idx_ori = lin_idx_b * self.na + trace_idx_ori   # b,a
                    k_permute = k.flatten(0,1)[trace_idx_ori] #bahmc
                    attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                            / self.d_model_per_head ** 0.5

            return attention_scores, trace_idx_ori
            ### attention_scores: local attention matrix; trace_idx_ori: anchor permutation unber the global best matching rotation
        else:
            raise NotImplementedError(f"self.attn_mode={self.attn_mode} not recognized")

    def cross_anchor_attn_ra(self, q, k):
        '''q: bahnc, k: bahmc -> bahnm'''
        ### try all permutations corresponding to rotations, and calculate attention

        if self.attn_on_sub:
            trace_idx_ori2 = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
            k2_permute = k[:,trace_idx_ori2]   # b r a(=2) h m c
            attention_scores_ra = torch.einsum('bahnc,brahmc->brahnm', q[:,[0,self.adj0]], k2_permute) \
                                    / self.d_model_per_head ** 0.5
        else:
            k_permute = k[:,self.trace_idx_ori]   # b r a h m c
            attention_scores_ra = torch.einsum('bahnc,brahmc->brahnm', q, k_permute) \
                                    / self.d_model_per_head ** 0.5

        attention_scores_ra_raw = attention_scores_ra

        attention_scores_ra = torch.sigmoid(attention_scores_ra)   # 0-1

        ### pool spatial-wise
        if not self.attn_r_multihead:
            attention_scores_ra = attention_scores_ra.mean(3) # branm
        if self.attn_r_positive == 'sq':
            attention_scores_ra = attention_scores_ra**2    # make sure not negative before this
        elif self.attn_r_positive == 'abs':
            attention_scores_ra = torch.abs(attention_scores_ra)    # make sure not negative before this
        elif self.attn_r_positive == 'relu':
            attention_scores_ra = F.relu(attention_scores_ra)
        elif self.attn_r_positive == 'sigmoid':
            attention_scores_ra = F.sigmoid(attention_scores_ra)

        if self.attn_r_summ == 'mean':
            attn_ra = attention_scores_ra.mean([-2,-1])   # bra(h)
        elif self.attn_r_summ == 'double_norm':
            ref_matching_scores = attention_scores_ra / attention_scores_ra.sum(dim=-1, keepdim=True)
            src_matching_scores = attention_scores_ra / attention_scores_ra.sum(dim=-2, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
            num_correspondences = min(self.num_correspondences, matching_scores.numel())
            corr_scores, corr_indices = matching_scores.flatten(-2).topk(k=num_correspondences, largest=True)    # bra(h)k
            attn_ra = corr_scores.mean(-1)          # bra(h)
        else:
            raise NotImplementedError(f'attn_r_summ ={self.attn_r_summ} not recognized')

        ### pick the optimal rotation and permute k accordingly
        if self.attn_r_soft:
            attn_ra = attn_ra / attn_ra.sum(1, keepdim=True)     #bra(h)
            if self.attn_on_sub:
                pass
            else:
                if not self.attn_ra_soft:
                    attn_ra = attn_ra.mean(2, keepdim=True)   # br1(h)
                if not self.attn_r_multihead:
                    attn_ra_weight = attn_ra_weight[..., None]  #brah

                attn_ra_weight = attn_ra[..., None,None]    #brah -> brahnm
                attention_scores = (attention_scores_ra_raw * attn_ra_weight).sum(1)    # bahnm
                    
        else:
            attn_r = attn_ra.mean(2)   # br(h)
            attn_r_max, attn_r_max_idx = attn_r.max(dim=1) # b(h)
            ### calculate final attention
            if self.attn_on_sub:
                ### need to recalculate attention (initial attn is only a subset)
                ### first permute k
                trace_idx_ori = self.trace_idx_ori[attn_r_max_idx]  # ra -> b(h)a
                b = q.shape[0]
                n = q.shape[3]
                m = k.shape[3]
                if self.attn_r_multihead:
                    trace_idx_ori = trace_idx_ori.transpose(1,2)    #bah
                    lin_idx = torch.arange(b).reshape(-1,1,1) # b,1,1
                    trace_idx_ori = lin_idx * self.na + trace_idx_ori   # b,a,h

                    lin_idx_h = torch.arange(self.num_heads).reshape(1,1,-1)
                    trace_idx_ori = trace_idx_ori * self.num_heads + lin_idx_h

                    k_permute = k.flatten(0,2)[trace_idx_ori] #bahmc
                else:
                    lin_idx = torch.arange(b).reshape(-1,1) # b,1
                    trace_idx_ori = lin_idx * self.na + trace_idx_ori   # b,a

                    k_permute = k.flatten(0,1)[trace_idx_ori] #bahmc

                ### then calc attn scores
                attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                        / self.d_model_per_head ** 0.5

            else:
                ### gather from initial attention
                if self.attn_r_multihead:
                    attn_r_max_idx = attn_r_max_idx.unsqueeze(1)    # br(=1)h
                    lin_idx_b = torch.arange(b).reshape(-1,1,1) # b,1,1
                    attn_r_max_idx = lin_idx_b * self.nr + attn_r_max_idx   # b,r,h
                    
                    lin_idx_h = torch.arange(self.num_heads).reshape(1,1,-1)
                    attn_r_max_idx = attn_r_max_idx * self.num_heads + lin_idx_h    # b,r,h

                    attention_scores = attention_scores_ra_raw.transpose(2,3).flatten(0, 2 \
                        )[attn_r_max_idx].reshape(b,1,self.num_heads,self.na,n,m).transpose(2,3).squeeze(1)
                        # bahnm
                else:
                    attn_r_max_idx = attn_r_max_idx.unsqueeze(1)    # br(=1)
                    lin_idx_b = torch.arange(b).reshape(-1,1) # b,1
                    attn_r_max_idx = lin_idx_b * self.nr + attn_r_max_idx   # b,r
                    attention_scores = attention_scores_ra_raw.flatten(0,1)[attn_r_max_idx].reshape(b,self.na,self.num_heads,n,m)
                    # bahnm

        return attention_scores

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, A, N, C)
            input_k (Tensor): input tensor for key (B, A, M, C)
            input_v (Tensor): input tensor for value (B, A, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, A, N, C)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, A, A, H, N, M), local attention scores
                'attn_w' ('attn_idx'): global rotation or anchor attention scores (or best indices)
        """
        q = rearrange(self.proj_q(input_q), 'b a n (h c) -> b a h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b a m (h c) -> b a h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b a m (h c) -> b a h m c', h=self.num_heads)

        attention_scores, attn_w = self.cross_anchor_attn_aa(q, k)     
        ### local and global attention matrix, of the same shape: baehnm / brahnm / bahnm
        
        if self.attn_mode in ['a_best', 'r_best']:
            attn_idx = attn_w

        if self.attn_mode in ['a_best', 'r_best']:
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1).unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1).unsqueeze(1), float('-inf'))
            if attention_masks is not None:
                attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        else:   # None, 'a_soft', 'r_soft'
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1).unsqueeze(1).unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1), float('-inf'))
            if attention_masks is not None:
                attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
               

        ### normalize the local attention over the points in key
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        if self.attn_mode == 'a_soft':
            ### attention_scores: baehnm, attn_w: baehnm, v: behmc
            attention_scores = attention_scores * attn_w
            hidden_states = torch.einsum('baehnm,behmc->bahnc', attention_scores, v)
        elif self.attn_mode == 'a_best':
            if self.alternative_impl:
                attn_idx = attn_idx[:,:,0,:,0,:, None].expand_as(v)   # bahmc
                v_permute = torch.gather(v, 1, attn_idx)
            else:
                ### attention_scores: bahnm, attn_idx: bae(=1)(h), v: behmc
                attn_idx = attn_idx.squeeze(2)  # ba(h)
                if self.attn_r_multihead:
                    b = attention_scores.shape[0]
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1) # b,1,1
                    attn_idx = lin_idx_b * self.na + attn_idx
                    lin_idx_h = torch.arange(self.num_heads, device=q.device).reshape(1,1,-1) # b,1,1
                    attn_idx = attn_idx * self.num_heads + lin_idx_h  # bah
                    v_permute = v.flatten(0,2)[attn_idx] # bahmc
                else:
                    b = attention_scores.shape[0]
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1) # b,1
                    attn_idx = lin_idx_b * self.na + attn_idx
                    v_permute = v.flatten(0,1)[attn_idx] # bahmc
            hidden_states = torch.einsum('bahnm,bahmc->bahnc', attention_scores, v_permute)
        elif self.attn_mode == 'r_soft':
            ### brahnm, brahnm, behmc
            attn_w, attn_matrix = attn_w
            attention_scores = attention_scores * attn_w
            v_permute = v[:, self.trace_idx_ori]    # brahmc
            # print('v_permute_0000', v_permute[..., 0,0,0,0])
            hidden_states = torch.einsum('brahnm,brahmc->bahnc', attention_scores, v_permute)
        elif self.attn_mode == 'r_best':
            if self.alternative_impl:
                ### attn_idx: bahmc
                v_permute = torch.gather(v, 1, attn_idx)
            else:
                ### attention_scores: bahnm, attn_idx: ba(h), v: behmc
                if self.attn_r_multihead:
                    v_permute = v.flatten(0,2)[attn_idx]    # bahmc
                else:
                    v_permute = v.flatten(0,1)[attn_idx]    # bahmc
            hidden_states = torch.einsum('bahnm,bahmc->bahnc', attention_scores, v_permute)
        else:
            assert self.attn_mode is None, self.attn_mode
            hidden_states = torch.einsum('baehnm,behmc->bahnc', attention_scores, v)

        # hidden_states = torch.matmul(attention_scores, v)   # bahnm, bahmc -> bahnc

        hidden_states = rearrange(hidden_states, 'b a h n c -> b a n (h c)')

        if self.attn_mode in ['a_best', 'r_best']:
            return hidden_states, [attention_scores, attn_idx]
        elif self.attn_mode in ['r_soft']:
            return hidden_states, [attention_scores, attn_w, attn_matrix]
        else:
            return hidden_states, [attention_scores, attn_w]

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, equivariant=False, attn_mode=None, alternative_impl=False, kanchor=4):
        super(AttentionLayer, self).__init__()
        self.equivariant = equivariant
        if self.equivariant:
            self.attention = MultiHeadAttentionEQ(d_model, num_heads, dropout=dropout, attn_mode=attn_mode, alternative_impl=alternative_impl, kanchor=kanchor)
        else:
            self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', equivariant=False, attn_mode=None, alternative_impl=False, kanchor=4):
        super(TransformerLayer, self).__init__()
        self.equivariant = equivariant
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout, equivariant=equivariant, attn_mode=attn_mode, alternative_impl=alternative_impl, kanchor=kanchor)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.cross_attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(self, input_states, memory_states, input_masks=None, memory_masks=None):
        hidden_states, attention_scores = self.self_attention(input_states, input_states, memory_masks=input_masks)
        hidden_states, attention_scores = self.cross_attention(hidden_states, memory_states, memory_masks=memory_masks)
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=None, activation_fn='ReLU'):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)

    def forward(self, feats, weights=None, masks=None, attention_factors=None, attention_masks=None):
        r"""Transformer Encoder forward.

        Args:
            feats (Tensor): (B, N, C)
            weights (Tensor=None): (B, N)
            masks (BoolTensor=None): True if ignored (B, N)
            attention_factors (Tensor=None): (B, N, N)
            attention_masks (BoolTensor=None): (B, N, N)

        Returns:
            feats (Tensor): (B, N, C)
        """
        for i in range(self.num_layers):
            feats, _ = self.layers[i](
                feats,
                feats,
                memory_weights=weights,
                memory_masks=masks,
                attention_factors=attention_factors,
                attention_masks=attention_masks,
            )
        return feats


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=None, activation_fn='ReLU'):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerDecoderLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)

    def forward(self, q_feats, s_feats):
        r"""Transformer Decoder forward.

        Args:
            q_feats (Tensor): (B, N, C)
            s_feats (Tensor): (B, M, C)

        Returns:
            q_feats (Tensor): (B, N, C)
        """
        for i in range(self.num_layers):
            q_feats, _ = self.layers[i](q_feats, s_feats)
        return q_feats
