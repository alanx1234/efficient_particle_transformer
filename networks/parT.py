''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from networks.logger import _logger


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)
            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)

    def forward(self, x=None, uu=None):
        assert (x is not None) or (uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        x: (seq_len, batch, embed_dim)
        x_cls: (1, batch, embed_dim) optional class token
        padding_mask: (batch, seq_len) True = padded
        """
        if x_cls is not None:
            with torch.no_grad():
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)[0]

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class PatchAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True,
                 patch_size=10, use_patch_messages=True, message_proj=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        self.patch_size = patch_size
        self.use_patch_messages = use_patch_messages

        # ── local intra-patch attention ───────────────────────────────────
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

        # ── hierarchical patch-level attention ────────────────────────────
        if use_patch_messages:
            # no dropout on global stage, matching phatjet
            self.patch_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)
            self.patch_proj = nn.Linear(embed_dim, embed_dim) if message_proj else None

        # ── FFN ───────────────────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

    def _intra_patch_attn(self, x, pm):
        """
        x:  (patch_size, num_patches*N, C)
        pm: (num_patches*N, patch_size) bool or None
        returns: (patch_size, num_patches*N, C)
        """
        residual = x
        xn = self.norm1(x)
        xn, _ = self.attn(xn, xn, xn, key_padding_mask=pm)

        if self.c_attn is not None:
            P = xn.size(0)
            xn = xn.view(P, -1, self.num_heads, self.head_dim)
            xn = torch.einsum('tbhd,h->tbdh', xn, self.c_attn)
            xn = xn.reshape(P, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            xn = self.post_attn_norm(xn)
        xn = self.dropout(xn)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        return xn + residual

    def _patch_message(self, x_4d):
        """
        x_4d: (num_patches, patch_size, N, C)
        returns msg: (num_patches, patch_size, N, C)
        """
        NP, P, N, C = x_4d.shape

        # mean pool over patch -> (NP, N, C), treat N as batch dim for MHA
        patch_tokens = x_4d.mean(dim=1)           # (NP, N, C)

        # MHA expects (seq, batch, embed): seq=NP, batch=N
        pt_out, _ = self.patch_attn(patch_tokens, patch_tokens, patch_tokens)  # (NP, N, C)

        if self.patch_proj is not None:
            pt_out = self.patch_proj(pt_out)       # (NP, N, C)

        # broadcast to all particles in each patch: (NP, 1, N, C) -> (NP, P, N, C)
        msg = pt_out.unsqueeze(1).expand(NP, P, N, C)
        return msg

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None,
                gmp=None, gmp_coords=None):
        """
        x:            (P_orig, N, C)
        padding_mask: (N, P_orig)  True = padded
        gmp:          GeometricMessagePassing module or None
        gmp_coords:   (c1, c2) tuple of (N, P) tensors, pre-computed in ParticleTransformer.forward
        attn_mask:    ignored (no interaction matrix in phat mode)
        x_cls:        ignored (cls_blocks use vanilla Block, not PatchAttentionBlock)
        """
        P_orig, N, C = x.shape
        P = self.patch_size

        # ── 1. GMP at the start of every block (matching PTv3Block) ──────
        if gmp is not None and gmp_coords is not None:
            c1, c2 = gmp_coords
            x_bpc = x.permute(1, 0, 2).contiguous()   # (N, P_orig, C)
            x_bpc = gmp(x_bpc, c1, c2, pad=padding_mask)
            x = x_bpc.permute(1, 0, 2).contiguous()   # (P_orig, N, C)

        # ── 2. pad to multiple of patch_size ─────────────────────────────
        # use local variable so we never mutate the caller's padding_mask
        local_pm = padding_mask
        remainder = P_orig % P
        if remainder != 0:
            pad_len = P - remainder
            x = torch.cat([x, x.new_zeros(pad_len, N, C)], dim=0)
            if local_pm is not None:
                local_pm = torch.cat(
                    [local_pm, local_pm.new_ones(N, pad_len)], dim=1
                )
        P_pad = x.shape[0]
        NP = P_pad // P

        # ── 3. reshape: (P_pad, N, C) -> (NP, P, N, C) -> (P, NP*N, C) ─
        x_4d = x.reshape(NP, P, N, C)                         # (NP, P, N, C)
        x_flat = x_4d.permute(1, 0, 2, 3).reshape(P, NP * N, C)

        # per-patch padding mask: (NP*N, P)
        if local_pm is not None:
            pm = local_pm.view(N, NP, P).permute(1, 0, 2).reshape(NP * N, P)
        else:
            pm = None

        # ── 4. intra-patch attention ──────────────────────────────────────
        x_flat = self._intra_patch_attn(x_flat, pm)           # (P, NP*N, C)

        # reshape back: (NP, P, N, C)
        x_4d = x_flat.reshape(P, NP, N, C).permute(1, 0, 2, 3).contiguous()

        # ── 5. patch message broadcast ────────────────────────────────────
        # matching PTv3Block: norm1 applied before patch_msg, dropout after
        if self.use_patch_messages:
            x_4d_normed = self.norm1(x_4d)
            msg = self._patch_message(x_4d_normed)             # (NP, P, N, C)
            x_4d = x_4d + self.dropout(msg)

        # ── 6. FFN ────────────────────────────────────────────────────────
        # x_4d is (NP, P, N, C); flatten NP*P -> P_pad keeping N,C order
        x = x_4d.contiguous().reshape(P_pad, N, C)

        residual = x
        x = self.norm2(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        # ── 7. strip padding ──────────────────────────────────────────────
        x = x[:P_orig]

        return x


class GeometricMessagePassing(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        grid_size: float = 0.05,
        scatter_reduce: str = "sum",
        max_delta_r: float = 0.8,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert scatter_reduce in ("sum", "mean")
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = float(grid_size)
        self.scatter_reduce = scatter_reduce
        self.max_delta_r = max_delta_r
        self.eps = eps

        self.conv2d = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Linear(channels, channels, bias=True)
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor,
                pad: torch.Tensor | None = None) -> torch.Tensor:
        B, P, C = x.shape
        assert C == self.channels
        residual = x

        if pad is not None:
            c1_for_min = c1.masked_fill(pad, float("inf"))
            c2_for_min = c2.masked_fill(pad, float("inf"))
            c1_min = c1_for_min.min(dim=1, keepdim=True).values
            c2_min = c2_for_min.min(dim=1, keepdim=True).values
            c1_min = torch.where(torch.isfinite(c1_min), c1_min, torch.zeros_like(c1_min))
            c2_min = torch.where(torch.isfinite(c2_min), c2_min, torch.zeros_like(c2_min))
            c1_shift = (c1 - c1_min).masked_fill(pad, 0.0)
            c2_shift = (c2 - c2_min).masked_fill(pad, 0.0)
        else:
            c1_shift = c1 - c1.min(dim=1, keepdim=True).values
            c2_shift = c2 - c2.min(dim=1, keepdim=True).values

        # cap grid dims to max_delta_r/grid_size — fixes H,W at a constant size
        # so conv2d always operates on the same shape, removing dynamic resizing overhead
        max_cells = max(1, int(self.max_delta_r / self.grid_size))
        H, W = max_cells, max_cells

        grid_eta = (c1_shift / self.grid_size).floor().to(torch.long).clamp(0, H - 1)
        grid_phi = (c2_shift / self.grid_size).floor().to(torch.long).clamp(0, W - 1)

        HW = H * W
        b_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, P)
        flat_idx = (b_idx * HW + grid_eta * W + grid_phi).reshape(-1)

        grid_flat = x.new_zeros((B * HW, C))
        grid_flat.scatter_add_(0, flat_idx[:, None].expand(-1, C), x.reshape(-1, C))

        if self.scatter_reduce == "mean":
            ones = x.new_ones((B * P,))
            counts = x.new_zeros((B * HW,))
            counts.scatter_add_(0, flat_idx, ones)
            grid_flat = grid_flat / (counts[:, None] + self.eps)

        grid = grid_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        grid = self.conv2d(grid)

        grid_bhwc = grid.permute(0, 2, 3, 1).contiguous()
        out = grid_bhwc[b_idx, grid_eta, grid_phi]

        out = self.pointwise(out)
        out = self.norm(out)
        return residual + out


def compute_eta_phi_from_p4(v: torch.Tensor, eps: float = 1e-8):
    """v: [N, 4, P] -> eta, phi each [N, P]"""
    px, py, pz = v[:, 0, :], v[:, 1, :], v[:, 2, :]
    pt = torch.sqrt(px * px + py * py + eps)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / (pt + eps))
    return eta, phi


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def unwrap_phi_per_jet(phi: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
    """phi: (B, P); pad: (B, P) True=padded. Returns seam-safe centered dphi."""
    if pad is None:
        sin_mean = torch.sin(phi).mean(dim=1, keepdim=True)
        cos_mean = torch.cos(phi).mean(dim=1, keepdim=True)
    else:
        w = (~pad).to(phi.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        sin_mean = (torch.sin(phi) * w).sum(dim=1, keepdim=True) / denom
        cos_mean = (torch.cos(phi) * w).sum(dim=1, keepdim=True) / denom
    phi0 = torch.atan2(sin_mean, cos_mean)
    return wrap_to_pi(phi - phi0)


class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 # GMP
                 use_gmp=False,
                 gmp_coords="raw",
                 gmp_kernel=3,
                 gmp_grid=0.05,
                 gmp_reduce="sum",
                 gmp_max_delta_r=0.8,
                 # PHAT
                 use_phat=False,
                 phat_patch_size=10,
                 phat_use_patch_messages=True,
                 phat_message_proj=True,
                 phat_gmp_per_block=False,
                 # standard parT GMP per block (independent of phat)
                 gmp_per_block=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.for_inference = for_inference
        self.use_phat = use_phat
        self.use_gmp = use_gmp
        self.gmp_coords = gmp_coords
        self.phat_gmp_per_block = phat_gmp_per_block
        self.gmp_per_block = gmp_per_block

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim

        # GMP: single module, shared (called once per block inside PatchAttentionBlock.forward)
        self.gmp = None
        if use_gmp:
            self.gmp = GeometricMessagePassing(
                channels=embed_dim,
                kernel_size=gmp_kernel,
                grid_size=gmp_grid,
                scatter_reduce=gmp_reduce,
                max_delta_r=gmp_max_delta_r,
            )
        _logger.info(f"GMP: use_gmp={use_gmp}, grid={gmp_grid}, coords={gmp_coords}, kernel={gmp_kernel}, max_delta_r={gmp_max_delta_r}")

        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()

        # pair_embed only constructed (and used) in standard parT mode
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference,
        ) if (not use_phat) and pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None

        if use_phat:
            self.blocks = nn.ModuleList([
                PatchAttentionBlock(
                    **cfg_block,
                    patch_size=phat_patch_size,
                    use_patch_messages=phat_use_patch_messages,
                    message_proj=phat_message_proj,
                ) for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])

        # cls_blocks always use vanilla Block (class-token attention unchanged)
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real=1, padded=0

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P) True=padded

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding: (P, N, C)
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)

            if self.use_phat:
                # compute gmp coords (needed for both upfront and per-block modes)
                gmp_coords = None
                if self.gmp is not None and v is not None:
                    eta, phi = compute_eta_phi_from_p4(v)
                    pad = padding_mask
                    phi_centered = unwrap_phi_per_jet(phi, pad=pad)
                    if self.gmp_coords == "pt":
                        px, py = v[:, 0, :], v[:, 1, :]
                        pt = torch.sqrt(px * px + py * py + 1e-8)
                        c1, c2 = pt * eta, pt * phi_centered
                    else:
                        c1, c2 = eta, phi_centered
                    gmp_coords = (c1, c2)

                if self.gmp is not None and not self.phat_gmp_per_block:
                    # GMP once upfront before block loop (default, matches our existing GMP+ParT)
                    x_bpc = x.permute(1, 0, 2).contiguous()
                    x_bpc = self.gmp(x_bpc, gmp_coords[0], gmp_coords[1], pad=padding_mask)
                    x = x_bpc.permute(1, 0, 2).contiguous()
                    gmp_pass = None   # blocks receive no gmp; coords already baked in
                else:
                    gmp_pass = self.gmp  # blocks will call GMP themselves each time

                for block in self.blocks:
                    x = block(
                        x,
                        padding_mask=padding_mask,
                        gmp=gmp_pass,
                        gmp_coords=gmp_coords,
                    )

            else:
                # standard parT: GMP then global attention + pair_embed
                # pre-compute coords once if gmp is active
                gmp_coords_std = None
                if self.gmp is not None and v is not None:
                    eta, phi = compute_eta_phi_from_p4(v)
                    pad = padding_mask
                    phi_centered = unwrap_phi_per_jet(phi, pad=pad)
                    if self.gmp_coords == "pt":
                        px, py = v[:, 0, :], v[:, 1, :]
                        pt = torch.sqrt(px * px + py * py + 1e-8)
                        c1, c2 = pt * eta, pt * phi_centered
                    else:
                        c1, c2 = eta, phi_centered
                    gmp_coords_std = (c1, c2)

                    if not self.gmp_per_block:
                        # default: GMP once upfront
                        x_bpc = x.permute(1, 0, 2).contiguous()
                        x_bpc = self.gmp(x_bpc, c1, c2, pad=pad)
                        x = x_bpc.permute(1, 0, 2).contiguous()

                attn_mask = None
                if (v is not None or uu is not None) and self.pair_embed is not None:
                    if v is not None:
                        v = v.masked_fill(~mask.expand_as(v), 0.0)
                    attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))

                for block in self.blocks:
                    if self.gmp_per_block and self.gmp is not None and gmp_coords_std is not None:
                        # GMP at the start of every block before attention
                        c1, c2 = gmp_coords_std
                        x_bpc = x.permute(1, 0, 2).contiguous()
                        x_bpc = self.gmp(x_bpc, c1, c2, pad=padding_mask)
                        x = x_bpc.permute(1, 0, 2).contiguous()
                    x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # class token (same for both paths)
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output


class ParticleTransformerTagger(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 use_gmp=False,
                 gmp_coords="raw",
                 gmp_kernel=3,
                 gmp_grid=0.05,
                 gmp_reduce="sum",
                 gmp_max_delta_r=0.8,
                 use_phat=False,
                 phat_patch_size=10,
                 phat_use_patch_messages=True,
                 phat_message_proj=True,
                 phat_gmp_per_block=False,
                 gmp_per_block=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp,
                                        use_gmp=use_gmp,
                                        gmp_coords=gmp_coords,
                                        gmp_kernel=gmp_kernel,
                                        gmp_grid=gmp_grid,
                                        gmp_reduce=gmp_reduce,
                                        gmp_max_delta_r=gmp_max_delta_r,
                                        use_phat=use_phat,
                                        phat_patch_size=phat_patch_size,
                                        phat_use_patch_messages=phat_use_patch_messages,
                                        phat_message_proj=phat_message_proj,
                                        phat_gmp_per_block=phat_gmp_per_block,
                                        gmp_per_block=gmp_per_block)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        with torch.no_grad():
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)
            return self.part(x, v, mask)


class ParticleTransformerTaggerWithExtraPairFeatures(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.for_inference = for_inference

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
        with torch.no_grad():
            if not self.for_inference:
                if pf_uu_idx is not None:
                    pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.size(-1))

            pf_x, pf_v, pf_mask, pf_uu = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)
            uu = torch.zeros(v.size(0), pf_uu.size(1), v.size(2), v.size(2), dtype=v.dtype, device=v.device)
            uu[:, :, :pf_x.size(2), :pf_x.size(2)] = pf_uu

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)
            return self.part(x, v, mask, uu)