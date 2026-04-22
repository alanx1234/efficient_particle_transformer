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

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import pyjet
    PYJET_AVAILABLE = True
except ImportError:
    PYJET_AVAILABLE = False


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
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
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

    # the following features are not symmetric for (i, j)
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
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
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
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None, points=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
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
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                    if points is not None:
                        points = torch.gather(points, -1, perm.expand_as(points))
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
                    if points is not None:
                        points = points[:, :, :maxlen]

        return x, v, mask, uu, points


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
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
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
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
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
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
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
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
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
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)

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
class GeometricMessagePassing(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        grid_size: float = 0.05,
        scatter_reduce: str = "sum",   # "sum" or "mean"
        eps: float = 1e-6,
        cluster_mode: str = "grid",
        cluster_k: int = 16,
        cluster_beta: bool = True,
        cluster_combine_mlp: bool = True,
        antikt_R: float = 0.2,
    ):
        super().__init__()
        assert scatter_reduce in ("sum", "mean")
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = float(grid_size)
        self.scatter_reduce = scatter_reduce
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

        self.cluster_mode = cluster_mode
        self.cluster_k = cluster_k
        self.antikt_R = antikt_R
        self.cluster_combine_mlp = cluster_combine_mlp

        if cluster_mode == "kmeans":
            self.centroids = nn.Parameter(torch.randn(cluster_k, 2) * 0.1)
            if cluster_beta:
                self.beta = nn.Parameter(torch.ones(1))
            else:
                self.register_buffer("beta", torch.ones(1))

        elif cluster_mode == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                raise ImportError(
                    "hdbscan package required for cluster_mode='hdbscan'. "
                    "Run: pip install hdbscan"
                )

        elif cluster_mode == "antikt":
            if not PYJET_AVAILABLE:
                raise ImportError(
                    "pyjet package required for cluster_mode='antikt'. "
                    "Run: pip install pyjet"
                )

        if cluster_combine_mlp and cluster_mode != "grid":
            self.cluster_mlp = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.GELU(),
                nn.Linear(channels, channels)
            )

    def forward(
        self,
        x: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        pad: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cluster_mode != "grid":
            return self._forward_cluster(x, c1, c2, pad, v=v)

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
    
        grid_eta = (c1_shift / self.grid_size).floor().to(torch.long)
        grid_phi = (c2_shift / self.grid_size).floor().to(torch.long)
    
        # dynamic grid sizing (no static cap)
        H = int(grid_eta.max().item()) + 1
        W = int(grid_phi.max().item()) + 1
        H = max(H, 1)
        W = max(W, 1)
    
        # --- issue #3 fix: redirect padded particles to scratch cells ---
        # padded particles land at a (H, W) scratch cell that is outside
        # the real [0..H-1, 0..W-1] grid, so they never contaminate real cells
        if pad is not None:
            grid_eta_scatter = torch.where(pad, torch.full_like(grid_eta, H), grid_eta)
            grid_phi_scatter = torch.where(pad, torch.full_like(grid_phi, W), grid_phi)
        else:
            grid_eta_scatter = grid_eta.clamp(0, H - 1)
            grid_phi_scatter = grid_phi.clamp(0, W - 1)
    
        # grid_flat has one extra row and col to absorb the scratch cell
        H_alloc, W_alloc = H + 1, W + 1
        HW_alloc = H_alloc * W_alloc
    
        b_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, P)
        flat_idx = (b_idx * HW_alloc + grid_eta_scatter * W_alloc + grid_phi_scatter).reshape(-1)
    
        grid_flat = x.new_zeros((B * HW_alloc, C))
        grid_flat.scatter_add_(0, flat_idx[:, None].expand(-1, C), x.reshape(-1, C))
    
        # --- issue #4 fix: exclude padded particles from mean count ---
        if self.scatter_reduce == "mean":
            real = (~pad).to(x.dtype).reshape(-1) if pad is not None else x.new_ones(B * P)
            counts = x.new_zeros((B * HW_alloc,))
            counts.scatter_add_(0, flat_idx, real)
            grid_flat = grid_flat / (counts[:, None] + self.eps)
    
        # slice off the scratch row/col before conv — conv only sees real cells
        grid = grid_flat.view(B, H_alloc, W_alloc, C)[:, :H, :W, :]
        grid = grid.permute(0, 3, 1, 2).contiguous()
        grid = self.conv2d(grid)
    
        # gather back using the original (unclamped) real indices
        grid_bhwc = grid.permute(0, 2, 3, 1).contiguous()
        grid_eta_real = grid_eta.clamp(0, H - 1)
        grid_phi_real = grid_phi.clamp(0, W - 1)
        out = grid_bhwc[b_idx, grid_eta_real, grid_phi_real]
    
        out = self.pointwise(out)
        out = self.norm(out)
        return residual + out

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    def _cluster_aggregate_broadcast(self, x, a, residual):
        # x: (B, P, C)  a: (B, P, K)  residual: (B, P, C)
        c = torch.einsum("bpk,bpc->bkc", a, x)  # (B, K, C)
        weights = a.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6  # (B, K, 1)
        c = c / weights
        pool = torch.einsum("bpk,bkc->bpc", a, c)  # (B, P, C)
        if self.cluster_combine_mlp:
            out = self.cluster_mlp(torch.cat([x, pool], dim=-1))
        else:
            out = pool
        out = self.norm(out)
        return residual + out

    def _forward_cluster(self, x, c1, c2, pad, v=None):
        if self.cluster_mode == "kmeans":
            return self._forward_kmeans(x, c1, c2, pad)
        elif self.cluster_mode == "hdbscan":
            return self._forward_hdbscan(x, c1, c2, pad)
        elif self.cluster_mode == "antikt":
            return self._forward_antikt(x, pad, v=v)
        else:
            raise ValueError(f"Unknown cluster_mode: {self.cluster_mode}")

    def _forward_kmeans(self, x, c1, c2, pad):
        residual = x

        coords = torch.stack([c1, c2], dim=-1)  # (B, P, 2)
        diff = coords.unsqueeze(2) - self.centroids.unsqueeze(0).unsqueeze(0)
        dists = (diff ** 2).sum(dim=-1)  # (B, P, K)
        a = torch.softmax(-self.beta * dists, dim=-1)  # (B, P, K)

        if pad is not None:
            a = a.masked_fill(pad.unsqueeze(-1), 0.0)

        return self._cluster_aggregate_broadcast(x, a, residual)

    def _forward_hdbscan(self, x, c1, c2, pad):
        import numpy as np

        B, P, _ = x.shape
        residual = x

        c1_np = c1.detach().cpu().numpy()
        c2_np = c2.detach().cpu().numpy()
        pad_np = pad.cpu().numpy() if pad is not None else None

        a = torch.zeros(B, P, self.cluster_k, dtype=x.dtype, device=x.device)

        for b in range(B):
            if pad_np is not None:
                mask_b = ~pad_np[b]
            else:
                mask_b = np.ones(P, dtype=bool)

            original_indices = np.where(mask_b)[0]
            n_real = len(original_indices)
            if n_real == 0:
                continue

            real_coords = np.stack(
                [c1_np[b][mask_b], c2_np[b][mask_b]], axis=1
            )

            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=max(2, n_real // (self.cluster_k * 2)),
                min_samples=1,
                core_dist_n_jobs=1,
            )
            labels = clusterer.fit_predict(real_coords)

            valid_labels = np.unique(labels[labels >= 0])
            if len(valid_labels) == 0:
                labels = np.zeros_like(labels)
                valid_labels = np.array([0])

            if np.any(labels == -1):
                centroids_np = np.array([
                    real_coords[labels == lbl].mean(axis=0)
                    for lbl in valid_labels
                ])
                noise_idx = np.where(labels == -1)[0]
                for ni in noise_idx:
                    d = np.sum((centroids_np - real_coords[ni]) ** 2, axis=1)
                    nearest_lbl = valid_labels[np.argmin(d)]
                    labels[ni] = nearest_lbl

            unique_labels = np.unique(labels)

            if len(unique_labels) > self.cluster_k:
                counts = np.array([np.sum(labels == lbl) for lbl in unique_labels])
                sorted_labels = unique_labels[np.argsort(-counts)]
                keep_labels = sorted_labels[:self.cluster_k]

                keep_centroids = np.array([
                    real_coords[labels == lbl].mean(axis=0)
                    for lbl in keep_labels
                ])

                for lbl in sorted_labels[self.cluster_k:]:
                    lbl_centroid = real_coords[labels == lbl].mean(axis=0)
                    d = np.sum((keep_centroids - lbl_centroid) ** 2, axis=1)
                    nearest_keep = keep_labels[np.argmin(d)]
                    labels[labels == lbl] = nearest_keep

            final_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(final_labels)}
            remapped = np.array([label_map[lbl] for lbl in labels])

            for p_local, p_orig in enumerate(original_indices):
                cluster_idx = remapped[p_local]
                if cluster_idx < self.cluster_k:
                    a[b, p_orig, cluster_idx] = 1.0

        return self._cluster_aggregate_broadcast(x, a, residual)

    def _forward_antikt(self, x, pad, v=None):
        import numpy as np

        B, P, _ = x.shape
        residual = x

        if v is None:
            raise ValueError("anti-kt clustering requires raw 4-vectors v")

        # v is in original (N, 4, P) layout, as expected by compute_eta_phi_from_p4
        eta_raw, phi_raw = compute_eta_phi_from_p4(v)
        px, py = v[:, 0, :], v[:, 1, :]
        pt = torch.sqrt(px * px + py * py + 1e-8)

        eta_np = eta_raw.detach().cpu().numpy()
        phi_np = phi_raw.detach().cpu().numpy()
        pt_np = pt.detach().cpu().numpy()
        pad_np = pad.cpu().numpy() if pad is not None else None

        a = torch.zeros(B, P, self.cluster_k, dtype=x.dtype, device=x.device)

        for b in range(B):
            if pad_np is not None:
                mask_b = ~pad_np[b]
            else:
                mask_b = np.ones(P, dtype=bool)

            original_indices = np.where(mask_b)[0]
            n_real = len(original_indices)
            if n_real == 0:
                continue

            pseudojets = np.zeros(
                n_real,
                dtype=np.dtype([
                    ("pt", "f8"),
                    ("eta", "f8"),
                    ("phi", "f8"),
                    ("mass", "f8"),
                ])
            )
            pseudojets["pt"] = pt_np[b][mask_b]
            pseudojets["eta"] = eta_np[b][mask_b]
            pseudojets["phi"] = phi_np[b][mask_b]
            pseudojets["mass"] = 0.0

            R_val = float(np.clip(self.antikt_R, 0.05, 1.0))
            sequence = pyjet.cluster(pseudojets, R=R_val, p=-1)
            subjets = sequence.inclusive_jets(ptmin=0.0)
            subjets_sorted = sorted(subjets, key=lambda j: j.pt, reverse=True)

            if len(subjets_sorted) == 0:
                a[b, original_indices, 0] = 1.0
                continue

            real_eta = eta_np[b][mask_b]
            real_phi = phi_np[b][mask_b]

            kept_subjets = subjets_sorted[:self.cluster_k]
            kept_centroids = np.array([[sj.eta, sj.phi] for sj in kept_subjets])

            local_assign = np.full(n_real, -1, dtype=np.int64)
            # Tracks which local indices have been claimed to prevent double-assignment.
            used_global = set()

            def match_constituent(c_eta, c_phi):
                # pyjet 1.9+ does not expose userindex; match by exact kinematics
                # (constituent eta/phi equals the original input particle's values).
                dphi = np.arctan2(np.sin(real_phi - c_phi), np.cos(real_phi - c_phi))
                dists = (real_eta - c_eta) ** 2 + dphi ** 2
                for taken in used_global:
                    dists[taken] = np.inf
                local_idx = int(np.argmin(dists))
                return local_idx if np.isfinite(dists[local_idx]) else -1

            def assign_subjet(subjet, cluster_idx):
                for constituent in subjet.constituents():
                    local_idx = match_constituent(constituent.eta, constituent.phi)
                    if local_idx >= 0 and local_idx not in used_global:
                        local_assign[local_idx] = cluster_idx
                        used_global.add(local_idx)

            for subjet_idx, subjet in enumerate(kept_subjets):
                assign_subjet(subjet, subjet_idx)

            for subjet in subjets_sorted[self.cluster_k:]:
                subjet_eta, subjet_phi = subjet.eta, subjet.phi
                dphi = np.arctan2(np.sin(kept_centroids[:, 1] - subjet_phi),
                                  np.cos(kept_centroids[:, 1] - subjet_phi))
                dists = (kept_centroids[:, 0] - subjet_eta) ** 2 + dphi ** 2
                nearest_keep = int(np.argmin(dists))
                assign_subjet(subjet, nearest_keep)

            # Final fallback: any still-unassigned particle goes to its nearest kept subjet.
            if np.any(local_assign < 0):
                for local_idx in np.where(local_assign < 0)[0]:
                    dphi = np.arctan2(np.sin(kept_centroids[:, 1] - real_phi[local_idx]),
                                      np.cos(kept_centroids[:, 1] - real_phi[local_idx]))
                    dists = (kept_centroids[:, 0] - real_eta[local_idx]) ** 2 + dphi ** 2
                    local_assign[local_idx] = int(np.argmin(dists))

            for p_local, p_orig in enumerate(original_indices):
                a[b, p_orig, local_assign[p_local]] = 1.0

        return self._cluster_aggregate_broadcast(x, a, residual)


def compute_eta_phi_from_p4(v: torch.Tensor, eps: float = 1e-8):
    """
    v: [N, 4, P] with [px, py, pz, E]
    returns eta, phi: [N, P], [N, P]
    """
    px, py, pz = v[:, 0, :], v[:, 1, :], v[:, 2, :]
    pt = torch.sqrt(px * px + py * py + eps)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / (pt + eps))
    return eta, phi

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

def unwrap_phi_per_jet(phi: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
    """
    phi: (B, P) in radians
    pad: (B, P) bool, True where padded
    returns: dphi in (-pi, pi], centered per jet so seam is not a problem
    """
    if pad is None:
        sin_mean = torch.sin(phi).mean(dim=1, keepdim=True)
        cos_mean = torch.cos(phi).mean(dim=1, keepdim=True)
    else:
        w = (~pad).to(phi.dtype)  # 1 for real, 0 for padded
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        sin_mean = (torch.sin(phi) * w).sum(dim=1, keepdim=True) / denom
        cos_mean = (torch.cos(phi) * w).sum(dim=1, keepdim=True) / denom

    phi0 = torch.atan2(sin_mean, cos_mean)          # (B,1) circular mean direction
    dphi = wrap_to_pi(phi - phi0)                   # (B,P) now seam-safe
    return dphi

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
                 gmp_coords = "raw",
                 use_gmp = False,
                 gmp_kernel = 3,
                 gmp_grid = 0.2,
                 gmp_reduce = "sum",
                 gmp_cluster = "grid",
                 gmp_k = 16,
                 gmp_cluster_beta = True,
                 gmp_cluster_combine_mlp = True,
                 antikt_R = 0.2,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim

        self.gmp_coords = gmp_coords
        self.use_gmp = use_gmp
        self.gmp = None
        if self.use_gmp:
            self.gmp = GeometricMessagePassing(
                channels=embed_dim,
                kernel_size=gmp_kernel,
                grid_size=gmp_grid,
                scatter_reduce=gmp_reduce,
                cluster_mode=gmp_cluster,
                cluster_k=gmp_k,
                cluster_beta=gmp_cluster_beta,
                cluster_combine_mlp=gmp_cluster_combine_mlp,
                antikt_R=antikt_R,
            )

        _logger.info(
            f"GMP ENABLED: use_gmp={use_gmp}, grid={gmp_grid}, "
            f"coords={gmp_coords}, kernel={gmp_kernel}, "
            f"cluster={gmp_cluster}, K={gmp_k}, "
            f"antikt_R={antikt_R}, combine_mlp={gmp_cluster_combine_mlp}"
        )

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
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
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

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, points=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu, points = self.trimmer(x, v, mask, uu, points=points)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)

            if self.gmp is not None and (v is not None):
                x_bpc = x.permute(1, 0, 2).contiguous()
                pad = ~mask.squeeze(1)
            
                if self.gmp_coords == "relative" and points is not None:
                    c1 = points[:, 0, :]  # part_deta (N, P)
                    c2 = points[:, 1, :]  # part_dphi (N, P)
                else:
                    eta, phi = compute_eta_phi_from_p4(v)
                    phi_centered = unwrap_phi_per_jet(phi, pad=pad)
                    if self.gmp_coords == "pt":
                        px, py = v[:, 0, :], v[:, 1, :]
                        pt = torch.sqrt(px*px + py*py + 1e-8)
                        c1, c2 = pt * eta, pt * phi_centered
                    else:
                        c1, c2 = eta, phi_centered
            
                x_bpc = self.gmp(x_bpc, c1, c2, pad=pad, v=v)
                x = x_bpc.permute(1, 0, 2).contiguous() # back to (P,N,C)
            
            if v is not None and self.pair_embed is not None:
                v = v.masked_fill(~mask.expand_as(v), 0.0)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


class ParticleTransformerTagger(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
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
                 use_gmp = False,
                 gmp_coords = "raw",
                 gmp_kernel = 3,
                 gmp_grid = 0.05,
                 gmp_reduce = "sum",
                 gmp_cluster = "grid",
                 gmp_k = 16,
                 gmp_cluster_beta = True,
                 gmp_cluster_combine_mlp = True,
                 antikt_R = 0.2,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if gmp_cluster != "grid" and gmp_coords == "relative":
            raise ValueError(
                "gmp_coords='relative' requires `points` (deta/dphi) to be passed into "
                "ParticleTransformer.forward, but ParticleTransformerTagger never passes "
                "points. Use gmp_coords='raw' for clustering inside the tagger."
            )

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        # network configurations
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
                                        # misc
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp,
                                        use_gmp=use_gmp,
                                        gmp_coords=gmp_coords,
                                        gmp_kernel=gmp_kernel,
                                        gmp_grid=gmp_grid,
                                        gmp_reduce=gmp_reduce,
                                        gmp_cluster=gmp_cluster,
                                        gmp_k=gmp_k,
                                        gmp_cluster_beta=gmp_cluster_beta,
                                        gmp_cluster_combine_mlp=gmp_cluster_combine_mlp,
                                        antikt_R=antikt_R,)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            pf_x, pf_v, pf_mask, _, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask)


class ParticleTransformerTaggerWithExtraPairFeatures(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
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
                                        # network configurations
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
                                        # misc
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            if not self.for_inference:
                if pf_uu_idx is not None:
                    pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.size(-1))

            pf_x, pf_v, pf_mask, pf_uu, _ = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
            sv_x, sv_v, sv_mask, _, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)
            uu = torch.zeros(v.size(0), pf_uu.size(1), v.size(2), v.size(2), dtype=v.dtype, device=v.device)
            uu[:, :, :pf_x.size(2), :pf_x.size(2)] = pf_uu

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask, uu)
