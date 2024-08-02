from typing import Optional, Union, Sequence, Tuple, Callable
from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat
from mamba_ssm import selective_scan_fn


class FlexiPatchEmbed3d(nn.Module):
    def __init__(
            self,
            input_size: tuple[int, int, int],
            patch_size: tuple[int, int, int],
            stride: tuple[int, int, int],
            channels: int,
            d_embed: int,
            patch_size_seq: Sequence[tuple[int, int, int]],
            patch_size_probs: Optional[Sequence[float]] = None,
            norm_layer: Optional[Union[nn.LayerNorm, nn.BatchNorm3d, nn.Module]] = None,
            bias: bool = True,
    ):
        super().__init__()
        assert len(input_size) == len(patch_size) == len(stride), (
            f"Length of the input size ({input_size}), patch size ({patch_size}) and stride ({stride}) should be equal"
        )
        self.input_size = input_size
        self.patch_size = patch_size
        self.stride = stride
        self.ratio = tuple(float(s / p) for s, p in zip(self.stride, self.patch_size))

        self.proj = nn.Conv3d(
            in_channels=channels,
            out_channels=d_embed,
            kernel_size=patch_size,
            stride=stride,
            bias=bias
        )
        self.norm = norm_layer(d_embed) if norm_layer else nn.Identity()

        self.patch_size_seq = patch_size_seq
        if self.patch_size_seq is not None:
            if patch_size_probs is None:
                n = len(self.patch_size_seq)
                self.patch_size_probs = [1. / n] * n
            else:
                self.patch_size_probs = [p / sum(patch_size_probs) for p in patch_size_probs]
        else:
            self.patch_size_probs = list()
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        pinvs = dict()
        for ps in self.patch_size_seq:
            pinvs[ps] = self._calculate_pinv(self.patch_size, ps)
        return pinvs

    def _calculate_pinv(self, old_shape: tuple, new_shape: tuple) -> torch.Tensor:
        mat = list()
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resized_matrix = torch.stack(mat)
        return torch.linalg.pinv(resized_matrix)

    @staticmethod
    def _resize(x: torch.Tensor, shape: tuple) -> torch.Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            size=shape,
            mode='trilinear',
            antialias=False
        )
        return x_resized[0, 0, ...]

    def resize_patch_embed(self, patch_embed: torch.Tensor, new_patch_size: tuple) -> torch.Tensor:
        if self.patch_size == new_patch_size:
            return patch_embed

        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)

        pinv = self.pinvs[new_patch_size].to(patch_embed.device)

        # inner function
        def resample_patch_embed(patch_embed: torch.Tensor):
            t, h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            resampled_kernel = rearrange(resampled_kernel, '(t h w) -> t h w', t=t, h=h, w=w)
            return resampled_kernel

        v_resampled_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resampled_patch_embed(patch_embed)

    def forward(
            self,
            x: torch.Tensor,
            patch_size: Optional[tuple] = None,
            return_patch_size: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        if patch_size is None and not self.training:
            patch_size = self.patch_size
        elif patch_size is None:
            assert self.patch_size_seq, "No patch size specified during forward and no patch_size_seq given"
            patch_size = self.patch_size_seq[np.random.choice(len(self.patch_size_seq), p=self.patch_size_probs)]

        # Resize
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        stride = tuple(int(p * r) for p, r in zip(patch_size, self.ratio))

        x = F.conv3d(x, weight, bias=self.proj.bias, stride=stride)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        if return_patch_size:
            return x, patch_size
        else:
            return x


class UnPatchEmbed3d(nn.Module):
    def __init__(
            self,
            patch_size: tuple[int, int, int],
            stride: tuple[int, int, int],
            channels: int,
            d_embed: int,
            bias: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose3d(
            in_channels=d_embed,
            out_channels=channels,
            kernel_size=patch_size,
            stride=stride,
            bias=bias
        )

    def _calculate_pinv(self, old_shape: Tuple, new_shape: Tuple) -> torch.Tensor:
        mat = list()
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    @staticmethod
    def _resize(x: torch.Tensor, shape: Tuple) -> torch.Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            size=shape,
            mode='trilinear',
            antialias=False
        )
        return x_resized[0, 0, ...]

    def resize_patch_embed(self, patch_embed: torch.Tensor, new_patch_size: Tuple):
        if self.patch_size == new_patch_size:
            return patch_embed
        pinv = self._calculate_pinv(self.patch_size, new_patch_size).to(patch_embed.device)

        def resample_patch_embed(patch_embed):
            t, h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            resampled_kernel = rearrange(resampled_kernel, '(t h w) -> t h w', t=t, h=h, w=w)
            return resampled_kernel

        v_resample_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)

    def forward(
            self,
            x: torch.Tensor,
            patch_size: Tuple,
            stride: Tuple
    ) -> torch.Tensor:
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        x = F.conv_transpose3d(x, weight, bias=self.proj.bias, stride=stride)

        return x


class SS3D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=6, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=6, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=6, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=6, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=6, D, N)

        self.forward_core = self.forward_core_v0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core_v0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, T, H, W = x.shape
        L = T * H * W
        K = 8

        x_thw_f = x.view(B, -1, L)
        x_twh_f = torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)
        x_hwt_f = x.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, L)
        x_wht_f = x.permute(0, 1, 4, 3, 2).contiguous().view(B, -1, L)
        x_f = torch.stack([x_thw_f, x_twh_f, x_hwt_f, x_wht_f], dim=1).view(B, 4, -1, L)

        xs = torch.cat([x_f, torch.flip(x_f, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y_twh_f = torch.transpose(out_y[:, 1].view(B, -1, T, W, H), dim0=3, dim1=4).contiguous().view(B, -1, L)
        y_hwt_f = out_y[:, 2].view(B, -1, H, W, T).permute(0, 1, 4, 2, 3).contiguous().view(B, -1, L)
        y_wht_f = out_y[:, 3].view(B, -1, W, H, T).permute(0, 1, 4, 3, 2).contiguous().view(B, -1, L)

        inv_y = torch.flip(out_y[:, 4:8], dims=[-1]).view(B, 4, -1, L)
        y_twh_b = torch.transpose(inv_y[:, 1].view(B, -1, T, W, H), dim0=3, dim1=4).contiguous().view(B, -1, L)
        y_hwt_b = inv_y[:, 2].view(B, -1, H, W, T).permute(0, 1, 4, 2, 3).contiguous().view(B, -1, L)
        y_wht_b = inv_y[:, 3].view(B, -1, W, H, T).permute(0, 1, 4, 3, 2).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + y_twh_f + y_twh_b + y_hwt_f + y_hwt_b + y_wht_f + y_wht_b

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, T, H, W, -1)  # (b, t, h, w, d)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_core_v0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, T, H, W = x.shape
        L = T * H * W
        K = 6

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, T, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, t, h, w, d)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, d, t, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Mamba3dBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        # self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.self_attention(self.ln_1(input))
        return x


class Mamba3dLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            d_state=16,
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            Mamba3dBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
