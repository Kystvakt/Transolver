import torch
from einops import rearrange, repeat
from torch import nn

from .layers import FlexiPatchEmbed3d, UnPatchEmbed3d, Mamba3dLayer


class Model(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.__name__ = "DMamba"

        # Configuration
        self.input_size = tuple(config.input_size)
        self.patch_size = tuple(config.patch_size)
        self.stride = tuple(config.stride)
        self.channels = config.channels
        self.dim = config.dim
        self.mlp_ratio = config.mlp_ratio
        self.patch_size_seq = (tuple(config.patch_size),)  # Use single patch size
        self.patch_size_probs = None
        self.norm_layer = nn.LayerNorm
        self.depth = config.depth
        self.dropout = config.dropout
        self.dropout_embed = config.dropout_embed
        self.mamba_kw = config.get('mamba_kw', dict())
        self.height = self.input_size[1]
        self.width = self.input_size[2]

        # Patch Embedding
        self.patch_embedding = FlexiPatchEmbed3d(
            input_size=self.input_size,
            patch_size=self.patch_size,
            stride=self.stride,
            channels=self.channels,
            d_embed=self.dim,
            patch_size_seq=self.patch_size_seq,
            patch_size_probs=self.patch_size_probs,
            norm_layer=self.norm_layer,
            bias=True,
        )
        self.t = int((self.input_size[0] - self.patch_size[0]) / self.stride[0] + 1)
        self.h = int((self.input_size[1] - self.patch_size[1]) / self.stride[1] + 1)
        self.w = int((self.input_size[2] - self.patch_size[2]) / self.stride[2] + 1)

        # Backbone
        dims = [self.dim * (1 ** i) for i in range(self.depth + 1)]

        # Encoder layers
        self.encoder = nn.ModuleList(
            nn.Sequential(
                Mamba3dLayer(dim=dims[i], depth=2, attn_drop=self.dropout),
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.SiLU()
            ) for i in range(self.depth)
        )

        # Bottleneck layer
        self.bridge = Mamba3dLayer(dim=dims[-1], depth=9, attn_drop=self.dropout)

        # Decoder layers
        self.decoder = nn.ModuleList(
            nn.Sequential(
                Mamba3dLayer(dim=dims[-(i + 1)], depth=2, attn_drop=self.dropout),
                nn.Linear(dims[-(i + 1)], dims[-(i + 2)]),
                nn.LayerNorm(dims[-(i + 2)]),
                nn.SiLU()
            ) for i in range(self.depth)
        )

        # Other layers
        self.dropout = nn.Dropout(self.dropout_embed)
        self.to_latent = nn.Identity()

        # Head
        self.head = UnPatchEmbed3d(
            patch_size=self.patch_size,
            stride=self.stride,
            channels=self.channels,
            d_embed=self.dim,
            bias=True
        )

    def forward(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Positions.
            fx (torch.Tensor): Input data of shape (B, H*W, C)

        Return:
            torch.Tensor: Output data of shape (B, H*W, C)
        """
        # fx: (B, H*W, T*C)
        fx = torch.cat([x, fx], dim=-1)

        # Patch embeddings
        fx = rearrange(fx, 'b (h w) (t c) -> b c t h w', h=self.height, w=self.width, c=1)
        fx = self.patch_embedding(fx, return_patch_size=False)
        fx = self.dropout(fx)
        fx = rearrange(fx, 'b d t h w -> b t h w d')

        # Encoder
        resid = []
        for layer in self.encoder:
            fx = layer(fx)
            resid.append(fx)

        # Bottleneck
        fx = self.bridge(fx)

        # Decoder
        for i, layer in enumerate(self.decoder):
            fx = fx + resid[-(i + 1)]
            fx = layer(fx)

        fx = self.to_latent(fx)

        # Head
        fx = rearrange(fx, 'b t h w d -> b d t h w')
        new_patch_size = (1, self.patch_size[1], self.patch_size[2])
        new_stride = (1, self.stride[1], self.stride[2])
        fx = self.head(fx, patch_size=new_patch_size, stride=new_stride)  # (B, C, T, H, W)
        fx = fx.squeeze(dim=2)  # no time dim.
        fx = rearrange(fx, 'b c h w -> b h w c')

        return fx
