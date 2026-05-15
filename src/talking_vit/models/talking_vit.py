import torch
from einops import rearrange
from torch import nn


class PatchEmbedder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_dim: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_dim = in_dim
        self.embeded_dim = embed_dim
        self.patch_size = patch_size

        self.n_patches = (image_size // patch_size) ** 2

        self.patch_proj = nn.Conv2d(
            self.in_dim, self.embeded_dim, self.patch_size, self.patch_size, bias=False
        )

        self._init_weights()

    def _init_weights(self):
        """
        Init module weights
        """
        nn.init.trunc_normal_(
            self.patch_proj.weight.data.view(self.embeded_dim, -1), std=0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_proj(x)

        return rearrange(feat, "b d h w -> b (h w) d")
