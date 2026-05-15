import torch
from einops import rearrange
from torch import nn


class PatchEmbedder(nn.Module):
    """Image Embedder used to turn an image into patch tokens.

    Args:
        image_size: Height and width of the input image (assumed square).
        patch_size: Height and width of each patch (assumed square).
        in_dim: Number of input channels.
        embed_dim: Dimension of the output patch embeddings.
        add_cls: If True, prepends a learnable [CLS] token to the sequence.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_dim: int = 3,
        embed_dim: int = 768,
        add_cls: bool = False
    ):
        super().__init__()
        self.image_size = image_size
        self.in_dim = in_dim
        self.embeded_dim = embed_dim
        self.patch_size = patch_size

        self.add_cls = add_cls

        self.n_patches = (image_size // patch_size) ** 2

        self.patch_proj = nn.Conv2d(
            self.in_dim, self.embeded_dim, self.patch_size, self.patch_size, bias=False
        )
        if self.add_cls:
            self.cls_token = nn.Parameter(torch.zeros((1, 1, self.embeded_dim)))

        self._init_weights()

    def _init_weights(self):
        """
        Init module weights
        """
        nn.init.trunc_normal_(
            self.patch_proj.weight.data.view(self.embeded_dim, -1), std=0.02
        )

        if self.add_cls:
            nn.init.trunc_normal_(
                self.cls_token, std=0.02
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bz, _, h, w = x.shape

        assert h % self.patch_size == 0 and w % self.patch_size == 0, "Image size must be proportionnal to patch size"

        feat = self.patch_proj(x)
        feat = rearrange(feat, "b d h w -> b (h w) d")

        if self.add_cls:
            cls = self.cls_token.expand(bz, -1, -1)
            feat = torch.cat([cls, feat], dim=1)
        return feat

class TextEmbedder(nn.Module):
    """Text Embedder used to turn token ids to dense vectors.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        embed_dim: Dimension of the output embedding vectors.
    """
    def __init__(self, vocab_size:int, embed_dim: int = 768):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embedder = nn.Embedding(self.vocab_size, self.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.embedder.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.embedder(x)
