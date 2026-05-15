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
        add_cls: bool = False,
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
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bz, _, h, w = x.shape

        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            "Image size must be proportionnal to patch size"
        )

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

    def __init__(self, vocab_size: int = 50257, embed_dim: int = 768):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embedder = nn.Embedding(self.vocab_size, self.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.embedder.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.embedder(x)


class MultiModEmbedder(nn.Module):
    """Multimodal embedder that combines image and text embeddings.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        image_size: Height and width of the input image.
        patch_size: Height and width of each patch.
        in_dim: Number of input image channels.
        embed_dim: Dimension of the output embeddings.
        add_cls: If True, prepends a learnable [CLS] token to image embeddings.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        image_size: int = 224,
        patch_size: int = 16,
        in_dim: int = 3,
        embed_dim: int = 768,
        add_cls: bool = False
        ,
    ):
        super().__init__()

        self.img_embedder = PatchEmbedder(
            image_size=image_size,
            patch_size=patch_size,
            in_dim=in_dim,
            embed_dim=embed_dim,
            add_cls=add_cls,
        )

        self.text_embedder = TextEmbedder(vocab_size=vocab_size, embed_dim=embed_dim)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Image tensor of shape (batch_size, in_dim, image_size, image_size)
            text_ids: Text token IDs of shape (batch_size, text_seq_len)

        Returns:
            Tuple of:
                - image_embeddings: Shape (batch_size, num_patches[+1], embed_dim)
                - text_embeddings: Shape (batch_size, text_seq_len, embed_dim)
        """
        image_embeddings = self.img_embedder(images)
        text_embeddings = self.text_embedder(text_ids)
        return image_embeddings, text_embeddings


class MultiHeadMixedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, full_attention_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0),  \
        "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        mask[:full_attention_length, :full_attention_length] = 0.
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, _ = x.shape

        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        head_dim = self.d_out // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)

        # Apply mask
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 1, -torch.inf)

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v

        # Combine heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_out)

        # Final projection
        out = self.out_proj(out)

        return out
