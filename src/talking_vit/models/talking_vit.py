from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


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


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
@dataclass
class GPT2Config:
    """Configuration for LookingGPT2.

    Args:
        vocab_size: Vocabulary size.
        context_length: Maximum total sequence length (image tokens + text tokens).
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        dropout: Dropout probability.
        image_size: Height/width of the input image.
        patch_size: Height/width of each image patch.
        in_dim: Number of image input channels.
    """
    vocab_size: int = 50257
    context_length: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    image_size: int = 224
    patch_size: int = 16
    in_dim: int = 3

    @property
    def num_img_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block with pre-LayerNorm and mixed attention.

    Args:
        config: GPT2Config instance.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadMixedAttention(
            d_in=config.embed_dim,
            d_out=config.embed_dim,
            context_length=config.context_length,
            full_attention_length=config.num_img_tokens,
            num_heads=config.num_heads,
            dropout=config.dropout,
            qkv_bias=True,
        )
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = FeedForward(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LookingGPT2(nn.Module):
    """Multimodal GPT-2 model that accepts image + text input.

    Image patches are prepended to the text token sequence and receive full
    attention among themselves, while text tokens use causal attention.
    Pretrained GPT-2 weights can be loaded via ``from_pretrained``.

    Args:
        config: GPT2Config instance.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.multimod_embedder = MultiModEmbedder(
            vocab_size=config.vocab_size,
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
        )
        self.pos_emb = nn.Embedding(config.context_length, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Tie lm_head weights to token embeddings (as in GPT-2)
        self.lm_head.weight = self.multimod_embedder.text_embedder.embedder.weight

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Image tensor of shape (batch_size, in_dim, H, W).
            text_ids: Token IDs of shape (batch_size, text_seq_len).

        Returns:
            Logits of shape (batch_size, text_seq_len, vocab_size).
        """
        img_emb, txt_emb = self.multimod_embedder(images, text_ids)
        x = torch.cat([img_emb, txt_emb], dim=1)

        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = self.drop(x + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Return only the text token logits
        num_img_tokens = img_emb.shape[1]
        img_tokens = self.lm_head(x[:, :num_img_tokens])
        text_tokens = self.lm_head(x[:, num_img_tokens:])
        return img_tokens, text_tokens

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate text tokens conditioned on an image.

        Args:
            images: Image tensor of shape (batch_size, in_dim, H, W).
            prompt_ids: Prompt token IDs of shape (batch_size, prompt_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. Values < 1 make distribution sharper.
            top_k: If set, restrict sampling to the top-k most likely tokens.
            eos_token_id: Stop when every sample in the batch produces this token.

        Returns:
            Token IDs of shape (batch_size, prompt_len + n_generated).
        """
        max_text_len = self.config.context_length - self.config.num_img_tokens
        generated = prompt_ids

        for _ in range(max_new_tokens):
            text_ids = generated[:, -max_text_len:]
            _, text_logits = self(images, text_ids)
            logits = text_logits[:, -1, :]  # (B, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -torch.inf)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2", config: GPT2Config | None = None) -> "LookingGPT2":
        """Load a LookingGPT2 model with pretrained GPT-2 weights.

        Text embeddings, positional embeddings, transformer blocks, and the
        final LayerNorm are loaded from the pretrained checkpoint. The image
        embedder is left randomly initialized.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"gpt2"``, ``"gpt2-medium"``).
            config: Optional GPT2Config override. If None, defaults matching the
                pretrained model are used.

        Returns:
            LookingGPT2 model with pretrained weights.
        """
        from transformers import GPT2Model

        hf_model = GPT2Model.from_pretrained(model_name)
        hf_state = hf_model.state_dict()

        hf_cfg = hf_model.config
        if config is None:
            config = GPT2Config(
                vocab_size=hf_cfg.vocab_size,
                context_length=hf_cfg.n_positions,
                embed_dim=hf_cfg.n_embd,
                num_heads=hf_cfg.n_head,
                num_layers=hf_cfg.n_layer,
            )

        model = cls(config)
        state = model.state_dict()

        # HuggingFace GPT-2 uses Conv1D with weight shape (in, out).
        # nn.Linear weight shape is (out, in) — so we transpose.
        def t(w: torch.Tensor) -> torch.Tensor:
            return w.T.contiguous()

        # Token and position embeddings
        state["multimod_embedder.text_embedder.embedder.weight"] = hf_state["wte.weight"]
        state["pos_emb.weight"] = hf_state["wpe.weight"]

        # Transformer blocks
        for i in range(config.num_layers):
            src = f"h.{i}"
            dst = f"blocks.{i}"

            # LayerNorms
            for ln, src_ln in [("ln_1", "ln_1"), ("ln_2", "ln_2")]:
                state[f"{dst}.{ln}.weight"] = hf_state[f"{src}.{src_ln}.weight"]
                state[f"{dst}.{ln}.bias"] = hf_state[f"{src}.{src_ln}.bias"]

            # Attention: split combined c_attn (in, 3*d) into q, k, v
            c_attn_w = t(hf_state[f"{src}.attn.c_attn.weight"])  # (3*d, d)
            c_attn_b = hf_state[f"{src}.attn.c_attn.bias"]       # (3*d,)
            d = config.embed_dim
            state[f"{dst}.attn.w_query.weight"] = c_attn_w[:d]
            state[f"{dst}.attn.w_key.weight"]   = c_attn_w[d:2*d]
            state[f"{dst}.attn.w_value.weight"] = c_attn_w[2*d:]
            state[f"{dst}.attn.w_query.bias"]   = c_attn_b[:d]
            state[f"{dst}.attn.w_key.bias"]     = c_attn_b[d:2*d]
            state[f"{dst}.attn.w_value.bias"]   = c_attn_b[2*d:]
            state[f"{dst}.attn.out_proj.weight"] = t(hf_state[f"{src}.attn.c_proj.weight"])
            state[f"{dst}.attn.out_proj.bias"]   = hf_state[f"{src}.attn.c_proj.bias"]

            # MLP
            state[f"{dst}.mlp.layers.0.weight"] = t(hf_state[f"{src}.mlp.c_fc.weight"])
            state[f"{dst}.mlp.layers.0.bias"]   = hf_state[f"{src}.mlp.c_fc.bias"]
            state[f"{dst}.mlp.layers.2.weight"] = t(hf_state[f"{src}.mlp.c_proj.weight"])
            state[f"{dst}.mlp.layers.2.bias"]   = hf_state[f"{src}.mlp.c_proj.bias"]

        # Final LayerNorm
        state["ln_f.weight"] = hf_state["ln_f.weight"]
        state["ln_f.bias"]   = hf_state["ln_f.bias"]

        model.load_state_dict(state)
        return model

