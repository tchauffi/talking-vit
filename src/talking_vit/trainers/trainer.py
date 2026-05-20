"""Trainer for LookingGPT2 on image-caption pairs."""

import dataclasses
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import bitsandbytes as bnb

from talking_vit.models.talking_vit import LookingGPT2
from talking_vit.trainers.dataset import HFImageCaptionDataset


@dataclass
class TrainConfig:
    # Data
    dataset_name: str = "laion/220k-GPT4Vision-captions-from-LIVIS"
    dataset_split: str = "train"
    image_col: str = ""
    caption_col: str = "short_caption"
    url_col: str | None = "url"
    prefetch_workers: int = 16
    image_cache_dir: str | None = "~/.cache/talking-vit/images"
    shuffle_buffer: int = 1000
    max_text_len: int = 128
    num_workers: int = 8   # 0 required for IterableDataset + streaming

    # Model
    pretrained: str = "gpt2"
    use_clip_weights: bool = True
    add_cls: bool = True

    # Optimisation
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-3
    backbone_lr: float = 2e-5  # GPT-2 backbone uses a much lower LR so the image path can compete
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    # Freeze the GPT-2 backbone for the first N optimizer steps so the image
    # path is forced to produce useful features before the language model
    # is allowed to adapt. 0 disables.
    freeze_backbone_steps: int = 1000
    grad_clip: float = 1.0
    # Weight of the CLIP-style InfoNCE image↔text contrastive loss added on
    # top of the LM loss. 0 disables. Provides a dense alignment signal that
    # is independent of the language prior — essential for forcing the model
    # to actually use image features.
    contrastive_weight: float = 0.5
    # Contrastive loss formulation. "softmax" = CLIP-style InfoNCE (needs large
    # batch or memory bank). "sigmoid" = SigLIP-style per-pair sigmoid BCE; works
    # at single-GPU batch sizes and does not require a memory bank.
    contrastive_loss_type: str = "sigmoid"
    # If True, derive the *text* contrastive feature from a second forward pass
    # with zeros instead of the real image. Removes image leakage into txt_feat
    # (the joint pass lets text attend to image tokens, which lets the
    # contrastive task be solved by trivial encoder-consistency features).
    # Costs ~2× the forward time per step.
    contrastive_text_only_pass: bool = True
    # FIFO memory bank of past contrastive features used as extra negatives in
    # the InfoNCE softmax (MoCo-style). Enlarges the effective batch for
    # contrastive learning without quadratic memory growth. 0 disables.
    contrastive_memory_bank_size: int = 0
    # Up-weight the LM loss on the *content-bearing* leading tokens. Position 0
    # is typically a stereotyped opener ("A", "The", "An") with no image
    # information, so up-weighting it amplifies language prior. Positions
    # 1..1+content_loss_window are where the noun/object usually sits and is
    # most image-conditional. alpha=1.0 ⇒ no upweighting.
    content_loss_weight: float = 10.0
    content_loss_window: int = 3
    # Apply image augmentation (RandomResizedCrop, HFlip, ColorJitter) on the
    # train stream. Effectively expands the dataset and forces the patch
    # encoder to learn invariances.
    augment: bool = True
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None  # stop early (smoke tests / quick validation)

    # Mixed precision: "no", "fp16", "bf16"
    mixed_precision: str = "fp16"

    # Generation samples logged at each checkpoint
    sample_prompts: tuple[str, ...] = ("The image shows", "In this picture,", "I can see", "")
    sample_max_new_tokens: int = 30
    sample_temperature: float = 0.8
    sample_top_k: int = 50

    # I/O
    run_name: str = ""  # auto-generated from timestamp + config if empty
    log_every: int = 10
    save_every: int = 250
    output_dir: str = "runs/coco"
    # Resume from a previously saved checkpoint directory (e.g.
    # "runs/coco/ckpt_0001000"), or "latest" to pick the highest-step ckpt
    # under ``output_dir``. None starts training from scratch.
    resume_from: str | None = None


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _make_run_name(cfg: "TrainConfig") -> str:
    model_tag = cfg.pretrained.split("/")[-1]
    vision_tag = "clip" if cfg.use_clip_weights else "rand"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_tag}_{vision_tag}_{ts}"


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _denormalize(img: torch.Tensor, use_clip: bool = False) -> torch.Tensor:
    """Reverse normalisation and clamp to [0, 1] for image preview."""
    mean = (_CLIP_MEAN if use_clip else _IMAGENET_MEAN).to(img.device)
    std = (_CLIP_STD if use_clip else _IMAGENET_STD).to(img.device)
    return (img * std + mean).clamp(0.0, 1.0)


def _grad_norm(params) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
    return total_sq ** 0.5


class MemoryBank:
    """FIFO buffer of detached feature vectors used as extra InfoNCE negatives.

    Holds the last ``capacity`` features in fp32 on the target device. Features
    are written without gradients (MoCo-style stale negatives): they enlarge
    the softmax denominator and make the contrastive task harder, but only the
    current batch's features carry gradient back through the encoders.

    Args:
        dim: Feature dimensionality.
        capacity: Maximum number of stored features.
        device: Storage device.
    """

    def __init__(self, dim: int, capacity: int, device) -> None:
        self.capacity = capacity
        self.buffer = torch.zeros(capacity, dim, device=device, dtype=torch.float32)
        self.size = 0
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor) -> None:
        b = features.shape[0]
        feats = features.detach().to(self.buffer.dtype)
        if b >= self.capacity:
            self.buffer.copy_(feats[-self.capacity:])
            self.size = self.capacity
            self.ptr = 0
            return
        end = self.ptr + b
        if end <= self.capacity:
            self.buffer[self.ptr:end].copy_(feats)
        else:
            first = self.capacity - self.ptr
            self.buffer[self.ptr:].copy_(feats[:first])
            self.buffer[:b - first].copy_(feats[first:])
        self.ptr = end % self.capacity
        self.size = min(self.size + b, self.capacity)

    def get(self) -> torch.Tensor:
        return self.buffer[:self.size]


def _resolve_resume_path(resume_from: str, output_dir: Path) -> Path:
    """Resolve a ``resume_from`` spec to a concrete checkpoint directory.

    "latest" picks the ckpt_NNNNNNN with the highest step under ``output_dir``.
    Any other value is treated as a direct path.
    """
    if resume_from == "latest":
        candidates = sorted(output_dir.glob("ckpt_*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found under {output_dir}")
        return candidates[-1]
    path = Path(resume_from)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _parse_step_from_ckpt(path: Path) -> int:
    name = path.name
    if name.startswith("ckpt_"):
        try:
            return int(name.removeprefix("ckpt_"))
        except ValueError:
            pass
    return 0


def _backbone_params(model: LookingGPT2) -> list:
    """GPT-2 language backbone parameters (trained at a lower LR)."""
    params = []
    params.extend(model.multimod_embedder.text_embedder.parameters())
    params.extend(model.pos_emb.parameters())
    for block in model.blocks:
        params.extend(block.parameters())
    params.extend(model.ln_f.parameters())
    return params


class Trainer:
    """Train LookingGPT2 with a causal-LM objective on image-caption pairs.

    The loss is cross-entropy over text positions only: given image patches and
    all previous text tokens, predict the next text token.

    Training is managed by HuggingFace ``accelerate``, which handles device
    placement, mixed-precision, and distributed runs transparently.
    Metrics are written to TensorBoard via the accelerate tracker.

    Args:
        config: Training hyperparameters.
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.out = Path(config.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, model: LookingGPT2 | None = None) -> None:
        """Run the training loop.

        Args:
            model: Pre-built model to train. If ``None``, loads from
                ``config.pretrained`` via ``LookingGPT2.from_pretrained``.
        """
        cfg = self.cfg

        resume_path: Path | None = None
        resumed_step = 0
        if cfg.resume_from:
            resume_path = _resolve_resume_path(cfg.resume_from, self.out)
            resumed_step = _parse_step_from_ckpt(resume_path)

        run_name = cfg.run_name or _make_run_name(cfg)

        proj_cfg = ProjectConfiguration(
            project_dir=str(self.out),
            # Each run gets its own subfolder so TensorBoard shows them as separate experiments.
            logging_dir=str(self.out / "tensorboard" / run_name),
        )
        accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            log_with="tensorboard",
            project_config=proj_cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )
        # TensorBoard hparams only accept int/float/str/bool — stringify complex fields.
        tb_config = {
            k: (str(v) if not isinstance(v, int | float | str | bool) else v)
            for k, v in dataclasses.asdict(cfg).items()
        }
        accelerator.init_trackers(run_name, config=tb_config)
        if accelerator.is_main_process:
            print(f"Run name   : {run_name}")

        tokenizer = GPT2Tokenizer.from_pretrained(cfg.pretrained)
        tokenizer.pad_token = tokenizer.eos_token

        if model is None:
            if accelerator.is_main_process:
                clip_note = " + CLIP patch encoder" if cfg.use_clip_weights else ""
                cls_note = " + CLS token" if cfg.add_cls else ""
                print(f"Loading pretrained model: {cfg.pretrained!r}{clip_note}{cls_note}")
            model = LookingGPT2.from_pretrained(cfg.pretrained, add_cls=cfg.add_cls)
            if cfg.use_clip_weights:
                model.multimod_embedder.img_embedder._load_clip_weights()

        vocab_size = model.config.vocab_size

        loader = self._build_loader(model.config.image_size, tokenizer)

        # With streaming datasets total_steps is unknown; use max_steps as budget.
        total_steps = cfg.max_steps or (cfg.num_epochs * 10_000)

        # Two param groups: image encoder trains fast, GPT-2 backbone trains slow
        # to adapt to visual tokens without destroying its language priors.
        # Contrastive heads are randomly initialised and go with the fast group.
        img_params = list(model.multimod_embedder.img_embedder.parameters())
        if model.config.contrastive_dim > 0:
            img_params.extend(model.img_contrastive_head.parameters())
            img_params.extend(model.txt_contrastive_head.parameters())
            img_params.append(model.logit_scale)
            img_params.append(model.logit_bias)
        backbone_params = _backbone_params(model)
        optimizer = bnb.optim.AdamW8bit(
            [
                {"params": img_params, "lr": cfg.lr},
                {"params": backbone_params, "lr": cfg.backbone_lr},
            ],
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        lr_lambda = lambda s: _lr_lambda(s, cfg.warmup_steps, total_steps)  # noqa: E731
        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda, lr_lambda])

        # Freeze the backbone for the warmup phase so the image path is forced
        # to produce useful features before GPT-2 is allowed to adapt. Without
        # this, the backbone overfits to caption priors and routes around the
        # (initially noisy) image tokens.
        backbone_frozen = cfg.freeze_backbone_steps > 0 and resumed_step < cfg.freeze_backbone_steps
        if backbone_frozen:
            for p in backbone_params:
                p.requires_grad = False
            if accelerator.is_main_process:
                print(f"Backbone frozen for first {cfg.freeze_backbone_steps} steps")

        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )

        if resume_path is not None:
            accelerator.load_state(str(resume_path))
            if accelerator.is_main_process:
                print(f"Resumed from   : {resume_path} (step {resumed_step})")

        # Hoist unwrapped reference so we can read logit_scale and config
        # without re-unwrapping every micro-batch.
        unwrapped_model = accelerator.unwrap_model(model)

        # Memory banks of stale (detached) contrastive features. They expand
        # the InfoNCE negative set well beyond the physical batch, which is
        # the only thing that actually makes the contrastive task harder.
        img_bank: MemoryBank | None = None
        txt_bank: MemoryBank | None = None
        # The memory bank only helps the softmax InfoNCE formulation; SigLIP
        # works directly at any batch size and stale entries would only add
        # noise.
        if (
            cfg.contrastive_weight > 0
            and cfg.contrastive_memory_bank_size > 0
            and cfg.contrastive_loss_type == "softmax"
            and unwrapped_model.config.contrastive_dim > 0
        ):
            cdim = unwrapped_model.config.contrastive_dim
            img_bank = MemoryBank(cdim, cfg.contrastive_memory_bank_size, accelerator.device)
            txt_bank = MemoryBank(cdim, cfg.contrastive_memory_bank_size, accelerator.device)
            if accelerator.is_main_process:
                print(f"Memory bank    : {cfg.contrastive_memory_bank_size} entries × {cdim}-D × 2")

        if accelerator.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            eff_batch = cfg.batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes
            print(f"Parameters : {n_params:,}  (trainable: {n_trainable:,})")
            print(f"Device     : {accelerator.device}  precision={cfg.mixed_precision}")
            print(f"Batch size : {cfg.batch_size} × {cfg.gradient_accumulation_steps} accum × {accelerator.num_processes} GPU = {eff_batch} effective\n")

        # global_step counts optimizer steps (not micro-batches). All budgets
        # — max_steps, warmup_steps, log_every, save_every — share this unit.
        global_step = resumed_step
        done = False
        running_loss = 0.0
        running_lm_loss = 0.0
        running_loss_content = 0.0
        running_loss_rest = 0.0
        running_contrastive_loss = 0.0
        running_micro_count = 0
        running_grad_img = 0.0
        running_grad_backbone = 0.0
        running_grad_count = 0
        sample_images: torch.Tensor | None = None  # fixed first-batch images for generation
        t0 = time.time()

        for epoch in range(1, cfg.num_epochs + 1):
            if done:
                break
            model.train()

            for images, text_ids, attn_mask in loader:
                # Pin two images from the very first batch for comparable generation samples.
                if sample_images is None:
                    sample_images = images[:2].detach()

                # Labels: clone input_ids but mask padding with -100 so cross_entropy
                # ignores them while keeping the appended EOS as a real target.
                labels = text_ids.clone()
                labels[attn_mask == 0] = -100

                with accelerator.accumulate(model):
                    use_contrastive = cfg.contrastive_weight > 0
                    if use_contrastive:
                        text_logits, img_feat, txt_feat = model(
                            images, text_ids, attention_mask=attn_mask,
                            return_contrastive=True,
                        )
                        if cfg.contrastive_text_only_pass:
                            # Run text alone through the transformer (pure
                            # causal mask, no image tokens) to derive a clean
                            # text contrastive feature. Cheaper than the
                            # zero-image trick and removes all cross-modal
                            # leakage into txt_feat.
                            txt_feat = unwrapped_model.text_contrastive_features(
                                text_ids, attention_mask=attn_mask
                            )
                    else:
                        text_logits = model(images, text_ids, attention_mask=attn_mask)
                    # Split the LM CE into "content" (the first few content
                    # tokens — typically the noun/object after a stereotyped
                    # opener) and "rest" (everything later). Position 0 is
                    # excluded from upweighting because it's almost always a
                    # language-prior token ("A", "The") with no image signal.
                    # text_logits shape: (B, L+1, V). text_logits[:, k] predicts
                    # text[k]; we upweight positions 1..1+content_loss_window.
                    cwin = cfg.content_loss_window
                    loss_content = F.cross_entropy(
                        text_logits[:, 1:1 + cwin].reshape(-1, vocab_size),
                        labels[:, 1:1 + cwin].reshape(-1),
                        ignore_index=-100,
                    )
                    # "rest" still covers position 0 (opener) and positions
                    # after the content window. text_logits[:, -1] is the
                    # generation-time prediction and has no matching label.
                    rest_logits = torch.cat(
                        [text_logits[:, :1], text_logits[:, 1 + cwin:-1]], dim=1
                    )
                    rest_labels = torch.cat(
                        [labels[:, :1], labels[:, 1 + cwin:]], dim=1
                    )
                    loss_rest = F.cross_entropy(
                        rest_logits.reshape(-1, vocab_size),
                        rest_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    alpha = cfg.content_loss_weight
                    # Convex combination so total magnitude is comparable to a
                    # plain CE; alpha=1 reproduces uniform weighting.
                    lm_loss = (alpha * loss_content + loss_rest) / (alpha + 1.0)

                    if use_contrastive:
                        # Gather across processes so each sample's similarity is
                        # computed against the global batch. No-op on single-GPU.
                        gathered_img = accelerator.gather(img_feat)
                        gathered_txt = accelerator.gather(txt_feat)
                        logit_scale = unwrapped_model.logit_scale.exp().clamp(max=100)
                        bsz = img_feat.shape[0]
                        rank_offset = accelerator.process_index * bsz

                        if cfg.contrastive_loss_type == "sigmoid":
                            # SigLIP: per-pair sigmoid BCE on the similarity
                            # matrix. Symmetric — one matrix scores both i→t and
                            # t→i positives. Works at small batch and doesn't
                            # need a memory bank.
                            logit_bias = unwrapped_model.logit_bias
                            sim = logit_scale * img_feat @ gathered_txt.t() + logit_bias
                            targets = -torch.ones_like(sim)
                            rows = torch.arange(bsz, device=sim.device)
                            cols = rows + rank_offset
                            targets[rows, cols] = 1.0
                            # -log σ(t·sim) — equivalent to BCE with logits,
                            # but in the SigLIP paper's form.
                            contrastive_loss = -F.logsigmoid(sim * targets).mean()
                        else:
                            # Append memory-bank entries as detached negatives.
                            # Positives stay at the first ``B*world_size`` indices,
                            # so labels_c below doesn't need adjustment.
                            if img_bank is not None and img_bank.size > 0:
                                all_img = torch.cat(
                                    [gathered_img, img_bank.get().to(gathered_img.dtype)], dim=0
                                )
                                all_txt = torch.cat(
                                    [gathered_txt, txt_bank.get().to(gathered_txt.dtype)], dim=0
                                )
                            else:
                                all_img = gathered_img
                                all_txt = gathered_txt

                            # CLIP-style InfoNCE.
                            logits_i2t = logit_scale * img_feat @ all_txt.t()
                            logits_t2i = logit_scale * txt_feat @ all_img.t()
                            labels_c = torch.arange(bsz, device=img_feat.device) + rank_offset
                            contrastive_loss = 0.5 * (
                                F.cross_entropy(logits_i2t, labels_c)
                                + F.cross_entropy(logits_t2i, labels_c)
                            )

                        loss = lm_loss + cfg.contrastive_weight * contrastive_loss

                        # Enqueue this step's gathered features for use as
                        # stale negatives in subsequent softmax steps.
                        if img_bank is not None:
                            img_bank.enqueue(gathered_img)
                            txt_bank.enqueue(gathered_txt)
                    else:
                        contrastive_loss = torch.zeros((), device=lm_loss.device)
                        loss = lm_loss

                    accelerator.backward(loss)
                    stepped = accelerator.sync_gradients
                    grad_norm_img = grad_norm_backbone = None
                    if stepped:
                        # Snapshot per-group grad norms before clipping so the
                        # logged values reflect the *actual* signal each
                        # parameter group received this step.
                        grad_norm_img = _grad_norm(img_params)
                        grad_norm_backbone = _grad_norm(backbone_params)
                        accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                running_loss += accelerator.gather(loss).mean().item()
                running_lm_loss += accelerator.gather(lm_loss).mean().item()
                running_loss_content += accelerator.gather(loss_content).mean().item()
                running_loss_rest += accelerator.gather(loss_rest).mean().item()
                running_contrastive_loss += accelerator.gather(contrastive_loss).mean().item()
                running_micro_count += 1
                if grad_norm_img is not None:
                    running_grad_img += grad_norm_img
                    running_grad_backbone += grad_norm_backbone
                    running_grad_count += 1

                if not stepped:
                    continue
                global_step += 1

                if backbone_frozen and global_step >= cfg.freeze_backbone_steps:
                    for p in backbone_params:
                        p.requires_grad = True
                    backbone_frozen = False
                    if accelerator.is_main_process:
                        print(f"  → unfreezing backbone at step {global_step}")

                if global_step % cfg.log_every == 0 and accelerator.is_main_process:
                    n = max(1, running_micro_count)
                    avg_loss = running_loss / n
                    avg_lm_loss = running_lm_loss / n
                    avg_loss_content = running_loss_content / n
                    avg_loss_rest = running_loss_rest / n
                    avg_contrastive_loss = running_contrastive_loss / n
                    elapsed = time.time() - t0
                    tok_per_sec = running_micro_count * cfg.batch_size * cfg.max_text_len / elapsed
                    lr_img, lr_backbone = scheduler.get_last_lr()
                    n_g = max(1, running_grad_count)
                    avg_grad_img = running_grad_img / n_g
                    avg_grad_backbone = running_grad_backbone / n_g
                    # While the backbone is frozen, |g_bb| is effectively zero
                    # and the ratio degenerates; hide it then.
                    if avg_grad_backbone > 1e-8:
                        grad_ratio = avg_grad_img / avg_grad_backbone
                        ratio_str = f"{grad_ratio:.2f}"
                    else:
                        grad_ratio = float("nan")
                        ratio_str = "N/A"

                    print(
                        f"epoch {epoch:02d} | step {global_step:6d} | "
                        f"loss {avg_loss:.4f} (lm {avg_lm_loss:.4f} "
                        f"[content {avg_loss_content:.4f}, rest {avg_loss_rest:.4f}], "
                        f"c {avg_contrastive_loss:.4f}) | "
                        f"lr_img {lr_img:.2e} | lr_bb {lr_backbone:.2e} | "
                        f"|g_img| {avg_grad_img:.2e} | |g_bb| {avg_grad_backbone:.2e} | "
                        f"ratio {ratio_str} | {tok_per_sec:,.0f} tok/s"
                    )
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/lm_loss": avg_lm_loss,
                            "train/loss_content": avg_loss_content,
                            "train/loss_rest": avg_loss_rest,
                            "train/contrastive_loss": avg_contrastive_loss,
                            "train/lr_img": lr_img,
                            "train/lr_backbone": lr_backbone,
                            "train/tokens_per_sec": tok_per_sec,
                            "train/grad_norm_img": avg_grad_img,
                            "train/grad_norm_backbone": avg_grad_backbone,
                            "train/grad_norm_ratio": grad_ratio,
                            "train/logit_scale": unwrapped_model.logit_scale.detach().item(),
                            "train/logit_bias": unwrapped_model.logit_bias.detach().item(),
                        },
                        step=global_step,
                    )
                    running_loss = 0.0
                    running_lm_loss = 0.0
                    running_loss_content = 0.0
                    running_loss_rest = 0.0
                    running_contrastive_loss = 0.0
                    running_micro_count = 0
                    running_grad_img = 0.0
                    running_grad_backbone = 0.0
                    running_grad_count = 0
                    t0 = time.time()

                if global_step % cfg.save_every == 0:
                    self._save(accelerator, global_step)
                    self._generate_sample(accelerator, model, tokenizer, sample_images, global_step)

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    done = True
                    break

        self._save(accelerator, global_step)
        self._generate_sample(accelerator, model, tokenizer, sample_images, global_step)
        accelerator.end_training()
        if accelerator.is_main_process:
            print("Training complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_loader(self, image_size: int, tokenizer: GPT2Tokenizer) -> DataLoader:
        cfg = self.cfg
        ds = HFImageCaptionDataset(
            dataset_name=cfg.dataset_name,
            split=cfg.dataset_split,
            image_col=cfg.image_col,
            caption_col=cfg.caption_col,
            tokenizer=tokenizer,
            max_text_len=cfg.max_text_len,
            image_size=image_size,
            shuffle_buffer=cfg.shuffle_buffer,
            use_clip_normalization=cfg.use_clip_weights,
            url_col=cfg.url_col,
            prefetch_workers=cfg.prefetch_workers,
            augment=cfg.augment,
        )
        num_shards = getattr(ds.hf_ds, "num_shards", 1) or 1
        num_workers = min(cfg.num_workers, num_shards)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            pin_memory=accelerator_pin_memory(),
        )

    def _save(self, accelerator: Accelerator, step: int) -> None:
        ckpt_dir = self.out / f"ckpt_{step:07d}"
        accelerator.save_state(str(ckpt_dir))
        if accelerator.is_main_process:
            print(f"  → checkpoint: {ckpt_dir}")

    def _generate_sample(
        self,
        accelerator: Accelerator,
        model: LookingGPT2,
        tokenizer: GPT2Tokenizer,
        sample_images: torch.Tensor | None,
        step: int,
    ) -> None:
        if not accelerator.is_main_process or sample_images is None:
            return

        cfg = self.cfg
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        writer = accelerator.get_tracker("tensorboard", unwrap=True)
        null_image = torch.zeros_like(sample_images[:1])  # blank image for conditioning check

        for idx, img in enumerate(sample_images):
            img_batch = img.unsqueeze(0)  # (1, C, H, W)

            img_display = _denormalize(img.float().cpu(), use_clip=cfg.use_clip_weights)
            writer.add_image(f"sample/image_{idx}", img_display, global_step=step)

            greedy_ids = tokenizer.encode("", return_tensors="pt").to(img_batch.device)

            # Greedy on real image — no sampling noise, best diagnostic for conditioning.
            greedy_out = unwrapped.generate(
                img_batch, greedy_ids,
                max_new_tokens=cfg.sample_max_new_tokens,
                temperature=0,
                eos_token_id=tokenizer.eos_token_id,
            )
            greedy_text = tokenizer.decode(greedy_out[0, greedy_ids.shape[1]:], skip_special_tokens=True)

            # Same prompt, blank image — identical output means image is being ignored.
            null_out = unwrapped.generate(
                null_image, greedy_ids,
                max_new_tokens=cfg.sample_max_new_tokens,
                temperature=0,
                eos_token_id=tokenizer.eos_token_id,
            )
            null_text = tokenizer.decode(null_out[0, greedy_ids.shape[1]:], skip_special_tokens=True)

            flag = "⚠ same — image ignored" if greedy_text == null_text else "✓ different"
            print(f"  [img {idx}] greedy: {greedy_text!r}")
            print(f"  [img {idx}] null  : {null_text!r}  {flag}")

            tb_text = [
                f"**greedy (real image):** {greedy_text}",
                f"**greedy (null image):** {null_text}",
            ]
            for prompt in cfg.sample_prompts:
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(img_batch.device)
                out_ids = unwrapped.generate(
                    img_batch, prompt_ids,
                    max_new_tokens=cfg.sample_max_new_tokens,
                    temperature=cfg.sample_temperature,
                    top_k=cfg.sample_top_k,
                    eos_token_id=tokenizer.eos_token_id,
                )
                continuation = tokenizer.decode(out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
                print(f"         {prompt!r} → {continuation!r}")
                tb_text.append(f"**{prompt}** {continuation}")

            writer.add_text(f"sample/captions_{idx}", "  \n".join(tb_text), global_step=step)

        unwrapped.train()


def accelerator_pin_memory() -> bool:
    return torch.cuda.is_available()
