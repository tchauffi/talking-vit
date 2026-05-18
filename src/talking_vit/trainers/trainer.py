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

from talking_vit.models.talking_vit import LookingGPT2
from talking_vit.trainers.dataset import HFImageCaptionDataset


@dataclass
class TrainConfig:
    # Data
    dataset_name: str = "clip-benchmark/wds_mscoco_captions"
    dataset_split: str = "train"
    image_col: str = "jpg"
    caption_col: str = "txt"
    shuffle_buffer: int = 1000
    max_text_len: int = 64
    num_workers: int = 0   # 0 required for IterableDataset + streaming

    # Model
    pretrained: str = "gpt2"
    use_clip_weights: bool = False
    add_cls: bool = False

    # Optimisation
    batch_size: int = 16
    num_epochs: int = 10
    lr: float = 3e-4
    backbone_lr: float = 3e-5  # GPT-2 backbone uses a lower LR to preserve language priors
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 2
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
        img_params = list(model.multimod_embedder.img_embedder.parameters())
        backbone_params = _backbone_params(model)
        optimizer = AdamW(
            [
                {"params": img_params, "lr": cfg.lr},
                {"params": backbone_params, "lr": cfg.backbone_lr},
            ],
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        lr_lambda = lambda s: _lr_lambda(s, cfg.warmup_steps, total_steps)  # noqa: E731
        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda, lr_lambda])

        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )

        if accelerator.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            eff_batch = cfg.batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes
            print(f"Parameters : {n_params:,}  (trainable: {n_trainable:,})")
            print(f"Device     : {accelerator.device}  precision={cfg.mixed_precision}")
            print(f"Batch size : {cfg.batch_size} × {cfg.gradient_accumulation_steps} accum × {accelerator.num_processes} GPU = {eff_batch} effective\n")

        # global_step counts optimizer steps (not micro-batches). All budgets
        # — max_steps, warmup_steps, log_every, save_every — share this unit.
        global_step = 0
        done = False
        running_loss = 0.0
        running_micro_count = 0
        sample_image: torch.Tensor | None = None  # fixed first-batch image for generation
        t0 = time.time()

        for epoch in range(1, cfg.num_epochs + 1):
            if done:
                break
            model.train()

            for images, text_ids, attn_mask in loader:
                # Pin one image from the very first batch for comparable generation samples.
                if sample_image is None:
                    sample_image = images[:1].detach()

                # Labels: clone input_ids but mask padding with -100 so cross_entropy
                # ignores them while keeping the appended EOS as a real target.
                labels = text_ids.clone()
                labels[attn_mask == 0] = -100

                with accelerator.accumulate(model):
                    text_logits = model(images, text_ids, attention_mask=attn_mask)
                    loss = F.cross_entropy(
                        text_logits[:, :-1].reshape(-1, vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
                    accelerator.backward(loss)
                    stepped = accelerator.sync_gradients
                    if stepped:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                running_loss += accelerator.gather(loss).mean().item()
                running_micro_count += 1

                if not stepped:
                    continue
                global_step += 1

                if global_step % cfg.log_every == 0 and accelerator.is_main_process:
                    avg_loss = running_loss / max(1, running_micro_count)
                    elapsed = time.time() - t0
                    tok_per_sec = running_micro_count * cfg.batch_size * cfg.max_text_len / elapsed
                    lr_img, lr_backbone = scheduler.get_last_lr()

                    print(
                        f"epoch {epoch:02d} | step {global_step:6d} | "
                        f"loss {avg_loss:.4f} | lr_img {lr_img:.2e} | lr_bb {lr_backbone:.2e} | {tok_per_sec:,.0f} tok/s"
                    )
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr_img": lr_img,
                            "train/lr_backbone": lr_backbone,
                            "train/tokens_per_sec": tok_per_sec,
                        },
                        step=global_step,
                    )
                    running_loss = 0.0
                    running_micro_count = 0
                    t0 = time.time()

                if global_step % cfg.save_every == 0:
                    self._save(accelerator, global_step)
                    self._generate_sample(accelerator, model, tokenizer, sample_image, global_step)

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    done = True
                    break

        self._save(accelerator, global_step)
        self._generate_sample(accelerator, model, tokenizer, sample_image, global_step)
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
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
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
        sample_image: torch.Tensor | None,
        step: int,
    ) -> None:
        if not accelerator.is_main_process or sample_image is None:
            return

        cfg = self.cfg
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        # Log the sample image (denormalized) so it's visible alongside captions.
        writer = accelerator.get_tracker("tensorboard", unwrap=True)
        img_display = _denormalize(sample_image[0].float().cpu(), use_clip=cfg.use_clip_weights)
        writer.add_image("sample/image", img_display, global_step=step)

        print("  [samples]")
        tb_text = []
        for prompt in cfg.sample_prompts:
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(sample_image.device)
            out_ids = unwrapped.generate(
                sample_image,
                prompt_ids,
                max_new_tokens=cfg.sample_max_new_tokens,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_ids = out_ids[0, prompt_ids.shape[1]:]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
            print(f"    {prompt!r} → {continuation!r}")
            tb_text.append(f"**{prompt}** {continuation}")

        writer.add_text("sample/captions", "  \n".join(tb_text), global_step=step)
        unwrapped.train()


def accelerator_pin_memory() -> bool:
    return torch.cuda.is_available()
