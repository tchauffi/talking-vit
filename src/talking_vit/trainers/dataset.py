"""HuggingFace image-caption dataset for LookingGPT2."""

import random

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from transformers import GPT2Tokenizer


class HFImageCaptionDataset(IterableDataset):
    """Wrap any HuggingFace streaming dataset that has an image and a text column.

    Supports newline-separated multi-caption fields (e.g. COCO's ``txt`` column
    stores all 5 captions joined by ``\\n``). One caption is picked at random
    each time a sample is yielded.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split (``"train"``, ``"validation"``, …).
        image_col: Name of the PIL-image column.
        caption_col: Name of the caption column (str or newline-joined str).
        tokenizer: GPT-2 tokenizer.
        max_text_len: Caption is truncated / padded to this length.
        image_size: Images are resized to ``(image_size, image_size)``.
        shuffle_buffer: Buffer size for streaming shuffle (0 = no shuffle).
        max_samples: Stop after this many samples (useful for smoke tests).
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_col: str,
        caption_col: str,
        tokenizer: GPT2Tokenizer,
        max_text_len: int = 64,
        image_size: int = 224,
        shuffle_buffer: int = 1000,
        max_samples: int | None = None,
        use_clip_normalization: bool = False,
    ):
        from datasets import load_dataset

        # stream the dataset to avoid downloading and storing it locally.
        self.hf_ds = load_dataset(dataset_name, split=split, streaming=True)
        if shuffle_buffer > 0:
            self.hf_ds = self.hf_ds.shuffle(buffer_size=shuffle_buffer)

        self.image_col = image_col
        self.caption_col = caption_col
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_samples = max_samples

        mean = self.CLIP_MEAN if use_clip_normalization else self.IMAGENET_MEAN
        std = self.CLIP_STD if use_clip_normalization else self.IMAGENET_STD

        # Resize-shorter + CenterCrop preserves aspect ratio. Resize((H, W)) with a
        # tuple stretches the image and silently misaligns captions like
        # "tall building" with squished pixels.
        self.transform = transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __iter__(self):
        count = 0
        for sample in self.hf_ds:
            if self.max_samples is not None and count >= self.max_samples:
                return

            try:
                image = sample[self.image_col].convert("RGB")
            except Exception:
                continue

            image = self.transform(image)

            raw = sample[self.caption_col]
            captions = raw.split("\n") if "\n" in raw else [raw]
            caption = random.choice(captions).strip()
            if not caption:
                continue
            
            enc = self.tokenizer(
                caption,
                max_length=self.max_text_len - 1,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)

            eos_id = self.tokenizer.eos_token_id
            ids = torch.cat([ids, torch.tensor([eos_id], dtype=ids.dtype)])
            mask = torch.cat([mask, torch.ones(1, dtype=mask.dtype)])

            pad_len = self.max_text_len - ids.shape[0]
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

            yield image, ids, mask
            count += 1
