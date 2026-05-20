"""HuggingFace image-caption dataset for LookingGPT2."""

import hashlib
import io
import random
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from PIL import Image
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
        url_col: Column containing image URLs. When set, images are fetched
            over HTTP instead of read from ``image_col``.
        url_timeout: Per-request timeout in seconds.
        prefetch_workers: Number of threads used to fetch images concurrently.
            Effective only when ``url_col`` is set.
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
        max_text_len: int = 32,
        image_size: int = 128,
        shuffle_buffer: int = 1000,
        max_samples: int | None = None,
        use_clip_normalization: bool = False,
        url_col: str | None = None,
        url_timeout: float = 2.0,
        prefetch_workers: int = 16,
        cache_dir: str | None = None,
        augment: bool = False,
    ):
        from datasets import load_dataset

        # stream the dataset to avoid downloading and storing it locally.
        self.hf_ds = load_dataset(dataset_name, split=split, streaming=True)
        if shuffle_buffer > 0:
            self.hf_ds = self.hf_ds.shuffle(buffer_size=shuffle_buffer)

        self.image_col = image_col
        self.caption_col = caption_col
        self.url_col = url_col
        self.url_timeout = url_timeout
        self.prefetch_workers = prefetch_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_samples = max_samples

        mean = self.CLIP_MEAN if use_clip_normalization else self.IMAGENET_MEAN
        std = self.CLIP_STD if use_clip_normalization else self.IMAGENET_STD

        # Resize-shorter + CenterCrop preserves aspect ratio. Resize((H, W)) with a
        # tuple stretches the image and silently misaligns captions like
        # "tall building" with squished pixels.
        if augment:
            # RandomResizedCrop + HFlip + light ColorJitter effectively expand
            # the dataset and force the patch encoder to learn invariances.
            # Scale/ratio kept conservative so captions still describe the crop.
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.7, 1.0),
                    ratio=(0.85, 1.18),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_image(self, sample: dict) -> Image.Image | None:
        try:
            if self.url_col is not None:
                url = sample[self.url_col]
                if self.cache_dir is not None:
                    key = hashlib.md5(url.encode()).hexdigest()
                    path = self.cache_dir / key[:2] / f"{key}.jpg"
                    if path.exists():
                        return Image.open(path).convert("RGB")
                    resp = requests.get(url, timeout=self.url_timeout)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    path.parent.mkdir(exist_ok=True)
                    # Pre-resize before caching so disk reads are fast and cache stays small.
                    cache_size = self.transform.transforms[0].size  # Resize target (int)
                    w, h = img.size
                    scale = cache_size / min(w, h)
                    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
                    img.save(path, format="JPEG", quality=90)
                    return img
                resp = requests.get(url, timeout=self.url_timeout)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            return sample[self.image_col].convert("RGB")
        except Exception:
            return None

    def _encode_caption(self, raw: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        captions = raw.split("\n") if "\n" in raw else [raw]
        caption = random.choice(captions).strip()
        if not caption:
            return None

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

        return ids, mask

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        if self.url_col is not None:
            yield from self._iter_with_prefetch()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self):
        count = 0
        for sample in self.hf_ds:
            if self.max_samples is not None and count >= self.max_samples:
                return
            image = self._fetch_image(sample)
            if image is None:
                continue
            result = self._encode_caption(sample[self.caption_col])
            if result is None:
                continue
            yield self.transform(image), *result
            count += 1

    def _iter_with_prefetch(self):
        """Fetch images concurrently with a sliding window of futures.

        Each future carries (pil_image_or_None, caption_str) so the main
        thread only does transform + tokenise (CPU-only, fast).
        """
        count = 0
        window: int = self.prefetch_workers * 2
        pending: deque[tuple[Future, str]] = deque()

        def submit(sample: dict) -> tuple[Future, str]:
            return executor.submit(self._fetch_image, sample), sample[self.caption_col]

        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as executor:
            src = iter(self.hf_ds)

            # Fill the initial window.
            for sample in src:
                pending.append(submit(sample))
                if len(pending) >= window:
                    break

            while pending:
                if self.max_samples is not None and count >= self.max_samples:
                    # Cancel remaining futures.
                    for fut, _ in pending:
                        fut.cancel()
                    return

                future, caption_raw = pending.popleft()

                # Refill: submit next sample while waiting for this future.
                try:
                    nxt = next(src)
                    pending.append(submit(nxt))
                except StopIteration:
                    pass

                image = future.result()
                if image is None:
                    continue

                result = self._encode_caption(caption_raw)
                if result is None:
                    continue

                yield self.transform(image), *result
                count += 1
