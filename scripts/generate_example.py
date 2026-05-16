"""Log a few generation examples from LookingGPT2."""

import torch
from transformers import GPT2Tokenizer

from talking_vit.models.talking_vit import GPT2Config, LookingGPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = [
    "The image shows",
    "In this picture,",
    "I can see",
]


def make_fake_image(config: GPT2Config) -> torch.Tensor:
    """Random RGB image matching the model's expected input."""
    return torch.randn(1, config.in_dim, config.image_size, config.image_size, device=DEVICE)


def run():
    print("Loading GPT-2 tokenizer and pretrained LookingGPT2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = LookingGPT2.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    config = model.config
    image = make_fake_image(config)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Image: {list(image.shape)}  →  {config.num_img_tokens} patch tokens\n")
    print("=" * 60)

    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        out_ids = model.generate(
            image,
            ids,
            max_new_tokens=40,
            temperature=0.8,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        new_ids = out_ids[0, ids.shape[1]:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        print(f"Prompt : {prompt!r}")
        print(f"Output : {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    run()
