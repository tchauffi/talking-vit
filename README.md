# Talking ViT

This project aims to act as a POC for a multimodal vision-language model, where vision token are stacked with language tokens. The model allow to generate caption autoregressively from image tokens.

This project is an educative project to explore the design of a multimodal model, and is not intended to be used in production.

## Dependancies

This porject is using UV as a dependancy manager, and the dependancies are listed in the `pyproject.toml` file. To install the dependancies, you can run the following command:

```bash
uv pip install -e .
```

It uses the `tokenizers` library to tokenize the text, and `torch` and `torchvision` for the model and image processing.
