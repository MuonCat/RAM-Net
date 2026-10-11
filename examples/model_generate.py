"""Download the pretrained RAM-Net 1.3B checkpoint and generate text on CUDA."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ramnet  # Register RAM-Net with the Hugging Face Auto classes.


def main() -> None:
    repo_id = "kaichengx/RAM-Net-Models"
    subfolder = "RAMNet-1.3B-100B-Top8"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, subfolder=subfolder, dtype=torch.bfloat16,
    ).cuda().eval()
    inputs = tokenizer("The future of artificial intelligence is", return_tensors="pt").to("cuda")

    with torch.inference_mode():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
