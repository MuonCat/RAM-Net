"""Use RAM-Net through Hugging Face Auto classes, training loss, and generation."""

import torch
from transformers import AutoModelForCausalLM

from ramnet import RAMNetConfig


def main() -> None:
    config = RAMNetConfig(
        hidden_size=128, num_attention_heads=4, num_hidden_layers=2,
        addr_order=3, addr_radix=4,
        embed_num=2, addr_top_k=8,
        output_norm=True,
        hidden_act="SwiGLU",
        vocab_size=1024, pad_token_id=0,
        dtype="float32",
    )
    model = AutoModelForCausalLM.from_config(config).cuda()
    input_ids = torch.randint(3, config.vocab_size, (2, 32), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    model.train()
    output = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=input_ids, use_cache=False,
    )
    output.loss.backward()
    print(output.logits.shape)  # (2, 32, 1024)

    model.eval()
    with torch.inference_mode():
        tokens = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=8, do_sample=False, use_cache=True,
        )
    print(tokens.shape)


if __name__ == '__main__':
    main()
