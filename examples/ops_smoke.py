"""Exercise public RAM attention and its gradients from an installed wheel."""

import argparse

import torch

from ramnet_ops import random_access_memory_attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    device = parser.parse_args().device
    backend = "cuda" if device == "cuda" else "torch"
    torch.manual_seed(0)

    B, T, H, U, P, C = 2, 32, 4, 3, 4, 32
    w_addr = torch.randn(B, T, H, U, P, device=device, requires_grad=True)
    r_addr = torch.randn(B, T, H, U, P, device=device, requires_grad=True)
    values = torch.randn(B, T, H, C, device=device, requires_grad=True)
    gamma = torch.randn(B, T, H, device=device, requires_grad=True)

    y, (slot, mass) = random_access_memory_attention(
        w_addr, r_addr, values, gamma,
        embed_num=2, top_k=8,
        impl_backend=backend,
    )
    loss = y.square().mean() + slot.square().mean() + mass.square().mean()
    loss.backward()
    print(y.shape)     # (B, T, H, C)
    print(slot.shape)  # (B, H, P ** U, C)
    print(mass.shape)  # (B, H, P ** U)

    assert y.shape == values.shape
    for tensor in (w_addr, r_addr, values, gamma):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()

    print(f'Public API forward/backward passed on {device}.')


if __name__ == '__main__':
    main()
