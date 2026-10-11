"""Small RAM cases and calls shared by CUDA contract and compile tests."""

from tests.ops.ram_attention_helpers import RamCase, make_ram_case


def make_cuda_case(*, order: int = 2, seq_len: int = 32) -> RamCase:
    return make_ram_case(
        batch=2, heads=2, order=order, radix=4,
        channels=7, seq_len=seq_len, masked=False,
    )


def run_cuda_ram(
        func,
        inputs,
        position_ids,
        mask,
        *,
        mass_aware: bool = False,
        eps: float = 1e-6,
        inplace: bool = False,
):
    return func(
        *inputs,
        mask, 1, position_ids,
        (2, 3),
        (False, True),
        mass_aware, eps,
        inplace,
    )
