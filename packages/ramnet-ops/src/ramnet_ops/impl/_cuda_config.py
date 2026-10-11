from ramnet_ops import _C


INT_BITS = _C.INT_BITS
INT_MAX = (1 << (INT_BITS - 1)) - 1
# Reserve one magnitude bit for intermediate address arithmetic.
ADDRESS_LIMIT = 1 << (INT_BITS - 2)


def check_int_capacity(*sizes: int) -> None:
    assert all(0 <= size < INT_MAX for size in sizes), f"CUDA index overflow: sizes={sizes}, limit={INT_MAX}."
