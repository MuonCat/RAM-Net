"""FP32 Torch/CUDA agreement and CUDA mixed-precision error."""

from dataclasses import replace
from itertools import product
import unittest

import torch

from ramnet_ops.impl.torch_impl import _torch_ram_attn

from tests.helpers import assert_close_with_stats, require_cuda_ops
from tests.ops.precision import low_precision_dtypes, requires_low_precision
from tests.ops.ram_attention_helpers import (
    TOP_K_RE_NORM_SETTINGS,
    cast_ram_case,
    gather_ram_result,
    make_ram_case,
    ram_masked_pairs,
)
from tests.ops.tolerances import FP32_TOLERANCE, PRECISION_TOLERANCES


# U, P, C, T: partial template widths and a long sequence.
SHAPE_CASES = ((4, 3, 29, 1024), (4, 5, 47, 33), (4, 7, 65, 128), (5, 4, 32, 1024))


class RamAttentionNumericsTest(unittest.TestCase):
    def setUp(self):
        self.ram = require_cuda_ops()

    def test_fp32_matches_torch(self):
        for (order, radix, channels, seq_len), re_norm, mass_aware in product(
                SHAPE_CASES, TOP_K_RE_NORM_SETTINGS, (False, True),
        ):
            with self.subTest(U=order, P=radix, C=channels, T=seq_len, re_norm=re_norm, mass_aware=mass_aware):
                case = make_ram_case(batch=2, heads=4, order=order, radix=radix, channels=channels, seq_len=seq_len, masked=False)
                options = dict(top_k_re_norm=re_norm, mass_aware=mass_aware)
                actual = gather_ram_result(self.ram, case, **options)
                expected = gather_ram_result(_torch_ram_attn, case, **options)
                assert_close_with_stats("CUDA == TORCH FP32", ram_masked_pairs(actual, expected), **FP32_TOLERANCE)

    def test_fp32_mask_and_position_modes_match_torch(self):
        source = make_ram_case(batch=2, heads=4, order=3, channels=16, seq_len=65)
        shifted = replace(source, position_ids=(source.position_ids * 3 + 11).masked_fill(~source.mask, 0))
        modes = (("unmasked", replace(source, mask=None)), ("masked", source), ("shifted", shifted))
        for (name, case), re_norm, mass_aware in product(modes, TOP_K_RE_NORM_SETTINGS, (False, True)):
            with self.subTest(mode=name, re_norm=re_norm, mass_aware=mass_aware):
                options = dict(top_k_re_norm=re_norm, mass_aware=mass_aware)
                actual = gather_ram_result(self.ram, case, **options)
                expected = gather_ram_result(_torch_ram_attn, case, **options)
                assert_close_with_stats(
                    "CUDA == TORCH FP32 MASK/POSITION",
                    ram_masked_pairs(actual, expected, res_mask=case.mask, ref_mask=case.mask),
                    **FP32_TOLERANCE,
                )

    def test_fp32_sequence_boundaries_match_torch(self):
        for seq_len, mass_aware in product((1, 31, 32, 33), (False, True)):
            with self.subTest(seq_len=seq_len, mass_aware=mass_aware):
                case = make_ram_case(batch=2, heads=4, order=2, channels=8, seq_len=seq_len, masked=False)
                actual = gather_ram_result(self.ram, case, top_k=(4, 8), mass_aware=mass_aware)
                expected = gather_ram_result(_torch_ram_attn, case, top_k=(4, 8), mass_aware=mass_aware)
                assert_close_with_stats("CUDA == TORCH FP32 BOUNDARIES", ram_masked_pairs(actual, expected), **FP32_TOLERANCE)

    @requires_low_precision
    def test_low_precision_parallel_matches_cuda_fp32(self):
        dtypes = low_precision_dtypes(required=True)
        for dtype, (order, radix, channels, seq_len) in product(dtypes, SHAPE_CASES):
            source = make_ram_case(batch=2, heads=4, order=order, radix=radix, channels=channels, seq_len=seq_len)
            low = cast_ram_case(source, dtype)
            # Promote already quantized inputs so input rounding does not change Top-K selection.
            fp32 = cast_ram_case(low, torch.float32)
            for re_norm, mass_aware in product(TOP_K_RE_NORM_SETTINGS, (False, True)):
                with self.subTest(dtype=dtype, U=order, P=radix, C=channels, T=seq_len, re_norm=re_norm, mass_aware=mass_aware):
                    options = dict(top_k_re_norm=re_norm, mass_aware=mass_aware)
                    actual = gather_ram_result(self.ram, low, **options)
                    expected = gather_ram_result(self.ram, fp32, **options)
                    assert_close_with_stats(
                        "CUDA LOW PRECISION == CUDA FP32",
                        ram_masked_pairs(actual, expected),
                        **PRECISION_TOLERANCES[dtype],
                    )

    @requires_low_precision
    def test_low_precision_autoregressive_matches_cuda_fp32(self):
        dtypes = low_precision_dtypes(required=True)
        source = make_ram_case(batch=2, heads=4, order=3, channels=16, seq_len=33)
        for dtype, re_norm, mass_aware in product(dtypes, TOP_K_RE_NORM_SETTINGS, (False, True)):
            with self.subTest(dtype=dtype, re_norm=re_norm, mass_aware=mass_aware):
                low = cast_ram_case(source, dtype)
                fp32 = cast_ram_case(low, torch.float32)
                options = dict(top_k_re_norm=re_norm, mass_aware=mass_aware, chunk_size=1, grad=False)
                actual = gather_ram_result(self.ram, low, **options)
                expected = gather_ram_result(self.ram, fp32, **options)
                assert_close_with_stats(
                    "CUDA LOW PRECISION == CUDA FP32 AR",
                    ram_masked_pairs(actual, expected),
                    **PRECISION_TOLERANCES[dtype],
                )
