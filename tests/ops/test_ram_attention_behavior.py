"""Sequence boundaries, mask/position semantics, and CUDA state carry."""

from dataclasses import replace
from itertools import product
import unittest

import torch

from ramnet_ops.impl.torch_impl import _torch_ram_attn

from tests.helpers import assert_close_with_stats, require_cuda_ops, select_valid_tokens
from tests.ops.precision import low_precision_dtypes
from tests.ops.ram_attention_helpers import (
    TOP_K_RE_NORM_SETTINGS,
    TOP_K_SETTINGS,
    cast_ram_case,
    compact_valid_tokens,
    gather_ram_result,
    make_ram_case,
    ram_masked_pairs,
)
from tests.ops.tolerances import BEHAVIOR_TOLERANCE


class RamAttentionBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.ram = require_cuda_ops()

    def test_mask_and_positions_match_compact_sequence(self):
        case = make_ram_case(batch=2, heads=4, order=3, seq_len=65)
        for shifted, mass_aware in product((False, True), repeat=2):
            with self.subTest(shifted=shifted, mass_aware=mass_aware):
                positions = case.position_ids * 3 + 11 if shifted else case.position_ids
                padded = replace(case, position_ids=positions.masked_fill(~case.mask, 0))
                compact = compact_valid_tokens(padded)
                actual = gather_ram_result(self.ram, padded, mass_aware=mass_aware)
                expected = gather_ram_result(self.ram, compact, mass_aware=mass_aware)
                assert_close_with_stats(
                    "MASK/POSITION == COMPACT",
                    ram_masked_pairs(actual, expected, res_mask=padded.mask),
                    **BEHAVIOR_TOLERANCE,
                )

    def test_chunked_state_and_gradients_match_full(self):
        for masked, re_norm, mass_aware in product((False, True), TOP_K_RE_NORM_SETTINGS, (False, True)):
            with self.subTest(masked=masked, re_norm=re_norm, mass_aware=mass_aware):
                case = make_ram_case(batch=2, heads=4, order=3, channels=8, seq_len=65, masked=False)
                if masked:
                    mask = torch.ones(2, 65, device="cuda", dtype=torch.bool)
                    mask[:, 22:44] = False  # An entire chunk must carry state without updates.
                    mask[0, ::5] = False
                    mask[1, 1::7] = False
                    positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
                    case = replace(case, mask=mask, position_ids=positions)
                options = dict(top_k=(4, 8), top_k_re_norm=re_norm, mass_aware=mass_aware)
                expected = gather_ram_result(self.ram, case, **options)
                for chunk_size in (22, 32, 33):
                    with self.subTest(chunk_size=chunk_size):
                        actual = gather_ram_result(self.ram, case, chunk_size=chunk_size, **options)
                        assert_close_with_stats(
                            "CHUNKED == FULL",
                            ram_masked_pairs(actual, expected),
                            **BEHAVIOR_TOLERANCE,
                        )

    def test_top_k_range_and_long_sequence(self):
        case = make_ram_case(batch=2, heads=4, order=5, channels=16, seq_len=1024, masked=False)
        for top_k, re_norm in product(TOP_K_SETTINGS, TOP_K_RE_NORM_SETTINGS):
            with self.subTest(top_k=top_k, re_norm=re_norm):
                options = dict(top_k=top_k, top_k_re_norm=re_norm)
                actual = gather_ram_result(self.ram, case, chunk_size=128, **options)
                expected = gather_ram_result(self.ram, case, **options)
                assert_close_with_stats(
                    "LONG SEQUENCE: CHUNKED == FULL",
                    ram_masked_pairs(actual, expected),
                    **BEHAVIOR_TOLERANCE,
                )

    def test_autoregressive_matches_parallel(self):
        case = make_ram_case(batch=3, heads=4, order=5, channels=16, seq_len=65, masked=False)
        mask = torch.ones(3, 65, device="cuda", dtype=torch.bool)
        mask[0, ::3] = False
        mask[1, 1::4] = False
        mask[2, 2::5] = False
        mask[:, 7::16] = False
        positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
        masked = replace(case, mask=mask, position_ids=positions)
        for top_k, re_norm, mass_aware in product(TOP_K_SETTINGS, TOP_K_RE_NORM_SETTINGS, (False, True)):
            for name, inputs in (("plain", case), ("masked", masked)):
                with self.subTest(top_k=top_k, re_norm=re_norm, mass_aware=mass_aware, mode=name):
                    options = dict(top_k=top_k, top_k_re_norm=re_norm, mass_aware=mass_aware, grad=False)
                    actual = gather_ram_result(self.ram, inputs, chunk_size=1, **options)
                    expected = gather_ram_result(self.ram, inputs, **options)
                    assert_close_with_stats(
                        "AR == PARALLEL",
                        ram_masked_pairs(actual, expected),
                        **BEHAVIOR_TOLERANCE,
                    )

    def test_sequence_and_empty_mask_boundaries(self):
        for seq_len, mask_mode, mass_aware in product(
                (1, 31, 32, 33), ("none", "all_masked", "empty_batch_item"), (False, True),
        ):
            with self.subTest(seq_len=seq_len, mask=mask_mode, mass_aware=mass_aware):
                case = make_ram_case(batch=2, heads=4, order=2, channels=8, seq_len=seq_len, masked=False)
                if mask_mode != "none":
                    mask = torch.zeros(2, seq_len, device="cuda", dtype=torch.bool)
                    if mask_mode == "empty_batch_item":
                        mask[1] = True
                        mask[1, 1::3] = False
                    positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
                    case = replace(case, mask=mask, position_ids=positions)
                # Gradients force the parallel dispatcher even at T=1.
                parallel = gather_ram_result(self.ram, case, top_k=(4, 8), mass_aware=mass_aware)
                ar = gather_ram_result(self.ram, case, top_k=(4, 8), mass_aware=mass_aware, chunk_size=1, grad=False)
                assert_close_with_stats("AR == PARALLEL BOUNDARIES", ram_masked_pairs(ar, parallel), **BEHAVIOR_TOLERANCE)
                for fwd, bwd in (parallel, ar):
                    for value in (*fwd, *(bwd or ())):
                        self.assertTrue(torch.isfinite(value).all())
                    if case.mask is None:
                        continue
                    y, slot, mass = fwd
                    empty = ~case.mask.any(dim=-1)
                    torch.testing.assert_close(slot[empty], case.init_slot[empty].double(), atol=0, rtol=0)
                    torch.testing.assert_close(mass[empty], case.init_mass[empty].double(), atol=0, rtol=0)
                    self.assertEqual(torch.count_nonzero(select_valid_tokens(y, ~case.mask, t_dim=-2)).item(), 0)
                    if bwd is not None:
                        d_w, d_r, d_x, d_gamma, d_slot, d_mass = bwd
                        for grad, t_dim in zip((d_w, d_r, d_x, d_gamma), (-1, -1, -2, -1)):
                            self.assertEqual(torch.count_nonzero(select_valid_tokens(grad, ~case.mask, t_dim)).item(), 0)
                        torch.testing.assert_close(d_slot[empty], case.d_slot[empty].double(), atol=0, rtol=0)
                        torch.testing.assert_close(d_mass[empty], case.d_mass[empty].double(), atol=0, rtol=0)

    def test_inplace_matches_separate_state(self):
        source = make_ram_case(batch=2, heads=4, order=3, radix=3, channels=8, seq_len=17, masked=False)
        for dtype, chunk_size, mass_aware in product(
                (torch.float32, *low_precision_dtypes()), (None, 1), (False, True),
        ):
            with self.subTest(dtype=dtype, chunk_size=chunk_size, mass_aware=mass_aware):
                case = cast_ram_case(source, dtype)
                options = dict(chunk_size=chunk_size, grad=False, mass_aware=mass_aware)
                actual = gather_ram_result(self.ram, case, inplace=True, **options)
                expected = gather_ram_result(self.ram, case, **options)
                # The helper checks storage aliasing on every inplace step.
                for _, val, ref in ram_masked_pairs(actual, expected):
                    torch.testing.assert_close(val, ref, atol=1e-5, rtol=1e-5)


class TorchRamAttentionBehaviorTest(unittest.TestCase):
    requires_cuda = False

    def test_mask_and_positions_match_compact_sequence(self):
        case = make_ram_case(batch=2, heads=4, order=3, channels=8, seq_len=33, device="cpu")
        compact = compact_valid_tokens(case)
        for mass_aware in (False, True):
            with self.subTest(mass_aware=mass_aware):
                actual = gather_ram_result(_torch_ram_attn, case, mass_aware=mass_aware)
                expected = gather_ram_result(_torch_ram_attn, compact, mass_aware=mass_aware)
                for _, val, ref in ram_masked_pairs(actual, expected, res_mask=case.mask):
                    torch.testing.assert_close(val, ref, atol=1e-4, rtol=1e-4)

    def test_inplace_updates_input_state(self):
        case = make_ram_case(batch=2, heads=4, order=3, channels=8, seq_len=17, device="cpu", masked=False)
        for mass_aware in (False, True):
            with self.subTest(mass_aware=mass_aware):
                actual = gather_ram_result(_torch_ram_attn, case, grad=False, mass_aware=mass_aware, inplace=True)
                expected = gather_ram_result(_torch_ram_attn, case, grad=False, mass_aware=mass_aware)
                for _, val, ref in ram_masked_pairs(actual, expected):
                    torch.testing.assert_close(val, ref, atol=1e-5, rtol=1e-5)
