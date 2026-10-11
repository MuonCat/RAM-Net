"""Address weights and gradients: FP32 backends and CUDA precision."""

from itertools import product
import unittest

import torch

from ramnet_ops.impl.torch_impl import addr_decode

from tests.helpers import assert_close_with_stats, require_cuda_ops
from tests.ops.cuda_helpers import make_cuda_case
from tests.ops.precision import low_precision_dtypes, requires_low_precision
from tests.ops.tolerances import FP32_TOLERANCE, PRECISION_TOLERANCES


class AddressNumericsTest(unittest.TestCase):
    def setUp(self):
        require_cuda_ops()
        from ramnet_ops.impl.cuda_impl import cuda_topk_addr

        self.addr = cuda_topk_addr

    def test_fp32_matches_torch(self):
        case = make_cuda_case(seq_len=31)
        for re_norm, neg in product((False, True), repeat=2):
            with self.subTest(re_norm=re_norm, neg=neg):
                source = case.r_addr if neg else case.w_addr
                addr = source.detach().requires_grad_()
                idx, val = self.addr(addr, 1, case.position_ids, 3, re_norm, neg)
                log_addr = addr.log_softmax(dim=-2).permute(4, 0, 1, 2, 3)
                dense = addr_decode(log_addr, case.position_ids, 1).permute(1, 2, 3, 0)
                # Check membership independently, then compare weights in native index order.
                selected = dense.gather(-2, (idx % 16).long())
                torch.testing.assert_close(
                    selected.sort(dim=-2).values,
                    dense.topk(3, dim=-2).values.sort(dim=-2).values,
                    atol=1e-5, rtol=1e-5,
                )
                ref = selected / selected.sum(dim=-2, keepdim=True) if re_norm else selected
                ref = -ref if neg else ref
                grad = torch.randn_like(val)
                actual, = torch.autograd.grad(val, addr, grad)
                expected, = torch.autograd.grad(ref, addr, grad)
                assert_close_with_stats(
                    "ADDRESS CUDA == TORCH FP32",
                    [("val", val, ref), ("D_addr", actual, expected)],
                    **FP32_TOLERANCE,
                )

    @requires_low_precision
    def test_low_precision_matches_cuda_fp32(self):
        dtypes = low_precision_dtypes(required=True)
        case = make_cuda_case(seq_len=31)
        for dtype, re_norm, neg in product(dtypes, (False, True), (False, True)):
            with self.subTest(dtype=dtype, re_norm=re_norm, neg=neg):
                source = case.r_addr if neg else case.w_addr
                low = source.to(dtype).detach().requires_grad_()
                fp32 = low.detach().float().requires_grad_()
                idx, val = self.addr(low, 1, case.position_ids, 3, re_norm, neg)
                ref_idx, ref_val = self.addr(fp32, 1, case.position_ids, 3, re_norm, neg)
                # Sorting removes arbitrary ordering among equal Top-K weights.
                idx, order = idx.sort(dim=-2)
                ref_idx, ref_order = ref_idx.sort(dim=-2)
                torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)
                val = val.gather(-2, order)
                ref_val = ref_val.gather(-2, ref_order)
                grad = torch.randn_like(val)
                actual, = torch.autograd.grad(val, low, grad)
                expected, = torch.autograd.grad(ref_val, fp32, grad)
                assert_close_with_stats(
                    "ADDRESS CUDA LOW PRECISION == CUDA FP32",
                    [("val", val, ref_val), ("D_addr", actual, expected)],
                    **PRECISION_TOLERANCES[dtype],
                )
