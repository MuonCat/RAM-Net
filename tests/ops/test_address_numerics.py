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
        for re_norm, neg, encode_batch_idx in product((False, True), repeat=3):
            with self.subTest(re_norm=re_norm, neg=neg, encode_batch_idx=encode_batch_idx):
                source = case.r_addr if neg else case.w_addr
                addr = source.detach().requires_grad_()
                idx, val = self.addr(addr, 1, case.position_ids, 3, re_norm, neg, encode_batch_idx=encode_batch_idx)
                B, H = addr.shape[:2]
                index_batches = B if encode_batch_idx else 1
                idx = idx - torch.arange(index_batches * H, device=idx.device, dtype=idx.dtype).reshape(index_batches, H, 1, 1) * 16
                log_addr = addr.log_softmax(dim=-2).permute(4, 0, 1, 2, 3)
                dense = addr_decode(log_addr, case.position_ids, 1).permute(1, 2, 3, 0)
                # Check membership independently, then compare weights in native index order.
                selected = dense.gather(-2, idx.long())
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
        for dtype, re_norm, neg, encode_batch_idx in product(dtypes, (False, True), (False, True), (False, True)):
            with self.subTest(dtype=dtype, re_norm=re_norm, neg=neg, encode_batch_idx=encode_batch_idx):
                source = case.r_addr if neg else case.w_addr
                low = source.to(dtype).detach().requires_grad_()
                fp32 = low.detach().float().requires_grad_()
                idx, val = self.addr(low, 1, case.position_ids, 3, re_norm, neg, encode_batch_idx=encode_batch_idx)
                ref_idx, ref_val = self.addr(fp32, 1, case.position_ids, 3, re_norm, neg, encode_batch_idx=encode_batch_idx)
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
