"""Address weights and gradients: Torch reference and CUDA precision."""

from itertools import product
import unittest

import torch

from ramnet_ops import topk_addr
from ramnet_ops.impl.torch_impl import addr_decode, _torch_topk_addr

from tests.helpers import assert_close_with_stats, require_cuda_ops
from tests.ops.cuda_helpers import make_cuda_case
from tests.ops.precision import low_precision_dtypes, requires_low_precision
from tests.ops.tolerances import FP32_TOLERANCE, PRECISION_TOLERANCES


class TorchAddressNumericsTest(unittest.TestCase):
    requires_cuda = False

    def test_weights_and_gradients_match_product_distribution(self):
        B, H, U, P, T = 2, 2, 2, 3, 5
        A = P ** U
        positions = torch.arange(T).unsqueeze(0) * 3 + torch.arange(B).unsqueeze(1)
        for dtype, k, re_norm, neg, encode_batch_idx in product(
                (torch.float32, torch.float64), (1, 4, A), (False, True), (False, True), (False, True),
        ):
            with self.subTest(dtype=dtype, k=k, re_norm=re_norm, neg=neg, encode_batch_idx=encode_batch_idx):
                addr = torch.randn(B, H, U, P, T * 2, dtype=dtype)[..., ::2].requires_grad_()
                original = addr.detach().clone()
                idx, val = _torch_topk_addr(addr, 1, positions, k, re_norm, neg, encode_batch_idx)
                self.assertEqual(idx.shape, (B, H, k, T))
                self.assertEqual(val.shape, idx.shape)
                self.assertEqual(idx.dtype, torch.int32)
                self.assertEqual(val.dtype, dtype)
                self.assertFalse(idx.requires_grad)

                index_batches = B if encode_batch_idx else 1
                slot_offset = torch.arange(index_batches * H).view(index_batches, H, 1, 1) * A
                local_idx = idx.long() - slot_offset
                self.assertTrue(((local_idx >= 0) & (local_idx < A)).all())
                shift = positions[:, None, None, :] * (torch.arange(H).view(1, H, 1, 1) < 1)
                content_idx = (local_idx + shift).remainder(A)

                # Independent probability-space reference; digit 0 is least significant.
                probs = addr.softmax(dim=-2)
                dense = (probs[:, :, 1, :, None, :] * probs[:, :, 0, None, :, :]).flatten(-3, -2)
                selected = dense.gather(-2, content_idx)
                torch.testing.assert_close(selected.sort(dim=-2).values, dense.topk(k, dim=-2).values.sort(dim=-2).values)
                ref = selected / selected.sum(dim=-2, keepdim=True) if re_norm else selected
                ref = -ref if neg else ref
                grad = torch.randn_like(val)
                actual, = torch.autograd.grad(val, addr, grad)
                expected, = torch.autograd.grad(ref, addr, grad)
                torch.testing.assert_close(val, ref, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(addr, original, atol=0, rtol=0)

    def test_inactive_addresses_have_zero_weights_and_gradients(self):
        source = torch.randn(2, 2, 2, 3, 3)
        source[..., 0] = -torch.inf
        source[:, :, 0, :, 1] = -torch.inf
        positions = torch.arange(3).expand(2, -1)
        for re_norm, neg in product((False, True), repeat=2):
            with self.subTest(re_norm=re_norm, neg=neg):
                addr = source.clone().requires_grad_()
                _, val = _torch_topk_addr(addr, 1, positions, 4, re_norm, neg)
                grad, = torch.autograd.grad(val, addr, torch.randn_like(val))
                torch.testing.assert_close(val[..., :2], torch.zeros_like(val[..., :2]), atol=0, rtol=0)
                torch.testing.assert_close(grad[..., :2], torch.zeros_like(grad[..., :2]), atol=0, rtol=0)
                self.assertTrue(torch.isfinite(val).all())
                self.assertTrue(torch.isfinite(grad).all())

    @requires_low_precision
    def test_low_precision_uses_fp32_weights(self):
        for dtype, re_norm in product(low_precision_dtypes(required=True), (False, True)):
            with self.subTest(dtype=dtype, re_norm=re_norm):
                addr = torch.randn(2, 2, 2, 3, 5).to(dtype).requires_grad_()
                fp32 = addr.detach().float().requires_grad_()
                positions = torch.arange(5).expand(2, -1)
                idx, val = _torch_topk_addr(addr, 1, positions, 4, re_norm)
                ref_idx, ref_val = _torch_topk_addr(fp32, 1, positions, 4, re_norm)
                self.assertEqual(val.dtype, torch.float32)
                torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)
                torch.testing.assert_close(val, ref_val, atol=0, rtol=0)
                grad = torch.randn_like(val)
                actual, = torch.autograd.grad(val, addr, grad)
                expected, = torch.autograd.grad(ref_val, fp32, grad)
                torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)


class AddressNumericsTest(unittest.TestCase):
    def setUp(self):
        require_cuda_ops()
        from ramnet_ops.impl.cuda_impl import _cuda_topk_addr

        self.addr = _cuda_topk_addr

    def test_public_backends_match(self):
        case = make_cuda_case(seq_len=31)
        for re_norm, neg, encode_batch_idx in product((False, True), repeat=3):
            with self.subTest(re_norm=re_norm, neg=neg, encode_batch_idx=encode_batch_idx):
                inputs, outputs = [], []
                for backend in ('cuda', 'torch'):
                    addr = case.w_addr.detach().requires_grad_()
                    idx, val = topk_addr(addr, 1, case.position_ids, 3, re_norm, neg, encode_batch_idx, impl_backend=backend)
                    idx, order = idx.sort(dim=-2)
                    inputs.append(addr)
                    outputs.append((idx, val.gather(-2, order)))
                (idx, val), (ref_idx, ref_val) = outputs
                torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)
                grad = torch.randn_like(val)
                actual, = torch.autograd.grad(val, inputs[0], grad)
                expected, = torch.autograd.grad(ref_val, inputs[1], grad)
                assert_close_with_stats(
                    "PUBLIC ADDRESS CUDA == TORCH",
                    [("val", val, ref_val), ("D_addr", actual, expected)],
                    **FP32_TOLERANCE,
                )

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
