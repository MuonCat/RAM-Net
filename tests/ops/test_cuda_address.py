"""CUDA address gradients and index bounds."""

from itertools import product
import unittest

import torch

from ramnet_ops.impl.torch_impl import addr_decode

from tests.helpers import require_cuda_ops
from tests.ops.cuda_helpers import make_cuda_case
from tests.ops.precision import low_precision_dtypes


class CudaAddressTest(unittest.TestCase):
    def setUp(self):
        require_cuda_ops()
        self.int_dtype = torch.int32

    def test_address_gradient_convention(self):
        for re_norm in (False, True):
            with self.subTest(re_norm=re_norm):
                case = make_cuda_case()
                w, r = (t.detach().requires_grad_() for t in (case.w_addr, case.r_addr))
                idx, val = torch.ops.ramnet_ops.topk_addr_forward(
                    w, r,
                    1, 16, case.position_ids.to(self.int_dtype),
                    2, 3,
                    re_norm, re_norm,
                )
                refs = []
                for addr, indices, k, sign in ((w, idx[:, :, :2], 2, 1), (r, idx[:, :, 2:], 3, -1)):
                    log_addr = addr.log_softmax(dim=-2).permute(4, 0, 1, 2, 3)
                    dense = addr_decode(log_addr, case.position_ids, 1, k, re_norm).permute(1, 2, 3, 0)
                    refs.append(sign * dense.gather(-2, (indices % 16).long()))
                ref = torch.cat(refs, dim=-2)
                # The private addr/segment chain passes gradients of positive read weights.
                w_ref, r_ref = refs
                ref_weights = torch.cat((w_ref, -r_ref), dim=-2)
                grad = torch.randn_like(val)
                actual_grads = torch.autograd.grad(val, (w, r), grad)
                expected_grads = torch.autograd.grad(ref_weights, (w, r), grad)
                torch.testing.assert_close(val, ref, atol=1e-5, rtol=1e-5)
                for actual, expected in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_single_address_wrapper_contract(self):
        from ramnet_ops.impl.cuda_impl import cuda_topk_addr

        case = make_cuda_case(seq_len=31)
        for dtype, re_norm, neg in product(
                (torch.float32, *low_precision_dtypes()), (False, True), (False, True),
        ):
            with self.subTest(dtype=dtype, re_norm=re_norm, neg=neg):
                source = case.r_addr if neg else case.w_addr
                addr = source.to(dtype).detach().requires_grad_()
                original = addr.detach().clone()
                idx, val = cuda_topk_addr(addr, 1, case.position_ids, 3, re_norm, neg)
                self.assertEqual(idx.shape, (2, 2, 3, 31))
                self.assertEqual(idx.dtype, self.int_dtype)
                self.assertEqual(val.dtype, torch.float32)
                self.assertFalse(idx.requires_grad)
                grad = torch.randn_like(val)
                actual, = torch.autograd.grad(val, addr, grad)
                self.assertTrue(torch.isfinite(val).all())
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(addr, original, atol=0, rtol=0)

    def test_address_and_padded_index_limits(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from ramnet_ops.impl._cuda_config import INT_BITS, INT_MAX
        from ramnet_ops.impl.cuda_impl import cuda_topk_addr

        # Reject invalid metadata before allocating outputs or launching native kernels.
        with FakeTensorMode():
            addr = torch.empty(1, 1, INT_BITS - 2, 2, 32, device="cuda", dtype=torch.float32)
            pos = torch.empty(1, 32, device="cuda", dtype=self.int_dtype)
            with self.assertRaisesRegex(AssertionError, "Address space overflow"):
                cuda_topk_addr(addr, 0, pos, 1, True)

            if INT_BITS == 32:
                # The original T fits; its next multiple of 32 exceeds the index limit.
                T = INT_MAX // 32 * 32 + 1
                addr = torch.empty(1, 1, 1, 1, T, device="cuda", dtype=torch.float32)
                pos = torch.empty(1, T, device="cuda", dtype=self.int_dtype)
                with self.assertRaisesRegex(AssertionError, "CUDA index overflow"):
                    cuda_topk_addr(addr, 0, pos, 1, True)
