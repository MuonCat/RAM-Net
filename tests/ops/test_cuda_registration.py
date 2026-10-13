"""Dispatcher schemas, fake tensors, and segment capacity contracts."""

from itertools import product
import unittest

import torch

from tests.helpers import require_cuda_ops
from tests.ops.cuda_helpers import make_cuda_case, run_cuda_ram


class CudaRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.ram = require_cuda_ops()
        self.int_dtype = torch.int32

    def test_functional_registrations(self):
        from ramnet_ops.impl.cuda_impl import gather_segment

        checks = ("test_schema", "test_autograd_registration", "test_faketensor")
        for order in (2, 4):
            with self.subTest(order=order):
                case = make_cuda_case(order=order)
                w = case.w_addr.requires_grad_()
                r = case.r_addr.requires_grad_()
                pos = case.position_ids.to(self.int_dtype)
                torch.library.opcheck(
                    torch.ops.ramnet_ops.topk_addr_single_forward,
                    (w, 1, 4 ** order, pos, 2, False),
                    test_utils=checks,
                )
                for (addr, k, re_norm, neg), encode_batch_idx in product(
                        ((w, 2, False, False), (r, 3, True, True)), (False, True),
                ):
                    torch.library.opcheck(
                        torch.ops.ramnet_ops.topk_addr_single_forward,
                        (
                            addr,
                            1, 4 ** order, pos,
                            k,
                            re_norm, neg,
                        ),
                        kwargs={"encode_batch_idx": encode_batch_idx},
                        test_utils=checks,
                    )
                args = (
                    w, r,
                    1, 4 ** order, pos,
                    2, 3,
                    False, True,
                )
                for encode_batch_idx in (False, True):
                    torch.library.opcheck(
                        torch.ops.ramnet_ops.topk_addr_forward, args,
                        kwargs={"encode_batch_idx": encode_batch_idx}, test_utils=checks,
                    )
                idx, val = torch.ops.ramnet_ops.topk_addr_forward(*args)
                meta = gather_segment(idx, None, 4 ** order)
                args = (
                    val,
                    meta.seg_idx, meta.seg_off, meta.seg_len, meta.event_i,
                    case.values.requires_grad_(), case.gamma.requires_grad_(),
                    case.init_slot, case.init_mass,
                    2, 3,
                    True, 1e-6,
                )
                torch.library.opcheck(torch.ops.ramnet_ops.topk_segment_forward, args, test_utils=checks)

    def test_inplace_schemas(self):
        from ramnet_ops.impl.cuda_impl import gather_segment

        case = make_cuda_case()
        pos = case.position_ids.to(self.int_dtype)
        idx, val = torch.ops.ramnet_ops.topk_addr_forward(
            case.w_addr, case.r_addr,
            1, 16, pos,
            2, 3,
            False, True,
        )
        meta = gather_segment(idx, None, 16)
        torch.library.opcheck(
            torch.ops.ramnet_ops.topk_segment_forward_,
            (
                val,
                meta.seg_idx, meta.seg_off, meta.seg_len, meta.event_i,
                case.values, case.gamma,
                case.init_slot.clone(), case.init_mass.clone(),
                2, 3,
                True, 1e-6,
            ),
            test_utils=("test_schema", "test_faketensor"),
        )
        torch.library.opcheck(
            torch.ops.ramnet_ops.topk_autoregressive_,
            (
                case.w_addr[..., 0], case.r_addr[..., 0],
                case.values[..., 0, :], case.gamma[..., 0],
                case.init_slot.clone(), case.init_mass.clone(),
                torch.empty(0, device="cuda", dtype=torch.bool), 1, pos[:, 0],
                2, 3,
                False, True,
                True, 1e-6,
            ),
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_all_masked_state_identity_and_capacity(self):
        from ramnet_ops.impl.cuda_impl import gather_segment

        for order in (2, 4):
            with self.subTest(order=order):
                case = make_cuda_case(order=order)
                inputs = tuple(t.detach().requires_grad_() for t in (
                    case.w_addr, case.r_addr, case.values, case.gamma, case.init_slot, case.init_mass))
                mask = torch.zeros(2, 32, device="cuda", dtype=torch.bool)
                idx, _ = torch.ops.ramnet_ops.topk_addr_forward(
                    inputs[0], inputs[1],
                    1, 4 ** order, case.position_ids.to(self.int_dtype),
                    2, 3,
                    False, True,
                )
                for layout_mask in (None, mask):
                    meta = gather_segment(idx, layout_mask, 4 ** order)
                    self.assertEqual(meta.seg_idx.numel(), 4 * (4 ** order))
                    self.assertEqual(meta.event_i.numel(), 4 * 32 * 5)
                outputs = run_cuda_ram(self.ram, inputs, case.position_ids, mask, mass_aware=True)
                torch.testing.assert_close(outputs[0], torch.zeros_like(outputs[0]), atol=0, rtol=0)
                for actual, initial in zip(outputs[1:], inputs[4:]):
                    self.assertNotEqual(actual.data_ptr(), initial.data_ptr())
                    torch.testing.assert_close(actual, initial, atol=0, rtol=0)
                grads = torch.autograd.grad(outputs[1].sum() + outputs[2].sum(), inputs)
                for grad in grads[:4]:
                    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)
                for grad in grads[4:]:
                    torch.testing.assert_close(grad, torch.ones_like(grad), atol=0, rtol=0)
