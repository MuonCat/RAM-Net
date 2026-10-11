"""Fixed-shape AOT/Inductor compilation and real CUDA Graph replay."""

import unittest

import torch

from tests.compile_helpers import COMPILE_BACKENDS, assert_compiled_ops, capture_cuda_graph, compile_case
from tests.helpers import require_cuda_ops
from tests.ops.cuda_helpers import make_cuda_case, run_cuda_ram


class CudaCompileTest(unittest.TestCase):
    def setUp(self):
        self.ram = require_cuda_ops()

    def test_compile_single_address_forward_and_backward(self):
        from ramnet_ops.impl.cuda_impl import cuda_topk_addr

        for backend in COMPILE_BACKENDS:
            for re_norm in (False, True):
                with self.subTest(backend=backend, re_norm=re_norm):
                    def forward(addr, position_ids):
                        return cuda_topk_addr(addr, 1, position_ids, 3, re_norm, True)

                    with compile_case(forward, backend) as (compiled, counter):
                        for batch_index, seq_len in enumerate((32, 33, 33)):
                            case = make_cuda_case(seq_len=seq_len)
                            results = []
                            for func in (forward, compiled):
                                addr = case.r_addr.detach().clone().requires_grad_()
                                idx, val = func(addr, case.position_ids)
                                grad, = torch.autograd.grad(val.square().sum(), addr)
                                results.append((idx, val, grad))
                            reference, actual = results
                            torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
                            for value, target in zip(actual[1:], reference[1:]):
                                torch.testing.assert_close(value, target, atol=1e-5, rtol=1e-5)
                            if batch_index == 2:
                                self.assertEqual(counter.frame_count, previous_frames)
                            previous_frames = counter.frame_count
                        assert_compiled_ops(counter, "topk_addr_single_forward")

    def test_compile_parallel_forward_and_backward(self):
        for backend in COMPILE_BACKENDS:
            for order, state_grad in ((2, False), (4, True)):
                with self.subTest(backend=backend, order=order, state_grad=state_grad):
                    def forward(w, r, x, gamma, slot, mass, position_ids, mask):
                        return run_cuda_ram(self.ram, (w, r, x, gamma, slot, mass), position_ids, mask, mass_aware=True)

                    with compile_case(forward, backend) as (compiled, counter):
                        for batch_index, seq_len in enumerate((31, 32, 33, 33)):
                            case = make_cuda_case(order=order, seq_len=seq_len)
                            mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
                            mask[:, ::4] = False
                            if batch_index == 3:
                                mask[1] = False
                            originals = (case.w_addr, case.r_addr, case.values, case.gamma, case.init_slot, case.init_mass)
                            upstream = (case.d_y, case.d_slot, case.d_mass)
                            results = []
                            for func in (forward, compiled):
                                inputs = tuple(t.detach().clone().requires_grad_(i < 4 or state_grad) for i, t in enumerate(originals))
                                y, slot, mass = func(*inputs, case.position_ids, mask)
                                outputs = (y.masked_fill(~mask[:, None, :, None], 0), slot, mass)
                                grads = torch.autograd.grad(outputs, inputs if state_grad else inputs[:4], upstream)
                                results.append((*outputs, *grads))
                                for current, original in zip(inputs[4:], originals[4:]):
                                    torch.testing.assert_close(current, original, atol=0, rtol=0)
                            reference, actual = results
                            for value, target in zip(actual, reference):
                                torch.testing.assert_close(value, target, atol=5e-4, rtol=5e-4)
                            if batch_index == 3:
                                self.assertEqual(counter.frame_count, previous_frames)
                            previous_frames = counter.frame_count
                        assert_compiled_ops(counter, "topk_addr_forward", "topk_segment_forward")

    def test_compile_autoregressive_inplace(self):
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_inplace(backend, seq_len=1)

    def test_compile_parallel_inplace(self):
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_inplace(backend, seq_len=33)

    def _check_compiled_inplace(self, backend: str, *, seq_len: int):
        case = make_cuda_case(seq_len=seq_len)
        ref_slot, ref_mass = case.init_slot.clone(), case.init_mass.clone()
        slot, mass = case.init_slot.clone(), case.init_mass.clone()

        def forward(w, r, x, gamma, slot, mass, position_ids, mask):
            return run_cuda_ram(
                self.ram, (w, r, x, gamma, slot, mass), position_ids, mask,
                mass_aware=True, inplace=True,
            )

        with compile_case(forward, backend) as (compiled, counter):
            for step in range(3):
                case = make_cuda_case(seq_len=seq_len)
                mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
                mask[step % 2, ::2] = False
                positions = case.position_ids + step * seq_len
                reference = forward(case.w_addr, case.r_addr, case.values, case.gamma, ref_slot, ref_mass, positions, mask)
                actual = compiled(case.w_addr, case.r_addr, case.values, case.gamma, slot, mass, positions, mask)
                _, post_slot, post_mass = actual
                self.assertEqual(post_slot.data_ptr(), slot.data_ptr())
                self.assertEqual(post_mass.data_ptr(), mass.data_ptr())
                for value, target in zip(actual, reference):
                    torch.testing.assert_close(value, target, atol=5e-4, rtol=5e-4)
                if step:
                    self.assertEqual(counter.frame_count, previous_frames)
                previous_frames = counter.frame_count
            operator = "topk_autoregressive_" if seq_len == 1 else "topk_segment_forward_"
            assert_compiled_ops(counter, operator)

    def test_inductor_cuda_graph_replay(self):
        for seq_len, inplace in ((1, True), (33, False), (33, True)):
            with self.subTest(seq_len=seq_len, inplace=inplace):
                case = make_cuda_case(seq_len=seq_len)
                static_inputs = tuple(t.clone() for t in (
                    case.w_addr, case.r_addr, case.values, case.gamma, case.init_slot, case.init_mass,
                    case.position_ids, torch.ones(2, seq_len, device="cuda", dtype=torch.bool),
                ))

                def forward(w, r, x, gamma, slot, mass, position_ids, mask):
                    return run_cuda_ram(
                        self.ram, (w, r, x, gamma, slot, mass), position_ids, mask,
                        mass_aware=True, inplace=inplace,
                    )

                with compile_case(forward, "inductor") as (compiled, counter):
                    graph, outputs = capture_cuda_graph(compiled, *static_inputs)
                    for step in range(3):
                        case = make_cuda_case(seq_len=seq_len)
                        mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
                        mask[step % 2, ::2] = False
                        inputs = (
                            case.w_addr, case.r_addr, case.values, case.gamma, case.init_slot, case.init_mass,
                            case.position_ids + step * seq_len, mask,
                        )
                        for target, value in zip(static_inputs, inputs):
                            target.copy_(value)
                        reference = forward(*(t.clone() for t in inputs))
                        graph.replay()
                        for actual, expected in zip(outputs, reference):
                            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
                        _, post_slot, post_mass = outputs
                        if inplace:
                            self.assertEqual(post_slot.data_ptr(), static_inputs[4].data_ptr())
                            self.assertEqual(post_mass.data_ptr(), static_inputs[5].data_ptr())
                        else:
                            torch.testing.assert_close(static_inputs[4], case.init_slot, atol=0, rtol=0)
                            torch.testing.assert_close(static_inputs[5], case.init_mass, atol=0, rtol=0)
                    operator = "topk_autoregressive_" if seq_len == 1 else "topk_segment_forward_" if inplace else "topk_segment_forward"
                    assert_compiled_ops(counter, operator)
