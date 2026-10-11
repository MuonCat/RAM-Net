"""Supported model configurations and training forward/backward contracts."""

import unittest

import torch

from tests.compile_helpers import COMPILE_BACKENDS, assert_compiled_ops, capture_cuda_graph, compile_case
from tests.helpers import assert_close_with_stats
from tests.models.helpers import TOLERANCES_CFG, assert_all_parameters_have_grad, make_cache, make_inputs, make_model

MODEL_SMOKE_CASES = (
    ("default", {}),
    ("default_fused", {
        "fuse_norm": True,
        "hidden_act": "fuse_SwiGLU",
    }),
    ("gelu", {
        "hidden_act": "GELU",
    }),
    ("silu", {
        "hidden_act": "SiLU",
    }),
    ("swish", {
        "hidden_act": "Swish",
    }),
    ("fixed_gamma", {
        "gamma_mode": "fixed",
    }),
    ("learnable_asymmetric_top_k", {
        "gamma_mode": "learnable",
        "addr_top_k": (4, 12),
        "addr_top_k_re_norm": (False, True),
    }),
    ("layerwise_fused_norm", {
        "embed_num": (0, 4),
        "fuse_norm": True,
    }),
    ("alternate_ram_shape", {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "addr_order": 8,
        "addr_radix": 4,
        "value_chs": 32,
        "embed_num": [3, 2],
    }),
    ("stability_controls", {
        "init_mass": 0.125,
        "addr_prior_range": 0.0,
        "gate_initializer_bias": 0.0,
        "ram_attn_eps": 1e-6,
        "initializer_range": 0.01,
    }),
)


class RamNetTrainingTest(unittest.TestCase):
    def test_compile_training_matches_eager(self) -> None:
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_training(backend, fuse_cross_entropy=False)

    def test_compile_fused_linear_training_matches_eager(self) -> None:
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_training(
                    backend, fuse_norm=True, hidden_act="fuse_SwiGLU",
                    fuse_linear_cross_entropy=True,
                )

    def test_compile_gradient_checkpointing_matches_eager(self) -> None:
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_training(
                    backend, gradient_checkpointing=True, fuse_cross_entropy=False,
                )

    def _check_compiled_training(self, backend: str, *, gradient_checkpointing: bool = False, **overrides) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
            **overrides,
        )
        model.train()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # FLA's fused-linear loss explicitly disables compilation.
        fullgraph = not config.fuse_linear_cross_entropy
        with compile_case(model, backend, fullgraph=fullgraph) as (compiled, counter):
            for batch_index, seq_len in enumerate((32, 33, 33)):
                input_ids, mask = make_inputs(config, batch=2, seq_len=seq_len)
                mask[1, 3 + batch_index::5] = False
                input_ids = input_ids.masked_fill(~mask, config.pad_token_id)
                labels = input_ids.masked_fill(~mask, -100)
                results = []
                for forward in (model, compiled):
                    model.zero_grad(set_to_none=True)
                    output = forward(
                        input_ids, attention_mask=mask, labels=labels, use_cache=False,
                    )
                    self.assertIsNone(output.past_key_values)
                    loss = output.loss.detach().clone()
                    logits = None if output.logits is None else output.logits.detach().clone()
                    output.loss.backward()
                    assert_all_parameters_have_grad("COMPILE training", model)
                    grads = {
                        name: parameter.grad.detach().clone()
                        for name, parameter in model.named_parameters() if parameter.requires_grad
                    }
                    results.append((loss, logits, grads))

                ref_loss, ref_logits, ref_grads = results[0]
                loss, logits, grads = results[1]
                pairs = [("loss", loss, ref_loss)]
                self.assertEqual(logits is None, config.fuse_linear_cross_entropy)
                self.assertEqual(ref_logits is None, config.fuse_linear_cross_entropy)
                if logits is not None:
                    pairs.append(("logits", logits, ref_logits))
                pairs.extend((name, grad, ref_grads[name]) for name, grad in grads.items())
                assert_close_with_stats(f"COMPILE {backend} training T={seq_len}", pairs, **TOLERANCES_CFG)
                if batch_index == 2:
                    self.assertEqual(counter.frame_count, previous_frames, "Same-shape training must reuse compiled graphs.")
                previous_frames = counter.frame_count

            assert_compiled_ops(counter, "topk_segment_forward")

    def test_inductor_cuda_graph_training_replay(self) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
            fuse_cross_entropy=False,
        )
        model.train()
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        names = ("loss", "logits", *(name for name, parameter in model.named_parameters() if parameter.requires_grad))
        input_ids, mask = make_inputs(config, batch=2, seq_len=33)
        labels = input_ids.masked_fill(~mask, -100)
        static_inputs = (input_ids.clone(), mask.clone(), labels.clone())

        with compile_case(model, "inductor") as (compiled, counter):
            def step(input_ids, mask, labels):
                output = compiled(input_ids, attention_mask=mask, labels=labels, use_cache=False)
                grads = torch.autograd.grad(output.loss, parameters)
                return (output.loss, output.logits, *grads)

            graph, outputs = capture_cuda_graph(step, *static_inputs)
            for batch_index in range(3):
                input_ids, mask = make_inputs(config, batch=2, seq_len=33)
                mask[1, 3 + batch_index::5] = False
                labels = input_ids.masked_fill(~mask, -100)
                for target, value in zip(static_inputs, (input_ids, mask, labels)):
                    target.copy_(value)

                expected = model(input_ids, attention_mask=mask, labels=labels, use_cache=False)
                grads = torch.autograd.grad(expected.loss, parameters)
                reference = (expected.loss, expected.logits, *grads)
                graph.replay()
                pairs = [(name, value, target) for name, value, target in zip(names, outputs, reference)]
                assert_close_with_stats("COMPILE CUDA Graph training replay", pairs, **TOLERANCES_CFG)

            assert_compiled_ops(counter, "topk_segment_forward")

    def test_gradient_checkpointing_disables_cache(self) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
        )
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        input_ids, mask = make_inputs(config, batch=1, seq_len=6)
        output = model(input_ids, attention_mask=mask, labels=input_ids, use_cache=True)
        self.assertIsNone(output.past_key_values)
        output.loss.backward()
        assert_all_parameters_have_grad("gradient checkpointing", model)

    def test_chunked_cache_preserves_input_gradients(self) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
        )
        block = model.model.layers[0].train()
        hidden = torch.randn(1, 8, config.hidden_size, device="cuda", requires_grad=True)
        positions = torch.arange(8, device="cuda").unsqueeze(0)
        upstream = torch.randn_like(hidden)
        full, _ = block.forward_with_state(hidden, position_ids=positions)
        full_grad, = torch.autograd.grad((full * upstream).sum(), hidden)
        cache = make_cache()
        first, _, _ = block(hidden[:, :4], past_key_values=cache, use_cache=True, position_ids=positions[:, :4])
        second, _, _ = block(hidden[:, 4:], past_key_values=cache, use_cache=True, position_ids=positions[:, 4:])
        chunked_grad, = torch.autograd.grad((torch.cat((first, second), dim=1) * upstream).sum(), hidden)
        torch.testing.assert_close(chunked_grad, full_grad, atol=5e-4, rtol=2e-3)

    def test_config_smoke_cases(self) -> None:
        # Exercises supported configuration branches inherited from MODEL_CFG.
        for name, config_overrides in MODEL_SMOKE_CASES:
            with self.subTest(name=name):
                config, model = make_model(**config_overrides)
                model.train()
                input_ids, attention_mask = make_inputs(config, batch=8, seq_len=512)

                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )

                self.assertEqual(out.logits.shape, (*input_ids.shape, config.vocab_size))
                self.assertTrue(torch.isfinite(out.logits).all())
                self.assertIsNotNone(out.loss)
                self.assertTrue(torch.isfinite(out.loss))
                self.assertIsNotNone(out.hidden_states)
                self.assertEqual(len(out.hidden_states), config.num_hidden_layers + 1)
                if config.tie_word_embeddings:
                    self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
                else:
                    self.assertIsNot(model.get_input_embeddings().weight, model.get_output_embeddings().weight)

                for layer_idx, layer in enumerate(model.model.layers):
                    self.assertEqual(layer.attn.gamma_mode, config.gamma_mode)
                    self.assertEqual(layer.attn.embed_num, config.get_embed_num(layer_idx))
                    self.assertEqual(layer.attn.top_k, config.addr_top_k)
                    self.assertEqual(layer.attn.top_k_re_norm, config.addr_top_k_re_norm)

                out.loss.backward()
                assert_all_parameters_have_grad(name, model)
                model.zero_grad(set_to_none=True)
