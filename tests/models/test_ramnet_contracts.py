"""Model initialization, API contracts, cache behavior, and configuration forwarding."""

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.models.helpers import (
    assert_cache_length,
    load_model_classes,
    make_cache,
    make_inputs,
    make_model,
)


class RamNetContractsTest(unittest.TestCase):
    def test_standalone_and_model_initialization_match(self) -> None:
        for gamma_mode, optional_parameters in (
                ("fixed", True), ("learnable", True), ("dynamic", True), ("dynamic", False),
        ):
            with self.subTest(gamma_mode=gamma_mode, optional_parameters=optional_parameters):
                config, model = make_model(
                    hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                    addr_order=2, addr_radix=3, value_chs=8, vocab_size=64,
                    gamma_mode=gamma_mode, initializer_range=0.03,
                    addr_prior_range=0.75, gate_initializer_bias=2.0,
                    addr_proj_bias=optional_parameters, value_proj_bias=optional_parameters,
                    address_scaling=optional_parameters, output_norm=optional_parameters,
                    output_gate="sigmoid" if optional_parameters else None,
                )
                reference = deepcopy(model.model.layers[0].attn).cpu()
                standalone = deepcopy(reference)
                with torch.no_grad():
                    for attention in (reference, standalone):
                        for parameter in attention.parameters():
                            parameter.fill_(float("nan"))

                with torch.random.fork_rng(devices=[]):
                    rng_state = torch.get_rng_state()
                    reference.apply(model._init_weights)
                    torch.set_rng_state(rng_state)
                    standalone._init_weights(
                        initializer_range=config.initializer_range,
                        addr_prior_range=config.addr_prior_range,
                        gate_initializer_bias=config.gate_initializer_bias,
                    )

                expected = {name: value.clone() for name, value in reference.state_dict().items()}
                # A parent callback must not reinitialize its initialized child modules.
                model._init_weights(reference)
                for attention in (reference, standalone):
                    for name, value in attention.state_dict().items():
                        torch.testing.assert_close(value, expected[name], atol=0, rtol=0, msg=name)

                for parameter in standalone.parameters(recurse=False):
                    torch.testing.assert_close(parameter, torch.zeros_like(parameter))
                if optional_parameters:
                    address_bias = torch.tensor([[0., 0., 0.], [-0.75, 0., 0.75]]).expand(4, 2, 3).flatten()
                    for projection in (standalone.r_proj, standalone.w_proj):
                        torch.testing.assert_close(projection.bias, address_bias)
                    torch.testing.assert_close(standalone.o_norm.weight, torch.ones_like(standalone.o_norm.weight))
                    gate_bias = standalone.o_gate[0].bias
                    torch.testing.assert_close(gate_bias, torch.full_like(gate_bias, 2.0))

    def test_mass_aware_config_reaches_attention_op(self) -> None:
        for mass_aware in (False, True):
            with self.subTest(mass_aware=mass_aware):
                config, model = make_model(
                    hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                    addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
                    mass_aware=mass_aware,
                )
                import ramnet.layers.ram_attn_block as ram_attn_block

                self.assertIs(config.mass_aware, mass_aware)
                for layer in model.model.layers:
                    self.assertIs(layer.attn.mass_aware, mass_aware)
                input_ids, attention_mask = make_inputs(config, batch=1, seq_len=5)
                model.eval()
                with torch.no_grad(), patch.object(
                        ram_attn_block, "random_access_memory_attention",
                        wraps=ram_attn_block.random_access_memory_attention,
                ) as ram_attn:
                    out = model(input_ids, attention_mask=attention_mask, use_cache=False)
                self.assertTrue(torch.isfinite(out.logits).all())
                self.assertEqual(ram_attn.call_count, config.num_hidden_layers)
                for call in ram_attn.call_args_list:
                    self.assertIs(call.kwargs["mass_aware"], mass_aware)

    def test_attention_inplace_selection(self) -> None:
        config, model = make_model()
        attention = model.model.layers[0].attn
        cache = make_cache()
        hidden_states = torch.randn(1, 1, config.hidden_size, device="cuda")

        def fake_ram_attention(*, values, init_ram, inplace, **kwargs):
            slot, mass = init_ram
            if slot is None:
                A = config.addr_radix ** config.addr_order
                slot = values.new_zeros(1, config.num_attention_heads, A, config.value_chs)
                mass = values.new_full((1, config.num_attention_heads, A), mass)
            return values, (slot, mass) if inplace else (slot.clone(), mass.clone())

        import ramnet.layers.ram_attn_block as ram_attn_block

        with patch.object(ram_attn_block, "random_access_memory_attention", side_effect=fake_ram_attention) as ram_attn:
            attention.eval()
            with torch.no_grad():
                attention(hidden_states, past_key_values=cache, use_cache=True)
            self.assertFalse(ram_attn.call_args.kwargs["inplace"])
            state = cache[0]["recurrent_state"]

            with torch.no_grad():
                attention(hidden_states, past_key_values=cache, use_cache=True)
            self.assertTrue(ram_attn.call_args.kwargs["inplace"])
            self.assertIs(cache[0]["recurrent_state"].slot, state.slot)

            attention.train()
            attention(hidden_states, past_key_values=cache, use_cache=True)
            self.assertFalse(ram_attn.call_args.kwargs["inplace"])

            attention.eval()
            attention(hidden_states, past_key_values=cache, use_cache=True)
            self.assertFalse(ram_attn.call_args.kwargs["inplace"])

            state = cache[0]["recurrent_state"]
            with torch.no_grad():
                _, post_state = attention.forward_with_state(
                    hidden_states, init_ram=(state.slot, state.mass), inplace=True,
                )
            self.assertTrue(ram_attn.call_args.kwargs["inplace"])
            self.assertIs(post_state.slot, state.slot)
            self.assertIs(post_state.mass, state.mass)
        self.assertEqual(cache.get_seq_length(), 4)

    def test_use_cache_flag_cases(self) -> None:
        RAMCache, _, _ = load_model_classes()
        config, model = make_model(use_cache=False)
        model.eval()
        input_ids, attention_mask = make_inputs(config, seq_len=10)
        cache = make_cache()

        with torch.no_grad():
            disabled = model(input_ids, attention_mask=attention_mask, past_key_values=cache)
            enabled = model(input_ids, attention_mask=attention_mask, use_cache=True)

        self.assertIsNone(disabled.past_key_values)
        assert_cache_length("USE_CACHE disabled", cache, config, 0)
        self.assertIsInstance(enabled.past_key_values, RAMCache)
        assert_cache_length("USE_CACHE enabled", enabled.past_key_values, config, input_ids.size(1))

    def test_hf_outputs_and_checkpoint_round_trip(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
            use_cache=False, tie_word_embeddings=True,
        )
        model.eval()
        input_ids, mask = make_inputs(config, batch=1, seq_len=6)
        with torch.no_grad():
            expected = model(input_ids, attention_mask=mask, output_hidden_states=True)
            actual = model(
                inputs_embeds=model.get_input_embeddings()(input_ids),
                attention_mask=mask, output_hidden_states=True, return_dict=False,
            )
            torch.testing.assert_close(actual[0], expected.logits)
            self.assertEqual(len(actual[1]), config.num_hidden_layers + 1)
            tail = model(input_ids, attention_mask=mask, logits_to_keep=2)
            torch.testing.assert_close(tail.logits, expected.logits[:, -2:])

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored_config = AutoConfig.from_pretrained(directory)
            self.assertFalse(restored_config.use_cache)
            restored = AutoModelForCausalLM.from_pretrained(directory).cuda().eval()
            self.assertIs(restored.get_input_embeddings().weight, restored.get_output_embeddings().weight)
            restored_state = restored.state_dict()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored_state[name], value, atol=0, rtol=0, msg=name)
            with torch.no_grad():
                torch.testing.assert_close(restored(input_ids, attention_mask=mask).logits, expected.logits)

    def test_cache_batch_operations_and_reset(self) -> None:
        load_model_classes()
        from ramnet import RAMState

        cache = make_cache()
        state = RAMState(
            torch.randn(2, 4, 16, 8, device="cuda"),
            torch.rand(2, 4, 16, device="cuda"),
        )
        positions = torch.tensor([[3], [5]], device="cuda")
        cache.update(state, layer_idx=0, offset=6, cache_kwargs={"next_position": positions})
        self.assertIs(cache[0]["recurrent_state"].slot, state.slot)

        cache.batch_repeat_interleave(2)
        indices = torch.tensor([3, 0, 2], device="cuda")
        cache.reorder_cache(indices)
        reference_indices = torch.tensor([1, 0, 1], device="cuda")
        for value, reference in zip(cache[0]["recurrent_state"], state):
            torch.testing.assert_close(value, reference[reference_indices])
        torch.testing.assert_close(cache.get_layer(0).next_position, positions[reference_indices])
        self.assertEqual(cache.get_seq_length(), 6)
        self.assertEqual(cache.max_batch_size, 3)
        self.assertEqual(cache.get_max_cache_shape(), -1)

        cache.batch_select_indices(torch.tensor([2], device="cuda"))
        self.assertEqual(cache[0]["recurrent_state"].slot.size(0), 1)
        with self.assertRaisesRegex(ValueError, "roll back"):
            cache.crop(5)
        cache.reset()
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertIsNone(cache[0]["recurrent_state"])
        self.assertIsNone(cache.get_layer(0).next_position)
        cache.update(state, layer_idx=0, offset=2)
        self.assertEqual(cache.get_seq_length(), 2)

    def test_model_rejects_unsupported_inputs(self) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
        )
        input_ids, mask = make_inputs(config, batch=1, seq_len=6)
        cases = (
            ({"cu_seqlens": torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)}, ValueError, "packed cu_seqlens"),
            ({"output_attentions": True}, ValueError, "attention matrices"),
            ({"past_key_values": ()}, TypeError, "RAMCache"),
        )
        for options, error, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(error, message):
                model(input_ids, attention_mask=mask, use_cache=False, **options)
