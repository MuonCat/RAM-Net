"""Manual and Hugging Face generation with and without RAMCache."""

import unittest
from unittest.mock import patch

import torch

from tests.compile_helpers import COMPILE_BACKENDS, assert_compiled_ops, compile_case
from tests.helpers import assert_close_with_stats
from tests.models.helpers import (
    TOLERANCES_CFG,
    assert_cache_length,
    assert_logits_close,
    assert_tokens_equal,
    make_cache,
    make_inputs,
    make_model,
)

GEN_MAX_NEW_TOKENS = 2


class RamNetGenerationTest(unittest.TestCase):
    def test_compile_hf_generation_matches_eager(self) -> None:
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_generation(backend, num_beams=1)

    def test_compile_hf_beam_generation_matches_eager(self) -> None:
        for backend in COMPILE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_compiled_generation(backend, num_beams=3)

    def _check_compiled_generation(self, backend: str, *, num_beams: int) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
        )
        model.eval()
        new_tokens = 4
        # Keep HF's generation loop in Python; cache bookkeeping may break graphs.
        with compile_case(model.forward, backend, fullgraph=False) as (compiled_forward, counter):
            for prompt_index, seq_len in enumerate((8, 9, 9)):
                input_ids, mask = make_inputs(config, batch=2, seq_len=seq_len)
                mask[1, :3 + prompt_index] = False
                input_ids = input_ids.masked_fill(~mask, config.pad_token_id)
                generation_kwargs = dict(
                    attention_mask=mask, max_new_tokens=new_tokens, do_sample=False,
                    num_beams=num_beams, use_cache=True,
                    pad_token_id=config.pad_token_id, eos_token_id=None,
                    return_dict_in_generate=True, output_logits=True,
                )
                with torch.no_grad():
                    expected = model.generate(input_ids, **generation_kwargs)
                    # Patch forward so generate cannot bypass the compiled callable.
                    with patch.object(model, "forward", compiled_forward):
                        actual = model.generate(input_ids, **generation_kwargs)

                assert_tokens_equal(f"COMPILE {backend} HF generation", actual.sequences, expected.sequences)
                self.assertEqual(len(actual.logits), new_tokens)
                self.assertEqual(len(expected.logits), new_tokens)
                assert_logits_close(
                    "COMPILE generation logits",
                    torch.stack(actual.logits, dim=1), torch.stack(expected.logits, dim=1),
                )
                # The last generated token has not yet been consumed by the model.
                cache_length = input_ids.size(1) + new_tokens - 1
                assert_cache_length("COMPILE generation", actual.past_key_values, config, cache_length)
                assert_cache_length("EAGER generation", expected.past_key_values, config, cache_length)
                next_position = (mask.long().sum(-1, keepdim=True) + new_tokens - 1).repeat_interleave(num_beams, dim=0)
                for layer_idx in range(config.num_hidden_layers):
                    state = actual.past_key_values[layer_idx]["recurrent_state"]
                    reference = expected.past_key_values[layer_idx]["recurrent_state"]
                    pairs = [(name, value, target) for name, value, target in zip(state._fields, state, reference)]
                    assert_close_with_stats("COMPILE generation state", pairs, **TOLERANCES_CFG)
                    torch.testing.assert_close(actual.past_key_values.get_layer(layer_idx).next_position, next_position)
                    torch.testing.assert_close(expected.past_key_values.get_layer(layer_idx).next_position, next_position)

            assert_compiled_ops(counter, "topk_segment_forward", "topk_autoregressive_")

    def test_beam_generation_cache_matches_full_fallback(self) -> None:
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
        )
        model.eval()
        input_ids, attention_mask = make_inputs(config, batch=2, seq_len=6)
        generation_kwargs = dict(
            attention_mask=attention_mask, num_beams=3, max_new_tokens=3,
            do_sample=False, pad_token_id=config.pad_token_id, eos_token_id=None,
        )
        with torch.no_grad():
            cached = model.generate(input_ids, use_cache=True, **generation_kwargs)
            full = model.generate(input_ids, use_cache=False, **generation_kwargs)
        assert_tokens_equal("HF BEAM cache == full fallback", cached, full)

    def test_manual_generation_cache_matches_full_fallback(self) -> None:
        # Verifies manual greedy decoding is identical with and without cache.
        config, model = make_model()
        model.eval()
        input_ids, attention_mask = make_inputs(config, batch=1, seq_len=8)

        def generate(use_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
            generated = input_ids.clone()
            mask = attention_mask.clone()
            cache = make_cache() if use_cache else None
            logits = []

            with torch.no_grad():
                for step in range(GEN_MAX_NEW_TOKENS):
                    curr_input_ids = generated[:, -1:] if use_cache and step > 0 else generated
                    out = model(curr_input_ids, attention_mask=mask, past_key_values=cache, use_cache=use_cache)
                    next_logits = out.logits[:, -1]
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    logits.append(next_logits)
                    generated = torch.cat([generated, next_token], dim=-1)
                    mask = torch.cat([mask, torch.ones_like(next_token, dtype=torch.bool)], dim=-1)

            return generated, torch.stack(logits, dim=1)

        cache_out, cache_logits = generate(use_cache=True)
        full_out, full_logits = generate(use_cache=False)
        assert_tokens_equal("  MANUAL GENERATE cache == full fallback  ", cache_out, full_out)
        assert_logits_close("  MANUAL GENERATE logits cache == full fallback  ", cache_logits, full_logits)

    def test_hf_generation_cache_matches_full_fallback(self) -> None:
        # Verifies Hugging Face generate is identical with and without cache.
        config, model = make_model()
        model.eval()
        input_ids, attention_mask = make_inputs(config, batch=1, seq_len=8)

        with torch.no_grad():
            cache_out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False, use_cache=True, pad_token_id=config.pad_token_id, eos_token_id=None, )
            full_out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False, use_cache=False, pad_token_id=config.pad_token_id, eos_token_id=None, )

        assert_tokens_equal("  HF GENERATE cache == full fallback  ", cache_out, full_out)
