"""Mask/position semantics and equivalence of cached and full execution."""

import unittest

import torch

from tests.helpers import assert_close_with_stats
from tests.models.helpers import (
    TOLERANCES_CFG,
    assert_all_parameters_have_grad,
    assert_cache_length,
    assert_logits_close,
    compact_input_ids,
    make_cache,
    make_inputs,
    make_model,
    make_position_ids,
    model_logits,
)

CACHE_POSITION_MODES = ("auto_mask", "explicit_position_ids")


class RamNetSequenceTest(unittest.TestCase):
    def test_explicit_block_state_supports_custom_time_axis(self) -> None:
        for fuse_norm in (False, True):
            with self.subTest(fuse_norm=fuse_norm):
                config, model = make_model(
                    hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                    addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
                    fuse_norm=fuse_norm,
                )
                block = model.model.layers[0].eval()
                hidden = torch.randn(2, 6, config.hidden_size, device="cuda")
                mask = torch.tensor([[0, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0]], device="cuda", dtype=torch.bool)
                positions = make_position_ids(mask)
                slot = torch.randn(2, 4, 16, 8, device="cuda")
                mass = torch.rand(2, 4, 16, device="cuda") + 0.1
                initial_slot, initial_mass = slot.clone(), mass.clone()
                cache = make_cache()
                cache.update((slot.clone(), mass.clone()), layer_idx=0, offset=0)

                with torch.no_grad():
                    ref, ref_state = block.forward_with_state(hidden, (slot, mass), mask=mask, position_ids=positions)
                    val, state = block.forward_with_state(
                        hidden.movedim(1, 0), (slot, mass),
                        mask=mask.movedim(1, 0), position_ids=positions.movedim(1, 0), t_dim=0,
                    )
                    cached, attentions, returned_cache = block(
                        hidden, attention_mask=mask, past_key_values=cache,
                        use_cache=True, position_ids=positions,
                    )
                torch.testing.assert_close(val.movedim(0, 1), ref)
                torch.testing.assert_close(cached, ref)
                self.assertIsNone(attentions)
                self.assertIs(returned_cache, cache)
                self.assertEqual(cache.get_seq_length(), hidden.size(1))
                for actual_state in (state, cache[0]["recurrent_state"]):
                    for value, reference in zip(actual_state, ref_state):
                        torch.testing.assert_close(value, reference)
                torch.testing.assert_close(slot, initial_slot)
                torch.testing.assert_close(mass, initial_mass)

    def test_recurrent_state_survives_masked_chunks(self) -> None:
        # Includes different valid lengths, an all-masked chunk, and a one-token decode.
        config, model = make_model(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
            addr_order=2, addr_radix=4, value_chs=8, vocab_size=64,
        )
        model.eval()
        input_ids = torch.randint(3, config.vocab_size, (2, 8), device="cuda")
        mask = torch.tensor([[1, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1, 1, 0]], device="cuda", dtype=torch.bool)
        with torch.no_grad():
            ref = model(input_ids, attention_mask=mask, use_cache=True)
            cache = make_cache()
            chunks = []
            for start, stop in ((0, 2), (2, 5), (5, 6), (6, 8)):
                chunks.append(model_logits(
                    model, input_ids[:, start:stop], attention_mask=mask[:, start:stop],
                    past_key_values=cache, use_cache=True,
                ))
        assert_logits_close("CACHE masked chunks", torch.cat(chunks, dim=1), ref.logits, mask)
        assert_cache_length("CACHE masked chunks", cache, config, input_ids.size(1))
        for layer_idx in range(config.num_hidden_layers):
            state = cache[layer_idx]["recurrent_state"]
            reference = ref.past_key_values[layer_idx]["recurrent_state"]
            pairs = [(name, value, target) for name, value, target in zip(state._fields, state, reference)]
            assert_close_with_stats("CACHE final state", pairs, **TOLERANCES_CFG)
            torch.testing.assert_close(cache.get_layer(layer_idx).next_position, mask.long().sum(-1, keepdim=True))

    def test_read_cache_without_updating(self) -> None:
        config, model = make_model()
        model.eval()
        input_ids, mask = make_inputs(config, seq_len=8)
        with torch.no_grad():
            prefix = model(input_ids[:, :5], attention_mask=mask[:, :5], use_cache=True)
            cache = prefix.past_key_values
            slot = cache[0]["recurrent_state"].slot.clone()
            actual = model(input_ids[:, 5:], past_key_values=cache, use_cache=False)
            self.assertIsNone(actual.past_key_values)
            torch.testing.assert_close(cache[0]["recurrent_state"].slot, slot)
            assert_cache_length("CACHE read only", cache, config, 5)
            expected = model_logits(model, input_ids, attention_mask=mask, use_cache=False)[:, 5:]
        assert_logits_close("CACHE read only", actual.logits, expected)

    def test_attention_mask_matches_explicit_position_ids(self) -> None:
        # Verifies attention_mask-derived positions match explicit position_ids.
        config, model = make_model()
        model.eval()
        input_ids, attention_mask = make_inputs(config, padding="position")

        with torch.no_grad():
            val = model_logits(model, input_ids, attention_mask=attention_mask, use_cache=False)
            ref = model_logits(model, input_ids, attention_mask=attention_mask, position_ids=make_position_ids(attention_mask), use_cache=False)

        assert_logits_close("  FORWARD attention_mask == explicit position_ids  ", val, ref, attention_mask)

    def test_random_mask_matches_per_sequence_compacted_forward_backward(self) -> None:
        # Verifies random masked batches match independently compacted sequences in forward and backward.
        batch, seq_len = 4, 32

        config, model = make_model()
        model.eval()

        mask_rows: list[torch.Tensor] = []
        while len(mask_rows) < batch:
            row = torch.rand(seq_len) < 0.6
            valid_count = int(row.sum())
            if valid_count in (0, seq_len):
                continue
            if any(torch.equal(row, prev) or valid_count == int(prev.sum()) for prev in mask_rows):
                continue
            mask_rows.append(row)

        attention_mask = torch.stack(mask_rows).cuda()
        input_ids = torch.randint(3, config.vocab_size, (batch, seq_len)).cuda()
        input_ids = input_ids.masked_fill(~attention_mask, config.pad_token_id)

        masked_hidden = model.model(input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        masked_valid = masked_hidden[attention_mask]
        upstream = torch.randn(masked_valid.shape, dtype=masked_valid.dtype).to(masked_valid.device)
        (masked_valid * upstream).sum().backward()
        assert_all_parameters_have_grad("masked", model.model)
        masked_grads = {
            param_name: param.grad.detach().clone()
            for param_name, param in model.model.named_parameters()
        }
        masked_valid = masked_valid.detach()
        model.zero_grad(set_to_none=True)

        compact_values = []
        valid_offset = 0
        for batch_idx in range(batch):
            compact_ids = compact_input_ids(input_ids, attention_mask, batch_idx)
            compact_hidden = model.model(compact_ids, use_cache=False).last_hidden_state.squeeze(0)
            compact_values.append(compact_hidden.detach())

            valid_len = compact_hidden.size(0)
            compact_upstream = upstream[valid_offset:valid_offset + valid_len]
            (compact_hidden * compact_upstream).sum().backward()
            valid_offset += valid_len

        self.assertEqual(valid_offset, masked_valid.size(0))
        assert_all_parameters_have_grad("compacted", model.model)
        assert_close_with_stats(
            "  FORWARD random mask == per-sequence compacted  ",
            [("hidden", masked_valid, torch.cat(compact_values, dim=0))],
            **TOLERANCES_CFG,
        )
        gradient_pairs = []
        stats_chunk_size = 1 << 22
        for param_name, param in model.model.named_parameters():
            grad_chunks = param.grad.flatten().split(stats_chunk_size)
            ref_chunks = masked_grads[param_name].flatten().split(stats_chunk_size)
            for chunk_idx, (grad_chunk, ref_chunk) in enumerate(zip(grad_chunks, ref_chunks)):
                chunk_name = param_name if len(grad_chunks) == 1 else f"{param_name}[{chunk_idx}]"
                gradient_pairs.append((chunk_name, grad_chunk, ref_chunk))
        assert_close_with_stats(
            "  BACKWARD random mask == per-sequence compacted  ",
            gradient_pairs,
            **TOLERANCES_CFG,
        )
        model.zero_grad(set_to_none=True)

    def test_cache_decode_matches_full_fallback(self) -> None:
        # Verifies cached parallel suffix decoding matches full-sequence fallback logits.
        config, model = make_model()
        model.eval()
        prompt_len, suffix_len = 6, 3
        input_ids, attention_mask = make_inputs(config, seq_len=prompt_len + suffix_len)
        position_ids = make_position_ids(attention_mask) * 3 + 5
        position_ids = position_ids.masked_fill(~attention_mask, 0)

        for position_mode in CACHE_POSITION_MODES:
            with self.subTest(position_mode=position_mode):
                explicit_position_ids = position_mode == "explicit_position_ids"
                with torch.no_grad():
                    ref = model_logits(
                        model,
                        input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids if explicit_position_ids else None,
                        use_cache=False,
                    )[:, prompt_len:]

                    cache = make_cache()
                    model_logits(
                        model,
                        input_ids[:, :prompt_len],
                        attention_mask=attention_mask[:, :prompt_len],
                        position_ids=position_ids[:, :prompt_len] if explicit_position_ids else None,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    assert_cache_length(f"CACHE {position_mode} PREFILL", cache, config, prompt_len)

                    val = model_logits(
                        model,
                        input_ids[:, prompt_len:],
                        attention_mask=attention_mask,
                        position_ids=position_ids[:, prompt_len:] if explicit_position_ids else None,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    assert_cache_length(f"CACHE {position_mode} DECODE", cache, config, prompt_len + suffix_len)

                assert_logits_close(
                    f"  CACHE {position_mode} decode cache == full fallback  ",
                    val,
                    ref,
                    attention_mask[:, prompt_len:],
                )

    def test_mixed_padding_cache_decode_matches_compacted(self) -> None:
        # Verifies cached parallel suffix decoding with mixed padding matches compacted per-sample caches.
        config, model = make_model()
        model.eval()
        prompt_len, suffix_len = 8, 4
        input_ids, attention_mask = make_inputs(config, seq_len=prompt_len + suffix_len, padding="mixed")

        with torch.no_grad():
            padded_cache = make_cache()
            model_logits(model, input_ids[:, :prompt_len], attention_mask=attention_mask[:, :prompt_len], past_key_values=padded_cache, use_cache=True)
            padded = model_logits(model, input_ids[:, prompt_len:], attention_mask=attention_mask, past_key_values=padded_cache, use_cache=True)
            assert_cache_length("CACHE mixed padding decode", padded_cache, config, prompt_len + suffix_len)

            padded_valid = []
            compact_refs = []
            for batch_idx in range(input_ids.size(0)):
                compact_cache = make_cache()
                compact_prefix = compact_input_ids(input_ids, attention_mask, batch_idx, end=prompt_len)
                compact_prefix_mask = torch.ones(1, compact_prefix.size(1), device=input_ids.device, dtype=torch.bool)
                model_logits(model, compact_prefix, attention_mask=compact_prefix_mask, past_key_values=compact_cache, use_cache=True)

                compact_suffix = compact_input_ids(input_ids, attention_mask, batch_idx)[:, compact_prefix.size(1):]
                compact_len = compact_prefix.size(1) + compact_suffix.size(1)
                compact_mask = torch.ones(1, compact_len, device=input_ids.device, dtype=torch.bool)
                compact = model_logits(model, compact_suffix, attention_mask=compact_mask, past_key_values=compact_cache, use_cache=True)
                padded_valid.append(padded[batch_idx, attention_mask[batch_idx, prompt_len:]])
                compact_refs.append(compact.squeeze(0))
                assert_cache_length(f"CACHE compact batch={batch_idx} decode", compact_cache, config, compact_len)

        assert_logits_close(
            "  CACHE mixed padding == compacted tokens  ",
            torch.cat(padded_valid, dim=0),
            torch.cat(compact_refs, dim=0),
        )
