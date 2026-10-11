"""Public API option forwarding without CUDA execution."""

import unittest
from unittest.mock import patch

import torch

import ramnet_ops.ram_attn as ram_attn_module


class RamAttentionApiTest(unittest.TestCase):
    requires_cuda = False

    def test_api_mass_aware_reaches_backend(self) -> None:
        # Isolate public API forwarding from backend execution.
        addresses = torch.zeros(1, 2, 1, 1, 8)
        values = torch.zeros(1, 2, 1, 4)

        def fake_ram_attn(w_addr, r_addr, x, gamma, init_slot, init_mass, *args):
            return x, init_slot, init_mass

        for backend in ("torch", "cuda"):
            for mass_aware in (False, True):
                with self.subTest(backend=backend, mass_aware=mass_aware):
                    with patch.object(ram_attn_module, f"_{backend}_ram_attn", side_effect=fake_ram_attn) as impl:
                        ram_attn_module.random_access_memory_attention(
                            addresses, addresses,
                            values,
                            mass_aware=mass_aware,
                            impl_backend=backend,
                        )
                    impl.assert_called_once()
                    self.assertIs(impl.call_args.args[-3], mass_aware)

    def test_api_top_k_re_norm_preserves_explicit_values(self) -> None:
        seen = {}

        def fake_ram_attn(
                w_addr,
                r_addr,

                x,
                gamma,

                init_slot,
                init_mass,

                mask,
                embed_num,
                position_ids,

                top_k,

                top_k_re_norm,

                mass_aware,
                eps,

                inplace,
        ):
            seen["top_k_re_norm"] = top_k_re_norm
            return x, init_slot, init_mass

        r_addr = torch.zeros(1, 1, 1, 1, 16)
        w_addr = torch.zeros_like(r_addr)
        values = torch.zeros(1, 1, 1, 4)

        cases = (
            (False, (False, False)),
            (True, (True, True)),
            ((True, False), (True, False)),
            ([False, True], (False, True)),
        )
        for option, expected in cases:
            with self.subTest(top_k_re_norm=option):
                with patch.object(ram_attn_module, "_torch_ram_attn", side_effect=fake_ram_attn):
                    ram_attn_module.random_access_memory_attention(
                        w_addr, r_addr,
                        values,
                        top_k=(4, 12),
                        top_k_re_norm=option,
                        impl_backend='torch',
                    )
                self.assertEqual(seen["top_k_re_norm"], expected)

    def test_api_top_k_re_norm_rejects_invalid_values(self) -> None:
        addresses = torch.zeros(1, 1, 1, 1, 16)
        values = torch.zeros(1, 1, 1, 4)
        invalid_options = (None, 1, (0, 1), [True], (True, False, True))

        with patch.object(ram_attn_module, "_torch_ram_attn") as impl:
            for option in invalid_options:
                with self.subTest(top_k_re_norm=option), self.assertRaisesRegex(TypeError, 'top_k_re_norm'):
                    ram_attn_module.random_access_memory_attention(
                        addresses, addresses,
                        values,
                        top_k_re_norm=option,
                        impl_backend='torch',
                    )
            impl.assert_not_called()
