"""Numerical budgets; shape, dtype, aliasing, and mask invariants stay exact."""

import torch


BEHAVIOR_TOLERANCE = dict(atol=1e-4, rtol=1e-4, max_err_rate=0.01, max_q99=1.5)
FP32_TOLERANCE = dict(atol=5e-4, rtol=5e-4, max_err_rate=0.01, max_q99=1.5)
PRECISION_TOLERANCES = {
    torch.float16: dict(atol=1e-2, rtol=1e-2, max_err_rate=0.01, max_q99=1.5),
    torch.bfloat16: dict(atol=7.5e-2, rtol=7.5e-2, max_err_rate=0.02, max_q99=2.0),
}
