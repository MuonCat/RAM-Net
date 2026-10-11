"""Optional CUDA precision coverage, configured by the shared test runner."""

import unittest

import torch


# FP32 always runs. Direct unittest discovery also defaults to no optional dtypes.
LOW_PRECISION_DTYPES: tuple[torch.dtype, ...] = ()
DISABLED_REASON = "Optional low-precision coverage is disabled; use --fp16 and/or --bf16."


def requires_low_precision(method):
    """Mark precision-only methods for filtering before setup or execution."""
    method.requires_low_precision = True
    return method


def low_precision_dtypes(*, required: bool = False) -> tuple[torch.dtype, ...]:
    # Keep direct unittest discovery safe when it bypasses the shared filters.
    if required and not LOW_PRECISION_DTYPES:
        raise unittest.SkipTest(DISABLED_REASON)
    return LOW_PRECISION_DTYPES
