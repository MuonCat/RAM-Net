// PyTorch 2.5 does not set this macro when building a limited-API extension.
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030A0000
#endif
#include <Python.h>
#include "ram_attn_ops.h"
#include "ram_attn.cuh"
#include <limits>

#if RAMNET_TORCH_STABLE_ABI
#include <torch/csrc/stable/library.h>
#define RAMNET_LIBRARY STABLE_TORCH_LIBRARY
#define RAMNET_LIBRARY_IMPL STABLE_TORCH_LIBRARY_IMPL
#define RAMNET_FN(fn) TORCH_BOX(&fn)
#else
#include <torch/library.h>
#define RAMNET_LIBRARY TORCH_LIBRARY
#define RAMNET_LIBRARY_IMPL TORCH_LIBRARY_IMPL
#define RAMNET_FN(fn) TORCH_FN(fn)
#endif

RAMNET_LIBRARY(ramnet_ops, m) {
#if !RAMNET_TORCH_STABLE_ABI
    m.set_python_module("ramnet_ops.impl._cuda_autograd");
#endif
    // cuda_impl explicitly imports _cuda_autograd; ABI 2.10 has no set_python_module.
    m.def(
        "topk_addr_single_forward("
        "Tensor addr, "
        "int embed_num, int A, "
        "Tensor position_ids, "
        "int top_k, bool top_k_re_norm, "
        "bool addr_val_neg"
        ") -> (Tensor, Tensor)");
    m.def(
        "topk_addr_single_backward("
        "Tensor idx, Tensor val, "
        "Tensor? addr, "
        "Tensor D_val, "
        "int embed_num, Tensor position_ids, "
        "int top_k, bool top_k_re_norm, "
        "bool addr_val_neg, "
        "int[] shape, ScalarType dtype"
        ") -> Tensor");
    m.def(
        "topk_addr_forward("
        "Tensor w_addr, Tensor r_addr, "
        "int embed_num, int A, "
        "Tensor position_ids, "
        "int w_top_k, int r_top_k, "
        "bool w_re_norm, bool r_re_norm"
        ") -> (Tensor, Tensor)");
    m.def(
        "topk_addr_backward("
        "Tensor idx, Tensor val, "
        "Tensor? w_addr, Tensor? r_addr, "
        "Tensor D_val, "
        "int embed_num, Tensor position_ids, "
        "int w_top_k, int r_top_k, "
        "bool w_re_norm, bool r_re_norm, "
        "int[] shape, ScalarType dtype"
        ") -> (Tensor, Tensor)");
    m.def(
        "topk_segment_forward("
        "Tensor addr_val, "
        "Tensor seg_idx, Tensor seg_off, Tensor seg_len, Tensor event_i, "
        "Tensor x, Tensor gamma, "
        "Tensor init_slot, Tensor init_mass, "
        "int w_top_k, int r_top_k, "
        "bool mass_aware, float eps"
        ") -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "topk_segment_backward("
        "Tensor seg_idx, Tensor seg_off, Tensor seg_len, Tensor event_i, Tensor event_v, "
        "Tensor x, Tensor gamma, "
        "Tensor init_slot, Tensor init_mass, "
        "Tensor D_y, Tensor D_post_slot, Tensor D_post_mass, "
        "int w_top_k, int r_top_k, "
        "bool mass_aware, float eps"
        ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "topk_segment_forward_("
        "Tensor addr_val, "
        "Tensor seg_idx, Tensor seg_off, Tensor seg_len, Tensor event_i, "
        "Tensor x, Tensor gamma, "
        "Tensor! slot, Tensor! mass, "
        "int w_top_k, int r_top_k, "
        "bool mass_aware, float eps"
        ") -> Tensor");
    m.def(
        "topk_autoregressive_("
        "Tensor w_addr, Tensor r_addr, "
        "Tensor x, Tensor gamma, "
        "Tensor! slot, Tensor! mass, "
        "Tensor mask, int embed_num, Tensor position_ids, "
        "int w_top_k, int r_top_k, "
        "bool w_re_norm, bool r_re_norm, "
        "bool mass_aware, float eps"
        ") -> Tensor");
}

RAMNET_LIBRARY_IMPL(ramnet_ops, CUDA, m) {
    m.impl("topk_addr_single_forward", RAMNET_FN(ramnet_ops::topk_addr_single_forward));
    m.impl("topk_addr_single_backward", RAMNET_FN(ramnet_ops::topk_addr_single_backward));
    m.impl("topk_addr_forward", RAMNET_FN(ramnet_ops::topk_addr_forward));
    m.impl("topk_addr_backward", RAMNET_FN(ramnet_ops::topk_addr_backward));
    m.impl("topk_segment_forward", RAMNET_FN(ramnet_ops::topk_segment_forward));
    m.impl("topk_segment_forward_", RAMNET_FN(ramnet_ops::topk_segment_forward_));
    m.impl("topk_segment_backward", RAMNET_FN(ramnet_ops::topk_segment_backward));
    m.impl("topk_autoregressive_", RAMNET_FN(ramnet_ops::topk_autoregressive_));
}

// Importing _C loads dispatcher registrations; tensors never cross the Python C API.
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr
};

PyMODINIT_FUNC PyInit__C() {
    auto *result = PyModule_Create(&module);
    if (result == nullptr)
        return nullptr;
    if (PyModule_AddIntConstant(result, "INT_BITS", std::numeric_limits<int_t>::digits + 1) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
