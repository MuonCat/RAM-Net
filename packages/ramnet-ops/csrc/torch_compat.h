#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>

#ifndef RAMNET_TORCH_STABLE_ABI
#define RAMNET_TORCH_STABLE_ABI 0
#endif

#if RAMNET_TORCH_STABLE_ABI
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#include <torch/headeronly/core/ScalarType.h>
#define RAMNET_CHECK STD_TORCH_CHECK
#else
#include <ATen/ATen.h>
#define RAMNET_CHECK TORCH_CHECK
#endif

namespace ramnet_ops::torch_api {
#if RAMNET_TORCH_STABLE_ABI

    using Tensor = torch::stable::Tensor;
    using ScalarType = torch::headeronly::ScalarType;
    using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
    using Device = torch::stable::Device;
    using torch::headeronly::toString;

    template<ScalarType dtype>
    using ScalarTypeToCPPTypeT = torch::headeronly::impl::ScalarTypeToCPPTypeT<dtype>;

    namespace detail {
        // wrapper for PyTorch Dispatcher (return Tensor)
        template<typename... Args>
        Tensor call_tensor(const char *const name, const char *const overload, const Args &... args) {
            std::array<StableIValue, sizeof...(Args)> stack{torch::stable::detail::from(args)...};
            STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(name, overload, stack.data(), TORCH_ABI_VERSION));
            return torch::stable::detail::to<Tensor>(stack[0]);
        }
    }

    inline Tensor empty(const IntArrayRef shape, const ScalarType dtype, const Device device) {
        return torch::stable::empty(shape, dtype, std::nullopt, device);
    }

    inline Tensor zeros(const IntArrayRef shape, const ScalarType dtype, const Device device) {
        return torch::stable::full(shape, 0.0, dtype, std::nullopt, device);
    }

    inline Tensor contiguous(const Tensor &tensor) {
        return torch::stable::contiguous(tensor);
    }

    inline Tensor clone(const Tensor &tensor) {
        // Always allocate independent, contiguous storage, including contiguous inputs.
        return detail::call_tensor("aten::clone", "", tensor, std::optional{torch::headeronly::MemoryFormat::Contiguous});
    }

    inline Tensor to(const Tensor &tensor, const ScalarType dtype) {
        return torch::stable::to(tensor, dtype);
    }

    inline Tensor narrow(const Tensor &tensor, const int64_t dim, const int64_t start, const int64_t length) {
        AtenTensorHandle result = nullptr;
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_aten_narrow(tensor.get(), dim, start, length, &result));
        return Tensor(result);
    }

    inline Tensor transpose(const Tensor &tensor, const int64_t dim0, const int64_t dim1) {
        return torch::stable::transpose(tensor, dim0, dim1);
    }

    inline Tensor reshape(const Tensor &tensor, const IntArrayRef shape) {
        return torch::stable::reshape(tensor, shape);
    }

    inline Tensor index_select(const Tensor &tensor, const int64_t dim, const Tensor &index) {
        return torch::stable::index_select(tensor, dim, index);
    }

    inline Tensor unsqueeze(const Tensor &tensor, const int64_t dim) {
        return torch::stable::unsqueeze(tensor, dim);
    }

    inline Tensor sum(const Tensor &tensor, const int64_t dim) {
        return torch::stable::sum(tensor, IntArrayRef{dim});
    }

    inline Tensor add(const Tensor &tensor, const double scalar) {
        // Scalar boxing is unavailable at ABI 2.10; this helper is CUDA-only.
        AtenTensorHandle result = nullptr;
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_add_Scalar(tensor.get(), scalar, 1.0, &result));
        return Tensor(result);
    }

    inline Tensor mul(const Tensor &a, const Tensor &b) {
        return detail::call_tensor("aten::mul", "Tensor", a, b);
    }

    inline void mul_(const Tensor &a, const Tensor &b) {
        detail::call_tensor("aten::mul_", "Tensor", a, b);
    }

    inline Tensor div(const Tensor &a, const Tensor &b) {
        return detail::call_tensor("aten::div", "Tensor", a, b);
    }

    inline Tensor neg(const Tensor &tensor) {
        return detail::call_tensor("aten::neg", "", tensor);
    }

    template<typename T, bool ReadOnly = false>
    std::conditional_t<ReadOnly, const T *, T *> data_ptr(const Tensor &tensor) {
        if constexpr (ReadOnly)
            return tensor.template const_data_ptr<T>();
        else
            return tensor.template mutable_data_ptr<T>();
    }

#else

    using Tensor = at::Tensor;
    using ScalarType = at::ScalarType;
    using IntArrayRef = at::IntArrayRef;
    using Device = at::Device;
    using at::toString;

    template<ScalarType dtype>
    using ScalarTypeToCPPTypeT = typename c10::impl::ScalarTypeToCPPType<dtype>::type;

    inline Tensor empty(const IntArrayRef shape, const ScalarType dtype, const Device device) {
        return at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
    }

    inline Tensor zeros(const IntArrayRef shape, const ScalarType dtype, const Device device) {
        return at::zeros(shape, at::TensorOptions().dtype(dtype).device(device));
    }

    inline Tensor contiguous(const Tensor &tensor) {
        return tensor.contiguous();
    }

    inline Tensor clone(const Tensor &tensor) {
        return tensor.clone(at::MemoryFormat::Contiguous);
    }

    inline Tensor to(const Tensor &tensor, const ScalarType dtype) {
        return tensor.to(dtype);
    }

    inline Tensor narrow(const Tensor &tensor, const int64_t dim, const int64_t start, const int64_t length) {
        return tensor.narrow(dim, start, length);
    }

    inline Tensor transpose(const Tensor &tensor, const int64_t dim0, const int64_t dim1) {
        return tensor.transpose(dim0, dim1);
    }

    inline Tensor reshape(const Tensor &tensor, const IntArrayRef shape) {
        return tensor.reshape(shape);
    }

    inline Tensor index_select(const Tensor &tensor, const int64_t dim, const Tensor &index) {
        return tensor.index_select(dim, index);
    }

    inline Tensor unsqueeze(const Tensor &tensor, const int64_t dim) {
        return tensor.unsqueeze(dim);
    }

    inline Tensor sum(const Tensor &tensor, const int64_t dim) {
        return tensor.sum({dim});
    }

    inline Tensor add(const Tensor &tensor, const double scalar) {
        return tensor + scalar;
    }

    inline Tensor mul(const Tensor &a, const Tensor &b) {
        return a * b;
    }

    inline void mul_(const Tensor &a, const Tensor &b) {
        a.mul_(b);
    }

    inline Tensor div(const Tensor &a, const Tensor &b) {
        return a / b;
    }

    inline Tensor neg(const Tensor &tensor) {
        return tensor.neg();
    }

    template<typename T, bool ReadOnly = false>
    std::conditional_t<ReadOnly, const T *, T *> data_ptr(const Tensor &tensor) {
        return tensor.template data_ptr<T>();
    }

#endif

    using Half = ScalarTypeToCPPTypeT<ScalarType::Half>;
    using BFloat16 = ScalarTypeToCPPTypeT<ScalarType::BFloat16>;

    inline std::string shape_string(const IntArrayRef shape) {
        // HeaderOnlyArrayRef does not provide the ordinary ArrayRef stream operator.
        std::ostringstream result;
        result << '[';
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0)
                result << ", ";
            result << shape[i];
        }
        result << ']';
        return result.str();
    }
}

namespace torch_api = ramnet_ops::torch_api;
