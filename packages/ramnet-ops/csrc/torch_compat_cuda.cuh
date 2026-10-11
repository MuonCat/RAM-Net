#pragma once
#include "torch_compat.h"
#include <cuda_runtime.h>

#if RAMNET_TORCH_STABLE_ABI
#include <memory>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace ramnet_ops::torch_api {
#if RAMNET_TORCH_STABLE_ABI

    class CUDAGuard {
        CUDAGuardHandle guard_ = nullptr;

    public:
        explicit CUDAGuard(const Device device) {
            STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_create_cuda_guard(device.index(), &guard_));
        }

        ~CUDAGuard() {
            STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_cuda_guard(guard_));
        }

        CUDAGuard(const CUDAGuard &) = delete;

        CUDAGuard &operator=(const CUDAGuard &) = delete;
    };

    inline cudaStream_t current_cuda_stream(const Device device) {
        void *stream = nullptr;
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device.index(), &stream));
        return static_cast<cudaStream_t>(stream);
    }

    class CUDAStreamGuard {
        CUDAStreamGuardHandle guard_ = nullptr;

    public:
        CUDAStreamGuard(const cudaStream_t stream, const Device device) {
            STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_create_cuda_stream_guard(stream, device.index(), &guard_));
        }

        ~CUDAStreamGuard() {
            STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_cuda_stream_guard(guard_));
        }

        CUDAStreamGuard(const CUDAStreamGuard &) = delete;

        CUDAStreamGuard &operator=(const CUDAStreamGuard &) = delete;
    };

    inline void check_cuda_launch(const cudaError_t error, const char *const file, const char *const function, const uint32_t line) {
        char *raw_message = nullptr;
        const auto status = torch_c10_cuda_check_msg(error, file, function, line, true, &raw_message);
        const std::unique_ptr<char, decltype(&torch_c10_cuda_free_error_msg)> message(raw_message, torch_c10_cuda_free_error_msg);
        RAMNET_CHECK(!message, message.get());
        STABLE_TORCH_ERROR_CODE_CHECK(status);
    }

#define RAMNET_CUDA_KERNEL_LAUNCH_CHECK() \
    torch_api::check_cuda_launch(cudaGetLastError(), __FILE__, __func__, __LINE__)

#else

    using CUDAGuard = c10::cuda::CUDAGuard;

    inline c10::cuda::CUDAStream current_cuda_stream(const Device device) {
        return c10::cuda::getCurrentCUDAStream(device.index());
    }

    class CUDAStreamGuard {
        c10::cuda::CUDAStreamGuard guard_;

    public:
        CUDAStreamGuard(const c10::cuda::CUDAStream stream, const Device) : guard_(stream) {
        }
    };

#define RAMNET_CUDA_KERNEL_LAUNCH_CHECK C10_CUDA_KERNEL_LAUNCH_CHECK

#endif
}
