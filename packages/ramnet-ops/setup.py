import os
import sys
from pathlib import Path
from setuptools import setup

cuda_src_path = Path('csrc')
cuda_source_files = (
    'ram_attn_bind.cpp',
    'ram_attn_ops.cpp',
    'ram_attn_topk_addr.cu',
    'ram_attn_segment_parallel.cu',
    'ram_attn_topk_autoregressive.cu',
)

ext_modules = []
cmdclass = {}
options = {}
dependencies = ['torch>=2.5', 'einops', 'numpy']
if os.getenv("SKIP_CUDA", "0") != "1":
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != 'torch':
            raise
        raise RuntimeError(
            'CUDA source builds require a preinstalled PyTorch build. '
            'Install the intended PyTorch/CUDA version, then use pip with --no-build-isolation. '
            'Set SKIP_CUDA=1 for a reference-only build.'
        ) from exc
    from torch.utils import cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Python == 3.10 stable ABI
    cpp_extension.min_supported_cpython = '0x030A0000'

    # PyTorch normal ABI / stable ABI (for PyTorch >= 2.10 compile environment >= 2.14)
    torch_version, _ = torch.__version__.split('+')
    torch_release = tuple(int(part) for part in torch_version.split('.')[:2])
    abi_mode = os.getenv('RAMNET_TORCH_STABLE_ABI', 'auto')
    if abi_mode not in ('auto', '0', '1'):
        raise ValueError('RAMNET_TORCH_STABLE_ABI must be auto, 0, or 1.')
    stable_abi = torch_release >= (2, 14) if abi_mode == 'auto' else abi_mode == '1'
    if stable_abi:
        if torch_release < (2, 14):
            raise RuntimeError('The Torch stable ABI build requires Torch >= 2.14; its runtime target is Torch >= 2.10.')
        dependencies[0] = 'torch>=2.10'
        defines = [
            ('RAMNET_TORCH_STABLE_ABI', '1'),
            ('TORCH_TARGET_VERSION', '0x020a000000000000'), # stable ABI 2.10
            ('USE_CUDA', None),
        ]
        torch_tag = 'abi2_10'
    else:
        dependencies[0] = f'torch=={torch_version}'
        defines = [('RAMNET_TORCH_STABLE_ABI', '0')]
        torch_tag = torch_version.replace('.', '_')

    sources = [cuda_src_path / source for source in cuda_source_files]
    if sys.platform == 'win32':
        msvc_defines = ['/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/DVC_EXTRALEAN']
        msvc_args = ['/Zc:preprocessor', '/permissive-', *msvc_defines]
        cxx_args = ['/O2', *msvc_defines]
        nvcc_args = ['-O3', '-lineinfo', '--use_fast_math', *(arg for flag in msvc_args for arg in ('-Xcompiler', flag)), ]
    else:
        cxx_args = ['-O3']
        nvcc_args = ['-O3', '-lineinfo', '--use_fast_math']
    extension = CUDAExtension(
        name='ramnet_ops._C',
        sources=[source.as_posix() for source in sources],
        define_macros=defines,
        py_limited_api=True,
        extra_compile_args={
            'cxx': cxx_args,
            'nvcc': nvcc_args,
        },
    )

    # Older PyTorch builders link torch_python even with py_limited_api=True.
    extension.libraries = [library for library in extension.libraries if library != 'torch_python']

    ext_modules.append(extension)
    cmdclass = {'build_ext': BuildExtension}

    # Wheel tags do not encode the Torch/CUDA ABI; keep artifact names distinct.
    cuda_tag = torch.version.cuda.replace('.', '')
    options = {'bdist_wheel': {
        'py_limited_api': 'cp310',
        'build_number': f'1torch{torch_tag}cu{cuda_tag}',
    }}

setup(
    install_requires=dependencies,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options=options,
    zip_safe=False,
)
