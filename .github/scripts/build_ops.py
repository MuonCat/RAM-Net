"""Build the ramnet-ops sdist and native wheel without executing the wheel."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(stage, *args, env=None):
    command = [sys.executable, "-u", "-m", *map(str, args)]
    log(f"{stage}: {subprocess.list2cmdline(command)}")
    started = time.perf_counter()
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as error:
        log(f"{stage} failed after {time.perf_counter() - started:.1f}s (exit code {error.returncode}).")
        raise
    log(f"{stage} completed in {time.perf_counter() - started:.1f}s.")


def main():
    started = time.perf_counter()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Release wheels must be built with CPython 3.10.")

    root = Path(__file__).resolve().parents[2]
    work = Path(tempfile.mkdtemp(prefix="ramnet-ops-build-", dir=os.environ["RUNNER_TEMP"]))
    dist = work / "dist"
    dist.mkdir()
    torch_version = os.environ["TORCH_VERSION"]
    cuda_tag = "cu" + os.environ["CUDA_VERSION"].replace(".", "")
    os.environ["SKIP_CUDA"] = "0"
    os.environ.setdefault("MAX_JOBS", "4")
    # Prefer tools installed alongside this Python over system executables.
    os.environ["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"]))

    log(f"Python {sys.version.split()[0]} on {sys.platform}: {sys.executable}")
    log(f"Source: {root}; build directory: {work}")
    for name in ("TORCH_VERSION", "CUDA_VERSION", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST",
                 "RAMNET_TORCH_STABLE_ABI", "MAX_JOBS", "CC", "CXX", "CUDAHOSTCXX", "MSVC_VERSION"):
        if name in os.environ:
            log(f"{name}={os.environ[name]}")

    run("Upgrade pip", "pip", "install", "--upgrade", "pip")
    run("Install build dependencies", "pip", "install", "setuptools>=77.0.3", "wheel", "build", "ninja", "einops", "numpy")
    run("Install Torch", "pip", "install", f"torch=={torch_version}+{cuda_tag}",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
        "--extra-index-url", "https://pypi.org/simple")
    run("Build sdist (SKIP_CUDA=1)", "build", "--sdist", "--no-isolation", "--outdir", dist, root / "packages" / "ramnet-ops",
        env=os.environ | {"SKIP_CUDA": "1"})
    sdist, = dist.glob("*.tar.gz")
    run("Build wheel", "pip", "wheel", "--verbose", sdist, "--no-deps", "--no-build-isolation", "--wheel-dir", work / "wheels")
    wheel, = (work / "wheels").glob("*.whl")

    if sys.platform == "linux":
        run("Install repair tools", "pip", "install", "auditwheel>=6", "patchelf")
        # PyTorch provides these libraries; the NVIDIA driver provides libcuda.
        exclusions = ("libc10*.so", "libtorch*.so", "libcudart.so.*", "libcuda.so.*")
        run("Repair Linux wheel", "auditwheel", "repair", wheel,
            "--plat", "manylinux_2_35_x86_64", "--only-plat", "--wheel-dir", dist,
            *(arg for library in exclusions for arg in ("--exclude", library)))
    else:
        log(f"Copy wheel: {wheel} -> {dist}")
        shutil.copy2(wheel, dist / wheel.name)

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"dist={dist.as_posix()}\n")
    log(f"Release artifacts: {dist}")
    for artifact in sorted(dist.iterdir()):
        log(f"  {artifact.name}: {artifact.stat().st_size / (1024 ** 2):.2f} MiB")
    log(f"Build completed in {time.perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()
