"""Compiler isolation and graph assertions shared by ops and model tests."""

from contextlib import contextmanager

import torch


COMPILE_BACKENDS = ("aot_eager", "inductor")


@contextmanager
def compile_case(func, backend: str, *, fullgraph: bool = True):
    from torch._dynamo import config
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.compiler.reset()
    counter = CompileCounterWithBackend(backend)
    try:
        # A failed compilation must not silently turn into an eager test.
        with config.patch(suppress_errors=False):
            yield torch.compile(func, backend=counter, fullgraph=fullgraph, dynamic=False), counter
    finally:
        torch.compiler.reset()


def assert_compiled_ops(counter, *operators: str) -> None:
    # Checkpointed regions live in nested GraphModules.
    targets = {
        str(node.target)
        for graph in counter.graphs for module in graph.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    }
    for operator in operators:
        prefix = f"ramnet_ops.{operator}"
        if not any(target == prefix or target.startswith(prefix + ".") for target in targets):
            raise AssertionError(f"{counter.backend} must capture ramnet_ops.{operator}; got {sorted(targets)}")


def capture_cuda_graph(func, *inputs):
    """Finish compilation and warmup before recording a real CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            func(*inputs)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = func(*inputs)
    return graph, outputs
