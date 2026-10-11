"""Run operator and model tests with recorded, repeatable random seeds.

Selection hierarchy:
    part          Top-level scope: ops or models.
      group       Responsibility within that part: contracts, behavior, etc.
        module    Python test file belonging to the group.
          class   unittest.TestCase containing related test methods.
            test  Individual test method; its subTest cases run together.

TEST_GROUPS is the only registry. Collect individual tests, then intersect filters:
    --part / --group / --test select scope, responsibility, and dotted ID prefix.
    --device cpu excludes CUDA tests; cuda includes CPU checks as well.
    --fp16 / --bf16 enable optional low-precision tests and subTest cases.
An empty selection is an error. --seeds controls execution, not selection.
"""

import argparse
from dataclasses import dataclass
import sys
import unittest

import torch

from tests.ops import precision

# Registry: part -> group -> test modules. Groups are logical, not directories.
TEST_GROUPS = {
    'ops': {
        'contracts': (
            'tests.ops.test_ram_attention_api',
            'tests.ops.test_cuda_registration',
            'tests.ops.test_cuda_address',
            'tests.ops.test_cuda_compile',
        ),
        'behavior': (
            'tests.ops.test_ram_attention_behavior',
        ),
        'numerics': (
            'tests.ops.test_ram_attention_numerics',
            'tests.ops.test_address_numerics',
        ),
    },
    'models': {
        'contracts': ('tests.models.test_ramnet_contracts',),
        'training': ('tests.models.test_ramnet_training',),
        'behavior': ('tests.models.test_ramnet_sequence',),
        'generation': ('tests.models.test_ramnet_generation',),
    },
}


@dataclass(frozen=True)
class TestInfo:
    name: str
    part: str
    group: str
    requires_cuda: bool
    requires_low_precision: bool


def iter_tests(suite: unittest.TestSuite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_tests(test)
        else:
            yield test


def get_tests() -> list[TestInfo]:
    """Expand the registry without running tests or their setup methods."""
    loader = unittest.TestLoader()
    tests = []
    for part, groups in TEST_GROUPS.items():
        for group, modules in groups.items():
            suite = loader.loadTestsFromNames(modules)
            if loader.errors:
                raise RuntimeError('\n'.join(loader.errors))
            for test in iter_tests(suite):
                method = getattr(test, test._testMethodName)
                # CUDA is required unless the class explicitly declares CPU coverage.
                tests.append(TestInfo(
                    test.id(), part, group,
                    getattr(test, 'requires_cuda', True),
                    getattr(method, 'requires_low_precision', False),
                ))
    return tests


def filter_tests(tests: list[TestInfo], args: argparse.Namespace) -> list[TestInfo]:
    """Apply independent filters; an explicit test ID never bypasses another filter."""
    if args.part != 'all':
        tests = [test for test in tests if test.part == args.part]
    if args.group:
        tests = [test for test in tests if test.group == args.group]
    if args.test:
        tests = [test for test in tests if test.name == args.test or test.name.startswith(args.test + '.')]
    if args.device == 'cpu':
        tests = [test for test in tests if not test.requires_cuda]
    if not (args.fp16 or args.bf16):
        tests = [test for test in tests if not test.requires_low_precision]
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--part', choices=('ops', 'models', 'all'), default='all', help='Top-level test scope; default: both ops and models.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='cpu: CPU checks only; cuda: include CUDA checks.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--fp16', action='store_true', help='Also run optional CUDA FP16 checks.')
    parser.add_argument('--bf16', action='store_true', help='Also run optional CUDA BF16 checks; requires BF16 support.')
    groups = sorted({name for part in TEST_GROUPS.values() for name in part})
    parser.add_argument('--group', choices=groups, help='Responsibility group within --part; default: all groups.')
    parser.add_argument('--test', help='Dotted unittest module/class/method ID; intersects all other filters.')
    args = parser.parse_args()

    selected = filter_tests(get_tests(), args)
    if not selected:
        parser.error('No tests match the selected part, group, test, device, and precision options.')
    parts = {test.part for test in selected}
    if (args.fp16 or args.bf16) and not any(test.part == 'ops' and test.requires_cuda for test in selected):
        parser.error('--fp16/--bf16 apply only to CUDA operator tests.')
    precision.LOW_PRECISION_DTYPES = tuple(dtype for enabled, dtype in ((args.fp16, torch.float16), (args.bf16, torch.bfloat16)) if enabled)

    print(f'Python: {sys.executable}\nPyTorch: {torch.__version__}', flush=True)
    if any(test.requires_cuda for test in selected):
        if not torch.cuda.is_available():
            parser.error('CUDA validation requires a CUDA GPU.')
        if args.bf16 and not torch.cuda.is_bf16_supported():
            parser.error('--bf16 requires a CUDA GPU with BF16 support.')
        # Import directly so an unavailable extension cannot become a skipped test.
        import ramnet_ops._C as extension

        print(f'Extension: {extension.__file__}\nGPU: {torch.cuda.get_device_name(0)}', flush=True)
    if 'ops' in parts:
        import ramnet_ops

        print(f'Ops package: {ramnet_ops.__file__}', flush=True)
        print(f'Ops dtypes: {(torch.float32, *precision.LOW_PRECISION_DTYPES)}', flush=True)
    if 'models' in parts:
        import ramnet

        print(f'Model package: {ramnet.__file__}', flush=True)

    successful = True
    for seed in args.seeds:
        print(f'\nPart: {args.part}; device: {args.device}; group: {args.group or "all"}; tests: {len(selected)}; seed: {seed}', flush=True)

        class SeededResult(unittest.TextTestResult):
            def startTest(self, test):
                # Reset per test so a focused rerun uses the same random inputs.
                torch.manual_seed(seed)
                super().startTest(test)

        loader = unittest.TestLoader()
        # Recreate test instances for each seed so no test-local state is reused.
        suite = loader.loadTestsFromNames([test.name for test in selected])
        result = unittest.TextTestRunner(verbosity=2, resultclass=SeededResult).run(suite)
        if result.skipped:
            print('Validation does not allow unexpected skipped tests.', flush=True)
        successful = successful and result.wasSuccessful() and not result.skipped and result.testsRun > 0
    return 0 if successful else 1


if __name__ == '__main__':
    raise SystemExit(main())
