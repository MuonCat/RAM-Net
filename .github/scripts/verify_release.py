"""Verify an existing build and its distributions without importing package code."""

import argparse
import base64
import csv
from email.parser import BytesParser
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from wheel.wheelfile import WheelFile


# Keep the native release contract aligned with ramnet-ops-build.yml.
TORCH_CUDA = {
    '2.6.0': ('118', '126'),
    '2.8.0': ('126', '128'),
    '2.14.0': ('126', '130'),
}
PLATFORMS = {'windows': 'win_amd64', 'linux': 'manylinux_2_35_x86_64'}
MANIFEST = Path('release.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def github(path, *, not_found=None):
    url = f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/{path}"
    request = Request(url, headers={
        'Accept': 'application/vnd.github+json',
        'Authorization': f"Bearer {os.environ['GH_TOKEN']}",
        'X-GitHub-Api-Version': '2022-11-28',
    })
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404 and not_found is not None:
            raise ValueError(not_found) from None
        raise ValueError(f'GitHub API request failed (HTTP {error.code} {error.reason}): {url}') from None


def expected_artifacts(version, package):
    if package == 'ramnet':
        return {
            'ramnet-dist-wheel': f'ramnet-{version}-py3-none-any.whl',
            'ramnet-dist-source': f'ramnet-{version}.tar.gz',
        }
    expected = {'ramnet-ops-dist-source': f'ramnet_ops-{version}.tar.gz'}
    for torch, cuda_versions in TORCH_CUDA.items():
        stable = torch == '2.14.0'
        artifact_tag = 'torchabi2_10' if stable else f'torch{torch}'
        wheel_tag = 'torchabi2_10' if stable else f"torch{torch.replace('.', '_')}"
        for cuda in cuda_versions:
            for platform, wheel_platform in PLATFORMS.items():
                name = f'ramnet-ops-dist-{platform}-{artifact_tag}-cu{cuda}'
                expected[name] = f'ramnet_ops-{version}-1{wheel_tag}cu{cuda}-cp310-abi3-{wheel_platform}.whl'
    return expected


def verify_source(package):
    run_id = os.environ['BUILD_RUN_ID']
    tag = os.environ['RELEASE_TAG']
    repository = os.environ['GITHUB_REPOSITORY']
    repo_url = f"{os.environ['GITHUB_SERVER_URL']}/{repository}"
    workflow = f'{package}-build.yml'
    require(run_id.isascii() and run_id.isdecimal(),
            f'Invalid build_run_id {run_id!r}. Enter only the numeric ID from {repo_url}/actions/runs/<ID> '
            '(for example, 1234567890), not the full URL, job ID, or run number (#42).')
    require(tag, f'tag is required. Enter an existing remote tag named {package}/v<VERSION>.')
    require(tag.startswith(f'{package}/v'),
            f'Invalid tag {tag!r}. Enter the full tag name {package}/v<VERSION>, not a branch or release URL.')
    run = github(
        f'actions/runs/{run_id}',
        not_found=f'Build run {run_id} was not found or is inaccessible in {repository} (HTTP 404). '
        f'Select a successful build from {repo_url}/actions/workflows/{workflow} '
        'and copy its run ID from the Actions URL. Check actions: read permission if the run exists.',
    )
    require(run['path'] == f'.github/workflows/{workflow}',
            f'Run {run_id} belongs to {run["path"]!r}. Select the successful {package} build, '
            f'not the publish workflow: {repo_url}/actions/workflows/{workflow}')
    require(run['status'] == 'completed' and run['conclusion'] == 'success', 'The original build must have succeeded.')
    require(run['event'] in ('push', 'workflow_dispatch'), 'Publish a push or manual build, not a pull-request build.')
    require(run['head_repository']['full_name'] == os.environ['GITHUB_REPOSITORY'], 'The build must belong to this repository.')

    # Resolve both lightweight and annotated tags to the underlying commit.
    ref = github(
        f"git/ref/tags/{quote(tag, safe='')}",
        not_found=f'Tag {tag!r} was not found or is inaccessible in {repository} (HTTP 404). '
        f'Check the tag name, or create and push it at build commit {run["head_sha"]} before retrying. '
        'A local-only tag is insufficient; this workflow does not create tags. '
        'Check contents: read permission if the remote tag exists.',
    )['object']
    while ref['type'] == 'tag':
        ref = github(f"git/tags/{ref['sha']}")['object']
    require(ref['type'] == 'commit' and ref['sha'] == run['head_sha'], 'Tag and build must identify the same commit.')
    content = github(f"contents/packages/{package}/pyproject.toml?ref={run['head_sha']}")
    project = tomllib.loads(base64.b64decode(content['content']).decode('utf-8'))['project']
    require(project['name'] == package, 'Unexpected project name.')
    require(tag == f"{package}/v{project['version']}", 'Tag must match the version at the build commit.')
    urls = project.get('urls', {})
    require(all(urls.get(key, '').startswith('https://') and '<' not in urls[key]
                for key in ('Source', 'Issues', 'Documentation')), 'Complete project.urls before publishing.')
    version = str(Version(project['version']))
    expected = expected_artifacts(version, package)

    artifacts = []
    page = 1
    while True:
        result = github(f'actions/runs/{run_id}/artifacts?per_page=100&page={page}')
        artifacts.extend(result['artifacts'])
        if len(artifacts) >= result['total_count']:
            break
        page += 1
    selected = {}
    ids = []
    for artifact in artifacts:
        name = artifact['name']
        if not name.startswith(f'{package}-dist-'):
            continue
        require(not artifact['expired'], f'Artifact expired: {name}; rebuild before publishing.')
        require(name in expected, f'Unexpected or duplicate artifact: {name}')
        selected[name] = expected.pop(name)
        ids.append(str(artifact['id']))
    require(not expected, f'Missing build artifacts: {sorted(expected)}')
    MANIFEST.write_text(json.dumps({
        'package': package, 'version': version, 'tag': tag, 'commit': run['head_sha'], 'artifacts': selected,
    }, indent=2), encoding='utf-8')
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write(f"artifact-ids={','.join(ids)}\n")
        output.write(f"prerelease={str(Version(version).is_prerelease).lower()}\n")
    print(f"Verified {tag} at {run['head_sha']}: {run['html_url']}; {len(ids)} artifacts.", flush=True)


def verify_metadata(data, version, package):
    metadata = BytesParser().parsebytes(data)
    require(canonicalize_name(metadata['Name']) == package, 'Unexpected distribution name.')
    require(Version(metadata['Version']) == Version(version), 'Distribution metadata version differs from the tag.')


def verify_wheel(path, version, package):
    with WheelFile(path) as wheel:
        names = wheel.namelist()
        require(len(names) == len(set(names)), f'Duplicate wheel members: {path.name}')
        records = csv.reader(io.StringIO(wheel.read(wheel.record_path).decode('utf-8')))
        require(all(row[0] in names for row in records), f'Missing wheel members: {path.name}')
        # Reading every member verifies both ZIP CRC and the RECORD hashes.
        for member in wheel.infolist():
            if not member.is_dir():
                with wheel.open(member) as stream:
                    while stream.read(1024 * 1024):
                        pass
        verify_metadata(wheel.read(f'{wheel.dist_info_path}/METADATA'), version, package)
        info = BytesParser().parsebytes(wheel.read(f'{wheel.dist_info_path}/WHEEL'))
        tags = {tag for value in info.get_all('Tag', []) for tag in parse_tag(value)}
        _, _, _, expected_tags = parse_wheel_filename(path.name)
        require(tags == expected_tags, f'Wheel metadata tags differ from its filename: {path.name}')
        if package == 'ramnet':
            require(info['Root-Is-Purelib'] == 'true' and tags == parse_tag('py3-none-any'),
                    f'Expected a pure Python wheel: {path.name}')
            require('ramnet/__init__.py' in names, f'Missing ramnet package: {path.name}')
            require(not any(name.endswith(('.so', '.pyd', '.dll', '.dylib')) for name in names),
                    f'Unexpected native library in the model wheel: {path.name}')
        else:
            suffix = '.pyd' if path.name.endswith('-win_amd64.whl') else '.so'
            require(any(name.startswith('ramnet_ops/_C.') and name.endswith(suffix) for name in names),
                    f'Missing native extension: {path.name}')


def verify_artifacts():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    package = manifest['package']
    expected = manifest['artifacts']
    root = Path('artifacts')
    require({path.name for path in root.iterdir()} == set(expected), 'Downloaded artifact inventory differs from the verified run.')
    files = []
    for name, filename in expected.items():
        directory = root / name
        path = directory / filename
        require(list(directory.iterdir()) == [path] and path.is_file(), f'Expected exactly {filename} in {name}.')
        if path.suffix == '.whl':
            verify_wheel(path, manifest['version'], package)
        else:
            with tarfile.open(path, 'r:gz') as archive:
                metadata = archive.extractfile(f"{package.replace('-', '_')}-{manifest['version']}/PKG-INFO")
                require(metadata is not None, 'Source archive has no PKG-INFO.')
                verify_metadata(metadata.read(), manifest['version'], package)
        files.append(path)
        print(f'Verified {filename}', flush=True)
    dist = Path('dist')
    dist.mkdir()
    for path in files:
        shutil.copyfile(path, dist / path.name)
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as summary:
        wheels = sum(path.suffix == '.whl' for path in files)
        summary.write(f"Verified **{manifest['tag']}** at `{manifest['commit']}`: {wheels} wheel(s) and one sdist.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('source', 'artifacts'))
    parser.add_argument('--package', choices=('ramnet', 'ramnet-ops'), default='ramnet-ops',
                        help='Package to release; artifact verification uses the recorded source manifest.')
    args = parser.parse_args()
    try:
        if args.stage == 'source':
            verify_source(args.package)
        else:
            verify_artifacts()
    except ValueError as error:
        message = str(error).replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
        print(f'::error title=Release verification failed::{message}', flush=True)
        raise SystemExit(1)
