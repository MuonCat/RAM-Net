"""Create a GitHub Release or verify an existing one without modifying it."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def main() -> None:
    tag = os.environ['RELEASE_TAG']
    repository = os.environ['GH_REPO']
    request = Request(
        f"{os.environ['GITHUB_API_URL']}/repos/{repository}/releases/tags/{quote(tag, safe='')}",
        headers={
            'Accept': 'application/vnd.github+json',
            'Authorization': f"Bearer {os.environ['GH_TOKEN']}",
            'X-GitHub-Api-Version': '2022-11-28',
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            release = json.load(response)
    except HTTPError as error:
        if error.code != 404:
            raise
        release = None

    files = sorted(Path('dist').iterdir())
    if release is None:
        subprocess.run([
            'gh', 'release', 'create', tag, *map(str, files),
            '--verify-tag', '--title', tag, '--generate-notes',
            f"--prerelease={os.environ['PRERELEASE']}",
        ], check=True)
        return

    if release['draft']:
        raise SystemExit(f'Cannot reuse GitHub Release {tag}: it is still a draft; publish it before retrying.')

    assets = {asset['name']: asset for asset in release['assets']}
    for path in files:
        asset = assets.get(path.name)
        if asset is None:
            raise SystemExit(f'Cannot reuse GitHub Release {tag}: missing asset {path.name}; existing releases are never modified.')
        with path.open('rb') as stream:
            digest = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
        if asset['state'] != 'uploaded' or asset.get('digest') != digest:
            raise SystemExit(f'Cannot reuse GitHub Release asset {path.name}: SHA-256 differs or is unavailable; existing assets are never overwritten.')
        print(f'Reusing verified asset: {path.name}', flush=True)

    print(f"Reusing GitHub Release without uploading: {release['html_url']}", flush=True)


if __name__ == '__main__':
    main()
