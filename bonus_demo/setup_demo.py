"""Download the isolated, local-only bonus demo's pinned dependencies."""

import hashlib
import json
import subprocess
import tarfile
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
DIMOS_VERSION = "0.0.15"
LIVEKIT_VERSION = "1.13.11"
CLIENT_VERSION = "2.23.0"


def download(url, target):
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, target)


def main():
    RUNTIME.mkdir(exist_ok=True)
    env_dir = ROOT / ".venv"
    if not env_dir.exists():
        venv.create(env_dir, with_pip=True)
    python = env_dir / "Scripts" / "python.exe"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )

    # Import the official source distribution directly. Its Python recorder
    # doesn't need the unrelated native navigation extension built by pip.
    with urllib.request.urlopen(f"https://pypi.org/pypi/dimos/{DIMOS_VERSION}/json") as response:
        metadata = json.load(response)
    source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    archive = RUNTIME / f"dimos-{DIMOS_VERSION}.tar.gz"
    download(source["url"], archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == source["digests"]["sha256"]
    with tarfile.open(archive) as package:
        package.extractall(RUNTIME, filter="data")

    server_url = (
        f"https://github.com/livekit/livekit/releases/download/v{LIVEKIT_VERSION}/"
        f"livekit_{LIVEKIT_VERSION}_windows_amd64.zip"
    )
    server_zip = RUNTIME / "livekit.zip"
    download(server_url, server_zip)
    with zipfile.ZipFile(server_zip) as package:
        package.extractall(RUNTIME / "livekit")

    sdk_url = f"https://registry.npmjs.org/livekit-client/-/livekit-client-{CLIENT_VERSION}.tgz"
    sdk_tar = RUNTIME / "livekit-client.tgz"
    download(sdk_url, sdk_tar)
    with tarfile.open(sdk_tar) as package:
        sdk = package.extractfile("package/dist/livekit-client.umd.js")
        (RUNTIME / "livekit-client.umd.js").write_bytes(sdk.read())
        license_file = package.extractfile("package/LICENSE")
        (RUNTIME / "livekit-client.LICENSE").write_bytes(license_file.read())

    manifest = {
        "dimos": {"version": DIMOS_VERSION, "url": source["url"], "sha256": source["digests"]["sha256"]},
        "livekit_server": {"version": LIVEKIT_VERSION, "url": server_url},
        "livekit_client": {"version": CLIENT_VERSION, "url": sdk_url},
    }
    (RUNTIME / "versions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready. Run: .\\bonus_demo\\start.ps1", flush=True)


if __name__ == "__main__":
    main()
