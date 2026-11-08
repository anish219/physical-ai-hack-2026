"""Install the isolated Linux sponsor viewer; never starts robot hardware."""

import argparse
import hashlib
import ipaddress
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"


def download(url, target, expected=None):
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {url}")
    target.write_bytes(payload)
    return {"url": url, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="NUC LAN IPv4 address reachable by operator and judges")
    args = parser.parse_args()
    host = str(ipaddress.IPv4Address(args.host))
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        parser.error("This launcher targets Linux x86-64; use it on the NUC.")
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        parser.error("Run setup with Python 3.12 or 3.13, for example ~/dream-env/bin/python.")
    RUNTIME.mkdir(exist_ok=True)
    env = ROOT / ".venv"
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", "--python", sys.executable, "--allow-existing", str(env)], check=True)
        install = [uv, "pip", "install", "--python", str(env / "bin/python")]
    else:
        if not (env / "bin/python").exists():
            venv.create(env, with_pip=True)
        install = [str(env / "bin/python"), "-m", "pip", "install"]
    subprocess.run([*install, "-r", str(ROOT / "requirements.txt")], check=True)
    versions = {}
    with urllib.request.urlopen("https://pypi.org/pypi/dimos/0.0.15/json", timeout=30) as response:
        metadata = json.load(response)
    source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    archive = RUNTIME / "dimos-0.0.15.tar.gz"
    versions["dimos"] = download(source["url"], archive, source["digests"]["sha256"])
    with tarfile.open(archive) as package:
        package.extractall(RUNTIME, filter="data")
    server_url = "https://github.com/livekit/livekit/releases/download/v1.13.11/livekit_1.13.11_linux_amd64.tar.gz"
    archive = RUNTIME / "livekit-linux.tar.gz"
    versions["livekit"] = download(server_url, archive)
    (RUNTIME / "livekit").mkdir(exist_ok=True)
    with tarfile.open(archive) as package:
        package.extractall(RUNTIME / "livekit", filter="data")
    archive = RUNTIME / "livekit-client.tgz"
    versions["livekit_client"] = download(
        "https://registry.npmjs.org/livekit-client/-/livekit-client-2.23.0.tgz", archive
    )
    with tarfile.open(archive) as package:
        for source, dest in [
            ("package/dist/livekit-client.umd.js", "livekit-client.js"),
            ("package/LICENSE", "livekit-client.LICENSE"),
        ]:
            member = package.extractfile(source)
            if member is None:
                raise ValueError(f"Missing {source}")
            (RUNTIME / dest).write_bytes(member.read())
    credentials = ROOT / "livekit-secret.json"
    if not credentials.exists():
        # Restrictive permissions from creation, before any secret is written.
        with open(credentials, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as output:
            json.dump({"key": "recorderkey", "secret": secrets.token_urlsafe(36)}, output)
    credentials.chmod(0o600)
    config = json.loads(credentials.read_text())
    yaml = ROOT / "livekit.yaml"
    yaml.touch(mode=0o600, exist_ok=True)
    yaml.chmod(0o600)
    yaml.write_text(
        "port: 7880\nbind_addresses:\n  - 0.0.0.0\nrtc:\n  tcp_port: 7881\n  udp_port: 7882\n"
        f"  node_ip: {host}\n  use_external_ip: false\nkeys:\n  {config['key']}: {config['secret']}\n"
        "logging:\n  level: warn\n"
    )
    (ROOT / "station.json").write_text(json.dumps({"livekit_host": host}))
    (ROOT / "versions.json").write_text(json.dumps(versions, indent=2))
    print("Installed in isolated environment. Start: bash run_recorder/start.sh")


if __name__ == "__main__":
    main()
