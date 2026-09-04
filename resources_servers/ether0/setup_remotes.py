# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import subprocess
from pathlib import Path

from assets import (
    ZINC20_CATALOG_SHA256,
    ZINC20_CATALOG_URL,
    ZINC20_RELATIVE_PATH,
    file_sha256,
)
from setup_ether0 import ETHER0_PATH, ensure_ether0


DEFAULT_VENV_PATH = Path(__file__).parent / ".remotes-venv"
DEFAULT_RUNTIME_HOME = Path(__file__).parent / ".remotes-home"
CONSTRAINTS_PATH = Path(__file__).parent / "remotes-constraints.txt"


def prefetch_zinc20(runtime_home: Path) -> Path:
    catalog_path = runtime_home / ZINC20_RELATIVE_PATH
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if catalog_path.is_file() and file_sha256(catalog_path) == ZINC20_CATALOG_SHA256:
        return catalog_path

    partial_path = catalog_path.with_suffix(".bloom.partial")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(partial_path),
            ZINC20_CATALOG_URL,
        ],
        check=True,
    )
    digest = file_sha256(partial_path)
    if digest != ZINC20_CATALOG_SHA256:
        raise ValueError(f"ZINC20 catalog SHA256 mismatch: expected {ZINC20_CATALOG_SHA256}, got {digest}")
    partial_path.replace(catalog_path)
    return catalog_path


def _install_shared_python(python_version: str, install_dir: Path) -> Path:
    install_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["UV_PYTHON_INSTALL_DIR"] = str(install_dir)
    subprocess.run(
        ["uv", "python", "install", "--install-dir", str(install_dir), "--no-bin", python_version],
        check=True,
        env=environment,
    )
    result = subprocess.run(
        ["uv", "python", "find", "--no-project", "--managed-python", python_version],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    python_executable = Path(result.stdout.strip()).resolve()
    if not python_executable.is_file() or not python_executable.is_relative_to(install_dir.resolve()):
        raise RuntimeError(f"uv resolved Python outside the shared install directory: {python_executable}")
    return python_executable


def setup_remotes_venv(
    venv_path: Path,
    runtime_home: Path,
    python_version: str,
    python_install_dir: Path,
) -> Path:
    """Build the legacy Ether0 model runtime outside Gym's Python 3.13 venv."""
    ensure_ether0()
    venv_path = venv_path.resolve()
    runtime_home = runtime_home.resolve()
    shared_python = _install_shared_python(python_version, python_install_dir.resolve())
    subprocess.run(
        [
            "uv",
            "venv",
            "--clear",
            "--seed",
            "--python",
            str(shared_python),
            str(venv_path),
        ],
        check=True,
        cwd=ETHER0_PATH,
    )
    python_executable = venv_path / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_executable),
            "--torch-backend",
            "cpu",
            "--constraint",
            str(CONSTRAINTS_PATH),
            f"{ETHER0_PATH / 'packages' / 'remotes'}[serve]",
            "molbloom==2.3.4",
        ],
        check=True,
        cwd=ETHER0_PATH,
    )
    prefetch_zinc20(runtime_home)
    environment = os.environ.copy()
    environment["HOME"] = str(runtime_home)
    subprocess.run(
        [
            str(python_executable),
            "-c",
            "import ether0.server, onmt, torch; print(torch.__version__)",
        ],
        check=True,
        env=environment,
    )
    return python_executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ether0's Python 3.12 remotes sidecar venv")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV_PATH)
    parser.add_argument("--runtime-home", type=Path, default=DEFAULT_RUNTIME_HOME)
    parser.add_argument("--python", default="3.12", dest="python_version")
    parser.add_argument(
        "--python-install-dir",
        type=Path,
        default=None,
        help="Shared directory for the managed Python runtime (default: sibling of --venv)",
    )
    args = parser.parse_args()
    python_install_dir = args.python_install_dir or args.venv.parent / "python"
    print(
        setup_remotes_venv(
            args.venv.resolve(),
            args.runtime_home.resolve(),
            args.python_version,
            python_install_dir.resolve(),
        )
    )


if __name__ == "__main__":
    main()
