#!/usr/bin/env python3
"""Extract Ether0's small X11 runtime dependency closure from a container."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


_LIBRARY_GLOBS = (
    "libX11.so*",
    "libXau.so*",
    "libXdmcp.so*",
    "libXext.so*",
    "libXrender.so*",
    "libbsd.so*",
    "libmd.so*",
    "libxcb.so*",
)
_REQUIRED_SONAMES = ("libX11.so.6", "libXext.so.6", "libXrender.so.1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the native libraries required by RDKit drawing imports. "
            "The source must be an x86_64 squashfs/Enroot image compatible with "
            "the NeMo-RL container."
        )
    )
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--library-dir",
        default="usr/lib/x86_64-linux-gnu",
        help="Library directory inside the source image",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    container = args.container.resolve(strict=True)
    output_dir = args.output_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="ether0-native-libs-") as tmp:
        extract_root = Path(tmp)
        patterns = [f"{args.library_dir}/{name}" for name in _LIBRARY_GLOBS]
        subprocess.run(
            ["unsquashfs", "-f", "-d", str(extract_root), str(container), *patterns],
            check=True,
        )
        extracted_dir = extract_root / args.library_dir
        if not extracted_dir.is_dir():
            raise FileNotFoundError(f"No libraries were extracted from {container}:{args.library_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in extracted_dir.iterdir():
            destination = output_dir / path.name
            if path.is_symlink():
                destination.unlink(missing_ok=True)
                destination.symlink_to(path.readlink())
            elif path.is_file():
                shutil.copy2(path, destination)

    missing = [name for name in _REQUIRED_SONAMES if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Native library extraction is incomplete; missing {', '.join(missing)}")
    print(f"Prepared Ether0 native libraries in {output_dir}")


if __name__ == "__main__":
    main()
