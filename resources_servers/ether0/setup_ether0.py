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

import logging
import subprocess
import sys
from pathlib import Path


_REPO_URL = "https://github.com/Future-House/ether0"
ETHER0_COMMIT = "b77e0d8eb06f62c8528da72bff22d4c7fee11371"
ETHER0_PATH = Path(__file__).parent / ".ether0"
ETHER0_SRC_PATH = ETHER0_PATH / "src"


def ensure_ether0() -> None:
    if not ETHER0_PATH.exists():
        logging.info("Cloning ether0 into %s", ETHER0_PATH)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", _REPO_URL, str(ETHER0_PATH)],
            check=True,
        )
    current_commit = subprocess.run(
        ["git", "-C", str(ETHER0_PATH), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != ETHER0_COMMIT:
        subprocess.run(
            ["git", "-C", str(ETHER0_PATH), "fetch", "--depth=1", "origin", ETHER0_COMMIT],
            check=True,
        )
    if current_commit != ETHER0_COMMIT or not (ETHER0_SRC_PATH / "ether0").is_dir():
        logging.info("Checking out ether0 commit %s", ETHER0_COMMIT)
        subprocess.run(
            ["git", "-C", str(ETHER0_PATH), "checkout", "--detach", ETHER0_COMMIT],
            check=True,
        )
    if str(ETHER0_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(ETHER0_SRC_PATH))
