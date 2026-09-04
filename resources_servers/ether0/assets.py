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

import hashlib
from pathlib import Path


MOLTRANS_CHECKPOINT_SHA256 = "1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587"
ZINC20_CATALOG_SHA256 = "7c6939251f2c86b7ee996faa727478bb6ceeffc675ab27f7f7e3a181a288b766"
ZINC20_CATALOG_URL = (
    "https://www.dropbox.com/scl/fi/4y3979xonia1ifjdro4ib/zinc20-new.bloom"
    "?rlkey=mi8sbbm4qrc9y0v1ssc29o2nd&st=c8iukvsp&dl=1"
)
ZINC20_RELATIVE_PATH = Path(".cache/molbloom/zinc20.bloom")


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
