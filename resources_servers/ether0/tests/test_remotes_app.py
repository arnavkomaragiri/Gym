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

from unittest.mock import MagicMock, patch

import numpy as np
import remotes_app
from fastapi.testclient import TestClient


def test_models_are_shared_across_requests() -> None:
    transformer = MagicMock()
    transformer.run.return_value = ("CCO", "job-id")
    purchasability = MagicMock()
    purchasability.run.return_value = True
    solubility = MagicMock()
    solubility.run.return_value = np.array([1.0, 0.1, 0.2], dtype=np.float32)
    headers = {"Authorization": "Bearer test-token"}

    with (
        patch.dict("os.environ", {"ETHER0_REMOTES_API_TOKEN": "test-token"}),
        patch.object(
            remotes_app,
            "_build_models",
            return_value=(transformer, purchasability, solubility),
        ) as build_models,
        patch.object(remotes_app, "_canonicalize_reaction", return_value="CC.O"),
        TestClient(remotes_app.app) as client,
    ):
        assert client.get("/health", headers=headers).status_code == 200
        assert client.post("/translate", json={"reaction": "CC.O>>CCO"}, headers=headers).status_code == 200
        assert client.post("/translate", json={"reaction": "CC.O>>CCO"}, headers=headers).status_code == 200
        assert client.post("/is_purchasable", json={"smiles": "CCO"}, headers=headers).json() == {"CCO": True}
        assert client.post("/compute_solubility", json={"smiles": "CCO"}, headers=headers).json()["mean"] == 1.0

    build_models.assert_called_once()
    assert transformer.run.call_count == 2
    purchasability.run.assert_called_once_with("CCO")
    solubility.run.assert_called_once_with("CCO")


def test_authentication_is_required() -> None:
    with patch.dict("os.environ", {"ETHER0_REMOTES_API_TOKEN": "test-token"}):
        response = TestClient(remotes_app.app).get("/health")
    assert response.status_code == 401
