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

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import remotes_app
from fastapi.testclient import TestClient


_REMOTES_ENV = {
    "ETHER0_REMOTES_API_TOKEN": "test-token",
    "ETHER0_REMOTES_SOLUBILITY_MAX_BATCH_SIZE": "32",
    "ETHER0_REMOTES_SOLUBILITY_BATCH_WAIT_MILLISECONDS": "10",
}


def test_models_are_shared_across_requests() -> None:
    transformer = MagicMock()
    transformer.run.return_value = ("CCO", "job-id")
    purchasability = MagicMock()
    purchasability.run.return_value = True
    solubility = MagicMock()
    solubility.run_batch.return_value = [np.array([1.0, 0.1, 0.2], dtype=np.float32)]
    headers = {"Authorization": "Bearer test-token"}

    with (
        patch.dict("os.environ", _REMOTES_ENV),
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
    solubility.run_batch.assert_called_once_with(["CCO"])


def test_solubility_requests_are_batched() -> None:
    solubility = MagicMock()
    solubility.run_batch.side_effect = lambda values: [
        np.array([float(index), 0.1, 0.2], dtype=np.float32) for index, _ in enumerate(values)
    ]

    async def run_requests() -> list[np.ndarray]:
        batcher = remotes_app.SolubilityBatcher(
            solubility,
            max_batch_size=4,
            batch_wait_seconds=0.05,
        )
        await batcher.start()
        try:
            return await asyncio.gather(
                batcher.predict("CC"),
                batcher.predict("CCC"),
                batcher.predict("CCCC"),
            )
        finally:
            await batcher.close()

    predictions = asyncio.run(run_requests())

    solubility.run_batch.assert_called_once_with(["CC", "CCC", "CCCC"])
    assert [prediction[0] for prediction in predictions] == [0.0, 1.0, 2.0]


def test_singleton_solubility_batch_preserves_batch_axis() -> None:
    member = MagicMock()
    member.return_value.numpy.return_value = np.array(
        [[1.0, 0.1], [1.0, 0.1]],
        dtype=np.float32,
    )
    solubility = remotes_app.SharedSolubility.__new__(remotes_app.SharedSolubility)
    solubility._model = MagicMock()
    solubility._model.model.models = [member]

    prediction = solubility._predict_batch([[1, 2, 3]])

    assert len(prediction) == 1
    np.testing.assert_array_equal(
        prediction[0],
        np.array([1.0, 0.1, 0.0], dtype=np.float32),
    )
    assert member.call_args.args[0].shape == (2, 3)


def test_authentication_is_required() -> None:
    with patch.dict("os.environ", _REMOTES_ENV):
        response = TestClient(remotes_app.app).get("/health")
    assert response.status_code == 401
