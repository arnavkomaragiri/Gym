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

"""Ether0 remotes server with one eagerly loaded instance of each model."""

import asyncio
import io
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


auth_scheme = HTTPBearer()
logger = logging.getLogger(__name__)

SolubilityPrediction = npt.NDArray[np.float32] | Literal[False]


def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),  # noqa: B008
) -> str:
    expected_token = os.environ["ETHER0_REMOTES_API_TOKEN"]
    if not secrets.compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


class SharedMolecularTransformer:
    """An OpenNMT translator loaded once and serialized across requests."""

    def __init__(self, model_path: Path) -> None:
        from onmt import opts  # noqa: PLC0415
        from onmt.translate.translator import build_translator  # noqa: PLC0415
        from onmt.utils.logging import init_logger  # noqa: PLC0415
        from onmt.utils.parse import ArgumentParser  # noqa: PLC0415

        parser = ArgumentParser()
        opts.config_opts(parser)
        opts.translate_opts(parser)
        args = [
            f"--model={model_path}",
            "--src=/dev/null",
            "--output=/dev/null",
            "--batch_size=64",
            "--beam_size=50",
            "--max_length=300",
        ]
        options = parser.parse_args(args)
        ArgumentParser.validate_translate_opts(options)
        self._output = io.StringIO()
        self._translator = build_translator(
            options,
            logger=init_logger(options.log_file),
            out_file=self._output,
            report_score=False,
        )

    def run(self, reaction: str) -> tuple[str, uuid.UUID]:
        from ether0.server import MolecularTransformer  # noqa: PLC0415

        self._output.seek(0)
        self._output.truncate(0)
        _, predictions = self._translator.translate(
            src=[MolecularTransformer.smiles_tokenizer(reaction)],
            src_feats={},
            tgt=None,
            batch_size=64,
            batch_type="sents",
            attn_debug=False,
            align_debug=False,
        )
        return predictions[0][0].replace(" ", "").strip(), uuid.uuid4()


class SharedSolubility:
    """KDESol inference without Keras predict() creating a pool per ensemble member."""

    def __init__(self) -> None:
        from molsol import KDESol  # noqa: PLC0415

        self._model = KDESol()

    def _predict_batch(
        self,
        encoded_smiles: list[list[int]],
    ) -> list[npt.NDArray[np.float32]]:
        batch = np.asarray(encoded_smiles)
        singleton_batch = len(batch) == 1
        if singleton_batch:
            # KDESol's SqueezeLayer drops the batch axis for B=1.
            batch = np.repeat(batch, 2, axis=0)
        predictions = np.stack(
            [member(batch, training=False).numpy() for member in self._model.model.models],
            axis=0,
        )
        means = np.mean(predictions, axis=0)
        standard_deviations = np.std(predictions, axis=0)
        if singleton_batch:
            means = means[:1]
            standard_deviations = standard_deviations[:1]
        return [
            np.array(
                [mean[0], mean[1], standard_deviation[0]],
                dtype=np.float32,
            )
            for mean, standard_deviation in zip(means, standard_deviations, strict=True)
        ]

    def _encode(self, smiles: str) -> list[int] | None:
        return self._model.stoi(self._model.encoder(smiles))

    def run_batch(self, smiles_batch: list[str]) -> list[SolubilityPrediction]:
        from rdkit import Chem  # noqa: PLC0415

        results: list[SolubilityPrediction] = [False] * len(smiles_batch)
        valid_indices: list[int] = []
        encoded_smiles: list[list[int]] = []
        for index, smiles in enumerate(smiles_batch):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            canonical_smiles = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
            )
            encoded = self._encode(canonical_smiles)
            if encoded is None:
                encoded = self._encode(smiles)
            if encoded is None:
                continue
            valid_indices.append(index)
            encoded_smiles.append(encoded)

        if encoded_smiles:
            predictions = self._predict_batch(encoded_smiles)
            for index, prediction in zip(valid_indices, predictions, strict=True):
                results[index] = prediction
        return results

    def run(self, smiles: str) -> SolubilityPrediction:
        return self.run_batch([smiles])[0]


@dataclass
class _SolubilityWorkItem:
    smiles: str
    future: asyncio.Future[SolubilityPrediction]
    queued_at: float


class SolubilityBatcher:
    """Coalesce independent requests into bounded KDESol inference batches."""

    def __init__(
        self,
        model: SharedSolubility,
        *,
        max_batch_size: int,
        batch_wait_seconds: float,
    ) -> None:
        self._model = model
        self._max_batch_size = max_batch_size
        self._batch_wait_seconds = batch_wait_seconds
        self._queue: asyncio.Queue[_SolubilityWorkItem | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._batch_count = 0
        self._request_count = 0

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("Solubility batcher is already running")
        self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        worker = self._worker
        if worker is None:
            return
        await self._queue.put(None)
        await worker
        self._worker = None

    async def predict(self, smiles: str) -> SolubilityPrediction:
        if self._worker is None:
            raise RuntimeError("Solubility batcher is not running")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SolubilityPrediction] = loop.create_future()
        await self._queue.put(
            _SolubilityWorkItem(
                smiles=smiles,
                future=future,
                queued_at=loop.time(),
            )
        )
        return await future

    async def _collect_batch(
        self,
        first_item: _SolubilityWorkItem,
    ) -> tuple[list[_SolubilityWorkItem], bool]:
        batch = [first_item]
        should_stop = False
        deadline = asyncio.get_running_loop().time() + self._batch_wait_seconds
        while len(batch) < self._max_batch_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except TimeoutError:
                break
            if item is None:
                should_stop = True
                break
            batch.append(item)
        return batch, should_stop

    async def _run(self) -> None:
        while True:
            first_item = await self._queue.get()
            if first_item is None:
                return
            batch, should_stop = await self._collect_batch(first_item)
            started_at = time.monotonic()
            try:
                predictions = await asyncio.to_thread(
                    self._model.run_batch,
                    [item.smiles for item in batch],
                )
            except Exception as error:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(error)
            else:
                for item, prediction in zip(batch, predictions, strict=True):
                    if not item.future.done():
                        item.future.set_result(prediction)

            self._batch_count += 1
            self._request_count += len(batch)
            if self._batch_count == 1 or self._batch_count % 100 == 0:
                oldest_queue_seconds = max(0.0, started_at - batch[0].queued_at)
                logger.info(
                    "Ether0 solubility batches=%d requests=%d last_batch=%d "
                    "oldest_queue_seconds=%.3f inference_seconds=%.3f",
                    self._batch_count,
                    self._request_count,
                    len(batch),
                    oldest_queue_seconds,
                    time.monotonic() - started_at,
                )
            if should_stop:
                return


class PurchasabilityModel(Protocol):
    def run(self, smiles: str) -> bool: ...


@dataclass
class RemotesModels:
    transformer: SharedMolecularTransformer
    purchasability: PurchasabilityModel
    solubility: SharedSolubility
    transformer_lock: asyncio.Lock
    purchasability_lock: asyncio.Lock
    solubility_batcher: SolubilityBatcher


def _build_models() -> tuple[SharedMolecularTransformer, PurchasabilityModel, SharedSolubility]:
    from ether0.server import MolBloom  # noqa: PLC0415

    model_path = Path(os.environ["ETHER0_REMOTES_MOLTRANS_MODEL_PATH"])
    return SharedMolecularTransformer(model_path), MolBloom(), SharedSolubility()


def _models_with_locks(
    loaded_models: tuple[SharedMolecularTransformer, PurchasabilityModel, SharedSolubility],
) -> RemotesModels:
    transformer, purchasability, solubility = loaded_models
    max_batch_size = int(os.environ["ETHER0_REMOTES_SOLUBILITY_MAX_BATCH_SIZE"])
    batch_wait_milliseconds = float(os.environ["ETHER0_REMOTES_SOLUBILITY_BATCH_WAIT_MILLISECONDS"])
    if max_batch_size < 1:
        raise ValueError("ETHER0_REMOTES_SOLUBILITY_MAX_BATCH_SIZE must be positive")
    if batch_wait_milliseconds < 0:
        raise ValueError("ETHER0_REMOTES_SOLUBILITY_BATCH_WAIT_MILLISECONDS must be non-negative")
    return RemotesModels(
        transformer=transformer,
        purchasability=purchasability,
        solubility=solubility,
        transformer_lock=asyncio.Lock(),
        purchasability_lock=asyncio.Lock(),
        solubility_batcher=SolubilityBatcher(
            solubility,
            max_batch_size=max_batch_size,
            batch_wait_seconds=batch_wait_milliseconds / 1000.0,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = _models_with_locks(await asyncio.to_thread(_build_models))
    app.state.models = models
    await models.solubility_batcher.start()
    try:
        yield
    finally:
        await models.solubility_batcher.close()


app = FastAPI(
    title="ether0 remotes server",
    dependencies=[Depends(validate_token)],
    lifespan=lifespan,
)


class MolTransRequest(BaseModel):
    reaction: str


class MolBloomRequest(BaseModel):
    smiles: list[str] | str


class SmilesRequest(BaseModel):
    smiles: str


def _canonicalize_reaction(reaction: str) -> str:
    from ether0.server import MolecularTransformer  # noqa: PLC0415

    reaction = reaction.replace(" ", "")
    if reaction.count(">") != 2:  # noqa: PLR2004
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The reaction should have two '>' characters, and no spaces.",
        )
    reactants_and_reagents = reaction.split(">")[:-1]
    return MolecularTransformer.canonicalize_smiles(".".join(part for part in reactants_and_reagents if part))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/translate")
async def translate_endpoint(request: MolTransRequest) -> dict[str, str | uuid.UUID]:
    reaction = _canonicalize_reaction(request.reaction)
    models: RemotesModels = app.state.models
    async with models.transformer_lock:
        product, job_id = await asyncio.to_thread(models.transformer.run, reaction)
    return {
        "product": product,
        "id": job_id,
        "reaction": reaction + ">>" + product,
    }


@app.post("/is_purchasable")
async def is_purchasable_endpoint(request: MolBloomRequest) -> dict[str, bool]:
    smiles = [request.smiles] if isinstance(request.smiles, str) else request.smiles
    models: RemotesModels = app.state.models
    async with models.purchasability_lock:
        results = await asyncio.to_thread(lambda: {value: models.purchasability.run(value) for value in smiles})
    return results


@app.post("/compute_solubility")
async def compute_solubility_endpoint(
    request: SmilesRequest,
) -> dict[str, float] | dict[str, str]:
    if "." in request.smiles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only single molecules are supported",
        )
    models: RemotesModels = app.state.models
    prediction = await models.solubility_batcher.predict(request.smiles)
    if prediction is False:
        return {"error": "Solubility prediction failed."}
    mean, aleatoric_uncertainty, epistemic_uncertainty = prediction.tolist()
    return {
        "mean": mean,
        "au": aleatoric_uncertainty,
        "eu": epistemic_uncertainty,
    }
