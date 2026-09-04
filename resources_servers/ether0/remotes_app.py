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
import os
import secrets
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

    def _predict(self, smiles: str) -> npt.NDArray[np.float32] | None:
        encoded = self._model.stoi(self._model.encoder(smiles))
        if encoded is None:
            return None
        batch = np.array([encoded, encoded])
        predictions = np.stack(
            [member(batch, training=False).numpy() for member in self._model.model.models],
            axis=0,
        )
        mean = np.mean(predictions, axis=0)[0]
        standard_deviation = np.std(predictions, axis=0)[0]
        return np.array(
            [mean[0], mean[1], standard_deviation[0]],
            dtype=np.float32,
        )

    def run(self, smiles: str) -> npt.NDArray[np.float32] | Literal[False]:
        from rdkit import Chem  # noqa: PLC0415

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
        )
        prediction = self._predict(canonical_smiles)
        if prediction is None:
            prediction = self._predict(smiles)
        return prediction if prediction is not None else False


class PurchasabilityModel(Protocol):
    def run(self, smiles: str) -> bool: ...


@dataclass
class RemotesModels:
    transformer: SharedMolecularTransformer
    purchasability: PurchasabilityModel
    solubility: SharedSolubility
    transformer_lock: asyncio.Lock
    purchasability_lock: asyncio.Lock
    solubility_lock: asyncio.Lock


def _build_models() -> tuple[SharedMolecularTransformer, PurchasabilityModel, SharedSolubility]:
    from ether0.server import MolBloom  # noqa: PLC0415

    model_path = Path(os.environ["ETHER0_REMOTES_MOLTRANS_MODEL_PATH"])
    return SharedMolecularTransformer(model_path), MolBloom(), SharedSolubility()


def _models_with_locks(
    loaded_models: tuple[SharedMolecularTransformer, PurchasabilityModel, SharedSolubility],
) -> RemotesModels:
    transformer, purchasability, solubility = loaded_models
    return RemotesModels(
        transformer=transformer,
        purchasability=purchasability,
        solubility=solubility,
        transformer_lock=asyncio.Lock(),
        purchasability_lock=asyncio.Lock(),
        solubility_lock=asyncio.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = _models_with_locks(await asyncio.to_thread(_build_models))
    yield


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
    async with models.solubility_lock:
        prediction: npt.NDArray[np.float32] | Literal[False] = await asyncio.to_thread(
            models.solubility.run,
            request.smiles,
        )
    if prediction is False:
        return {"error": "Solubility prediction failed."}
    mean, aleatoric_uncertainty, epistemic_uncertainty = prediction.tolist()
    return {
        "mean": mean,
        "au": aleatoric_uncertainty,
        "eu": epistemic_uncertainty,
    }
