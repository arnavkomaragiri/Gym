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
import logging
import os
import re
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from tenacity import RetryError

from resources_servers.ether0.assets import (
    MOLTRANS_CHECKPOINT_SHA256,
    ZINC20_CATALOG_SHA256,
    ZINC20_RELATIVE_PATH,
    file_sha256,
)
from resources_servers.ether0.setup_ether0 import ensure_ether0


ensure_ether0()

import ether0.clients as ether0_clients  # noqa: E402
from ether0.model_prompts import extract_answer_loose  # noqa: E402
from ether0.models import RewardFunctionInfo  # noqa: E402
from ether0.rewards import EVAL_FUNCTIONS  # noqa: E402

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.judge import JudgeError


logger = logging.getLogger(__name__)

_REMOTE_EVAL_FUNCTIONS = frozenset({"sol_eval", "rxn_forward"})


class _LocalHttpxAdapter:
    """The subset of httpx used by the pinned Ether0 clients module."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._client.post(url, **kwargs)


class Ether0RemotesConfig(BaseModel):
    """Configuration for the Ether0-owned local model sidecar."""

    python_executable: Path = Field(description="Python 3.12 executable in the prebuilt remotes venv")
    model_path: Path = Field(description="Path to the Molecular Transformer checkpoint")
    runtime_home: Path = Field(description="Home containing the prewarmed .cache/molbloom catalog")
    model_sha256: str = Field(
        default=MOLTRANS_CHECKPOINT_SHA256,
        description="Expected Molecular Transformer checkpoint SHA256",
    )
    zinc20_sha256: str = Field(
        default=ZINC20_CATALOG_SHA256,
        description="Expected MolBloom ZINC20 catalog SHA256",
    )
    startup_timeout_seconds: float = Field(
        default=600.0,
        gt=0,
        description="Maximum time for all remotes models to load",
    )
    shutdown_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Grace period before the remotes process is killed",
    )
    cpu_threads: int = Field(
        default=4,
        gt=0,
        description="Thread cap applied to Torch, TensorFlow, BLAS, and OpenMP",
    )


class Ether0ResourcesServerConfig(BaseResourcesServerConfig):
    max_concurrent_verifications: int = Field(
        default=32,
        gt=0,
        description="Maximum synchronous Ether0 reward functions evaluated concurrently",
    )
    remotes: Optional[Ether0RemotesConfig] = Field(
        default=None,
        description="Local remotes sidecar; required by sol_eval and rxn_forward rows",
    )


class Ether0RunRequest(BaseRunRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class Ether0VerifyRequest(Ether0RunRequest, BaseVerifyRequest):
    pass


class Ether0VerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str] = None
    eval_function: Optional[str] = None
    problem_type: Optional[str] = None


class Ether0ResourcesServer(SimpleResourcesServer):
    config: Ether0ResourcesServerConfig
    _verification_semaphore: asyncio.Semaphore = PrivateAttr()
    _remotes_process: Optional[asyncio.subprocess.Process] = PrivateAttr(default=None)
    _remotes_base_url: Optional[str] = PrivateAttr(default=None)
    _remotes_token: Optional[str] = PrivateAttr(default=None)
    _remotes_http_client: Optional[httpx.Client] = PrivateAttr(default=None)
    _previous_client_state: Optional[tuple[Optional[str], dict[str, str], ModuleType]] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.remotes is not None and self.config.num_workers not in (None, 1):
            raise ValueError("Ether0 remotes require num_workers=1 so the models are loaded exactly once")
        self._verification_semaphore = asyncio.Semaphore(self.config.max_concurrent_verifications)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        main_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app: FastAPI):
            async with main_lifespan(app) as maybe_state:
                try:
                    if self.config.remotes is not None:
                        await self._start_remotes()
                    yield maybe_state
                finally:
                    await self._stop_remotes()

        app.router.lifespan_context = lifespan_wrapper
        return app

    def _validate_remotes_files(self, config: Ether0RemotesConfig) -> None:
        if not config.python_executable.is_file():
            raise FileNotFoundError(
                f"Ether0 remotes Python executable not found: {config.python_executable}. "
                "Run setup_remotes.py before starting Gym."
            )
        if not config.model_path.is_file():
            raise FileNotFoundError(f"Ether0 MolTrans checkpoint not found: {config.model_path}")
        if not config.runtime_home.is_dir():
            raise FileNotFoundError(f"Ether0 remotes runtime home not found: {config.runtime_home}")
        zinc20_path = config.runtime_home / ZINC20_RELATIVE_PATH
        if not zinc20_path.is_file():
            raise FileNotFoundError(
                f"Ether0 ZINC20 MolBloom catalog not found: {zinc20_path}. Run setup_remotes.py before starting Gym."
            )
        model_digest = file_sha256(config.model_path)
        if model_digest != config.model_sha256:
            raise ValueError(
                f"Ether0 MolTrans checkpoint SHA256 mismatch: expected {config.model_sha256}, got {model_digest}"
            )
        zinc20_digest = file_sha256(zinc20_path)
        if zinc20_digest != config.zinc20_sha256:
            raise ValueError(
                f"Ether0 ZINC20 catalog SHA256 mismatch: expected {config.zinc20_sha256}, got {zinc20_digest}"
            )

    async def _start_remotes(self) -> None:
        config = self.config.remotes
        if config is None:
            return
        await asyncio.to_thread(self._validate_remotes_files, config)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.set_inheritable(True)
        port = listener.getsockname()[1]
        self._remotes_base_url = f"http://127.0.0.1:{port}"
        self._remotes_token = secrets.token_urlsafe(32)

        environment = os.environ.copy()
        environment.update(
            {
                "ETHER0_REMOTES_API_TOKEN": self._remotes_token,
                "ETHER0_REMOTES_MOLTRANS_MODEL_PATH": str(config.model_path),
                "HOME": str(config.runtime_home),
                "MKL_NUM_THREADS": str(config.cpu_threads),
                "OMP_NUM_THREADS": str(config.cpu_threads),
                "OPENBLAS_NUM_THREADS": str(config.cpu_threads),
                "PYTHONUNBUFFERED": "1",
                "TF_NUM_INTEROP_THREADS": str(config.cpu_threads),
                "TF_NUM_INTRAOP_THREADS": str(config.cpu_threads),
            }
        )
        try:
            self._remotes_process = await asyncio.create_subprocess_exec(
                str(config.python_executable),
                "-m",
                "uvicorn",
                "remotes_app:app",
                "--fd",
                str(listener.fileno()),
                cwd=Path(__file__).parent,
                env=environment,
                pass_fds=(listener.fileno(),),
            )
        finally:
            listener.close()

        ready = False
        try:
            await self._wait_for_remotes_ready(config.startup_timeout_seconds)
            ready = True
        finally:
            if not ready:
                await self._stop_remotes()

        self._previous_client_state = (
            ether0_clients.BASE_URL,
            dict(ether0_clients.HEADERS),
            ether0_clients.httpx,
        )
        self._remotes_http_client = httpx.Client(trust_env=False)
        ether0_clients.BASE_URL = self._remotes_base_url
        ether0_clients.HEADERS = {
            "Authorization": f"Bearer {self._remotes_token}",
            "Content-Type": "application/json",
        }
        ether0_clients.httpx = _LocalHttpxAdapter(self._remotes_http_client)
        logger.info("Ether0 remotes sidecar ready at %s", self._remotes_base_url)

    async def _wait_for_remotes_ready(self, timeout_seconds: float) -> None:
        if self._remotes_process is None or self._remotes_base_url is None or self._remotes_token is None:
            raise RuntimeError("Ether0 remotes process was not initialized")
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        headers = {"Authorization": f"Bearer {self._remotes_token}"}
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._remotes_process.returncode is not None:
                    raise RuntimeError(
                        f"Ether0 remotes exited during startup with code {self._remotes_process.returncode}"
                    )
                try:
                    response = await client.get(f"{self._remotes_base_url}/health", headers=headers)
                    if response.is_success:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError(f"Ether0 remotes did not become ready within {timeout_seconds} seconds")

    async def _stop_remotes(self) -> None:
        if self._previous_client_state is not None:
            (
                ether0_clients.BASE_URL,
                ether0_clients.HEADERS,
                ether0_clients.httpx,
            ) = self._previous_client_state
            self._previous_client_state = None
        if self._remotes_http_client is not None:
            self._remotes_http_client.close()
            self._remotes_http_client = None

        process = self._remotes_process
        self._remotes_process = None
        self._remotes_base_url = None
        self._remotes_token = None
        if process is None or process.returncode is not None:
            return

        process.terminate()
        if self.config.remotes is None:
            raise RuntimeError("Ether0 remotes process exists without remotes configuration")
        try:
            await asyncio.wait_for(process.wait(), timeout=self.config.remotes.shutdown_timeout_seconds)
        except TimeoutError:
            logger.warning("Ether0 remotes did not stop gracefully; killing PID %s", process.pid)
            process.kill()
            await process.wait()

    async def verify(self, body: Ether0VerifyRequest) -> Ether0VerifyResponse:
        text = _extract_last_assistant_text(body)

        meta = body.verifier_metadata or {}
        solution_str = meta.get("solution", "")
        problem_type = meta.get("problem_type", "")

        try:
            reward_info = RewardFunctionInfo.model_validate(solution_str)
        except ValidationError:
            logger.warning("Malformed solution string: %r", solution_str)
            return _response(body, 0.0, None, None, problem_type)

        eval_fn_name = reward_info.fxn_name
        answer_info = reward_info.answer_info

        text = text.replace("<|answer_start|>", "<answer>").replace("<|answer_end|>", "</answer>")
        answer = _extract_answer_multi_format(text)
        if answer is None:
            return _response(body, 0.0, None, eval_fn_name, problem_type)

        choices = meta.get("choices", {})
        if choices and len(answer) == 1 and answer.isalpha():
            answer = choices.get(answer.upper(), answer)

        eval_fn = EVAL_FUNCTIONS.get(eval_fn_name)
        if eval_fn is None:
            logger.warning("Unknown eval function %r", eval_fn_name)
            return _response(body, 0.0, answer, eval_fn_name, problem_type)

        if eval_fn_name in _REMOTE_EVAL_FUNCTIONS and self.config.remotes is None:
            raise JudgeError(
                f"Ether0 eval function {eval_fn_name!r} requires the remotes sidecar, but remotes is not configured"
            )

        try:
            async with self._verification_semaphore:
                reward = await asyncio.to_thread(eval_fn, answer, answer_info)
        except (httpx.HTTPError, RetryError) as error:
            raise JudgeError(
                f"Ether0 remotes call failed for {eval_fn_name!r}: {type(error).__name__}: {error}"
            ) from error

        return _response(body, reward, answer, eval_fn_name, problem_type)


def _response(
    body: Ether0VerifyRequest,
    reward: float,
    extracted_answer: Optional[str],
    eval_function: Optional[str],
    problem_type: str,
) -> Ether0VerifyResponse:
    return Ether0VerifyResponse(
        **body.model_dump(exclude={"extracted_answer", "eval_function", "problem_type"}),
        reward=reward,
        extracted_answer=extracted_answer,
        eval_function=eval_function,
        problem_type=problem_type,
    )


_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
_ANSWER_LETTER_RE = re.compile(r"Answer\s*:\s*([A-Za-z])\s*$", re.MULTILINE)


def _extract_answer_multi_format(text: str) -> str | None:
    # <answer> tags
    ans = extract_answer_loose(text).strip()
    if ans:
        return ans
    # \boxed{}, rightmost
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip() or None
    # Answer: LETTER, rightmost
    matches = _ANSWER_LETTER_RE.findall(text)
    if matches:
        return matches[-1].strip() or None
    return None


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


if __name__ == "__main__":
    Ether0ResourcesServer.run_webserver()
