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

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app import (
    EVAL_FUNCTIONS,
    Ether0RemotesConfig,
    Ether0ResourcesServer,
    Ether0ResourcesServerConfig,
    Ether0VerifyRequest,
)
from fastapi.testclient import TestClient

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_server(config: Ether0ResourcesServerConfig | None = None) -> Ether0ResourcesServer:
    config = config or Ether0ResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return Ether0ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_request(
    text: str,
    solution: str,
    problem_type: str,
    choices: dict | None = None,
) -> Ether0VerifyRequest:
    meta: dict = {"solution": solution, "problem_type": problem_type}
    if choices is not None:
        meta["choices"] = choices
    return Ether0VerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q"}]},
        response=_make_response(text),
        verifier_metadata=meta,
    )


class TestVerify:
    def test_sanity(self) -> None:
        _make_server()

    async def test_str_eval_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>FCC(=O)O</answer>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_str_eval_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>CCCCCC</answer>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_ether0_special_tokens(self) -> None:
        server = _make_server()
        req = _make_request(
            "<|think_start|>reasoning here<|think_end|><|answer_start|>FCC(=O)O<|answer_end|>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_no_answer_tag(self) -> None:
        server = _make_server()
        req = _make_request(
            "I have no idea",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.extracted_answer is None

    async def test_malformed_solution(self) -> None:
        server = _make_server()
        req = _make_request("<answer>CCO</answer>", "bad_format", "")
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_boxed_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "Reasoning here.\n\n\\boxed{FCC(=O)O}",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "FCC(=O)O"

    async def test_boxed_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "Reasoning here.\n\n\\boxed{CCCCCC}",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    _MCQ_SOLUTION = "str_eval!:!C1=CC(NN)=CC=C1C!:!property-regression-pka/pKaH1"
    _MCQ_CHOICES = {
        "A": "NNC1=CC=C(C=C1)Cl",
        "B": "ClC1C=CC(=CC=1)SC1C=C(N=C(N)N=1)N",
        "C": "C1=CC(NN)=CC=C1C",
    }

    async def test_letter_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think...\nAnswer: C",
            self._MCQ_SOLUTION,
            "property-regression-pka/pKaH1",
            choices=self._MCQ_CHOICES,
        )
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "C1=CC(NN)=CC=C1C"

    async def test_letter_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think...\nAnswer: A",
            self._MCQ_SOLUTION,
            "property-regression-pka/pKaH1",
            choices=self._MCQ_CHOICES,
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.extracted_answer == "NNC1=CC=C(C=C1)Cl"

    async def test_letter_lowercase(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think...\nAnswer: c",
            self._MCQ_SOLUTION,
            "property-regression-pka/pKaH1",
            choices=self._MCQ_CHOICES,
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_sync_verifier_runs_off_event_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event_loop_thread = threading.get_ident()

        def thread_eval(answer: str, answer_info: str) -> float:
            assert threading.get_ident() != event_loop_thread
            return float(answer == answer_info)

        monkeypatch.setitem(EVAL_FUNCTIONS, "thread_eval", thread_eval)
        server = _make_server()
        request = _make_request(
            "<answer>CCO</answer>",
            "thread_eval!:!CCO!:!test",
            "test",
        )
        result = await server.verify(request)
        assert result.reward == 1.0

    async def test_remote_eval_requires_sidecar(self) -> None:
        server = _make_server()
        request = _make_request(
            "<answer>CCO</answer>",
            "sol_eval!:!('scaffold', 'CC', 0.0, 'increase')!:!oracle-solubility",
            "oracle-solubility",
        )
        with pytest.raises(JudgeError, match="requires the remotes sidecar"):
            await server.verify(request)


def _remotes_config(tmp_path: Path) -> Ether0RemotesConfig:
    python_executable = tmp_path / "python"
    model_path = tmp_path / "model.pt"
    runtime_home = tmp_path / "runtime-home"
    python_executable.touch()
    model_path.touch()
    (runtime_home / ".cache" / "molbloom").mkdir(parents=True)
    (runtime_home / ".cache" / "molbloom" / "zinc20.bloom").touch()
    return Ether0RemotesConfig(
        python_executable=python_executable,
        model_path=model_path,
        runtime_home=runtime_home,
    )


def test_remotes_reject_multiple_fastapi_workers(tmp_path: Path) -> None:
    config = Ether0ResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        num_workers=2,
        remotes=_remotes_config(tmp_path),
    )
    with pytest.raises(ValueError, match="num_workers=1"):
        _make_server(config)


def test_remotes_follow_fastapi_lifespan(tmp_path: Path) -> None:
    config = Ether0ResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        num_workers=1,
        remotes=_remotes_config(tmp_path),
    )
    server = _make_server(config)
    start = AsyncMock()
    stop = AsyncMock()
    with (
        patch.object(Ether0ResourcesServer, "_start_remotes", start),
        patch.object(Ether0ResourcesServer, "_stop_remotes", stop),
        TestClient(server.setup_webserver()) as client,
    ):
        assert client.get("/").status_code == 404
    start.assert_awaited_once()
    stop.assert_awaited_once()


def test_remotes_are_stopped_when_startup_fails(tmp_path: Path) -> None:
    config = Ether0ResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        num_workers=1,
        remotes=_remotes_config(tmp_path),
    )
    server = _make_server(config)
    start = AsyncMock(side_effect=RuntimeError("startup failed"))
    stop = AsyncMock()
    with (
        patch.object(Ether0ResourcesServer, "_start_remotes", start),
        patch.object(Ether0ResourcesServer, "_stop_remotes", stop),
        pytest.raises(RuntimeError, match="startup failed"),
        TestClient(server.setup_webserver()),
    ):
        pass
    start.assert_awaited_once()
    stop.assert_awaited_once()
