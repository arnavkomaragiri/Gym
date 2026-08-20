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
"""Harbor environment backed by the NeMo Gym sandbox API.

Makes every provider ``nemo_gym.sandbox`` supports a Harbor execution backend,
selected purely via config:

    harbor_environment_import_path: "responses_api_agents.harbor_agent.\
custom_envs.nemo_gym_sandbox.environment:NemoGymSandboxEnvironment"
    harbor_environment_kwargs:
      sandbox_provider:
        opensandbox:
          connection: {domain: ..., api_key: ..., use_server_proxy: true}

The task's ``docker_image`` must be a pullable image ref; Harbor-side image
builds are not supported. Logging directories are not mounted, so Harbor
downloads ``/logs`` at the end of the trial via :meth:`download_dir`.
"""

import json
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Mapping, Optional

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.definition import should_upload_environment_dir
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode
from harbor.models.trial.paths import EnvironmentPaths
from pydantic import BaseModel, ConfigDict, model_validator

from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
    rewrite_image,
)


# The sandbox-side scratch directory used for tar-based directory transfer.
_TRANSFER_DIR = "/tmp"
_SHARED_ARTIFACT_MARKER = ".nemo-gym-shared-artifact.json"
_SHARED_ARTIFACT_SCHEMA_VERSION = 1


class SandboxPathCopy(BaseModel):
    """Copy a sandbox-visible directory into the task workspace before setup."""

    model_config = ConfigDict(extra="forbid")

    source: str
    destination: str

    @model_validator(mode="after")
    def validate_paths(self) -> "SandboxPathCopy":
        for field_name, value in (("source", self.source), ("destination", self.destination)):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or path == PurePosixPath("/")
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(
                    f"sandbox_path_copies.{field_name} must be an absolute non-root path without "
                    f"'.' or '..' components (got {value!r})."
                )
        if PurePosixPath(self.source) == PurePosixPath(self.destination):
            raise ValueError("sandbox_path_copies source and destination must differ.")
        return self


class SharedArtifactTransferConfig(BaseModel, extra="forbid"):
    """Sandbox-local artifact relay through a filesystem shared by both roles."""

    root: str
    sources: list[str]
    timeout_s: float = 600
    stale_after_s: float = 86400

    @model_validator(mode="after")
    def validate_paths(self) -> "SharedArtifactTransferConfig":
        normalized: list[str] = []
        for field_name, values in (("root", [self.root]), ("sources", self.sources)):
            if not values:
                raise ValueError(f"shared_artifact_transfer.{field_name} must not be empty")
            for value in values:
                path = PurePosixPath(value)
                if (
                    not path.is_absolute()
                    or path == PurePosixPath("/")
                    or any(part in {"", ".", ".."} for part in path.parts)
                ):
                    raise ValueError(
                        f"shared_artifact_transfer.{field_name} entries must be absolute non-root paths "
                        f"without '.' or '..' components (got {value!r})"
                    )
                normalized.append(str(path))
        self.root = normalized[0]
        self.sources = normalized[1:]
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("shared_artifact_transfer.sources must not contain duplicates")
        root = PurePosixPath(self.root)
        if any(
            root == PurePosixPath(source)
            or root in PurePosixPath(source).parents
            or PurePosixPath(source) in root.parents
            for source in self.sources
        ):
            raise ValueError("shared_artifact_transfer.root and sources must not overlap")
        if self.timeout_s <= 0:
            raise ValueError("shared_artifact_transfer.timeout_s must be positive")
        if self.stale_after_s <= 0:
            raise ValueError("shared_artifact_transfer.stale_after_s must be positive")
        return self


@dataclass(frozen=True)
class SharedArtifactMarker:
    """Host-side capability for one sandbox-local shared-filesystem snapshot."""

    context_id: str
    source: str
    remote_path: str
    sha256: str
    source_environment_stopped: bool = False
    consumed: bool = False

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema_version": _SHARED_ARTIFACT_SCHEMA_VERSION,
                    "context_id": self.context_id,
                    "source": self.source,
                    "remote_path": self.remote_path,
                    "sha256": self.sha256,
                    "source_environment_stopped": self.source_environment_stopped,
                    "consumed": self.consumed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_path(cls, path: Path) -> "SharedArtifactMarker":
        raw = json.loads(path.read_text())
        expected = {
            "schema_version",
            "context_id",
            "source",
            "remote_path",
            "sha256",
            "source_environment_stopped",
            "consumed",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"Invalid shared artifact marker fields in {path}")
        if raw["schema_version"] != _SHARED_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shared artifact marker schema in {path}")
        for field_name in ("context_id", "source", "remote_path", "sha256"):
            if not isinstance(raw[field_name], str) or not raw[field_name]:
                raise ValueError(f"Invalid {field_name} in shared artifact marker {path}")
        if len(raw["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in raw["sha256"]):
            raise ValueError(f"Invalid sha256 in shared artifact marker {path}")
        for field_name in ("source_environment_stopped", "consumed"):
            if not isinstance(raw[field_name], bool):
                raise ValueError(f"Invalid {field_name} in shared artifact marker {path}")
        return cls(
            context_id=raw["context_id"],
            source=raw["source"],
            remote_path=raw["remote_path"],
            sha256=raw["sha256"],
            source_environment_stopped=raw["source_environment_stopped"],
            consumed=raw["consumed"],
        )


def _cpu_pin_prefix(width: int) -> str:
    """Shell prefix pinning the following command to a random contiguous
    ``width``-core block.

    Containers see the host's full core count (cgroup cpu limits don't shrink
    ``nproc``), so build tools fan out and get CFS throttled. A random block
    also spreads co-resident sandboxes instead of stacking them on cores 0..n.

    Must be POSIX sh, and fails open: no ``taskset``, ``nproc`` <= width, or
    unreadable urandom leaves the command unpinned.
    """
    return (
        f"__osb_w={width}; __osb_n=$(nproc 2>/dev/null || echo 0); "
        f'if [ "$__osb_n" -gt "$__osb_w" ] && command -v taskset >/dev/null 2>&1; then '
        f'__osb_s=$(( $(od -An -N2 -tu2 /dev/urandom | tr -d " ") % (__osb_n - __osb_w) )); '
        f'__osb_pin="taskset -c $__osb_s-$((__osb_s + __osb_w - 1))"; '
        f'else __osb_pin=""; fi; $__osb_pin'
    )


class NemoGymSandboxEnvironment(BaseEnvironment):
    """Harbor ``BaseEnvironment`` that runs the task in a NeMo Gym sandbox.

    Optional kwargs (via ``harbor_environment_kwargs``):
        sandbox_provider: Required. Single-key mapping ``{provider_name: kwargs}``
            (optionally wrapped with a reserved ``default_metadata`` key, same
            shape as the shipped ``sandbox:`` config blocks).
        sandbox_metadata: Extra ``SandboxSpec.metadata`` entries.
        sandbox_provider_options: ``SandboxSpec.provider_options`` passed through
            to the provider, e.g. ``resource_requests`` to schedule sandboxes
            below their resource limits. String values may contain
            ``{environment_name}``, ``{task_id}``, or ``{session_id}``; these
            are expanded per trial. This is useful for a volume ``sub_path``.
        environment_upload_excludes: Relative paths omitted when Harbor uploads
            the task's ``environment/`` directory. Use this only when the image
            or a sandbox volume supplies those paths, e.g. ``["data"]`` with a
            task-specific volume mounted at ``/app/data``.
        sandbox_path_copies: Ordered ``[{source: ..., destination: ...}]``
            directory copies performed inside the sandbox before Harbor uploads
            the task environment. Strings support the same per-trial placeholders
            as ``sandbox_provider_options``. This lets a read-only dataset mount
            seed a writable workspace without transferring the data through the
            OpenSandbox API.
        sandbox_path_copy_timeout_s: Timeout for each sandbox-local directory
            copy (default 1200).
        shared_artifact_transfer: Optional ``{root, sources, timeout_s}``
            configuration for relaying selected artifact directories through a
            sandbox-visible shared filesystem. The source environment snapshots
            into an opaque per-trial directory; a separate verifier can consume
            it only after source sandbox teardown succeeds.
        sandbox_env: Extra environment variables set in the sandbox.
        sandbox_ttl_s: Sandbox server-side TTL safety net (default 21600).
        sandbox_ready_timeout_s: Create/readiness timeout incl. image pull
            (default 900).
        default_exec_timeout_s: Per-command bound applied when Harbor calls
            ``exec`` without ``timeout_sec`` (default 1800), not a trial bound.

            Do not set this small. Harbor's verifier never passes
            ``timeout_sec``, so this is what bounds verification for every
            benchmark, and benchmarks routinely declare multi-minute budgets.
            A default below the task's own budget silently truncates
            verification and scores the task 0 rather than failing loudly.
        exec_shell: Shell prefix wrapped around every Harbor-issued command
            (default ``"bash -ic"``, matching Harbor's docker/daytona
            backends). Set to null to run commands verbatim.
        cpu_pin_enabled: Pin every Harbor-issued command to a random
            contiguous core block sized by the task's cpu count (default
            False). Terminus-2's tmux server is itself launched via ``exec``,
            so the whole agent session inherits the affinity. See
            ``_cpu_pin_prefix``.
        image_rewrites: Ordered ``[{from: ..., to: ...}]`` prefix rewrites
            applied to the task's ``docker_image`` (see
            ``nemo_gym.sandbox.rewrite_image``).
        image_override: Optional image replacing the task's ``docker_image``.
            Prefer a digest-pinned reference for reproducible runs.
        entrypoint: Optional command replacing the sandbox image's startup
            command. This is useful when Harbor starts task services later.
        workdir: Container working directory override (defaults to the image's
            own WORKDIR).
        allow_unenforced_internet_isolation: Accept tasks that request
            ``network_mode = "no-network"`` even though this environment cannot
            enforce network isolation (default False). Each affected trial
            logs a prominent warning.
    """

    def __init__(
        self,
        *args,
        sandbox_provider: Optional[Mapping[str, Any]] = None,
        sandbox_metadata: Optional[Mapping[str, Any]] = None,
        sandbox_provider_options: Optional[Mapping[str, Any]] = None,
        environment_upload_excludes: Optional[Sequence[str]] = None,
        sandbox_path_copies: Optional[Sequence[Mapping[str, str]]] = None,
        sandbox_path_copy_timeout_s: Optional[float] = 1200,
        shared_artifact_transfer: Optional[Mapping[str, Any] | SharedArtifactTransferConfig] = None,
        sandbox_env: Optional[Mapping[str, str]] = None,
        sandbox_ttl_s: Optional[float] = 21600,
        sandbox_ready_timeout_s: Optional[float] = 900,
        default_exec_timeout_s: Optional[float] = 1800,
        exec_shell: Optional[str] = "bash -ic",
        cpu_pin_enabled: bool = False,
        image_rewrites: Optional[list[Mapping[str, str]]] = None,
        image_override: Optional[str] = None,
        entrypoint: Optional[list[str]] = None,
        workdir: Optional[str] = None,
        allow_unenforced_internet_isolation: bool = False,
        **kwargs,
    ):
        # Set before super().__init__: the base constructor runs the
        # _validate_* hooks, which read these.
        self._sandbox_provider = sandbox_provider
        self._sandbox_metadata = dict(sandbox_metadata or {})
        self._sandbox_provider_options = dict(sandbox_provider_options or {})
        self._environment_upload_excludes = self._validate_upload_excludes(environment_upload_excludes or ())
        self._sandbox_path_copies = tuple(SandboxPathCopy.model_validate(item) for item in (sandbox_path_copies or ()))
        self._sandbox_path_copy_timeout_s = sandbox_path_copy_timeout_s
        self._shared_artifact_transfer = (
            shared_artifact_transfer
            if isinstance(shared_artifact_transfer, SharedArtifactTransferConfig)
            else SharedArtifactTransferConfig.model_validate(shared_artifact_transfer)
            if shared_artifact_transfer is not None
            else None
        )
        self._shared_transfer_markers: set[Path] = set()
        self._shared_transfer_cleanup_paths: set[PurePosixPath] = set()
        self._sandbox_env = {str(k): str(v) for k, v in dict(sandbox_env or {}).items()}
        self._sandbox_ttl_s = sandbox_ttl_s
        self._sandbox_ready_timeout_s = sandbox_ready_timeout_s
        self._default_exec_timeout_s = default_exec_timeout_s
        self._exec_shell = exec_shell
        self._cpu_pin_enabled = cpu_pin_enabled
        self._image_rewrites = [dict(rewrite) for rewrite in (image_rewrites or [])]
        self._image_override = image_override
        self._entrypoint = list(entrypoint) if entrypoint is not None else None
        self._workdir = workdir
        self._allow_unenforced_internet_isolation = allow_unenforced_internet_isolation
        self._sandbox: Optional[AsyncSandbox] = None

        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> EnvironmentType:
        # Only used in validation error messages, and the pinned harbor release
        # has no member for external environments, so DOCKER is display-only.
        return getattr(EnvironmentType, "NEMO_GYM_SANDBOX", EnvironmentType.DOCKER)

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return self._allow_unenforced_internet_isolation

    def _validate_definition(self):
        if not self._sandbox_provider:
            raise ValueError(
                "NemoGymSandboxEnvironment requires harbor_environment_kwargs.sandbox_provider "
                "({provider_name: kwargs})."
            )
        # Fails fast on malformed provider blocks (e.g. multiple provider keys).
        resolve_provider_config(self._sandbox_provider)
        if not self.task_env_config.docker_image:
            raise ValueError(
                f"Task {self.environment_name!r} does not define environment.docker_image; "
                "NemoGymSandboxEnvironment cannot build images from a Dockerfile."
            )

    @property
    def _resolved_image(self) -> str:
        if self._image_override is not None:
            return self._image_override
        return rewrite_image(self.task_env_config.docker_image, self._image_rewrites)

    @staticmethod
    def _validate_upload_excludes(excludes: Sequence[str]) -> tuple[PurePosixPath, ...]:
        normalized: list[PurePosixPath] = []
        for value in excludes:
            path = PurePosixPath(str(value))
            if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
                raise ValueError(
                    "environment_upload_excludes entries must be non-empty relative paths "
                    f"without '.' or '..' components (got {value!r})."
                )
            normalized.append(path)
        return tuple(normalized)

    def _render_provider_option_templates(self, value: Any) -> Any:
        if isinstance(value, str):
            task_id = self.environment_name.rsplit("__", 1)[-1]
            replacements = {
                "{environment_name}": self.environment_name,
                "{task_id}": task_id,
                "{session_id}": self.session_id,
            }
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            return value
        if isinstance(value, Mapping):
            return {key: self._render_provider_option_templates(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._render_provider_option_templates(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._render_provider_option_templates(item) for item in value)
        return value

    def _is_environment_upload(self, source: Path) -> bool:
        return source.resolve() == Path(self.environment_dir).resolve()

    @property
    def _is_separate_verifier(self) -> bool:
        # Harbor assigns separate verifier environments a stable
        # ``<trial>__verifier__<step>`` session id.
        return "__verifier__" in self.session_id

    def _is_upload_excluded(self, relative: PurePosixPath) -> bool:
        return any(
            relative == excluded or excluded in relative.parents for excluded in self._environment_upload_excludes
        )

    def _build_spec(self) -> SandboxSpec:
        config = self.task_env_config
        resources: dict[str, Any] = {}
        if config.cpus:
            resources["cpu"] = float(config.cpus)
        if config.memory_mb:
            resources["memory_mib"] = int(config.memory_mb)
        if config.storage_mb:
            resources["disk_gib"] = max(1, round(config.storage_mb / 1024))
        if config.gpus:
            resources["gpu"] = int(config.gpus)

        metadata = {
            "harbor-session": self.session_id,
            "harbor-task": self.environment_name,
            "harbor-role": "verifier" if self._is_separate_verifier else "agent",
            **resolve_provider_metadata(self._sandbox_provider),
            **self._sandbox_metadata,
        }

        return SandboxSpec(
            image=self._resolved_image,
            ttl_s=self._sandbox_ttl_s,
            ready_timeout_s=self._sandbox_ready_timeout_s,
            workdir=self._workdir if self._workdir is not None else self.task_env_config.workdir,
            env={**self._startup_env(), **self._sandbox_env},
            metadata=metadata,
            resources=resources,
            entrypoint=self._entrypoint,
            provider_options=self._render_provider_option_templates(self._sandbox_provider_options),
        )

    async def _upload_environment_dir_after_start(self) -> None:
        if not self._is_separate_verifier:
            await super()._upload_environment_dir_after_start()
            return

        # In Harbor separate-verifier mode ``environment_dir`` is the hidden
        # tests directory, not the task's policy environment directory. Local
        # Harbor backends build this context as the verifier image; prebuilt
        # OpenSandbox images need the equivalent files uploaded to /tests.
        if not should_upload_environment_dir(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
        ):
            return
        self.logger.debug("Uploading separate verifier context to /tests")
        await self.upload_dir(self.environment_dir, str(EnvironmentPaths.tests_dir))

    async def _copy_sandbox_paths(self) -> None:
        sandbox = self._require_sandbox()
        for configured_copy in self._sandbox_path_copies:
            rendered = SandboxPathCopy.model_validate(
                self._render_provider_option_templates(configured_copy.model_dump())
            )
            source = shlex.quote(rendered.source)
            destination = shlex.quote(rendered.destination)
            result = await sandbox.exec(
                f"test -d {source} && mkdir -p {destination} && cp -a -- {source}/. {destination}/",
                cwd="/",
                timeout_s=self._sandbox_path_copy_timeout_s,
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(
                    f"Failed to copy sandbox directory {rendered.source!r} to {rendered.destination!r}: {output}"
                )

    def _uses_shared_artifact_transfer(self, source_dir: str) -> bool:
        config = self._shared_artifact_transfer
        return config is not None and str(PurePosixPath(source_dir)) in config.sources

    @staticmethod
    def _write_shared_artifact_marker(path: Path, marker: SharedArtifactMarker) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(marker.to_json())
        temporary.replace(path)

    def _validate_shared_remote_path(self, marker: SharedArtifactMarker) -> PurePosixPath:
        config = self._shared_artifact_transfer
        if config is None:
            raise RuntimeError("Shared artifact marker found but shared_artifact_transfer is disabled")
        if marker.context_id != str(self.context_id):
            raise RuntimeError("Shared artifact marker belongs to a different Harbor trial")
        if marker.source not in config.sources:
            raise RuntimeError("Shared artifact marker source is not enabled in this environment")
        expected_parent = PurePosixPath(config.root) / "v1" / marker.context_id
        remote_path = PurePosixPath(marker.remote_path)
        if remote_path.parent != expected_parent:
            raise RuntimeError("Shared artifact marker path is outside its opaque per-trial namespace")
        return remote_path

    async def _snapshot_dir_to_shared_transfer(
        self,
        *,
        source_dir: str,
        target_dir: Path,
        exclude: Sequence[str],
    ) -> None:
        config = self._shared_artifact_transfer
        if config is None:
            raise RuntimeError("Shared artifact transfer is not configured")
        target_dir.mkdir(parents=True, exist_ok=True)
        if any(target_dir.iterdir()):
            raise RuntimeError(f"Shared artifact marker directory is not empty: {target_dir}")

        context_id = str(self.context_id)
        transfer_id = uuid.uuid4().hex
        root = PurePosixPath(config.root)
        version_dir = root / "v1"
        context_dir = version_dir / context_id
        remote_path = context_dir / transfer_id
        staging_path = context_dir / f".staging-{transfer_id}"
        payload_path = staging_path / "payload"
        archive_path = staging_path / "workspace.tar"
        exclude_flags = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in exclude)
        stale_after_minutes = max(1, int(config.stale_after_s // 60))
        script = (
            "set -euo pipefail; umask 077; "
            f"test -d {shlex.quote(source_dir)}; "
            f"test ! -L {shlex.quote(str(root))}; mkdir -p {shlex.quote(str(root))}; "
            f"test ! -L {shlex.quote(str(version_dir))}; mkdir -p {shlex.quote(str(version_dir))}; "
            f"find {shlex.quote(str(version_dir))} -mindepth 1 -maxdepth 1 -type d "
            f"-mmin +{stale_after_minutes} -exec rm -rf -- {{}} +; "
            f"test ! -L {shlex.quote(str(context_dir))}; mkdir -p {shlex.quote(str(context_dir))}; "
            f"test ! -e {shlex.quote(str(remote_path))}; rm -rf {shlex.quote(str(staging_path))}; "
            f"trap 'rm -rf {shlex.quote(str(staging_path))}' EXIT; "
            f"mkdir -p {shlex.quote(str(payload_path))}; "
            f"tar {exclude_flags} -C {shlex.quote(source_dir)} -cf - . "
            f"| tar -C {shlex.quote(str(payload_path))} -xf -; "
            f"bad=$(find {shlex.quote(str(payload_path))} ! -type d ! -type f -print -quit); "
            'if [ -n "$bad" ]; then printf \'unsupported artifact entry: %s\\n\' "$bad"; exit 73; fi; '
            f"tar -C {shlex.quote(str(payload_path))} -cf {shlex.quote(str(archive_path))} .; "
            f"rm -rf {shlex.quote(str(payload_path))}; "
            f": > {shlex.quote(str(staging_path / 'ready'))}; "
            f"mv {shlex.quote(str(staging_path))} {shlex.quote(str(remote_path))}; trap - EXIT; "
            f"sha256sum {shlex.quote(str(remote_path / 'workspace.tar'))} | awk '{{print $1}}'"
        )
        result = await self._require_sandbox().exec(
            f"bash -o pipefail -c {shlex.quote(script)}",
            cwd="/",
            timeout_s=config.timeout_s,
            user="root",
        )
        if result.return_code != 0:
            output = (
                "\n".join(
                    output
                    for output in (
                        (result.stdout or "").strip(),
                        (result.stderr or "").strip(),
                    )
                    if output
                )
                or "<no output>"
            )
            raise RuntimeError(f"Failed to snapshot {source_dir!r} into shared artifact storage: {output}")
        digest = (result.stdout or "").strip().splitlines()[-1:]
        if not digest or len(digest[0]) != 64 or any(character not in "0123456789abcdef" for character in digest[0]):
            await self._require_sandbox().exec(
                f"rm -rf {shlex.quote(str(remote_path))}",
                cwd="/",
                timeout_s=60,
                user="root",
            )
            raise RuntimeError("Shared artifact snapshot did not return a valid SHA-256 digest")

        marker_path = target_dir / _SHARED_ARTIFACT_MARKER
        marker = SharedArtifactMarker(
            context_id=context_id,
            source=str(PurePosixPath(source_dir)),
            remote_path=str(remote_path),
            sha256=digest[0],
        )
        try:
            self._write_shared_artifact_marker(marker_path, marker)
        except OSError:
            await self._require_sandbox().exec(
                f"rm -rf {shlex.quote(str(remote_path))}",
                cwd="/",
                timeout_s=60,
                user="root",
            )
            raise
        self._shared_transfer_markers.add(marker_path)

    async def _upload_shared_artifact(self, source: Path, target_dir: str) -> bool:
        marker_path = source / _SHARED_ARTIFACT_MARKER
        if not marker_path.is_file():
            return False
        if {path.name for path in source.iterdir()} != {_SHARED_ARTIFACT_MARKER}:
            raise RuntimeError(f"Shared artifact marker directory contains unexpected entries: {source}")

        marker = SharedArtifactMarker.from_path(marker_path)
        remote_path = self._validate_shared_remote_path(marker)
        if str(PurePosixPath(target_dir)) != marker.source:
            raise RuntimeError("Shared artifact handoff target does not match its original source")
        if not marker.source_environment_stopped:
            raise RuntimeError("Refusing shared artifact handoff before source sandbox teardown is confirmed")
        if not self._is_separate_verifier:
            raise RuntimeError("Shared artifact handoff may only be consumed by a separate verifier environment")
        if marker.consumed:
            raise RuntimeError("Shared artifact handoff has already been consumed")

        config = self._shared_artifact_transfer
        if config is None:
            raise RuntimeError("Shared artifact transfer is not configured")
        archive = remote_path / "workspace.tar"
        ready = remote_path / "ready"
        restore_path = PurePosixPath(_TRANSFER_DIR) / f".nemo-gym-shared-restore-{uuid.uuid4().hex}"
        # The verifier owns the relay after accepting its host-side capability.
        # Keep it until stop() so failed judging/setup paths cannot leak EFS state.
        self._shared_transfer_cleanup_paths.add(remote_path)
        script = (
            "set -euo pipefail; "
            f"test -f {shlex.quote(str(ready))}; test -f {shlex.quote(str(archive))}; "
            f"actual=$(sha256sum {shlex.quote(str(archive))} | awk '{{print $1}}'); "
            f'test "$actual" = {shlex.quote(marker.sha256)}; '
            f"rm -rf {shlex.quote(str(restore_path))}; trap 'rm -rf {shlex.quote(str(restore_path))}' EXIT; "
            f"mkdir -p {shlex.quote(str(restore_path))}; tar -C {shlex.quote(str(restore_path))} "
            f"-xf {shlex.quote(str(archive))}; "
            f"bad=$(find {shlex.quote(str(restore_path))} ! -type d ! -type f -print -quit); "
            'if [ -n "$bad" ]; then printf \'unsupported artifact entry: %s\\n\' "$bad"; exit 73; fi; '
            f"mkdir -p {shlex.quote(target_dir)}; "
            f"cp -a -- {shlex.quote(str(restore_path))}/. {shlex.quote(target_dir)}/; "
            f"rm -rf {shlex.quote(str(restore_path))}; trap - EXIT"
        )
        result = await self._require_sandbox().exec(
            f"bash -o pipefail -c {shlex.quote(script)}",
            cwd="/",
            timeout_s=config.timeout_s,
            user="root",
        )
        if result.return_code != 0:
            output = (
                "\n".join(
                    output
                    for output in (
                        (result.stdout or "").strip(),
                        (result.stderr or "").strip(),
                    )
                    if output
                )
                or "<no output>"
            )
            raise RuntimeError(f"Failed to restore shared artifacts into {target_dir!r}: {output}")
        self._write_shared_artifact_marker(marker_path, replace(marker, consumed=True))
        return True

    async def _cleanup_shared_artifact_transfers(self) -> None:
        """Delete verifier-owned relay paths before its EFS volume is detached."""
        if not self._shared_transfer_cleanup_paths:
            return
        config = self._shared_artifact_transfer
        if config is None:
            raise RuntimeError("Shared artifact cleanup is pending but transfer is disabled")

        commands: list[str] = ["set -euo pipefail"]
        for remote_path in sorted(self._shared_transfer_cleanup_paths, key=str):
            context_dir = remote_path.parent
            commands.extend(
                [
                    f"rm -rf -- {shlex.quote(str(remote_path))}",
                    f"rmdir {shlex.quote(str(context_dir))} 2>/dev/null || true",
                ]
            )
        result = await self._require_sandbox().exec(
            f"bash -o pipefail -c {shlex.quote('; '.join(commands))}",
            cwd="/",
            timeout_s=config.timeout_s,
            user="root",
        )
        if result.return_code != 0:
            output = (
                "\n".join(
                    output
                    for output in (
                        (result.stdout or "").strip(),
                        (result.stderr or "").strip(),
                    )
                    if output
                )
                or "<no output>"
            )
            raise RuntimeError(f"Failed to clean shared artifact relay during verifier teardown: {output}")
        self._shared_transfer_cleanup_paths.clear()

    def _confirm_shared_artifact_source_stopped(self) -> None:
        for marker_path in self._shared_transfer_markers:
            marker = SharedArtifactMarker.from_path(marker_path)
            self._write_shared_artifact_marker(
                marker_path,
                replace(marker, source_environment_stopped=True),
            )

    def _sandbox_volume_mount_paths(self) -> tuple[PurePosixPath, ...]:
        rendered = self._render_provider_option_templates(self._sandbox_provider_options)
        volumes = rendered.get("volumes", []) if isinstance(rendered, Mapping) else []
        mount_paths: list[PurePosixPath] = []
        for volume in volumes:
            if not isinstance(volume, Mapping):
                continue
            value = volume.get("mountPath", volume.get("mount_path"))
            if isinstance(value, str) and PurePosixPath(value).is_absolute():
                mount_paths.append(PurePosixPath(value))
        return tuple(mount_paths)

    def _empty_dirs_preserving_mounts_command(
        self,
        dirs: Sequence[str | PurePath],
        *,
        chmod: bool,
    ) -> str:
        mounts = self._sandbox_volume_mount_paths()
        commands: list[str] = []
        for value in dirs:
            path = PurePosixPath(str(value))
            quoted = shlex.quote(str(path))
            if path in mounts:
                commands.append("true")
                continue
            preserved_children = {path / mount.relative_to(path).parts[0] for mount in mounts if path in mount.parents}
            commands.extend(
                [
                    f"if [ -L {quoted} ] || {{ [ -e {quoted} ] && [ ! -d {quoted} ]; }}; then rm -rf {quoted}; fi",
                    f"mkdir -p {quoted}",
                ]
            )
            exclusions = " ".join(f"! -path {shlex.quote(str(child))}" for child in sorted(preserved_children))
            commands.append(f"find {quoted} -mindepth 1 -maxdepth 1 {exclusions} -exec rm -rf -- {{}} +")
            if chmod:
                commands.append(f"chmod 777 {quoted}")
        return " && ".join(commands)

    async def start(self, force_build: bool) -> None:
        if force_build:
            self.logger.warning(
                "force_build is not supported by NemoGymSandboxEnvironment; using the task's prebuilt image %r.",
                self._resolved_image,
            )
        if self._network_policy.network_mode == NetworkMode.NO_NETWORK:
            self.logger.warning(
                "Task %r requests network_mode='no-network' but NemoGymSandboxEnvironment does "
                "not enforce network isolation; the sandbox keeps cluster-default egress.",
                self.environment_name,
            )

        sandbox = AsyncSandbox(
            resolve_provider_config(self._sandbox_provider),
            self._build_spec(),
        )
        await sandbox.start()
        self._sandbox = sandbox

        # Harbor's local backends bind mount these convention directories. The
        # remote provider must create all of them before agents start writing.
        log_dirs = f"{EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} {EnvironmentPaths.artifacts_dir}"
        result = await self._sandbox.exec(f"mkdir -p {log_dirs}", cwd="/", timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to create log directories in sandbox: {result.stderr or result.stdout or '<no output>'}"
            )
        await self._copy_sandbox_paths()
        await self._upload_environment_dir_after_start()

    def _emptying_removes_sandbox_copy(self, dirs: Sequence[str | PurePath]) -> bool:
        emptied = [PurePosixPath(str(path)) for path in dirs]
        for configured_copy in self._sandbox_path_copies:
            rendered = SandboxPathCopy.model_validate(
                self._render_provider_option_templates(configured_copy.model_dump())
            )
            destination = PurePosixPath(rendered.destination)
            if any(path == destination or path in destination.parents for path in emptied):
                return True
        return False

    async def ensure_dirs(
        self,
        dirs: Sequence[str | PurePath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        """Create infrastructure directories without relying on the task workdir."""
        if not dirs:
            return None
        return await self.exec(
            self._ensure_dirs_command(dirs, chmod=chmod),
            cwd="/",
            user=self._reset_dirs_user() if chmod else None,
        )

    async def empty_dirs(
        self,
        dirs: Sequence[str | PurePath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        """Empty directories, then restore verifier-only sandbox-local baselines."""
        if not dirs:
            return None
        result = await self.exec(
            self._empty_dirs_preserving_mounts_command(dirs, chmod=chmod),
            cwd="/",
            user=self._reset_dirs_user(),
        )
        if result.return_code == 0 and self._is_separate_verifier and self._emptying_removes_sandbox_copy(dirs):
            await self._copy_sandbox_paths()
        return result

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            self._path_kind_check_command(path, require_dir=True),
            cwd="/",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            self._path_kind_check_command(path, require_dir=False),
            cwd="/",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox is not running; call start() first.")
        return self._sandbox

    async def stop(self, delete: bool):
        if self._sandbox is None:
            return
        if not delete:
            # Remote sandboxes are single-use, so honoring delete=False would
            # leak cluster resources.
            self.logger.debug(
                "delete=False is ignored by NemoGymSandboxEnvironment; the sandbox is always terminated on stop()."
            )
        sandbox = self._sandbox
        cleanup_error: Exception | None = None
        if self._is_separate_verifier:
            try:
                await self._cleanup_shared_artifact_transfers()
            except Exception as error:  # noqa: BLE001 - sandbox termination must still run
                cleanup_error = error
                self.logger.exception("Failed to clean verifier shared artifact relay before sandbox termination")
        stop_error: Exception | None = None
        try:
            await sandbox.stop()
        except Exception as error:  # noqa: BLE001 - preserve cleanup and stop failures
            stop_error = error
        finally:
            self._sandbox = None
        if stop_error is None:
            self._confirm_shared_artifact_source_stopped()
        if cleanup_error is not None:
            if stop_error is not None:
                raise RuntimeError(
                    "Failed to clean shared artifact relay and terminate verifier sandbox: "
                    f"cleanup_error={cleanup_error!r}, stop_error={stop_error!r}"
                ) from cleanup_error
            raise cleanup_error
        if stop_error is not None:
            raise stop_error

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        timeout_s = timeout_sec if timeout_sec is not None else self._default_exec_timeout_s
        # Harbor's docker/daytona backends use an interactive bash, so
        # .bashrc-based task setups (conda, pyenv, PATH) must behave the same.
        if self._exec_shell:
            command = f"{self._exec_shell} {shlex.quote(command)}"
        if self._cpu_pin_enabled:
            width = int(self.task_env_config.cpus or 0)
            if width > 0:
                command = f"{_cpu_pin_prefix(width)} {command}"
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd,
            env=self._merge_env(env),
            timeout_s=timeout_s,
            user=self._resolve_user(user),
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str):
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        sandbox = self._require_sandbox()
        parent = str(PurePosixPath(target_path).parent)
        if parent and parent != ".":
            await sandbox.exec(f"mkdir -p {shlex.quote(parent)}", cwd="/", timeout_s=60)
        await sandbox.upload(source, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")
        if await self._upload_shared_artifact(source, target_dir):
            return
        sandbox = self._require_sandbox()
        excludes = self._environment_upload_excludes if self._is_environment_upload(source) else ()

        remote_tar = f"{_TRANSFER_DIR}/.nemo-gym-upload-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "upload.tar.gz"
            with tarfile.open(local_tar, "w:gz") as tar:
                # Archive the *contents* of source_dir so they land directly in
                # target_dir (Harbor's upload_dir contract).
                def _filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
                    relative = PurePosixPath(member.name).relative_to(".")
                    if excludes and self._is_upload_excluded(relative):
                        return None
                    return member

                tar.add(source, arcname=".", filter=_filter)
            await sandbox.upload(local_tar, remote_tar)

        quoted_target = shlex.quote(target_dir)
        quoted_tar = shlex.quote(remote_tar)
        result = await sandbox.exec(
            f"mkdir -p {quoted_target} && tar -xzf {quoted_tar} -C {quoted_target}; "
            f"status=$?; rm -f {quoted_tar}; exit $status",
            cwd="/",
            timeout_s=600,
        )
        if result.return_code != 0:
            self.logger.warning(
                "tar-based upload_dir failed (rc=%s, stderr=%r); falling back to per-file upload.",
                result.return_code,
                (result.stderr or "")[:500],
            )
            await self._upload_dir_file_by_file(source, target_dir, excludes=excludes)

    async def _upload_dir_file_by_file(
        self,
        source: Path,
        target_dir: str,
        *,
        excludes: tuple[PurePosixPath, ...] = (),
    ):
        sandbox = self._require_sandbox()
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            relative_posix = PurePosixPath(*relative.parts)
            if excludes and self._is_upload_excluded(relative_posix):
                continue
            remote_path = str(PurePosixPath(target_dir) / relative_posix)
            if path.is_dir():
                await sandbox.exec(f"mkdir -p {shlex.quote(remote_path)}", cwd="/", timeout_s=60)
            elif path.is_file():
                await self.upload_file(path, remote_path)

    async def download_file(self, source_path: str, target_path: Path | str):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._require_sandbox().download(source_path, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        if self._uses_shared_artifact_transfer(source_dir):
            await self._snapshot_dir_to_shared_transfer(
                source_dir=source_dir,
                target_dir=Path(target_dir),
                exclude=(),
            )
            return
        await self._download_dir_archive(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=(),
            snapshot=True,
        )

    async def download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        if self._uses_shared_artifact_transfer(source_dir):
            await self._snapshot_dir_to_shared_transfer(
                source_dir=source_dir,
                target_dir=Path(target_dir),
                exclude=exclude,
            )
            return
        await self._download_dir_archive(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=exclude,
            snapshot=False,
        )

    async def _download_dir_archive(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: Sequence[str],
        snapshot: bool,
    ) -> None:
        sandbox = self._require_sandbox()
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        transfer_id = uuid.uuid4().hex
        remote_tar = f"{_TRANSFER_DIR}/.nemo-gym-download-{transfer_id}.tar.gz"
        remote_snapshot = f"{_TRANSFER_DIR}/.nemo-gym-download-{transfer_id}"
        quoted_source = shlex.quote(source_dir)
        quoted_tar = shlex.quote(remote_tar)
        quoted_snapshot = shlex.quote(remote_snapshot)
        exclude_flags = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in exclude)
        if snapshot:
            archive_command = (
                f"rm -rf {quoted_snapshot}; mkdir -p {quoted_snapshot} "
                f"&& cp -a -- {quoted_source}/. {quoted_snapshot}/ "
                f"&& tar -czf {quoted_tar} -C {quoted_snapshot} .; "
                f"status=$?; rm -rf {quoted_snapshot}; exit $status"
            )
        else:
            archive_command = f"tar -czf {quoted_tar} {exclude_flags} -C {quoted_source} ."
        result = await sandbox.exec(
            archive_command,
            cwd="/",
            timeout_s=600,
        )
        if result.return_code != 0:
            await sandbox.exec(f"rm -f {quoted_tar}", cwd="/", timeout_s=60)
            self.logger.warning(
                "tar-based download_dir failed (rc=%s, stderr=%r); falling back to per-file download.",
                result.return_code,
                (result.stderr or "")[:500],
            )
            await self._download_dir_file_by_file(source_dir, target, exclude=exclude)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "download.tar.gz"
            try:
                await sandbox.download(remote_tar, local_tar)
            finally:
                await sandbox.exec(f"rm -f {quoted_tar}", cwd="/", timeout_s=60)
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(target, filter="data")

    @staticmethod
    def _is_download_excluded(relative: PurePosixPath, exclude: Sequence[str]) -> bool:
        candidates = [relative.as_posix(), *(parent.as_posix() for parent in relative.parents if parent.parts)]
        return any(fnmatch(candidate, pattern) for candidate in candidates for pattern in exclude)

    async def _download_dir_file_by_file(
        self,
        source_dir: str,
        target: Path,
        *,
        exclude: Sequence[str] = (),
    ) -> None:
        sandbox = self._require_sandbox()
        listing = await sandbox.exec(
            f"find {shlex.quote(source_dir)} -type f",
            cwd="/",
            timeout_s=120,
        )
        if listing.return_code != 0:
            raise RuntimeError(
                f"Failed to list sandbox directory {source_dir!r}: {listing.stderr or listing.stdout or '<no output>'}"
            )
        source_root = PurePosixPath(source_dir)
        for line in (listing.stdout or "").splitlines():
            remote_path = line.strip()
            if not remote_path:
                continue
            relative = PurePosixPath(remote_path).relative_to(source_root)
            if self._is_download_excluded(relative, exclude):
                continue
            await self.download_file(remote_path, target / Path(*relative.parts))
