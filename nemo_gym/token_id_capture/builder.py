# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Turn a rollout's captured token records into trainable trajectories.

The builder is a pure function over a list of ``TokenEntry`` records (whatever a
``TokenSource`` returns). It has two strategies:

  per_request     assumes nothing about how the calls relate; every call becomes
                  its own training sequence. Always valid.
  prefix_merging  chains calls by their token-prefix relationship. It also
                  handles text APIs that re-tokenize the end of a sampled turn:
                  the token-identical part remains generated and the changed
                  suffix becomes masked context. A prompt that no longer extends
                  most of an earlier call starts a new root (a compacted or
                  rewritten context). Equally good candidate parents are
                  ambiguous, so that subtree is quarantined rather than guessed.

Both are order-independent: they do not depend on arrival order or any sequence
number. ``prefix_merging`` processes entries by increasing prompt length, which
is derived from the tokens themselves (a parent's prompt is shorter than its
child's), so concurrent or out-of-order capture yields the same result.

Loss masks follow provenance: tokens the policy generated are marked 1 (with
their captured log probabilities), and everything re-fed into a prompt (history,
tool output, tokens added between calls) is marked 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from nemo_gym.token_id_capture.records import TokenEntry


@dataclass
class ChainLink:
    entry: TokenEntry
    interstitial: list[int]  # prompt tokens added since the parent (tool output, new user turn); mask 0
    # A later prompt may re-tokenize the end of this generation even when it
    # preserves the decoded text. Keep only the token-identical prefix so the
    # later call can start from its exact sampled prompt.
    generation_length: int | None = None

    def kept_generation_length(self) -> int:
        return len(self.entry.generation_token_ids) if self.generation_length is None else self.generation_length


@dataclass
class Chain:
    chain_id: str
    links: list[ChainLink] = field(default_factory=list)
    root_prompt: list[int] = field(default_factory=list)

    def validate(self) -> None:
        """Every generated token needs its log probability; a trainer cannot use a chain
        where they disagree, and the mismatch is easier to read here than downstream."""
        for link in self.links:
            generated = link.entry.generation_token_ids
            log_probs = link.entry.generation_log_probs
            if len(log_probs) != len(generated):
                raise ValueError(
                    f"log-prob/token length mismatch on {link.entry.model_call_id}: "
                    f"{len(log_probs)} vs {len(generated)}"
                )
            if not 0 <= link.kept_generation_length() <= len(generated):
                raise ValueError(
                    f"invalid generation prefix on {link.entry.model_call_id}: "
                    f"{link.kept_generation_length()} of {len(generated)} tokens"
                )


@dataclass
class BuildNotes:
    """What the build kept, dropped and had to guess at.

    Typed rather than a free-form dict because the consumer turns these into the metrics a run is
    judged by, and a renamed key in an untyped dict reads as a zero on a dashboard instead of an
    error.
    """

    builder: str
    roots: int = 0
    chains: int = 0
    generated_tokens_captured: int = 0
    generated_tokens_delivered: int = 0
    # Only one chain is delivered per rollout today, so sub-agent branches and everything after a
    # context compaction are dropped. That is a deliberate limitation and has to stay visible.
    delivered_fraction: float = 0.0
    # Calls whose sibling was a retry the harness may or may not have kept. Unresolvable for the
    # final call of a rollout, because no later call names the survivor.
    unresolved_retries: list[str] = field(default_factory=list)
    # Retry responses that a later request proves the harness did not consume.
    # They are not policy actions in the episode delivered to the harness and must
    # not appear in the independent-call training payload.
    unused_retry_calls: list[str] = field(default_factory=list)
    # Calls the model returned with no generated tokens, kept out of the chain entirely.
    empty_generation_calls: list[str] = field(default_factory=list)
    # A text-based API can decode and re-tokenize a sampled turn before
    # sending it back. Those boundaries remain trainable on either side, but
    # the token-changed suffix itself must be masked.
    retokenized_boundaries: int = 0
    retokenized_tokens_masked: int = 0


@dataclass
class BuildOutput:
    chains: list[Chain]
    quarantined: list[str] = field(default_factory=list)  # model_call_ids
    notes: BuildNotes = field(default_factory=lambda: BuildNotes(builder=""))


def per_request(entries: list[TokenEntry]) -> BuildOutput:
    ordered = sorted(entries, key=lambda e: (len(e.prompt_token_ids), e.model_call_id))
    chains = [
        Chain(chain_id=f"req-{i}", root_prompt=list(e.prompt_token_ids), links=[ChainLink(entry=e, interstitial=[])])
        for i, e in enumerate(ordered)
    ]
    return BuildOutput(chains=chains, notes=BuildNotes(builder="per_request", chains=len(chains)))


def _is_prefix(a: list[int], b: list[int]) -> bool:
    return len(a) <= len(b) and b[: len(a)] == a


@dataclass(eq=False)  # identity-based, so nodes are hashable for set membership
class _Node:
    entry: TokenEntry
    cumulative: list[int]  # prompt + generation for this call
    parent: "_Node | None" = None
    parent_prefix_length: int = 0
    children: list["_Node"] = field(default_factory=list)
    quarantined: bool = False


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _infer_parent(prompt: list[int], candidates: list["_Node"]) -> tuple["_Node | None", bool, int]:
    """Choose the earlier call that preserves the longest prefix of this prompt.

    Exact extensions are preferred naturally by their full cumulative length.
    A near-prefix can win over a shorter exact ancestor when only the end of the
    immediately preceding generation was re-tokenized. Equal-length candidates
    are indistinguishable, so the subtree is quarantined rather than guessed.
    """
    matches = [
        (candidate, len(candidate.cumulative)) for candidate in candidates if _is_prefix(candidate.cumulative, prompt)
    ]

    # Text-based APIs decode a sampled turn and later re-tokenize that same
    # text as history. If the difference is confined to the end of the prior
    # generation, retain its token-identical prefix and let the child's exact
    # prompt replace the changed suffix as masked context. Requiring the whole
    # prior prompt plus most of its generation prevents context compaction or
    # ordinary history rewrites from being stitched this way.
    for candidate in candidates:
        generated = candidate.entry.generation_token_ids
        if _is_prefix(candidate.cumulative, prompt) or not generated or len(prompt) <= len(candidate.cumulative):
            continue
        prefix_length = _common_prefix_length(candidate.cumulative, prompt)
        generated_prefix_length = prefix_length - len(candidate.entry.prompt_token_ids)
        if generated_prefix_length >= 0.8 * len(generated) and len(generated) - generated_prefix_length <= 1024:
            matches.append((candidate, prefix_length))

    if not matches:
        return None, False, 0
    best_len = max(prefix_length for _, prefix_length in matches)
    best = [candidate for candidate, prefix_length in matches if prefix_length == best_len]
    return best[0], len(best) > 1, best_len


def prefix_merging(entries: list[TokenEntry]) -> BuildOutput:
    # A call that generated nothing (content filter, or the output budget exhausted before the
    # first token) has no trainable tokens, and its cumulative sequence equals its prompt. Left in,
    # it matches the prefix test against any call sharing that prompt and becomes its parent, so a
    # filtered call followed by a retry reads as a two-turn chain. Retry resolution only compares
    # siblings under a shared parent, so it would never see the pair.
    empty_generation = [e.model_call_id for e in entries if not e.generation_token_ids]
    entries = [e for e in entries if e.generation_token_ids]
    if not entries:
        return BuildOutput(
            chains=[], notes=BuildNotes(builder="prefix_merging", empty_generation_calls=empty_generation)
        )

    # Increasing prompt length is an order derived from the tokens: a parent's
    # cumulative sequence is a prefix of its child's prompt, so the parent's
    # prompt is shorter. This makes the pass order-independent.
    ordered = sorted(entries, key=lambda e: (len(e.prompt_token_ids), e.model_call_id))
    nodes: list[_Node] = []
    roots: list[_Node] = []
    quarantined: list[str] = []

    for entry in ordered:
        prompt = list(entry.prompt_token_ids)
        node = _Node(entry=entry, cumulative=prompt + list(entry.generation_token_ids))
        parent, ambiguous, parent_prefix_length = _infer_parent(prompt, nodes)
        if parent is not None:
            node.parent = parent
            node.parent_prefix_length = parent_prefix_length
            if ambiguous:
                # Two candidate parents with identical sequences: quarantine rather than guess.
                node.quarantined = True
                quarantined.append(entry.model_call_id)
            parent.children.append(node)
        else:
            roots.append(node)
        nodes.append(node)

    # Resolve retry siblings. A harness (Claude Code) retries on timeout / 5xx / dropped SSE, and the
    # capture point records a call even if the client never received it, so a retry yields two nodes
    # with identical prompt ids under the same parent and divergent generations.
    #
    # A recorded parent link settles this exactly: a later call names the sibling the harness actually
    # kept, so the other is provably unused. Without one we fall back to "the sibling a later call
    # extended wins". Neither can resolve a retry of the last call, because no later call names the
    # survivor, so that case is flagged as unresolved rather than tie-broken silently, and the
    # caller masks the rollout instead of training on a generation the client may never have received.
    unresolved_retries: list[str] = []
    unused_retry_calls: list[str] = []
    # Group by parent identity. Roots share the ROOTS key on purpose: a retry of a rollout's first
    # call produces two roots with the same prompt, and they have to be compared as siblings like
    # any other pair. An explicit key says so, where id(None) would leave it looking accidental.
    ROOTS = "roots"
    siblings_by_parent: dict[object, list[_Node]] = {}
    for node in nodes:
        siblings_by_parent.setdefault(ROOTS if node.parent is None else id(node.parent), []).append(node)
    for group in siblings_by_parent.values():
        by_prompt: dict[tuple, list[_Node]] = {}
        for node in group:
            by_prompt.setdefault(tuple(node.entry.prompt_token_ids), []).append(node)
        for retry_group in by_prompt.values():
            if len(retry_group) < 2:
                continue
            extended = [n for n in retry_group if n.children]
            if extended:
                keep = set(extended)
            else:
                keep = {min(retry_group, key=lambda n: n.entry.model_call_id)}
                unresolved_retries.extend(n.entry.model_call_id for n in retry_group)
            for node in retry_group:
                if node not in keep and not node.quarantined:
                    node.quarantined = True
                    quarantined.append(node.entry.model_call_id)
                    unused_retry_calls.append(node.entry.model_call_id)

    chains: list[Chain] = []

    def walk(node: _Node, path: list[_Node]) -> None:
        path = path + [node]
        if not node.children:
            if any(p.quarantined for p in path):
                return
            root = path[0]
            chain = Chain(chain_id="", root_prompt=list(root.entry.prompt_token_ids))
            for step, p in enumerate(path):
                interstitial = [] if step == 0 else list(p.entry.prompt_token_ids[p.parent_prefix_length :])
                generation_length = None
                if step + 1 < len(path):
                    child = path[step + 1]
                    kept = child.parent_prefix_length - len(p.entry.prompt_token_ids)
                    if kept < len(p.entry.generation_token_ids):
                        generation_length = kept
                chain.links.append(
                    ChainLink(
                        entry=p.entry,
                        interstitial=interstitial,
                        generation_length=generation_length,
                    )
                )
            chains.append(chain)
            return
        for child in node.children:
            walk(child, path)

    for root in roots:
        walk(root, [])

    # Only one chain is delivered per rollout, so when a rollout has more than one root, one has
    # to be chosen. The choice is the root whose call completed first.
    #
    # That is dispatch order for a sequential harness. capture_tokens is awaited inside the model
    # server's response path, so a call's record is durable before its response reaches the
    # harness, which means the next call has not been made yet. created_at is stamped at that
    # point. Ordering on the record's own timestamp rather than on file order matters because a
    # server running num_workers > 1 has several processes appending, and their interleaving is
    # lock order, not completion order.
    #
    # Two known shapes are not handled, both involving a second agent:
    #
    # An auxiliary call the harness makes on its own account, such as generating a conversation
    # title, is short and can complete before the agent's first real turn. It would then be picked
    # as the root and the agent's own chain relabelled a branch.
    #
    # Parallel sub-agents overlap, so completion order stops meaning dispatch order at all, and
    # nothing in a record says which agent made the call. Ordering cannot recover that. Keeping
    # sub-agent calls out of the records, by pointing them at a model server that is not the
    # policy's, can; some harnesses support configuring that.
    #
    # Both are follow-up work. Until then the split is visible rather than silent: a rollout that
    # split reports chains > 1 and a delivered_fraction below 1.
    def selection_key(c: Chain) -> tuple:
        # Earliest root first. Ties break on call id so the result is deterministic when two
        # records share a timestamp, and a chain with no links sorts last rather than raising.
        if not c.links:
            return (float("inf"), "")
        root = c.links[0].entry
        return (root.created_at, root.model_call_id)

    if chains:
        main = min(chains, key=selection_key)
        main.chain_id = "main"
        branch = 0
        for c in chains:
            if c is not main:
                c.chain_id = f"branch-{branch}"
                branch += 1

    delivered = sum(link.kept_generation_length() for link in main.links) if chains else 0
    captured = sum(len(e.generation_token_ids) for e in entries)
    retokenized_links = [link for link in main.links if link.generation_length is not None] if chains else []
    notes = BuildNotes(
        builder="prefix_merging",
        roots=len(roots),
        chains=len(chains),
        generated_tokens_captured=captured,
        generated_tokens_delivered=delivered,
        delivered_fraction=round(delivered / captured, 4) if captured else 0.0,
        unresolved_retries=unresolved_retries,
        unused_retry_calls=unused_retry_calls,
        empty_generation_calls=empty_generation,
        retokenized_boundaries=len(retokenized_links),
        retokenized_tokens_masked=sum(
            len(link.entry.generation_token_ids) - link.kept_generation_length() for link in retokenized_links
        ),
    )
    return BuildOutput(chains=chains, quarantined=quarantined, notes=notes)


_BUILDERS: dict[str, Callable[[list[TokenEntry]], BuildOutput]] = {
    "per_request": per_request,
    "prefix_merging": prefix_merging,
}


def run_builder(entries: list[TokenEntry], builder: str = "prefix_merging") -> BuildOutput:
    """Chain a rollout's records using the named strategy."""
    if builder not in _BUILDERS:
        raise ValueError(f"unknown builder {builder!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[builder](entries)


# --- Projection to a contiguous, token-bearing response ---


def project_chain_to_output_items(chain: Chain) -> list[dict]:
    """Project the chain into content-bearing Responses output items whose prompts are
    contiguous. For each call, emit its captured output items (assistant text, tool
    calls preserved) and set the contiguous prompt on the item that carries the
    generation, so each generated item's prompt extends the previous one. That is the
    shape a trainer ingests, with the text intact for anything that scores it. Falls back to a
    synthesized token-only item only when a call captured no content items."""
    items: list[dict] = []
    cumulative = list(chain.root_prompt)
    for step, link in enumerate(chain.links):
        cumulative = cumulative + (link.interstitial if step > 0 else [])
        entry = link.entry
        generation_length = link.kept_generation_length()
        generation_token_ids = list(entry.generation_token_ids[:generation_length])
        generation_log_probs = list(entry.generation_log_probs[:generation_length])

        def attach_execution_metadata(item: dict) -> None:
            if entry.routed_experts is not None:
                item["routed_experts"] = entry.routed_experts
            for field in (
                "ng_generation_replica_id",
                "ng_generation_weight_version",
                "ng_kv_cache_scheduler_block_size",
                "ng_kv_cache_hash_block_size",
            ):
                value = getattr(entry, field)
                if value is not None:
                    item[field] = value

        content_items = [dict(item) for item in (entry.output_items or [])]
        index = entry.token_item_index
        if index is not None and 0 <= index < len(content_items):
            # The item the arrays were taken off, recorded at capture time.
            generated = [content_items[index]]
        else:
            # Records written before the arrays were de-duplicated still carry them inline.
            generated = [item for item in content_items if item.get("generation_token_ids") is not None]
        if not generated and content_items:
            # No item carried token fields (unexpected); attach to the last so tokens are not lost.
            generated = content_items[-1:]
        if content_items:
            for item in generated:
                item["prompt_token_ids"] = list(cumulative)
                item["generation_token_ids"] = generation_token_ids
                item["generation_log_probs"] = generation_log_probs
                attach_execution_metadata(item)
            items.extend(content_items)
        else:
            item = {
                "type": "message",
                "prompt_token_ids": list(cumulative),
                "generation_token_ids": generation_token_ids,
                "generation_log_probs": generation_log_probs,
            }
            attach_execution_metadata(item)
            items.append(item)
        cumulative = cumulative + generation_token_ids
    return items


def project_main_chain_response(rollout_id: str, out: BuildOutput, model: str = "") -> dict:
    """Rebuild the main chain as a Responses object whose output items are contiguous.

    The result is an ordinary Gym-native Responses payload: ``object: "response"``, a list
    of ``output`` items, and ``usage``. The only thing that distinguishes it from what the
    model server returned is that the token fields on each generated item now describe an
    unbroken sequence across the whole rollout, because the items come from several model
    calls stitched together rather than one.
    """
    mains = [c for c in out.chains if c.chain_id == "main"] or out.chains[:1]
    output = project_chain_to_output_items(mains[0]) if mains else []
    # Token fields ride only on generated items; a content-only leading item (e.g. assistant
    # text emitted before a tool call) carries none. Read the usage counts from the items that
    # actually have token fields so a leading content item does not KeyError or skew the totals.
    generated = [item for item in output if item.get("generation_token_ids") is not None]
    n_in = len(generated[0]["prompt_token_ids"]) if generated else 0
    n_out = sum(len(item["generation_token_ids"]) for item in generated)
    return {
        "id": f"proj-{rollout_id}",
        "model": model,
        "object": "response",
        "output": output,
        "usage": {"input_tokens": n_in, "output_tokens": n_out},
    }


def project_independent_call_responses(
    rollout_id: str,
    entries: list[TokenEntry],
    out: BuildOutput,
    model: str = "",
) -> list[dict]:
    """Project every consumed call with its exact sampled prompt and generation.

    Calls are independent causal-attention segments. Unlike the legacy main-chain
    projection, this never truncates an earlier generation or rewrites a later
    prompt to make the two tokenizations contiguous. Context compactions and
    sub-agent roots are ordinary additional segments.
    """
    unused_retries = set(out.notes.unused_retry_calls)
    usable = [entry for entry in entries if entry.generation_token_ids and entry.model_call_id not in unused_retries]
    responses: list[dict] = []
    for index, entry in enumerate(sorted(usable, key=lambda item: (item.created_at, item.model_call_id))):
        chain = per_request([entry]).chains[0]
        output = project_chain_to_output_items(chain)
        responses.append(
            {
                "id": f"call-{rollout_id}-{index}",
                "model": model or entry.model,
                "object": "response",
                "output": output,
                "usage": {
                    "input_tokens": len(entry.prompt_token_ids),
                    "output_tokens": len(entry.generation_token_ids),
                },
            }
        )
    return responses


def assert_prefix_contiguity(response: dict) -> None:
    """Check the invariant the projection promises: each output item's
    prompt_token_ids must extend the tokens seen so far (prompt plus generation
    of all prior items). Raises AssertionError otherwise."""
    seen: list[int] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("generation_token_ids") is None:
            continue
        prompt = item.get("prompt_token_ids") or []
        if prompt[: len(seen)] != seen:
            raise AssertionError(
                "projection is not prefix-contiguous: an output item's prompt_token_ids "
                "does not extend the tokens seen so far"
            )
        seen = list(prompt) + list(item["generation_token_ids"])
