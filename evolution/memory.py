"""Memory bank for the evolution loop (창종설 보고서 §5).

Four retrieval modes (Phase 4 — research direction C1+C2):

  - keyword     : scenario tag + |perf_delta| ranking (lightweight default)
  - cosine      : text-embedding cosine top-k (ReasoningBank-style baseline)
  - state       : runtime state-vector nearest-neighbor (C1, scenario-state-conditioned)
  - contrastive : pair retrieval — matched (success, failure) for the same state (C2)
  - state_contrastive : C1 + C2 combined

Each `MemoryItem` now optionally carries a `state_signature` (the
supply-chain aggregates at the time the lesson was captured) and an
`embedding` (text vector, lazily computed). Items written before the
new fields existed will load with these as None.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
from typing import Iterable, Literal, Optional

MemoryType = Literal["success", "failure", "strategy"]


@dataclass
class StateSignature:
    """Snapshot of the simulator's supply-chain aggregates at the moment
    a memory item was captured. Used as a structured retrieval key (C1)."""
    disruption_level: float = 0.0
    supply_delay_level: float = 0.0
    urgent_ratio: float = 0.0
    avg_inbound_delay: float = 0.0
    mat_risk_mean: float = 0.0

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.disruption_level,
            self.supply_delay_level / 2.0,           # rescale to [0,1] roughly
            self.urgent_ratio,
            min(self.avg_inbound_delay / 120.0, 1.0),  # 0..1
            self.mat_risk_mean,
        )

    @staticmethod
    def distance(a: "StateSignature", b: "StateSignature") -> float:
        """Weighted L2 in [0,1]-normalised state space."""
        va, vb = a.as_vector(), b.as_vector()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))


@dataclass
class MemoryItem:
    title: str           # one-line strategy summary
    description: str     # when it applies (scenario + condition)
    content: str         # concrete logic / DSL snippet / lesson
    scenario: str        # S0..S5 it was observed under
    perf_delta: float    # signed: positive = improvement vs baseline (%)
    type: MemoryType = "strategy"
    # Phase-4 fields (Optional → backward-compatible loading).
    state_signature: Optional[StateSignature] = None
    embedding: Optional[list[float]] = None

    def is_success(self) -> bool: return self.type == "success"
    def is_failure(self) -> bool: return self.type == "failure"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


RetrievalMode = Literal["keyword", "cosine", "state", "contrastive", "state_contrastive"]


@dataclass
class MemoryBank:
    items: list[MemoryItem] = field(default_factory=list)

    def add(self, item: MemoryItem) -> None:
        # Deduplicate by (title, scenario) — newer entry wins on perf_delta.
        for i, existing in enumerate(self.items):
            if existing.title == item.title and existing.scenario == item.scenario:
                # Keep the entry with the larger absolute perf_delta.
                if abs(item.perf_delta) > abs(existing.perf_delta):
                    self.items[i] = item
                return
        self.items.append(item)

    def add_many(self, items: Iterable[MemoryItem]) -> None:
        for it in items:
            self.add(it)

    # ------------------------------------------------------------------ #
    # Retrieval — 5 modes for the Phase-4 ablation                       #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        scenario: str,
        top_k: int = 3,
        only_type: Optional[MemoryType] = None,
        *,
        mode: RetrievalMode = "keyword",
        current_state: Optional[StateSignature] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> list[MemoryItem]:
        """Dispatch to the requested retrieval mode.

        Default `keyword` mode preserves the existing behaviour so older
        call sites keep working unchanged. The other modes need either a
        `current_state` (C1) or `query_embedding` (cosine baseline).
        """
        if mode == "keyword":
            return self._retrieve_keyword(scenario, top_k, only_type)
        if mode == "cosine":
            return self._retrieve_cosine(query_embedding, top_k, only_type)
        if mode == "state":
            return self._retrieve_state(current_state, top_k, only_type)
        if mode == "contrastive":
            return self._retrieve_contrastive(scenario, top_k, query_embedding)
        if mode == "state_contrastive":
            return self._retrieve_state_contrastive(current_state, top_k)
        raise ValueError(f"unknown retrieval mode: {mode}")

    # --- (existing) keyword mode -------------------------------------

    def _retrieve_keyword(
        self,
        scenario: str,
        top_k: int,
        only_type: Optional[MemoryType],
    ) -> list[MemoryItem]:
        candidates = [m for m in self.items if (only_type is None or m.type == only_type)]
        same = [m for m in candidates if m.scenario == scenario]
        same.sort(key=lambda m: abs(m.perf_delta), reverse=True)
        if len(same) >= top_k:
            return same[:top_k]
        rest = [m for m in candidates if m.scenario != scenario]
        rest.sort(key=lambda m: abs(m.perf_delta), reverse=True)
        return (same + rest)[:top_k]

    # --- cosine (ReasoningBank-style baseline) -----------------------

    def _retrieve_cosine(
        self,
        query_embedding: Optional[list[float]],
        top_k: int,
        only_type: Optional[MemoryType],
    ) -> list[MemoryItem]:
        if not query_embedding:
            return []
        scored: list[tuple[float, MemoryItem]] = []
        for m in self.items:
            if only_type is not None and m.type != only_type:
                continue
            if not m.embedding:
                continue
            scored.append((_cosine(query_embedding, m.embedding), m))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    # --- state-conditioned (C1) ---------------------------------------

    def _retrieve_state(
        self,
        current_state: Optional[StateSignature],
        top_k: int,
        only_type: Optional[MemoryType],
    ) -> list[MemoryItem]:
        if current_state is None:
            return []
        scored: list[tuple[float, MemoryItem]] = []
        for m in self.items:
            if only_type is not None and m.type != only_type:
                continue
            if m.state_signature is None:
                continue
            d = StateSignature.distance(current_state, m.state_signature)
            scored.append((d, m))
        scored.sort(key=lambda kv: kv[0])  # smaller distance first
        return [m for _, m in scored[:top_k]]

    # --- contrastive pair (C2) ----------------------------------------

    def _retrieve_contrastive(
        self,
        scenario: str,
        top_k: int,
        query_embedding: Optional[list[float]],
    ) -> list[MemoryItem]:
        """Return up to `top_k` interleaved (success, failure) pairs as
        a flat list. Each pair is for the same scenario where possible."""
        out: list[MemoryItem] = []
        used: set[int] = set()
        for _ in range(top_k):
            s = self._best_unused(scenario, "success", used, query_embedding)
            f = self._best_unused(scenario, "failure", used, query_embedding)
            if s is None and f is None:
                break
            if s is not None:
                out.append(s)
                used.add(id(s))
            if f is not None:
                out.append(f)
                used.add(id(f))
        return out

    def _best_unused(
        self,
        scenario: str,
        kind: MemoryType,
        used: set[int],
        query_embedding: Optional[list[float]],
    ) -> Optional[MemoryItem]:
        # Prefer same scenario; if cosine query supplied, rank by cosine.
        same = [m for m in self.items
                if m.scenario == scenario and m.type == kind and id(m) not in used]
        if query_embedding:
            same = [m for m in same if m.embedding]
            same.sort(key=lambda m: _cosine(query_embedding, m.embedding or []), reverse=True)
        else:
            same.sort(key=lambda m: abs(m.perf_delta), reverse=True)
        if same:
            return same[0]
        rest = [m for m in self.items
                if m.type == kind and id(m) not in used]
        rest.sort(key=lambda m: abs(m.perf_delta), reverse=True)
        return rest[0] if rest else None

    # --- state + contrastive (C1 + C2) --------------------------------

    def _retrieve_state_contrastive(
        self,
        current_state: Optional[StateSignature],
        top_k: int,
    ) -> list[MemoryItem]:
        """For each of top_k state-nearest contexts, return a matched
        (success, failure) pair drawn from items closest to that state."""
        if current_state is None:
            return []

        def nearest_of_type(kind: MemoryType, exclude: set[int]) -> Optional[MemoryItem]:
            scored: list[tuple[float, MemoryItem]] = []
            for m in self.items:
                if m.type != kind or id(m) in exclude or m.state_signature is None:
                    continue
                scored.append((StateSignature.distance(current_state, m.state_signature), m))
            scored.sort(key=lambda kv: kv[0])
            return scored[0][1] if scored else None

        out: list[MemoryItem] = []
        used: set[int] = set()
        for _ in range(top_k):
            s = nearest_of_type("success", used)
            f = nearest_of_type("failure", used)
            if s is None and f is None:
                break
            if s is not None:
                out.append(s); used.add(id(s))
            if f is not None:
                out.append(f); used.add(id(f))
        return out

    # --- persistence ----------------------------------------------------

    def save(self, path: Path) -> None:
        # asdict recursively converts StateSignature too; embeddings are
        # plain lists.
        path.write_text(json.dumps([asdict(m) for m in self.items],
                                   indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        items: list[MemoryItem] = []
        for d in raw:
            sig = d.pop("state_signature", None)
            if isinstance(sig, dict):
                sig = StateSignature(**sig)
            items.append(MemoryItem(state_signature=sig, **d))
        return cls(items=items)
