"""P3 experience storage – JSON-backed memory of successful/failed rules.

Schema of experience_store.json
--------------------------------
{
  "<scenario_tag>": [          // e.g. "S1_ratio0.20_k1.0"
    {
      "id": "exp_001",
      "iteration": 5,
      "rule_code": "...",
      "at_score": 12.3,
      "best_baseline_at": 14.0,
      "improvement_ratio": 0.121,
      "is_success": true,
      "pattern": "Natural language summary of what worked / failed."
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Optional

DEFAULT_PATH = Path(__file__).parent.parent.parent / "results" / "experience_store.json"
SUCCESS_THRESHOLD = 0.95   # AT ≤ 95 % of best-baseline AT → success
MAX_PER_SCENARIO = 20      # cap entries per scenario tag


class ExperienceStore:

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def record(
        self,
        scenario_tag: str,
        iteration: int,
        rule_code: str,
        at_score: float,
        best_baseline_at: float,
        pattern: str,
    ) -> None:
        """Save one experience entry for the given scenario."""
        is_success = (best_baseline_at > 0) and (at_score <= best_baseline_at * SUCCESS_THRESHOLD)
        improvement = (best_baseline_at - at_score) / max(best_baseline_at, 1e-9)

        entry = {
            "id": f"exp_{uuid.uuid4().hex[:8]}",
            "iteration": iteration,
            "rule_code": rule_code,
            "at_score": round(at_score, 4),
            "best_baseline_at": round(best_baseline_at, 4),
            "improvement_ratio": round(improvement, 4),
            "is_success": is_success,
            "pattern": pattern,
        }

        bucket = self._data.setdefault(scenario_tag, [])
        bucket.append(entry)

        # Trim oldest if over cap
        if len(bucket) > MAX_PER_SCENARIO:
            bucket.sort(key=lambda e: e["improvement_ratio"], reverse=True)
            self._data[scenario_tag] = bucket[:MAX_PER_SCENARIO]

        self._save()

    def record_lesson(
        self,
        scenario_tag: str,
        iteration: int,
        lesson: dict,
    ) -> None:
        """Save one structured LESSON block from LLM-S reflector."""
        entry = {
            "id":        f"lesson_{__import__('uuid').uuid4().hex[:8]}",
            "source":    "LLM-S",
            "iteration": iteration,
            "type":      lesson.get("type", "strategy"),
            "title":     lesson.get("title", ""),
            "description": lesson.get("description", ""),
            "content":   lesson.get("content", ""),
            "perf_delta": lesson.get("perf_delta", "0"),
            "is_success": lesson.get("type") == "success",
            "improvement_ratio": self._delta_to_ratio(lesson.get("perf_delta", "0")),
        }
        bucket = self._data.setdefault(scenario_tag, [])
        bucket.append(entry)
        if len(bucket) > MAX_PER_SCENARIO:
            bucket.sort(key=lambda e: e["improvement_ratio"], reverse=True)
            self._data[scenario_tag] = bucket[:MAX_PER_SCENARIO]
        self._save()

    @staticmethod
    def _delta_to_ratio(perf_delta) -> float:
        try:
            return float(str(perf_delta).replace('%', '').strip()) / 100
        except (ValueError, TypeError):
            return 0.0

    def retrieve_top(self, scenario_tag: str, k: int = 3) -> List[dict]:
        """Return up to k best experiences for the scenario (by improvement_ratio)."""
        bucket = self._data.get(scenario_tag, [])
        sorted_bucket = sorted(bucket, key=lambda e: e["improvement_ratio"], reverse=True)
        return sorted_bucket[:k]

    def format_for_prompt(self, scenario_tag: str, k: int = 3) -> str:
        """Return a human-readable block to inject into LLM-A prompts."""
        entries = self.retrieve_top(scenario_tag, k)
        if not entries:
            return "(No prior experiences for this scenario.)"

        lines = [f"Top-{len(entries)} lessons for scenario {scenario_tag!r}:"]
        for i, e in enumerate(entries, 1):
            status  = "SUCCESS" if e.get("is_success") else "FAILURE"
            # LLM-S structured lesson
            if e.get("source") == "LLM-S":
                lines.append(
                    f"{i}. [{status} | {e.get('type','?')}] {e.get('title','')}\n"
                    f"   applies: {e.get('description','')}\n"
                    f"   detail:  {e.get('content','')}\n"
                    f"   delta:   {e.get('perf_delta','?')}%"
                )
            else:
                # legacy string-pattern entry
                lines.append(
                    f"{i}. [{status}] iter={e.get('iteration','?')}  "
                    f"improvement={e.get('improvement_ratio', 0):.1%}\n"
                    f"   {e.get('pattern', e.get('content', ''))}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)
