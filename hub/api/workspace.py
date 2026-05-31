"""Per-user workspace state.

Each workspace lives at `platform/workspace/<user>/` and contains:
  - baselines.json  — user-defined dispatching rules (additions to BASELINES)
  - experiments/    — launched experiment outputs

Source files in evolution/ are never modified by the platform; the
runtime composes "effective baselines" = BASELINES (from code) +
workspace.baselines.json (user additions) at experiment-launch time.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT / "hub" / "workspace"


def _safe_user(user: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,32}", user or ""):
        raise ValueError(f"invalid user id: {user!r}")
    return user


@dataclass
class Workspace:
    user: str
    root: Path

    @property
    def baselines_path(self) -> Path:
        return self.root / "baselines.json"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    def load_baselines(self) -> dict[str, str]:
        if not self.baselines_path.exists():
            return {}
        return json.loads(self.baselines_path.read_text())

    def save_baselines(self, bls: dict[str, str]) -> None:
        self.baselines_path.write_text(json.dumps(bls, indent=2, ensure_ascii=False))

    def add_baseline(self, name: str, expr: str, *, description: str = "") -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,32}", name or ""):
            raise ValueError(f"invalid baseline name: {name!r}")
        bls = self.load_baselines()
        bls[name] = expr
        self.save_baselines(bls)
        # Side log of NL/description for audit.
        log = self.root / "baselines_log.jsonl"
        with log.open("a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "name": name,
                "expr": expr,
                "description": description,
            }, ensure_ascii=False) + "\n")
        return {"name": name, "expr": expr}

    def remove_baseline(self, name: str) -> bool:
        bls = self.load_baselines()
        if name in bls:
            del bls[name]
            self.save_baselines(bls)
            return True
        return False


def get_workspace(user: str) -> Workspace:
    u = _safe_user(user)
    root = WS_ROOT / u
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(exist_ok=True)
    return Workspace(user=u, root=root)
