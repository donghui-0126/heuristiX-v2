"""Copy tracked seed_results/ (heuristiX v3 measurements) into results/
so a fresh clone has demo data without re-running the batteries.

Run once after clone:
    python3 seed.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).parent
src = HERE / "seed_results"
dst = HERE / "results"
dst.mkdir(exist_ok=True)
n = 0
for p in src.glob("*.json"):
    shutil.copy2(p, dst / p.name)
    n += 1
print(f"seeded {n} result files into {dst}")
