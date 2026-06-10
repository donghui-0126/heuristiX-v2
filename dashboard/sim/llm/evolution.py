"""EoH 기반 LLM 규칙 진화 엔진.

개체군 구성: 15개 (B1-B10 기본 규칙 + LLM 초기 생성 5개)
세대 수: 20
생존 규칙 수 (top-k): 7
세대당 신규 생성: 8개 (E1×3, E2×2, M1×2, M2×1)
적합도 평가 시드: 5개

각 세대마다 생성된 규칙의 프롬프트 요약·코드·평가 결과를 generation_log에 기록합니다.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..core.simulator import FJSSPSimulator
from ..core.job import Job
from ..rules.baseline import RULES as BASELINE_RULES
from ..rules.interface import load_priority_fn, validate_priority_fn
from ..data.loader import generate_due_dates
from . import client as llm
from .operators import (
    e1_mutation, e2_crossover, m1_reflection,
    m2_simplification, initial_generation_prompt,
)
from .client import generate_lessons
from .experience import ExperienceStore

N_ITER      = 20
POOL_SIZE   = 15
N_TOP       = 7
N_EVAL      = 5
N_FINAL     = 100

_BASELINE_IDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B9","B10"]

_OPERATOR_KR = {
    "E1":       "변이 (Mutation)",
    "E2":       "교차 (Crossover)",
    "M1":       "반성적 개선 (Reflection)",
    "M2":       "단순화 (Simplification)",
    "BASELINE": "기본 규칙",
    "LLM_INIT": "LLM 초기 생성",
}


@dataclass
class RuleEntry:
    rule_id:    str
    code:       str
    fn:         Callable
    at_scores:  List[float] = field(default_factory=list)
    generation: int = 0
    parents:    List[str] = field(default_factory=list)
    operator:   str = "BASELINE"

    @property
    def avg_at(self) -> float:
        return sum(self.at_scores) / len(self.at_scores) if self.at_scores else float("inf")

    def to_dict(self) -> dict:
        return {
            "rule_id":    self.rule_id,
            "code":       self.code,
            "avg_at":     self.avg_at if self.at_scores else None,
            "generation": self.generation,
            "parents":    self.parents,
            "operator":   self.operator,
            "operator_kr": _OPERATOR_KR.get(self.operator, self.operator),
        }


@dataclass
class GenerationLogEntry:
    """세대별 규칙 생성 기록 — 프롬프트 요약·코드·평가 결과 포함."""
    iteration:      int
    rule_id:        str
    operator:       str
    operator_kr:    str
    parents:        List[str]
    prompt_summary: str      # 프롬프트 앞 600자
    code:           str
    avg_at:         float = float("inf")
    entered_pool:   bool = False

    def to_dict(self) -> dict:
        return {
            "iteration":      self.iteration,
            "rule_id":        self.rule_id,
            "operator":       self.operator,
            "operator_kr":    self.operator_kr,
            "parents":        self.parents,
            "prompt_summary": self.prompt_summary,
            "code":           self.code,
            "avg_at":         self.avg_at if self.avg_at != float("inf") else None,
            "entered_pool":   self.entered_pool,
        }


@dataclass
class EvolutionResult:
    best_rule:           RuleEntry
    pool_history:        List[List[RuleEntry]]
    iteration_best_at:   List[float]
    genealogy:           List[Dict]
    generation_log:      List[Dict]   # GenerationLogEntry.to_dict() 목록


class EoHEvolution:

    def __init__(
        self,
        jobs: List[Job],
        n_machines: int,
        scenario_factory,
        scenario_tag: str,
        scenario_description: str,
        use_external_vars: bool = True,
        use_experience:    bool = False,
        experience_store:  Optional[ExperienceStore] = None,
        ddt:               float = 1.5,
        seed_offset:       int   = 0,
        progress_callback  = None,
    ):
        self.jobs               = jobs
        self.n_machines         = n_machines
        self.scenario_factory   = scenario_factory
        self.scenario_tag       = scenario_tag
        self.scenario_desc      = scenario_description
        self.use_external_vars  = use_external_vars
        self.use_experience     = use_experience
        self.store              = experience_store or ExperienceStore()
        self.ddt                = ddt
        self.seed_offset        = seed_offset
        self.progress_callback  = progress_callback
        self.eval_seeds         = list(range(seed_offset, seed_offset + N_EVAL))
        self._gen_log: List[GenerationLogEntry] = []

    # ------------------------------------------------------------------
    def run(self) -> EvolutionResult:
        pool = self._init_pool()
        genealogy  = [r.to_dict() for r in pool]
        history:   List[List[RuleEntry]] = []
        iter_best: List[float] = []

        for iteration in range(1, N_ITER + 1):
            self._evaluate_pool(pool)
            pool.sort(key=lambda r: r.avg_at)
            survivors       = pool[:N_TOP]
            current_best_at = survivors[0].avg_at
            iter_best.append(current_best_at)

            # 생성 로그에 이번 세대 평가 결과 반영
            self._update_log_scores(pool, iteration)

            msg = (f"[세대 {iteration:02d}/{N_ITER}] "
                   f"최적 AT={current_best_at:.4f}  규칙={survivors[0].rule_id}")
            print(f"  {msg}")
            if self.progress_callback:
                self.progress_callback(iteration, current_best_at, msg)

            history.append(deepcopy(survivors))

            if iteration == N_ITER:
                break

            new_rules = self._generate_new_rules(survivors, iteration)
            valid_new = [r for r in new_rules if r is not None]
            genealogy.extend(r.to_dict() for r in valid_new)

            # 신규 규칙이 pool에 진입했는지 기록
            survivor_ids = {r.rule_id for r in survivors}
            new_pool = survivors + valid_new
            fallback_ids = list(BASELINE_RULES.keys())
            while len(new_pool) < POOL_SIZE:
                bid = random.choice(fallback_ids)
                new_pool.append(RuleEntry(
                    rule_id=f"{bid}_pad", code="",
                    fn=self._wrap_fn(BASELINE_RULES[bid]),
                    generation=iteration, operator="BASELINE",
                ))
            for entry in self._gen_log:
                if entry.iteration == iteration and entry.rule_id in {r.rule_id for r in valid_new}:
                    entry.entered_pool = True

            if self.use_experience:
                self._run_reflector(pool, iteration)

            pool = new_pool

        best = min(pool, key=lambda r: r.avg_at)
        return EvolutionResult(
            best_rule=best,
            pool_history=history,
            iteration_best_at=iter_best,
            genealogy=genealogy,
            generation_log=[e.to_dict() for e in self._gen_log],
        )

    # ------------------------------------------------------------------
    # 초기화 (B1-B10 + LLM 5개 = 15)
    # ------------------------------------------------------------------
    def _init_pool(self) -> List[RuleEntry]:
        pool: List[RuleEntry] = []
        for bid in _BASELINE_IDS:
            pool.append(RuleEntry(
                rule_id=bid, code="",
                fn=self._wrap_fn(BASELINE_RULES[bid]),
                generation=0, operator="BASELINE",
            ))

        prompt = initial_generation_prompt(self.scenario_desc)
        for i in range(1, 6):
            code = llm.generate_rule(prompt)
            if code:
                try:
                    fn = load_priority_fn(code)
                    if validate_priority_fn(fn):
                        rid = f"LLM_init_{i}"
                        pool.append(RuleEntry(
                            rule_id=rid, code=code,
                            fn=self._wrap_fn(fn),
                            generation=0, operator="LLM_INIT",
                        ))
                        self._gen_log.append(GenerationLogEntry(
                            iteration=0, rule_id=rid,
                            operator="LLM_INIT",
                            operator_kr=_OPERATOR_KR["LLM_INIT"],
                            parents=[],
                            prompt_summary=prompt[:600],
                            code=code,
                        ))
                        continue
                except Exception:
                    pass
            # fallback
            fallback_id = f"LLM_init_{i}_fallback"
            pool.append(RuleEntry(
                rule_id=fallback_id, code="",
                fn=self._wrap_fn(BASELINE_RULES["B3"]),
                generation=0, operator="LLM_INIT",
            ))

        return pool[:POOL_SIZE]

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    def _evaluate_pool(self, pool: List[RuleEntry]) -> None:
        for rule in pool:
            rule.at_scores = [self._eval_single(rule.fn, s) for s in self.eval_seeds]

    def _eval_single(self, fn: Callable, seed: int) -> float:
        jobs   = generate_due_dates(deepcopy(self.jobs), ddt=self.ddt, seed=seed)
        events = self.scenario_factory(seed)
        sim    = FJSSPSimulator(jobs, self.n_machines, fn, scenario_events=events)
        try:
            return sim.run().at
        except Exception:
            return float("inf")

    def _update_log_scores(self, pool: List[RuleEntry], iteration: int):
        """이번 세대에 생성된 로그 항목에 평가 결과 반영."""
        score_map = {r.rule_id: r.avg_at for r in pool}
        for entry in self._gen_log:
            if entry.iteration == iteration and entry.rule_id in score_map:
                entry.avg_at = score_map[entry.rule_id]

    # ------------------------------------------------------------------
    # 신규 규칙 생성 (E1×3, E2×2, M1×2, M2×1)
    # ------------------------------------------------------------------
    def _generate_new_rules(
        self, top: List[RuleEntry], iteration: int
    ) -> List[Optional[RuleEntry]]:
        results = []
        exp_block    = (self.store.format_for_prompt(self.scenario_tag)
                        if self.use_experience else None)
        best_baseline = self._best_baseline_at(top)

        # Build elite_rules list for context: [(rule_id, code, avg_at)]
        elite_rules = [
            (r.rule_id, r.code or _fn_to_code(r.fn), r.avg_at)
            for r in top[:3]
        ]

        # E1 × 3
        for rank in range(min(3, len(top))):
            t      = top[rank]
            prompt = e1_mutation(
                t.rule_id, t.code or _fn_to_code(t.fn), t.avg_at,
                scenario_desc=self.scenario_desc,
                top_rules=elite_rules,
            )
            r = self._try_generate(
                f"E1_{iteration}_{rank+1}", prompt,
                operator="E1", parents=[t.rule_id], generation=iteration)
            results.append(r)

        # E2 × 2
        for k in range(min(2, len(top) - 1)):
            a, b   = top[k], top[k + 1]
            prompt = e2_crossover(
                a.rule_id, a.code or _fn_to_code(a.fn), a.avg_at,
                b.rule_id, b.code or _fn_to_code(b.fn), b.avg_at,
                scenario_desc=self.scenario_desc,
            )
            r = self._try_generate(
                f"E2_{iteration}_{k+1}", prompt,
                operator="E2", parents=[a.rule_id, b.rule_id], generation=iteration)
            results.append(r)

        # M1 × 2
        for rank in range(min(2, len(top))):
            t      = top[rank]
            prompt = m1_reflection(
                t.rule_id, t.code or _fn_to_code(t.fn), t.avg_at,
                best_baseline, self.scenario_desc, exp_block,
            )
            r = self._try_generate(
                f"M1_{iteration}_{rank+1}", prompt,
                operator="M1", parents=[t.rule_id], generation=iteration)
            results.append(r)

        # M2 × 1
        t      = top[0]
        prompt = m2_simplification(
            t.rule_id, t.code or _fn_to_code(t.fn), t.avg_at,
            scenario_desc=self.scenario_desc,
        )
        r = self._try_generate(
            f"M2_{iteration}", prompt,
            operator="M2", parents=[t.rule_id], generation=iteration)
        results.append(r)

        return results

    def _try_generate(
        self, rule_id: str, prompt: str,
        operator: str = "E1", parents: List[str] = None,
        generation: int = 0,
    ) -> Optional[RuleEntry]:
        code = llm.generate_rule(prompt)

        # 생성 로그 기록 (성공 여부와 무관)
        log_entry = GenerationLogEntry(
            iteration=generation,
            rule_id=rule_id,
            operator=operator,
            operator_kr=_OPERATOR_KR.get(operator, operator),
            parents=parents or [],
            prompt_summary=prompt[:600],
            code=code or "(생성 실패)",
        )
        self._gen_log.append(log_entry)

        if not code:
            return None
        try:
            fn = load_priority_fn(code)
            if not validate_priority_fn(fn):
                return None
            return RuleEntry(
                rule_id=rule_id, code=code,
                fn=self._wrap_fn(fn),
                generation=generation,
                parents=parents or [],
                operator=operator,
            )
        except Exception as exc:
            print(f"  [EoH] {rule_id} 실패: {exc}")
            return None

    # ------------------------------------------------------------------
    def _wrap_fn(self, fn: Callable) -> Callable:
        if self.use_external_vars:
            return fn

        def masked(job, operation, machine, state):
            j = dict(job)
            j["part_available_time"] = 0.0
            j["urgent_order_flag"]   = 0
            return fn(j, operation, machine, state)

        return masked

    def _best_baseline_at(self, pool: List[RuleEntry]) -> float:
        entries = [r for r in pool if r.rule_id in BASELINE_RULES]
        if not entries:
            return float("inf")
        return min(r.avg_at for r in entries)

    def _run_reflector(self, pool: List[RuleEntry], iteration: int) -> None:
        """LLM-S: extract structured lessons from this generation and store them (P3)."""
        best_baseline = self._best_baseline_at(pool)
        llm_rules = [r for r in pool if r.operator not in ("BASELINE",) and r.code]

        success = [
            (r.rule_id, r.code, r.avg_at, best_baseline)
            for r in llm_rules if r.avg_at <= best_baseline * 0.95
        ]
        failure = [
            (r.rule_id, r.code, r.avg_at, best_baseline)
            for r in llm_rules if r.avg_at > best_baseline * 1.05
        ]

        if not success and not failure:
            return

        lessons = generate_lessons(self.scenario_desc, success, failure)
        for lesson in lessons:
            self.store.record_lesson(self.scenario_tag, iteration, lesson)

        if lessons:
            print(f"  [LLM-S] {iteration}세대 교훈 {len(lessons)}개 저장")


def _fn_to_code(fn: Callable) -> str:
    import inspect
    try:
        return inspect.getsource(fn)
    except Exception:
        return f"# 내장 규칙: {fn.__name__}"
