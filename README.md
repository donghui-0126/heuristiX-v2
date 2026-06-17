<div align="center">

# ⚙️ heuristiX

**LLM-Evolved Dispatching Rules for Dynamic FJSSP under Supply-Chain Disruption**

*외부 공급망 충격 환경에서, LLM이 인간 전문가의 스케줄링 휴리스틱을 분 단위·커피값에 진화시킨다.*

![Rust](https://img.shields.io/badge/engine-Rust%20DES-orange?logo=rust)
![Python](https://img.shields.io/badge/evolution-Python%203.12-blue?logo=python)
![LLM](https://img.shields.io/badge/LLM-gpt--4o--mini-10a37f?logo=openai)
![Dashboard](https://img.shields.io/badge/UI-Flask%20Workbench-000?logo=flask)
![License](https://img.shields.io/badge/status-research-purple)

### 🌐 [프로젝트 페이지 · 발표 슬라이드 →](https://donghui-0126.github.io/heuristiX-v2/)

</div>

---

## 한 장 요약

| | 인간 전문가 | **heuristiX** |
|---|---|---|
| 새 dispatching rule 1개 | 12~36개월 / $120k+ | **~1분 / $0.07** |
| 외란 적응 | 정적 — 재설계 필요 | **재진화로 즉시 대응** |
| 결과 형태 | 논문 속 수식 | **실행 가능한 코드 + 자연어 설명** |

27개 실험 셀(3 scenario × 3 variant × 3 flexibility) 전부에서 12종 인간 휴리스틱
(FIFO·EDD·SPT·CR·Urgency·WMDD·COVERT·ATC·MDD·MWKR·LWKR·LPT)의 최강자와
동률 이상, **최대 ARI +9.7%** (moderate FJSSP × 부품 지연 시나리오).
전체 재현 비용: **$1.95 / 27분.**

---

## 아키텍처

```
                        ┌─────────────────────────────────┐
   자연어 외란 묘사  ──▶ │  dashboard/  Flask Workbench    │ ◀── 팀원·교수님
   "항만 폐쇄 2주"      │  벤치마크·시나리오·진화·보고서   │     (브라우저)
                        └───────┬─────────────────────────┘
                                │ 규칙 import / NL 설명 / 재실험
                        ┌───────▼─────────────────────────┐
   EoH 4연산 진화    ──▶ │  evolution/  LLM-A + LLM-S 루프  │ ──▶ gpt-4o-mini
   P1/P2/P3 ablation    │  MemoryBank · 9-combo 병렬 배터리 │
                        └───────┬─────────────────────────┘
                                │ subprocess (JSON metrics)
                        ┌───────▼─────────────────────────┐
   100-seed 평가     ──▶ │  src/  Rust DES Simulator        │
   ~0.2s / 100 reps     │  FJSSP 행렬 · S0/S1/S2 · evalexpr │
                        └─────────────────────────────────┘
```

| 레이어 | 역할 | 왜 이 기술 |
|---|---|---|
| `src/` | 이산사건 시뮬레이터 — 적합도 평가의 진실 공급원 | Rust: SimPy 대비 10–50×, 18k 시뮬레이션을 분 단위에 |
| `evolution/` | LLM-A(생성) + LLM-S(반성) 진화 루프, 배터리 드라이버 | EoH(ICML'24) 4연산 + EvoDR 듀얼-LLM + ReasoningBank 메모리 |
| `dashboard/` | 13페이지 연구 워크벤치 — 실험·비교·자연어 보고서 | Flask: 비전공자 self-service, 규칙을 한국어로 풀이 |

---

## 연구 질문과 답

| RQ | 질문 | 답 (v3 실측) |
|---|---|---|
| **RQ1** | LLM이 12종 인간 휴리스틱의 최강자를 이기는가? | ✅ 27/27 cell 동률 이상, peak +9.7% |
| **RQ2** | 외란 변수(긴급 플래그·부품 가용 시각) 노출이 가치 있는가? | ✅ S0/S1에서 P2 > P1 — 단, 환경 의존적 |
| **RQ3** | 성공/실패 경험 메모리가 추가 기여하는가? | ⚠️ marginal — 단일 시나리오 내 lesson 포화 (한계로 명시) |

**핵심 발견** — LLM의 sweet spot은 **moderate FJSSP (flex≈0.5) × 비단순 외란**.
strict JSSP(flex=0)에선 SPT가, full FJSSP(flex=1)에선 EDD/CR이 이미 거의 최적이라
LLM의 추가 여지가 줄어든다. Routing 자유도 자체는 AT를 65~81% 줄이는 가장 큰 지렛대.

LLM이 발견한 규칙 예시 (S1 부품지연 × flex=0.5, ARI +9.7% vs WMDD):

```text
exp_(-0.125 * max_(0, due_date - (current_time + remaining_pt + part_available_time)))
  / remaining_pt
```

> *"납기 위험이 클수록 지수적으로 우선하되, 남은 작업량으로 정규화"* — 대시보드가
> 모든 규칙을 이런 자연어로 자동 풀이한다.

---

## Quickstart

### A. 대시보드 (팀원·시연용 — 코드 없이)

```bash
git clone https://github.com/donghui-0126/heuristiX-v2.git && cd heuristiX-v2
echo "OPENAI_API_KEY=sk-..." > .env

cd dashboard
pip install -r requirements.txt
python3 seed.py          # v3 실측 결과 37파일 시드
python3 app.py           # → http://localhost:5000
```

또는 Docker 한 줄: `docker compose up --build`

대시보드에서 할 수 있는 것:
1. **규칙 탐색기** — 37개 규칙(인간 10 + LLM 27) 코드·변수·계보 + **자연어 설명**
2. **이 규칙으로 직접 실험** — 임의 인스턴스×시나리오에 재실행, ARI 즉시 확인
3. **이벤트 변환기** — "호르무즈 봉쇄 2주" → 시뮬레이션 파라미터 자동 변환
4. **진화 센터** — P1/P2/P3 LLM 진화를 브라우저에서 launch
5. **보고서 생성** — MD/PDF/DOCX, *"작업당 5.5분 덜 늦음"* 식의 실무 환산 포함

📖 **페이지별 상세 사용법: [`dashboard/GUIDE.md`](dashboard/GUIDE.md)** — 처음 10분
코스부터 전체 실험 사이클·FAQ까지.

### B. 연구 파이프라인 (Rust 엔진 — 본 실험용)

```bash
cargo build --release

# 단일 진화 (S1 부품지연, P3 변형, flex=0.5)
python3 -m evolution.evolve \
    --scenario S1 --variant P3 --flexibility 0.5 \
    --iterations 20 --replications 100 \
    --provider openai --model gpt-4o-mini

# 9-combo 병렬 배터리 (~9분) → 12-baseline 비교 → flex sweep 통합
python3 -m evolution.p123_battery
python3 -m evolution.compare_p123_vs_baselines
python3 -m evolution.flex_sweep_report
```

---

## 실험 설계 (실험설계서_수정 기준)

**시나리오** — 단일 변수 통제

| 코드 | 외란 | 파라미터 |
|---|---|---|
| S0 | 없음 (대조군) | — |
| S1 | 부품 지연: head-op `part_available_time` 미래로 | ratio {10/20/40}% × k {0.5/1/2} |
| S2 | 긴급 주문: tight-due 작업 1개 삽입 | 납기계수 {0.3/0.5/1.0} |

**P1 / P2 / P3 변형** — 정보 노출 ablation

| 변형 | LLM이 보는 것 | 측정 대상 |
|---|---|---|
| P1 | 기본 변수만 | 외란 모르는 휴리스틱의 한계 |
| P2 | + 외란 변수 | 충격 인지의 marginal 가치 |
| P3 | + 경험 메모리 | 누적 학습의 추가 가치 |

**평가** — AT(평균 납기초과) primary · MIT / PTJ / ARI 부지표 · 100 seeds ·
paired comparison (공통 시드) · fitness ≡ 보고 지표

---

## 저장소 구조

```
heuristiX-v2/
├── src/                 Rust DES 엔진 (jssp/ sim/ rules/ scenarios.rs)
├── evolution/           LLM 진화 루프 + 배터리 + 비교 리포트
├── dashboard/           Flask 연구 워크벤치 (13 페이지)
│   ├── sim/             경량 Python 시뮬레이터 + DSL transpiler
│   ├── ui/              템플릿·정적 자원
│   └── seed_results/    v3 실측 데이터 (시드용, 추적됨)
├── data/brandimarte/    표준 FJSSP 벤치마크 로더
├── docs/                PIPELINE · PROMPTS 명세
└── runs/                실험 출력 (gitignored, 재생성 가능)
```

**두 시뮬레이터?** — 의도된 설계. Rust는 100-seed 배터리용 정밀 엔진(논문 수치의
출처), dashboard 내장 Python 시뮬레이터는 브라우저 인터랙션용 경량 엔진. LLM이
발견한 DSL 규칙은 `dashboard/sim/llm/dsl_translator.py`가 transpile해 양쪽에서 실행된다.

---

## 관련 연구 대비 위치

|  | 듀얼-LLM | 메모리 | 외란 ladder | FJSSP flex sweep | Ablation | 비용 공개 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| EoH (ICML'24) | — | — | — | — | — | — |
| SeEvo (T-FS'26) | — | — | — | ✓ | — | — |
| EvoDR ('26) | ✓ | — | △ | ✓ | — | — |
| ReasoningBank ('26) | — | ✓ | — | — | — | — |
| **heuristiX** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

상세: [`runs/comparison_synthesis.md`](runs/comparison_synthesis.md) ·
중간보고서: [`runs/interim_report.md`](runs/interim_report.md)

---

## Roadmap

- [ ] Multi-urgent S2 + urgent-AT 분리 지표 → RQ3 재검증
- [ ] Paired t-test — 통계적 유의성 명시
- [ ] Brandimarte Mk01–Mk10 실인스턴스 (로더 완료, 데이터 drop-in 대기)
- [ ] Flex sweep 세분화 (0.3 / 0.7) — ARI peak 정밀 위치
- [ ] Cross-scenario memory bank
- [ ] 모델 비교 (gpt-4.1 · Claude)

---

<div align="center">
<sub>창종설 연구 프로젝트 · 실험설계서_수정 §5-2/§8 구현 ·
EoH(Liu '24) × EvoDR(Qiu '26) × ReasoningBank(Ouyang '26) 계보</sub>
</div>
