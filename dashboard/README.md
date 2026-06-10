# FJSSP Research Workbench

FJSSP(Flexible Job Shop Scheduling Problem) 시뮬레이션 및 LLM 기반 규칙 진화 연구 플랫폼.

## 실행 방법

```bash
# 1. 패키지 설치
pip install flask numpy scikit-learn openai fpdf2 python-docx

# 2. 환경변수 설정 (.env 파일 생성)
echo OPENAI_API_KEY=sk-... > .env
echo OPENAI_MODEL=gpt-4o-mini >> .env

# 3. 벤치마크 데이터 다운로드
git clone https://github.com/SchedulingLab/fjsp-instances.git fjsp-instances

# 4. 서버 실행
python app.py

# 5. 브라우저에서 접속
# http://localhost:5000
```

## 사용 흐름

1. **벤치마크 선택** → `/benchmark-manager`에서 Brandimarte/Hurink/Behnke 중 선택
2. **시나리오 설정 & 실험** → `/scenario-manager`에서 S0/S1/S2 구성 및 B1~B10 평가
3. **결과 분석** → `/simulation-results`에서 AT/PTJ/MIT/Makespan 비교
4. **AI 규칙 진화** → `/evolution-center`에서 P1/P2/P3 방식 LLM 진화 실행
5. **인간 vs AI** → `/human-vs-ai`에서 기본 규칙 vs AI 생성 규칙 비교
6. **보고서 생성** → `/report-generator`에서 MD/PDF/DOCX 내보내기

## 시나리오 종류

| 코드 | 이름 | 설명 |
|------|------|------|
| S0 | 정상 운영 | 외부 교란 없음 |
| S1 | 부품 지연 | 일정 비율 작업의 부품이 k배 지연 |
| S2 | 긴급 주문 | 시뮬레이션 중 긴급 작업 삽입 |

## 디스패칭 규칙 (B1~B10)

| ID | 이름 | 핵심 변수 |
|----|------|---------|
| B1 | FIFO | release_time |
| B2 | EDD | due_date |
| B3 | SPT | processing_time |
| B4 | CR | CR = (d-t)/remaining_pt |
| B5 | Urgency | urgent_order_flag |
| B6 | PT+WINQ+SL | PT, WINQ, Slack |
| B7 | CR+SPT | CR, SPT |
| B8 | AT+RPT | release_time + remaining_pt |
| B9 | PDDR | machine_utilization |
| B10 | ATCS | ATCS 공식 |

## 구조

```
sim/        # 시뮬레이션 엔진, 시나리오, 규칙, LLM 진화
ui/         # Flask 템플릿 및 정적 파일
app.py      # Flask 메인 애플리케이션
results/    # 실험 결과 저장 (자동 생성)
benchmarks/ # 사용자 업로드 벤치마크
```
