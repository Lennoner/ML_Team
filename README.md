# 시세선 — 중고차 적정가 분석

한동대학교 ML 팀프로젝트 · 러닝(Running) 머신팀 · 2026

매물 정보를 입력하면 **AI가 적정가를 예측**하고, 시세 대비 편차를 분석해
**허위매물·사기 위험도를 5단계로 진단**하는 웹 애플리케이션입니다.

> **100% 정적 사이트입니다.** 서버·데이터베이스·백엔드가 전혀 필요 없습니다.
> Ridge + XGBoost 모델(1,200 트리)을 `model.json`으로 추출하고, 추론 로직을
> 브라우저 JavaScript(`engine.js`)로 재현했습니다. 사용자의 브라우저 안에서
> 모델이 직접 돌아가므로 항상 켜져 있고, 호스팅 비용이 들지 않습니다.

---

## 파일 구성

| 파일 | 용도 |
|---|---|
| `index.html` | 메인 웹페이지 (UI + 분석 로직) |
| `engine.js` | 브라우저용 추론 엔진 (Ridge + XGBoost 재현) |
| `model.json` | 추출된 모델 (계수·트리·임계값·카테고리, 약 2.9MB) |

이 세 파일이 전부입니다. **셋을 같은 폴더에 두기만 하면 동작합니다.**

---

## 로컬에서 실행

`index.html`을 더블클릭하면 `model.json`을 `fetch`할 때 CORS 정책에 막힙니다.
간단한 로컬 서버로 열어주세요.

```bash
# 이 폴더에서
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

또는 VS Code의 "Live Server" 확장을 사용해도 됩니다.

---

## 배포 (셋 중 하나 선택)

### A. Vercel (가장 쉬움 · 추천)

1. [vercel.com](https://vercel.com) 가입 (GitHub 계정 연동)
2. 이 폴더를 GitHub 저장소에 push
3. Vercel에서 **Add New → Project → 저장소 선택 → Deploy**
4. 빌드 설정 불필요 (정적 파일 자동 감지). 끝.

또는 CLI:
```bash
npm i -g vercel
vercel        # 프롬프트에서 Enter만 누르면 배포
vercel --prod # 프로덕션 배포
```

### B. Netlify

1. [netlify.com](https://app.netlify.com/drop) 접속
2. **이 폴더를 통째로 드래그&드롭** → 즉시 배포 URL 발급

또는 GitHub 연동 후 자동 배포 (빌드 명령 없음, publish 디렉토리 = 루트).

### C. GitHub Pages

1. 이 폴더를 GitHub 저장소에 push
2. 저장소 **Settings → Pages → Source: Deploy from a branch → main / (root)**
3. 몇 분 뒤 `https://<사용자명>.github.io/<저장소명>/` 에서 접속

> ⚠️ GitHub Pages는 큰 JSON도 잘 서빙하지만, 저장소에 `model.json`(약 2.9MB)이
> 포함되어야 합니다. Git LFS는 필요 없습니다.

---

## 모델 성능

| 지표 | 값 |
|---|---|
| R² (원본 스케일) | **0.933** |
| R² (log 스케일) | 0.955 |
| MAPE (평균 절대 오차율) | **11.1%** |
| RMSE | 358만원 |
| MAE | 213만원 |
| 학습 데이터 | 35,698 대 (전체 44,623 중 80%) |
| 가격대 | 240 ~ 7,350만원 |
| 모델 | Ridge + XGBoost Residual Learning |

---

## 위험도 분류 (5단계)

학습 데이터 잔차 분포 기반 비대칭 임계값:

| 분류 | 편차율 | 의미 |
|---|---|---|
| 🚨 Danger (저가) | ≤ -16.7% | 허위매물·사기 강한 의심 |
| ⚠️ Warning (저가) | -16.7% ~ -8.3% | 저렴, 확인 권장 |
| ✅ Fair | -8.3% ~ +8.3% | 정상 시세 |
| ⚠️ Warning (고가) | +8.3% ~ +16.7% | 다소 비쌈 |
| 🚨 Danger (고가) | ≥ +16.7% | 과도하게 비쌈 |

---

## 시연용 예시 케이스

페이지 하단의 프리셋 버튼으로 바로 체험 가능:

| 시나리오 | 입력값 | 결과 |
|---|---|---|
| ✅ 정상 매물 | 현대 / 팰리세이드 / 2021 / 72,000km / 2,690만 | Fair (위험 3%) |
| 🚨 저가 의심 | 기아 / 카니발 4세대 / 2021 / 78,000km / 1,200만 | Danger 저가 (위험 100%) |
| 🚨 고가 의심 | 제네시스 / G80 (RG3) / 2021 / 80,000km / 6,000만 | Danger 고가 (위험 100%) |

---

## 모델을 다시 추출하려면

데이터나 하이퍼파라미터를 바꿔 모델을 재학습한 경우, `export_model.py`로
`model.json`을 다시 생성하면 됩니다. (원본 프로젝트의 `detector.py`,
`cleaned_data_v2.csv` 필요)

```bash
python3 export_model.py   # detector_retrained.joblib → model.json
```

---

*본 결과는 통계적 추정이며 실제 매물 거래 시 직접 점검을 권장합니다.*
