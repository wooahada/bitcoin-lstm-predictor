# bitcoin
bitcoin predict

## 📘 비트코인 단기 예측 웹앱 (LSTM + PyTorch + Streamlit)

---

### 🧠 프로젝트 개요
• 실시간 **Bybit 비트코인 가격 데이터**를 기본으로
• **LSTM 단기 예측 모델**(PyTorch)을 통해 `5m`, `15m`, `1h`, `4h` 예측
• 아무리 도구에도 **Streamlit으로 다음 가격을 계산 결과 표시**

---

### 🔗 GitHub 프로젝트 사용법

> 📅 필요 조건
> - Python 3.8 이상
> - pip 또는 conda 환경 (venv 권장)

#### 📂 생성 & 실행
```bash
git clone https://github.com/yourusername/bitcoin-lstm-app.git
cd bitcoin-lstm-app

python -m venv venv
venv\Scripts\activate    # Windows

pip install -r requirements.txt

streamlit run streamlit_app.py
```

🚀 자동으로 http://localhost:8501 에서 홈 게시 가능

---

### 🔅 가상화 

| 할 수 있는 기능 | 설명 |
|------------------|--------|
| 📢 실시간 데이터 | Bybit에서 1000회 가격 수집 |
| 🌐 예측 시간단위 | 5m / 15m / 1h / 4h 선택 |
| 📊 예측 수 | 1~5 step (각 step은 시간과 같음) |
| 🧪 모델 | PyTorch LSTM 기본 |
| 📈 결과 구현 | 전체 가격 그래프 + 예측가격 선 |

---

### 📊 LSTM 모델 성능 검토 결과

| 시간단위 | MAE | RMSE | R² Score |
|------------|------|-------|------------|
| `5m`       | 326.58 | 453.23 | 0.8018 |
| `15m`      | 261.83 | 337.55 | 0.8814 |
| `1h`       | 966.01 | 1380.58 | 0.8292 |
| `4h`       | 2167.46 | 2658.86 | **0.9430** ✅ |

> 📆 **R² Score** 수치가 1에 가\uuae30 때 예측력이 높습니다.
> 다음 가격을 예측하는 모델 중 `4시간` 기본이 가장 적절한 결과를 보여주어요.

---

### 🔹 계층구조

```bash
bitcoin-lstm-app/
├── streamlit_app.py            # 웹앱 UI 배포
├── lstm_pytorch_trainer.py     # LSTM 학습용 파일
├── test_predict.py             # 예측 테스트 (console)
├── models/                     # .pt 파일 저장 포더
├── utils/
│   ├── bybit_api.py            # Bybit API에서 데이터 가져오기
│   ├── preprocess.py           # 전처리 모델 시간 연속 형식으로 변\uud658
│   └── predict.py              # 예측 결과 구현 함수
├── requirements.txt            # 필수 패키지 목록
```

---

### 🚀 사용하는 도움이 공부에 유용
- 단가의 Streamlit 웹가이드를 지원
- PyTorch로 가능한 시계열 LSTM 모델 구조
- 비트코인 가격 데이터와 무료 API 통화 연습
- 통계 구현, MAE/RMSE/R2 목표 평가 경험

---

### 🚫 패턴 / 오류
- 아무 데이터도 안 나오면 Bybit API 문제
- 결과가 표시되지 않으면 matplotlib font 문제
- streamlit_app.py에서 st.set_page_config() 메인에 걸면 ì \xec \xec\x84  \xec\x84\xa0 \xec\x84\xa0\uud574되어야 합니다.

---

### 🌟 가장 중요한 필요 명령
```bash
python lstm_pytorch_trainer.py   # 메인 LSTM 모델 학습
streamlit run streamlit_app.py   # 결과 웹에서 표시
```

---

### 🔗 GitHub 로고에 추가할 상의
```bash
https://github.com/yourusername/bitcoin-lstm-app
```

