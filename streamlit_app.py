# streamlit_app.py

# ✅ 반드시 첫 번째 줄에 위치해야 함
import streamlit as st
st.set_page_config(page_title="📈 비트코인 예측", layout="centered")

# 📌 나머지 import
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib import font_manager
from utils.predict import predict_lstm_price
import traceback

# ✅ 한글 폰트 설정 (배포 환경에서도 안전하게)
try:
    font_path = "assets/micross.ttf"  # 프로젝트 내 폰트 (micross.ttf 등)
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.family'] = font_name
        print(f"✅ [LOG] 폰트 적용 완료: {font_name}")
    else:
        print("⚠️ [LOG] 폰트 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"⚠️ [LOG] 폰트 설정 실패: {e}")

matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ Streamlit UI
st.title("📈 비트코인 단기 예측 (LSTM)")
st.markdown("실시간 바이비트 데이터를 기반으로 LSTM으로 다음 가격을 예측합니다.")

# ✅ 사용 방법 안내
with st.expander("ℹ️ 프로그램 사용 방법 보기"):
    st.markdown("""
    1. **예측 시간 단위**: 몇 분/시간 간격의 데이터를 기반으로 예측할지 선택합니다.
       - `5m`: 5분봉
       - `15m`: 15분봉
       - `1h`: 1시간봉
       - `4h`: 4시간봉
    2. **예측 스텝 수**: 몇 단계 뒤의 가격을 예측할지 슬라이더로 선택하세요. `1`이면 바로 다음 데이터 포인트를 예측합니다.
    3. **[예측 시작] 버튼 클릭**: 선택한 설정으로 LSTM 모델이 작동해 다음 가격을 예측하고 시각화합니다.
    """)

# ✅ 사용자 입력
interval = st.selectbox("예측 시간 단위", ['5m', '15m', '1h', '4h'])
steps = st.slider("몇 스텝 후 가격을 예측할까요?", 1, 5, 1)

# ✅ 세션 상태 초기화
if "previous_prediction" not in st.session_state:
    st.session_state["previous_prediction"] = None

# ✅ 실행 버튼
if st.button("예측 시작"):
    st.write("⏳ 예측 중입니다...")

    try:
        predicted, last_price, df = predict_lstm_price(interval=interval, steps=steps)

        if predicted is None:
            st.error("❌ 예측 실패. 서버 로그를 확인하세요.")
        else:
            # ✅ 날짜 범위 표시
            st.write("🗓️ 데이터 기간:", df.index.min(), "~", df.index.max())

            # ✅ 이전 예측과 비교
            previous = st.session_state["previous_prediction"]
            st.session_state["previous_prediction"] = predicted

            if previous is not None:
                if abs(predicted - previous) < 1e-6:
                    st.warning("⚠️ 예측 결과는 이전과 동일합니다.")
                else:
                    st.success("✅ 예측이 새로 업데이트되었습니다!")

            # ✅ 변화량 계산
            diff = predicted - last_price
            diff_percent = (diff / last_price) * 100
            is_up = diff > 0

            color = "green" if is_up else "red"
            emoji = "📈" if is_up else "📉"
            direction = "상승" if is_up else "하락"

            # ✅ 텍스트 출력
            st.success(f"✅ 현재 가격: {last_price:.2f} USDT")
            st.info(f"🔮 예측 가격: {predicted:.2f} USDT")
            st.markdown(
                f"<h3 style='color:{color}'>{emoji} {direction} 예측: {abs(diff):.2f} USDT ({abs(diff_percent):.2f}%)</h3>",
                unsafe_allow_html=True
            )

            # ✅ 그래프 시각화
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index[-100:], df['close'][-100:], label='실제 가격', linewidth=2)
            ax.plot(df.index[-1], last_price, 'bo', label='현재 가격')
            ax.plot(df.index[-1], predicted, 'o', color=color, label='예측 가격', markersize=9)
            ax.axhline(y=predicted, color=color, linestyle='--', label='예측 수평선')
            ax.fill_between(df.index[-5:], last_price, predicted, color=color, alpha=0.2)

            # ✅ 날짜 포맷
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            fig.autofmt_xdate()

            ax.set_title(f'BTCUSDT {interval} 예측 결과', fontsize=14)
            ax.set_xlabel('시간')
            ax.set_ylabel('가격 (USDT)')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            st.pyplot(fig)

    except Exception as e:
        st.error("🚨 예측 도중 오류가 발생했습니다. 콘솔 로그와 아래 에러 메시지를 확인하세요.")
        st.code(traceback.format_exc())
    