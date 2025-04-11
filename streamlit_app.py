# streamlit_app.py (Streamlit Cloud 호환 및 예외 로그 표시 포함)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from utils.predict import predict_lstm_price
import traceback
from matplotlib import font_manager
import os
import matplotlib.dates as mdates   


# 프로젝트 경로에 있는 폰트 설정
font_path = "assets/micross.ttf"
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['font.family'] = font_name
else:
    print("❗ 폰트 파일 없음: 기본 폰트로 표시됩니다.")

st.set_page_config(page_title="📈 비트코인 예측", layout="centered")

# ✅ Streamlit 화면 UI
st.title("📈 비트코인 단기 예측 (LSTM)")
st.markdown("실시간 바이비트 데이터를 기반으로 LSTM으로 다음 가격을 예측합니다.")

interval = st.selectbox("예측 시간 단위", ['5m', '15m', '1h', '4h'])
steps = st.slider("몇 스텝 후 가격을 예측할까요?", 1, 5, 1)

if st.button("예측 시작"):
    st.write("⏳ 예측 중입니다...")
    try:
        predicted, last_price, df = predict_lstm_price(interval=interval, steps=steps)

        if predicted is None:
            st.error("❌ 예측 실패. 모델 또는 데이터 오류가 있습니다.")
        else:
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

            st.markdown(f"<h3 style='color:{color}'>{emoji} {direction} 예측: {abs(diff):.2f} USDT ({abs(diff_percent):.2f}%)</h3>", unsafe_allow_html=True)

            # ✅ 그래프 출력
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['close'][-100:], label='실제 가격', linewidth=2)
            ax.plot(df.index[-1], last_price, 'bo', label='현재 가격')
            ax.plot(df.index[-1], predicted, 'o', color=color, label='예측 가격', markersize=9)
            ax.axhline(y=predicted, color=color, linestyle='--', label='예측 수평선')
            ax.fill_between(df.index[-5:], last_price, predicted, color=color, alpha=0.2)
            ax.set_title(f'BTCUSDT {interval} 예측 결과', fontsize=14)
            ax.set_xlabel('시간')
            ax.set_ylabel('가격 (USDT)')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"🔥 예외 발생: {str(e)}")
        st.code(traceback.format_exc())
