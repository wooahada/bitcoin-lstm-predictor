import matplotlib
print("📦 현재 matplotlib 백엔드:", matplotlib.get_backend())
matplotlib.use('tkagg')  # ✅ 첫 줄에 위치

from utils.predict import predict_lstm_price, plot_prediction

def test():
    print("🔍 테스트 시작...")

    predicted, last_price, df = predict_lstm_price(interval='15m', steps=1)

    if predicted is None:
        print("❌ 예측 실패. 콘솔 로그를 확인하세요.")
    else:
        print(f"📈 현재 가격: {last_price:.2f} USDT")
        print(f"🔮 예측 가격: {predicted:.2f} USDT")
        plot_prediction(df, predicted, interval='15m')

if __name__ == "__main__":
    print("🔥 FILE EXECUTED")
    test()