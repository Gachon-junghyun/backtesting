import yfinance as yf
import matplotlib.pyplot as plt

try:
    # 데이터 요청 기간을 3개월로 축소
    data = yf.download("TSLA", start="2024-01-01", end="2024-03-20", progress=False)
    
    if data.empty:
        print("데이터를 받아오지 못했습니다. 인터넷 연결을 확인해주세요.")
    else:
        print("데이터 크기:", data.shape)
        print("\n처음 5개 행:")
        print(data.head())
        
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Close Price')
        plt.title('Tesla 주가 추이')
        plt.xlabel('날짜')
        plt.ylabel('주가 (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

except Exception as e:
    print(f"오류가 발생했습니다: {str(e)}")