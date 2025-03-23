import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.backend_bases import key_press_handler
import matplotlib.dates as mdates
from datetime import datetime

# 데이터 불러오기
ticker = 'spy'
start_date = '2024-01-01'
end_date = '2025-03-20'
data = yf.download(ticker, start=start_date, end=end_date)

# RSI 계산 함수
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 데이터에 RSI 추가
data['RSI'] = calculate_rsi(data)

# 로컬 최고점과 최저점 찾기
data['RSI_Max'] = data.iloc[argrelextrema(data['RSI'].values, np.greater_equal, order=10)[0]]['RSI']
data['RSI_Min'] = data.iloc[argrelextrema(data['RSI'].values, np.less_equal, order=10)[0]]['RSI']

# 그래프 그리기
fig = plt.figure(figsize=(30, 10))

# 종가 그래프
ax1 = plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Close Price')
plt.xlim([data['Close'].index[0], data['Close'].index[-1] + pd.DateOffset(days=100)])
plt.title(ticker + ' CLOSE and RSI')
plt.legend()

# RSI 그래프
ax2 = plt.subplot(2, 1, 2)
plt.plot(data['RSI'], label='RSI', color='blue')
plt.xlim([data['Close'].index[0], data['Close'].index[-1] + pd.DateOffset(days=100)])
plt.scatter(data.index, data['RSI_Max'], color='red', label='RSI Max', marker='^', alpha=1)
plt.scatter(data.index, data['RSI_Min'], color='green', label='RSI Min', marker='v', alpha=1)

# RSI Max 연장선
max_indices = data['RSI_Max'].dropna().index
df_max = data.loc[max_indices]
for i in range(1, len(df_max)):
    slope = (df_max['RSI_Max'].iloc[i] - df_max['RSI_Max'].iloc[i-1]) / ((df_max.index[i] - df_max.index[i-1]).days)
    end_x = df_max.index[i] + pd.Timedelta(days=100)
    end_y = df_max['RSI_Max'].iloc[i] + slope * 100
    plt.plot([df_max.index[i], end_x], [df_max['RSI_Max'].iloc[i], end_y], 'r--')

# RSI Min 연장선
min_indices = data['RSI_Min'].dropna().index
df_min = data.loc[min_indices]
for i in range(1, len(df_min)):
    slope = (df_min['RSI_Min'].iloc[i] - df_min['RSI_Min'].iloc[i-1]) / ((df_min.index[i] - df_min.index[i-1]).days)
    end_x = df_min.index[i] + pd.Timedelta(days=100)
    end_y = df_min['RSI_Min'].iloc[i] + slope * 100
    plt.plot([df_min.index[i], end_x], [df_min['RSI_Min'].iloc[i], end_y], 'g--')

plt.axhline(70, color='r', linestyle='--', linewidth=1)
plt.axhline(30, color='g', linestyle='--', linewidth=1)
plt.legend()

# 전역 변수로 수직선 저장
vertical_lines = []

# 마우스 클릭 이벤트 처리 함수
def onclick(event):
    if event.inaxes == ax2:  # RSI 그래프에서 클릭한 경우
        x_date = mdates.num2date(event.xdata)
        # matplotlib의 날짜를 pandas datetime으로 변환
        x_date = pd.Timestamp(x_date).tz_localize(None)
        # 가장 가까운 날짜 찾기
        closest_date = data.index[data.index.get_indexer([x_date], method='nearest')[0]]
        rsi_value = float(data.loc[closest_date, 'RSI'])
        close_value = float(data.loc[closest_date, 'Close'])
        
        # 이전 수직선 제거
        for line in vertical_lines:
            line.remove()
        vertical_lines.clear()
        
        # 새로운 수직선 그리기
        vline1 = ax1.axvline(x=closest_date, color='purple', linestyle='--', alpha=0.5)
        vline2 = ax2.axvline(x=closest_date, color='purple', linestyle='--', alpha=0.5)
        vertical_lines.extend([vline1, vline2])
        
        print(f'날짜: {closest_date.strftime("%Y-%m-%d")}')
        print(f'RSI: {rsi_value:.2f}')
        print(f'종가: {close_value:.2f}')
        print('-' * 50)
        
        # 그래프 업데이트
        plt.draw()

# 마우스 클릭 이벤트 연결
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
