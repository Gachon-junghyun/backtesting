from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import io
import base64
from datetime import datetime

app = Flask(__name__)

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 폼 데이터에서 티커와 날짜 받아오기
        ticker = request.form.get('ticker', 'TSLA')  # 기본값으로 TSLA 설정
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        
        # 문자열을 datetime 객체로 변환
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # 데이터 불러오기
        try:
            print(f"데이터 다운로드 시작: {ticker}, {start_date}, {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)
            print(f"다운로드된 데이터:\n{data.head()}")
            print(f"데이터 크기: {len(data)}")
        except Exception as e:
            print(f"데이터 다운로드 중 오류 발생: {str(e)}")
            return jsonify({'error': f'데이터를 다운로드하는 중 오류가 발생했습니다: {str(e)}'}), 500
        
        if data.empty:
            return jsonify({'error': '해당 기간에 데이터가 없습니다. 다른 기간을 선택해주세요.'}), 400
        
        if len(data) < 14:  # RSI 계산에 필요한 최소 데이터 포인트
            return jsonify({'error': '선택한 기간이 너무 짧습니다. 최소 14일 이상의 기간을 선택해주세요.'}), 400
        
        # RSI 계산
        data['RSI'] = calculate_rsi(data)
        
        # 로컬 최고점과 최저점 찾기
        data['RSI_Max'] = data.iloc[argrelextrema(data['RSI'].values, np.greater_equal, order=10)[0]]['RSI']
        data['RSI_Min'] = data.iloc[argrelextrema(data['RSI'].values, np.less_equal, order=10)[0]]['RSI']
        
        plt.figure(figsize=(30, 10))
        
        # 종가 그래프
        plt.subplot(2, 1, 1)
        plt.plot(data['Close'], label='Close Price')
        plt.xlim([data['Close'].index[0], data['Close'].index[-1] + pd.DateOffset(days=100)])
        plt.title(f'{ticker} CLOSE and RSI')
        plt.legend()
        
        # RSI 그래프
        plt.subplot(2, 1, 2)
        plt.plot(data['RSI'], label='RSI', color='blue')
        plt.xlim([data['Close'].index[0], data['Close'].index[-1] + pd.DateOffset(days=100)])
        plt.scatter(data.index, data['RSI_Max'], color='red', label='RSI Max', marker='^', alpha=1)
        plt.scatter(data.index, data['RSI_Min'], color='green', label='RSI Min', marker='v', alpha=1)
        
        # RSI Max 연장선
        max_indices = data['RSI_Max'].dropna().index
        if len(max_indices) > 1:
            df_max = data.loc[max_indices]
            for i in range(1, len(df_max)):
                slope = (df_max['RSI_Max'].iloc[i] - df_max['RSI_Max'].iloc[i-1]) / ((df_max.index[i] - df_max.index[i-1]).days)
                end_x = df_max.index[i] + pd.Timedelta(days=100)
                end_y = df_max['RSI_Max'].iloc[i] + slope * 100
                plt.plot([df_max.index[i], end_x], [df_max['RSI_Max'].iloc[i], end_y], 'r--')
        
        # RSI Min 연장선
        min_indices = data['RSI_Min'].dropna().index
        if len(min_indices) > 1:
            df_min = data.loc[min_indices]
            for i in range(1, len(df_min)):
                slope = (df_min['RSI_Min'].iloc[i] - df_min['RSI_Min'].iloc[i-1]) / ((df_min.index[i] - df_min.index[i-1]).days)
                end_x = df_min.index[i] + pd.Timedelta(days=100)
                end_y = df_min['RSI_Min'].iloc[i] + slope * 100
                plt.plot([df_min.index[i], end_x], [df_min['RSI_Min'].iloc[i], end_y], 'g--')
        
        plt.axhline(70, color='r', linestyle='--', linewidth=1)
        plt.axhline(30, color='g', linestyle='--', linewidth=1)
        plt.legend()
        
        # 그래프를 base64로 인코딩
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({'graph': graph_data})
        
    except Exception as e:
        print(f"전체 프로세스 중 오류 발생: {str(e)}")
        return jsonify({'error': f'분석 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 