import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def analyze_future_price_changes(symbol, days_ahead=7, volume_threshold=None):
    # 주식 데이터 가져오기
    stock = yf.Ticker(symbol)
    
    # 최근 2년간의 데이터 가져오기 (충분한 데이터를 위해)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # 주가 데이터 다운로드
    df = stock.history(start=start_date, end=end_date)
    
    # 거래량 기준 설정 (기본값: 평균 거래량의 1.2배)
    if volume_threshold is None:
        volume_threshold = df['Volume'].mean() * 1.2
    
    # 결과를 저장할 리스트
    price_changes = []
    dates = []
    volumes = []
    
    # 각 거래일마다 15일 후의 가격 변화 계산
    for i in range(len(df) - days_ahead):
        # 거래량이 기준 이상일 때만 분석
        if df['Volume'].iloc[i] >= volume_threshold:
            current_price = df['Close'].iloc[i]
            future_price = df['Close'].iloc[i + days_ahead]
            price_change = ((future_price - current_price) / current_price) * 100
            price_changes.append(price_change)
            dates.append(df.index[i])
            volumes.append(df['Volume'].iloc[i])
    
    if not price_changes:
        print(f"\n{symbol} 주식에서 거래량 {volume_threshold:,.0f} 이상인 거래가 없습니다.")
        return None
    
    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame({
        'Date': dates,
        'Price_Change_Percent': price_changes,
        'Volume': volumes
    })
    
    # 통계 계산
    mean_change = np.mean(price_changes)
    std_change = np.std(price_changes)
    max_change = np.max(price_changes)
    min_change = np.min(price_changes)
    avg_volume = np.mean(volumes)
    
    # 결과 출력
    print(f"\n{symbol} 주식의 {days_ahead}일 후 가격 변동 분석 (거래량 {volume_threshold:,.0f} 이상):")
    print(f"분석된 거래 수: {len(price_changes)}")
    print(f"평균 변동률: {mean_change:.2f}%")
    print(f"표준편차: {std_change:.2f}%")
    print(f"최대 상승률: {max_change:.2f}%")
    print(f"최대 하락률: {min_change:.2f}%")
    print(f"평균 거래량: {avg_volume:,.0f}")
    
    return results_df, mean_change, std_change, len(price_changes)

def plot_comparison_grid(symbol, days_ahead=7):
    # 주식 데이터 가져오기
    stock = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    df = stock.history(start=start_date, end=end_date)
    
    # 3x3 서브플롯 생성
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'{symbol} 주식의 거래량 임계값별 가격 변동 비교', fontsize=16)
    
    # 거래량 임계값 설정 (평균의 0.8배부터 2.0배까지)
    thresholds = [0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    mean_volume = df['Volume'].mean()
    
    for idx, threshold in enumerate(thresholds):
        volume_threshold = mean_volume * threshold
        results, mean_change, std_change, num_trades = analyze_future_price_changes(
            symbol, days_ahead, volume_threshold)
        
        if results is not None:
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # 히스토그램 그리기
            ax.hist(results['Price_Change_Percent'], bins=50, edgecolor='black')
            ax.set_title(f'거래량 임계값: {volume_threshold:,.0f}\n(평균의 {threshold:.1f}배)')
            ax.set_xlabel('가격 변동률 (%)')
            ax.set_ylabel('빈도')
            ax.grid(True)
            
            # 통계 정보 표시
            stats_text = f'거래 수: {num_trades}\n'
            stats_text += f'평균: {mean_change:.2f}%\n'
            stats_text += f'표준편차: {std_change:.2f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 3x3 비교 그래프 생성
    plot_comparison_grid('TSLA')
