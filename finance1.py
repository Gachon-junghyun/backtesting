import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

class StockPlotter:
    def __init__(self):
        self.points = []
        self.lines = []
        self.is_drawing = False
        
    def plot_tesla_stock(self):
        # Tesla 주식 데이터 가져오기
        tesla = yf.Ticker("TSLA")
        
        # 최근 1년간의 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # 주가 데이터 다운로드
        self.df = tesla.history(start=start_date, end=end_date)
        
        # 그래프 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.plot(self.df.index, self.df['Close'], label='종가', color='blue')
        
        # 그래프 꾸미기
        self.ax.set_title('Tesla (TSLA) 주가 차트 - 최근 1년\n(선을 그리려면 두 번 클릭하세요)')
        self.ax.set_xlabel('날짜')
        self.ax.set_ylabel('주가 (USD)')
        self.ax.grid(True)
        self.ax.legend()
        
        # x축 날짜 포맷 설정
        self.fig.autofmt_xdate()
        
        # 마우스 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 그래프 표시
        plt.show()
    
    def on_click(self, event):
        if event.inaxes is not None:
            if not self.is_drawing:
                # 첫 번째 클릭: 시작점 설정
                self.points = [(event.xdata, event.ydata)]
                self.is_drawing = True
            else:
                # 두 번째 클릭: 끝점 설정 및 선 그리기
                self.points.append((event.xdata, event.ydata))
                
                # 이전 선 제거
                for line in self.lines:
                    line.remove()
                self.lines.clear()
                
                # 새로운 선 그리기
                x_coords = [p[0] for p in self.points]
                y_coords = [p[1] for p in self.points]
                line, = self.ax.plot(x_coords, y_coords, 'r-', linewidth=2)
                self.lines.append(line)
                
                # 그래프 업데이트
                self.fig.canvas.draw()
                
                # 상태 초기화
                self.is_drawing = False
                self.points = []
                

if __name__ == '__main__':
    plotter = StockPlotter()
    plotter.plot_tesla_stock()
