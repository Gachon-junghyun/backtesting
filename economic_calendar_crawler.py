import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

class EconomicCalendarCrawler:
    def __init__(self):
        self.base_url = "https://www.investing.com/economic-calendar/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_economic_calendar(self, date=None):
        try:
            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 경제 캘린더 테이블 찾기
            table = soup.find('table', {'id': 'economicCalendarData'})
            if not table:
                raise Exception("경제 캘린더 테이블을 찾을 수 없습니다.")

            # 데이터 추출
            events = []
            rows = table.find_all('tr')[1:]  # 헤더 제외
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 7:
                    event = {
                        '시간': cols[0].text.strip(),
                        '통화': cols[1].text.strip(),
                        '영향': cols[2].text.strip(),
                        '이벤트': cols[3].text.strip(),
                        '실제': cols[4].text.strip(),
                        '예측': cols[5].text.strip(),
                        '이전': cols[6].text.strip()
                    }
                    events.append(event)

            # DataFrame 생성
            df = pd.DataFrame(events)
            
            # 현재 시간을 파일명에 추가
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'economic_calendar_{current_time}.csv'
            
            # CSV 파일로 저장
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"데이터가 {filename}에 저장되었습니다.")
            
            return df

        except Exception as e:
            print(f"에러 발생: {str(e)}")
            return None

if __name__ == "__main__":
    crawler = EconomicCalendarCrawler()
    df = crawler.get_economic_calendar()
    if df is not None:
        print("\n크롤링된 데이터 미리보기:")
        print(df.head()) 