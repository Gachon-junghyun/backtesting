# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt

def create_heart():
    # 파라미터 방정식을 사용하여 하트 모양 생성
    t = np.linspace(0, 2*np.pi, 1000)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    
    # 그래프 그리기
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'r-', linewidth=2)
    plt.fill(x, y, 'red', alpha=0.3)
    plt.axis('equal')
    plt.axis('off')
    plt.title('하트 그래프')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_heart()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
