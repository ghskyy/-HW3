import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
import streamlit as st

st.title("3D Scatter Plot with Separating Hyperplane (Square Distribution for Y=0)")

# 添加滑桿來控制資料點數量和距離閾值
num_points = st.slider("Number of Points", min_value=100, max_value=1000, value=600, step=50)
threshold = st.slider("Square Side Length", min_value=1.0, max_value=20.0, value=10.0, step=0.1)

# 生成資料點 - 在不同範圍內生成點
np.random.seed(0)
x1 = np.random.uniform(-10, 10, num_points)
x2 = np.random.uniform(-10, 10, num_points)

# 設置方形的條件來決定點的分佈
# 使用方形邊界條件來產生 Y=0 的區域
condition1 = (np.abs(x1) < threshold / 2)
condition2 = (np.abs(x2) < threshold / 2)
Y = np.where(condition1 & condition2, 0, 1)  # 方形區域內的點為 Y=0，其餘為 Y=1

# 計算 x3 作為 x1 和 x2 的高斯函數
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# 訓練 SVM 模型
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# 創建 3D 圖形函數
def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製資料點
    ax.scatter(x1[Y == 0], x2[Y == 0], x3[Y == 0], c='blue', marker='o', s=10, alpha=0.7, label='Y=0')  # 藍點（方形分佈）
    ax.scatter(x1[Y == 1], x2[Y == 1], x3[Y == 1], c='red', marker='s', s=10, alpha=0.7, label='Y=1')   # 紅點

    # 創建網格平面並計算分離平面
    xx, yy = np.meshgrid(np.linspace(-10, 10, 30), np.linspace(-10, 10, 30))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)  # 半透明的分離平面

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('3D Scatter Plot with Y Color and Separating Hyperplane')
    ax.legend()
    
    return fig

st.pyplot(plot_3d())

