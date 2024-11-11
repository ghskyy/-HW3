import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
import streamlit as st

st.title("3D Scatter Plot with Separating Hyperplane")

# 添加滑桿來控制資料點數量和距離閾值
num_points = st.slider("Number of Points", min_value=100, max_value=1000, value=600, step=50)
threshold = st.slider("Distance Threshold", min_value=0.1, max_value=10.0, value=6.0, step=0.1)

# 生成資料點
np.random.seed(0)
x1 = np.random.uniform(-10, 10, num_points)
x2 = np.random.uniform(-10, 10, num_points)

# 計算距離並根據閾值分配標籤
distances = np.sqrt(x1**2 + x2**2)
Y = np.where(distances < threshold, 0, 1)

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
    ax.scatter(x1[Y == 0], x2[Y == 0], x3[Y == 0], c='blue', marker='o', s=10, alpha=0.7, label='Y=0')  # 藍點
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
