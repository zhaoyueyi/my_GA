from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

figure = plt.figure()
ax = Axes3D(figure)  # 设置图像为三维格式
X = np.arange(-10.0, 10.0, 0.1)
Y = np.arange(-10.0, 10.0, 0.1)  # X,Y的范围
X, Y = np.meshgrid(X, Y)  # 绘制网格
s = (X, Y)
# Z = (np.sin(X) * np.sin(Y)) / (X * Y)  # f(x,y)=(sin(x)*sin(y))/(x*y),注意括号
# Z = 100 * ((Y - X ** 2) ** 2) + (X - 1) ** 2
# Z = 1/(1 + (s[0] + s[1] + 1) ** 2 + (19 - 14 * s[0] + 3 * s[0] ** 2 - 14 * s[1] + 6 * s[0] + s[1])) \
#              * ((30 + 2 * s[0] - 3 * s[1] ** 2) ** 2
#                 * (18 - 32 * s[0] + 12 * s[0] ** 2 + 48 * s[1] - 36 * s[0] * s[1] + 27 * s[1]) + 1)
Z = (s[0] ** 2 + s[1] - 11) ** 2 + (s[0] + s[1] ** 2 - 7) ** 2
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# 绘制3D图，后面的参数为调节图像的格式
plt.show()  # 展示图片
