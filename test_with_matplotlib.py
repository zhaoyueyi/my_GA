import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

figure = plt.figure()
ax = Axes3D(figure)  # 设置图像为三维格式
X = np.arange(-1, 1, 0.01)
Y = np.arange(-1, 1, 0.01)  # X,Y的范围
X, Y = np.meshgrid(X, Y)  # 绘制网格
s = (X, Y)
# Z = (np.sin(X) * np.sin(Y)) / (X * Y)  # f(x,y)=(sin(x)*sin(y))/(x*y),注意括号
# Z = s[0] ** 2 + s[1] ** 2

# Z = 100 * ((Y - X ** 2) ** 2) + (X - 1) ** 2
# Z = 1/(1 + (s[0] + s[1] + 1) ** 2 + (19 - 14 * s[0] + 3 * s[0] ** 2 - 14 * s[1] + 6 * s[0] + s[1])) \
#              * ((30 + 2 * s[0] - 3 * s[1] ** 2) ** 2
#                 * (18 - 32 * s[0] + 12 * s[0] ** 2 + 48 * s[1] - 36 * s[0] * s[1] + 27 * s[1]) + 1)
# Z = (s[0] ** 2 + s[1] - 11) ** 2 + (s[0] + s[1] ** 2 - 7) ** 2
# Z = 4 * s[0] ** 2 + 2.1 * s[0] ** 4 + s[0] ** 6 / 3 + s[0] * s[1] - 4 * s[1] ** 2 + 4 * s[1] ** 4
Z = s[0] ** 2 + 2 * s[1] ** 2 - 0.3 * np.cos(3 * np.pi * s[0]) - 0.4 * np.cos(4 * np.pi * s[1] + 0.7)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# 绘制3D图，后面的参数为调节图像的格式
plt.show()  # 展示图片
