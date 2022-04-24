# @Time : 2022/4/24 13:23 

# @Author : Yüehi

# @File : t_with_mynsgaii.py 

# @Software: PyCharm

# coding:utf-8
from myNSGAII import ZDT1, MyNSGAII, DTLZ1

ga = MyNSGAII(100, 50, 20, 20)
ga.load_problem(DTLZ1())
ga.run()