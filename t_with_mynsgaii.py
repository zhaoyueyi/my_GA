# @Time : 2022/4/24 13:23 

# @Author : Yüehi

# @File : t_with_mynsgaii.py 

# @Software: PyCharm

# coding:utf-8
from myNSGAII import ZDT1, MyNSGAII

ga = MyNSGAII(100, 50, 20, 20)
prob = ZDT1()
ga.load_problem(prob)
ga.run()