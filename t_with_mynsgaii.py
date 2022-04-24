# @Time : 2022/4/24 13:23 

# @Author : Yüehi

# @File : t_with_mynsgaii.py 

# @Software: PyCharm

# coding:utf-8
from myNSGAII import ZDT1, ZDT2, ZDT3, MyNSGAII, DTLZ1, DTLZ5, DTLZ4

ga = MyNSGAII(20, 20, 20, 20)
ga.load_problem(ZDT1())
ga.run()
ga.load_problem(ZDT2())
ga.run()
ga.load_problem(ZDT3())
ga.run()
ga.load_problem(DTLZ1())
ga.run()
ga.load_problem(DTLZ4())
ga.run()
ga.load_problem(DTLZ5())
ga.run()