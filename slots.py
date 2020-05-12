#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
from prettytable import PrettyTable

t1 = time.time()
# ==============================基础数据==============================
SPIN_NUM = 10000000  # 模拟次数
REEL_NUM = 5  # 几轴
ROW_NUM = 3  # 几行
ELEMENT_NUM = 10  # 图标个数，图标id命名为0~ELEMENT_NUM-1，0默认为wild
BET = 9
PAY_TABLE = np.array([  # pay_table，和图标id相对应
    [0, 10, 100, 1000, 10000],  # wild
    [0, 5, 50, 250, 1000],
    [0, 4, 40, 200, 500],
    [0, 3, 30, 150, 300],
    [0, 0, 25, 100, 200],
    [0, 0, 20, 75, 150],
    [0, 0, 15, 50, 100],
    [0, 0, 10, 30, 80],
    [0, 0, 5, 20, 60],
    [0, 0, 4, 15, 50],
])
LINE = np.array([  # 赢钱线
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2],
    [0, 1, 2, 1, 0],
    [2, 1, 0, 1, 2],
    [1, 0, 0, 0, 1],
    [1, 2, 2, 2, 1],
    [0, 0, 1, 2, 2],
    [2, 2, 1, 0, 0],
])
REEL_SET = [  # 卷轴
    [0,1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9],
]
# =================================pay_table初始化==================================
'''
pay_array，实质是一个 N*N*N*N*N 的5维数组
pay_array[index1][index2][index3][index4][index5]的值代表了 一条连线index1-5对应的赔率是多少
假设：   3个wild 10倍，4个wild 100倍，5个wild 1000倍，wild图标定义为0
那么：   pay_array[0][0][0][0][0]=1000
        pay_array[0][0][0][0][任意非0]=100
        pay_array[0][0][0][任意非0][任意]=10
'''
pay_array_size = [ELEMENT_NUM for i in range(REEL_NUM)]
pay_array = np.zeros(pay_array_size, dtype=np.int)
any_element = list(range(ELEMENT_NUM))  # 代表任意元素
# 遍历设置 每个元素element，若干连击hits(3连4连5连..)，对应的赢钱数
# 倒序设置，保证wild最后设置
for hits in range(1,REEL_NUM+1):  # hits代表几连的图标
    for element in range(ELEMENT_NUM - 1, -1, -1):

        pay_num = PAY_TABLE[element][hits-1]
        if pay_num > 0:
            # wild+图标组合 可以同样赢钱
            win_element = [0, element]
            # 具体一条线的赢钱组合，连续若干个赢钱图标+若干个任意图标
            element_list = [win_element for i in range(hits)] + [any_element for i in range(REEL_NUM - hits)]
            '''
            np.ix_函数，能把多个一维数组 转换为 一个用于选取方形区域的索引器
            利用np.ix_批量将 某赢钱图标 345连的赢钱情况及对应的赢钱倍数设置好
            先设置3连，再设置4连，再设置5连，这样4连的情况会覆盖原本一部分3连的情况，也正好符合设计
            '''
            pay_array[np.ix_(*element_list)] = pay_num
# 因为每一轴的卷轴长度不一样，所以这一块不能用矩阵乘法，只能用for循环
for i in range(REEL_NUM):
    # 卷轴长度
    length = len(REEL_SET[i])
    # 每轴转化成np.array
    now_reel = np.array(REEL_SET[i])
    # 将原本长度x的一维卷轴,转换成 ROW_NUM*x 大小的矩阵，为了方便后续进行取随机结果的运算
    reel_array = np.empty((ROW_NUM, length), dtype=np.int)
    for j in range(ROW_NUM):
        # 第j个为 错位j格
        reel_array[j] = np.append(now_reel[j:], now_reel[:j])

    # 随机SPIN_NUM个随机下标，通过随机下标映射到卷轴，取出随机结果
    random_index = np.random.randint(0, length, size=SPIN_NUM)
    # result代表 ROW_NUM*SPIN_NUM的随机结果
    result = reel_array[:, random_index]
    # 将每一轴的result合并，变成（ROW_NUM*REEL_NUM）*SPIN_NUM的结果，把盘面化成1维方便取各条线的结果
    if i == 0:
        total_result = result
    else:
        total_result = np.append(total_result, result, axis=0)
# 由于盘面转换成一维方便计算，LINE也要相应处理
for i in range(REEL_NUM):
    LINE[:, i] += i * ROW_NUM
# 根据LINE得到每一条赢钱线的结果
result_line = total_result[LINE, :]
# 把所有线的每个图标 按照下标索引的方式带入到pay_table矩阵中，得到每条线的赢钱结果
win_line = pay_array[
    result_line[:, 0, :], result_line[:, 1, :], result_line[:, 2, :], result_line[:, 3, :], result_line[:, 4, :]]
# 求和，得到每次spin的总赢钱倍数
win_total = win_line.sum(axis=0)/ BET

# ==================统计数据=============================
win_rate = win_total.sum() / SPIN_NUM
n = win_total[win_total == 0]
hit_rate = 1 - n.size / SPIN_NUM
std_dev = np.sqrt(np.square(win_total - win_rate ).sum() / SPIN_NUM)

base_table = PrettyTable()
base_table.add_row(["spin_num", SPIN_NUM])
base_table.add_row(["win_rate", win_rate])
base_table.add_row(["hit_rate", hit_rate])
base_table.add_row(["std_dev", std_dev])
base_table.align = "l"
base_table.header = False
print(base_table)

# ===================赢钱分布========================
win_table = PrettyTable(['win_multi', 'hit_rate', 'win_rate'])
multi_range = [0, 1, 5, 10, 20, 50, 100, 200, 500]
for i in range(len(multi_range)):
    if i == 0:
        n = win_total[(win_total > 0) & (win_total < multi_range[0])]
        title = "(0," + str(multi_range[0]) + ")"
    elif i == len(multi_range) - 1:
        n = win_total[(win_total >= multi_range[-1])]
        title = "[" + str(multi_range[-1]) + ",)"
    else:
        n = win_total[(win_total >= multi_range[i]) & (win_total < multi_range[i + 1])]
        title = "[" + str(multi_range[i]) + "," + str(multi_range[i + 1]) + ")"

    win_table.add_row([title, n.size / SPIN_NUM, n.sum() / SPIN_NUM ])
win_table.align = "l"
print(win_table)

t2 = time.time()
print("cost_time:", t2 - t1)
