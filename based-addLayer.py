# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# -*- coding:utf8 -*-

# ------代码训练流程及目的------#
'''
add_layer 传入最基本的4个参数 ：输入值、输入值大小、输出值大小、激活函数
经过一系列初始化及定义的计算规则后，返回输出值
'''
from __future__ import print_function
import tensorflow as tf

# 设定默认的激励函数是None
def add_layer(inputs, in_size, out_size, activation_function=None): # 有4个参数： 输入值、输入/出值大小、激励函数

# 两个参数初始化
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) # Weights为一个n_size行 out_size列的随机变量矩阵--因为在生成初始参数时，random_normal（随机变量）会比全为0好，
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # 机器学习中，biases的推荐值不为0，此处在0向量基础上+0.1

# 计算规则
    Wx_plus_b = tf.matmul(inputs, Weights) + biases # 神经网络 未激活 的值 先存储在 Wx_plus_b 中

# 激活函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
# 返回输出值
    return outputs