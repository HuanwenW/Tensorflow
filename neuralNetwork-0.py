# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# -*- coding:utf8 -*-

# ------代码训练流程及目的------#
'''
回归学习

构建一个神经网络，通常包括：输入层、隐藏层和输出层（本网络构建的是————输入层=1，隐藏层=10，输出层=1）

构建真实数据（及x和y的表达式）--定义神经网络层结构（隐藏层、输出层）--计算loss--训练优化--session激活训练--目的：使得loss越来越小，与真实数据更接近

优化器
tf.train.GradientDescentOptimizer 梯度下降
tf.train.AdadeltaOptimizer
tf.train.AdagradDAOptimizer
tf.train.AdamOptimizer     常用
tf.train.FtrlOptimizer
tf.train.MomentumOptimizer 常用
tf.train.RMSPropOptimizer  Alpha Go用的就是这个

'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data

x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis] #生成300个-1到1之间数值，再转换为300X1的维度
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

##plt.scatter(x_data, y_data)
##plt.show()

# define placeholder for inputs to network 利用占位符定义我们所需要的神经网络的输入
xs = tf.placeholder(tf.float32, [None, 1]) # 占位符类型；None代表无论输入多少都可以；因为输入只有一个特征，所以这里是1
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer 隐藏层可以自己假设
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) # 调用add_layer函数，激励函数用relu
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data 取方差后再取平均值计算loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 以0.1的效率来最小化误差loss

# important step
sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement 输出误差loss
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))