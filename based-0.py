# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial


# ------代码训练流程及目的------#
'''
1. 先给出数据 x_data 和 y_data
2. 构建具有tensorflow结构的网络
3. 通过一步步的优化 weight 和 biase，结果同预先给出的值会非常接近
'''

#
# -*- coding:utf8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32) # 随机生成包含100个float32类型数（的数组）
y_data = x_data*0.1 + 0.3  # y=Weights * x + biase

### --- create tensorflow structure start ---- ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #参数变量 1维结构 数值大小在-1到1的数
biases = tf.Variable(tf.zeros([1])) # 参数变量 biases 1维结构 数值初始化为0

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) # 计算 预测值与实际值的差别
optimizer = tf.train.GradientDescentOptimizer(0.5) # 学习效率一般选取小于1的数
train = optimizer.minimize(loss) # 使用优化器GradientDescentOptimizer（梯度下降）减少误差
### ---- create tensorflow structure end ---- ###

sess = tf.Session() # session是执行命令控制语句，----- hh  类似main（）函数
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

# init = tf.initialize_all_variables() # 老版本代码

sess.run(init) # 非常重要，必须激活才能运行

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))