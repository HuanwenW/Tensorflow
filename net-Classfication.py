# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# -*- coding:utf8 -*-

# ------代码训练流程及目的------#
"""
分类学习

同 回归学习 的区别：在于输出变量类型上

定量 输出是回归，或者说是连续变量预测 （eg. 预测房价是回归任务）
定性 输出是分类，或者说是离散变量预测 （eg. 把东西分成几类，比如 猫、狗、猴）

"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # 加载mnist模块
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 数据包含55000张训练图片，每张图片分辨率28*28

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

# 计算准确度的功能
def compute_accuracy(v_xs, v_ys):
    global prediction # 设为全局变量
    y_pre = sess.run(prediction, feed_dict={xs: v_xs}) # 得到预测值-- 即1行10列的概率值，（10列代表10个类，取最大的概率【0-1之间】对应的位置类）
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) # 预测值 同 真实值（为1的位置所属的类）做比较，看是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 计算预测的均值；tf.cast数据类型转换
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network 搭建神经网络
xs = tf.placeholder(tf.float32, [None, 784]) # 不规定sample，但规定sample大小为28x28
ys = tf.placeholder(tf.float32, [None, 10]) # 每张图片都表示一个数字，所以输出是数字0到9，共10类

# add output layer 只有输入层和输出层的简单网络
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data 交叉熵（cross_entropy）函数--用来衡量预测值和真实值的相似程度，若完全相同，交叉熵为0
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

#
for i in range(10000): # 随着循环（训练）次数的增加，准确率越高
# 从下载好的mnist中提取出部分数据训练
    batch_xs, batch_ys = mnist.train.next_batch(100) # 每次取100张图片训练，免得数据太多训练的慢
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 500 == 0: # 每训练50次输出一下预测精度
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels)) # 测试集计算准确度