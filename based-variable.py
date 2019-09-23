# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# -*- coding:utf8 -*-

# ------代码训练流程及目的------#
'''
variable(变量)的使用：只有定义为变量才能当变量用（用法不同于python）
通过 变量 和 常量 相加运算，理解变量的用法及使用session的流程
'''
from __future__ import print_function
import tensorflow as tf

state = tf.Variable(0, name='counter') # 定义state为变量
#print(state.name)
one = tf.constant(1) # 定义常量

new_value = tf.add(state, one) # 此句代码：仅仅是定义加法步骤，并没有直接计算
update = tf.assign(state, new_value)

# 只要定义变量，就要记得变量的初始化！非常重要！！
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # 初始化所有变量
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
