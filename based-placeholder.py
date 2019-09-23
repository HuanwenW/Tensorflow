
# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# -*- coding:utf8 -*-

# ------代码训练流程及目的------#
'''
placeholder(占位符)的作用：暂时存储变量
Tensorflow 如果想从外部传入data，那就需要用到tf.placeholder(),然后以这种形式传输数据sess.run(****,feed_dict={input:****})
placeholder 与 feed_dict={} 是绑定在一起出现的！
'''

from __future__ import print_function
import tensorflow as tf

input1 = tf.placeholder(tf.float32) # 定义占位符及类型
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2) # 7*2 =14

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]})) # 传入的值以放在字典里的形式传入