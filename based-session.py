# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# -*- coding:utf8 -*-

# ------代码训练流程及目的------#
'''
通过 矩阵相乘 的例子，展示 两种形式使用会话控制Session
'''

from __future__ import print_function
import tensorflow as tf

matrix1 = tf.constant([[3, 3]]) # 1行2列的矩阵
matrix2 = tf.constant([[2],
                       [2]])    # 2行1列
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

# method 1
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

# method 2  自动关闭 session
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
