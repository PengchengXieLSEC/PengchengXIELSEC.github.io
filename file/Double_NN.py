import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

x0 = np.linspace(50,55,50)[:,np.newaxis]
NF = len(x0)
for k in range(50):
    fx = 2*x0**2-2
    y = fx

    x = x0

    tf.compat.v1.disable_eager_execution()
    xs = tf.compat.v1.placeholder(tf.float32,[None,1])
    ys = tf.compat.v1.placeholder(tf.float32,[None,1])

    global weights,biases

    def add_layer(inputs,in_size,out_size,activation_function = None):
        global weights, biases

        weights = tf.Variable(tf.compat.v1.random_normal([in_size,out_size]))
        biases  = tf.Variable(tf.zeros([1,out_size])+0.1)
        Wx_plus_b = tf.matmul(inputs,weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    # 构建具有10个神经元的隐藏层
    h1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
    # 构建具有5个神经元的隐藏层
    h2 = add_layer(h1,10,5,activation_function = tf.nn.relu)
    # 构建具有一个神经元的输出层
    prediction = add_layer(h2 ,5,1,activation_function = None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),axis=[1]))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss) # 设置学习率为0.1

    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()                      # 创建会话
    record_loss = np.zeros(1000)             # 记录损失值变化
    sess.run(init)                           # 初始化所有变量
    for i in range(1000):
        sess.run(train_step,feed_dict={xs: x,ys: y})
        record_loss[i] = sess.run(loss,feed_dict={xs: x,ys: y})
    #以上得到拟合函数，下面求出其最小值点
    yy0 = min(y)
    flag = where(yy0==y)
    flag = flag[0][0]#迭代初始点索引
    xx0 = x0[flag]
    ###############
    all_vars = tf.compat.v1.trainable_variables()
    for v in all_vars:
        FLAG = all_vars.index(v)
        if FLAG == 0:
            weights_i_h1 = np.array(sess.run(v))[0]
        if FLAG == 1:
            biases_i_h1 = np.array(sess.run(v))[0]
        if FLAG == 2:
            weights_h1_h2 = np.array(sess.run(v))
        if FLAG == 3:
            biases_h1_h2 = np.array(sess.run(v))[0]
        if FLAG == 4:
            weights_h2_o = np.array(sess.run(v))
        if FLAG == 5:
            biases_h2_o = np.array(sess.run(v))[0]

    def rho(x):
        A=sess.run(tf.nn.relu(x* weights_i_h1.T + biases_i_h1))
        B=sess.run(tf.nn.relu(dot(A, weights_h1_h2) + biases_h1_h2))
        C=dot(B, weights_h2_o) + biases_h2_o
        return C

    begin = xx0[0]-2
    end = xx0[0]+2
    xx0 = np.linspace(begin, end, 20)[:, np.newaxis]
    ff0 = rho(xx0)
    xx0 = xx0[where(ff0 == min(ff0))[0][0]]

    ###############
    x0 = append(x0,xx0)
    x0 = x0.reshape(len(x0),1)
    NF = NF+1
