import random

import tensorflow as tf
# 载入plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/boston.csv", header=0)

# 显示摘要信息
print(df.describe())

# 转换成为np
df = np.array(df)
# 归一化
for i in range(12):
    df[:, i] = df[:, i] / (df[:, i].max() - df[:, i].min())

# 取前12列特征数据  x
x_data = df[:, :12]



# 最后一列为标签数据 y
y_data = df[:, 12]


# 定义占位符
# 行不明确 列明确 实际训练进行数据带入
x = tf.placeholder(tf.float32, [None, 12], name="X")
y = tf.placeholder(tf.float32, [None, 1], name="Y")

# 定义模型
# 定义命名空间
with tf.name_scope("Model"):
    # w 初始化为shape=(12,1) 矩阵的随机数 列向量
    w = tf.Variable(tf.random_normal([12,1], stddev=0.01), name="W")
    b = tf.Variable(1.0, name="b")

    # w 和 x 矩阵叉乘 带入行向量 行向量和列向量进行叉乘法，1 × 12  12 × 1 得出 1 × 1 矩阵 得到的是结果 y y的矩阵为 1 × 1
    # Y = x1xw1 + x2xw2 + ....   转化为矩阵 12 × 1 行向量 叉乘 列向量 1 × 12 得到1 × 1 向量
    def model(x, w, b):
        return tf.matmul(x, w) + b

    # 预测计算结果
    pred = model(x, w, b)

# 超参数
# 迭代次数
train_epochs = 50
# 学习率
learning_rate = 0.001

# 定义损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred, 2))

# 创建优化器 使用梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 声明会话
sess = tf.Session()

# 定义初始化
init = tf.global_variables_initializer()

#设置日志
logdir = "log"
# 创建操作
sum_loss_op = tf.summary.scalar("loss", loss_function)
# 写入
merged = tf.summary.merge_all()


# 启动会话
sess.run(init)

# 写入
writer = tf.summary.FileWriter(logdir, sess.graph)

# 保存loss list
loss_list =  []
loss_list1 = []


# 训练
for epoch in range(train_epochs):
    # 记录损失
    loss_sum = 0.0

    # 数据合并
    for xs, ys in zip(x_data, y_data):
        # 标量 转 向量
        # 1 * 12
        xs = xs.reshape(1,12)
        # 1 * 1
        ys = ys.reshape(1,1)

        # 数据填入
        loss = sess.run([optimizer, sum_loss_op, loss_function], feed_dict={x:xs, y:ys})

        loss_sum = loss_sum + loss[2]


        writer.add_summary(loss[1], epoch)



        loss_list1.append(loss[1])

    # 再次学习
    seed = random.randint(1,100)
    random.seed(seed)
    random.shuffle(x_data)
    random.seed(seed)
    random.shuffle(y_data)

    # 得到b 和 w
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    # 记录平均
    loss_average = loss_sum / len(y_data)

    # 保存进入列表
    loss_list.append(loss_average)

    print("第几轮   ", epoch + 1)
    print("损失值   ", loss_average)
    print("b   ", b0temp)
    print("w   ", w0temp)

# 出图
plt.plot(loss_list)
plt.show()
plt.plot(loss_list1)
plt.show()



# 模型预测
n = np.random.randint(506)
print(n)
x_test = x_data[n]
x_test = x_test.reshape(1,12)
predict = sess.run(pred, feed_dict={x:x_test})
print("预测值", predict)
target = y_data[n]
print("标签值", target)

