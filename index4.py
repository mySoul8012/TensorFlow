import random
import tensorflow as tf
# 载入plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/39324970_2.csv", header=0)

# 显示摘要信息
print(df.describe())

# 转换成为np
df = np.array(df)


# 取前11 38列特征数据  x
x_data = df[:,11:38]
print(x_data[233][26])



# 最后一列为标签数据 y
y_data = df[:, 8:11]
print(y_data[233][2])


plt.show()



# 定义占位符
# 行不明确 列明确 实际训练进行数据带入
x = tf.placeholder(tf.float32, [None, 27], name="X")
y = tf.placeholder(tf.float32, [None, 3], name="Y")

# 定义模型
# 定义命名空间
with tf.name_scope("Model"):
    # w 初始化为shape=(27,3) 矩阵的随机数 列向量
    w = tf.Variable(tf.random_normal([27,3], stddev=0.01), name="W")
    b = tf.Variable(1.0, name="b")


    def model(x, w, b):
        print(w)
        return tf.matmul(x, w) + b

    # 预测计算结果
    pred = model(x, w, b)

# 超参数
# 迭代次数
train_epochs = 50
# 学习率
learning_rate = 0.01

# 定义损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred, 2))

# 创建优化器 使用梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 声明会话
sess = tf.Session()

# 定义初始化
init = tf.global_variables_initializer()

# 启动会话
sess.run(init)

# 保存loss
loss_list = []
loss_list11 = []

# 训练
for epoch in range(train_epochs):
    # 记录损失
    loss_sum = 0.0

    # 数据合并
    for xs, ys in zip(x_data, y_data):
        # 标量 转 向量
        # 1 * 27
        print(xs)
        xs = xs.reshape(1,27)
        # 1 * 3
        ys = ys.reshape(1,3)

        # 数据填入
        loss = sess.run([optimizer, loss_function], feed_dict={x:xs, y:ys})


        loss_sum = loss_sum + loss[1]

        loss_list11.append(loss[1])


    # 得到b 和 w
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    # 记录平均
    loss_average = loss_sum / len(y_data)

    # 再次学习
    seed = random.randint(1,100)
    random.seed(seed)
    random.shuffle(x_data)
    random.seed(seed)
    random.shuffle(y_data)

    loss_list.append(loss_average)

    print("第几轮   ", epoch + 1)
    print("损失值   ", loss_average)
    print("b   ", b0temp)
    print("w   ", w0temp)


plt.plot(loss_list)
plt.show()

plt.plot(loss_list11)
plt.show()




# 模型预测
n = np.random.randint(234)
print(n)
x_test = x_data[n]
x_test = x_test.reshape(1,27)
predict = sess.run(pred, feed_dict={x:x_test})
print("预测值", predict)
target = y_data[n]
print("标签值", target)


