import tensorflow as tf
# 载入plt
import matplotlib.pyplot as plt
import numpy as np




# 设置种子
np.random.seed(5)

# 采用等差数列
x_data = np.linspace(-1, 1, 100)

# y = 2x + 1 + 噪声 噪声维度与x_data 一支

y_data = 2 * x_data + 1.0 + np.random.random(*x_data.shape) * 0.4

plt.scatter(x_data, y_data)
plt.show()

# 定义占位符
x = tf.placeholder("float", name="x")
y = tf.placeholder("float", name="y")

# 定义模型 训练模型请求w和b 使得最小 w * x + b
def model(x, w, b):
    return tf.multiply(x, w) + b


# 构建线性函数斜率 变量w
w = tf.Variable(1.0, name="w0")
b = tf.Variable(0.0, name="b0")

# 定义预测值
pred = model(x, w, b)

# 设置迭代次数
train_epochs = 10
# 学习率
learning_rate = 0.05

# 定义损失函数
# 使用均方差
loss_function = tf.reduce_mean(tf.square(y-pred))

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)


sess = tf.Session()

# 初始化
init = tf.global_variables_initializer()
sess.run(init)

# 训练
step = 0 # 步数
loss_list = [] # 保存loss列表 损失值
display_step = 10 # 报告粒度

# 训练
for epoch in range(train_epochs):
    # 组合成为一维数组
    for xs, ys in zip(x_data, y_data):
        loss = sess.run([optimizer, loss_function], feed_dict={x:xs, y:ys})
        # 显示损失值
        loss_list.append(loss)
        step = step + 1
        if step % display_step == 0:
            print("第几轮" , (epoch + 1))
            print("第几次", step)
            print("损失值", format(loss))
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    plt.plot(x_data, w0temp * x_data + b0temp)

# 显示损失值
plt.plot(loss_list)
plt.plot(loss_list, "r+")
plt.show()

# 结果打印
print("w" , sess.run(w))
print("b", sess.run(b))

# 根据求出 w 和 b进行画图
plt.scatter(x_data, y_data, label = "Original data")
plt.plot(x_data, x_data * sess.run(w) + sess.run(b), label = "Fitted line", color="r", linewidth=3)
plt.legend(loc = 2)
plt.show()

# 预测结果
x_test = 3.21
predict = sess.run(pred, feed_dict={x:x_test})
print(predict)