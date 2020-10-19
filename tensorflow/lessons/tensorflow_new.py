import tensorflow as tf
import matplotlib.pyplot as plt
import random


# Generating the Dataset¶


def synthetic_data(w, b, num_examples):  # @save
    """Generate y = Xw + b + noise."""  # Y = w0 + w1 * X1 + w2 * X2 + ... + wN * XN   5000 usd = w1 * 6 year + w2 * 1.5L
    # = [w0 ... wN] x [x0 ... xN]^T, x0 == 1
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y


true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('X_shape:', features.shape)
print('y_shape:', labels.shape)

# The semicolon is for displaying the plot only
plt.scatter(features[:, 1], labels, 1)
plt.show()

# 3D-plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0], features[:, 1], labels)
plt.show()
print('features:', features[0],'\nlabel:', labels[0])

#Initializing Model Parameters
w = tf.Variable(tf.random.normal(shape=(2, 1),
                                 mean=0, stddev=0.01), trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

#Reading the Dataset
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices) # Ключевая часть алгоритма обучения на мини-батчах!
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)


# Batch optimization
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# Defining the Model
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return tf.matmul(X, w) + b


# Defining the Loss Function
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

# # Линейная регрессия имеет решение в замкнутой форме.
# # Поскольку ни одна из других моделей не может быть решена аналитически,
# # мы воспользуемся этой возможностью, чтобы протестировать первый рабочий пример
# # мини-пакетного стохастического градиентного спуска.
# #
# # На каждом этапе, используя один мини-батч, случайно выбранный из нашего набора данных,
# # мы будем оценивать градиент потерь относительно заданных нами параметров.
# # Далее мы обновим эти параметры в направлении, которое может уменьшить потери.
# # Следующий код применяет обновление стоха
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient Обновить параметры (w, b) ← (w, b) − ηg
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

# В каждую эпоху мы будем перебирать весь набор данных (используя функцию data_iter) один раз,
# пройдя через каждый пример в обучающем наборе данных (при условии, что количество примеров делится на размер пакета).
# Количество эпох num_epochs и скорость обучения lr являются гиперпараметрами,
# которые мы установили здесь на 3 и 0,03 соответственно.

print(f'error in estimating w: {true_w - tf.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')

print('---- Parameters of Linear Regression ----')
print('w      : ', w)
print('true_w : ', true_w)

print('b      : ', b)
print('true_b : ', true_b)




