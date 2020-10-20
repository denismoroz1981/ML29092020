import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import random
import scipy.stats
import logging
import numpy as np

# Generating the Dataset¶


def synthetic_data(w, b, num_examples):   #@save
    """Generate y = Xw + b + noise."""    # Y = w0 + w1 * X1 + w2 * X2 + ... + wN * XN   5000 usd = w1 * 6 year + w2 * 1.5L
                                          # = [w0 ... wN] x [x0 ... xN]^T, x0 == 1
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y


#setting true values
true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#print('X_shape:', features.shape)
#print('y_shape:', labels.shape)
#print('features:', features[0],'\nlabel:', labels[0])

# The semicolon is for displaying the plot only
# plt.scatter(features[:, 1], labels, 1)
# plt.show()

# 3D-plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(features[:, 0], features[:, 1], labels)
# plt.show()
#

#Initializing Model Parameters
w = tf.Variable(tf.random.normal(shape=(2, 1),
                                 mean=0, stddev=0.01), trainable=True) #+-3 sigma = 99.7%
b = tf.Variable(tf.zeros(1), trainable=True)
#
#
#Reading the Dataset
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices) # Ключевая часть алгоритма обучения на мини-батчах!
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)




#for X, y in data_iter(batch_size, features, labels):
#    print(X, '\n', y)
#    break


# Defining the Model
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return tf.matmul(X, w) + b


# Defining the Loss Function
#Squared error
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

#Mean absolute error
def mae(y_hat, y):  #@save
    """Mean absolute error."""
    return tf.math.abs(y_hat - tf.reshape(y, y_hat.shape)) / 2


#
#
# # Линейная регрессия имеет решение в замкнутой форме.
# # Поскольку ни одна из других моделей не может быть решена аналитически,
# # мы воспользуемся этой возможностью, чтобы протестировать первый рабочий пример
# # мини-пакетного стохастического градиентного спуска.
# #
# # На каждом этапе, используя один мини-батч, случайно выбранный из нашего набора данных,
# # мы будем оценивать градиент потерь относительно заданных нами параметров.
# # Далее мы обновим эти параметры в направлении, которое может уменьшить потери.
# # Следующий код применяет обновление стохастического градиентного спуска по мини-батчу,
# # учитывая набор параметров, скорость обучения и размер пакета.
#
# # Размер шага обновления определяется скоростью обучения lr.
# # Поскольку потери рассчитываются как сумма по мини-партии примеров,
# # мы нормализуем размер нашего шага на размер партии (batch_size),
# # так что величина типичного размера шага не сильно зависит от нашего выбора размера партии.

def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)

#setting hyperparams
#lr = 0.03
#num_epochs = 5
#batch_size = 10

#net = linreg
#loss = squared_loss
#loss = mae


def train (net=linreg, loss=squared_loss, lr =0.03, num_epochs=3,batch_size=10):

    """Training model and plotting results"""

    #setting empty lists to store progress of loss and gradients
    loss_list, dw1_list, dw2_list, db_list = [],[],[],[]

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):

            with tf.GradientTape() as g:
                l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
            # Compute gradient on l with respect to [`w`, `b`]
            dw, db = g.gradient(l, [w, b])

            #adding chaging parameters to list for plotting
            loss_list.append(tf.math.reduce_sum(l)/batch_size)
            dw1_list.append(dw[0])
            dw2_list.append(dw[1])
            db_list.append(db)
            # Update parameters using their gradient Обновить параметры (w, b) ← (w, b) − ηg
            sgd([w, b], [dw, db], lr, batch_size)

        train_l = loss(net(features, w, b), labels)
        le_loss = float(tf.reduce_mean(train_l))
        print(f'epoch {epoch + 1}, loss {le_loss:f}')

    #plotting train progres
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(7,7))
    fig.suptitle(f"Train progress at lf = {loss.__name__}, bs = {batch_size}, lr = {lr}, ne = {num_epochs}")

    loss_list=np.array(loss_list).ravel()
    dw1_list=np.array(dw1_list).ravel()
    dw2_list=np.array(dw2_list).ravel()
    db_list=np.array(db_list).ravel()
    number_loss=len(loss_list)
    number_d=len(dw1_list)

    #Plotting loss
    ax1.set_title("Loss",fontsize=10, fontweight="bold")
    ax1.plot(range(number_loss),loss_list, color="r",label="loss")
    ax1.text(ax1.get_xlim()[1]*0.5, ax1.get_ylim()[1]*0.9,"Final loss = {:.6f}".format(le_loss))

    #Plotting gradients
    ax2.set_title("Gradients",fontsize=10, fontweight="bold")
    ax2.plot(range(number_d),dw1_list,color="b",label="dw1")
    ax2.plot(range(number_d),dw2_list,color="b",label="dw2")
    ax2.plot(range(number_d),db_list,color="g",label="db")
    ax2.text(ax2.get_xlim()[1] * 0.5, ax2.get_ylim()[1] * 0.8, "w1 = {:.6f}".format(w.numpy().item(0)))
    ax2.text(ax2.get_xlim()[1] * 0.5, ax2.get_ylim()[1] * 0.6, "w2 = {:.6f}".format(w.numpy().item(1)))
    ax2.text(ax2.get_xlim()[1] * 0.5, ax2.get_ylim()[1] * 0.4, "b = {:.6f}".format(b.numpy().item(0)))
    ax2.set_xlabel("batches")
    ax2.legend(loc="upper left")

    plt.savefig("figures/train"+f"_{loss.__name__}"+f"_lr_{lr}"+f"_ne_{num_epochs}"+f"_bs_{batch_size}"+".png")

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

if __name__ == '__main__':
    #we will vary hyperparametets to see train progress by batch
    #train(net=linreg, loss=squared_loss, lr =0.03, num_epochs=3,batch_size=10)
    #train(net=linreg, loss=squared_loss, lr =0.03, num_epochs=6,batch_size=10)
    #train(net=linreg, loss=squared_loss, lr =0.006, num_epochs=17,batch_size=10)
    #train(net=linreg, loss=squared_loss, lr =1, num_epochs=1,batch_size=10)
    #train(net=linreg, loss=squared_loss, lr =0.03, num_epochs=3,batch_size=5)
    #train(net=linreg, loss=squared_loss, lr =0.03, num_epochs=3,batch_size=1)
    train(net=linreg, loss=squared_loss, lr =0.03, num_epochs=40,batch_size=1000)
    #train(net=linreg, loss=mae, lr =0.03, num_epochs=6,batch_size=10)
    #train(net=linreg, loss=squared_loss, lr =0.05, num_epochs=2,batch_size=10)
