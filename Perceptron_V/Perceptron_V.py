import tensorflow as tf
import numpy as np


#importing parts of code written in the outside files
from animator import *
from accumulator import *

class Perceptron_V:

    """Perseptron for image recognition"""

    def __init__(self,num_inputs,num_hiddens,num_outputs):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        tf.random.set_seed(54)
        self.W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
        self.b1 = tf.Variable(tf.zeros(num_hiddens))

        self.W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
        self.b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.01))

        self.batch_size = 1
        self.num_epochs = 1

        self.params = [self.W1, self.b1, self.W2, self.b2]

    class Updater():  # @save
        """For updating parameters using minibatch stochastic gradient descent."""

        def __init__(self, params, lr):
            self.params = params
            self.lr = lr

        def __call__(self, batch_size, grads):
            self.sgd(self.params, grads, self.lr, batch_size)

        def sgd(self, params, grads, lr, batch_size):  # @save
            """Minibatch stochastic gradient descent."""
            for param, grad in zip(params, grads):
                param.assign_sub(lr * grad / batch_size)



    def train_epoch(self,train_iter, updater):  #@save
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        for X, y in train_iter:
            # Compute gradients and update parameters
            with tf.GradientTape() as tape:
                y_hat = self.net(X)
                # Keras implementations for loss takes (labels, predictions)
                # instead of (predictions, labels) that users might implement
                # in this book, e.g. `cross_entropy` that we implemented above
                if isinstance(self.loss_f, tf.keras.losses.Loss):
                    l = self.loss_f(y, y_hat)
                else:
                    l = self.loss_f(y_hat, y)
            if isinstance(updater, tf.keras.optimizers.Optimizer):
                params = self.net.trainable_variables
                grads = tape.gradient(l, params)
                updater.apply_gradients(zip(grads, params))
            else:
                updater(X.shape[0], tape.gradient(l, updater.params))
            # Keras loss by default returns the average loss in a batch
            l_sum = l * float(tf.size(y)) if isinstance(
                self.loss_f, tf.keras.losses.Loss) else tf.reduce_sum(l)
            metric.add(l_sum, self.accuracy(y_hat, y), tf.size(y))
        # Return training loss and training accuracy
        return metric[0] / metric[2], metric[1] / metric[2]



    def train(self,train_iter, test_iter, loss="cross-enthropy", af="relu", batch_size=256, num_epochs=10, lr=0.1):  #@save
        """Train a model"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
        if loss == "cross-enthropy":
            self.loss_f = self.loss_cross

        if af == "relu":
            self.af=self.relu
        if af == "sigmoid":
            self.af = tf.math.sigmoid
        if af == "identical":
            self.af=self.identical
        if af == "lrelu":
            self.af = self.lrelu
        updater = self.Updater([self.W1, self.W2, self.b1, self.b2], lr)
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_iter, updater)
            test_acc = self.evaluate_accuracy(test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.51, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc


    def evaluate_accuracy(self, data_iter):  #@save
        """Compute the accuracy for a model on a dataset."""
        metric = Accumulator(2)  # No. of correct predictions, no. of predictions
        for _, (X, y) in enumerate(data_iter):
            metric.add(self.accuracy(self.net(X), y), tf.size(y).numpy())
        return metric[0] / metric[1]


    def accuracy(self,y_hat, y):  #@save
        """Compute the number of correct predictions."""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = tf.argmax(y_hat, axis=1)
        cmp = tf.cast(y_hat, y.dtype) == y
        return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))





    # # ReLU Function
    # #
    # # The most popular choice, due to both simplicity of implementation and its good performance
    # # on a variety of predictive tasks, is the rectified linear unit (ReLU).
    # # ReLU provides a very simple nonlinear transformation. Given an element x
    # # , the function is defined as the maximum of that element and 0:
    #
    # x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
    # y = tf.nn.relu(x)
    # plt.plot(x.numpy(), y.numpy(), 'x', 'relu(x)')
    # plt.show()
    #
    #
    #
    # with tf.GradientTape() as t:
    #     y = tf.nn.relu(x)
    # plt.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu')
    # plt.show()
    #
    # # Sigmoid Function
    #
    # y = tf.nn.sigmoid(x)
    # plt.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)')
    # plt.show()
    #
    # with tf.GradientTape() as t:
    #     y = tf.nn.sigmoid(x)
    # plt.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid')
    # plt.show()
    #
    # batch_size = 256
    #
    # y = tf.nn.tanh(x)
    # plt.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)')
    # plt.show()


    # relu activation function
    def relu(self,X):
        return tf.math.maximum(X, 0)


    def identical(self,X):
        """identical activation function"""
        return X


    def lrelu(self,X):
        """leaky relu activation function"""
        layer = tf.keras.layers.LeakyReLU(alpha=0.3)
        return layer(X)

    def net(self,X):
        """calculation of predicted results"""
        X = tf.reshape(X, (-1, self.num_inputs))
        H = self.af(tf.matmul(X, self.W1) + self.b1)
        return tf.matmul(H, self.W2) + self.b2


    def loss_cross(self,y_hat, y):
        """cross-enthropy loss function"""
        return tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)


    def predict(self, test_iter, n=6):  #@save
        """Predict labels (defined in Chapter 3)."""
        for X, y in test_iter:
            break
        trues = self.get_fashion_mnist_labels(y)
        preds = self.get_fashion_mnist_labels(tf.argmax(self.net(X), axis=1))
        titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
        self.show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
        plt.show()


    def get_fashion_mnist_labels(self,labels):  #@save
        """Return text labels for the Fashion-MNIST dataset."""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]


    def show_images(self,imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
        """Plot a list of images."""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            ax.imshow(np.asarray(img))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        return axes



