import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from IPython import display


class Updater():  #@save
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


class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(15, 5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib."""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

    def use_svg_display(self):  # @save
        """Use the svg format to display a plot in Jupyter."""
        display.set_matplotlib_formats('svg')

    def set_figsize(self, figsize=(4, 4)):  # @save
        """Set the figure size for matplotlib."""
        self.use_svg_display()
        plt.rcParams['figure.figsize'] = figsize


def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(batch_size).map(resize_fn))


def train_epoch(net, train_iter, loss, updater, num_inputs, W1, b1, W2, b2):  #@save
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X, num_inputs=num_inputs, W1=W1, b1=b1, W2=W2, b2=b2)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]



def train(net, train_iter, test_iter, loss, num_epochs, updater, num_inputs, W1, b1, W2, b2):  #@save
    """Train a model"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater, num_inputs, W1, b1, W2, b2)
        test_acc = evaluate_accuracy(net, test_iter, num_inputs, W1, b1, W2, b2)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def evaluate_accuracy(net, data_iter, num_inputs, W1, b1, W2, b2):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X, num_inputs, W1, b1, W2, b2), y), tf.size(y).numpy())
    return metric[0] / metric[1]


def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))


def relu(X):
    return tf.math.maximum(X, 0)


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





def net(X, num_inputs, W1, b1, W2, b2):
    X = tf.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2


def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)


def predict(net, test_iter, num_inputs, W1, b1, W2, b2, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(tf.argmax(net(X, num_inputs, W1, b1, W2, b2), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()


def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
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


batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs, num_hiddens, num_outputs,  = 784, 256, 10

W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))

W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.01))

num_epochs, lr = 10, 0.1
updater = Updater([W1, W2, b1, b2], lr)
params = [W1, b1, W2, b2]

train(net, train_iter, test_iter, loss, num_epochs, updater, num_inputs, W1, b1, W2, b2)

predict(net, test_iter, num_inputs, W1, b1, W2, b2)
plt.show()
