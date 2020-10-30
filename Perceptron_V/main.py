
from Perceptron_V import *

def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(batch_size).map(resize_fn))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #setting hyperparameters
    batch_size = 256
    num_inputs, num_hiddens, num_outputs, = 784, 256, 10
    num_epochs, lr = 10, 0.1
    loss = "cross-enthropy"
    af = "lrelu" #you may also select "relu", "identical", "sigmoid" or "lrelu" (leaky relu)

    #loading fashion-mnist dataset with seggregation on train and test samples
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # creation of Perceptron_V object with neuron network layout
    perceptron = Perceptron_V(num_inputs,num_hiddens,num_outputs)

    #training the perceptron on the training sample with validation on the test sample
    perceptron.train(train_iter, test_iter, loss, af, batch_size, num_epochs, lr)

    #predicting labels for the test sample
    perceptron.predict(test_iter)

    plt.show()


