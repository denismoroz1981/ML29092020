import numpy
# сигмоид expit()
import scipy.special
#to get current time as a key of dict
import datetime

#we will store weights matrixes in the complex dict
#common dict type does not allow such an assignment - dict["key1"]["key2"] = x
#so special user-defined dict type Vividict created
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

# описание класса нейронной сети
class neuralNetwork:

    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задание количества узлов входного, скрытого и выходного слоя
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #creating dict for weights storage
        self.weights_data = Vividict()

        # связь весовых матриц, wih и who
        # вес внутри массива w_i_j, где связь идет из узла i в узел j
        # следующего слоя
        # w11 w21
        # w12 w22 и т д7
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # уровень обучения
        self.lr = learningrate

        # функция активации - сигмоид
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #function which is called when needed to store current states of weight matrixes
    def save_weights(self):
        #current time is used as a key for our dict with weights
        #so we have the following dict data structure for traking weights state when needed
        # { 1603390164.517186 {
        #                     "wih":
        #                     {
        #                      array([[ w0, w1 ... wn]])
        #                      }
        #                      "who":
        #                     {
        #                      array([[ w0, w1 ... wn]])
        #                      }
        #                     }


        current_time = (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
        self.weights_data[current_time]["wih"] = self.wih
        self.weights_data[current_time]["who"] = self.who

    #getter to get weights matrixes
    def get_weights(self):
        return self.weights_data

    # запрос к нейронной сети
    def query(self, inputs_list):
        # преобразование входного списка 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # вычисление сигналов на входе в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычисление сигналов на выходе из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычисление сигналов на входе в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    #example of usage of "save-weights" function, we increase weights (w1 and w2) in weights matrixes by 100
    #and then save weights
    def increment(self):
        self.wih += 100
        self.save_weights()


if __name__ == '__main__':
    # Задание архитектуры сети:
    # количество входных, скрытых и выходных узлов
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # уровень обучения
    learning_rate = 0.1


    # создание экземпляра класса нейронной сети
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    #for example we increase weights in weights matrixes by 100 and save weights
    n.increment()

    #printing weights saved after increment
    print(n.get_weights())

