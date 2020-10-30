
import numpy
# сигмоид expit()
import scipy.special

import matplotlib.pyplot as plt


# описание класса нейронной сети
class neuralNetwork:

    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задание количества узлов входного, скрытого и выходного слоя
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # связь весовых матриц, wih и who
        # вес внутри массива w_i_j, где связь идет из узла i в узел j
        # следующего слоя
        # w11 w21
        # w12 w22 и т д7
        numpy.random.seed(42)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # уровень обучения
        self.lr = learningrate
        
        # функция активации - сигмоид
        self.sigmoid = lambda x: scipy.special.expit(x)
        self.activation_function = self.sigmoid
        #for loss progress storage
        self.loss_list = []
        #just for plotting loss
        self.loss_demo = []
        pass

        # обучение нейронной сети
        #we also transfer number of correct label and sample size to function
    def train(self, inputs_list, targets_list,label,i,n):
        # преобразование входного списка 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # вычисление сигналов на входе в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычисление сигналов на выходе из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # вычисление сигналов на входе в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        # ошибка на выходе (целевое значение - рассчитанное)
        output_errors = targets - final_outputs  #
        # распространение ошибки по узлам скрытого слоя
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # пересчет весов между скрытым и выходным слоем
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), # d_sigma() = sigma()*(1-sigma())
                                        numpy.transpose(hidden_outputs))
        
        # пересчет весов между входным и скрытым слоем
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), # d_sigma() = sigma()*(1-sigma())
                                        numpy.transpose(inputs))

        #loss calculation between correct label probability (0.99) and obtained targets
        err = targets[label]-final_outputs[label]
        #traking the loss
        self.loss_list.append(err)
        #we divide our train size by 10 batches to monitor average loss per batch at Terminal and on the graph
        if not i%(n//10):
            err_mean = sum(self.loss_list[i-(n//10):i])/len(self.loss_list[i-(n//10):i])
            #to monitor at Terminal
            print(f"Step {i}, error {err_mean}")
            #for plotting
            self.loss_demo.append(err_mean)

        pass

    
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

    #plotting loss
    def plot_train(self, score):
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

        ax1.set_title(f"Loss progress @ sigmoid, hidden nodes = {hidden_nodes}, "
                      f"epochs = {epochs}, lr = {learning_rate}")
        ax1.plot(range(len(self.loss_demo)),self.loss_demo)
        ax1.set_ylabel("average loss")
        ax1.set_xlabel("iterations")
        ax1.text(ax1.get_xlim()[1] * 0.5, ax1.get_ylim()[1] * 0.9, "Performance = {:.4f}".format(score))
        ax1.grid()
        #saving plot
        plt.savefig("figures/train" + f"_af_sigmoid" + f"_hn_{hidden_nodes}"
                    + f"_e_{epochs}" + f"_lr_{learning_rate}" + ".png")
# Задание архитектуры сети:
# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 300
output_nodes = 10

# уровень обучения
learning_rate = 0.05

# создание экземпляра класса нейронной сети
n = neuralNetwork(input_nodes,
                  hidden_nodes,
                  output_nodes,
                  learning_rate)


# Загрузка тренировочного набора данных
training_data_file = open("mnist\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# Обучение нейронной сети
# количество эпох
epochs = 3
#for monitoring and plotting as we divide our sample into 10 batches to calculate average loss per batch
i = 1

for e in range(epochs):
    # итерирование по всем записям обучающего набора
    for record in training_data_list:
        # разделение записей по запятым ','
        all_values = record.split(',')
        # масштабирование и сдвиг исходных данных
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создание целевых  выходов
        targets = numpy.zeros(output_nodes) + 0.01
        # элемент all_values[0] является целевым для этой записи
        targets[int(all_values[0])] = 0.99
        #call train function with additional params - i and sample size
        n.train(inputs, targets,int(all_values[0]),i,len(training_data_list)*epochs)
        i+=1
        pass
    pass


# Загрузка тестового набора данных
test_data_file = open("mnist\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Тестирование нейронной сети

# Создание пустого накопителя для оценки качества
scorecard = []

# итерирование по тестовому набору данных
for record in test_data_list:
    # разделение записей по запятым ','
    all_values = record.split(',')
    # правильный ответ - в первой ячейке
    correct_label = int(all_values[0])
    # масштабирование и сдвиг исходных данных
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # получение ответа от нейронной сети
    outputs = n.query(inputs)
    # получение выхода
    label = numpy.argmax(outputs)
    # добавление в список единицы, если ответ совпал с целевым значением
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    
    pass

# расчет точности классификатора
scorecard_array = numpy.asarray(scorecard)
score = scorecard_array.sum()/scorecard_array.size
print("performance = ", score)
#transfer performance demonstrated on test samle to show on the graph
n.plot_train(score)

