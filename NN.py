import numpy as np
import gzip
import pickle as cPickle
import random
import time


f = gzip.open(r'C:\Users\Jean\Desktop\Cuza info\S1\RN\mnist.pkl.gz', 'rb')
fTemp = cPickle._Unpickler(f)
fTemp.encoding = 'latin1'
train_set, valid_set, test_set = fTemp.load()
f.close()


# compute sigmoid nonlinearity
def sigmoid(x):
    # if x<-100000000:
    #     return 0
    # else:
    #     return [if i <-100000000 (1.0 / (1.0 + np.exp(-i))) else 0
    #             for i in x ]
    return (1.0 / (1.0 + np.exp(-x)))

# convert output of sigmoid function to its derivative
def deriv_sig(output):
        return output * (1 - output)

class neural_network(object):
    def __init__(self,connections,training_data, learning_rate, regularization_param):

        self.structure = connections  # pointers
        self.training_data = training_data
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param

        self.bias = [0] * len(self.structure)
        self.weights = [0] * len(self.structure)
        self.neurons_value = [0] * len(self.structure)
        self.delta_weights = [0] * len(self.structure)

        self.structure[0] = [0] * (len(training_data[0])-1)  # assign number of neurons to first layer

        for layer in range(1,len(self.structure)):  # for each layer (except input layer)

            self.bias[layer] =[0] * len(self.structure[layer])
            self.weights[layer] = [0] * len(self.structure[layer])
            self.neurons_value[layer] = [0] * len(self.structure[layer])
            self.delta_weights[layer] = [0] * len(self.structure[layer])
            for neuron in range(len(self.structure[layer])):  # for each neuron of current layer
                nr_connections = len(connections[layer][neuron])
                self.bias[layer][neuron] = np.random.randn()
                self.delta_weights[layer][neuron] = [0] * nr_connections
                # initialise all weigths enterring in current neuron
                self.weights[layer][neuron] = [0] * len(self.structure[layer-1])
                for active_con in connections[layer][neuron]:
                    self.weights[layer][neuron][active_con] = np.random.randn() * np.sqrt(2/nr_connections) #relu


    def compute_feed_forward(self, inputs_value):
        self.neurons_value[0] = inputs_value #copy by memory address
        for layer in range(1,len(self.structure)): # for each layer (except input layer)
            self.neurons_value[layer] = sigmoid(np.add([np.dot(self.neurons_value[layer-1], (self.weights[layer][neuron]))
                                         for neuron in range(len(self.structure[layer]))],self.bias[layer]))
    def update_mini_batch(self, mini_batch):
        for training_unit in mini_batch :  # for each training unit in the minibatch
            start = time.time()
            self.backprop(training_unit[:-1], training_unit[-1])
            stop = time.time()
            print('backprop')
            print (stop-start)
        # initialise the delta of each weigths (how much they will be modificated)
        for layer in range(1, len(self.structure)):  # for each layer except input layer
            for neuron in range(len(self.structure[layer])):  # for each neuron of current layer
                for weight in self.structure[layer][neuron]:  # for each connection of current neuron to previous layer
                    self.weights[layer][neuron][weight] = (1-self.learning_rate*(self.regularization_param/len(mini_batch)))*\
                                                          self.weights[layer][neuron][weight] -  self.learning_rate * self.delta_weights[layer][neuron][weight]

                    # reinitialise the delta of each weigths (how much they will be modificated for next minibatch)
                    self.delta_weights[layer][neuron][weight] = 0

    def backprop(self, inputs, target):

        # feedforward
        self.compute_feed_forward(inputs)
        #calculate softmax
        # nr_outputs = len(self.neurons_value[-1])
        # targets = [0] * nr_outputs
        # targets[target] = 1
        # new_delta = [0]*nr_outputs
        # sum_out = sum(self.neurons_value[-1])
        # for out_neuron in range(nr_outputs):
        #     self.neurons_value[-1][out_neuron] /= sum_out
        #     new_delta[out_neuron] = self.neurons_value[-1][out_neuron] - targets[out_neuron]

        # calculate cross entropy cost
        targets = [0] * len(self.neurons_value[-1])
        targets[int(target)] = 1
        error_total_respect_net_input = np.array(np.subtract(self.neurons_value[-1], targets))
        a = [np.transpose(self.neurons_value[-2]),]
        self.delta_weights[-1] = np.dot(np.transpose(error_total_respect_net_input), np.transpose(self.neurons_value[-2]) )
        for layer in range(2,len(self.structure)): # for each layer except input and output layer (starting from last hidden layer)
            error_total_respect_net_input = np.dot(np.transpose(self.delta_weights[-layer+1]), error_total_respect_net_input)
            self.delta_weights[-layer] = np.dot(error_total_respect_net_input, np.transpose(self.neurons_value[-layer-1]))


    def SGD(self, epochs, mini_batch_size,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """

        # early stopping functionality:
        best_accuracy=1

        n = len(self.training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(self.training_data) # shuffle datas to learn uniformly on all epochs
            mini_batches = [ #separe training datas in groups (minibatch)
                self.training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            print("Epoch %s training complete" % j)

            # if monitor_training_cost:
            #     cost = self.total_cost(training_data, lmbda)
            #     training_cost.append(cost)
            #     print("Cost on training data: {}".format(cost))
            # if monitor_training_accuracy:
            #     accuracy = self.accuracy(training_data, convert=True)
            #     training_accuracy.append(accuracy)
            #     print("Accuracy on training data: {} / {}".format(accuracy, n))
            # if monitor_evaluation_cost:
            #     cost = self.total_cost(evaluation_data, lmbda, convert=True)
            #     evaluation_cost.append(cost)
            #     print("Cost on evaluation data: {}".format(cost))
            # if monitor_evaluation_accuracy:
            #     accuracy = self.accuracy(evaluation_data)
            #     evaluation_accuracy.append(accuracy)
            #     print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if self.accuracy > best_accuracy:
                    best_accuracy = self.accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    # def total_cost(self, data, lmbda, convert=False):
    #     """Return the total cost for the data set ``data``.  The flag
    #     ``convert`` should be set to False if the data set is the
    #     training data (the usual case), and to True if the data set is
    #     the validation or test data.  See comments on the similar (but
    #     reversed) convention for the ``accuracy`` method, above.
    #     """
    #     cost = 0.0
    #     for x, y in data:
    #         a = self.feedforward(x)
    #         if convert: y = vectorized_result(y)
    #         cost += self.cost.fn(a, y)/len(data)
    #         cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
    #     return cost
    #
    # def save(self, filename):
    #     """Save the neural network to the file ``filename``."""
    #     data = {"sizes": self.sizes,
    #             "weights": [w.tolist() for w in self.weights],
    #             "biases": [b.tolist() for b in self.biases],
    #             "cost": str(self.cost.__name__)}
    #     f = open(filename, "w")
    #     json.dump(data, f)
    #     f.close()

    def save(self):
        f = open('savedNet.txt', 'w')
        for i in range(len(self.structure)-1): # for each layer
            for j in range(len(self.structure[i])): #for each neurone of current layer
                for k in range(len(self.structure[i+1])): #for each connection with next layer
                    f.write(str(self.structure[i][j][k]))
                    f.write(" ")
        for i in range(10):
            #f.write(str(b[i]))
            f.write(" ")
        f.close()



