import numpy as np
import gzip
import pickle as cPickle
import GA
import mnistLoader as data
import time

#load data
f = gzip.open('data/mnist.pkl.gz', 'rb')
fTemp = cPickle._Unpickler(f)
fTemp.encoding = 'latin1'
train_set, validation_set, test_set = fTemp.load()
f.close()

classes = list(set(train_set[1]))
nr_inputs = len(train_set[0][0])
nr_training_units = len(train_set[0][:5000])
nr_output = len(classes)

#rules of thumb applications
nr_max_neurons = int(nr_inputs/4)
nr_max_hidden = min(int(nr_training_units/(nr_max_neurons)),10)
nr_max_hidden = max(nr_max_hidden,1)

train_set, validation_set, test_set = data.load_data_wrapper()

pop_size = 40
crossover_rate = 0.5
mutation_rate = 0.05
nr_bits_learning_rate = 10
nr_bits_regularization_param = 10
start = time.time()
myGA = GA.genetic_algorithm(train_set, validation_set, nr_max_hidden, nr_max_neurons, nr_output, nr_inputs,
                            nr_bits_learning_rate, nr_bits_regularization_param,
                            pop_size, crossover_rate, mutation_rate)
myGA.minimal_accuracy_gain_slope = 1.5
precision = 0.1
while time.time() - start < 3600 * 17: # 17 hours of run
    myGA.nn_population = []
    # train and select neural networks, also best 10% are saved in best_nn and their genotype is
    # reused for next population
    selected_pop = myGA.selection()
    myGA.crossover_operation(selected_pop) # will complete the new population
    myGA.mutation_operation() # randomly mutate some individuals
    myGA.population = myGA.new_population
    myGA.minimal_accuracy_gain_slope=max(myGA.minimal_accuracy_gain_slope-precision,1)
file_name_inc = 1
for bestNN in myGA.best_nn:
    bestNN.save('Trained_network'+str(file_name_inc)+'.txt')
    file_name_inc+=1



#
