import numpy as np
import NNmodified as NN
import random
import copy


class genetic_algorithm(object):
    def __init__(self,train_set, validation_set, nr_max_hidden, nr_max_neurons, nr_outputs,
                 nr_inputs, nr_bits_learning_rate, nr_bits_regularization_param, pop_size,
                 crossover_rate, mutation_rate, elitism_rate = 0.1):

        # global values to calculate only once :
        self.index_begin_connections = nr_max_hidden + nr_max_hidden * nr_max_neurons
        self.nr_bits_learning_rate = nr_bits_learning_rate
        self.denominator_learning_rate = pow(2, nr_bits_learning_rate) * 2  # range learning rate = [0.000001, 0.5000001]
        self.nr_bits_regularization_param = nr_bits_regularization_param
        self.denominator_regularization_param = pow(2, nr_bits_regularization_param)  # range regularization param = [0.01, 0.51]


        self.nr_max_hidden = nr_max_hidden
        self.nr_max_neurons = nr_max_neurons
        self.nr_outputs = nr_outputs
        self.nr_inputs = nr_inputs
        self.population = []
        chromosome = [0] * 5
        for individual in range(pop_size):
            chromosome = [0] * 5
            chromosome[0] = np.random.choice([True, False], self.nr_max_hidden,0.35)
            chromosome[1] = np.random.choice([True, False], nr_max_hidden*nr_max_neurons,0.5)
            chromosome[2] = np.random.choice([True, False], nr_max_hidden * nr_max_neurons * nr_max_neurons + nr_max_neurons * nr_outputs, 0.65)
            chromosome[3] = np.random.choice([True, False], self.nr_bits_learning_rate,0.5)
            chromosome[4] = np.random.choice([True, False], self.nr_bits_regularization_param, 0.5)
            self.population.append(chromosome)

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.train_set = train_set
        self.validation_set = validation_set
        self.elitism_rate = elitism_rate
        self.nn_population = []

    def get_fitness(self, chromosome):
        connections = [[0]*self.nr_inputs]  # set connections of input layer empty (no connection before input layer)
        index_neurons = 0
        index_connections = 0
        first_hidden_layer_activated = False
        neurons_numeral_previous_layer = []
        neurons_numeral_cur_layer = []
        for cur_layer_index in range(len(chromosome[0])):  # for each bit corresponding to hidden layer presence
            if chromosome[0][cur_layer_index]:

                connections.append([])  # create layers list
                neuron_number = 0
                for cur_neuron_index in range(index_neurons, index_neurons+self.nr_max_neurons):  # for each bit corresponding to neuron presence in current hidden layer
                    index_end_connections = index_connections + self.nr_max_neurons
                    if chromosome[1][cur_neuron_index]:
                        neurons_numeral_cur_layer.append(neuron_number)
                        connections[-1].append([])  # create neurons list
                        connection_number=0
                        for connection in range(index_connections, index_end_connections):  # for each bit corresponding to connection presence toward current neuron
                            # verify existence of connection and neuron sender from previous layer
                            if chromosome[2][connection]:
                                if not first_hidden_layer_activated:
                            # connect neuron n° 'neuron' from current layer to neuron n° 'connection_number' from previous activatedlayer
                                    connections[-1][-1].append(connection_number)  # every neurons on input layer are existing
                                else:
                                    try:
                                        # connect neuron n° 'neuron' from current layer to neuron n° 'connection_number' from previous activatedlayer
                                        connections[-1][-1].append(neurons_numeral_previous_layer.index(connection_number))
                                    except:
                                        pass
                            connection_number += 1
                    index_connections = index_end_connections
                    neuron_number += 1
                neurons_numeral_previous_layer = list(neurons_numeral_cur_layer)
                neurons_numeral_cur_layer = []
                first_hidden_layer_activated = True

            index_neurons += self.nr_max_neurons

        # output layer connections
        if not first_hidden_layer_activated:
            return 0  # no hidden layer activated, not viable chromosome
        connections.append([])  # create layers list
        for output_neuron in range(self.nr_outputs):  # for each bit corresponding to neuron presence in current hidden layer
            index_end_connections = index_connections + self.nr_max_neurons
            connections[-1].append([])  # create neurons list
            connection_number = 0
            # for each bit corresponding to connection presence toward current neuron
            for connection in range(index_connections, index_end_connections):
                    # verify existence of connection and previous neuron
                    if chromosome[2][connection]:
                        try:
        # connect neuron n° 'neuron' from current layer to neuron n° 'connection_number' from previous activatedlayer
                            connections[-1][-1].append(neurons_numeral_previous_layer.index(connection_number))
                        except:
                            pass
                    connection_number += 1
            index_connections = index_end_connections

        learning_rate = 0
        power_of_two = self.denominator_learning_rate/4
        for bit_learning_rate in chromosome[3]:  # for last nr_bits_learning_rate bits of chr
            if bit_learning_rate:
                learning_rate += power_of_two
            power_of_two /= 2
        learning_rate /= self.denominator_learning_rate
        learning_rate+=0.000001

        regularization_param = 0
        power_of_two = self.denominator_regularization_param/4
        for bit_regularization_param in chromosome[4]:  # for last nr_bits_regularization_param bits of chr
            if bit_regularization_param:
                regularization_param += power_of_two
            power_of_two /= 2
        regularization_param /= self.denominator_regularization_param
        regularization_param += 0.01
        try:
            nn = NN.Network(connections)  # , self.train_set, learning_rate, regularization_param
            fitness = nn.SGD(self.train_set,1000,20, learning_rate, regularization_param, self.validation_set, True, 3,
                             self.minimal_accuracy_gain_slope)
            self.nn_population.append(nn)
            return fitness
        except:
            return 0

    def selection(self):
        all_fit = [self.get_fitness(individual) for individual in self.population]
        ordered_fit = sorted(all_fit)
        # remember indexes of corresponding individuals
        index_population=sorted(range(len(all_fit)), key=lambda k: all_fit[k])

        # save elits for next generation
        self.new_population = [copy.deepcopy(self.population[index_population[i]])
                               for i in range(1,int(len(self.population)*self.elitism_rate)+1)]
        self.best_nn = [self.nn_population[all_fit.index(ordered_fit[-i])]
                               for i in range(1,int(len(self.population)*self.elitism_rate)+1)]
        ordered_fit = np.divide(ordered_fit, sum(all_fit)) # make fitness on a probability form
        cumulativ_fit=0
        selected_pop = []
        for fit_index in range(len(ordered_fit)):
            if ordered_fit[fit_index]+cumulativ_fit > random.random():
                selected_pop.append(self.population[index_population[fit_index]])
            cumulativ_fit += ordered_fit[fit_index]
        return selected_pop

    def crossover_operation(self, selected_pop):
        while len(self.new_population) < len(self.population):
            index_parent1 = random.randint(0,len(selected_pop))
            index_parent2 = random.randint(0,len(selected_pop))
            while index_parent1 == index_parent2:
                index_parent1 = random.randint(0, len(selected_pop)-1)
                index_parent2 = random.randint(0, len(selected_pop)-1)
            if self.crossover_rate > random.random():
                child1 = []
                child2 = []
                for chr_index in range(len(self.population[index_parent1])):
                    child1.append([])
                    child2.append([])
                    cutting_point = random.randint(1,len(self.population[index_parent1][chr_index]-1))
                    child1[-1] = np.append(self.population[index_parent1][chr_index][:cutting_point],
                                           self.population[index_parent2][chr_index][cutting_point:])
                    child2[-1] = np.append(self.population[index_parent2][chr_index][:cutting_point],
                                           self.population[index_parent1][chr_index][cutting_point:])
                self.new_population.append(child1)
                self.new_population.append(child2)
            else:
                self.new_population.append(copy.deepcopy(self.population[index_parent1]))
                self.new_population.append(copy.deepcopy(self.population[index_parent2]))

        if len(self.new_population) > len(self.population):
            self.new_population.pop(-1)
        return

    def mutation_operation(self):
        for individual_index in range(int(len(self.new_population)*self.elitism_rate), len(self.new_population)):
            if random.random() < self.mutation_rate: #determine if the current individual will undergo mutation
                chromosome_index = random.randint(0,len(self.population[0])-1)
                gene_index  = random.randint(0,len(self.population[0][chromosome_index])-1)
                self.new_population[individual_index][chromosome_index][gene_index] = \
                    not self.new_population[individual_index][chromosome_index][gene_index]
        return
