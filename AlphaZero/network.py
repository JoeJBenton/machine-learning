import numpy as np
import tictactoe as game

class Network (object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x,y) in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Feedforward to penultimate layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)

        # Final softmax layer
        b, w = self.biases[-1], self.weights[-1]
        a = np.dot(w,a) + b
        policy = np.exp(a[:-1])/np.sum(np.exp(a[:-1]))
        value = sigmoid(a[-1])*2-1

        return (policy, value)

    def SGD(self, training_data, epochs, batch_size, eta, display=False):
        n = len(training_data)
        for epoch in range(epochs):
            if display:
                print("Epoch %s" % epoch)
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k+batch_size] for k in xrange(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for (x,y) in mini_batch:
            (delta_nabla_b, delta_nabla_w) = self.back_prop((x, y))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (eta/len(mini_batch))*nb for nb, b in zip(nabla_b, self.biases)]
        self.weights = [w - (eta/len(mini_batch))*nw for nw, w in zip(nabla_w, self.weights)]

    def back_prop(self, (input, output)):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Perform feedforward to get activations
        activations = [input]
        zs = []
        a = input

        # Feedforward to penultimate layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        # Final softmax layer
        b, w = self.biases[-1], self.weights[-1]
        z = np.dot(w,a) + b
        zs.append(z)
        network_policy = np.exp(z[:-1])/np.sum(np.exp(z[:-1]))
        network_value = sigmoid(z[-1])*2-1
        activations.append((network_policy, network_value))

        # First backpropagate through the final softmax layer
        policy_cost_derivative = -output[:-1]/network_policy
        value_cost_derivative = 2*(network_value - output[-1])
        inverse_matrix = np.dot(np.exp(zs[-1][:-1]), \
                                    np.exp(zs[-1][:-1].transpose()))
        inverse_matrix = -inverse_matrix/np.sum(inverse_matrix)
        inverse_matrix = inverse_matrix - np.diag(inverse_matrix.sum(1))

        policy_delta = 0*np.dot(inverse_matrix, policy_cost_derivative)
        value_delta = sigmoid_prime(zs[-1][-1])*value_cost_derivative

        delta = np.concatenate((policy_delta, [value_delta]), 0)

        #Backward pass
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return(nabla_b, nabla_w)
            


    def policy(self, state):
        a = state.network_input()
        (output_policy, value)  = self.feedforward(a)
        policy = {}
        for (index, probability) in zip(range(self.sizes[-1]-1), output_policy):
            policy[game.move_list[index]] = probability
        return policy

    def value(self, state):
        a = state.network_input()
        (policy, value)  = self.feedforward(a)
        return value


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


