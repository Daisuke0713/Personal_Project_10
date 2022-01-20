# network.py

import numpy as np
import random
import mnist_loader as ml
import matplotlib.pyplot as plt
import os

class Network:
    
    def __init__(self, sizes):
        # sizes is array representing the number of neurons in each layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    """
    Returns an output for the given input (vector indicating pixels) by applying
    the sigmoid function for each layer
    param: a is a vector indicating the values of the first layer (input layer)
    """
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    """
    Train our network given training data
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            # create a list of mini batches, each of size minibatch-size
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                # use the function update_mini_batch to update each mini_batch
                self.update_mini_batch(mini_batch, eta)
            if test_data: # None by defualt
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
    
    """
    Update weights and biases by applying GD 
    params: eta is the learning rate
            mini_batch is a randomly chosen data in the form of (x,y)
    """
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
            self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    """
    Returns (nabla_b, nabla_w) indicating gradient for the cost fn
    """
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] # list to store all the z vectors
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    """
    Return the accuracy of the network 
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    def test(self, training_data):
        test_pixels = training_data[np.random.randint(0, len(training_data)-1)][0]
        #print(test_pixels)
        #test_pixels = np.array(test_pixels, dtype='uint8')
        pixels = test_pixels.reshape((28,28))
        #print(test_pixels)
        #Plot
        plt.title('Test Image')
        plt.imshow(pixels, cmap='gray')
        plt.show()
        return test_pixels
        
    def test_image(self, test_pixels):
        print("The digit is {}".format(np.argmax(self.feedforward(test_pixels))))
        

# computes the output of the activation function given a vector z
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1-sigmoid(z))

def main():
    training_data, validation_data, test_data = ml.load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
    input()
    for x in range(1, 10):
        pixels = net.test(training_data)
        net.test_image(pixels)
    

if __name__ == "__main__":
    main()        