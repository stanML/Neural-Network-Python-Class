"""
A python class for the implemtation of
Two-Layer Perceptrons (Neural Network)
----------------------------------------
Ben Stanley Dec' 2015
https://github.com/stanML
benstanley@live.co.uk

"""

# TODO: allow for more than one layer as a network parameter
#       (use a generator??)
# TODO: remove methods such as sigmoid, put in base clase that nn extends
# TODO: Include set_parameter method and an assertion for data size compatability



import numpy as np
from scipy import optimize
import pdb

class neuralNetwork(object):
    
    # Initialize neural network parameters
    # and random weights (theta1 & theta2)
    def __init__(self, input_layer_size, hidden_layer_size, num_labels, lamda):

        self.il_size = input_layer_size
        self.hl_size = hidden_layer_size
        self.n_labels = num_labels
        self.lamda = lamda
        
        """
        Initialises weights as a uniform 
        distribution between two functions
        based on the input size for each 
        perceptron. Includes weights for the
        bias terms added to the input data.
        """
        
        self.theta1 = np.random.uniform((-1/np.sqrt(self.il_size)),
                                        (1/np.sqrt(self.il_size)),
                                        [self.hl_size, self.il_size + 1])
        
        self.theta2 = np.random.uniform((-1/np.sqrt(self.hl_size)),
                                        (1/np.sqrt(self.hl_size)),
                                        [self.n_labels, self.hl_size + 1])
   
    # 'One-Hot' Matrix
    def oneHot(self, y):
        
        y_ohm = np.zeros([len(y), self.n_labels])
        i = len(y) - 1

        while i > 0:
            y_vec = np.zeros(self.n_labels)
            y_vec[y[i]] = 1
            y_ohm[i,:] = y_vec
            i -= 1
    
        return y_ohm
    
    # Add biases
    def addBias(self, array):
    
        ones  = np.ones((len(array),1))
        array = np.concatenate((ones,array), axis=1)
    
        return array
    
    # Sigmoid Activation function
    def sigmoid(self, a):
    
        """
        Takes as input a scalar value 
        or array of values and applies
        the sigmoidal activation.
        """

        return 1 / (1 + np.exp(-a))
    
    # Calculate the sigmoid gradient
    def sigmoidGradient(self, z):

        """
        For each value of z (activations 
        after forward propagation), evaluate
        the gradient. Takes as input a 
        scalar value or array of values.
        """
        gz = self.sigmoid(z)
        
        return np.multiply(gz, (1 - gz))

    # Perform Fowarad Propagation
    def forwardProp(self,X):
    
        """
        A vectorized implementation of 
        forward propagation. Calculating
        the activation of each perceptron
        from the input vector (X) and the
        network weights (theta1 & theta2).
        """
        
        # Add a column of ones to X
        a1 = self.addBias(X)
        
        # Apply the weights and sum each column for the
        # activation of each unit in the hidden layer
        z2 = np.dot(a1, np.transpose(self.theta1))

        # Apply the sigmoid function to the activations
        # in the hidden layer
        a2 = self.sigmoid(z2)
        
        # Add bias term to hidden layer activations
        a2 = np.insert(a2, 0, 1, axis=1)
        
        # Apply the weights and sum each column for the
        # activation of each unit in the ouput layer
        z3 = np.dot(a2, np.transpose(self.theta2))

        # Apply the sigmoid function to the activations in
        # the ouput layer
        hyp = self.sigmoid(z3)

        return hyp, a1, a2, z2

    # Cost Function Calculation
    def nnCost(self, X, y):
        
        """
        Calculates the cross-entropy objective 
        function that is used to optimise the 
        parameters of the network. 
        Regularisation is used to prevent over
        -fitting, this is weighted with lamda.
        """
        
        # Convert target values to one hot matrix
        y_matrix = self.oneHot(y)
        
        # Scaling term for regularization
        m = float(len(X))

        hyp, a1, a2, z2 = self.forwardProp(X)
        
        # Cross-entropy cost function
        J = (1/m) * (np.sum(y_matrix * np.log(hyp))) - (np.sum((1 - y_matrix) *
                                                              np.log(1 - hyp)))
    
        # Regularisation term
        reg_term = (self.lamda/(2*m)) * np.sum(np.square(self.theta1[:,1::])) \
                    + np.sum(np.square(self.theta2[:,1::]))
        
        # Regularised cost function
        J += reg_term
        
        """
        Run the neural network optimisation
        algorithm 'backpropagation' that 
        calculatest the error gradient at 
        each node of the network and updates
        each parameter to minimise the cost.
        """
        
        # Compute the error at the output layer
        d_3 = hyp - y_matrix
        
        # Compute the error at the activations of the hidden layer
        d_2 = np.multiply(np.dot(d_3, self.theta2[:, 1::]), self.sigmoidGradient(z2))
        
        # Compute the error at each layer
        delta_1 = np.dot(np.transpose(d_2), a1)
        delta_2 = np.dot(np.transpose(d_3), a2)
            
        # Compute the gradient for each parameter
        theta1_grad = delta_1 / m
        theta2_grad = delta_2 / m
        
        # Create a local copy of theta1 & theta2
        theta1_reg = self.theta1
        theta2_reg = self.theta2
        
        # Exclude the 1st column from the local
        # parameter values from the regularisation
        theta1_reg[:, 0] = 0
        theta2_reg[:, 0] = 0
    
        # Apply regularisation to the gradients
        theta1_grad += theta1_reg * (self.lamda / m)
        theta2_grad += theta2_reg * (self.lamda / m)
        
        nn_params = np.concatenate([theta1_grad.ravel(),
                                   theta2_grad.ravel()])

        return J, nn_params














