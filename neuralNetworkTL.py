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
# TODO: Include public predict method and an assertion for
#       data size/weights compatability
# TODO: public method getParameters
# TODO: "No bias detected, during cost function"

import numpy as np
from scipy.optimize import minimize
import pdb

class neuralNetwork(object):
    
    # Initialize neural network parameters
    # and random weights (theta1 & theta2)
    def __init__(self, input_layer_size, num_labels,
                 hidden_layer_size=None, lamda=None,
                 add_bias=False):
        
        """
        Use default network parameters
        if none are specified
        """
        
        if (hidden_layer_size == None):
            self.hl_size = 25
            print "Warning used default parameter: 'hidden_layer_size'"

        else:
            self.hl_size = hidden_layer_size

        if (lamda == None):
            self.lamda = 0.1
            print "Warning used default parameter: 'lamda'"

        else:
            self.lamda = lamda

        # Parse bias initial parameter
        if (add_bias == False):
            init_bias = 0

        elif (add_bias == True):
            init_bias = 1

        else:
            raise ValueError("Only boolean values can be used for 'add_bias'")
    
        self.il_size = input_layer_size
        self.n_labels = num_labels
        self.bias = 1
        self.iter = 0
        
        """
        Initialises weights as a uniform 
        distribution between two functions
        based on the input size for each 
        perceptron. Includes weights for the
        bias terms added to the input data.
        """
        
        theta1 = np.random.uniform(
                                   (-1/np.sqrt(self.il_size)),
                                   (1/np.sqrt(self.il_size)),
                                   [self.hl_size, self.il_size + init_bias])
        
        theta2 = np.random.uniform(
                                   (-1/np.sqrt(self.hl_size)),
                                   (1/np.sqrt(self.hl_size)),
                                   [self.n_labels, self.hl_size + init_bias])
            
        self.nn_weights = np.concatenate(
                                        [theta1.ravel(),
                                         theta2.ravel()])
    
    # Unroll the vector of network weights
    def __unrollWeights(self, weights):
        
        theta1 = np.reshape(self.nn_weights[0:self.hl_size * (self.il_size + self.bias)],
                            [self.hl_size, (self.il_size + self.bias)])
            
        theta2 = np.reshape(self.nn_weights[(self.hl_size * (self.il_size + self.bias))::],
                                                [self.n_labels, (self.hl_size + self.bias)])
                            
        return theta1, theta2
    
    # Set weights to user specified values
    def setWeights(self, new_theta1, new_theta2, add_bias=False):
    
        """
        Public method to set global parameter 
        values to user specified values
        """

        # unroll parameters into theta1 & theta 2
        theta1, theta2 = self.__unrollWeights(self.nn_weights)
        
        print "Previous Theta1 parameter size = ", theta1.shape
        print "Previous Theta1 parameter size = ", theta2.shape
        print "------------------------------------"
        
        # Parse bias parameter
        if add_bias == False:
            pass
        
        elif add_bias == True:
            np.insert(new_theta1, 0, 1, axis=1)
            np.insert(new_theta2, 0, 1, axis=1)
        
        elif add_bias == False:
            pass
        
        else:
            raise ValueError("Only boolean values can be used for 'add_bias'")
        
        # Set new parameter values
        theta1 = new_theta1
        theta2 = new_theta2
        
        print "New Theta1 parameter size = ", theta1.shape
        print "New Theta1 parameter size = ", theta2.shape

        self.nn_weights = np.concatenate([theta1.ravel(),
                                        theta2.ravel()])
    
    # 'One-Hot' Matrix
    def __oneHot(self, y):
        
        """
        Converts a vector of target 
        integers into a sparse matrix
        where each row is a traget 
        vector.
        """
        
        y_ohm = np.zeros([len(y), self.n_labels])
        i = len(y) - 1

        while i > 0:
            y_vec = np.zeros(self.n_labels)
            y_vec[y[i] - 1] = 1
            y_ohm[i,:] = y_vec
            i -= 1
    
        return y_ohm
    
    # Sigmoid Activation function
    def __sigmoid(self, a):
    
        """
        Takes as input a scalar value 
        or array of values and applies
        the sigmoidal activation.
        """

        return 1 / (1 + np.exp(-a))
    
    # Calculate the sigmoid gradient
    def __sigmoidGradient(self, z):

        """
        For each value of z (activations 
        after forward propagation), evaluate
        the gradient. Takes as input a 
        scalar value or array of values.
        """
        gz = self.__sigmoid(z)
        
        return np.multiply(gz, (1 - gz))

    # Perform Fowarad Propagation
    def __forwardProp(self, X, theta1, theta2):
    
        """
        A vectorized implementation of 
        forward propagation. Calculating
        the activation of each perceptron
        from the input vector (X) and the
        network weights (theta1 & theta2).
        """
        
        # Add a column of ones to X
        a1 = np.insert(X, 0, 1, axis=1)
        
        # Apply the weights and sum each column for the
        # activation of each unit in the hidden layer
        z2 = np.dot(a1, np.transpose(theta1))

        # Apply the sigmoid function to the activations
        # in the hidden layer
        a2 = self.__sigmoid(z2)
        
        # Add bias term to hidden layer activations
        a2 = np.insert(a2, 0, 1, axis=1)
        
        # Apply the weights and sum each column for the
        # activation of each unit in the ouput layer
        z3 = np.dot(a2, np.transpose(theta2))

        # Apply the sigmoid function to the activations in
        # the ouput layer
        hyp = self.__sigmoid(z3)

        return hyp, a1, a2, z2

    # Cost Function Calculation
    def __nnCost(self, initial_theta, X, y):
        
        """
        Calculates the cross-entropy objective 
        function that is used to optimise the 
        parameters of the network. 
        Regularisation is used to prevent over
        -fitting, this is weighted with lamda.
        """
        
        # unroll parameters into theta1 & theta 2
        theta1, theta2 = self.__unrollWeights(initial_theta)
        
        # Convert target values to one hot matrix
        y_matrix = self.__oneHot(y)
        
        # Scaling term for regularization
        m = float(len(X))

        hyp, a1, a2, z2 = self.__forwardProp(X, theta1, theta2)
        
        # Cross-entropy cost function
        J = (1/m) * (np.sum(y_matrix * np.log(hyp))) - (np.sum((1 - y_matrix) *
                                                              np.log(1 - hyp)))
    
        # Regularisation term
        reg_term = (self.lamda/(2*m)) * np.sum(np.square(theta1[:,1::])) \
                    + np.sum(np.square(theta2[:,1::]))
        
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
        d_2 = np.multiply(np.dot(d_3, theta2[:, 1::]),
                          self.__sigmoidGradient(z2))
        
        # Compute the error at each layer
        delta_1 = np.dot(np.transpose(d_2), a1)
        delta_2 = np.dot(np.transpose(d_3), a2)
            
        # Compute the gradient for each parameter
        theta1_grad = delta_1 / m
        theta2_grad = delta_2 / m
        
        # Create a local copy of theta1 & theta2
        theta1_reg = theta1
        theta2_reg = theta2
        
        # Exclude the 1st column from the local
        # parameter values from the regularisation
        theta1_reg[:, 0] = 0
        theta2_reg[:, 0] = 0
    
        # Apply regularisation to the gradients
        theta1_grad += theta1_reg * (self.lamda / m)
        theta2_grad += theta2_reg * (self.lamda / m)
        
        nn_weights = np.concatenate([theta1_grad.ravel(),
                                   theta2_grad.ravel()])
                                   
        print J

        return J, nn_weights

    # Public method for training the network
    def train(self, X, y, max_it=None):
        
        """
        Utilises the 'minimize' module
        from the scipy library to train
        the network, finding the optimum
        set of parameters (weights).
        """
        
        print "Initial cost is - ", self.__nnCost(self.nn_weights,X,y)[0]
    
        # Parse the input variable 'max_it'
        if (max_it == None):
            max_it = 100
        
        # Train the network
        result = minimize(
                          self.__nnCost,
                          self.nn_weights,
                          args=(X, y),
                          method='CG',
                          jac=True,
                          options={'maxiter': 50, 'disp': False})
        
        # Extract the parameters from
        # the results object
        model = result['x']
        
        # Set the global network parameters
        self.nn_weights = model

        return model

    






