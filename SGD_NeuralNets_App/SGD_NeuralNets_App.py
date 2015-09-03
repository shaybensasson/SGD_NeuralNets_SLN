from numpy import *
import matplotlib.pyplot as plt

from numpy.random import random_sample
from sklearn.cross_validation import train_test_split


# This is the neural network class, for your information.
class NeuralNetwork(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        #print 'in ctor'
    
    def _activation(self, x):
        """ Funny tanh function. """
        z = x*2/3
        y = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        return 1.7159*y
    
    def _da(self, x):
        return (1.7159 - multiply(x, x) / 1.7159) * 2/3
    
    def feed_forward(self, X):
        """From the input X, calculate the activations at the hidden layer and the output layer."""
        #print 'in FF'
        #print c_[X, ones((X.shape[0], 1))].shape
        #print self.W_hidden
        
        Z      = self._activation(dot(c_[X, ones((X.shape[0], 1))], self.W_hidden))
        return   self._activation(dot(c_[Z, ones((X.shape[0], 1))], self.W_output)), Z
                                        
    def back_propagate(self, inputs, hidden, output, errors):
        """Back-propagate the errors and update the weights."""
        #print 'in BP'
        
        #print self.W_output[:-1]
        #print self.W_output
        d_output = self._da(output) * errors
        d_hidden = self._da(hidden) * dot(d_output, self.W_output[:-1].T)
        
        n_samples = inputs.shape[0]
        bias = ones((n_samples, 1))
        # Update momentum and weights
        #print (hidden)
        #print (bias)
        #print (c_[hidden, bias].T)
        #print (d_output)
        #print r_['1',hidden, bias].T.shape
        self.V_output = self.learning_rate * dot(c_[hidden, bias].T, d_output) / n_samples + \
                        self.momentum_learning_rate * self.V_output
        self.W_output+= self.V_output
        
        self.V_hidden = self.learning_rate * dot(c_[inputs, bias].T, d_hidden) / n_samples + \
                       self.momentum_learning_rate * self.V_hidden
        self.W_hidden+= self.V_hidden 
    
    def train(self, epochs = 100, n_input = 2, n_hidden = 2, n_output = 1, 
            learning_rate = 0.1, momentum_learning_rate = 0.9,          
            test_size=0.2):
        """Initialize the network and start training."""

        #print ('inside train()')
        # Initialize variables
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.momentum_learning_rate = momentum_learning_rate
        
        
        self.V_hidden = zeros((self.n_input + 1, self.n_hidden))
        self.W_hidden = random_sample(self.V_hidden.shape)
        self.V_output = zeros((self.n_hidden + 1, self.n_output))
        self.W_output = random_sample(self.V_output.shape)
        
        #print'Wh,Wo'
        #print self.W_hidden
        #print self.W_output

        # Start the training
        rmse=zeros((epochs,2))
        for t in arange(epochs):
            # Test then Train, since we'll use the training errors
            outputs, hidden = self.feed_forward(self.inputs)
            #print 'FF'
            #print hidden.shape
            #print outputs.shape
            #target=ones(outputs.shape)*(-1.0)
            #target[arange(target.shape[0]),self.targets-1]=1.0
            #print (target)
            errors = self.targets - outputs
            #print (hidden.shape)
            #print (outputs.shape)
            #print (self.targets.shape)

            #errors = self.targets - outputs
            i=0
            RMSE = sqrt((errors**2).mean())
            rmse[t, i] = sqrt((errors**2).mean()) 
            yield rmse, t, epochs
            
            if (RMSE<1e-4):
                break
            
            # Update weights using backpropagation
            #print 'Pre BP'
            #print self.inputs.shape
            #print hidden.shape
            #print outputs.shape
            #print errors.shape
            
            self.back_propagate(self.inputs, hidden, outputs, errors)

    def predict(self, n):
        """Returns the prediction and the reconstruction for the sample n."""
        outputs, hidden = self.feed_forward(self.inputs[n:n+1])
        return outputs

def plot_training(axs, rmse, t, epochs):
    """Draw the plot to the specified axis."""
    axs.set_title("RMSE")
    axs.set_xlabel("Training epoch")
    axs.set_ylabel("RMSE")
    axs.grid()

    axs.plot(arange(t), rmse[:t])
    axs.set_xlim([0, epochs])
    axs.set_ylim([0, 2.0])
    axs.legend(['Test', 'Training'], loc="best")


def train_network(inputs, targets, **kwargs):
    net = NeuralNetwork(inputs, targets)
    fig, axs = plt.subplots(1,1,figsize=(10,5))
    for rmse, t, epochs in net.train(**kwargs):
        if mod(t, 10) != 0:
            continue

        plot_training(axs, rmse, t, epochs)
        invalidatePlot()

        axs.cla()

    plot_training(axs, rmse, t, epochs)
    invalidatePlot()
    
    print 'Total epochs till convergence: ' + str(t)

    return net

def invalidatePlot():
    plt.draw()      # force a draw
    plt.pause(0.1)

if __name__ == "__main__":
    X=array([
        [ 1,-1],
        [-1, 1],
        [ 1, 1],
        [-1,-1]], dtype=float)
    T=array([
        [ 1],
        [ 1],
        [-1],
        [-1]], dtype=float)

    net = train_network(
        inputs=X,
        targets=T,
        epochs= 1000,
        n_input = 2,
        n_hidden = 2,
        n_output = 1,
        learning_rate = 0.1,
        momentum_learning_rate = 0.9,         
        test_size=0.2)

    for n in arange(X.shape[0]):
        print X[n]
        print T[n]
        print net.predict(n)
        print '\n'

    plt.waitforbuttonpress(timeout=-1)
    #exit()