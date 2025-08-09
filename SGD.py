import numpy as np
import random as r

# based on the formulae in http://neuralnetworksanddeeplearning.com/chap2.html
class NN:
    def __init__(self, sizes, activation_function):
        # we take sizes as a list: [a1, a2, a3, ... , an]
        # with a1 the entry size (the size of the vector) (which is a column)
        # a2 to an-1 is the size of the neuron layer (the number of neurons)
        # an is the size of the output vector
        # activation_function is the type of activation_function that will be used for every layer 
        # except a1 (entry)
        # activation_function = "sigmoid", "tanh" , "relu"
        self.sizes = sizes
        self.activation_function = activation_function
        xavier_limits = self.xavier_init()
        self.biases = [
            np.random.uniform(-xavier_limits[k+1], xavier_limits[k+1], (sizes[k+1], 1)) 
            for k in range(len(sizes)-1)
            ]
        # we initialize uniformly the biases between -limit limit, based on Xavier initialization
        self.weights = [
            np.random.uniform(-xavier_limits[i+1], xavier_limits[i+1], (y, x)) 
            for i, (x, y) in enumerate(zip(sizes[:-1], sizes[1:]))
            ]

    def xavier_init(self):
        #gives the limits for a uniform random to initialize the biases and weights
        res = []
        for k in range(len(self.sizes)):
            if k ==0:
                res.append(0) #we dont put any biase to the first layer
            else:
                value = np.sqrt(6/(self.sizes[k]+self.sizes[k-1]))
                res.append(value)
        return(res)
    
    def activation(self, z):
        # uses the right activation function
        if self.activation_function == "sigmoid":
            return(self.sigmoid(z))
        elif self.activation_function == "relu":
            return(self.relu(z))
        elif self.activation_function == "tanh":
            return(self.tanh(z))
        else:
            raise ValueError(f"Activation '{self.activation_function}' not implemented.")
        
    def activation_prime(self, z):
        # uses the right activation function to compute derivative
        if self.activation_function == "sigmoid":
            return(self.sigmoid_prime(z))
        elif self.activation_function == "relu":
            return(self.relu_prime(z))
        elif self.activation_function == "tanh":
            return(self.tanh_prime(z))
        else:
            raise ValueError(f"Activation '{self.activation_function}' not implemented.")
        
    
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
    
    def relu(self, z):
        return (z > 0).astype(float)
    
    def tanh(self, z):
        return(np.tanh(z))
    
    def sigmoid_prime(self, z):
        return(self.sigmoid(z)*(1-self.sigmoid(z)))
    
    def relu_prime(self, z):
        return 0 if z <= 0 else 1
    
    def tanh_prime(self, z):
        return(1-self.tanh(z)**2)
    
    def feedforward(self, entry):
        # process the result of the network with a certain entry
        for b, w in zip(self.biases, self.weights):
            entry = self.activation(np.dot(w, entry)+b)
        return entry
    
    def training(self, train_data, nb_epochs, batch_size, learn_rate):
        # train the network using SGD with mini batches (the size of batch size)
        # training data format: list of tuples (X, Y) , Y beeing the desiered result
        for epoch in range(nb_epochs):
            r.shuffle(train_data) # we shuffle the dataset
            batches = []
            for k in range (0, len(train_data), batch_size):
                batch = train_data[k:k+batch_size]
                batches.append(batch)
                self.back_propagation(batch, learn_rate)
            print("epoch "+str(epoch)+" completed")
            
    def back_propagation(self, batch, learn_rate):
        # updates the weights and biases using backpropagation with the results on batch
        grad_b = [np.zeros(bias.shape) for bias in self.biases]
        grad_w = [np.zeros(weight.shape) for weight in self.weights] #gradients init
        for X, Y in batch:
            #formulae:
            # new_w = w- learn_rate/batch_size*grad(w), same for b
            # we note z(l)= W(l)activation(z(l-1))+b(l)
            
            # we note at the last layer L, delta(L) = dC/dz(L) (depends on the cost function)
            # for the moment we consider C = 1/2(activation(z(L))-y)**2
            # so delta(L) = (activation(L) - y)°activation_prime(z(L))
            # with ° the hadamard product
            # at the last layer L gradb(L)= delta(L), gradw(L) = delta(L)activation(z(L-1))transposed
            # we note at layer l<L, delta(l)=(W(l+1)transposed*delta(l+1))°activation_prime(z(l))
            # gradb(l)= delta(l), gradw(l) = delta(l)activation(z(l-1))transposed
            # we note activation(z) = activation
            deltagrad_b = [np.zeros(b.shape) for b in self.biases]
            deltagrad_w = [np.zeros(w.shape) for w in self.weights]
            activation = X
            all_activation = [X] # we store the previous values
            all_z = []
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b # calculate z(l)
                activation = self.activation(z)
                all_activation.append(activation)
                all_z.append(z)
            delta = (all_activation[-1]- Y)*self.activation_prime(all_z[-1]) 
            # here all_activation[-1]- Y is the derivative of our cost func, if we want to handle different cost functions
            # we need to implement the derivatives and call them here
            deltagrad_b[-1] = delta # gradb(L)= delta(L)
            deltagrad_w[-1] = np.dot(delta, all_activation[-2].transpose()) #gradw(L) = delta(L)activation(z(l-1))transposed
            L = len(self.sizes)
            # now for l<L except entry
            for l in range(L-1, 1, -1): # l = L-1,L-2,...,2
                z = all_z[l-2]
                aprime = self.activation_prime(z)
                delta = np.dot(self.weights[l-1].T, delta)*aprime
                deltagrad_b[l-2] = delta
                deltagrad_w[l-2] = np.dot(delta, all_activation[l-2].T)
            # we sum the gradient on all the minibatch
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, deltagrad_b)] 
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, deltagrad_w)] 
        # new_w = w- learn_rate/batch_size*grad(w), same for b
        self.weights = [w-(learn_rate/len(batch))*gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b-(learn_rate/len(batch))*gb for b, gb in zip(self.biases, grad_b)]
        # the weight and biases are now updated

if __name__ == "__main__":
    #try to learn the AND logic
    net = NN([2, 3, 1], activation_function="sigmoid") 
    #entry size: 2,
    #3 layers
    #output: 1
    training_data = [
    (np.array([[0],[0]]), np.array([[0]])),
    (np.array([[0],[1]]), np.array([[0]])),
    (np.array([[1],[0]]), np.array([[0]])),
    (np.array([[1],[1]]), np.array([[1]])),
]
    net.training(training_data, nb_epochs=1000, batch_size=4, learn_rate=0.5)  # training
    for x, y in training_data:
        output = net.feedforward(x)
        print(f"Input: {x.ravel()}, Expected: {y.ravel()}, Output: {output.ravel()}")