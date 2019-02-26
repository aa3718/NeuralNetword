import numpy as np
import pickle
import math

np.set_printoptions(threshold=np.inf)

def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """

    # initialises the weights to keep the varianace stable
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        #######################################################################
        #** START OF YOUR CODE **
        #######################################################################
        # calculate sigmoid value for given x
        sigmoid = 1 / (1 + np.exp(-x))
        self._cache_current = x, sigmoid
        return sigmoid

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):


        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x, sigmoid = self._cache_current
        derivativeActivation = sigmoid * (1 - sigmoid)
        derivative = derivativeActivation * grad_z
        return derivative

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #relu = x * max(0, x.all())
        #print("Relu 1" + str(relu))
        relu = np.maximum(np.zeros(x.shape), x)
        #print("Relu 2" + str(relu))
        self._cache_current = x, relu
        return relu
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x, relu = self._cache_current
        if(x.all() > 0):
            derivativeActivation = 1
        else:
            derivativeActivation = 0
        derivative = derivativeActivation * grad_z
        return derivative
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.
        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = None
        self._b = None

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        # Initialize the weights matrix and the bias vector
        # Weights is a matrix of in x out
        self._W = xavier_init([self.n_in, self.n_out])
        #print("Weights: " + str(self._W))
        # The bias is a M x 1 column vector, set to zero (typically)
        self._b = np.zeros(self.n_out)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).
        Logs information needed to compute gradient at a later stage in
        `_cache_current`.
        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).
        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Perform the matrix multiplication
        z = np.matmul(x, self._W) + self._b

        # store the results locally for use with the back-propagation
        self._cache_current = x

        return z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).
        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).
        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # grad_z contains the dLoss/dZ from the layer above, i.e dLoss/dZ
        # is given!
        # dLoss/dX = dLoss/dZ * W^transpose

        grad_x = np.matmul(grad_z, self._W.transpose())

        # Also need to store dW and dB,
        # dLoss/dW = X^T * dLoss/dZ
        self._grad_W_current = np.matmul(self._cache_current.transpose(), grad_z)

        # dLoss/dB = 1^T * dLoss/dZ
        rows = grad_z.shape
        ident = np.identity(rows[0])
        self._grad_b_current = np.matmul(ident.transpose(), grad_z)

        # Return the gradient, dLoss/dX
        return grad_x

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.
        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Gradient decent is defined as W <- W - alpha*dL/dW
        # This updates the weights for the next round of propagation
        # Alpha is the learning rate

        self._W = self._W - (learning_rate * self._grad_W_current)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.
        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # self.activation_classes = []

        network = []

        # if neurons list is empty then no layers
        if (len(self.neurons) < 1):
            raise Exception("Not enough neurons have been inputted")

        # input_dim matches to first layer
        layer = LinearLayer(input_dim, neurons[0])
        network.append(layer)

        if (self.activations[0] == "relu"):
            network.append(ReluLayer())
        elif (self.activations[0] == "sigmoid"):
            network.append(SigmoidLayer())

        # if number of layers greater than 1 then append rest of layers
        print("length is " + str(len(self.neurons)))
        if (len(self.neurons) > 1):

            for i in range(0, len(self.neurons) - 1):
                print(i)
                layer = LinearLayer(neurons[i], neurons[i+1])
                network.append(layer)

                if (self.activations[i+1] == "relu"):
                    network.append(ReluLayer())
                elif (self.activations[i+1] == "sigmoid"):
                    network.append(SigmoidLayer())

        self._layers = network

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):

        # implement forward pass to collect one batch at a time
        # implement sigmoid transformation

        """
        #Performs forward pass through the network.
        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).
        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_input = x
        # iterate through layers

        for i in range(0, len(self._layers)):
            x_input = self._layers[i](x_input)

        x_output = x_input

        # an array of size (batch_size, output_layer_length) is returned
        return x_output
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.
        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).
        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # iterate backwards through layers
        for i in reversed(range(len(self._layers))):

            grad_z = (self._layers[i].backward(grad_z))

        return grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.
        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for i in range(0, len(self._layers)):

            self._layers[i].update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.
        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = None

        if(self.loss_fun == "cross_entropy"):
            self._loss_layer = CrossEntropyLossLayer()

        elif(self.loss_fun == "mse"):
            self._loss_layer = MSELossLayer()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).
        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Create an randomly shuffled array to use as the new order
        new_order = np.random.permutation(len(input_dataset))

        # Use this order to rearrange the input datasets
        shuffled_inputs = input_dataset[new_order]
        shuffled_targets = target_dataset[new_order]

        # Return the shuffled data
        return(shuffled_inputs, shuffled_targets)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for i in range(self.nb_epoch):
            print("Epoch Number: " + str(i))
            # Shuffle the data if required
            if(self.shuffle_flag):
                trainingData = self.shuffle(input_dataset, target_dataset)
            else:
                trainingData = (input_dataset, target_dataset)

            # Split data into batches of size bacch_size
            input_batches = np.array_split(trainingData[0], self.batch_size)
            target_batches = np.array_split(trainingData[1], self.batch_size)

            for j in range(self.batch_size):

                #print("Input Values: " + str(input_batches[j]))
                # Perform forward pass
                output_values = self.network(input_batches[j])

                #print("Output Values: " + str(output_values))
                # Evaluate the loss function
                self._loss_layer(output_values, target_batches[j])
                #loss = -1*((target_batches[j]*(1/output_values) + (1-target_batches[j])*(1/(1-output_values))))

                # Send gradients backward
                gradients = self.network.backward(self._loss_layer.backward()) # TODO: Check this is correct
                #gradients = network.backward(loss)

                # Perform gradient descent
                self.network.update_params(self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if(self._loss_layer != None):
            output_values = self.network(input_dataset);
            return self._loss_layer(output_values, target_dataset)
        raise Exception("Loss layer has not been set")
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)
        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.max_value = np.amax(data);
        self.min_value = np.amin(data);
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.
        Arguments:
            - data {np.ndarray} dataset to be normalized.
        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return (data - self.min_value)/(self.max_value - self.min_value)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.
        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.
        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return (data*(self.max_value-self.min_value))+self.min_value;
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
    
