import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np

from illustrate import illustrate_results_ROI

def main():

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # LOAD DATA
    dataset = np.loadtxt("ROI_dataset.dat")
    number_of_classes = 4

    x = dataset[:, :3]
    y = dataset[:, 3:]

    split_index = int(0.8 * len(x))

    x_training = x[:split_index]
    y_training = y[:split_index]
    x_testing = x[split_index:]
    y_testing = y[split_index:]

    x_dim = x_training.shape[1] # get number of input neurons
    y_dim = y_training.shape[1] # get number of output neurons

    print("x_dim is ")
    print(x_dim)

    print("y_dim is ")
    print(y_dim)

    # CREATE NETWORK

    model = torch.nn.Sequential()

    # layers x_dim -> 16 -> 16 -> y_dim using ReLU activation function
    model.add_module("dense1", torch.nn.Linear(x_dim, 16))
    model.add_module("relu1", torch.nn.ReLU())
    model.add_module("dense2", torch.nn.Linear(16, 16))
    model.add_module("relu2", torch.nn.ReLU())
    model.add_module("dense3", torch.nn.Linear(16, y_dim))

    # define an optimiser and learning rate
    optimiser = optim.SGD(model.parameters(), lr=0.1)

    # define a loss function (cross entropy needed for classification problem)
    loss = nn.CrossEntropyLoss(size_average=True)

    batch_size = 125
    num_epoch = 1000

    training_samples = x_training.shape[0]
    training_batches = training_samples/batch_size

    # convert numpy array into torch tensors
    x_training = torch.from_numpy(x_training).float()
    y_training = torch.from_numpy(y_training).long()
    x_testing = torch.from_numpy(x_testing).float()

    training_accuracy = []
    training_loss = []

    current_loss = 0
    total_loss = []

    for i in range(num_epoch):
        current_loss = 0

        for j in range(int(training_batches)):
            x_batch = x_training[j*batch_size:(j+1)*batch_size]
            y_batch = y_training[j*batch_size:(j+1)*batch_size]
            current_loss += train(model, loss, optimiser, x_batch, y_batch)

        y_predict = predict(model, x_testing)
        accuracy = np.mean(y_testing == y_predict)

        print("Epoch: %d, cost: %f, accuracy: %.2f" % (i, current_loss/training_batches, accuracy))

        total_loss.append(total_loss.append(current_loss/batch_size))
        training_accuracy.append(accuracy)

    #illustrate_results_ROI(network, prep)

# function to training data
def train(model, loss, optimiser, inputs, labels):

    inputs = Variable(inputs, requires_grad = False)
    labels = Variable(labels, requires_grad = False)

    optimiser.zero_grad()

    # forward
    logits = model.forward(inputs)
    output = loss.forward(logits, torch.max(labels,1)[1])

    # backward
    output.backward()

    # update parameters
    optimiser.step()

    return output.item()

# prediction procedure
def predict(model, inputs):
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits.data.numpy().argmax(axis=1)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
