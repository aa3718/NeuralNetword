import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import itertools as it
import pickle
import math


def main():

    #######################################################################
    #                   LOAD HIDDEN DATASET HERE
    #######################################################################

    dataset = np.loadtxt("ROI_dataset.dat") # <-- Load hidden dataset
    np.random.shuffle(dataset)

    model = torch.load("best_model_ROI.pt")

    prediction_results = predict_hidden(model, dataset)

    accuracy = evaluate_architecture(model, dataset) #<- Please see Results-Part-3.txt for evaluation results

    return accuracy, prediction_results # <- Returns accuracy and an MxN array where N = sample size and M = number of outputs

    #######################################################################
    #           MAIN WILL RETURN ACCURACY AND ARRAY OF PREDICTED RESULTS
    #######################################################################


    split_index_1 = int(0.6 * len(dataset))
    split_index_2 = int(0.8 * len(dataset))

    split_index_3 = int(0.99 * len(dataset))

    data = dataset[split_index_3:,:]

    f = open("hypersearch2.txt", "a+")

    accuracy_best = -1;
    batches = [25, 50, 75]
    neurons =  [4, 8, 12, 16]
    epochs = [50, 100, 150, 200]
    learning_rates = [0.05, 0.1, 0.01]

    iterators = it.product(batches, neurons, epochs, learning_rates)
    best_stats = (0)

    for iter in iterators:
        accuracy, model = config_and_train(dataset, iter)

        current_param = [iter[0], iter[1], iter[2], iter[3]]
        f.write("accuracy: %f batches: %d neurons: %d epochs: %d learningrate: %f \r\n" % (accuracy, iter[0], iter[1], iter[2], iter[3]))

        if (accuracy > accuracy_best):
            accuracy_best = accuracy
            del(best_stats)
            best_stats = [accuracy, iter[0], iter[1], iter[2], iter[3]]
            torch.save(model, 'best_model.pt')

    # evaluate architecture
    testing_data = dataset[split_index_2:,:]
    evaluate_architecture(model, testing_data)

    return


def config_and_train(dataset, iterator):

    batch_size = iterator[0]
    neurons = iterator[1]
    num_epoch = iterator[2]
    learning_rate = iterator[3]

    # Hyper-parameters
    input_size = 3
    output_size = 4

    # split data into input and output columns

    x = dataset[:, :input_size]
    y = dataset[:, input_size:]

    split_index_1 = int(0.6 * len(x))
    split_index_2 = int(0.8 * len(x))

    # split data
    x_training = x[:split_index_1,:]
    y_training = y[:split_index_1,:]
    x_validation = x[split_index_1:split_index_2,:]
    y_validation = y[split_index_1:split_index_2,:]

    # convert numpy array into torch tensors
    x_training = torch.from_numpy(x_training).float()
    y_training = torch.from_numpy(y_training).long()

    x_validation = torch.from_numpy(x_validation).float()
    y_validation = torch.from_numpy(y_validation).long()

    # CREATE NETWORK
    model = torch.nn.Sequential()

    # layers x_dim -> 200 -> 200 -> y_dim using ReLU activation function
    model.add_module("dense1", torch.nn.Linear(input_size, neurons))
    model.add_module("relu1", torch.nn.ReLU())
    model.add_module("dense2", torch.nn.Linear(neurons, neurons))
    model.add_module("relu2", torch.nn.ReLU())
    model.add_module("dense3", torch.nn.Linear(neurons, output_size))

    # define an optimiser and learning rate
    optimiser = optim.Adam(model.parameters(), learning_rate, weight_decay = 0.0001)

    # learning rate decay
    #scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 10, gamma = 0.99)

    # define a loss function (cross entropy needed for classification problem)
    criterion = nn.CrossEntropyLoss()

    # get samples
    training_samples = len(x_training)
    training_batches = int(training_samples/batch_size)

    training_accuracy = []
    training_loss = []
    current_loss = 0


    for i in range(num_epoch):
        current_loss = 0

        # adjust the learning rate
        #scheduler.step()

        for j in range(training_batches):
            x_batch = x_training[j*batch_size:(j+1)*batch_size]
            y_batch = y_training[j*batch_size:(j+1)*batch_size]

            current_loss += train(model, criterion, optimiser, x_batch, y_batch)

        y_predict = predict(model, x_validation)
        y_actual = y_validation.data.numpy().argmax(axis=1)
        accuracy = np.mean(y_actual == y_predict)

        val_loss = validation_loss(model, criterion, x_validation, y_validation)

        print("Epoch: %d, cost for training: %f, cost for validation: %f, accuracy: %.5f" % (i, current_loss/training_batches, val_loss, accuracy))

    return accuracy, model


def validation_loss(model, criterion, inputs, labels):
    inputs = Variable(inputs, requires_grad = False)
    labels = Variable(labels, requires_grad = False)

    # forward
    outputs = model(inputs)
    loss = criterion(outputs, torch.max(labels,1)[1])

    return loss.item()

# function to training data
def train(model, criterion, optimiser, inputs, labels):

    inputs = Variable(inputs, requires_grad = False)
    labels = Variable(labels, requires_grad = False)

    optimiser.zero_grad()

    # forward
    outputs = model(inputs)
    loss = criterion(outputs, torch.max(labels,1)[1])

    # backward
    loss.backward()

    # update parameters
    optimiser.step()

    return loss.item()

# prediction procedure
def predict(model, inputs):

    inputs = Variable(inputs, requires_grad=False)
    logits = model(inputs)

    return logits.data.numpy().argmax(axis=1)

def evaluate_architecture(model, dataset):

    f = open("Results-Part-3.txt", "w")

    x_testing = dataset[:, :3]
    y_testing = dataset[:, 3:]
    x_testing = torch.from_numpy(x_testing).float()
    y_testing = torch.from_numpy(y_testing).long()

    y_predict = predict(model, x_testing)
    y_actual = y_testing.data.numpy().argmax(axis=1)

    f.write("Confusion Matrix: \r\n\r\n")
    f.write(str(confusion_matrix(y_actual,y_predict)))
    f.write("\r\n\r\nClassification Report: \r\n\r\n")
    f.write(str(classification_report(y_actual,y_predict)))

    accuracy = np.mean(y_actual == y_predict)
    return accuracy


def predict_hidden(model, dataset):

    input_size = 3

    x = dataset[:, :input_size]
    y = dataset[:, input_size:]

    x_testing = torch.from_numpy(x).float()
    y_testing = torch.from_numpy(y).long()

    y_predict = model(x_testing).data.numpy().argmax(axis=1)

    results = np.zeros(shape=(len(y), len(y[1])))

    for i in range(0, len(y)):

        if (y_predict[i] == 0):
            results[i] = [1, 0, 0, 0]
        if (y_predict[i] == 1):
            results[i] = [0, 1, 0, 0]
        if (y_predict[i] == 2):
            results[i] = [0, 0, 1, 0]
        if (y_predict[i] == 3):
            results[i] = [0, 0, 0, 1]

    return results



if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
