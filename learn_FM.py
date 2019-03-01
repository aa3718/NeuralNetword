import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset

from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV

import numpy as np
import itertools as it
import pickle
import math

from nn_lib import Preprocessor
#from illustrate import illustrate_results_FM

# Perform a hyper-parameter search
def hyperparam(training_set, validation_set):
    # Log results
    f = open("hypersearch.txt", "a+")

    r2_best_score = -1;
    #paramgrid = {'neurons':[5,10,15,20,25,30], 'epoch':[500, 1000, 1500], 'learning':[0.01, 0.1], 'batch':[20,40,60,80,100,120]}
    batches = [25, 75, 125, 175, 225]
    neurons =  [5,10,15,20,25,30]
    epochs = [100, 200, 400, 800, 1000]
    learning_rates = [0.01, 0.1]
    dropout = [0.0, 0.25, 0.5]

    iterators = it.product(batches, neurons, epochs, learning_rates, dropout)
    best_stats = (0,0,0)
    for iter in iterators:
        network, mse, evs, r2 = config_and_train(training_set, validation_set, iter)

        current_param = [iter[0], iter[1], iter[2], iter[3], iter[4]]
        f.write("mse: %d evs: %f r2: %f batches: %d neurons: %d epochs: %d learningrate: %f dropout: %f \r\n" % (mse, evs, r2, iter[0], iter[1], iter[2], iter[3], iter[4]))

        if(r2 > r2_best_score):
            # save the r2 score and the best model
            r2_best_score = r2
            del(best_stats)
            best_stats = [mse, evs, r2, iter[0], iter[1], iter[2], iter[3], iter[4]]
            torch.save(network, 'best_model.pt')

    f.close()
    # return the best model
    return torch.load('best_model.pt'), best_stats

# Evaluate the performance of the model using validation data
def evaluate_architecture(network, x_val_for_model, y_val):
    results = network(x_val_for_model)
    print("Mean Squared Error: " + str(mean_squared_error(y_val, results.data.numpy())))
    print("Explained Variance Score: " + str(explained_variance_score(y_val, results.data.numpy())))
    print("R2 Score: " + str(r2_score(y_val, results.data.numpy())))

    return mean_squared_error(y_val, results.data.numpy()), explained_variance_score(y_val, results.data.numpy()), r2_score(y_val, results.data.numpy())

# Configure the network based on the call from hyperparam
def config_and_train(training, validation, iterator):
#def config_and_train(dataset, iterator):

    batchsize = iterator[0]
    neurons = iterator[1]
    epochs = iterator[2]
    learning_rate = iterator[3]
    dropout = iterator[4]

    # Confgure the dataset
    np.random.shuffle(training)
    np.random.shuffle(validation)

    #x = dataset[:, :3]
    #y = dataset[:, 3:]

    #split_index = int(0.8 * len(x))

    x_training = training[:, :3]
    y_training = training[:, 3:]
    x_validation = validation[:, :3]
    y_validation = validation[:, 3:]

    prep_input = Preprocessor(x_training)

    x_train_pre = prep_input.apply(x_training)
    x_val_pre = prep_input.apply(x_validation)

    x_dim = x_training.shape[1] # get number of input neurons
    y_dim = y_training.shape[1] # get number of output neurons

    # Create the network
    model = torch.nn.Sequential();

    model.add_module("dense1", torch.nn.Linear(x_dim, neurons))
    model.add_module("sig1", torch.nn.Sigmoid())
    model.add_module("dropout1", torch.nn.Dropout(dropout))
    model.add_module("dense2", torch.nn.Linear(neurons, y_dim))


    # define an optimiser and learning rate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Look into Adam?

    # define a loss function (MSE needed for regression problem)
    loss_function = nn.MSELoss()

    # convert numpy array into torch tensors
    x_training_pre = torch.from_numpy(x_train_pre).float()
    y_training = torch.from_numpy(y_training).float()
    x_validation_pre = torch.from_numpy(x_val_pre).float()
    y_validation = torch.from_numpy(y_validation).float()

    # Set up the dataset to use the dataloader
    tensor_training_set = TensorDataset(x_training_pre, y_training)
    train_dataloader = DataLoader(tensor_training_set, batch_size = batchsize, shuffle = True)

    training_samples = x_training.shape[0]
    training_batches = training_samples/batchsize

    valid_ds = TensorDataset(x_validation_pre, y_validation)
    valid_dataloader = DataLoader(valid_ds, batch_size = batchsize * 2)

    for i in range(epochs):

        #print("Epoch Number: " + str(i))
        # Shuffle the data

        current_loss = 0
        # set the model to training mode
        model.train()
        for xb, yb in train_dataloader:
            pred = model(xb)
            #loss = loss_function(pred, (torch.max(yb, 1)[1]))
            loss = loss_function(pred, yb)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            #print("Epoch: %d, cost: %f, accuracy: %.2f" % (i, current_loss/training_batches, loss_function(model(xb), torch.max(yb, 1)[1]))

            #illustrate_results_FM(model, prep_input)
    print(loss_function(model(xb), yb))
    #print("Training End")

    x_prime = x_val_pre

    # Evaluate in the hyperparam function
    mse, evs, r2 = evaluate_architecture(model, x_validation_pre, y_validation)

    # return the model, the mean squared error, the explained variance score and the r2
    return model, mse, evs, r2

# Loads the best model and returns a matrix of observations
def predict_hidden(dataset):
    # Confgure the dataset
    np.random.shuffle(dataset)

    # Preprocess the incoming dataset
    prep_input = Preprocessor(dataset)
    test_set = prep_input.apply(dataset)

    testset = torch.from_numpy(test_set).float()

    model = torch.load('best_model_FM.pt')

    predictions = model(testset)

    print("Predictions from best model: " + str(predictions.data.numpy()))
    return predictions.data.numpy()

def test_model(network, testset):
    # shuffle the testset
    np.random.shuffle(testset)

    x_test = testset[:, :3]
    y_test = testset[:, 3:]

    prep_input = Preprocessor(x_test)
    x_test_pre = prep_input.apply(x_test)

    x_test_torch = torch.from_numpy(x_test_pre).float()

    # get the result of testing on the best model
    mse, evs, r2 = evaluate_architecture(network, x_test_torch, y_test)

    # return the the mean squared error, the explained variance score and the r2
    stats = (mse, evs, r2)
    return stats


def main():
    # Import the dataset

    dataset = np.loadtxt("FM_dataset.dat")
    np.random.shuffle(dataset)

    ###### Lines for testing. Please remove for other uses.
    print(predict_hidden(dataset[:,:3]))
    return
    ###### End of testing.

    # Split into training, validation
    # (and test?)
    split_index = int(0.7 * len(dataset)) # take the top 70% of the dataset, and 20% for validation
    split_test = int(0.9 * len(dataset)) # take 10% for test

    training = dataset[:split_index]
    validation = dataset[split_index:split_test]
    test = dataset[split_test:]

    # Get out the best model from training and validation
    best_model, best_stats = hyperparam(training, validation)
    print("Loaded best model, with validation results (mse, evs, r2, batch, neurons, epochs, lr, dropout): " + str(best_stats))

    best_model = torch.load('best_model_FM.pt')

    test_results = test_model(best_model, test)
    print("Best model gave the following results on the test data (mse, evs, r2, batch, neurons, epochs, lr, dropout)")

    #predict_hidden(dataset)

    print("End of main")




if __name__ == "__main__":
    main()
