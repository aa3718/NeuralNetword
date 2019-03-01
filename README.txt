_   _                            _   _   _        _                           _
| \ | |                          | | | \ | |      | |                         | |
|  \| |  ___  _   _  _ __   __ _ | | |  \| |  ___ | |_ __      __  ___   _ __ | | __ ___
| . ` | / _ \| | | || '__| / _` || | | . ` | / _ \| __|\ \ /\ / / / _ \ | '__|| |/ // __|
| |\  ||  __/| |_| || |   | (_| || | | |\  ||  __/| |_  \ V  V / | (_) || |   |   < \__ \
\_| \_/ \___| \__,_||_|    \__,_||_| \_| \_/ \___| \__|  \_/\_/   \___/ |_|   |_|\_\|___/


README

This collection of files contain an implementation of a neural network library along with two neural network
functions to learn both forward motion and a region of interest for an ABB IRB 120 Robot.

The Neural Network Library is implemented in Python and contains everything needed to create, train and evaluate
a simple neural network.

The learn_FM function is designed to be able to learn and predict the forward model of the ABB IRB 120 and predict
the location of the gripper in 3D space, given the angle of each of the three motors in the arm.

The learn_ROI function is designed to predict which of four different zones the robots gripper is in. These zones are
designated as: 1) The workspace infront of the robot, 2) Two small zones either side of the main workspace,
3) The ground and 4) The rest of the workspace. These are encoded through a one hot encoding scheme.


FILE MANIFEST

 README.txt
 nn_lib.py - Neural Network Library.
 learn_FM.py - Learn forward movement of the ABB IRB 120 Robot.
 learn_ROI.py - Learn reigon of interest of the ABB IRB 120 Robot.
 iris.dat - Iris data set used for testing.
 FM_dataset.dat - Data set for the forward motion of the robot.
 ROI_dataset.dat - Data set for the Reigon Of Interest detector.
 illustrate.py - Code for showing the ABB IRB 120 (when used with nn_lib)
 simulator.py - Simulator for use with nn_lib.py file.
 best_mode.pt - Best model for use with predict_hidden.

PREREQUISITS

 The nn_lib only requires Python 3 and Numpy to run whilst the learn_FM and learn_ROI require
 Python 3, Numpy, Scikit-Learn, Conda and Pytorch.


SETTINGS

 Neural Networks Library

     The settings of the Neural Network are initialised on the creation of the Neural Network class through
     calling MultiLayerNetwork(input_dim, neurons, activations), where input_dim is the number of inputs,
     neurons is an array containing the number of neurons in each layer and activations is an array containing
     the activation function to use between each layer.

     The training parameters are changed through the arguments passed to
     Trainer(net, batch_size, epoch, learning_rate, loss_fun, shuffle_flag) where net is the network created in
     the previous paragraph, batch_size is the batch size, epoch is the number of epochs, learning_rate is the
     learning rate to be used, loss_fun is the name of the loss function and shuffle flag is if the data set should
     be shuffled each time.

 Learn FM

     Learn FM uses the Pytorch Library to implement a neural network. This is created in the config_and_train function
     and then trained on the dataset which is passed into it. Settings are passed in through the iterator variable which
     is an array containing batchsize, neurons, epochs, learningrate and dropout.

 Learn ROI
     Learn ROI uses the Pytorch Library to implement a neural network. Settings are passed in through the iterator variable which
     is an array containing batchsize, neurons, epochs, and learningrate.



RUNNING

 Neural Networks Library

     To run the Neural Network Library, a main function must be created which includes the library before creating
     a network which is then passed into the trainer class to be trained. Optionally, a Preprocessor class can be
     created to pre-process the data and convert it into the range 0-1 before being passed into the neural network.

     The nn_lib function contains an example on how to structure the code to run the library. This can be called by
     running Python3 nn_lib.py.

 Learn FM

     Learn_FM contains several functions which perform the traning and evaluation of the network on a given dataset.

     evaluate_architecture(model, x_data, y_data) takes a trained model as an argument along with some input data and
     expected outputs and prints and returns the R2 Score, Mean Squared Error and Explained Variance Score for that data.

     predict_hidden(dataset) loads the best model, as saved under best_model.pt and evaluates the given dataset on this
     model, returning an array of the predictions. It expects dataset to contain 3 columns to match the number of inputs
     to the network.

     For the evaluation of the hidden dataset, the function can be imported from the library and called on the desired
     dataset, or alternatively the main can be changed to load the desired dataset and executed through Python3 learn_FM.py

 Learn ROI

     At the beginning of the main of the learn_ROI.py, after the comment “load given dataset here”, enter the file name of
     the dataset and run the program.

     Both the predict_hidden( ) and evaluate_architecture( ) functions will be executed on the hidden dataset.

     The predict_hidden( ) results of your best model will be returned as an (n * m) size array, where n is equal to the
     sample size and m is equal to the number of outputs.

     The evaluate_architecture( ) results will be streamed into a file named “Results-Part-3”. The results printed in the
     file are the confusion matrix and the classification report.



OUTPUT

 Neural Networks Library

     To run the Neural Network Library, a main function must be created which includes the library before creating
     a network which is then passed into the trainer class to be trained. Optionally, a Preprocessor class can be
     created to pre-process the data and convert it into the range 0-1 before being passed into the neural network.

     The nn_lib function contains an example on how to structure the code to run the library. This can be called by
     running Python3 nn_lib.py.

 Learn FM

     Depending on which functions are used as part of the learn_FM, the learn_FM function will either return the
     predictions given the input angles or if the model is being trained, the accuracy of the model for that dataset.

 Learn ROI

     The main( ) function return the final accuracy and prediction results of the hidden dataset.



TROUBLESHOOTING


CONTACT

 Alexander Luisi - alexander.luisi15@imperial.ac.uk
 Allegra Anka-Mueller - allegra.anka-mueller18@imperial.ac.uk
 Olivia Stannah - olivia.stannah18@imperial.ac.uk
 Owen Harcombe - owen.harcombe15@imperial.ac.uk
