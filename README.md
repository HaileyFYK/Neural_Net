# Neural Net and Machine Learning

### Introduction
This program is a neural network that allows to recognize things or image through image classification and machine learning. Neural networks enable computers to make intelligent decisions with limited human assistance and it have numerous use cases across many industries, such as medical diagnosis through medical image classification, targeted marketing through social network screening and behavioral data analysis, and financial forecasting by processing historical data on financial instruments.

### Instruction

When it's given a set of data by user, the data is split into three groups, training set, validation set, and test set. Training set is used to teach the model, validation set is used to check how well the model is learning during training, and test set is used at the end to test how well the model can recognize new images it hasn't seen before. After the model is trained on the training data, it makes predictions on all three sets to see how accurately it can guess correctly.

Then, the code counts how many predictions were correct for each type of condition and calculates the overall success rate for each dataset, which helps to understand how well the model has learned and whether it's able to correctly recognize signs it hasn’t seen before.

### Explanation and File Overview

*Main.py is the main program.

*ClassViewer.py displays the first 10 images of the class `class_number`.
User can run the program by typing `python ClassViewer.py` in the Shell tab.

*DataDumper.py
This utility program displays the first 5 and last 5 rows  and the first two and last two columns of the SignMNIST data set.
User can run the program by typing `python ClassViewer.py` in the Shell tab.


*DisplayClassHist.py
This utility program displays a histogram of the SignMNIST data set classes.
User can run the program by typing `python DisplayClassHist.py` in the Shell tab.

*ShowPixelGrid.py
This utility program displays a one image from the SignMNIST data set classes with the numerical pixel values overlayed  on each pixel. You can select which row of the dataset is displayed by changing the row number in the iloc location: `row = df.iloc[7]`
User can run the program by typing `python DisplayClassHist.py` in the Shell tab.

*sign_mnist_13bal_test.csv
This is a comma separated (CSV) formatted file containing a test portion from a balanced sample of 13 images from each class in the SignMNIST dataset.

*sign_mnist_13bal_train.csv
This is a comma separated (CSV) formatted file containing a training portion a balanced sample of 13 images from each class in the SignMNIST dataset.


### Additional Information
Sign MNIST dataset source: https://www.openml.org/search?type=data&status=active&id=45082