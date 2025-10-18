import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X = X_train / 255.0
y = train_data['class']# Target (first column)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 50, random_state = 2, stratify = y) 

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the  classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Use this line to get you started on adding a validation dataset
#X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=10, random_state=0)

neural_net_model = MLPClassifier( hidden_layer_sizes=(35),random_state=42,tol=0.005)

neural_net_model.fit(X_train, y_train)
# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Training set size: {len(y_train)}")
print(f"Layer sizes: {layer_size_str}")


# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred_validate = neural_net_model.predict(X_validate)
y_pred_test = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts_validate = defaultdict(int)
total_counts_validate = defaultdict(int)
overall_correct_validate = 0
correct_counts_test = defaultdict(int)
total_counts_test = defaultdict(int)
overall_correct_test = 0

for true, pred in zip(y_validate, y_pred_validate):
    total_counts_validate[true] += 1
    if true == pred:
        correct_counts_validate[true] += 1
        overall_correct_validate += 1

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred_test):
    total_counts_test[true] += 1
    if true == pred:
        correct_counts_test[true] += 1
        overall_correct_test += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1


# Calculate and print accuracy for each class and overall test accuracy

for class_id in sorted(total_counts_validate.keys()):
    accuracy = correct_counts_validate[class_id] / total_counts_validate[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")

overall_accuracy_test = overall_correct_test / len(y_test)*100
print(f"Overall Test Accuracy: {overall_accuracy_test:3.1f}%")

overall_accuracy_validate = overall_correct_validate / len(y_validate)*100
print(f"Overall Validate Accuracy: {overall_accuracy_validate:3.1f}%")

overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")