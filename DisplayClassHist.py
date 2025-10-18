import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets - assuming they are in the current working directory
# and have the filenames 'sign_mnist_13bal_train.csv'
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Plot histograms for the 'class' column in both datasets
plt.figure(figsize=(6, 4))

# Histogram for the training data

plt.hist(train_data['class'], bins=len(train_data['class'].unique()),
         rwidth=0.8, alpha=0.7, color='blue')
plt.title('Histogram of class column in training data')
plt.xlabel('Class')
plt.ylabel('Frequency')


# Show the plot
plt.tight_layout()
plt.show()