import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_train = pd.read_csv('sign_mnist_13bal_train.csv')
# Label is in the first column
y_train = df_train.iloc[:, 0]  
# Features are in all other columns, normalized
X_train = df_train.iloc[:, 1:] / 255.0  

print(df_train)