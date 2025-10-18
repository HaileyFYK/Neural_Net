import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 8})
# Load one row of data from the CSV file
df = pd.read_csv('sign_mnist_13bal_train.csv')
row = df.iloc[7]

# Extract the image data (excluding the label)
image_data = row[1:].values
image_data = image_data.astype(np.uint8)  # Convert to unsigned integer for image display

# Reshape the data into a 28x28 grid
image_28x28 = image_data.reshape(28, 28)

# Display the image with numerical values overlaid and row/column labels
fig, ax = plt.subplots(figsize=(7, 7))
cax = ax.matshow(image_28x28, cmap='gray')

# Add color bar for reference
plt.colorbar(cax)

# Adding grid labels
for (i, j), val in np.ndenumerate(image_28x28):
    ax.text(j, i, val, ha='center', va='center', color='red', fontsize=5)

# Label rows and columns
ax.set_xticks(np.arange(0, 28, 1))
ax.set_yticks(np.arange(0, 28, 1))
ax.set_xticklabels(np.arange(1, 29, 1))
ax.set_yticklabels(np.arange(1, 29, 1))

plt.show()