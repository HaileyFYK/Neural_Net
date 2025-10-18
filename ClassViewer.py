import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('sign_mnist_13bal_train.csv')

# which class to display
class_number = 7

# Filter the dataset for one class, and select the first 10 rows
class_0_df = df[df['class'] == class_number].head(10)

# Plotting the first 10 images of class 0
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(6, 4))

for i, row in enumerate(class_0_df.iterrows()):
    image_data = row[1][1:].values  # Exclude the label
    # Convert to unsigned integer for image display
    image_data = image_data.astype(np.uint8)  
    # Reshape the data into a 28x28 grid
    image_28x28 = image_data.reshape(28, 28)  

    # Plot each image
    ax = axes[i // 5, i % 5]
    ax.imshow(image_28x28, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()