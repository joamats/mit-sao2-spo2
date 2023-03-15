import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Load data, already split 80-20
X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')['hidden_hypoxemia']

# Create a t-SNE model with 2 components
tsne = TSNE(n_components=2)

# Fit the t-SNE model to the data
X_tsne = tsne.fit_transform(X_train)

for i in range(len(X_tsne)):
    if y_train[i] == 0:
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='b')
    else:
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='r')

# Add title and axis labels
plt.title("t-SNE Plot")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# Save the plot as a PNG file
plt.savefig("results/tsne_train.png")