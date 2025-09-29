import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Load the embeddings file
data = np.load("embeddings.npz")

# Extract embeddings and labels
embeddings = data['arr_0']
labels = data['arr_1']

# Reduce dimensionality using PCA before t-SNE (helps with stability)
pca = PCA(n_components=50)  # Reduce to 50 dimensions before t-SNE
embeddings_pca = pca.fit_transform(embeddings)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_pca)

# Plot the 2D visualization
plt.figure(figsize=(12, 8))

# Map string labels to integers for coloring
unique_labels = np.unique(labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = np.array([label_mapping[label] for label in labels])

# Create scatter plot
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], 
    c=numeric_labels, cmap='jet', alpha=0.7
)

# Add class names to the plot
for i, label in enumerate(labels):
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=7, alpha=0.8, color='black')

# Create a legend for class names
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=scatter.cmap(scatter.norm(idx)), markersize=10) 
           for idx in range(len(unique_labels))]

plt.legend(handles, unique_labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()  # Adjust layout for better visualization
plt.show()
