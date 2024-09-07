import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# Load the embeddings CSV
file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\word_embeddings.csv'
embeddings_df = pd.read_csv(file_path)

# Assuming the first column is the word and the rest are the embedding vectors
words = embeddings_df.iloc[:, 0].values  # First column contains words
vectors = embeddings_df.iloc[:, 1:].values  # Remaining columns contain the embeddings

# Apply t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=0)
reduced_vectors = tsne.fit_transform(vectors)

# Plotting
plt.figure(figsize=(12, 10))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], marker='o')


plt.title('Word Embeddings Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()