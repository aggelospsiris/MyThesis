import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load text embeddings
text_embeddings_df = pd.read_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\text_embeddings.csv', header=None, index_col=0)

# Load query embeddings
query_embeddings_df = pd.read_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\query_embeddings.csv', header=None, index_col=0)

# Combine text and query embeddings for visualization
combined_embeddings_df = pd.concat([text_embeddings_df, query_embeddings_df])

# Prepare data for t-SNE
embeddings = combined_embeddings_df.values
node_labels = combined_embeddings_df.index

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Create a DataFrame with t-SNE results
tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
tsne_df['Node'] = node_labels

# Plot t-SNE results
plt.figure(figsize=(12, 8))

# Plot text embeddings
text_tsne_df = tsne_df[tsne_df['Node'].str.startswith('text')]
plt.scatter(text_tsne_df['Dimension 1'], text_tsne_df['Dimension 2'], color='blue', label='Text Nodes', alpha=0.6)

# Plot query embeddings with annotations
query_tsne_df = tsne_df[tsne_df['Node'].str.startswith('query')]
plt.scatter(query_tsne_df['Dimension 1'], query_tsne_df['Dimension 2'], color='red', label='Query Nodes', alpha=0.6)

# Annotate query nodes with their IDs
for i, row in query_tsne_df.iterrows():
    plt.annotate(row['Node'], (row['Dimension 1'], row['Dimension 2']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red')

# Set zoom by adjusting the limits of the plot
x_min, x_max = tsne_df['Dimension 1'].min() - 5, tsne_df['Dimension 1'].max() + 5
y_min, y_max = tsne_df['Dimension 2'].min() - 5, tsne_df['Dimension 2'].max() + 5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.title('t-SNE Visualization of Node Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
