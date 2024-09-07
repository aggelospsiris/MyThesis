import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Function to calculate precision and recall
def calc_precision_recall(doc_sims, relevant):
    cnt = 0
    retrieved = 1
    recall = []
    precision = []
    for doc in doc_sims:
        if doc in relevant:
            cnt += 1
            p = cnt / retrieved
            r = cnt / len(relevant)
            precision.append(p)
            recall.append(r)
        retrieved += 1

    avg_pre = sum(precision) / len(precision) if precision else 0
    avg_rec = sum(recall) / len(recall) if recall else 0

    return avg_pre, avg_rec


# Load embeddings
query_embeddings_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\query_embeddings.csv'
text_embeddings_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\text_embeddings.csv'
relevant_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\Relevant.txt'

df_query_embeddings = pd.read_csv(query_embeddings_path, index_col=0)
df_text_embeddings = pd.read_csv(text_embeddings_path, index_col=0)

# Load the relevant texts for each query
with open(relevant_path, 'r') as file:
    relevant_lines = file.readlines()

relevant_dict = {i: list(map(int, line.strip().split())) for i, line in enumerate(relevant_lines)}

# Calculate cosine similarity between each query and all texts
results = []

for query_id, query_embedding in df_query_embeddings.iterrows():
    query_embedding = query_embedding.values.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, df_text_embeddings.values)
    similarities = similarities.flatten()

    # Create a list of text IDs sorted by similarity
    sorted_texts = [df_text_embeddings.index[i] for i in np.argsort(-similarities)]
    sorted_texts_ids = [int(text_id.replace('text_', '')) for text_id in sorted_texts]
    # Ensure the query_id is an integer for accessing relevant_dict
    query_idx = int(query_id.replace('query_', ''))
    relevant = relevant_dict[query_idx]

    # Calculate precision and recall
    avg_precision, avg_recall = calc_precision_recall(sorted_texts_ids, relevant)
    
    results.append({
        'query_id': query_id,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall
    })

# Convert results to DataFrame and save
df_results = pd.DataFrame(results)
results_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\precision_recall_results_random_walk.csv'
df_results.to_csv(results_path, index=False)

# Calculate mean precision and recall
mean_precision = df_results['avg_precision'].mean()
mean_recall = df_results['avg_recall'].mean()

print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")