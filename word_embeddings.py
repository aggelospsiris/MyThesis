import networkx as nx
import re
import numpy as np
import pandas as pd
from pecanpy.node2vec import SparseOTF
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from typing import Dict, Tuple, Set,List

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt')

def load_inverted_index(data_file_path: str) -> Tuple[Dict[str, Dict[int, int]], Dict[str, float], Dict[int, Dict[str, int]], Set[int]]:
    """
    Load inverted index data from the given file path and return the required components for building the graph.

    Args:
        data_file_path (str): Path to the file containing the inverted index data.

    Returns:
        Tuple[Dict[str, Dict[int, int]], Dict[str, float], Dict[int, Dict[str, int]], Set[int]]:
            - word_docs: Dictionary where keys are words and values are dictionaries with document IDs and term frequencies.
            - word_importance: Dictionary where keys are words and values are document frequencies.
            - doc_words: Dictionary where keys are document IDs and values are dictionaries with words and term frequencies.
            - doc_ids: Set of document IDs.
    """
    word_docs = {}  # {word: {doc_id: tf}}
    word_importance = {}    # {word: importance}
    doc_words = {}  # {doc_id: {word: tf}}
    doc_ids = set()

    with open(data_file_path, 'r') as file:
        for line in file:
            match = re.match(r'^(\d+);([^;]+);([^;]+);(.+)$', line.strip())
            if match:
                word_id = match.group(1)
                importance = float(match.group(2))
                label = match.group(3).lower()
                word_importance[label] = importance
                pairs = re.findall(r'\[(\d+), (\d+)\]', match.group(4))
                word_docs[label] = {int(doc_id): int(tf) for doc_id, tf in pairs}
                for doc_id, tf in pairs:
                    doc_id = int(doc_id)
                    tf = int(tf)
                    doc_ids.add(doc_id)
                    if doc_id not in doc_words:
                        doc_words[doc_id] = {}
                    doc_words[doc_id][label] = tf

    return word_docs, word_importance, doc_words, doc_ids

def filter_low_importance_words(word_docs: Dict[str, Dict[int, int]], word_importance: Dict[str, float], doc_words: Dict[int, Dict[str, int]], threshold_divisor: float) -> Tuple[Dict[str, Dict[int, int]], Dict[str, float], Dict[int, Dict[str, int]]]:
    """
    Calculate the average importance, filter out low importance words, and update doc_words to exclude low importance words.

    Args:
        word_docs (Dict[str, Dict[int, int]]): Dictionary with words and their document term frequencies.
        word_importance (Dict[str, float]): Dictionary with words and their importance scores.
        doc_words (Dict[int, Dict[str, int]]): Dictionary with document IDs and their words with term frequencies.
        threshold_divisor (float): The divisor for calculating the importance threshold.

    Returns:
        Tuple[Dict[str, Dict[int, int]], Dict[str, float], Dict[int, Dict[str, int]]]:
            - filtered_word_docs: Filtered dictionary with words and their document term frequencies.
            - filtered_word_importance: Filtered dictionary with words and their importance scores.
            - filtered_doc_words: Updated dictionary with document IDs and their words with term frequencies.
    """
    average_importance = np.mean(list(word_importance.values()))
    threshold = average_importance / threshold_divisor

    filtered_word_docs = {word: docs for word, docs in word_docs.items() if word_importance[word] >= threshold}
    filtered_word_importance = {word: importance for word, importance in word_importance.items() if importance >= threshold}

    filtered_doc_words = {}
    for doc_id, words in doc_words.items():
        filtered_doc_words[doc_id] = {word: tf for word, tf in words.items() if word in filtered_word_importance}

    return filtered_word_docs, filtered_word_importance, filtered_doc_words

def create_word_graph(filtered_word_docs: Dict[str, Dict[int, int]], filtered_word_importance: Dict[str, float]) -> nx.Graph:
    """
    Create a graph with word nodes and weighted edges based on their co-occurrence in common texts,
    weighted by the importance of the words.
    """
    G = nx.Graph()
    
    # Add word nodes with importance attribute
    for word in filtered_word_docs.keys():
        G.add_node(word, importance=filtered_word_importance.get(word, 1.0))
    
    # Dictionary to hold edge weights
    word_pairs = {}
    
    # Iterate over all pairs of words
    for word1, docs1 in filtered_word_docs.items():
        for word2, docs2 in filtered_word_docs.items():
            if word1 >= word2:  # Avoid duplicate pairs and self-loops
                continue
            
            # Find common texts where both words appear
            common_texts = set(docs1.keys()).intersection(docs2.keys())
            if not common_texts:
                continue
            
            # Compute total weight for this pair
            total_weight = 0
            for doc_id in common_texts:
                occ_word_1 = docs1[doc_id]
                occ_word_2 = docs2[doc_id]
                importance_word_1 = filtered_word_importance.get(word1, 1.0)
                importance_word_2 = filtered_word_importance.get(word2, 1.0)
                weight = (importance_word_1 * occ_word_1 + importance_word_2 * occ_word_2)
                total_weight += weight
            
            # Add the edge if weight is positive
            if total_weight > 0:
                G.add_edge(word1, word2, weight=total_weight)
    
    return G


def generate_word_embeddings(word_graph: nx.Graph, embedding_size: int, walk_length: int, num_walks: int, window_size: int, p: float, q: float, edg_file_path: str) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for the word nodes in the provided graph using Node2Vec.

    Args:
        word_graph (nx.Graph): The graph with word nodes and edges.
        embedding_size (int): The size of the embeddings.
        walk_length (int): Length of each random walk.
        num_walks (int): Number of random walks to perform.
        window_size (int): Window size for Word2Vec model.
        p (float): Return parameter for Node2Vec.
        q (float): In-out parameter for Node2Vec.
        edg_file_path (str): Path to save the edge list file.

    Returns:
        Dict[str, np.ndarray]: A dictionary with nodes as keys and their embeddings as values.
    """
    # Write the graph to an edge list file in the correct format
    with open(edg_file_path, 'w') as f:
        for u, v, data in word_graph.edges(data=True):
            f.write(f"{u}\t{v}\t{data['weight']}\n")
    
    # Use pecanpy for generating embeddings
    g = SparseOTF(p=p, q=q, workers=1, verbose=False)
    g.read_edg(edg_file_path, weighted=True, directed=False)
    walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length, n_ckpts=10, pb_len=50)
    model = Word2Vec(walks, vector_size=embedding_size, window=window_size, min_count=0, sg=1, workers=1, epochs=1)
    
    # Generate embeddings for nodes
    embeddings = {}
    for node in word_graph.nodes:
        try:
            embeddings[node] = model.wv[str(node)]
        except KeyError:
            print(f"Warning: Embedding for node {node} not found. Setting to random embedding.")
            embeddings[node] = np.random.rand(embedding_size)
    
    return embeddings

# Initialize dictionaries and load data
data_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\complete_index.dat'
# Load inverted index data
word_docs, word_importance, doc_words, doc_ids = load_inverted_index(data_file_path)
# Filter low importance words
filtered_word_docs, filtered_word_importance, filtered_doc_words = filter_low_importance_words(word_docs, word_importance, doc_words, threshold_divisor=46)

# Assuming you have already created your filtered_word_docs and filtered_word_importance
word_graph = create_word_graph(filtered_word_docs, filtered_word_importance)

# Parameters for Node2Vec and Word2Vec
embedding_size = 64
walk_length = 30
num_walks = 500
window_size = 5
p = 0.5
q = 2
edg_file_path = 'word_graph.edgelist'

# Generate word embeddings
embeddings = generate_word_embeddings(word_graph, embedding_size, walk_length, num_walks, window_size, p, q, edg_file_path)

# Save embeddings to CSV
word_embeddings_df = pd.DataFrame(embeddings).T
word_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\word_embeddings.csv', header=False)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(embeddings: Dict[str, np.ndarray], title: str = 'Word Embeddings') -> None:
    """
    Visualize word embeddings using t-SNE.

    Args:
        embeddings (Dict[str, np.ndarray]): A dictionary with nodes as keys and their embeddings as values.
        title (str): The title for the plot.
    """
    # Convert embeddings to a numpy array
    words = list(embeddings.keys())
    vectors = np.array([embeddings[word] for word in words])

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=0)
    reduced_vectors = tsne.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], marker='o')

    # Annotate each point with the corresponding word
    for i, word in enumerate(words):
        plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word, fontsize=9)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()