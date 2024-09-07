import networkx as nx
import re
import numpy as np
import pandas as pd
from pecanpy.node2vec import SparseOTF
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from typing import Dict, Tuple, Set,List
from sklearn.neighbors import NearestNeighbors

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

def create_graph(doc_ids: Set[int], filtered_word_docs: Dict[str, Dict[int, int]], filtered_word_importance: Dict[str, float], filtered_doc_words: Dict[int, Dict[str, int]]) -> nx.Graph:
    """
    Create a graph with text nodes, word nodes, and weighted edges based on term frequencies and importance.

    Args:
        doc_ids (Set[int]): Set of document IDs.
        filtered_word_docs (Dict[str, Dict[int, int]]): Filtered dictionary with words and their document term frequencies.
        filtered_word_importance (Dict[str, float]): Filtered dictionary with words and their importance scores.
        filtered_doc_words (Dict[int, Dict[str, int]]): Filtered dictionary with document IDs and their words with term frequencies.

    Returns:
        nx.Graph: The created graph with nodes and edges.
    """
    # Create an undirected graph
    G = nx.Graph()

    # Add text nodes
    for doc_id in doc_ids:
        G.add_node(f'text_{doc_id}', type='text')

    # Add word nodes and edges with weights between words and texts
    for word, docs in filtered_word_docs.items():
        G.add_node(word, type='word')
        for doc_id, tf in docs.items():
            importance = filtered_word_importance[word]
            weight = importance * tf
            G.add_edge(word, f'text_{doc_id}', weight=weight)

    # Add edges between text nodes based on shared words with the calculated threshold
    for doc1 in doc_ids:
        for doc2 in doc_ids:
            if doc1 >= doc2:
                continue
            common_words = set(filtered_doc_words[doc1].keys()).intersection(set(filtered_doc_words[doc2].keys()))
            if not common_words:
                continue
            weight = sum((max(filtered_word_docs[word][doc2], filtered_word_docs[word][doc1]) * filtered_word_importance[word]) for word in common_words if word in filtered_word_docs)
            if weight > 0:
                G.add_edge(f'text_{doc1}', f'text_{doc2}', weight=weight)

    # Create the text subgraph
    text_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'text']
    text_subgraph = G.subgraph(text_nodes).copy()

    return text_subgraph

def generate_text_embeddings(text_subgraph: nx.Graph, embedding_size: int, walk_length: int, num_walks: int, window_size: int, p: float, q: float, edg_file_path: str) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for the text nodes in the provided graph using Node2Vec.

    Args:
        text_subgraph (nx.Graph): The graph with text nodes and edges.
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
    # Write the subgraph to an edge list file in the correct format
    with open(edg_file_path, 'w') as f:
        for u, v, data in text_subgraph.edges(data=True):
            f.write(f"{u}\t{v}\t{data['weight']}\n")

    # Use pecanpy for generating embeddings
    g = SparseOTF(p=p, q=q, workers=1, verbose=False)
    g.read_edg(edg_file_path, weighted=True, directed=False)
    walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length, n_ckpts=10, pb_len=50)
    model = Word2Vec(walks, vector_size=embedding_size, window=window_size, min_count=0, sg=1, workers=1, epochs=1)

    # Generate embeddings for nodes
    embeddings = {}
    for node in text_subgraph.nodes:
        try:
            embeddings[node] = model.wv[str(node)]
        except KeyError:
            print(f"Warning: Embedding for node {node} not found. Setting to random embedding.")
            embeddings[node] = np.random.rand(embedding_size)

    return embeddings

# Function to simulate biased random walks for a query
def simulate_biased_random_walks_for_query(G, query_node, text_embeddings, embedding_size, walk_length, num_walks, p, q):
    query_embedding = np.zeros(embedding_size)
    walks = []

    def get_alias_edge(src, dst):
        unnormalized_probs = []
        for nbr in G.neighbors(dst):
            weight = G[dst][nbr]['weight']
            if nbr == src:
                unnormalized_probs.append(weight / p)
            elif G.has_edge(nbr, src):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return normalized_probs

    for _ in range(num_walks):
        current_node = query_node
        walk_embedding = np.zeros(embedding_size)
        prev_node = None
        walk = [str(current_node)]

        for _ in range(walk_length):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break

            if current_node == query_node:
                probabilities = [G[current_node][neighbor]['weight'] for neighbor in neighbors]
                probabilities = probabilities / np.sum(probabilities)
            else:
                probabilities = get_alias_edge(prev_node, current_node)

            next_node = np.random.choice(neighbors, p=probabilities)
            walk.append(str(next_node))
            if next_node in text_embeddings:
                walk_embedding += text_embeddings[next_node]
            prev_node = current_node
            current_node = next_node

        walks.append(walk)
        query_embedding += walk_embedding / walk_length

    # Train a Word2Vec model on the generated walks
    query_model = Word2Vec(walks, vector_size=embedding_size, window=5, min_count=0, sg=1, workers=1, epochs=1)

    # Combine embeddings from biased random walks and node2vec model
    for node in G.nodes:
        if node.startswith('query_'):
            try:
                query_embedding += query_model.wv[str(node)]
            except KeyError:
                print(f"Warning: Embedding for query node {node} not found in node2vec model.")
                query_embedding += np.random.rand(embedding_size)

    return query_embedding / num_walks

def load_and_tokenize_queries(queries_file_path: str) -> List[List[str]]:
    """
    Load and process queries from a file and tokenize them.

    Args:
        queries_file_path (str): Path to the file containing queries.

    Returns:
        List[List[str]]: A list of tokenized queries, where each query is a list of words.
    """
    queries = []
    
    # Read queries from the file
    with open(queries_file_path, 'r') as file:
        for line in file:
            query = line.strip().lower()
            queries.append(query)

    # Tokenize queries
    query_words = [word_tokenize(query) for query in queries]
    
    return query_words

# Expand queries using KNN
def expand_query(query, word_embeddings, k=5):
    expanded_query = set(query)
    all_word_vecs = np.array(list(word_embeddings.values()))
    all_words = list(word_embeddings.keys())
    
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(all_word_vecs)
    
    for word in query:
        if word in word_embeddings:
            distances, indices = knn.kneighbors([word_embeddings[word]])
            for idx in indices[0]:
                neighbor_word = all_words[idx]
                expanded_query.add(neighbor_word)
    
    return list(expanded_query)


def generate_query_embeddings(
    text_embeddings: Dict[str, np.ndarray],
    word_embeddings: Dict[str, np.ndarray],
    text_graph: nx.Graph,
    query_words: List[List[str]],
    embedding_size: int,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    simulate_biased_random_walks_for_query: callable,
    filtered_word_docs: Dict[str, Dict[int, int]],
    filtered_word_importance: Dict[str, float],
    filtered_doc_words: Dict[int, Dict[str, int]],
    doc_ids: set
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for queries based on the provided text embeddings and graph.

    Args:
        text_embeddings (Dict[str, np.ndarray]): Text node embeddings.
        text_graph (nx.Graph): Graph with text nodes and edges.
        query_words (List[List[str]]): Tokenized queries.
        embedding_size (int): The size of the embeddings.
        num_walks (int): Number of random walks to perform.
        walk_length (int): Length of each random walk.
        p (float): Return parameter for Node2Vec.
        q (float): In-out parameter for Node2Vec.
        simulate_biased_random_walks_for_query (callable): Function to simulate biased random walks.
        filtered_word_docs (Dict[str, Dict[int, int]]): Filtered word-doc term frequency dictionary.
        filtered_word_importance (Dict[str, float]): Filtered word importance dictionary.
        filtered_doc_words (Dict[int, Dict[str, int]]): Filtered doc-word term frequency dictionary.
        doc_ids (set): Set of document IDs.

    Returns:
        Dict[str, np.ndarray]: A dictionary with query nodes as keys and their embeddings as values.
    """
    query_embeddings = {}
    
    for i, query in enumerate(query_words):
        #query = expand_query(query, word_embeddings, k=5)
        query_node = f'query_{i}'
        text_graph.add_node(query_node, type='query', embedding=np.zeros(embedding_size))
        
        # Add edges based on common words with text nodes if weight is greater than average
        for doc_id in doc_ids:
            text_node = f'text_{doc_id}'
            if text_node not in text_graph:
                continue
            common_words = set(query).intersection(set(filtered_doc_words[doc_id].keys()))
            if not common_words:
                continue
            weight = sum(filtered_word_importance[word] * filtered_word_docs[word][doc_id] for word in common_words if word in filtered_word_importance)
            text_graph.add_edge(query_node, text_node, weight=weight)
        
        # Simulate biased random walks for the query node
        updated_embedding = simulate_biased_random_walks_for_query(
            text_graph, query_node, text_embeddings, embedding_size, walk_length, num_walks, p, q
        )
        query_embeddings[query_node] = updated_embedding
        
        # Remove the query node from the graph
        text_graph.remove_node(query_node)
    
    return query_embeddings




# Initialize dictionaries and load data
data_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\complete_index.dat'
# Load inverted index data
word_docs, word_importance, doc_words, doc_ids = load_inverted_index(data_file_path)
# Filter low importance words
filtered_word_docs, filtered_word_importance, filtered_doc_words = filter_low_importance_words(word_docs, word_importance, doc_words, threshold_divisor=46)
#create text subgraph
text_subgraph = create_graph(doc_ids, filtered_word_docs, filtered_word_importance, filtered_doc_words)
# Parameters for node2vec
embedding_size = 64
walk_length = 30
num_walks = 300
window_size = 5
p = 0.5
q = 2
edg_file_path = 'text_subgraph.edg'
# Generate embeddings
text_embeddings = generate_text_embeddings(text_subgraph, embedding_size, walk_length, num_walks, window_size, p, q, edg_file_path)

# Save embeddings to CSV
text_embeddings_df = pd.DataFrame(text_embeddings).T
text_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\text_embeddings.csv', header=False)

print("Saved text embeddings to CSV.")

queries_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\Queries.txt'
query_words = load_and_tokenize_queries(queries_file_path)

# Load precomputed word embeddings
word_embeddings_df = pd.read_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\word_embeddings.csv', header=None, index_col=0)
word_embeddings = {str(idx): np.array(embedding, dtype=np.float64) for idx, embedding in word_embeddings_df.iterrows()}

# Example usage
query_embeddings = generate_query_embeddings(
    text_embeddings=text_embeddings,
    word_embeddings=word_embeddings,
    text_graph=text_subgraph,
    query_words=query_words,
    embedding_size=64,
    num_walks=200,
    walk_length=2,
    p=0.5,
    q=2,
    simulate_biased_random_walks_for_query=simulate_biased_random_walks_for_query,
    filtered_word_docs=filtered_word_docs,
    filtered_word_importance=filtered_word_importance,
    filtered_doc_words=filtered_doc_words,
    doc_ids=doc_ids
)

# Save query embeddings to CSV
query_embeddings_df = pd.DataFrame(query_embeddings).T
query_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\query_embeddings.csv', header=False)

print("Saved query embeddings to CSV.")
