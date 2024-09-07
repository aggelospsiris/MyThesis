import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.data import Data
import torch.nn.functional as F
import random
import pandas as pd

# Ensure you have downloaded the stopwords corpus
import nltk
nltk.download('stopwords')
nltk.download('punkt')

data_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\invertedindex.dat'
queries_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\Queries.txt'

# Initialize dictionaries
word_docs = {}  # {word: {doc_id: tf}}
word_df = {}  # {word: df}
doc_words = {}  # {doc_id: {word: tf}}
doc_ids = set()
stop_words = set(stopwords.words('english'))

# Parse the file to fill word_docs and doc_words
with open(data_file_path, 'r') as file:
    for line in file:
        match = re.match(r'^(\d+);([^;]+);([^;]+);(.+)$', line.strip())
        if match:
            word_id = match.group(1)
            label = match.group(2).lower()
            df = float(match.group(3))
            word_df[label] = df
            if label in stop_words:
                continue
            pairs = re.findall(r'txtfiles\\(\d+),(\d+)', match.group(4))
            word_docs[label] = {int(doc_id): int(tf) for doc_id, tf in pairs}
            for doc_id, tf in pairs:
                doc_id = int(doc_id)
                tf = int(tf)
                doc_ids.add(doc_id)
                if doc_id not in doc_words:
                    doc_words[doc_id] = {}
                doc_words[doc_id][label] = tf

# Create the graph
G = nx.DiGraph()  # Use directed graph

# Add text nodes
for doc_id in doc_ids:
    G.add_node(f'text_{doc_id}', type='text')

# Add word nodes and edges with weights between words and texts
for word, docs in word_docs.items():
    G.add_node(word, type='word')
    for doc_id, tf in docs.items():
        df = word_df[word]
        weight = df * tf
        G.add_edge(word, f'text_{doc_id}', weight=weight)

# Add edges between text nodes based on shared words
for doc1 in doc_ids:
    for doc2 in doc_ids:
        if doc1 >= doc2:
            continue
        common_words = set(doc_words[doc1].keys()).intersection(set(doc_words[doc2].keys()))
        if not common_words:
            continue
        weight = sum((((word_docs[word][doc1] * word_df[word]) + (word_docs[word][doc2] * word_df[word])) / 2) for word in common_words)
        if weight > 0:
            G.add_edge(f'text_{doc1}', f'text_{doc2}', weight=weight)
            G.add_edge(f'text_{doc2}', f'text_{doc1}', weight=weight)

# Read and process the queries
queries = []
with open(queries_file_path, 'r') as file:
    for line in file:
        query = line.strip().lower()
        queries.append(query)

# Tokenize and remove stop words from queries
query_words = [word_tokenize(query) for query in queries]
query_words = [[word for word in query if word not in stop_words] for query in query_words]

# Define a custom GAT layer that uses edge weights for attention
class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super(CustomGATConv, self).__init__(aggr='add')
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, heads * out_channels)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_weight):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        alpha = (x_j * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = alpha * edge_weight.view(-1, 1)
        alpha = F.softmax(alpha, dim=1)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=1)

# Define GAT model with custom GAT layers emphasizing edge weights
class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATNet, self).__init__()
        self.conv1 = CustomGATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = CustomGATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = CustomGATConv(hidden_channels * heads, out_channels, heads=heads)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x

# Initialize model with increased capacity
embedding_size = 128
hidden_channels = 64
out_channels = 128
heads = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNet(embedding_size, hidden_channels, out_channels, heads).to(device)

# Convert the networkx graph to PyTorch Geometric data
def convert_to_pyg_data(G, node_features):
    node_mapping = {node: i for i, node in enumerate(G.nodes)}
    reverse_node_mapping = {i: node for i, node in enumerate(G.nodes)}
    
    edges = []
    edge_weights = []
    
    for u, v, data in G.edges(data=True):
        edges.append((node_mapping[u], node_mapping[v]))
        edge_weights.append(data['weight'])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Ensure all indices in edge_index are within the valid range
    max_index = edge_index.max().item()
    if max_index >= len(node_features):
        raise ValueError(f"Edge index contains invalid indices. Max index is {max_index}, but node features length is {len(node_features)}.")
    
    # Convert list of numpy arrays to a single numpy array and then to a tensor
    x = torch.tensor(np.array([node_features[node] for node in G.nodes]), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight), node_mapping, reverse_node_mapping

# Prepare data for text nodes only
text_nodes = [n for n in G.nodes if G.nodes[n]['type'] == 'text']
text_subgraph = G.subgraph(text_nodes).copy()

# Random initial features for text nodes
text_node_features = {node: np.random.randn(embedding_size) for node in text_nodes}

data, node_mapping, reverse_node_mapping = convert_to_pyg_data(text_subgraph, text_node_features)
data = data.to(device)

# Forward pass to get text node embeddings
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index, data.edge_weight)

# Save text node embeddings into the graph as their features
for i, node in reverse_node_mapping.items():
    G.nodes[node]['embedding'] = out[i].detach().cpu().numpy()

# Save text node embeddings into CSV file in the specified format
text_embeddings = {node: G.nodes[node]['embedding'] for node in text_nodes}
text_embeddings_df = pd.DataFrame.from_dict(text_embeddings, orient='index')
text_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\text_embeddings3.csv', header=False)

# Add query nodes, initialize, run GAT, and calculate embeddings for each query
query_embeddings = {}
for i, query in enumerate(query_words):
    query_node = f'query_{i}'
    G.add_node(query_node, type='query', embedding=np.random.randn(embedding_size))
    for doc_id in doc_ids:
        common_words = set(query).intersection(set(doc_words[doc_id].keys()))
        if not common_words:
            continue
        weight = sum(doc_words[doc_id][word] for word in common_words)
        if weight > 0:
            G.add_edge(query_node, f'text_{doc_id}', weight=weight)
    
    # Convert the modified graph to PyTorch Geometric data
    query_subgraph = G.subgraph(text_nodes + [query_node]).copy()
    
    # Use the precomputed embeddings for text nodes and initialize random for the query node
    query_node_features = {node: G.nodes[node]['embedding'] for node in text_nodes}
    query_node_features[query_node] = np.random.randn(embedding_size)
    
    data, node_mapping, reverse_node_mapping = convert_to_pyg_data(query_subgraph, query_node_features)
    data = data.to(device)
    
    # Ensure all indices in edge_index are within the valid range
    max_index = data.edge_index.max().item()
    if max_index >= data.x.size(0):
        raise ValueError(f"Edge index contains invalid indices. Max index is {max_index}, but node features length is {data.x.size(0)}.")
    
    # Run GAT for the query node
    query_index = node_mapping[query_node]
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
    
    # Extract the embedding for the query node
    query_embeddings[query_node] = out[query_index].detach().cpu().numpy()
    
    # Remove the query node from the graph
    G.remove_node(query_node)

# Save query node embeddings into CSV file in the specified format
query_embeddings_df = pd.DataFrame.from_dict(query_embeddings, orient='index')
query_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\query_embeddings3.csv', header=False)
