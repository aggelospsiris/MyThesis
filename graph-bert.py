import networkx as nx
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import nltk
from nltk.tokenize import word_tokenize

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt')

# Function to initialize the process group for distributed training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Function to cleanup the process group after training
def cleanup():
    dist.destroy_process_group()

# Define the dataset
class GraphDataset(Dataset):
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes)
        
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        node = self.nodes[idx]
        neighbors = list(self.graph.neighbors(node))
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=[node] + neighbors).todense()
        edge_weights = np.array([self.graph[node][neighbor]['weight'] for neighbor in neighbors])
        edge_weights = np.insert(edge_weights, 0, 1.0)  # Add self-loop with weight 1.0
        return torch.tensor([node] + neighbors, dtype=torch.long), torch.tensor(adj_matrix, dtype=torch.float), torch.tensor(edge_weights, dtype=torch.float)

# Define the model
class WeightedGraphBERT(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super(WeightedGraphBERT, self).__init__()
        self.bert = BertModel(BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=hidden_dim * 4
        ))
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        
    def forward(self, node_ids, adj_matrix, edge_weights):
        node_features = self.node_embeddings(node_ids)
        attention_mask = self.create_attention_mask(adj_matrix, edge_weights)
        outputs = self.bert(inputs_embeds=node_features, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def create_attention_mask(self, adj_matrix, edge_weights):
        # Modify attention mask to include edge weights
        mask = adj_matrix * edge_weights
        return mask

# Training function for each process
def train(rank, world_size, graph, num_nodes, hidden_dim, epochs, batch_size=16):
    setup(rank, world_size)
    
    model = WeightedGraphBERT(num_nodes=num_nodes, hidden_dim=hidden_dim).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    dataset = GraphDataset(graph)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        ddp_model.train()
        epoch_loss = 0
        for node_ids, adj_matrix, edge_weights in dataloader:
            node_ids, adj_matrix, edge_weights = node_ids.to(rank), adj_matrix.to(rank), edge_weights.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(node_ids, adj_matrix, edge_weights)
            loss = criterion(outputs, adj_matrix)  # Example loss function
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch + 1}, Loss: {epoch_loss}")
    
    cleanup()

# Main function to initiate training with multiple GPUs
def main():
    # Load inverted index data and preprocess
    word_docs = {}  # {word: {doc_id: tf}}
    word_df = {}  # {word: df}
    doc_words = {}  # {doc_id: {word: tf}}
    doc_ids = set()

    data_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\invertedindex.dat'
    queries_file_path = 'C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\Queries.txt'

    with open(data_file_path, 'r') as file:
        for line in file:
            match = re.match(r'^(\d+);([^;]+);([^;]+);(.+)$', line.strip())
            if match:
                word_id = match.group(1)
                label = match.group(2).lower()
                df = float(match.group(3))
                word_df[label] = df
                pairs = re.findall(r'txtfiles\\(\d+),(\d+)', match.group(4))
                word_docs[label] = {int(doc_id): int(tf) for doc_id, tf in pairs}
                for doc_id, tf in pairs:
                    doc_id = int(doc_id)
                    tf = int(tf)
                    doc_ids.add(doc_id)
                    if doc_id not in doc_words:
                        doc_words[doc_id] = {}
                    doc_words[doc_id][label] = tf

    average_importance = np.mean(list(word_df.values()))
    threshold = average_importance / 46

    filtered_word_docs = {word: docs for word, docs in word_docs.items() if word_df[word] >= threshold}
    filtered_word_df = {word: df for word, df in word_df.items() if df >= threshold}

    filtered_doc_words = {}
    for doc_id, words in doc_words.items():
        filtered_doc_words[doc_id] = {word: tf for word, tf in words.items() if word in filtered_word_df}

    G = nx.Graph()  # Use undirected graph

    for doc_id in doc_ids:
        G.add_node(f'text_{doc_id}', type='text')

    for word, docs in filtered_word_docs.items():
        G.add_node(word, type='word')
        for doc_id, tf in docs.items():
            df = filtered_word_df[word]
            weight = df * tf
            G.add_edge(word, f'text_{doc_id}', weight=weight)

    for doc1 in doc_ids:
        for doc2 in doc_ids:
            if doc1 >= doc2:
                continue
            common_words = set(filtered_doc_words[doc1].keys()).intersection(set(filtered_doc_words[doc2].keys()))
            if not common_words:
                continue
            weight = sum((max(filtered_word_docs[word][doc2], filtered_word_docs[word][doc1]) * filtered_word_df[word]) for word in common_words if word in filtered_word_docs)
            if weight > 0:
                G.add_edge(f'text_{doc1}', f'text_{doc2}', weight=weight)

    text_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'text']
    text_subgraph = G.subgraph(text_nodes).copy()

    # Verify the text subgraph structure
    print("Text subgraph construction completed.")
    print(f"Number of nodes in text subgraph: {text_subgraph.number_of_nodes()}")
    print(f"Number of edges in text subgraph: {text_subgraph.number_of_edges()}")

    num_nodes = G.number_of_nodes()
    hidden_dim = 256
    epochs = 1000
    batch_size = 16
    world_size = torch.cuda.device_count()

    mp.spawn(train,
             args=(world_size, G, num_nodes, hidden_dim, epochs, batch_size),
             nprocs=world_size,
             join=True)

    # Extract and save text node embeddings
    model = WeightedGraphBERT(num_nodes=num_nodes, hidden_dim=hidden_dim)
    node_embeddings = model.node_embeddings.weight.detach().cpu().numpy()
    
    # Extract only text node embeddings
    text_node_embeddings = {node: node_embeddings[idx] for idx, node in enumerate(G.nodes) if G.nodes[node]['type'] == 'text'}
    text_node_ids = list(text_node_embeddings.keys())
    text_node_embeddings = np.array(list(text_node_embeddings.values()))

    text_embeddings_df = pd.DataFrame(text_node_embeddings, index=text_node_ids)
    text_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\text_embeddings.csv', header=False)
    print("Saved text embeddings to CSV.")

    # Load and process the queries
    queries = []
    with open(queries_file_path, 'r') as file:
        for line in file:
            query = line.strip().lower()
            queries.append(query)

    query_words = [word_tokenize(query) for query in queries]

    # Load precomputed text embeddings
    text_embeddings_df = pd.read_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\text_embeddings.csv', header=None, index_col=0)
    text_embeddings = {str(idx): np.array(embedding, dtype=np.float64) for idx, embedding in text_embeddings_df.iterrows()}

    # Function to use pre-trained Graph-BERT for query embeddings
    def generate_query_embeddings(model, query_words, text_embeddings, embedding_size):
        query_embeddings = {}
        for i, query in enumerate(query_words):
            query_node = f'query_{i}'
            text_subgraph.add_node(query_node, type='query', embedding=np.zeros(embedding_size))
            
            # Add edges based on common words with text nodes if weight is greater than average
            for doc_id in doc_ids:
                text_node = f'text_{doc_id}'
                if text_node not in text_subgraph:
                    continue
                common_words = set(query).intersection(set(filtered_doc_words[doc_id].keys()))
                if not common_words:
                    continue
                weight = sum(filtered_word_df[word] * filtered_word_docs[word][doc_id] for word in common_words if word in filtered_word_df)
                text_subgraph.add_edge(query_node, text_node, weight=weight)
            
            # Use pre-trained Graph-BERT model to get query embedding
            node_ids = [query_node] + [f'text_{doc_id}' for doc_id in doc_ids if text_subgraph.has_edge(query_node, f'text_{doc_id}')]
            node_ids_tensor = torch.tensor([text_node_ids.index(node_id) if node_id in text_node_ids else len(text_node_ids) for node_id in node_ids], dtype=torch.long)
            adj_matrix = nx.adjacency_matrix(text_subgraph, nodelist=node_ids).todense()
            edge_weights = np.array([text_subgraph[query_node][text_node]['weight'] for text_node in node_ids[1:]])
            edge_weights = np.insert(edge_weights, 0, 1.0)  # Add self-loop with weight 1.0
            
            node_ids_tensor = node_ids_tensor.unsqueeze(0)  # Add batch dimension
            adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).unsqueeze(0)  # Add batch dimension
            edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                outputs = model(node_ids_tensor, adj_matrix_tensor, edge_weights_tensor)
            query_embedding = outputs.squeeze().cpu().numpy().mean(axis=0)
            
            query_embeddings[query_node] = query_embedding
            text_subgraph.remove_node(query_node)
        
        return query_embeddings

    # Generate query embeddings using pre-trained Graph-BERT model
    query_embeddings = generate_query_embeddings(model, query_words, text_embeddings, hidden_dim)

    # Save query node embeddings into CSV file
    query_embeddings_df = pd.DataFrame(query_embeddings).T
    query_embeddings_df.to_csv('C:\\Users\\dionusia\\Downloads\\pythonProject\\pythonProject\\query_embeddings.csv', header=False)
    print("Saved query embeddings to CSV.")

if __name__ == "__main__":
    main()