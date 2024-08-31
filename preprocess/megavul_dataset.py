import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import json
from dataclasses import dataclass

from pathlib import Path
from typing import List, Optional, Dict

graph_dir='/home/ngan/Documents/SamVulDetection/dataset/megavul/megavul_graph/' 

@dataclass
class MegaVulFunction:
    cve_id: str
    cwe_ids: List[str]
    cvss_vector: Optional[str]
    cvss_base_score: Optional[float]
    cvss_base_severity: Optional[str]
    cvss_is_v3: Optional[bool]
    publish_date: str

    repo_name: str
    commit_msg: str
    commit_hash: str
    parent_commit_hash: str
    commit_date: int
    git_url: str

    file_path: str
    func_name: str
    parameter_list_signature_before: Optional[str]
    parameter_list_before: Optional[List[str]]
    return_type_before: Optional[str]
    func_before: Optional[str]
    abstract_func_before: Optional[str]
    abstract_symbol_table_before: Optional[Dict[str, str]]
    func_graph_path_before: Optional[str]

    parameter_list_signature: str
    parameter_list: List[str]
    return_type: str
    func: str
    abstract_func: str
    abstract_symbol_table: Dict[str, str]
    func_graph_path: Optional[str]

    diff_func: Optional[str]
    diff_line_info: Optional[Dict[str, List[int]]]  # [deleted_lines, added_lines]

    is_vul: bool
class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self,
                 sequence_tokens=None,
                 sequence_ids=None,
                 graph_features=None,
                 idx=None,
                 label=None,
                 attention_mask=None):
        self.sequence_tokens = sequence_tokens  # Tokens for sequence data
        self.sequence_ids = sequence_ids        # Token IDs for sequence data
        self.graph_features = graph_features    # Features for graph data
        self.idx = str(idx)                     # Unique identifier
        self.label = label                      # Label for classification
        self.attention_mask =attention_mask


def create_graph_from_json(data_item):
    # print('data_item')
    # print(data_item)

    """
    Create a DGL graph from a JSON item based on MegaVulFunction dataclass.

    Args:
        data_item (dict): A dictionary containing graph data in JSON format.

    Returns:
        dgl.DGLGraph: The DGL graph created from the JSON data.
    """
    # Extract graph-related information from the JSON data
    func_graph_path = data_item.get('func_graph_path_before', None)
    # print(func_graph_path)

    if func_graph_path:
        func_graph_path = graph_dir + func_graph_path
        # print(func_graph_path)

        # Here you would typically load the graph data from a file
        # For this example, we assume `func_graph_path` contains graph data in JSON format
        # If `func_graph_path` is a path to a file, load it accordingly
        # For example, if the graph data is directly embedded in JSON:
        
        with open(func_graph_path, 'r') as f:
            graph_data = json.load(f)
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        # print(f'edges: {edges}')
        # Tạo danh sách các node id và edge connections
        edge_tuples = [(edge['inNode'], edge['outNode']) for edge in edges]

        # Tạo đồ thị với DGL
        g = dgl.graph(edge_tuples)

        # Debug: Check number of nodes in the graph
        print(f"Number of nodes in the graph---(edge_tuples): {g.num_nodes()}")
        print(f"Number of nodes len(nodes): {len(nodes)}")
        
        # Thêm đặc trưng cho các nodes
        # Ví dụ: chỉ sử dụng 'typeFullName' và 'code' như là đặc trưng của các nodes
        node_features = []
        for node in nodes:
            features = {
                # 'type': node.get('typeFullName', 'unknown'),
                'type': node.get('_label', 'unknown'),  # Adapt _label to type feature
                'code': node.get('code', ''),
                'name': node.get('name', ''),
                'language': node.get('language', '')
                # Thêm các đặc trưng khác nếu cần
            }
            node_features.append(features)
        
        # Debug: Check number of nodes in the graph
        print(f"Number of nodes in the graph: after add nodes {g.num_nodes()}")
        
        print(f"g.num_nodes() {g.num_nodes()}")
        print(f"len(node_features) {len(node_features)}")

        # Ensure node features match number of nodes
        if len(node_features) < g.num_nodes():
            missing_count = g.num_nodes() - len(node_features)
            placeholder_features = {
                'type': 'unknown',
                'code': '',
                'name': '',
                'language': ''
            }
            for _ in range(missing_count):
                node_features.append(placeholder_features)


        if len(node_features) == g.num_nodes():
            print(f'len(node_features) == g.num_nodes()')
            type_feature = torch.tensor([hash(f['type']) % 1000 for f in node_features], dtype=torch.float32)
            # Thêm một chiều mới để tạo tensor 2D với kích thước [num_nodes, 1]
            type_feature_2d = type_feature.unsqueeze(1)  # hoặc type_feature.view(-1, 1)
            g.ndata['type'] = type_feature_2d

            # code_feature = torch.tensor([hash(f['code']) % 1000 for f in node_features], dtype=torch.float32)
            # name_feature = torch.tensor([hash(f['name']) % 1000 for f in node_features], dtype=torch.float32)
            # language_feature = torch.tensor([hash(f['language']) % 1000 for f in node_features], dtype=torch.float32)

            # g.ndata['code'] = code_feature
            # g.ndata['name'] = name_feature
            # g.ndata['language'] = language_feature
        else:
            raise ValueError("Mismatch between the number of nodes and node features")

        # Map edge types
        edge_labels = [edge.get('etype', 'unknown') for edge in edges]
        # print(f"edge_labels {edge_labels}")
        edge_type_mapping = {
            'AST': 0,
            'CDG': 1,
            'CFG': 2,
            'REACHING_DEF': 3,
            'CALL': 4,
            'OTHER': 5
        }
        edge_types = [edge_type_mapping.get(label, edge_type_mapping['OTHER']) for label in edge_labels] #Embedding Layer
        # Ensure edge types are within the valid range
        g.edata['label'] = torch.tensor(edge_types, dtype=torch.long)

        return g
    else:
        # Handle the case where func_graph_path is None
        return None
# def pad_graph_features(graph_features, max_size):
#     padded_graphs = []
#     for g in graph_features:
#         size = g.size(0)  # Assuming square matrices for simplicity
#         if size < max_size:
#             # Pad the graph with zeros
#             padded_graph = torch.zeros((max_size, max_size))
#             padded_graph[:size, :size] = g
#             padded_graphs.append(padded_graph)
#         else:
#             # Truncate the graph
#             padded_graph = g[:max_size, :max_size]
#             padded_graphs.append(padded_graph)
#     return torch.stack(padded_graphs)

def pad_graph_features(graph, max_size):
    """
    Pad the adjacency matrix of a DGL graph to a specified size.

    Args:
        graph (dgl.DGLGraph): A single DGL graph object.
        max_size (int): The maximum size for padding/truncating.

    Returns:
        torch.Tensor: Padded adjacency matrix of the graph.
    """
    # Get the adjacency matrix of the graph in scipy's coo format
    adj = graph.adjacency_matrix(scipy_fmt="coo")
    
    # Convert the adjacency matrix to a dense PyTorch tensor
    adj_dense = torch.tensor(adj.toarray(), dtype=torch.float32)

    # Get the current size of the adjacency matrix
    size = adj_dense.shape[0]

    if size < max_size:
        # Pad the adjacency matrix with zeros
        padded_adj = torch.zeros((max_size, max_size), dtype=torch.float32)
        padded_adj[:size, :size] = adj_dense
    else:
        # Truncate the adjacency matrix
        padded_adj = adj_dense[:max_size, :max_size]

    return padded_adj

def convert_examples_to_features(js, tokenizer, args, idx, label):
    # Function to process sequence data and return tokens, ids
    code = js['func']

    code_tokens = tokenizer.tokenize(code)
    source_tokens = code_tokens[:args['block_size']]
    # if args.model_type in ["codegen"]:
        # code_tokens = tokenizer.tokenize(code)
        # source_tokens = code_tokens[:args.block_size]
    # elif args.model_type in ["starcoder"]:
    #     code_tokens = tokenizer.tokenize(code)
    #     source_tokens = code_tokens[:args.block_size]
    # else:
    #     code_tokens = tokenizer.tokenize(code)
    #     source_tokens = [tokenizer.cls_token] + code_tokens[:args.block_size-2] + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # Create attention mask: 1 for real tokens and 0 for padding tokens
    attention_mask = [1] * len(input_ids)
    padding_length = args['block_size'] - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length  # Pad the attention mask with 0s

    return InputFeatures(sequence_tokens=source_tokens,
                         sequence_ids=input_ids,
                         attention_mask=attention_mask,  # Add the attention mask here
                         idx=idx,
                         label=label)


class MegaVulDataset(Dataset):
    def __init__(self, data, labels, tokenizer, args):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.args = args
        self.max_graph_size = args['max_graph_size']  # Set this to a fixed size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels[idx]
        # Convert boolean label to integer
        label = int(label)  # Convert True/False to 1/0

        # Sequence input processing
        sequence_features = convert_examples_to_features(item, self.tokenizer, self.args, idx, label)
        # Graph input processing
        graph_features = create_graph_from_json(item)
        if graph_features is not None:
            padded_graph_features = pad_graph_features(graph_features, self.max_graph_size)
        else:
            # Create a placeholder tensor filled with zeros if graph_features is None
            padded_graph_features = torch.zeros(self.max_graph_size, self.max_graph_size)

        return {
            'sequence_ids': sequence_features.sequence_ids,
            'attention_mask': sequence_features.attention_mask,
            'graph_features': graph_features,
            'label': label
        }