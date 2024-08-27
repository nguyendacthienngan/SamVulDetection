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
                 label=None):
        self.sequence_tokens = sequence_tokens  # Tokens for sequence data
        self.sequence_ids = sequence_ids        # Token IDs for sequence data
        self.graph_features = graph_features    # Features for graph data
        self.idx = str(idx)                     # Unique identifier
        self.label = label                      # Label for classification


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
        
        # Process the loaded graph data to create DGL graph
        # Example structure of graph_data
        # {
        #     'nodes': {'node_ids': [0, 1, 2], 'node_features': [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]},
        #     'edges': {'src': [0, 1], 'dst': [1, 2]}
        # }
        
        # nodes = graph_data.get('nodes', {})
        # edges = graph_data.get('edges', {})
        nodes, edges = graph_data['nodes'] , graph_data['edges']
        #print(nodes)    # [{'version': '0.1', 'language': 'NEWC', '_label': 'META_DATA', 'overlays': ....
        #print(edges)    # [{'innode': 196, 'outnode': 2, 'etype': 'AST', 'variable': None}, ...]

        # Tạo danh sách các node id và edge connections
        node_ids = [node['id'] for node in nodes]
        edge_tuples = [(edge['inNode'], edge['outNode']) for edge in edges]

        # Tạo đồ thị với DGL
        g = dgl.graph(edge_tuples)

        # Thêm đặc trưng cho các nodes
        # Ví dụ: chỉ sử dụng 'typeFullName' và 'code' như là đặc trưng của các nodes
        node_features = []
        for node in nodes:
            features = {
                'type': node.get('typeFullName', 'unknown'),
                'code': node.get('code', ''),
                'name': node.get('name', ''),
                'language': node.get('language', '')
                # Thêm các đặc trưng khác nếu cần
            }
            node_features.append(features)

        if (len(node_features) == g.num_nodes()):
            'Match between nodes ({g.num_nodes()}) and features ({len(node_features)})'
            # Bạn có thể thêm các đặc trưng khác như tensors (giả sử dùng nhúng (embeddings))
            # Sử dụng các đặc trưng đơn giản như sau (ở đây chuyển đổi string thành embedding có thể là phức tạp hơn tùy theo yêu cầu của bạn):
            g.ndata['type'] = torch.tensor([hash(f['type']) % 1000 for f in node_features], dtype=torch.float32)
            g.ndata['code'] = torch.tensor([hash(f['code']) % 1000 for f in node_features], dtype=torch.float32)
            g.ndata['name'] = torch.tensor([hash(f['name']) % 1000 for f in node_features], dtype=torch.float32)
            g.ndata['language'] = torch.tensor([hash(f['language']) % 1000 for f in node_features], dtype=torch.float32)

        # Bạn cũng có thể thêm đặc trưng cho các edges nếu cần
        edge_labels = [edge.get('label', 'unknown') for edge in edges]
        g.edata['label'] = torch.tensor([hash(label) % 1000 for label in edge_labels], dtype=torch.float32)
        return g
    else:
        # Handle the case where func_graph_path is None
        return None
def convert_examples_to_features(js, tokenizer, args):
    # Function to process sequence data and return tokens, ids
    code = js['func']

    code_tokens = tokenizer.tokenize(code)
    source_tokens = code_tokens[:args.block_size]
    # if args.model_type in ["codegen"]:
        # code_tokens = tokenizer.tokenize(code)
        # source_tokens = code_tokens[:args.block_size]
    # elif args.model_type in ["starcoder"]:
    #     code_tokens = tokenizer.tokenize(code)
    #     source_tokens = code_tokens[:args.block_size]
    # else:
    #     code_tokens = tokenizer.tokenize(code)
    #     source_tokens = [tokenizer.cls_token] + code_tokens[:args.block_size-2] + [tokenizer.sep_token]

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(sequence_tokens=source_tokens,
                         sequence_ids=source_ids,
                         idx=js['idx'],
                         label=js['target'])


class MegaVulDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Sequence input processing
        sequence_features = convert_examples_to_features(item['sequence_data'], self.tokenizer, self.args)
        # Graph input processing
        graph_features = self.create_graph_from_json(item['graph_data'])
        
        return {
            'sequence_ids': sequence_features.input_ids,
            'attention_mask': sequence_features.attention_mask,
            'graph_features': graph_features,
            'label': self.labels[idx]
        }