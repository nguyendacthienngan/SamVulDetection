import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import json

graph_dir='/home/ngan/Documents/code/dataset/megavul/megavul_graph/' 

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


class MegaVulDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.graphs = [create_graph_from_json(item) for item in self.data]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        return {'graph': graph, 'label': torch.tensor(label, dtype=torch.float32)}
