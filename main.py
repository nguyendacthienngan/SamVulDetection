import json
from models.graph.devign import DevignModel
from models.combined import CombinedModel
from models.sequence.clr import CLRModel
from preprocess.megavul_dataset import MegaVulDataset
import torch
import dgl
import json
from pathlib import Path
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
import torch.nn as nn

from dataclasses import dataclass

from utils.evaluate import evaluate_model

from sklearn.model_selection import train_test_split

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

# def load_megavul_data(json_file: str):
#     with open(json_file, 'r') as file:
#         data = json.load(file)
    
#     functions = []
#     for item in data:
#         functions.append(MegaVulFunction(**item))
#     return functions

# def extract_data_for_training(functions: List[MegaVulFunction]):
#     sequence_data = []
#     graph_data = []
#     labels = []

#     for func in functions:
#         # Extract sequence data (e.g., abstract_func)
#         sequence_data.append(func.abstract_func)

#         # Extract graph data (e.g., path to func_graph)
#         graph_data.append(func.func_graph_path)

#         # Extract labels (is_vul)
#         labels.append(int(func.is_vul))
    
#     return sequence_data, graph_data, labels

import json
from sklearn.model_selection import train_test_split

def load_and_split_data(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract data and labels
    data_items = [item for item in data]
    labels = [item['is_vul'] for item in data]

    # Split data into train, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data_items, labels, test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


# Split the data into train, validation, and test sets
# def split_data(sequence_data, graph_data, labels, train_size=0.7, val_size=0.15):
    # First split to separate train data
    seq_train, seq_temp, graph_train, graph_temp, label_train, label_temp = train_test_split(
        sequence_data, graph_data, labels, train_size=train_size, random_state=42, stratify=labels)

    # Then split the remaining data into validation and test sets
    val_ratio = val_size / (1 - train_size)
    seq_val, seq_test, graph_val, graph_test, label_val, label_test = train_test_split(
        seq_temp, graph_temp, label_temp, train_size=val_ratio, random_state=42, stratify=label_temp)

    return (seq_train, graph_train, label_train), (seq_val, graph_val, label_val), (seq_test, graph_test, label_test)

def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    labels = [item['label'] for item in batch]
    # Process the graphs and labels as needed for your model
    # Example: padding graphs if necessary
    return {'graph': graphs, 'label': torch.stack(labels)}


# Main code
# Example usage
json_file = "/home/ngan/Documents/SamVulDetection/dataset/megavul-001.json"
# data = load_megavul_data(json_file)
# sequence_data, graph_data, labels = extract_data_for_training(data)


# (train_sequence_data, train_graph_data, train_labels), (val_sequence_data, val_graph_data, val_labels), (test_sequence_data, test_graph_data, test_labels) = split_data(sequence_data, graph_data, labels)
# Load and split data
train_data, val_data, test_data, train_labels, val_labels, test_labels = load_and_split_data(json_file)

# Sau khi tách
print("extract_data_for_training sucessfully")

# Create datasets
train_dataset = MegaVulDataset(train_data, train_labels)
val_dataset = MegaVulDataset(val_data, val_labels)
test_dataset = MegaVulDataset(test_data, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("Create DataLoaders")


# Assuming you have already split your data as shown in the previous steps

# Create dataset instances
# train_dataset = MegaVulDataset(train_sequence_data, train_graph_data, train_labels)
# val_dataset = MegaVulDataset(val_sequence_data, val_graph_data, val_labels)
# test_dataset = MegaVulDataset(test_sequence_data, test_graph_data, test_labels)

# # Create DataLoader instances
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize the model
clr_model = CLRModel(model_name='roberta-base', num_labels=2)
devign_model = DevignModel(input_dim=100, output_dim=128)  # Adjust dimensions based on your data
combined_model = CombinedModel(clr_model, devign_model)

print('Init model sucessfully')
num_epochs = 10
# Assuming you have a dataloader for the validation or test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
combined_model.to(device)

# Train model
print('Train model')

optimizer = torch.optim.Adam(combined_model.parameters(), lr=1e-4)

# Binary Cross-Entropy Loss for binary classification
loss_function = nn.BCELoss()
for epoch in range(num_epochs):
    combined_model.train()
    for batch in train_loader:
        graph_data = batch['graph']
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = combined_model(graph_data)
        loss = loss_function(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    val_metrics = evaluate_model(combined_model, val_loader, device)
    print(f'Epoch {epoch + 1}, Validation Metrics: {val_metrics}')


# Evaluate the model
print('Evaluate the model')

val_metrics = evaluate_model(combined_model, val_loader, device)

# Print the results
for metric_name, value in val_metrics.items():
    if metric_name == 'Confusion Matrix':
        print(f"{metric_name}:\n{value}")
    else:
        print(f"{metric_name}: {value:.4f}")
