import json
from models.graph.devign import DevignModel
from models.combined import CombinedModel
from models.sequence.clr import CLRModel
from preprocess.megavul_dataset import MegaVulDataset
import torch
import json
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
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


def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    labels = [item['label'] for item in batch]
    # Process the graphs and labels as needed for your model
    # Example: padding graphs if necessary
    return {'graph': graphs, 'label': torch.stack(labels)}
def train(args, device, train_loader, val_loader, model, optimizer, loss_function):
    model.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        
        for batch in train_loader:
            sequence_inputs, attention_mask, graph_inputs, labels = batch
            
            # Chuyển dữ liệu sang device
            sequence_inputs = sequence_inputs.to(device)
            attention_mask = attention_mask.to(device)
            graph_inputs = graph_inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(sequence_inputs, attention_mask, graph_inputs)
            
            # Tính loss
            loss = loss_function(logits.squeeze(), labels.float())
            
            # Backward pass và cập nhật
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}")
        
        # Đánh giá mô hình trên validation set
        evaluate(args, device, val_loader, model)
        
    print("Training complete.")

def evaluate(args, device, val_loader, model):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            sequence_inputs, attention_mask, graph_inputs, labels = batch
            
            # Chuyển dữ liệu sang device
            sequence_inputs = sequence_inputs.to(device)
            attention_mask = attention_mask.to(device)
            graph_inputs = graph_inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(sequence_inputs, attention_mask, graph_inputs)
            preds = torch.sigmoid(logits).squeeze()  # Sử dụng sigmoid cho binary classification
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Tính các số liệu đánh giá
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")


# Main code
# Example usage
json_file = "/home/ngan/Documents/SamVulDetection/dataset/megavul-001.json"
# Load and split data
train_data, val_data, test_data, train_labels, val_labels, test_labels = load_and_split_data(json_file)

# Sau khi tách
print("extract_data_for_training sucessfully")

# Update to include tokenizer and args
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
args = {
    'model_type': 'codegen',  # Or your specific model type
    'block_size': 512  # Example block size
}

train_dataset = MegaVulDataset(train_data, train_labels, tokenizer, args)
val_dataset = MegaVulDataset(val_data, val_labels, tokenizer, args)
test_dataset = MegaVulDataset(test_data, test_labels, tokenizer, args)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("Create DataLoaders")


# Assuming you have already split your data as shown in the previous steps

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
# for epoch in range(num_epochs):
#     combined_model.train()
#     for batch in train_loader:
#         graph_data = batch['graph']
#         labels = batch['label'].to(device)

#         # Forward pass
#         optimizer.zero_grad()
#         outputs = combined_model(graph_data)
#         loss = loss_function(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#     # Evaluate on validation set
#     val_metrics = evaluate_model(combined_model, val_loader, device)
#     print(f'Epoch {epoch + 1}, Validation Metrics: {val_metrics}')


# # Evaluate the model
# print('Evaluate the model')

# val_metrics = evaluate_model(combined_model, val_loader, device)

# # Print the results
# for metric_name, value in val_metrics.items():
#     if metric_name == 'Confusion Matrix':
#         print(f"{metric_name}:\n{value}")
#     else:
#         print(f"{metric_name}: {value:.4f}")

# Define arguments

# Set training arguments
args = {
    'learning_rate': 1e-4,
    'epoch': 10,
    'warmup_ratio': 0.1,
    'gradient_accumulation_steps': 1,
    'logging_steps': 100,
    'max_grad_norm': 1.0,
    'max_patience': 5,
    'do_test': True,
    'project': 'your_project',
    'model_dir': 'your_model_dir',
    'output_dir': './output',
    'adam_epsilon': 1e-8,
    'weight_decay': 0.01,
    'start_epoch': 0,
    'per_gpu_eval_batch_size': 32,
    'n_gpu': 1
}

# Initialize DataLoader, model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
combined_model.to(device)

optimizer = torch.optim.Adam(combined_model.parameters(), lr=args.learning_rate)
loss_function = nn.BCEWithLogitsLoss()

# Call train function
train(args, device, train_loader, val_loader, combined_model, optimizer, loss_function)

