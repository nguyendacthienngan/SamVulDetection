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
import matplotlib.pyplot as plt
import os
import numpy as np
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

import dgl

def collate_fn(batch):
    # Tách các phần tử của batch
    sequence_ids = [torch.tensor(item['sequence_ids']) for item in batch]
    attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
    graph_features = [item['graph_features'] for item in batch if item['graph_features']  is not None]
    labels = [item['label'] for item in batch]

    # Chuyển đổi các dữ liệu sequence thành tensor
    sequence_ids_tensor = torch.stack(sequence_ids)
    attention_masks_tensor = torch.stack(attention_masks)
    labels_tensor = torch.tensor(labels)
    # graph_features_tensor = torch.stack(graph_features)
    if len(graph_features) > 0:
        graph_features_tensor = dgl.batch(graph_features)  # Use DGL's batch function
    else:
        graph_features_tensor = None

    return {
        'sequence_ids': sequence_ids_tensor,
        'attention_mask': attention_masks_tensor,
        'graph_features': graph_features_tensor,
        'label': labels_tensor
    }
def train(args, device, train_loader, val_loader, model, optimizer, loss_function):
    model.train()
    train_losses = []
    best_val_loss = float('inf')
    save_dir = '/home/ngan/Documents/SamVulDetection/saved_models'
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    for epoch in range(args['epoch']):
        print(f'Epoch: {epoch}')
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            print(f'batch_idx: {batch_idx}')
            sequence_inputs = batch['sequence_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_inputs = batch['graph_features'].to(device) if batch['graph_features'] is not None else None
            labels = batch['label'].to(device).long()  # Convert labels to long
            
            # Forward pass
            if graph_inputs is not None:
                outputs = model(sequence_inputs, attention_mask, graph_inputs)
            else:
                outputs = model(sequence_inputs, attention_mask)
                
            # Print shapes and types for debugging
            print(f"Sequence logits shape: {outputs.shape}, dtype: {outputs.dtype}")
            print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")

            # Calculate loss
            loss = loss_function(outputs, labels)
            
            # Backward pass và cập nhật
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args['epoch']}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args['epoch']}, Average Loss: {avg_loss:.4f}")
        
        # Đánh giá mô hình trên validation set
        val_loss = evaluate(args, device, val_loader, model)

        # Save the model after each epoch
        model_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Optionally save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")

        
    print("Training complete.")

    # Plot the training loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args['epoch'] + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.show()

def evaluate(args, device, val_loader, model):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            sequence_inputs = batch['sequence_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            graph_inputs = batch['graph_features']
            if graph_inputs is not None:
                graph_inputs = graph_inputs.to(device)
                # graph_inputs = [g.to(device) for g in graph_inputs]
            
            # Forward pass
            outputs = model(sequence_inputs, attention_mask, graph_inputs)
            
            # Combined logits shape is [batch_size, 2], we need to get class predictions (0 or 1)
            preds = torch.argmax(outputs, dim=1)  # Get the predicted class for each sample
            labels_np = labels.cpu().numpy()  # Convert labels to numpy
            
            all_preds.append(preds.cpu().numpy())  # Convert predictions to numpy and append
            all_labels.append(labels_np)
            
        # Flatten the accumulated predictions and labels
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # Debug print
        print(f"all_preds shape: {all_preds.shape}")
        print(f"all_labels shape: {all_labels.shape}")

        # Calculate evaluation metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
            
        print(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return total_loss



# Main code
# Example usage
json_file = "/home/ngan/Documents/SamVulDetection/dataset/megavul-001.json"
# Load and split data
train_data, val_data, test_data, train_labels, val_labels, test_labels = load_and_split_data(json_file)

# Sau khi tách
print("extract_data_for_training sucessfully")

# Update to include tokenizer and args
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
data_args = {
    'model_type': 'codegen',  # Or your specific model type
    'block_size': 128,  # Example block size
    'max_graph_size': 128
}

train_dataset = MegaVulDataset(train_data, train_labels, tokenizer, data_args)
val_dataset = MegaVulDataset(val_data, val_labels, tokenizer, data_args)
test_dataset = MegaVulDataset(test_data, test_labels, tokenizer, data_args)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
print("Create DataLoaders")


# Assuming you have already split your data as shown in the previous steps

# Initialize the model
clr_model = CLRModel(num_labels=2)
devign_model = DevignModel()  # Adjust dimensions based on your data
combined_model = CombinedModel(clr_model, devign_model)

print('Init model sucessfully')
num_epochs = 10
# Assuming you have a dataloader for the validation or test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
combined_model.to(device)

# Train model
print('Train model')

optimizer = torch.optim.Adam(combined_model.parameters(), lr=1e-4)

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
optimizer = torch.optim.Adam(combined_model.parameters(), lr=args['learning_rate'])
loss_function = torch.nn.CrossEntropyLoss()
# Call train function
train(args, device, train_loader, val_loader, combined_model, optimizer, loss_function)

