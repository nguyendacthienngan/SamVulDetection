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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN  # Combines SMOTE (oversampling) and Edited Nearest Neighbors (undersampling)

import numpy as np
import dgl

def load_and_split_data(json_file_path, apply_combined_sampling=True):
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract data and labels
    data_items = [item for item in data]
    labels = [item['is_vul'] for item in data]

    # Split data into train, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data_items, labels, test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

    # Check label distribution before resampling
    print(f"Training label distribution before resampling: {np.bincount(train_labels)}")

    if apply_combined_sampling:
        # First apply undersampling to the majority class
        undersampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Retain 80% of the majority class
        train_data, train_labels = undersampler.fit_resample(np.array(train_data).reshape(-1, 1), train_labels)
        train_data = [item[0] for item in train_data]  # Flatten data back

        # Then apply oversampling to the minority class
        oversampler = RandomOverSampler(sampling_strategy=1.0, random_state=42)  # Match minority class to majority
        train_data_resampled, train_labels_resampled = oversampler.fit_resample(np.array(train_data).reshape(-1, 1), train_labels)
        train_data_resampled = [item[0] for item in train_data_resampled]

        # Check label distribution after resampling
        print(f"Training label distribution after combined resampling: {np.bincount(train_labels_resampled)}")
    else:
        train_data_resampled = train_data
        train_labels_resampled = train_labels

    return train_data_resampled, val_data, test_data, train_labels_resampled, val_labels, test_labels



def collate_fn(batch):
    sequence_ids = [torch.tensor(item['sequence_ids']) for item in batch]
    attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
    graph_features = [item['graph_features'] for item in batch if item['graph_features'] is not None]
    labels = [item['label'] for item in batch]
    raw_codes = [item['raw_code'] for item in batch]

    sequence_ids_tensor = torch.stack(sequence_ids)
    attention_masks_tensor = torch.stack(attention_masks)
    labels_tensor = torch.tensor(labels)

    if len(graph_features) > 0:
        graph_features_tensor = dgl.batch(graph_features)
    else:
        graph_features_tensor = None

    return {
        'sequence_ids': sequence_ids_tensor,
        'attention_mask': attention_masks_tensor,
        'graph_features': graph_features_tensor,
        'raw_code': raw_codes,
        'label': labels_tensor
    }

from torch.optim.lr_scheduler import StepLR
def train(args, device, train_loader, val_loader, model, optimizer, loss_function, freeze_epochs=5):
    train_losses = []
    best_val_loss = float('inf')
    save_dir = '/home/ngan/Documents/SamVulDetection/saved_models'
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # Freeze Roberta layers for the first `freeze_epochs` epochs
    if hasattr(model, 'clr_model'):
        for param in model.clr_model.parameters():
            param.requires_grad = False
    for epoch in range(args['epoch']):

        print(f'Epoch: {epoch}')
        
        # Unfreeze Roberta after `freeze_epochs`
        if epoch == freeze_epochs:
            print(f"Unfreezing Roberta parameters at epoch {epoch}")
            if hasattr(model, 'clr_model'):
                for param in model.clr_model.parameters():
                    param.requires_grad = True
                
        total_loss = 0.0
        model.train() 

        for batch_idx, batch in enumerate(train_loader):
            print(f'batch_idx: {batch_idx}')
            sequence_inputs = batch['sequence_ids']
            attention_mask = batch['attention_mask']
            graph_inputs = batch['graph_features']
            labels = batch['label'].to(device).long()  # Ensure labels are of dtype long (for classification)

            if graph_inputs is not None:
                graph_inputs = graph_inputs.to(device)

            # Move data to device
            sequence_inputs = sequence_inputs.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            if isinstance(model, PhpNetGraphTokensCombine):
                if graph_inputs is not None:
                    outputs = model(sequence_inputs, graph_inputs)
                else:
                    outputs = model(sequence_inputs)
            else:
                if graph_inputs is not None:
                    outputs = model(sequence_inputs, attention_mask, graph_inputs)
                else:
                    outputs = model(sequence_inputs, attention_mask)
    
            # Calculate loss
            loss = loss_function(outputs, labels)
            
            # Backward pass and update
            # if (batch_idx + 1) % args['gradient_accumulation_steps'] == 0:
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
            optimizer.step()
            total_loss += loss.item()
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args['epoch']}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradients for {name}: {param.grad.mean().item()}")
            else:
                print(f"Gradients for {name}: None")

        
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args['epoch']}, Average Loss: {avg_loss:.4f}")
        
        # Save the model after each epoch
        model_name = 'PhpNetGraphTokensCombine' if isinstance(model, PhpNetGraphTokensCombine) else 'CombinedModel'
        model_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate model on validation set
        val_loss = evaluate(args, device, val_loader, model)
        print(f"Epoch {epoch + 1}/{args['epoch']}, Validation Loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"Epoch {epoch + 1}: Learning rate: {scheduler.get_last_lr()}")

        # Optionally save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")



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
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(all_labels, all_preds)
        # precision = precision_score(all_labels, all_preds)
        # recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        unique, counts = np.unique(all_preds, return_counts=True)
        print(f"Predictions distribution: {dict(zip(unique, counts))}")

        recall = recall_score(all_labels, all_preds, average='binary')
        unique, counts = np.unique(all_preds, return_counts=True)
        print(f"Predictions distribution: {dict(zip(unique, counts))}")

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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
print("DataLoaders created with resampled training data.")

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

# Set training arguments
args = {
    'learning_rate': 1e-5,
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

# Initialize optimizer, and loss function
optimizer = torch.optim.Adam(combined_model.parameters(), lr=args['learning_rate'])

class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_labels)
print(f'class_weights: {class_weights}')
# class_weights = torch.tensor([1.0, class_weights[1]]).to(device)
class_weights = torch.tensor([1.0, class_weights[1]], dtype=torch.float32).to(device)
# class_weights = torch.tensor([1.0, 20.0], dtype=torch.float32).to(device)  # Adjust weight for class 1
print(f'new class_weights: {class_weights}')

loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
# Call train function
from models.phpnet import PhpNetGraphTokensCombine
php_model = PhpNetGraphTokensCombine().to(device)


# train(args, device, train_loader, val_loader, combined_model, optimizer, loss_function)
train(args, device, train_loader, val_loader, php_model, optimizer, loss_function, 0)

# Plot the training loss over epochs after the training loop
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, args['epoch'] + 1), train_losses, marker='o', linestyle='-', color='b')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss over Epochs')
# plt.grid(True)
# plt.show()


# model = combined_model
# model_path = '/home/ngan/Documents/SamVulDetection/saved_models/model_epoch_1.pth'
# model.load_state_dict(torch.load(model_path))
# total_loss = 0
# all_labels = []
# all_preds = []
# model.eval()
# with torch.no_grad():
#     for batch_idx, batch in enumerate(val_loader):
#         sequence_inputs = batch['sequence_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
        
#         graph_inputs = batch['graph_features']
#         if graph_inputs is not None:
#             graph_inputs = graph_inputs.to(device)
#             # graph_inputs = [g.to(device) for g in graph_inputs]
        
#         # Forward pass
#         outputs = model(sequence_inputs, attention_mask, graph_inputs)

        
#         # Combined logits shape is [batch_size, 2], we need to get class predictions (0 or 1)
#         preds = torch.argmax(outputs, dim=1)  # Get the predicted class for each sample

#         print(f"Labels: {labels.cpu().numpy()}")
#         print(f"Predictions: {preds.cpu().numpy()}")
#         labels_np = labels.cpu().numpy()  # Convert labels to numpy
        
#         all_preds.append(preds.cpu().numpy())  # Convert predictions to numpy and append
#         all_labels.append(labels_np)
#         break
        
#     # Flatten the accumulated predictions and labels
#     all_preds = np.concatenate(all_preds).flatten()
#     all_labels = np.concatenate(all_labels).flatten()
    
#     # Calculate evaluation metrics
#     accuracy = accuracy_score(all_labels, all_preds)
#     # precision = precision_score(all_labels, all_preds)
#     # recall = recall_score(all_labels, all_preds)
#     precision = precision_score(y_true, y_pred, zero_division=1)

#     precision = precision_score(all_labels, all_preds, average='binary')
#     unique, counts = np.unique(all_preds, return_counts=True)
#     print(f"Predictions distribution: {dict(zip(unique, counts))}")

#     recall = recall_score(y_true, y_pred, zero_division=1)
#     unique, counts = np.unique(all_preds, return_counts=True)
#     print(f"Predictions distribution: {dict(zip(unique, counts))}")

#     print(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
