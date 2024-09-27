import torch
import torch.nn as nn
import json
import os
import torch

from dig.xgraph.method import DeepLIFT
from transformers_interpret import SequenceClassificationExplainer
from transformers import RobertaTokenizer
from torch.optim.lr_scheduler import StepLR
"""
explanations/
    epoch_1/
        batch_0.json
        batch_1.json
        ...
    epoch_2/
        batch_0.json
        batch_1.json
        ...
    ...

"""
def save_to_file(data, file_path):
    """
    Save data to a JSON file.
    
    Parameters:
    - data: The data to save (can be a dict, list, etc.).
    - file_path (str): The path to the file where the data will be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)  # Use indent for pretty printing
    print(f"Data saved to {file_path}")
    

def store_explanations(epoch, batch_idx, sequence_explanations, graph_explanations, expl_dir, node_attributions=None):
    """
    Store explanations for a given epoch and batch.

    Parameters:
    - epoch (int): The current epoch number.
    - batch_idx (int): The index of the current batch.
    - sequence_explanations (dict): Explanations for the sequence model.
    - graph_explanations (dict): Explanations for the graph model.
    - expl_dir (str): The directory where the explanations will be saved.
    - node_attributions (dict, optional): Node attributions to save.
    """
    # Create the directory for the current epoch if it doesn't exist
    epoch_dir = os.path.join(expl_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Prepare the file path for sequence explanations
    sequence_file_path = os.path.join(epoch_dir, f'sequence_explanations_batch_{batch_idx}.json')
    save_to_file(sequence_explanations, sequence_file_path)

    # Store graph explanations if available
    if graph_explanations is not None:
        # Save edge masks and other graph explanations as needed
        edge_masks_file_path = os.path.join(epoch_dir, f'graph_edge_masks_batch_{batch_idx}.json')
        save_to_file(graph_explanations['edge_masks'], edge_masks_file_path)

        node_idx_file_path = os.path.join(epoch_dir, f'graph_node_idx_batch_{batch_idx}.json')
        save_to_file(graph_explanations['node_idx'], node_idx_file_path)
    
    # Store node attributions if provided
    if node_attributions is not None:
        node_attributions_file_path = os.path.join(epoch_dir, f'node_attributions_batch_{batch_idx}.json')
        save_to_file(node_attributions, node_attributions_file_path)

        
def load_explanations(epoch, batch_idx, expl_dir='explanations'):
    if epoch < 0:  # No previous explanations available if it's the first epoch
        return None
    
    epoch_dir = os.path.join(expl_dir, f'epoch_{epoch}')
    explanation_file = os.path.join(epoch_dir, f'batch_{batch_idx}.json')
    
    if os.path.exists(explanation_file):
        with open(explanation_file, 'r') as f:
            explanations = json.load(f)
        print(f"Loaded explanations from {explanation_file}")
        return explanations
    else:
        print(f"No explanations found for batch {batch_idx} in epoch {epoch}")
        return None


def sequence_explanation_loss(importance_scores, model_outputs, labels):
    """
    Calculate the loss based on sequence explanation importance scores.
    
    Penalize more important tokens when the model makes incorrect predictions.
    """
    # Convert model outputs and labels to probabilities or logits if needed
    model_outputs = torch.softmax(model_outputs, dim=1)  # Assuming binary classification
    labels = labels.float()

    seq_loss = 0
    for i, score in enumerate(importance_scores):
        error = torch.abs(model_outputs[i] - labels[i])  # Calculate error
        seq_loss += score * error  # Penalize based on importance score and error magnitude
    
    return seq_loss.mean()  # Return the average loss over the batch


def graph_explanation_loss(node_importance, model_outputs, labels):
    """
    Calculate the loss based on graph explanation node importance.
    
    Penalize more important nodes when the model makes incorrect predictions.
    """
    # Convert model outputs and labels to probabilities or logits if needed
    model_outputs = torch.softmax(model_outputs, dim=1)
    labels = labels.float()

    graph_loss = 0
    for i, node_score in enumerate(node_importance):
        error = torch.abs(model_outputs[i] - labels[i])
        graph_loss += node_score * error  # Penalize based on node importance and error magnitude
    
    return graph_loss.mean()  # Return the average loss over the batch


def explanation_loss(sequence_explanations, graph_explanations, model_outputs, labels):
    """
    Combine sequence and graph explanation losses.
    """
    # Calculate sequence explanation loss
    sequence_loss = sequence_explanation_loss(sequence_explanations['importance_scores'], model_outputs, labels)
    
    # Initialize graph_loss to zero if graph_explanations is None
    graph_loss = 0.0
    if graph_explanations is not None:
        # Calculate graph explanation loss
        # graph_loss = graph_explanation_loss(graph_explanations['node_attributions'], model_outputs, labels)

        # Unpack graph explanations to get edge masks and node attributions
        edge_masks, node_attributions = graph_explanations

        # Calculate graph explanation loss using node attributions
        graph_loss = graph_explanation_loss(node_attributions, model_outputs, labels)

    
    # Combine both losses
    total_explanation_loss = sequence_loss + graph_loss
    return total_explanation_loss

def compute_node_attributions(edge_masks, edge_index):
    # Initialize a tensor to accumulate node attributions
    node_attributions = torch.zeros(max_node_index, device=edge_masks.device)  # Adjust max_node_index as needed
    
    for edge, mask in zip(edge_index, edge_masks):
        src, dst = edge  # Extract source and destination nodes from the edge index
        node_attributions[src] += mask  # Aggregate attribution to the source node
        node_attributions[dst] += mask  # You can adjust this logic based on your needs

    return node_attributions

def explain_graph_outputs(model, outputs, g_batch, device):
    edge_explanations = []
    node_explanations = []
    node_features = g_batch.ndata['type'].to(device) # Ensure this is the right feature tensor
    edge_index = g_batch.edges() # This could return a tuple (src, dst)

    model.set_explainer_mode(True)
    # Convert the edge_index components to tensors
    if isinstance(edge_index, tuple):
        print(f'edge_index is a tuple')
        src, dst = edge_index  # Unpack the source and destination
        edge_index = torch.stack([src, dst], dim=0).to(device)  # Stack into a 2D tensor
    else:
        edge_index = edge_index.to(device)  # If already a tensor, just move to device

    # Debugging: Check shapes
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")


    graph_explainer = DeepLIFT(model, explain_graph=False) # false because node classification
    for i, g in enumerate(dgl.unbatch(g_batch)):
        print(outputs.shape)

        node_idx = torch.argmax(outputs[i], dim=0).item()
        
        # Generate edge explanations
        edge_masks, hard_edge_masks, related_preds = graph_explainer(node_features, edge_index, node_idx=node_idx)
        
        
        # Check the shapes of edge_masks
        print(f"Edge masks shape: {edge_masks.shape}")

        edge_explanations.append(edge_mask)

        # Compute node attributions from edge masks
        node_attribution = compute_node_attributions(edge_mask, g.edges())
        node_explanations.append(node_attribution)

    model.set_explainer_mode(False)

    return edge_explanations, node_explanations

# def explain_graph_outputs(model, outputs, g_batch, device):
#     # g_batch = g_batch.to(device)
#     # node_features = g_batch.ndata['type'].to(device)
#     edge_index = g_batch.edges()
#     node_idx = torch.argmax(outputs, dim=1).item()
    
#     graph_explainer = DeepLift(model)

#     # Initialize lists to store explanations for each graph in the batch
#     edge_explanations = []
#     node_explanations = []
#     # # Compute node attributions from edge masks if necessary
#     # node_attributions = compute_node_attributions(edge_masks, edge_index)
#     for i, node_idx in enumerate(node_idxs):
#         # Extract the graph corresponding to the current batch index
#         current_g = dgl.unbatch(g_batch)[i]

#         # Generate explanation for the current graph and node index
#         edge_mask = graph_explainer.attribute(
#             current_g, node_idx=node_idx.item(), target=node_idx.item()
#         )
#         # Calculate node attributions if supported by your implementation
#         node_mask = graph_explainer.attribute(
#             current_g, node_idx=node_idx.item(), target=node_idx.item(), return_node_attributions=True
#         )

#         # Store the explanations for this sample
#         edge_explanations.append(edge_mask)
#         node_explanations.append(node_mask)

#     return edge_explanations, node_explanations

def explain_sequence_outputs(model, sequence_inputs, predicted_class_indices, tokenizer):
    sequence_explanations = []
    
    # Decode token IDs into text (if sequence_inputs are token IDs)
    sequence_texts = tokenizer.batch_decode(sequence_inputs, skip_special_tokens=True)

    # Initialize the explainer with the model and tokenizer
    explainer = SequenceClassificationExplainer(model, tokenizer)

    # Loop through each sequence and generate explanations
    for i, text in enumerate(sequence_texts):
        # Explain the predicted output (i.e., argmax result for each input)
        explanation_result = explainer(text, index=predicted_class_indices[i])

        # If explanation_result is a list of importance scores, directly append it
        if isinstance(explanation_result, list):
            sequence_explanations.append(explanation_result)
        else:
            # If explanation_result has a get_importance_scores method, call it
            sequence_explanations.append(explanation_result.get_importance_scores())

    return sequence_explanations

def train(args, device, train_loader, val_loader, model, optimizer, loss_function):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Initialize the explainability model
    # graph_explainer = DeepLIFT(model=model.devign_model, explain_graph=False)
    cls_explainer = SequenceClassificationExplainer(
        model.clr_model.model,
        tokenizer)

    train_losses = []
    best_val_loss = float('inf')
    save_dir = '/home/ngan/Documents/SamVulDetection/saved_models'
    expl_dir = '/home/ngan/Documents/SamVulDetection/saved_explains'
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # Freeze Roberta layers for the first `freeze_epochs` epochs
    if hasattr(model, 'clr_model'):
        for param in model.clr_model.parameters():
            param.requires_grad = False
    for epoch in range(args['epoch']):

        print(f'Epoch: {epoch}')

        total_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            print(f'batch_idx: {batch_idx}')
            sequence_inputs = batch['sequence_ids']
            attention_mask = batch['attention_mask']
            graph_inputs = batch['graph_features']
            labels = batch['label'].to(device).long()  # Ensure labels are of dtype long (for classification)

            # Assuming you have sequence_inputs, attention_mask, and graph_inputs prepared
            print(f"Sequence inputs shape: {sequence_inputs.shape}")  # Should be [batch_size, seq_length]
            print(f"Attention mask shape: {attention_mask.shape}")    # Should match sequence_inputs

            if graph_inputs is not None:
                # print(f"Graph inputs shape: {graph_inputs.shape}")      # Make sure this is consistent in terms of batch size
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

            optimizer.zero_grad()

            # Calculate loss
            classification_loss  = loss_function(outputs, labels)

            # Load explanations from previous epoch
            prev_explanations = load_explanations(epoch - 1, batch_idx, expl_dir)
            expl_loss = 0.0
            if prev_explanations:
                sequence_explanations = prev_explanations['sequence_explanations']
                graph_explanations = prev_explanations['graph_explanations']
                
                # Calculate explanation loss based on importance scores and attributions
                expl_loss = explanation_loss(sequence_explanations, graph_explanations, outputs, labels)
            
            # Combine classification loss with explanation-based loss
            total_loss_combined = classification_loss + expl_loss

            # Backpropagation and optimization
            total_loss_combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            total_loss += total_loss_combined.item()


            # Generate new sequence explanations
            # Predict the class labels from model outputs
            predicted_class_indices = outputs.argmax(dim=1).tolist()

            # Call the sequence explanation function
            sequence_explanations = explain_sequence_outputs(
                model.clr_model.model, 
                sequence_inputs, 
                predicted_class_indices, 
                tokenizer
            )

            # Generate new graph explanations using DeepLIFT (with node_idx based on outputs)
            if graph_inputs is not None:
                edge_masks, node_attributions = explain_graph_outputs(
                    model.devign_model, outputs, g_batch=graph_inputs, device=device
                )
                graph_explanations = {
                    'edge_masks': edge_masks.tolist(),
                    'node_attributions': node_attributions  # Store node attributions here
                }
            else:
                graph_explanations = None

            # Assuming you have generated sequence_explanations, graph_explanations, and node_attributions
            # store_explanations(epoch, batch_idx, sequence_explanations, graph_explanations, expl_dir, node_attributions)
            store_explanations(epoch, batch_idx, sequence_explanations, graph_explanations, expl_dir)

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
        val_loss = evaluate(args, device, val_loader, model, loss_function)
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


