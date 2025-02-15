import torch
import torch.nn as nn
import json
import os

# from dig.xgraph.method import DeepLIFT
from transformers_interpret import SequenceClassificationExplainer
from transformers import RobertaTokenizer
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import dgl
def visualize_sequence_explanations(sequence_explanations, sequence_texts):
    for i, (explanations, text) in enumerate(zip(sequence_explanations, sequence_texts)):
        # Extract tokens and their importance scores
        tokens = [score[0] for score in explanations]
        scores = np.array([score[1] for score in explanations])

        # Plot token importance using a bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(tokens, scores, color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel("Tokens")
        plt.ylabel("Importance Score")
        plt.title(f"Token Importance for Sequence {i}")
        plt.tight_layout()

        # Save or show the plot
        plt.savefig(f"sequence_explanations_{i}.png")
        plt.show()
        

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
def convert_tensors_to_lists(data):
    """
    Recursively converts all tensors in a structure (dict, list, etc.) to lists.
    """
    if isinstance(data, torch.Tensor):
        return data.tolist()  # Convert tensor to list
    elif isinstance(data, dict):
        return {key: convert_tensors_to_lists(value) for key, value in data.items()}  # Recursively apply for dict
    elif isinstance(data, list):
        return [convert_tensors_to_lists(item) for item in data]  # Recursively apply for list
    elif isinstance(data, tuple):
        return tuple(convert_tensors_to_lists(item) for item in data)  # Recursively apply for tuple
    else:
        return data  # Return the data as is if it's not a tensor

def save_to_file(data, file_path):
    """
    Save data to a JSON file. If the data contains tensors, convert them to lists first.
    """
    # Convert all tensors in the structure to lists
    data = convert_tensors_to_lists(data)
    
    # Write to the JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)  # Use indent for pretty printing
    print(f"Data saved to {file_path}")

def store_explanations(epoch, batch_idx, node_explanations, edge_explanations, expl_dir='explanations'):
    os.makedirs(expl_dir, exist_ok=True)
    
    for i, (nodes, edges) in enumerate(zip(node_explanations, edge_explanations)):
        graph_data = {'nodes': [], 'edges': []}
        
        # Store node explanations with both label and attribution score
        for node_id, node_importance in nodes.items():
            graph_data['nodes'].append({
                'id': node_id,
                'label': node_importance['label_node'],  # Use 'label_node' instead of converting dict
                # 'attribution_score': node_importance['attribution_score']  # Store attribution score
            })
        
        # Store edge explanations
        for edge in edges:
            graph_data['edges'].append({
                'in_node': edge['in_node'],
                'out_node': edge['out_node'],
                'important_score': edge['important_score']
            })
        
        # Save explanations for this batch
        file_path = os.path.join(expl_dir, f'explanations_epoch_{epoch}_batch_{batch_idx}_graph_{i}.json')
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=4)


def explain_and_store(epoch, batch_idx, model, outputs, sequence_inputs, node_features, graph_inputs, device, tokenizer, expl_dir):
    """
    Generate and store explanations for both sequence and graph outputs during training.
    """
    # Explain sequence outputs
    # predicted_class_indices = outputs.argmax(dim=1).tolist()
    predicted_class_indices = torch.argmax(outputs, dim=1).tolist()
    sequence_explanations = explain_sequence_outputs(model.clr_model.model, sequence_inputs, predicted_class_indices, tokenizer)

    # Extract the importance scores for combining with graph explanations
    sequence_importance = [explanation[1] for explanation in sequence_explanations]  # Only use the importance scores
    
    # Explain graph outputs
    if graph_inputs is not None:
        node_explanations, edge_explanations = explain_graph_outputs(model.devign_model, outputs, graph_inputs, device, node_features)
    else:
        node_explanations, edge_explanations = [], []

    # Store the combined explanations
    store_explanations(epoch, batch_idx, node_explanations, edge_explanations, expl_dir)

def load_explanations(epoch, batch_idx, expl_dir='explanations'):
    if epoch < 0:  # No previous explanations available if it's the first epoch
        return None
    
    epoch_dir = os.path.join(expl_dir, f'epoch_{epoch}')
    
    # Load sequence explanations
    sequence_file = os.path.join(epoch_dir, f'sequence_explanations_batch_{batch_idx}.json')
    graph_file = os.path.join(epoch_dir, f'graph_node_attributions_batch_{batch_idx}.json')

    explanations = {}
    
    if os.path.exists(sequence_file):
        with open(sequence_file, 'r') as f:
            explanations['sequence_explanations'] = json.load(f)
        print(f"Loaded sequence explanations from {sequence_file}")
    else:
        print(f"No sequence explanations found for batch {batch_idx} in epoch {epoch}, file: {sequence_file}")
    
    # Load graph explanations if available
    if os.path.exists(graph_file):
        with open(graph_file, 'r') as f:
            explanations['graph_explanations'] = json.load(f)
        print(f"Loaded graph explanations from {graph_file}")
    else:
        print(f"No graph explanations found for batch {batch_idx} in epoch {epoch}, file: {graph_file}")

    return explanations if explanations else None

def sequence_explanation_loss(importance_scores, model_outputs, labels):
    seq_loss = torch.tensor(0.0, device=model_outputs.device)  # Initialize as tensor

    for i, scores in enumerate(importance_scores):
        # Extract only the numeric scores
        numeric_scores = [score[1] for score in scores]  # Get only the numeric part
        
        # Ensure the numeric_scores are in tensor format
        numeric_scores_tensor = torch.tensor(numeric_scores, device=model_outputs.device)

        # Calculate error (assuming model_outputs and labels are tensors)
        error = torch.abs(model_outputs[i] - labels[i])  # Calculate error

        # If no numeric scores, skip this instance
        if numeric_scores_tensor.numel() == 0:
            print(f"Warning: No valid scores for instance {i}. Skipping.")
            continue

        # Aggregate numeric scores (e.g., mean across tokens)
        if numeric_scores_tensor.shape[0] != 2:
            numeric_scores_tensor = numeric_scores_tensor.mean()  # Use mean if more than 2
        else:
            numeric_scores_tensor = numeric_scores_tensor.sum()  # If exactly 2, sum them

        # Compute the weighted error based on numeric scores
        weighted_error = numeric_scores_tensor * error.mean()  # Average score for loss calculation
        seq_loss += weighted_error

    return seq_loss.mean()  # Return the average loss over the batch


def graph_explanation_loss(node_importance, model_outputs, labels):
    """
    Calculate the loss based on graph explanation node importance.
    
    Penalize more important nodes when the model makes incorrect predictions.
    """
    model_outputs = torch.softmax(model_outputs, dim=1)
    labels = labels.float()

    graph_loss = 0
    for i, node_score in enumerate(node_importance):
        if isinstance(node_score, list):
            node_score = torch.tensor(node_score, device=model_outputs.device)

        # Skip if no importance
        if node_score.sum() == 0:
            continue

        # Calculate error
        error = torch.abs(model_outputs[i] - labels[i])

        # Normalize the node score
        norm_node_score = torch.sigmoid(node_score) / (torch.sum(node_score) + 1e-8)

        # print(f'Node Score: {node_score}, Error: {error.mean()}, Graph Loss Increment: {norm_node_score.mean() * error.mean()}')

        # Compute the weighted graph loss
        graph_loss += norm_node_score.mean() * error.mean()

    # Average loss over the batch
    return graph_loss.mean()  # Normalize the graph loss to prevent large values

def explanation_loss(sequence_explanations, graph_explanations, model_outputs, labels):
    """
    Calculate the combined explanation loss based on sequence and graph explanations.
    """
    # Calculate sequence explanation loss
    sequence_loss = sequence_explanation_loss(sequence_explanations, model_outputs, labels)

    graph_loss = 0.0
    if graph_explanations is not None:
        # Check if graph_explanations is a list and handle accordingly
        if isinstance(graph_explanations, list):
            # Assuming each element in graph_explanations is a list of attributions
            node_attributions = [explanation[0] for explanation in graph_explanations]  # Adjust index if necessary
            
            # Convert node_attributions to tensor if necessary
            node_attributions = [torch.tensor(attrib, device=model_outputs.device) for attrib in node_attributions]

        else:
            # Handle case where graph_explanations is not a list (if applicable)
            # Assuming graph_explanations is a dictionary or something else
            node_attributions = graph_explanations['node_attributions']
            node_attributions = torch.tensor(node_attributions, device=model_outputs.device)  # Convert to tensor
        graph_loss = graph_explanation_loss(node_attributions, model_outputs, labels)

    # Combine losses
    print(f'sequence_loss: {sequence_loss}')
    print(f'graph_loss: {graph_loss}')

    total_loss = sequence_loss + graph_loss
    print(f'total_loss: {graph_loss}')

    return total_loss
def explain_graph_outputs(model, outputs, g_batch, device, node_features):
    edge_explanations = []
    node_explanations = []
    node_features_type = g_batch.ndata['type'].to(device)  # Node features ('type' field)
    edge_index = g_batch.edges()  # This returns (src, dst) as a tuple of tensors

    model.set_explainer_mode(True)

    if isinstance(edge_index, tuple):
        src, dst = edge_index
        edge_index = torch.stack([src, dst], dim=0).to(device)

    # Initialize DeepLIFT for graph explanation
    graph_explainer = DeepLIFT(model, explain_graph=False)
    
    for i, g in enumerate(dgl.unbatch(g_batch)):  # Unbatch for individual graphs
        node_idx = torch.argmax(outputs[i], dim=0).unsqueeze(0).to(device)  # Get the most likely node (optional)
        
        # Explain node attributions using DeepLIFT
        results = graph_explainer(node_features_type, edge_index)  # Apply DeepLIFT on node features and edges
        node_attributions = results.sort(descending=True).indices.cpu()  # Sort attributions in descending order

        # Map node attributions to node indices and labels (using node_features)
        # node_expl = {idx: {'label_node': node_features[idx]}  # Store string type from node_features
        #              for idx in range(len(node_attributions))}


        # Map node attributions to node indices and labels (using node_features)
        node_expl = {}
        for idx in range(len(node_attributions)):
            if idx < len(node_features) and node_features[idx] is not None:  # Ensure node_features[idx] is valid
                label_node = node_features[idx][idx]['type']
                print(f'label_node: {label_node}')

                node_expl[idx] = {'label_node': label_node}  # Store string type from node_features

        node_explanations.append(node_expl)

        # Compute edge explanations based on node attributions
        edge_expl = []
        for src_node, dst_node in zip(src.tolist(), dst.tolist()):
            print(f"src_node: {src_node}, dst_node: {dst_node}")
            
            # Use the attributions of the connected nodes to calculate edge importance
            if src_node < len(node_attributions) and dst_node < len(node_attributions):
                edge_importance_score = (node_attributions[src_node].item() + node_attributions[dst_node].item()) / 2.0
            else:
                edge_importance_score = 0  # Fallback if node indices are out of bounds

            # Append edge explanation
            edge_expl.append({
                'in_node': src_node,
                'out_node': dst_node,
                'important_score': edge_importance_score
            })

        edge_explanations.append(edge_expl)

    return node_explanations, edge_explanations

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

        # Append the list of importance scores directly
        sequence_explanations.append(explanation_result)

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
            node_features = batch['node_features']
            labels = batch['label'].to(device).long()  # Ensure labels are of dtype long (for classification)
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

            # visualize_graph_explanations(graph_inputs, test_explanations, batch_idx)
            # visualize_ast(graph_inputs, batch_idx)

            # visualize_ast(graph_inputs, batch_idx, test_explanations)

            # return
            
            # Calculate loss
            classification_loss  = loss_function(outputs, labels)

            # Load explanations from previous epoch
            if test_mode is True:
                load_epoch = epoch
            else:
                load_epoch = epoch  -1
            prev_explanations = load_explanations(load_epoch, batch_idx, expl_dir)
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

            explain_and_store(epoch, batch_idx, model, outputs, sequence_inputs, node_features, graph_inputs, device, tokenizer, expl_dir)

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args['epoch']}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss_combined.item():.4f}")
            if test_mode is True:
                break

            # Visualize explanations
            # visualize_sequence_explanations(sequence_explanations, tokenizer.batch_decode(sequence_inputs, skip_special_tokens=True))
            # if graph_inputs is not None:
            #     visualize_graph_explanations(graph_inputs, graph_explanations, batch_idx)
        avg_loss = total_loss / len(train_loader)

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.mean().item()}")
        #     else:
        #         print(f"Gradients for {name}: None")


        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args['epoch']}, Average Loss: {avg_loss:.4f}")

        # Save the model after each epoch
        model_name = 'PhpNetGraphTokensCombine' if isinstance(model, PhpNetGraphTokensCombine) else 'CombinedModelExp'
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


