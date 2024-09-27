import torch
import torch.nn as nn
import json
import os

# from dig.xgraph.method import DeepLIFT
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

def store_explanations(epoch, batch_idx, sequence_explanations, graph_explanations, expl_dir):
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
        node_attributions_file_path = os.path.join(epoch_dir, f'graph_node_attributions_batch_{batch_idx}.json')
        save_to_file(graph_explanations['node_attributions'], node_attributions_file_path)
    
        
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


def explain_graph_outputs(model, outputs, g_batch, device):
    edge_explanations = []
    node_explanations = []
    node_features = g_batch.ndata['type'].to(device) # Ensure this is the right feature tensor
    edge_index = g_batch.edges() # This could return a tuple (src, dst)

    model.set_explainer_mode(True)
    num_classes = model.num_classes  # Ensure this is set in your model
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
        # print(outputs.shape) #[16,2]

        # node_idx = torch.argmax(outputs[i], dim=0).item()
        
        # Use tensor instead of item for node_idx
        node_idx = torch.argmax(outputs[i], dim=0).unsqueeze(0).to(device)

        # Generate edge explanations
        results = graph_explainer(node_features, edge_index)

        node_attributions = results.sort(descending=True).indices.cpu()

        # Compute node attributions from edge masks
        # node_attribution = compute_node_attributions(edge_mask, g.edges())
        node_explanations.append(node_attributions)

    model.set_explainer_mode(False)

    return node_explanations

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
                node_explanations = explain_graph_outputs(
                    model.devign_model, outputs, g_batch=graph_inputs, device=device
                )
                graph_explanations = {
                    'node_attributions': node_explanations  # Store node attributions here
                }
            else:
                graph_explanations = None

            # Assuming you have generated sequence_explanations, graph_explanations, and node_attributions
            # store_explanations(epoch, batch_idx, sequence_explanations, graph_explanations, expl_dir, node_attributions)
            store_explanations(epoch, batch_idx, sequence_explanations, graph_explanations, expl_dir)

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args['epoch']}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss_combined.item():.4f}")
            if test_mode is True:
                break
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


