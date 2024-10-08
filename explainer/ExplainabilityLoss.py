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



def visualize_ast(g_batch, batch_idx, save_dir='/home/ngan/Documents/SamVulDetection/saved_pdf/'):
    """
    Visualize the AST graph by filtering out non-AST edges.

    Args:
    - g_batch: DGL batch of graphs with edge types (AST, CFG, etc.).
    - batch_idx: Index of the current batch.
    - save_dir: Directory to save the graph image (optional).
    """
    # Unbatch the graph if g_batch contains multiple graphs
    graphs = dgl.unbatch(g_batch)
    g = graphs[batch_idx]  # Select the specific graph for this batch
    
    # Create a NetworkX graph from DGL graph
    AST_graph = nx.DiGraph()  # DiGraph for directed AST
    
    # Retrieve edge labels
    edge_labels = g.edata['label'].tolist()
    
    # Only include AST edges in the visualization
    for i, (u, v) in enumerate(zip(g.edges()[0], g.edges()[1])):
        if edge_labels[i] == 0:  # Assuming 0 corresponds to AST edge in your mapping
            AST_graph.add_edge(u.item(), v.item())

    # Add node attributes (e.g., 'type' or any other feature you want to visualize)
    node_types = g.ndata['type'].tolist()
    for node in AST_graph.nodes():
        AST_graph.nodes[node]['type'] = node_types[node]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(AST_graph, seed=42)  # Layout for graph visualization
    
    # Draw nodes with labels (node type)
    nx.draw(AST_graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
    
    # Draw edge labels if needed
    nx.draw_networkx_edge_labels(AST_graph, pos)
    
    plt.title(f"AST Graph Visualization - Batch {batch_idx}")
    
    # Save the graph or show it
    if save_dir:
        plt.savefig(f"{save_dir}/AST_graph_batch_{batch_idx}.png")
    else:
        plt.show()

def visualize_ast(g_batch, batch_idx, node_importance_scores, save_dir='/home/ngan/Documents/SamVulDetection/saved_pdf/'):
    """
    Visualize the AST graph by filtering out non-AST edges and highlighting important nodes.

    Args:
    - g_batch: DGL batch of graphs with edge types (AST, CFG, etc.).
    - batch_idx: Index of the current batch.
    - node_importance_scores: Dictionary of node importance scores for the current graph.
    - save_dir: Directory to save the graph image (optional).
    """
    # Check if 'node_attributions' is in the dictionary
    if 'node_attributions' in node_importance_scores:
        attributions_list = node_importance_scores['node_attributions']
        
        # Unbatch the graph if g_batch contains multiple graphs
        graphs = dgl.unbatch(g_batch)
        g = graphs[batch_idx]  # Select the specific graph for this batch

        # Check if the index batch_idx is valid for attributions_list
        if batch_idx < len(attributions_list):
            node_scores = attributions_list[batch_idx]

            # Ensure the node_scores are in a proper format (list or tensor)
            node_scores = node_scores.tolist() if isinstance(node_scores, torch.Tensor) else node_scores

            # Create a NetworkX graph from DGL graph
            AST_graph = nx.DiGraph()  # DiGraph for directed AST
            
            # Retrieve edge labels
            edge_labels = g.edata['label'].tolist()
            
            # Only include AST edges in the visualization
            for i, (u, v) in enumerate(zip(g.edges()[0], g.edges()[1])):
                if edge_labels[i] == 0:  # Assuming 0 corresponds to AST edge in your mapping
                    AST_graph.add_edge(u.item(), v.item())

            # Add node attributes (e.g., 'type' or any other feature you want to visualize)
            node_types = g.ndata['type'].tolist()
            for node in AST_graph.nodes():
                AST_graph.nodes[node]['type'] = node_types[node]

            # Ensure the node scores are aligned with the node IDs
            node_ids = [node.item() for node in g.nodes()]
            node_scores_dict = dict(zip(node_ids, node_scores))

            # Extract scores in the order of the nodes in the graph
            aligned_scores = [node_scores_dict.get(node_id, 0) for node_id in node_ids]

            # Normalize node scores
            min_score = min(aligned_scores)
            max_score = max(aligned_scores)
            normalized_scores = [(score - min_score) / (max_score - min_score + 1e-6) for score in aligned_scores]
            
            # Ensure the lengths match
            if len(normalized_scores) != len(AST_graph.nodes()):
                print("Warning: The length of normalized_scores does not match the number of nodes in the graph.")
                # Adjust the size of normalized_scores or node_sizes to match the number of nodes
                normalized_scores = normalized_scores[:len(AST_graph.nodes())]  # Trim to match if necessary

            
            # Set node size and color based on normalized importance scores
            node_sizes = [score * 1000 for score in normalized_scores]  # Scale sizes for visibility
            node_colors = normalized_scores  # Directly use scores for color
            
            # Visualization
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(AST_graph, seed=42)  # Layout for graph visualization

            # Draw nodes with labels (node type)
            nx.draw(AST_graph, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.Reds, alpha=0.8)
            
            # Draw edge labels if needed
            nx.draw_networkx_edge_labels(AST_graph, pos)
            
            plt.title(f"AST Graph Visualization - Batch {batch_idx}")
            
            # Save the graph or show it
            if save_dir:
                plt.savefig(f"{save_dir}/AST_graph_batch_{batch_idx}.png")
            else:
                plt.show()
        else:
            print(f"No explanations for graph index {batch_idx}.")
    else:
        print("No node attributions found in the provided dictionary.")
    """
    Visualize the AST graph by filtering out non-AST edges and highlighting important nodes.

    Args:
    - g_batch: DGL batch of graphs with edge types (AST, CFG, etc.).
    - batch_idx: Index of the current batch.
    - node_importance_scores: List of node importance scores for the current graph.
    - save_dir: Directory to save the graph image (optional).
    """
    node_importance_scores = node_importance_scores['node_attributions']
    processed_scores = []
    # Unbatch the graph if g_batch contains multiple graphs
    graphs = dgl.unbatch(g_batch)
    g = graphs[batch_idx]  # Select the specific graph for this batch
    
    # Create a NetworkX graph from DGL graph
    AST_graph = nx.DiGraph()  # DiGraph for directed AST
    
    # Retrieve edge labels
    edge_labels = g.edata['label'].tolist()
    
    # Only include AST edges in the visualization
    for i, (u, v) in enumerate(zip(g.edges()[0], g.edges()[1])):
        if edge_labels[i] == 0:  # Assuming 0 corresponds to AST edge in your mapping
            AST_graph.add_edge(u.item(), v.item())

    # Add node attributes (e.g., 'type' or any other feature you want to visualize)
    node_types = g.ndata['type'].tolist()
    for node in AST_graph.nodes():
        AST_graph.nodes[node]['type'] = node_types[node]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(AST_graph, seed=42)  # Layout for graph visualization

    # Check the type of node importance scores
    print("Node Importance Scores:", node_importance_scores)
    print("Types of scores:", [type(score) for score in node_importance_scores])

    # Convert scores to floats if they are tensors or strings
    # node_importance_scores = [score.item() if isinstance(score, torch.Tensor) else float(score) for score in node_importance_scores]
    # Check type and handle it
    num_nodes = g_batch.number_of_nodes()
    processed_scores = [0.0] * num_nodes  # Initialize with zero scores

    if isinstance(node_importance_scores, torch.Tensor):
        # Ensure the tensor is float type
        node_importance_scores = node_importance_scores.float()
        # processed_scores = node_importance_scores.flatten().tolist()
        for i in range(num_nodes):
            processed_scores[i] = node_importance_scores[i].item() if i < len(node_importance_scores) else 0.0
    
    elif isinstance(node_importance_scores, list):
        # for score in node_importance_scores:
        #     if isinstance(score, torch.Tensor):
        #         # Convert to float if it's a tensor
        #         score = score.float()
        #         if score.numel() == 1:
        #             processed_scores.append(float(score.item()))
        #         else:
        #             # Average the multi-element tensor
        #             avg_score = score.mean().item()  # Get the average
        #             processed_scores.append(float(avg_score))
        #             print("Averaged multi-element tensor to:", avg_score)
        #     else:
        #         processed_scores.append(float(score))
        for i, score in enumerate(node_importance_scores):
            if isinstance(score, torch.Tensor):
                score = score.float()  # Convert to float tensor
                processed_scores[i] = score.mean().item() if i < len(processed_scores) else 0.0
            else:
                processed_scores[i] = float(score)

    else:
        raise ValueError("node_importance_scores must be a Tensor or a list of Tensors or numbers.")
    # Check if processed_scores is empty
    if not processed_scores:
        raise ValueError("node_importance_scores is empty after processing. Please check the input.")

    # Ensure the length matches the number of nodes in the graph
    if len(processed_scores) != num_nodes:
        print(f"Warning: Length of processed_scores ({len(processed_scores)}) does not match number of nodes ({num_nodes}).")
        # Optionally, fill the remaining scores with a default value (like 0.0)
        processed_scores.extend([0.0] * (num_nodes - len(processed_scores)))

    min_score = min(processed_scores)
    max_score = max(processed_scores)

    normalized_scores = [(score - min_score) / (max_score - min_score + 1e-6) for score in processed_scores]

    # Check if normalized_scores length matches the number of nodes
    if len(normalized_scores) != num_nodes:
        raise ValueError(f"Length of normalized_scores ({len(normalized_scores)}) does not match number of nodes ({num_nodes}).")

     # Visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(AST_graph, seed=42)  # Layout for graph visualization
    
    # Set node size and color based on normalized importance scores
    node_sizes = [score * 1000 for score in normalized_scores]  # Scale sizes for visibility
    node_colors = normalized_scores  # Directly use scores for color
    
    # Draw nodes with labels (node type)
    nx.draw(AST_graph, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.Reds, alpha=0.8)
    
    # Draw edge labels if needed
    nx.draw_networkx_edge_labels(AST_graph, pos)
    
    plt.title(f"AST Graph Visualization - Batch {batch_idx}")
    
    # Save the graph or show it
    if save_dir:
        plt.savefig(f"{save_dir}/AST_graph_batch_{batch_idx}.png")
    else:
        plt.show()

def visualize_graph_explanations(g_batch, node_attributions, batch_idx):
    # Check if 'node_attributions' is in the dictionary
    if 'node_attributions' in node_attributions:
        attributions_list = node_attributions['node_attributions']
        
        for i, g in enumerate(dgl.unbatch(g_batch)):
            # Check if the index i is valid for attributions_list
            if i < len(attributions_list):
                # Access the node scores for the current graph
                node_scores = attributions_list[i]

                # Create a NetworkX graph from the DGL graph
                G = nx.Graph()
                for node in g.nodes():
                    G.add_node(node.item())  # Ensure node is added as an integer
                for edge in g.edges():
                    G.add_edge(edge[0].item(), edge[1].item())

                # Print node scores and node IDs for debugging
                node_scores = node_scores.tolist() if isinstance(node_scores, torch.Tensor) else node_scores
                # print(f"Node scores for graph {i}: {node_scores}")

                # Debug: Print the corresponding node IDs in the graph
                node_ids = [node.item() for node in g.nodes()]
                # print(f"Node IDs for graph {i}: {node_ids}")

                # Ensure the node scores are aligned with the node IDs
                node_scores_dict = dict(zip(node_ids, node_scores))

                # Extract scores in the order of the nodes in the graph
                aligned_scores = [node_scores_dict.get(node_id, 0) for node_id in node_ids]

                # Normalize node scores
                min_score = min(aligned_scores)
                max_score = max(aligned_scores)
                normalized_scores = [(score - min_score) / (max_score - min_score + 1e-6) for score in aligned_scores]

                # Set node size and color based on normalized importance scores
                node_sizes = [score * 1000 for score in normalized_scores]
                node_colors = normalized_scores

                # Debugging output for lengths
                # print(f"Number of nodes in graph: {len(G.nodes())}")
                # print(f"Node sizes length: {len(node_sizes)}")
                # print(f"Node colors length: {len(node_colors)}")

                # Ensure that sizes and colors match the number of nodes
                if len(node_sizes) != len(G.nodes()) or len(node_colors) != len(G.nodes()):
                    # print(f"Mismatch in lengths for graph {i}:")
                    # print(f"Node sizes: {len(node_sizes)}, Node colors: {len(node_colors)}, Nodes in graph: {len(G.nodes())}")
                    continue  # Skip drawing if there's a mismatch

                # Draw the graph
                plt.figure(figsize=(8, 8))
                pos = nx.spring_layout(G)  # Layout for visual clarity
                nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Reds)
                # plt.title(f"Graph Node Importance for Batch {batch_idx}, Graph {i}")
                # plt.savefig(f"graph_explanation_batch_{batch_idx}_graph_{i}.png")
                plt.show()
            else:
                print(f"No explanations for graph index {i}.")
                continue  # Skip if there are no explanations for this graph
    else:
        print("No node attributions found in the provided dictionary.")

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


