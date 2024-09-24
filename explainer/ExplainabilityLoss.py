import torch
import torch.nn as nn
import os
from dig.xgraph.method.deeplift import DeepLIFT
from transformers_interpret import SequenceClassificationExplainer
from transformers import RobertaTokenizer
from torch.optim.lr_scheduler import StepLR

# Sparsity Loss
def compute_explainability_loss(explanations_sequence, edge_masks, labels, sparsity=0.7):
    """
    Encourage sparse explanations where only a small fraction of input features (sparsity) have non-zero attributions.
    """
    # Define how many elements should be non-zero based on sparsity
    sequence_nonzero_threshold = int(explanations_sequence.numel() * sparsity)
    if edge_masks is not None:
        graph_nonzero_threshold = int(edge_masks.numel() * sparsity)
    else:
        graph_nonzero_threshold = 0

    # Penalize explanations where more elements than the threshold are non-zero
    sequence_nonzero_count = (explanations_sequence.abs() > 0).sum().item()
    graph_nonzero_count = (edge_masks.abs() > 0).sum().item() if edge_masks is not None else 0

    sequence_loss = max(0, sequence_nonzero_count - sequence_nonzero_threshold)
    graph_loss = max(0, graph_nonzero_count - graph_nonzero_threshold)

    explainability_loss = sequence_loss + graph_loss
    return explainability_loss


def train(args, device, train_loader, model, optimizer, loss_function):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Initialize the explainability model
    graph_explainer = DeepLIFT(model=model, explain_graph=False)
    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)

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

            optimizer.zero_grad()

            # Calculate loss
            classification_loss  = loss_function(outputs, labels)

            # Explanation for sequence data using SequenceClassificationExplainer
            explanations_sequence = cls_explainer(sequence_inputs)

            # Explanation for graph data using Deeplift if graph_inputs are available
            if graph_inputs is not None:
                g_batch = batch['graph_features']
                x = g_batch.ndata['type']  # Node features
                edge_index = g_batch.edges()  # Edge indices
                edge_masks, hard_edge_masks, related_preds = graph_explainer(
                    x=x, edge_index=edge_index, node_idx=None, num_classes=2, sparsity=0.7
                )
            else:
                edge_masks = None  # No graph data

            # Compute the explainability loss based on attributions
            explainability_loss = compute_explainability_loss(explanations_sequence, edge_masks, labels)

            # Total loss: classification loss + explainability loss
            total_loss_combined = classification_loss + explainability_loss

            # Backpropagation and optimization
            total_loss_combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            total_loss += total_loss_combined.item()

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
        # val_loss = evaluate(args, device, val_loader, model, loss_function)
        # print(f"Epoch {epoch + 1}/{args['epoch']}, Validation Loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"Epoch {epoch + 1}: Learning rate: {scheduler.get_last_lr()}")

        # Optionally save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")


