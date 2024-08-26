from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch

def evaluate_model(model, data_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for seq_batch, graph_batch, label_batch in data_loader:
            # Move data to the specified device (GPU/CPU)
            seq_batch = seq_batch.to(device)
            graph_batch = graph_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass to get model predictions
            outputs, _, _ = model(seq_batch, graph_batch)

            # Apply sigmoid to get probabilities (assuming binary classification)
            probs = torch.sigmoid(outputs).squeeze()

            # Binarize the probabilities to get predictions (0 or 1)
            preds = (probs > 0.5).float()

            # Append the results for later evaluation
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC Score: {auc:.4f}')

    return accuracy, precision, recall, f1, auc
