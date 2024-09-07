import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model=None):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
        # Since you have binary classification (0 or 1), the output size should be 2 (for 2 classes)
        self.classifier = nn.Linear(2, 2)  # Output 2 logits for binary classification
    
    def forward(self, sequence_inputs, attention_mask, graph_inputs=None):
        # Get logits from CLR model (sequence data)
        sequence_logits = self.clr_model(sequence_inputs, attention_mask)
        print(f"Sequence logits shape: {sequence_logits.shape}")  # Debugging shape
        
        # Ensure sequence_logits is of shape [batch_size, 1]
        if sequence_logits.dim() > 2:
            sequence_logits = sequence_logits.mean(dim=1, keepdim=True)
        
        # Handle case where graph inputs might be None
        if graph_inputs is not None:
            graph_outputs = self.devign_model(graph_inputs)
            
            # If graph_outputs is a tuple, extract the first element (the actual logits or output tensor)
            if isinstance(graph_outputs, tuple):
                graph_outputs = graph_outputs[0]
            
            print(f"Graph outputs shape before pooling: {graph_outputs.shape}")  # Debugging shape
            
            # Ensure graph_outputs is at least 2D
            if graph_outputs.dim() == 1:
                graph_outputs = graph_outputs.unsqueeze(dim=1)  # Add an extra dimension
            
            # Ensure graph_outputs is of shape [batch_size, 1]
            if graph_outputs.dim() > 2:
                graph_outputs = graph_outputs.mean(dim=1, keepdim=True)
            
            # Adjust graph_outputs size to match the batch size of sequence_logits
            if graph_outputs.shape[0] != sequence_logits.shape[0]:
                padding_size = sequence_logits.shape[0] - graph_outputs.shape[0]
                if padding_size > 0:
                    # Pad graph_outputs with zeros to match the sequence_logits batch size
                    graph_outputs = torch.cat([graph_outputs, torch.zeros(padding_size, 1).to(graph_outputs.device)], dim=0)
                else:
                    # In case graph_outputs has more samples, truncate it
                    graph_outputs = graph_outputs[:sequence_logits.shape[0], :]
            
            print(f"Graph outputs shape after adjustment: {graph_outputs.shape}")  # Debugging shape
        else:
            graph_outputs = torch.zeros(sequence_logits.shape).to(sequence_logits.device)
            print(f"Graph outputs (zeros) shape: {graph_outputs.shape}")  # Debugging shape

        # Concatenate or combine logits
        combined_logits = torch.cat((sequence_logits, graph_outputs), dim=1)  # Shape: [batch_size, 2]
        print(f"Combined logits shape: {combined_logits.shape}")  # Debugging shape
        
        final_output = self.classifier(combined_logits)  # Shape: [batch_size, 2]
        return final_output
