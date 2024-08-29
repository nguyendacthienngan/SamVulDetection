import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
        self.fc = nn.Linear(clr_model.config.hidden_size + devign_model.output_dim, 1)  # Điều chỉnh kích thước đầu ra nếu cần (num_classes=1)

    def forward(self, sequence_inputs, graph_inputs, attention_mask=None, labels=None):
        # Get sequence outputs from CLRModel
        sequence_logits = self.clr_model(sequence_inputs, attention_mask=attention_mask)
        graph_logits = []
        if graph_inputs.size(1) > 0:  # Ensure graph_inputs is not empty
            for g in graph_inputs:
                if g.numel() > 0:
                    graph_logits.append(self.devign_model(g))
            
            if len(graph_logits) > 0:
                graph_logits = torch.cat(graph_logits, dim=0)
                # graph_logits = self.devign_model(graph_inputs)
                combined_logits = torch.cat((sequence_logits, graph_logits), dim=1)  # Combine the outputs
            else:
                combined_logits = sequence_logits
        else:
            combined_logits = sequence_logits
        
        return combined_logits
    
