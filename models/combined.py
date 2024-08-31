import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
        self.fc = nn.Linear(clr_model.config.hidden_size + devign_model.output_dim, 1)  # Điều chỉnh kích thước đầu ra nếu cần (num_classes=1)

    def forward(self, sequence_inputs, attention_mask=None, graph_inputs=None):
        # Get sequence outputs from CLRModel
        sequence_outputs = self.clr_model(sequence_inputs, attention_mask=attention_mask)
        sequence_logits = sequence_outputs.logits
        graph_logits = []
        if graph_inputs is not None:
            if isinstance(graph_inputs, list):
                # Handle the case where graph_inputs is a list of DGLGraph objects
                for g in graph_inputs:
                    if g.numel() > 0:
                        graph_logits.append(self.devign_model(g))
            else:
                # Handle the case where graph_inputs is a batch of DGLGraphs or a single DGLGraph
                graph_logits = self.devign_model(graph_inputs)
                graph_logits = graph_logits.unsqueeze(0)  # Ensure it's the correct shape for concatenation
            
            if len(graph_logits) > 0:
                graph_logits = torch.cat(graph_logits, dim=0) if isinstance(graph_logits, list) else graph_logits
                combined_logits = torch.cat((sequence_logits, graph_logits), dim=1)
            else:
                combined_logits = sequence_logits
        else:
            combined_logits = sequence_logits
        # Pass the combined logits through the final fully connected layer
        return self.fc(combined_logits)

        # if graph_inputs is not None:  # Ensure graph_inputs is not None
        #     for g in graph_inputs:
        #         if g.numel() > 0:
        #             graph_logits.append(self.devign_model(g))
            
        #     if len(graph_logits) > 0:
        #         graph_logits = torch.cat(graph_logits, dim=0)
        #         # graph_logits = self.devign_model(graph_inputs)
        #         combined_logits = torch.cat((sequence_logits, graph_logits), dim=1)  # Combine the outputs
        #     else:
        #         combined_logits = sequence_logits
        # else:
        #     combined_logits = sequence_logits
        
        # return combined_logits
    
