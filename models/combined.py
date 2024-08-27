import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
        self.fc = nn.Linear(clr_model.config.hidden_size + devign_model.output_size, 1)  # Điều chỉnh kích thước đầu ra nếu cần (num_classes=1)

    def forward(self, sequence_inputs, graph_inputs, attention_mask=None, labels=None):
        # Get sequence outputs from CLRModel
        sequence_logits = self.clr_model(sequence_inputs, attention_mask=attention_mask)
        
        # Get graph outputs from DevignModel
        graph_logits = self.devign_model(graph_inputs)
        
        # Combine the outputs
        combined_output = torch.cat((sequence_logits, graph_logits), dim=1)
        logits = self.fc(combined_output)
        
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        #     return loss, logits
        return logits
