import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class CLRModel(nn.Module):
    def __init__(self, num_labels):
        super(CLRModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        self.config = self.model.config 

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_logits = outputs.logits  # [batch_size, num_labels]

        # Reduce the logits to a single value per sample (e.g., by averaging or taking the max)
        sequence_logits = sequence_logits.mean(dim=1, keepdim=True)  # Now [batch_size, 1]
        
        return sequence_logits