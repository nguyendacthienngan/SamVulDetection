import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

class CLRModel(nn.Module):
    def __init__(self, config):
        super(CLRModel, self).__init__()
        self.config = config
        self.model = RobertaModel.from_pretrained(config.model_name_or_path)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss, logits
        return logits
