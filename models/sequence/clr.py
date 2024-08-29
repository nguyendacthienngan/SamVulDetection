import torch
import torch.nn as nn
# from transformers import RobertaModel, RobertaConfig
# from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers import AutoModelForSequenceClassification

class CLRModel(nn.Module):
    def __init__(self, num_labels):
        super(CLRModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        self.config = self.model.config 

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask)
        # outputs = self.model(input_ids, attention_mask=attention_mask)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)
        
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     return loss, logits
        # return logits
