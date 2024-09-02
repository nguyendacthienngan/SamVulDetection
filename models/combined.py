import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model=None):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
    
    def forward(self, sequence_inputs, attention_mask, graph_inputs=None):
        # Get logits from CLR model (sequence data)
        sequence_outputs = self.clr_model(sequence_inputs, attention_mask)
        sequence_logits = sequence_outputs.logits
        # Handle case where graph inputs might be None
        if graph_inputs is not None:
            graph_outputs = self.devign_model(graph_inputs)
            # Aggregate graph outputs to match batch size (e.g., mean pooling)
            if isinstance(graph_outputs, tuple):
                graph_outputs = graph_outputs[0]  # Extract the tensor from the tuple

            graph_logits = graph_outputs.mean(dim=0, keepdim=True)
        else:
            # If no graph input, return zeros or a placeholder tensor
            graph_logits = torch.zeros(sequence_logits.shape[0], 1).to(sequence_logits.device)

        # Concatenate or combine logits
        combined_logits = torch.cat((sequence_logits, graph_logits.expand(sequence_logits.shape[0], -1)), dim=1)
        return combined_logits


# class CombinedModel(nn.Module):
#     def __init__(self, clr_model, devign_model):
#         super(CombinedModel, self).__init__()
#         self.clr_model = clr_model
#         self.devign_model = devign_model
#         print(f'clr_model.config.hidden_size: {clr_model.config.hidden_size}')
#         print(f'devign_model.concat_dim: {devign_model.concat_dim}')
#         self.fc = None  # Khởi tạo mà không có lớp FC


#     def forward(self, sequence_inputs, attention_mask=None, graph_inputs=None):
#         # Get sequence outputs from CLRModel
#         sequence_outputs = self.clr_model(sequence_inputs, attention_mask=attention_mask)
#         hidden_size = self.clr_model.config.hidden_size
#         sequence_logits = sequence_outputs.logits
#         graph_logits = []
#         if graph_inputs is not None:
#             graph_logits = self.devign_model(graph_inputs)
#             devign_output = graph_logits[0]
#             print(f'graph_logits[0] shape: {devign_output.shape}')

#             if len(graph_logits) > 0:
#                 if devign_output.dim() == 1:
#                     print(f'devign_output.dim() == 1')
#                     devign_output = devign_output.unsqueeze(1)
#                 if sequence_logits.dim() == 1:
#                     print(f'sequence_logits.dim() == 1')
#                     sequence_logits = sequence_logits.unsqueeze(1)
#                 # graph_logits = torch.cat(graph_logits, dim=0) if isinstance(graph_logits, list) else graph_logits
#                 # combined_logits = torch.cat((sequence_logits, graph_logits), dim=1)
#                 print(f'sequence_logits shape: {sequence_logits.shape}')
#                 print(f'devign_output shape: {devign_output.shape}')

#                 combined_logits = torch.cat((sequence_logits, devign_output), dim=1)
#             else:
#                 combined_logits = sequence_logits
#         else:
#             combined_logits = sequence_logits
        
#         # Cập nhật lớp FC nếu chưa có hoặc kích thước thay đổi
#         in_features = combined_logits.shape[1]
#         if self.fc is None or self.fc.in_features != in_features:
#             self.fc = nn.Linear(in_features, 1)

#         return self.fc(combined_logits)