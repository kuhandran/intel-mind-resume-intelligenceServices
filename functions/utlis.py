from torch import nn

class CustomModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Forward pass through the original model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract only the start_logits and end_logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Return only the required tensors
        return start_logits, end_logits