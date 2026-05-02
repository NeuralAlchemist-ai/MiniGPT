import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        self.activation = config.build_activation()
    
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x @ self.w2

        return x
