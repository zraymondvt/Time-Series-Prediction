import torch
import torch.nn as nn

class HybridTransformerESN(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, ff_dim, reservoir_size, spectral_radius, sparsity, dropout=0.1):
        super(HybridTransformerESN, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.reservoir_size = reservoir_size
        self.input_weights = nn.Parameter(torch.randn(reservoir_size, d_model) * 0.1)

        self.reservoir_weights = nn.Parameter(torch.empty(reservoir_size, reservoir_size))
        nn.init.orthogonal_(self.reservoir_weights)
        self.reservoir_weights.data *= spectral_radius / torch.linalg.norm(self.reservoir_weights, ord=2)

        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        self.reservoir_weights.data[mask] = 0

        self.readout = nn.Linear(reservoir_size, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)

        batch_size, seq_len, d_model = x.shape
        reservoir_state = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        for t in range(seq_len):
            u_t = x[:, t, :]
            reservoir_state = torch.tanh(
                torch.matmul(u_t, self.input_weights.T) + torch.matmul(reservoir_state, self.reservoir_weights.T)
            )

        output = self.readout(reservoir_state)
        return output
