import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class CybersecurityTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=512, dropout=0.1):
        super(CybersecurityTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.positional_encoding(self.embedding(src))
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1), device=src.device)

        # Utilisation de l'encodeur pour générer la sortie. 
        # Note: Si le modèle a aussi un décodeur, il faudrait ajuster cela
        output = self.transformer.encoder(
            src, 
            mask=src_mask
        )
        return self.fc_out(output)
