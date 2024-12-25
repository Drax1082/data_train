import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CybersecurityTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, num_languages, dim_feedforward=512, dropout=0.1):
        super(CybersecurityTransformer, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_languages)
        ])
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
                num_encoder_layers
            ) for _ in range(num_languages)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, language_idx=0):
        # Appliquer l'embedding correspondant au langage
        x = self.embeddings[language_idx](x)
        
        # Ajouter l'encodage positionnel
        x = self.positional_encoding(x)
        
        # Passer par l'encodeur du transformer spécifique au langage
        x = x.permute(1, 0, 2)  # Permutation pour correspondre à la forme attendue par le Transformer
        padding_mask = (x == 0).squeeze(-1)  # Masque pour le padding, supposant que 0 est l'indice de <PAD>
        x = self.transformer_encoders[language_idx](x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)  # Remettre dans la forme originale
        
        # Appliquer la couche fully-connected
        return self.fc(x.mean(dim=1))  # Moyenne sur la séquence, puis passage par la couche linéaire

# Exemple d'utilisation
if __name__ == "__main__":
    vocab_size = 100  # À ajuster selon votre vocabulaire
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3  # Non utilisé dans cet exemple, mais disponible pour une extension
    num_languages = 9  # Nombre de langages supportés (Python, C, C++, Java, HTML, CSS, Bash, PHP, Ruby)

    model = CybersecurityTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, num_languages)
    print(model)  # Affiche la structure du modèle pour vérification

    # Exemple de données (indices fictifs pour Python)
    src = torch.randint(1, vocab_size, (1, 32))  # Exemple de séquence d'entrée pour Python
    output = model(src, language_idx=0)  # Index 0 pour Python
    print(output.shape)  # Devrait être [1, vocab_size]