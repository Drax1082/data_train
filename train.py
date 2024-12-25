import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import string
import json
from torch.amp import GradScaler

# --- Dataset personnalisé pour cybersécurité ---
class CybersecurityDataset(Dataset):
    def __init__(self, data_path, max_length=128, split_ratio=0.8):
        self.train_data, self.valid_data = self.prepare_data(data_path, max_length, split_ratio)
        self.vocab = self.create_vocab()
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Max index in vocab: {max(self.vocab.values())}")

    def prepare_data(self, data_path, max_length, split_ratio):
        with open(data_path, 'r') as file:
            lines = file.readlines()
        
        cleaned_data = []
        for line in lines:
            cleaned_line = self.clean_line(line, max_length)
            if cleaned_line:  # Ignorer les lignes vides après nettoyage
                cleaned_data.append(cleaned_line)
        
        random.shuffle(cleaned_data)
        split_index = int(len(cleaned_data) * split_ratio)
        return cleaned_data[:split_index], cleaned_data[split_index:]

    def clean_line(self, line, max_length):
        line = ' '.join(line.split())
        line = line[:max_length]
        return f"<SOS> {line} <EOS>" if line else None

    def create_vocab(self):
        vocab = set()
        for line in self.train_data:  # Utiliser train_data pour le vocabulaire
            vocab.update(line.split())
        return {word: idx for idx, word in enumerate(['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(vocab))}

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        line = self.train_data[idx]
        src = [self.vocab.get(word, self.vocab['<UNK>']) for word in line.split()]
        src = [self.vocab['<SOS>']] + src + [self.vocab['<EOS>']]
        trg = src[1:] + [self.vocab['<PAD>']]  # Décalage pour la cible
        return torch.tensor(src), torch.tensor(trg)

# --- Fonction de Batching Dynamique ---
def dynamic_batching(batch):
    src_batch, trg_batch = zip(*batch)
    padded_src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    padded_trg = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0)
    print(f"Padded src shape: {padded_src.shape}, Padded trg shape: {padded_trg.shape}")
    print(f"Max index in src: {padded_src.max().item()}, Max index in trg: {padded_trg.max().item()}")
    return padded_src, padded_trg

def decode_output(output_indices, vocab):
    return ' '.join([list(vocab.keys())[list(vocab.values()).index(idx)] if idx in vocab.values() else '<UNK>' for idx in output_indices if idx != 0])

# --- Modèle Transformer ---
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

    def forward(self, src, trg):
        print(f"Source shape: {src.shape}, Target shape: {trg.shape}")
        print(f"Max src index: {src.max().item()}, Max trg index: {trg.max().item()}")
        if src.max().item() >= self.embedding.num_embeddings or trg.max().item() >= self.embedding.num_embeddings:
            raise ValueError("Index out of range in input data")
        
        src = self.positional_encoding(self.embedding(src))
        trg = self.positional_encoding(self.embedding(trg))

        # Assurez-vous que les masques de padding sont bien de dimension 2 pour des données en batch
        src_key_padding_mask = (src == 0).any(dim=-1)  # Cela devrait donner un bool tensor de shape [batch_size, seq_len]
        tgt_key_padding_mask = (trg == 0).any(dim=-1)  # Cela devrait donner un bool tensor de shape [batch_size, seq_len]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg.size(1), device=src.device)

        output = self.transformer(
            src, trg,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.fc_out(output)

# --- Paramètres ---
data_path = "C:/Users/enzo0/Desktop/ia/data_python.txt"  # Remplacer par le chemin réel
vocab_size = 2000  # Ajusté selon les indices max observés dans votre dataset
max_length = 128  # Longueur max des séquences
d_model = 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
batch_size = 128  # Augmenté pour RTX 4080
num_epochs = 100
learning_rate = 0.001

# Dataset et DataLoader
dataset = CybersecurityDataset(data_path, max_length)
train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dynamic_batching, pin_memory=True)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Initialisation du modèle
model = CybersecurityTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
print(f"Model Embedding size: {model.embedding.num_embeddings}")
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD>
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Précision mixte pour l'entraînement
scaler = GradScaler()

# Charger un modèle existant ou initialiser un nouveau
model_path = "trained_model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(checkpoint, strict=False)
        print("Modèle chargé avec succès, certaines dimensions peuvent ne pas correspondre.")
    except RuntimeError:
        print("Impossible de charger le modèle. Entraînement à partir de zéro.")
else:
    print("Aucun modèle trouvé. Entraînement à partir de zéro.")

# --- Boucle d'entraînement ---
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]

        optimizer.zero_grad()
        try:
            with torch.amp.autocast('cuda'):
                output = model(src, trg_input)
                output = output.reshape(-1, vocab_size)  # (batch * seq_length, vocab_size)
                trg_output = trg_output.reshape(-1)     # (batch * seq_length)
                loss = criterion(output, trg_output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
        except ValueError as e:
            print(f"Error during training: {e}")
            continue

    print(f"Époque {epoch+1}, Perte moyenne : {epoch_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        sample, _ = dataset[0]  # Exemple de la première séquence
        sample = sample.unsqueeze(0).to(device)
        trg_input = torch.zeros_like(sample).to(device)  # Commence avec une séquence vide
        try:
            with torch.cuda.amp.autocast():
                output = model(sample, trg_input)
            output_indices = output.argmax(dim=2)[0].cpu().numpy()  # Indices de la prédiction
            decoded_output = decode_output(output_indices, dataset.vocab)
            print(f"Exemple de prédiction : {decoded_output}")
        except ValueError as e:
            print(f"Error during evaluation: {e}")

    scheduler.step(epoch_loss / len(train_loader))
    torch.cuda.empty_cache()  # Libérer la mémoire GPU

# Sauvegarder le modèle
torch.save(model.state_dict(), model_path)  # Sauvegarde uniquement les poids avec cette méthode
print("Modèle sauvegardé dans", model_path)

# Sauvegarder le dataset nettoyé pour une utilisation future