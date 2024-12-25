import os
import torch
import torch.nn as nn
import re
import logging
from model import CybersecurityTransformer
import torch.nn.functional as F

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Nettoyage de l'entrée utilisateur pour éliminer les caractères spéciaux
def clean_input(text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s\(\)\[\]\{\}\=\+\-\*\/\.\<\>\:\;\,\?\!]", "", text).strip()
    logger.info(f"Entrée nettoyée: {cleaned}")
    return cleaned

# Génération de réponse basée sur un modèle Transformer
class Chatbot:
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        self.tokenizer = self.create_tokenizer()
        self.reverse_tokenizer = {v: k for k, v in self.tokenizer.items()}  # Inversion du tokenizer
        self.model = CybersecurityTransformer(vocab_size, d_model, num_heads, num_layers, num_layers)
        self.history = []

        # Charger le modèle pré-entraîné
        model_path = "trained_model.pth"
        if os.path.exists(model_path):
            self.load_trained_model(model_path)
        else:
            logger.warning(f"Aucun modèle trouvé à {model_path}. Le modèle sera initialisé aléatoirement.")

    # Tokenizer basique pour transformer les phrases en indices
    def create_tokenizer(self):
        vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + \
                list("abcdefghijklmnopqrstuvwxyz") + \
                [" ", ",", ".", "!", "?"]
        return {word: idx for idx, word in enumerate(vocab)}

    # Charger le modèle pré-entraîné
    def load_trained_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()  # Passer le modèle en mode évaluation
            logger.info("Modèle chargé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")

    # Convertir un texte en indices
    def text_to_indices(self, text):
        tokens = list(text.lower())  # Transformer le texte en liste de caractères
        indices = [self.tokenizer.get(token, self.tokenizer["<UNK>"]) for token in tokens]
        logger.info(f"Texte tokenisé: {tokens}")
        logger.info(f"Indices: {indices}")
        return indices

    # Convertir les indices en texte
    def indices_to_text(self, indices):
        decoded_text = []
        for idx in indices:
            word = self.reverse_tokenizer.get(idx, "<UNK>")
            if word not in ["<PAD>", "<SOS>", "<EOS>"]:  # Exclure les tokens spéciaux
                decoded_text.append(word)
        decoded_text = "".join(decoded_text)
        logger.info(f"Texte décodé: {decoded_text}")
        return decoded_text

    # Générer une réponse à partir de l'entrée de l'utilisateur
    def generate_reply(self, user_input, k=40):
        input_indices = [self.tokenizer["<SOS>"]] + self.text_to_indices(user_input) + [self.tokenizer["<EOS>"]]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)
        logger.info(f"Entrée convertie en tensor: {input_tensor}")

        with torch.no_grad():
            output = self.model(input_tensor)

        max_length = 50
        generated_sequence = [self.tokenizer["<SOS>"]]
        for _ in range(max_length):
            last_token = torch.tensor([generated_sequence[-1]]).unsqueeze(0)
            with torch.no_grad():
                next_token_logits = self.model(last_token)
            next_prob = F.softmax(next_token_logits[0, -1], dim=-1)
            
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(next_prob, k=k)
            next_token = torch.multinomial(top_k_probs, 1).item()
            next_token = top_k_indices[next_token].item()

            if next_token == self.tokenizer["<EOS>"]:  
                break
            generated_sequence.append(next_token)
        
        logger.info(f"Indices prédits: {generated_sequence}")
        return self.indices_to_text(generated_sequence[1:])  # Ignorer <SOS>

    # Réponse avec gestion de l'historique
    def get_response(self, user_input):
        user_input = clean_input(user_input)  # Nettoyer l'entrée
        self.history.append(f"User: {user_input}")

        # Limiter l'historique à 5 derniers échanges
        context = " ".join(self.history[-5:])
        logger.info(f"Contexte utilisé pour la génération: {context}")

        # Générer la réponse
        response = self.generate_reply(context)
        self.history.append(f"Bot: {response}")
        return response

# Fonction pour lancer le chat
def start_chat():
    vocab_size = 2000  # Ajusté à la taille du vocabulaire du modèle sauvegardé
    d_model = 128  # Dimension du modèle
    num_heads = 4  # Nombre de têtes dans l'attention
    num_layers = 3  # Nombre de couches du transformer

    chatbot = Chatbot(vocab_size, d_model, num_heads, num_layers)

    print("Chatbot prêt. Tapez 'exit' pour quitter.")

    while True:
        # Demander l'entrée de l'utilisateur
        user_input = input("Vous: ")

        # Quitter si l'utilisateur tape 'exit'
        if user_input.lower() == "exit":
            print("Fin de la conversation.")
            break

        # Obtenir la réponse du chatbot
        response = chatbot.get_response(user_input)

        # Afficher la réponse du chatbot
        print(f"Bot: {response}")

# Lancer le chatbot
if __name__ == "__main__":
    start_chat()
