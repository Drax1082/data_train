import os
import torch
import torch.nn as nn
import re
import logging
from model import CybersecurityTransformer  # Import du modèle modifié

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Nettoyage de l'entrée utilisateur pour éliminer les caractères spéciaux
def clean_input(text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s\(\)\[\]\{\}\=\+\-\*\/\.\<\>\:\;\,\?\!]", "", text).strip()
    logger.info(f"Entrée nettoyée: {cleaned}")
    return cleaned

# Génération de réponse basée sur un modèle Transformer adapté
class Chatbot:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_languages):
        self.tokenizers = self.create_tokenizers(num_languages)  # Création de tokenizers pour chaque langage
        self.reverse_tokenizers = [{v: k for k, v in tokenizer.items()} for tokenizer in self.tokenizers]
        self.model = CybersecurityTransformer(vocab_size, d_model, num_heads, num_layers, num_languages)
        self.history = []
        self.language_index = self.get_language_index()

        # Charger le modèle pré-entraîné
        model_path = "trained_model.pth"
        if os.path.exists(model_path):
            self.load_trained_model(model_path)
        else:
            logger.warning(f"Aucun modèle trouvé à {model_path}. Le modèle sera initialisé aléatoirement.")

    # Création de tokenizers pour chaque langage
    def create_tokenizers(self, num_languages):
        tokenizers = []
        for _ in range(num_languages):
            vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + \
                    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + \
                    [" ", "\n", "\t", "+", "-", "*", "/", "=", "(", ")", "[", "]", "{", "}", "<", ">", ".", ",", ";", ":", "?", "!"]
            tokenizers.append({word: idx for idx, word in enumerate(vocab)})
        return tokenizers

    def get_language_index(self, language="python"):  # Par défaut, Python
        languages = {"python": 0, "c": 1, "cpp": 2, "java": 3, "html": 4, "css": 5, "bash": 6, "php": 7, "ruby": 8}
        return languages.get(language.lower(), 0)  # Retourne l'index de Python par défaut si le langage n'est pas reconnu

    # Charger le modèle pré-entraîné
    def load_trained_model(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))  # Chargement du modèle
            self.model.eval()  # Passer le modèle en mode évaluation
            logger.info("Modèle chargé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")

    # Convertir un texte en indices pour un langage donné
    def text_to_indices(self, text, language_idx=0):
        tokenizer = self.tokenizers[language_idx]
        tokens = list(text)
        indices = [tokenizer.get(token, tokenizer["<UNK>"]) for token in tokens]
        logger.info(f"Texte tokenisé: {tokens}")
        logger.info(f"Indices: {indices}")
        return indices

    # Convertir les indices en texte pour un langage donné
    def indices_to_text(self, indices, language_idx=0):
        reverse_tokenizer = self.reverse_tokenizers[language_idx]
        decoded_text = []
        for idx in indices:
            word = reverse_tokenizer.get(idx, "<UNK>")
            if word not in ["<PAD>", "<SOS>", "<EOS>"]:  # Exclure les tokens spéciaux
                decoded_text.append(word)
        decoded_text = "".join(decoded_text)
        logger.info(f"Texte décodé: {decoded_text}")
        return decoded_text

    # Générer une réponse à partir de l'entrée de l'utilisateur
    def generate_reply(self, user_input, language):
        language_idx = self.get_language_index(language)
        # Ajouter des tokens spéciaux pour le modèle
        input_indices = [self.tokenizers[language_idx]["<SOS>"]] + self.text_to_indices(user_input, language_idx) + [self.tokenizers[language_idx]["<EOS>"]]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)  # Créer un tenseur 2D
        logger.info(f"Entrée convertie en tensor: {input_tensor}")

        # Passer l'entrée dans le modèle
        with torch.no_grad():
            output = self.model(input_tensor, language_idx)
        logger.info(f"Sortie brute du modèle: {output.shape}")

        # Prendre la sortie du modèle, choisir le mot avec la probabilité la plus élevée
        predicted_indices = torch.argmax(output, dim=-1).squeeze(0).tolist()
        logger.info(f"Indices prédits: {predicted_indices}")

        # Convertir les indices en texte
        return self.indices_to_text(predicted_indices, language_idx)

    # Réponse avec gestion de l'historique
    def get_response(self, user_input, language="python"):
        user_input = clean_input(user_input)  # Nettoyer l'entrée
        self.history.append(f"User: {user_input}")

        # Limiter l'historique à 5 derniers échanges
        context = " ".join(self.history[-5:])
        logger.info(f"Contexte utilisé pour la génération: {context}")

        # Générer la réponse
        response = self.generate_reply(context, language)
        self.history.append(f"Bot: {response}")
        return response

# Fonction pour lancer le chat
def start_chat():
    vocab_size = 100  # Taille du vocabulaire (à ajuster selon votre tokenization)
    d_model = 128  # Dimension du modèle
    num_heads = 4  # Nombre de têtes dans l'attention
    num_layers = 3  # Nombre de couches du transformer
    num_languages = 9  # Nombre de langages supportés

    chatbot = Chatbot(vocab_size, d_model, num_heads, num_layers, num_languages)

    print("Chatbot prêt. Tapez 'exit' pour quitter. Indiquez le langage entre crochets [langage] pour changer de contexte de code.")

    while True:
        # Demander l'entrée de l'utilisateur
        user_input = input("Vous: ")
        
        # Vérifier si l'utilisateur veut changer de langage
        if user_input.startswith('[') and user_input.endswith(']'):
            language = user_input[1:-1]
            print(f"Langage changé à {language}")
            continue

        # Quitter si l'utilisateur tape 'exit'
        if user_input.lower() == "exit":
            print("Fin de la conversation.")
            break

        # Obtenir la réponse du chatbot
        response = chatbot.get_response(user_input, language="python")

        # Afficher la réponse du chatbot
        print(f"Bot: {response}")

# Lancer le chatbot
if __name__ == "__main__":
    start_chat()