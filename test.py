import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd # Utile pour re-créer le mapping des sentiments si non sauvegardé
import re # Pour le nettoyage des balises HTML


# --- 1. Configuration (ESSENTIEL : Doit correspondre à l'entraînement) ---

# Chemin vers votre modèle TensorFlow/Keras enregistré au format .h5
MODEL_PATH = 'sentiment_imdb_lstm_model.h5' # Assurez-vous que ce chemin est correct

# Paramètres du Tokenizer et des séquences
# CES VALEURS DOIVENT ÊTRE EXACTEMENT LES MÊMES QUE CELLES UTILISÉES LORS DE L'ENTRAÎNEMENT
VOCAB_SIZE = 10000  # Maximum de mots à considérer dans le vocabulaire
MAX_LEN = 256       # Longueur maximale des séquences de mots après padding
OOV_TOKEN = "<unk>" # Token pour les mots hors vocabulaire


# Mapping des sentiments numériques vers les labels textuels
# DOIT CORRESPONDRE À L'ORDRE UTILISÉ LORS DE L'ENTRAÎNEMENT (0 pour négatif, 1 pour positif)
sentiment_mapping = {'negative': 0, 'positive': 1}
# Pour l'affichage, nous inversons le mapping
id_to_sentiment = {id: sentiment for sentiment, id in sentiment_mapping.items()}


# --- 2. Fonctions de Prétraitement ---

def clean_html(text):
    """
    Supprime les balises HTML (<br /> par exemple) d'une chaîne de caractères.
    Cette fonction est appliquée car le dataset IMDB contient ces balises.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def preprocess_text(text, tokenizer_obj, max_len_val):
    """
    Prétraite un texte brut pour le rendre compatible avec l'entrée du modèle.
    Étapes : nettoyage HTML, tokenisation, et padding/troncation.
    """
    # Applique le nettoyage HTML
    cleaned_text = clean_html(text)
    
    # Convertit le texte nettoyé en séquence d'entiers via le tokenizer
    # Notez que texts_to_sequences attend une liste de textes, même pour un seul.
    sequence = tokenizer_obj.texts_to_sequences([cleaned_text])
    
    # Applique le padding/troncation pour obtenir une séquence de longueur fixe
    padded_sequence = pad_sequences(sequence, maxlen=max_len_val, padding='post', truncating='post')
    
    return padded_sequence


# --- 3. Chargement du Modèle et Initialisation du Tokenizer ---

print(f"Tentative de chargement du modèle depuis: {MODEL_PATH}...")
try:
    # Charge le modèle Keras sauvegardé (architecture, poids, configuration de compilation)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé avec succès.")
    # Affiche un résumé de l'architecture du modèle chargé (optionnel, pour vérification)
    # model.summary()
except Exception as e:
    print(f"ERREUR: Impossible de charger le modèle. Détails: {e}")
    print("Veuillez vérifier que le fichier .h5 existe bien au chemin spécifié et qu'il n'est pas corrompu.")
    print("Assurez-vous également que la version de TensorFlow utilisée pour charger est compatible avec celle utilisée pour sauvegarder.")
    exit() # Quitte le script si le modèle ne peut pas être chargé

print("Initialisation du tokenizer avec le vocabulaire d'entraînement...")
# RECOMMANDATION IMPORTANTE: La meilleure méthode est de sauvegarder l'objet tokenizer
# lui-même après l'entraînement (par exemple avec `pickle`) et de le charger ici.
# Si vous n'avez pas sauvegardé le tokenizer, cette partie tente de le "re-créer"
# en le refittant sur une partie des données d'origine pour qu'il ait le bon mapping mot-ID.
# Cette méthode est moins robuste si le tokenizer a des options complexes ou si
# le dataset d'origine est trop grand pour être rechargé ici.
try:
    # Charge juste le strict minimum du dataset IMDB pour ré-initialiser le tokenizer
    # (cela doit être les mêmes données que celles sur lesquelles il a été fitté)
    # Assurez-vous que le chemin vers le CSV est correct
    dummy_texts_for_tokenizer_init = pd.read_csv('IMDB reviews/IMDB Dataset.csv')['review'].apply(clean_html).tolist()
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(dummy_texts_for_tokenizer_init)
    print("Tokenizer initialisé avec le vocabulaire original.")
    # On peut vérifier la taille effective du vocabulaire du tokenizer
    print(f"Taille du vocabulaire du tokenizer (effective): {len(tokenizer.word_index)}")
except FileNotFoundError:
    print(f"ERREUR: Le fichier CSV 'IMDB reviews/IMDB Dataset.csv' n'a pas été trouvé.")
    print("Ce fichier est nécessaire pour initialiser le tokenizer si vous ne l'avez pas sauvegardé séparément.")
    print("Veuillez vérifier le chemin ou envisager de sauvegarder/charger le tokenizer.")
    exit()
except Exception as e:
    print(f"ERREUR: Impossible d'initialiser le tokenizer. Détails: {e}")
    print("Assurez-vous que les paramètres VOCAB_SIZE, OOV_TOKEN correspondent à l'entraînement.")
    exit()


# --- 4. Boucle d'Inférence via la Ligne de Commande (CMD) ---

print("\n--- Lancez l'analyse de sentiment ! ---")
print("Écrivez une revue de film et appuyez sur Entrée pour obtenir le sentiment.")
print("Tapez 'quitter' ou 'exit' (sans guillemets) pour arrêter.")

while True:
    # Demande à l'utilisateur d'entrer du texte dans la console
    user_input = input("\nVotre texte de revue (ou 'quitter'): ")
    
    # Vérifie si l'utilisateur veut quitter
    if user_input.lower() in ['quitter', 'exit']:
        print("Fin de l'analyse. Au revoir!")
        break
    
    # Gère les entrées vides
    if not user_input.strip():
        print("Veuillez entrer du texte pour l'analyse.")
        continue

    # Prétraitement du texte saisi par l'utilisateur
    processed_input = preprocess_text(user_input, tokenizer, MAX_LEN)
    
    # Effectue la prédiction avec le modèle chargé
    # verbose=0 supprime l'affichage de la barre de progression de predict
    predictions = model.predict(processed_input, verbose=0)
    
    # np.argmax retourne l'indice de la probabilité la plus élevée (0 pour négatif, 1 pour positif)
    predicted_class_id = np.argmax(predictions, axis=1)[0]
    
    # Récupère le nom du sentiment (ex: 'positive' ou 'negative')
    predicted_sentiment = id_to_sentiment[predicted_class_id]
    
    # Récupère les probabilités pour chaque classe (par exemple [0.1, 0.9])
    probabilities = predictions[0]

    # Affiche les résultats
    print(f"\n-> Sentiment prédit: {predicted_sentiment.upper()}")
    print(f"   Probabilités: Négatif={probabilities[0]:.4f}, Positif={probabilities[1]:.4f}")
    # Une petite astuce pour mettre en évidence la probabilité la plus haute
    if predicted_class_id == 0:
        print(f"   (Le modèle est {probabilities[0]*100:.2f}% certain que c'est Négatif)")
    else:
        print(f"   (Le modèle est {probabilities[1]*100:.2f}% certain que c'est Positif)")