import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Bidirectional, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

csv_path = 'IMDB reviews/IMDB Dataset.csv'

try:
    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded with {len(df)} rows")
except FileNotFoundError:
    print(f"File Not found at {csv_path}")
    exit()
    
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

df['review'] = df['review'].apply(clean_html)

sentiment_mapping = {'negative': 0, 'positive': 1}
df['sentiment_id'] = df['sentiment'].map(sentiment_mapping)

texts = df['review'].tolist()
labels = df['sentiment_id'].values

vocab_size = 10000
embedding_dim = 128
max_len = 256
oov_token = "<unk>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
print(f"Nombre de tokens uniques trouvés (incluant OOV): {len(word_index)}")
print(f"Vocabulaire effectif utilisé par le tokenizer: {min(vocab_size, len(word_index) + 1)}")

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTaille de l'ensemble d'entraînement (X, y): {X_train.shape}, {y_train.shape}")
print(f"Taille de l'ensemble de test (X, y): {X_test.shape}, {y_test.shape}")
print(f"Nombre de classes: {np.max(y_train) + 1}")

num_classes = np.max(y_train) + 1

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

epochs = 5
batch_size = 64

print("\nDébut de l'entraînement du modèle sur le dataset IMDB...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

print("\nEntraînement terminé.")


print("\nÉvaluation du modèle sur l'ensemble de test IMDB...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Perte sur l'ensemble de test: {loss:.4f}")
print(f"Précision sur l'ensemble de test: {accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nRapport de Classification:")
target_names = [sentiment for sentiment, id in sorted(sentiment_mapping.items(), key=lambda item: item[1])]
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nMatrice de Confusion:")
print(confusion_matrix(y_test, y_pred))

model_filename = 'sentiment_imdb_lstm_model.h5'
model.save(model_filename)
print(f"\nModèle enregistré sous: {model_filename}")