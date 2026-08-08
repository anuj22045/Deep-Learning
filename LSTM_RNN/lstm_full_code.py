##its output prediction is not good because i use very small dataset, you can use (IMDB Dataset of 50K Movie Reviews) dataset 
# from kaggle 

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences


# =========================================================
# 1. DATASET
# =========================================================

sentences = [
    # Positive
    "I love this movie",
    "This movie is amazing",
    "The film was excellent",
    "I enjoyed the movie",
    "This was a wonderful film",
    "The movie was fantastic",
    "I really liked this film",
    "The story was beautiful",
    "This film was great",
    "I enjoyed watching this movie",

    # Negative
    "I hate this movie",
    "This movie is terrible",
    "The film was horrible",
    "I disliked the movie",
    "This was a boring film",
    "The movie was awful",
    "I really disliked this film",
    "The story was disappointing",
    "This film was bad",
    "I hated watching this movie"
]

labels = np.array([
    # Positive
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    # Negative
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
])


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    sentences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# 3. TOKENIZER
# =========================================================

tokenizer = Tokenizer(
    num_words=100,
    oov_token="<OOV>"
)

# IMPORTANT:
# Fit tokenizer ONLY on training data

tokenizer.fit_on_texts(X_train)


# =========================================================
# 4. TRAINING PREPROCESSING
# =========================================================

X_train_sequences = tokenizer.texts_to_sequences(X_train)

print("\nTraining sequences:")
print(X_train_sequences)


# =========================================================
# 5. TEST PREPROCESSING
# =========================================================

X_test_sequences = tokenizer.texts_to_sequences(X_test)

print("\nTesting sequences:")
print(X_test_sequences)


# =========================================================
# 6. PADDING
# =========================================================

max_length = 6

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=max_length,
    padding="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=max_length,
    padding="post"
)

print("\nTraining padded shape:")
print(X_train_padded.shape)

print("\nTesting padded shape:")
print(X_test_padded.shape)


# =========================================================
# 7. CREATE LSTM MODEL
# =========================================================

model = Sequential([
    Embedding(
        input_dim=100,
        output_dim=16
    ),

    LSTM(32),

    Dense(
        1,
        activation="sigmoid"
    )
])


# =========================================================
# 8. COMPILE
# =========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 9. TRAIN
# =========================================================

model.fit(
    X_train_padded,
    y_train,
    epochs=30,
    batch_size=4,
    verbose=1
)


# =========================================================
# 10. EVALUATE
# =========================================================

loss, accuracy = model.evaluate(
    X_test_padded,
    y_test
)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)


# =========================================================
# 11. TEST NEW SENTENCES
# =========================================================

new_sentences = [
    "I really love this film",
    "This movie was awful",
    "The film was fantastic",
    "I hated this movie"
]

new_sequences = tokenizer.texts_to_sequences(new_sentences)

new_padded = pad_sequences(
    new_sequences,
    maxlen=max_length,
    padding="post"
)


# =========================================================
# 12. PREDICTION
# =========================================================

predictions = model.predict(new_padded)

print("\nPredictions:")

for sentence, prediction in zip(new_sentences, predictions):

    score = prediction[0]

    if score >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print("\nSentence:", sentence)
    print("Score:", score)
    print("Sentiment:", sentiment)