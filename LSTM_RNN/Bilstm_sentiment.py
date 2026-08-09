import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

sentences = [
    # Positive
    "I love this movie",
    "This movie is amazing",
    "The film was excellent",
    "I enjoyed the movie",
    "This was a wonderful film",
    "The movie was fantastic",
    "I really enjoyed this film",
    "This film was great",
    "The movie was brilliant",
    "I liked this movie",

    # Negative
    "I hate this movie",
    "This movie is terrible",
    "The film was horrible",
    "I disliked the movie",
    "This was a boring film",
    "The movie was awful",
    "I really hated this film",
    "This film was bad",
    "The movie was disappointing",
    "I did not like this movie"
]

labels = np.array([
    1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0
])

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
print("vocabulary:")
print(tokenizer.word_index)


max_length = 6
X = pad_sequences(X, maxlen=max_length, padding="post")

print("\n padded data: ")
print(X)

##create BiLSTM model
model = Sequential([
    Embedding(
        input_dim=100,
        output_dim=16,

    ),
    Bidirectional(LSTM(32)),
    Dense(1, activation="sigmoid")

])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(X, labels, epochs=50, batch_size=4, verbose=1)

##test data
test_sentences = [
    "I really love this film",
    "This movie was fantastic",
    "The film was wonderful",

    "I really hate this film",
    "This movie was horrible",
    "The film was boring"
]

test_X = tokenizer.texts_to_sequences(test_sentences)
test_X = pad_sequences(test_X, maxlen=max_length, padding="post")

predictions = model.predict(test_X)

print("\n predictions: ")
for sentence, prediction in zip(test_sentences, predictions):
    score = prediction[0]
    if score >= 0.5:
        result = "positive"
    else:
        result="negative"
    print("Sentence :", sentence)
    print("Score    :", round(float(score), 3))
    print("Result   :", result)
    print("-" * 50)
    