import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

sentences = [
    "I love this movie",
    "This movie is amazing",
    "The film was excellent",
    "I enjoyed the movie",
    "This was a wonderful film",

    "I hate this movie",
    "This movie is terrible",
    "The film was horrible",
    "I disliked the movie",
    "This was a boring film"
]

labels = np.array([
    1,1,1,1,1,0,0,0,0,0
])


tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

X = tokenizer.texts_to_sequences(sentences)

print("Tokenizer sentence: ",X)

#padding

X = pad_sequences(X, maxlen=5, padding="post")
print("Padded sentences: ", X)

##create LSTM model

model = Sequential([
    Embedding(input_dim=100, output_dim=16, input_length=5),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

##compile

model.compile(
    optimizer = "adam",
    loss= "binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, labels, epochs=100, verbose=1
)

test_sentences = [
    # Positive
    "I really love this film",
    "This movie was fantastic",
    "The film was wonderful",
    "I enjoyed watching this movie",
    "This was an excellent film",

    # Negative
    "I really hate this film",
    "This movie was horrible",
    "The film was boring",
    "I disliked watching this movie",
    "This was a terrible film"
]

test_X = tokenizer.texts_to_sequences(test_sentences)

test_X = pad_sequences(
    test_X, maxlen=5, padding="post"
)

#prediction
predictions = model.predict(test_X)
print("\npredictions: ")
for sentence, prediction in zip(test_sentences, predictions):
    if prediction[0]>0.5:
        result="positive"
    else:
        result="negative"
    print(sentence)
    print("Score: ", prediction[0])
    print("result: ", result)
    print()

