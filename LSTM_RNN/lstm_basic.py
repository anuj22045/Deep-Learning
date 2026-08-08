import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

X = np.array([
    [1,2,3],
    [2,3,4],
    [3,4,5],
    [4,5,6]
    
])

y = np.array([4,5,6,7])
X = X.reshape((4,3,1))

model = Sequential([
    LSTM(64, input_shape=(3,1)),
    Dense(1)

])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X,y, epochs=100, batch_size=2)

prediction = model.predict(X)

print("\n Prediction:")
print(prediction)

loss = model.evaluate(X,y)

print("\n Loss:", loss)

