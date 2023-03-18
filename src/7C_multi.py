from utils import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')
y_train['HH'] = ((y_train.SaO2 < 88) & (X_train.SpO2 >= 88)).astype(int)

# Create Model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))

# Define your loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer='adam', loss=['mse', loss_fn], metrics=['mse', 'accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32)

# Save the model to a file
filename = "models/multi"
tf.keras.models.save_model(model, filename)