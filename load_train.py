import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import numpy as np

# Read the dataset
df = pd.read_csv('mentalhealth_post_features_tfidf_256.csv')

# Extract features and labels
X = df['post']
selectedColumns = ['liwc_anger', 'liwc_anxiety', 'liwc_sadness', 'liwc_negative_emotion']
# Convert the selected columns to binary labels
for col in selectedColumns:
    df[col] = (df[col] > df[col].mean()).astype(int)
y = df[selectedColumns].values

# Define the maximum number of features
MAX_FEATURES = 200000

# Vectorize the text data
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=2000,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

# Build the model
model = Sequential([
    Embedding(MAX_FEATURES+1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(4, activation='sigmoid')
])

# Compile the model
model.compile(loss= BinaryCrossentropy(), optimizer='adam')

# Display model summary
model.summary()

# Define a function to evaluate model performance
def evaluate_model(dataset):
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in dataset:
        X_true, y_true = batch
        yhat = model.predict(X_true)
        y_true = np.asarray(y_true).flatten()  # Convert to NumPy array and then flatten
        yhat = yhat.flatten()
        pre.update_state(y_true, yhat)
        re.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)
    print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Train and evaluate the model
for i in range(3):
    print(f"Epoch {i+1}")
    history = model.fit(train, epochs=3, verbose=1, validation_data=val)
    evaluate_model(test)
    model.save(f'mentalProbNewest{i+1}.h5')