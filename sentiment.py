from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    StringInput: str

# Load the DataFrame and preprocess the text data
df = pd.read_csv('mentalhealth_post_features_tfidf_256.csv')
X = df['post']
y = df['sent_compound'].values
MAX_FEATURES = 200000
X = X.astype(str)

# Text vectorization
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=2000,
    output_mode='int'
)
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# Load the sentiment analysis model
sentiment_model = tf.keras.models.load_model('mentalState5.h5')

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = sentiment_model.predict(vectorized_comment)
    # Assuming results is a single prediction array
    map = ['Negative', "Neutral", "Positive"]
    max_index = results[0].argmax()  # Find the index of the maximum value in the prediction array
    print(results[0])
    if results[0][max_index] > 0.5:   # Check if the maximum value exceeds the threshold
        return map[max_index]
    return None
@app.post('/sentiment_pred')
def sentiment_pred(input_parameters: model_input):
    result = score_comment(input_parameters.StringInput)
    return result

nest_asyncio.apply()
uvicorn.run(app, port=8001)