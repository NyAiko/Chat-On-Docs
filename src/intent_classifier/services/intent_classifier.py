from src.vectordb.services import embeddings_function
from joblib import load
import numpy as np

emb_fn = embeddings_function.FastEmbeddingFunction()


async def classify_intent(text:str):
    vec = await emb_fn(text)
    sgd = load('sgd.p')
    return sgd.predict([np.array(vec)])[0]

