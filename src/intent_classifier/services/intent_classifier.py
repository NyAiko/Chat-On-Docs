from fastembed.embedding import TextEmbedding
from typing import List
import numpy as np
from joblib import load

emb = TextEmbedding(model_name = 'BAAI/bge-small-en-v1.5')

def compute_embeddings(text:List[str]):
  embeddings = np.array(list(emb.embed(text)))
  return embeddings

def classify_intent(text:str):
  model = load('intent_clf.p')
  return model.predict([text])[0]
