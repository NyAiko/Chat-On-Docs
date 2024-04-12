from fastembed.embedding import TextEmbedding
from typing import List
import numpy as np
from joblib import load

emb = TextEmbedding(model_name = 'BAAI/bge-small-en-v1.5')

def get_text_embeddings(text:List[str]):
  embeddings = np.array(list(emb.embed(text)))
  return embeddings[0].tolist()

