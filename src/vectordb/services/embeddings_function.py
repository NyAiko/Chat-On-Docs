from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import os
from src.compute_embeddings.services.compute_embeddings import get_text_embeddings as emb_func

load_dotenv()

async def clean_text(text):
    all_stopwords = set(stopwords.words())
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    filtered_tokens = [re.sub(r'[^\w\s]','',word.lower()) for word in tokens if word.lower() not in all_stopwords]
    filtered_text = ' '.join(filtered_tokens)
    filtered_text = re.sub(r'[\s\n]+',' ',filtered_text).strip()
    return filtered_text

class DumbEmbeddingFunction():
    def __call__(self, input:list):
        output = [[0.1,0.01,0.02]]
        return output

class FastEmbeddingFunction():
    async def __call__(self, input):
        input = await clean_text(input)
        embeddings = await emb_func(input)
        return embeddings.tolist()

class EmbedWithOpenai():
    def __init__(self):
        self.model = OpenAIEmbeddings(max_retries=5, show_progress_bar=True,api_key=os.getenv('OPENAI_API_KEY'))
    def __call__(self, input:str):
        return [self.model.embed_query(input)]