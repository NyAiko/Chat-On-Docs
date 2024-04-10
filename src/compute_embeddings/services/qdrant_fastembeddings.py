import fastembed
from fastembed import TextEmbedding
embed = TextEmbedding(model_name='intfloat/multilingual-e5-large')

def get_text_embeddings(text:str):
  embeddings = list(embed.embed('Hi'))[0].astype('float16').tolist() #Vector embeddings output
  return embeddings
