import fastembed
from fastembed import TextEmbedding
embed = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')

async def get_text_embeddings(text:str):
  embeddings = list(embed.embed('Hi'))[0].astype('float16').tolist() #Vector embeddings output
  return embeddings
