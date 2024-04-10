from src.vectordb.services.qdrant_setup import client, embedding_fn, models
import re

async def find_document(collection_name:str, query_text:str,filenames:list):
    vec = await embedding_fn(query_text)
    matches = []

    for filename in filenames:
        matches.append(models.FieldCondition(key='source',
                                             match=models.MatchValue(value=filename)))
    
    filter = models.Filter(should=matches)
    points = client.search(collection_name = collection_name,query_vector = vec,query_filter = filter)
    docs_str = ''
    for pt in points:
        #print(pt.dict().keys())
        text= pt.dict()['payload']['text']
        #print(text)
        #text= re.sub(r'[\s\n]+',' ',text).strip()
        docs_str += text
    
    return docs_str

async def get_all_docs_embeddings(collection_name:str,filenames:list):
    matches = []
    for filename in filenames:
        matches.append(models.FieldCondition(key='source',
                                             match=models.MatchValue(value=filename)))
    
    filter = models.Filter(should = matches)
    count_result = client.count(collection_name=collection_name, count_filter=filter)
    n_docs = count_result.count
    points = client.scroll(collection_name = collection_name, 
                           scroll_filter = filter,
                           with_payload=True, 
                           limit = n_docs,
                           with_vectors=True)
    
    docs = []
    embeddings = []
    points = points[0]
    for pt in points:
        docs.append(pt.dict()['payload']['text'])
        embeddings.append(pt.dict()['vector'])
    return docs, embeddings
