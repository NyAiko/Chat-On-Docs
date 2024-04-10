from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
from src.vectordb.services.embeddings_function import FastEmbeddingFunction
from langchain_core.documents import Document
import uuid
import asyncio
from typing import List

load_dotenv()

client = QdrantClient(url='https://8684201d-ef62-4e5f-9fa0-62f900cb1ac3.europe-west3-0.gcp.cloud.qdrant.io',
                      api_key=os.getenv('qdrant_api_key'))

embedding_fn = FastEmbeddingFunction()

async def create_collection(collection_name:str):
    client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    print('Collection created')

async def delete_collection(collection_name:str):
    client.delete_collection(collection_name=collection_name)

async def compute_embedding(doc):
    return (doc, await embedding_fn(doc.page_content))

async def insert_docs(collection_name: str, docs: List[Document]):
    if client.collection_exists(collection_name)==False:
        await create_collection(collection_name=collection_name)
        print('Collection created')
    try:
        print('Computing embeddings')
        embeddings = await asyncio.gather(*[compute_embedding(doc) for doc in docs])
        batch_size = 10
        print('Start adding docs to the database')
        if len(docs)>=batch_size:
            for i in range(0, len(embeddings), batch_size):
                batch_points = []
                for doc, vector_embedding in embeddings[i:i+batch_size]:
                    batch_points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector_embedding,
                            payload={"text": doc.page_content, 
                                    "source": doc.metadata['source'],
                                    "page": doc.metadata['page']},
                        )
                    )
                print(f'Adding {len(batch_points)} docs to the Vector Database')
                client.upsert(collection_name=collection_name, points=batch_points)
        else:
            print('Adding docs to the vector databse')
            for i in range(len(docs)):
                doc =docs[i]
                client.upsert(collection_name=collection_name,
                              points = [models.PointStruct(id=str(uuid.uuid4()),
                                        vector=embeddings[i],
                                        payload={"text": doc.page_content, 
                                                "source": doc.metadata['source'],
                                                "page": doc.metadata['page']})])        
            client.upsert(collection_name=collection_name, points=batch_points)
            
    except Exception as e:
        error = f'Error in adding docs: {e}'
        raise(error)