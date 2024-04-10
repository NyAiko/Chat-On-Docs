import asyncio
from src.query_vdb.services.find_docs import get_all_docs_embeddings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time
from src.summarize.services.llm_sum import summarize


async def calculate_distances(cluster_embeddings, centroids,i):
    distances = cdist(cluster_embeddings, [centroids[i]], 'cosine')
    closest_doc_indices = np.argsort(distances.flatten())[:2]
    return closest_doc_indices, distances.flatten()

async def fast_summarization(collection_id:str, target_metadata,llm=summarize):
    start_time = time.time()
    docs, embeddings = await get_all_docs_embeddings(collection_name=collection_id, filenames=target_metadata)
    num_docs = int(len(docs))
    if num_docs>=100:
        num_clusters = int(np.log(num_docs))
    else:
        num_clusters = int(np.sqrt(num_docs))

    print('Num of Clusters :',num_clusters)
    # Perform K-means clustering on embeddings
    print('Start K-means clustering')
    kmeans = KMeans(n_clusters=num_clusters,random_state=42)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_
    # Compute centroids
    centroids = kmeans.cluster_centers_
    # Find document closest to each centroid
    cluster_summaries = {}
    summary_str = ''
    for i in range(num_clusters):
        cluster_docs = [docs[j] for j in range(len(docs)) if clusters[j] == i]
        cluster_embeddings = [embeddings[j] for j in range(len(docs)) if clusters[j] == i]
        # Calculate distances of documents in the cluster to the centroid
        closest_doc_indices, closest_distances = await calculate_distances(cluster_embeddings, centroids,i)
        closest_docs = [cluster_docs[idx] for idx in closest_doc_indices]
        cluster_summaries[f'Cluster {i+1}'] = closest_docs
        summary_str += '\n'.join(closest_docs) + '\n'
    
    endtime = time.time()
    print('Execution time: ',endtime-start_time)
    input = {'text': summary_str}
    output = llm.invoke(input)
    return output
