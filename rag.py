import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def get_prompt(instruction: str, query: str) -> str:
    return f'Instruct: {instruction}\nQuery: {query}'

def get_verbalized_conv(conv: list) -> str:
    return f'User: {conv[0]}\nAssistant: {conv[1]}'

def retrieve_top_summary(query_conv, model, 
                          data_path='data/strategy_arsenal.json', 
                          index_path='data/index/cwe_convs.index', 
                          instruction="Given a search query, retrieve relevant passages to the query", 
                          topk=1):
    query_conv = get_verbalized_conv(query_conv)
    
    total_convs = []
    conv_id_to_data_id = {}
    conv_id = 0
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    for id, d in enumerate(data):
        convs = d['conversation']
        for c in convs:
            c = get_verbalized_conv(c)
            total_convs.append(c)
            conv_id_to_data_id[conv_id] = id
            conv_id += 1
    
    if not os.path.exists(index_path):
        print("Creating Conversation Index...")
        embeddings = model.encode(total_convs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
        assert embeddings.shape[0] == len(conv_id_to_data_id)
        index.add_with_ids(embeddings.detach().cpu(), np.array(range(embeddings.shape[0])))
        faiss.write_index(index, index_path)
    else:
        index = faiss.read_index(index_path)
    
    query_embedding = model.encode([get_prompt(instruction, query_conv)], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    scores, ids = index.search(query_embedding.detach().cpu(), k=topk)
    retrieved_data = [data[conv_id_to_data_id[conv_id]] for conv_id in ids.tolist()[0]]
    
    return retrieved_data[0]['summary'], retrieved_data[0]['jailberak_task'] if retrieved_data else None


