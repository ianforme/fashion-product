
import time
import openai
import faiss
import requests

import numpy as np
import pandas as pd

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def get_clip_text_embedding(text_list, id_list, model, batch_size=100):
    all_embeddings = []
    all_ids = []
    
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_text = text_list[i:i + batch_size]
        batch_ids = id_list[i:i + batch_size]
        text_embeddings = model.encode(batch_text)
        all_embeddings.append(text_embeddings)
        all_ids.extend(batch_ids)

    all_embeddings = np.vstack(all_embeddings)
    df = pd.DataFrame({'parent_asin': all_ids, 'text_embedding': list(all_embeddings)})
    return df

def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        response = requests.get(url_or_path, stream=True)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    else:
        return Image.open(url_or_path)
    
def get_clip_image_embedding(image_urls, id_list, model, batch_size=100):
    all_embeddings = []
    all_ids = []
    
    for i in tqdm(range(0, len(image_urls), batch_size)):
        batch_urls = image_urls[i:i + batch_size]
        batch_ids = id_list[i:i + batch_size]
        images = []
        valid_ids = []
        
        for url, id_ in zip(batch_urls, batch_ids):
            try:
                if url is None or pd.isna(url):
                    continue
                image = load_image(url)
                images.append(image)
                valid_ids.append(id_)
            except Exception as e:
                print(f"Error loading image {url}: {e}")
                continue
        
        if images:
            try:
                image_embeddings = model.encode(images)
                all_embeddings.append(image_embeddings)
                all_ids.extend(valid_ids)
            except Exception as e:
                print(f"Error encoding batch {i}-{i+batch_size}: {e}")
                continue
    
    all_embeddings = np.vstack(all_embeddings)
    df = pd.DataFrame({'parent_asin': all_ids, 'image_embedding': list(all_embeddings)})
    return df

def mix_embeddings(row, img_rato=0.75, txt_ratio=0.25):
    img_emb = np.array(row['image_embedding'])
    txt_emb = np.array(row['text_embedding'])
    return img_emb * img_rato + txt_emb * txt_ratio

