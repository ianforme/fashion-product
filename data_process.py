import pandas as pd
import numpy as np
from tqdm import tqdm
from embedding import *
from config import *
import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["                                  
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Misc symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U000025A0-\U000025FF"  # Geometric shapes
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_list(l, delimiter=' '):
    if l == []:
        return None
    else:
        return delimiter.join(l)
    
def proecss_dict(d, delimiter=' '):
    if d == {}:
        return None
    else:
        return delimiter.join([f"{k}: {v}" for k, v in d.items()])
    
def process_details(x, details_map):
    cleaned = {}
    for k, v in x.items():
        if k in details_map.keys():
            cleaned[details_map[k]] = v
    return cleaned
    
def create_text(title, features, description, details, store):
    text = f"Title: {title}\n"
    if features:
        text += f"Features: {features}\n"
    if description:
        text += f"Description: {description}\n"
    if details:
        text += f"Details: {details}\n"
    if store:
        text += f"Store: {store}\n"

    return text.strip().lower()

def add_category_to_text(text, category, subcategory):
    text += f"\nProduct Category: {category}\n"
    text += f"Product Subcategory: {subcategory}\n"
    return text.strip().lower()

def prepare_data(input_path, 
                 cols_to_drop, 
                 sample_size=None):
    
    data = pd.read_json(input_path, lines=True)
    
    # process the lists and dictionaries
    data['features_t'] = data['features'].apply(process_list)
    data['description_t'] = data['description'].apply(process_list)
    data['details'] = data['details'].apply(lambda x: process_details(x, details_map=details_map))
    data['details_t'] = data['details'].apply(lambda x: proecss_dict(x, delimiter='; '))
    data['main_image'] = data['images'].apply(lambda x: [i['large'] for i in x if i['variant'] == 'MAIN'])
    data['main_image'] = data['main_image'].apply(lambda x: x[0] if len(x) > 0 else None)

    # create the text column before embedding
    data['text'] = data.apply(lambda x: create_text(x['title'],
                                                    x['features_t'], 
                                                    x['description_t'], 
                                                    x['details_t'],
                                                    x['store']), axis=1)
    
    # remove emojis and multiple spaces
    data['text'] = data['text'].apply(remove_emojis)
    
    # drop the necessary columns
    data = data.drop(columns=['images', 'videos'] + cols_to_drop)
    
    # make sure there are both text / main images and prices
    data = data[data['text'].notna() & data['main_image'].notna() & data['price'].notna()] 
    data = data.drop_duplicates(subset=['title'], keep='first')
    
    # used during devleopment to limit the size of the dataset
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)
        
    data = data.reset_index(drop=True)
    
    return data


def create_category(fashion_categories, text_model):
    categories = []
    for category, subcategories in fashion_categories.items():
        categories.append([category, subcategories])
    categories = pd.DataFrame(categories, columns=['category', 'subcategories'])
    categories = categories.explode('subcategories')

    categories_embeddings = get_clip_text_embedding(
        categories['subcategories'].tolist(), 
        categories['subcategories'].tolist(), 
        text_model
    )
    categories_embeddings.columns = ['subcategories', 'category_embedding']
    categories = categories.merge(categories_embeddings, on='subcategories', how='inner')
    
    return categories

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_closest_category(row, categories):
    similarities = categories['category_embedding'].apply(lambda x: cosine_similarity(row['mixed_embedding'], x))
    closest_idx = similarities.idxmax()
    return pd.Series({
        'predicted_category': categories.loc[closest_idx, 'category'],
        'predicted_subcategory': categories.loc[closest_idx, 'subcategories'],
        'sim_scores': max(similarities)
    })