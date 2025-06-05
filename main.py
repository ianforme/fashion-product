from data_process import *
from embedding import *
from config import *
from index import *

def index_creation(sample_size=None):
    print("Preparing data...")
    data = prepare_data(data_path, cols_to_drop=cols_to_drop, sample_size=sample_size)
    print(f"Data loaded with {len(data)} records.")
    
    print("Load multimodal embedding models...")
    img_model = SentenceTransformer(img_model_path)
    text_model = SentenceTransformer(text_model_path)
    
    print("Creating text embeddings...")
    text_embeddings = get_clip_text_embedding(data['text'].tolist(),
                                              data['parent_asin'].tolist(), 
                                              text_model, 
                                              batch_size=batch_size)
    
    print("Creating image embeddings...")
    image_embeddings = get_clip_image_embedding(data['main_image'].tolist(), 
                                                data['parent_asin'].tolist(), 
                                                img_model,
                                                batch_size=batch_size)

    print("Mixing embeddings...")
    merged_embeddings = pd.merge(image_embeddings, text_embeddings, on='parent_asin', how='inner')
    merged_embeddings['mixed_embedding'] = merged_embeddings.apply(lambda x: mix_embeddings(x, img_ratio, text_ratio), axis=1)
    data = data.merge(merged_embeddings, on='parent_asin', how='inner')
    
    print("Tag categories...")
    categories = create_category(fashion_categories, text_model)
    data[['category', 'subcategory', 'sim_scores']] = data.apply(lambda x: find_closest_category(x, categories), axis=1)
    
    print("Data preparation complete. Saving index to disk...")
    data = data[['parent_asin', 
                'title', 'store', 'features', 'description', 'details',
                'category', 'subcategory', 
                'average_rating', 'rating_number', 'price', 
                'main_image', 
                'mixed_embedding']]
    
    save_faiss_index_and_metadata(
        index_path,
        metadata_path,
        embeddings=np.vstack(data["mixed_embedding"].values),
        metadata=data.drop(columns=["mixed_embedding"]).to_dict(orient="records")
    )
    
def retrieve(query_text, base_k=base_k):
    print("Querying FAISS index...")
    text_model = SentenceTransformer(text_model_path)
    
    results = query_faiss_index(
        index_path,
        metadata_path,
        query_text,
        text_model,
        openai_api_key,
        base_k=base_k,
    )
    
    return results