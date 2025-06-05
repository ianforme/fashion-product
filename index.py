import faiss
import pickle
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm

def save_faiss_index_and_metadata(index_path, metadata_path, embeddings, metadata):
    # Normalize embeddings for cosine similarity if needed
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Build index
    dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity
    index.add(normalized_embeddings.astype("float32"))

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")
    
def rephrase_query_for_embedding(user_query, system_prompt=None):
    system_prompt = system_prompt or (
        """You are a product search assistant for fashion products.
        Your task is to convert the user's query to product descriptions which are relevant to the original query
        e.g., if the user query is "I need an outfit to go to the beach this summer", you should return "swimwear|||sandals|||sunglasses".
        
        output should be delimitered by |||
        output should contain maximum 5 items
        if the user query is already a product description, you should return it as is.
        if the user query is not relevant to fashion products, you should return "not relevant to fashion products".
        """
    )

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {user_query}"}
        ],
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()

    if result.lower() == "not relevant to fashion products":
        return result

    # Split by comma and strip whitespace
    return [item.strip() for item in result.split('|||')]

def post_extraction_check(image_dict, query, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)

    # Build content list for messages
    message_content = []

    # Add text instruction
    image_ids_list = [f"{key}" for key in image_dict.keys()]
    message_content.append({
        "type": "input_text",
        "text": f"""
            You are given a user query and a list of fashion product images, each identified by an image ID.

            Query:
            "{query}"

            Below are the images, each associated with an ID.

            Your task:
            - Determine which images are relevant to the query.
            - If none of the images are relevant, return "no relevant images".
            - Return all relevant image **IDs**, delimitered by ||| (e.g., "id1|||id3").
            - Only return IDs from the provided list: {image_ids_list}
            - Do NOT make up or hallucinate any image IDs or external content.
        """
    })

    # Add each image with its ID to the message content
    for image_id, url in image_dict.items():
        message_content.append({
            "type": "input_image",
            "image_url": url  # model can't see the ID directly, so it's in prompt
        })

    # Make API call
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": message_content
            }
        ],
        temperature=0.3
    )
    
    if response.output_text == 'no relevant images':
        return []

    return [item.strip() for item in response.output_text.split('|||')]

def query_faiss_index(
    index_path,
    metadata_path,
    orig_query_text,
    text_model,
    openai_api_key,
    base_k=10,
):

    # Load FAISS index and metadata
    print(f"Loading FAISS index from {index_path} and metadata from {metadata_path}")
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    print("FAISS index and metadata loaded successfully.")
    
    
    # rephrase the query if necessary
    print(f"Rephrasing original query: {orig_query_text}")
    query_text = rephrase_query_for_embedding(orig_query_text)
    if query_text == "not relevant to fashion products":
        return pd.DataFrame()
    
    print(f"Rephrased query: {query_text}")
    res_list = []
    
    # for each key word in the query, we will search for the most relevant products
    print('Starting search for each keyword in the query...')
    for kw in query_text:
        # Embed the query
        query_vec = text_model.encode(kw).astype("float32").reshape(1, -1)

        # Perform search
        D, I = index.search(query_vec, base_k)

        # Build result list
        kw_list = [
            {**metadata[i], "score": float(D[0][rank])}
            for rank, i in enumerate(I[0])
        ]
        res_list.extend(kw_list)
    print(f"Found {len(res_list)} results for the query.")
    
    # post-extraction check for relevance
    print("Performing post-extraction relevance check...")
    check_dict = {}
    for res in res_list:
        # Use the image URL as the key and store the ID
        check_dict[res["parent_asin"]] = res["main_image"]
    
    relevant_images = []
    for i in tqdm(range(0, len(check_dict), 5)):
        batch_dict = {k: check_dict[k] for k in list(check_dict.keys())[i:i+5]}
        relevant_images_batch = post_extraction_check(batch_dict, orig_query_text, openai_api_key)
        if relevant_images_batch:
            relevant_images.extend(relevant_images_batch)
            
    print(f"Found {len(relevant_images)} relevant images after post-extraction check.")
    print("Filtering and sorting results...")
    # Filter out non-relevant results
    final_res = []
    for res in res_list:
        if res['parent_asin'] in relevant_images:
            final_res.append(res)
    print(f"Filtered down to {len(final_res)} results after relevance check.")
    
    if len(final_res) == 0:
        print("No relevant results found after filtering. Returning empty DataFrame.")
        return pd.DataFrame()
        
    # Sort by score, remove duplicates from multiple queries
    final_res = sorted(final_res, key=lambda x: x["score"], reverse=True)
    final_res = pd.DataFrame(final_res)
    final_res = final_res.drop(columns=['score']).drop_duplicates(subset=['title']).reset_index(drop=True)
    
    print(f"Complete!")
    return final_res
