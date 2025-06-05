import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from main import *

openai.api_key = st.secrets["openai_api_key"]

st.set_page_config(page_title="Product Explorer", layout="wide")
st.title("🛍️ Product Explorer")

# Text input for the search query
query = st.text_input("Enter your search query:")

# Retrieve data only when query changes
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Add a search button
search_clicked = st.button("Search")

# Only retrieve data when the button is clicked
if search_clicked and query and query != st.session_state.last_query:
    st.session_state.df = retrieve(query, st.secrets["openai_api_key"])
    st.session_state.last_query = query
    
df = st.session_state.df
        
if not df.empty:
    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Filter by store
    stores = df["store"].unique()
    selected_stores = st.sidebar.multiselect("Store", stores, default=stores)

    # Filter by category
    categories = df["category"].unique()
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
    
    # Filter by subcategory
    subcategories = df["subcategory"].unique()
    selected_subcategories = st.sidebar.multiselect("Subcategory", subcategories, default=subcategories)

    # Filter by price range
    min_price = float(df["price"].min())
    max_price = float(df["price"].max())
    if min_price == max_price:
        max_price += 1  # Ensure the slider has a range if all prices are the same
    price_range = st.sidebar.slider("Price Range", min_value=min_price, max_value=max_price, value=(min_price, max_price))

    # Filter by review count
    min_reviews = int(df["rating_number"].min())
    max_reviews = int(df["rating_number"].max())
    if min_reviews == max_reviews:
        max_reviews += 1  
    review_range = st.sidebar.slider("Review Count", min_value=min_reviews, max_value=max_reviews, value=(min_reviews, max_reviews))

    # Filter by review value 
    min_reviews_val = float(df["average_rating"].min())
    max_reviews_val = float(df["average_rating"].max())
    if min_reviews_val == max_reviews_val:
        max_reviews_val += 0.1 
    review_val_range = st.sidebar.slider("Review Rating", min_value=min_reviews_val, max_value=max_reviews_val, value=(min_reviews_val, max_reviews_val))

    # Apply all filters
    filtered_df = df[
        (df["store"].isin(selected_stores)) &
        (df["category"].isin(selected_categories)) &
        (df["subcategory"].isin(selected_subcategories)) &
        (df["price"].between(*price_range)) &
        (df["rating_number"].between(*review_range)) &
        (df["average_rating"].between(*review_val_range))
    ]

    st.subheader(f"🔎 Results for: '{query}'")
    st.write(f"Showing {len(filtered_df)} results")

    for idx, row in filtered_df.iterrows():
        with st.container():
            cols = st.columns([1, 4])
            # Image
            with cols[0]:
                try:
                    response = requests.get(row["main_image"], stream=True, timeout=3)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    st.image(image, use_container_width=True)
                except Exception:
                    st.write("No Image")

            with cols[1]:
                st.markdown(f"### {row['title']}")
                st.markdown(f"**Price:** <span style='color:green; font-size:1.2em;'>${row['price']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Store:** {row['store']}  |  **Category:** {row['category']}  |  **Subcategory:** {row['subcategory']}")
                st.markdown(
                    f"**Rating:** ⭐ {row['average_rating']} &nbsp;&nbsp;|&nbsp;&nbsp; **Reviews:** {row['rating_number']}",
                    unsafe_allow_html=True
                )
                if row['description']:
                    st.markdown("**Description:**")
                    st.markdown("\n".join([f"- {item}" for item in row['description']]))
                if row['features']:
                    st.markdown("**Features:**")
                    st.markdown("\n".join([f"- {item}" for item in row['features']]))
                if row['details'] and isinstance(row['details'], dict):
                    st.markdown("**More Details:**")
                    details_html = "<ul>"
                    for k, v in row['details'].items():
                        details_html += f"<li>{k}: {v}</li>"
                    details_html += "</ul>"
                    st.markdown(details_html, unsafe_allow_html=True)
            st.markdown("---")
            
else:
    if search_clicked:
        st.warning("No results found. Please try a different query.")
    else:
        st.info("Enter a search query to explore fashion products.")