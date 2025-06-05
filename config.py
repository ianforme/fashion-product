# Dynamic configuration for the fashion recommendation system
img_ratio = 0.7
text_ratio = 0.3
base_k = 10
batch_size = 100

# Static configuration for the OpenAI API key and model paths
openai_api_key = 'your_openai_api_key_here'  # Replace with your actual OpenAI API key
img_model_path = 'clip-ViT-B-32'
text_model_path = 'sentence-transformers/clip-ViT-B-32'

index_path = './index/faiss_index.index'
metadata_path = './index/faiss_metadata.pkl'

data_path = './data/meta_Amazon_Fashion.jsonl'
cols_to_drop = ['main_category', 'bought_together', 'categories']

# Predefined fashion categories and details mapping
fashion_categories = {
    "Apparel": [
        "Dresses",
        "Jackets & Coats",
        "Jeans & Pants",
        "Jumpsuits & Rompers",
        "Lingerie & Sleepwear",
        "T-Shirts & Tops",
        "Shorts & Skirts",
        "Sweaters & Knits",
        "Swimwear",
        "Sportswear",
    ],
    
    "Shoes": [
        "Boots",
        "Flats & Loafers",
        "Heels & Pumps",
        "Sandals & Espadrilles",
        "Sneakers & Athletic",
        "Clogs",
    ],

    "Bags": [
        "Backpacks",
        "Bucket & Tote Bags",
        "Clutches & Mini Bags",
        "Crossbody & Shoulder Bags",
        "Top Handle Bags",
        "Wallets & Card Cases",
    ],

    "Accessories": [
        "Belts",
        "Hats",
        "Jewelry",
        "Sunglasses",
        "Watches",
        "Gloves & Scarves",
        "Facemasks",
        "Other Accessories",
    ]
}

details_map = {
    "Date First Available": "Date First Available",
    "Package Dimensions": "Package Dimensions",
    "Item model number": "Item model number",
    "Is Discontinued By Manufacturer": "Is Discontinued By Manufacturer",
    "Product Dimensions": "Product Dimensions",
    "Department": "Department",
    "Manufacturer": "Manufacturer",
    "Brand": "Brand",
    "Age Range (Description)": "Age Range (Description)",
    "Material": "Material",
    "Item Weight": "Item Weight",
    "Style": "Style",
    "Color": "Color",
    "Closure Type": "Closure Type",
    "Size": "Size",
    "Shape": "Shape",
    "Reusability": "Reusability",
    "Theme": "Theme",
    "Special Feature": "Special Feature",
    "Pattern": "Pattern",
    "Country of Origin": "Country of Origin",
    "Unit Count": "Unit Count",
    "Item Package Quantity": "Item Package Quantity",
    "Clasp Type": "Clasp Type",
    "Sport": "Sport",
    "Neck Style": "Neck Style",
    "Batteries": "Batteries",
    "Item Dimensions LxWxH": "Item Dimensions LxWxH",
    "Fit Type": "Fit Type",
    "Sleeve Type": "Sleeve Type",
    "Chain Type": "Chain Type",
    "Manufacturer recommended age": "Manufacturer recommended age",
    "Number of Items": "Number of Items",
    "Number of Pieces": "Number of Pieces",
    "Fabric Type": "Fabric Type",
    "Occasion": "Occasion",
    "Product Care Instructions": "Product Care Instructions",
    "Frame Material": "Frame Material",
    "Collection Name": "Collection Name",
    "Metal Type": "Metal Type",
    "Item Length": "Item Length",
    "Shirt form type": "Shirt form type",
    "Brand Name": "Brand",
    "Metal Stamp": "Metal Stamp",
    "Part Number": "Part Number",
    "Target Audience": "Target Audience",
    "Item Package Dimensions L x W x H": "Item Dimensions LxWxH",
    "Cartoon Character": "Cartoon Character",
    "Package Weight": "Package Weight",
    "Included Components": "Included Components",
    "Collar Style": "Collar Style",
    "Hand Orientation": "Hand Orientation",
    "Item Length (Description)": "Item Length",
    "Sport Type": "Sport",
    "Lens Color": "Lens Color",
    "Form Factor": "Form Factor",
    "Lining Description": "Lining Description",
    "Frame Type": "Frame Type",
    "Lens Coating Description": "Lens Coating Description",
    "Suggested Users": "Target Audience",
    "Band Material Type": "Band Material Type",
    "Team Name": "Team Name",
    "Band Color": "Band Color",
    "Recommended Uses For Product": "Recommended Uses For Product",
    "Ply Rating": "Ply Rating",
    "Compatible Phone Models": "Compatible Phone Models",
    "Use for": "Recommended Uses For Product",
    "Filter Class": "Filter Class",
    "Handle Material": "Handle Material",
    "Vehicle Service Type": "Vehicle Service Type",
    "Opening Mechanism": "Opening Mechanism",
    "Item Form": "Item Form",
    "Manufacturer Part Number": "Part Number",
    "Number of Labels": "Number of Labels",
    "League": "League",
    "Hair Type": "Hair Type",
    "Mounting Type": "Mounting Type",
    "Item Dimensions  LxWxH": "Item Dimensions LxWxH",
    "Compatible Devices": "Compatible Devices",
    "Number of Sets": "Number of Sets",
    "Finish Type": "Finish Type",
    "Lens Material": "Lens Material",
    "Stone Color": "Stone Color",
    "Band Size": "Band Size",
    "Model Name": "Model Name",
    "Outer Material": "Outer Material",
    "Capacity": "Capacity",
    "Material Type": "Material",
    "Primary Stone Gem Type": "Primary Stone Gem Type",
    "": "",
    "Band Width": "Band Width",
    "Water Resistance Level": "Water Resistance Level",
    "Language": "Language",
    "Size Map": "Size Map",
    "UPC": "UPC",
    "Shaft Material": "Shaft Material",
    "Top Style": "Top Style",
    "Model Year": "Model Year",
    "Batteries required": "Batteries"
}