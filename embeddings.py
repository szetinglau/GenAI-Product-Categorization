import json
import os
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

# Define the fields to be used for embedding and search
fields = [
    "Item_Num", "Description_1", "Description_2", "GTIN", "Brand_Id", "Brand_Name", 
    "GDSN_Brand", "Long_Product_Name", "Pack", "Size", "Size_UOM", "Class_Id", 
    "Class_Name", "PBH_ID", "PBH", "Analytical_Hierarchy_CD", "Analytical_Hierarchy", 
    "Temp_Min", "Temp_Max", "Benefits", "General_Description"
]

def generate_embedding(text, aoai_client, text_embedding_model_name):
    """
    Generate embeddings using Azure OpenAI with retry logic
    """
    response = aoai_client.embeddings.create(input=[text], model=text_embedding_model_name)
    return response.data[0].embedding

def save_documents_to_json(documents, filename, output_dir):
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(documents, f)
    print(f"Documents saved to {filename}")

def load_documents_from_json(filename, output_dir):
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'r') as f:
        documents = json.load(f)
    print(f"Documents loaded from {file_path}")
    return documents

def clean_value(value):
    if pd.isna(value) or value is np.nan:
        return ""
    if isinstance(value, str):
        return value.strip().replace("'", '"')
    return str(value)

def create_embeddings_and_documents(product_data, aoai_client, text_embedding_model_name, output_dir):
    print(f"Preparing {len(product_data)} documents for upload...")
    documents = []
    embedding_fields = [
        "Item_Num", "Brand_Id", "Brand_Name", "GDSN_Brand", "Long_Product_Name", 
        "Class_Id", "Class_Name", "PBH_ID", "PBH", "Analytical_Hierarchy_CD", 
        "Analytical_Hierarchy", "Temp_Min", "Temp_Max", "Benefits", "General_Description"
    ]

    for idx, row in product_data.iterrows():
        text_to_embed = " ".join(clean_value(row[field]) for field in embedding_fields)

        document = {"id": clean_value(row["Item_Num"])}
        document.update({field: clean_value(row[field]) for field in fields})
        document["embedding"] = generate_embedding(text_to_embed, aoai_client, text_embedding_model_name)
        
        documents.append(document)
        
        if idx % 100 == 0:
            print(f"Processed {idx} documents")
            
    print(f"Finished preparing {len(documents)} documents for upload.")
    
    save_documents_to_json(documents, 'prepared_documents.json', output_dir)
    
    return documents
