from azure.core.exceptions import HttpResponseError
import pandas as pd
import requests, json, os
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv(override=True)

# Azure OpenAI credentials
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Replace with your Azure OpenAI endpoint
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")  # Replace with your Azure OpenAI API key
text_embedding_model = os.getenv("AZURE_OPENAI_TEXT_EMBEDDING_MODEL")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_api_version,
            api_key=azure_openai_key,
        )

# Azure Cognitive Search credentials
search_service_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")  # Replace with your Azure Cognitive Search endpoint
search_api_key = os.getenv("SEARCH_API_KEY")  # Replace with your Azure Cognitive Search API key
index_name = os.getenv("SEARCH_INDEX_NAME")  # Replace with your Azure Cognitive Search index name
search_client = SearchClient(endpoint=search_service_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))

# Load the input data
def load_data():
    # Replace these file paths with the actual paths
    analytical_hierarchy_file = "Analytical Hierarchy_Class_PBH.xlsx"
    item_hierarchy_file = "Item Hierarchy 11.14.2024.xlsx"
    october_list_file = "October Items.xlsx"

    # Load data into pandas DataFrames
    #class_pbh_detail = pd.read_excel(analytical_hierarchy_file, sheet_name=3, engine="openpyxl", skiprows=1)
    analytical_hierarchy = pd.read_excel(analytical_hierarchy_file, sheet_name=0, engine="openpyxl")
    item_hierarchy = pd.read_excel(item_hierarchy_file, engine="openpyxl")
    october_items = pd.read_excel(october_list_file, engine="openpyxl")
    
    october_items["Analytical_Hierarchy"] = october_items["Analytical_Hierarchy"].str.strip() # Remove leading/trailing whitespaces
    item_hierarchy["Hierarchy CD"] = item_hierarchy["Hierarchy CD"].astype(str).str.strip() # Remove leading/trailing whitespaces

    approved_october_items = october_items[october_items['Status'] == 'Approved'] # Filter approved items
    approved_october_items = approved_october_items[['Item_Num', 'Brand_Name', 'Long_Product_Name', 'Class_Name', 'PBH', 'Analytical_Hierarchy']] # Select relevant columns

    merged_data = pd.merge(
    approved_october_items,
    item_hierarchy,
    left_on="Analytical_Hierarchy",
    right_on="Hierarchy CD",
    how="inner"  # Use "inner" to keep only matching rows
    )

    # print(merged_data.head())

    return analytical_hierarchy, item_hierarchy, merged_data

# Generate embeddings for each item
def generate_embedding(text):
    """
    Generate embeddings using Azure OpenAI
    """

    return client.embeddings.create(input = [text], model=text_embedding_model).data[0].embedding

# Create an index in Azure Cognitive Search
def create_embedding_index(filtered_data, search_client):
    # Prepare documents for upload
    documents = []
    for idx, row in filtered_data.iterrows():
        embedding = generate_embedding(row["Hierarchy Detail"])
        documents.append({
            "id": str(row["Item_Num"]),
            "hierarchy_path": row["Hierarchy Detail"],
            "class_name": row["Class_Name"],
            "pbh": row["PBH"],
            "embedding": embedding,
            "hierarchy_cd": row["Analytical_Hierarchy"]
        })

    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer_name="myVectorizer"
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="myVectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=azure_openai_endpoint,
                    deployment_name="klauopenai",
                    model_name="text-embedding-ada-002",
                    api_key=azure_openai_key
                )
            )
        ]
    )


    # Define the embedding index schema
    index_schema = SearchIndex(
        name="hierarchy-index",
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            SearchField(name="hierarchy_path",  type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="class_name",  type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="pbh",  type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
            SimpleField(name="hierarchy_cd", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True) 
        ],
        vector_search=vector_search
    )

    # Create the index in Azure Cognitive Search
    search_index_client = SearchIndexClient(search_client._endpoint, search_client._credential)
 
    search_index_client.create_or_update_index(index=index_schema)

    # Upload documents in batches
    batch_size = 100

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            # Upload the batch
            response = search_client.upload_documents(documents=batch)
            print(f"Uploaded batch {i // batch_size + 1} successfully.")
        except HttpResponseError as e:
            print(f"Error uploading batch {i // batch_size + 1}: {e}")
            continue
    print("Embedding index created and documents uploaded successfully.")

# Perform vector search
def vector_search(query):
    embedding = generate_embedding(query)

    # 50 is an optimal value for k_nearest_neighbors when performing vector search
    # To learn more about how vector ranking works, please visit https://learn.microsoft.com/azure/search/vector-search-ranking
    vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="embedding")
    
    results = search_client.search(  
        search_text=query,  
        vector_queries= [vector_query],
        select=["hierarchy_path", "hierarchy_cd"],
        top=3
    )  
    # for result in results:
        # print(result["hierarchy_path"], result["hierarchy_cd"], result["@search.score"])

    return results

 # vector_search("DAIRY PROD & SUBS CHEESE CHEESE FRENCH") # = 1687 is False
# vector_search("PRODUCE PRE-CUT VEGETABLEs PRE-CUT VEGETABLE BLEND ROASTING CAULIFLOWER BUTTERNUT RED PEARL ONION BRUSSEL SPROUT") # should be 2539
# vector_search("CHEMICALS & CLEANING FOOD SAFETY & KITCHEN CLN GREASE CARTRIDGE FOOD GRADE") # should be 1121

# Save predictions
def save_predictions(items, filename):
    items.to_excel(filename, index=False, engine='openpyxl')

def run_test(data):
    predicted_hierarchies = []

    for idx, row in data.iterrows():
        results = vector_search(row["Class_Name"] + " " + row["PBH"] + " " + row["Long_Product_Name"])
        tmp = []
        # Create a list to store the predicted values

        for result in results:
            tmp.append(result["hierarchy_cd"]) 
        predicted_hierarchies.append(tmp)    
        
        if idx == 15:
            break
    # Add the predicted values as a new column to the DataFrame
    data = data.assign(Predicted_Hierarchy_CD=predicted_hierarchies)
    
    return data


def main():
     # Step 1: Load the data
    analytical_hierarchy, item_hierarchy, october_items = load_data()

    # Step 2: Build an index of items with embeddings
    #create_embedding_index(october_items, search_client)

    # Step 3: Query the model for each item
    results = run_test(october_items.head(15))

    save_predictions(results, "predicted_october_items.xlsx")

if __name__ == "__main__":
    # Apply this function to extract matching rows for each item
   
    main()
