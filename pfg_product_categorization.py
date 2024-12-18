
import pandas as pd
import numpy as np
import requests, json, os
import time

from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import RetryPolicy
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)
batch_size = 15  # Adjust batch size based on your dataset
max_retries = 3

# Azure OpenAI credentials
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Replace with your Azure OpenAI endpoint
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")  # Replace with your Azure OpenAI API key
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
openai_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

text_embedding_deployment_name = os.getenv("AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME")
text_embedding_model_name = os.getenv("AZURE_OPENAI_TEXT_EMBEDDING_MODEL_NAME")

# Initialize the Azure OpenAI client
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
    analytical_hierarchy = pd.read_excel(analytical_hierarchy_file, sheet_name=0, engine="openpyxl")
    item_hierarchy = pd.read_excel(item_hierarchy_file, engine="openpyxl")
    october_items = pd.read_excel(october_list_file, engine="openpyxl")

    # print(f"Total number of items in October list: {len(october_items)}")

    # # Output all initiated items
    # initiated_items = october_items[october_items['Status'] == 'Initiated']
    # print("Initiated Items:")
    # print(initiated_items)

    # # Output all approved items
    # approved_items = october_items[october_items['Status'] == 'Approved']
    # print("Approved Items:")
    # print(approved_items)
    
    october_items["Analytical_Hierarchy"] = october_items["Analytical_Hierarchy"].str.strip() # Remove leading/trailing whitespaces
    item_hierarchy["Hierarchy CD"] = item_hierarchy["Hierarchy CD"].astype(str).str.strip() # Remove leading/trailing whitespaces

    approved_october_items = filter_and_merge_items(october_items, 'Approved', item_hierarchy)
    initiated_october_items = filter_and_merge_items(october_items, 'Initiated', item_hierarchy)
    # print(f"Number of approved items: {len(approved_october_items)}")
    # print(f"Number of initiated items: {len(initiated_october_items)}")

    # Save the approved and initiated items to files
    approved_october_items.to_excel("approved_october_items.xlsx", index=False, engine='openpyxl')
    initiated_october_items.to_excel("initiated_october_items.xlsx", index=False, engine='openpyxl')

    print("Data loaded successfully.")
    return analytical_hierarchy, item_hierarchy, approved_october_items, initiated_october_items


# Filter and merge items based on status and hierarchy
def filter_and_merge_items(items, status, item_hierarchy):
    filtered_items = items[items['Status'] == status]
    filtered_items = filtered_items[[
        'Item_Num', 
        'Description_1',
        'Description_2',
        'GTIN',
        'Brand_Id',
        'Brand_Name',
        'GDSN_Brand',
        'Long_Product_Name',
        'Pack',
        'Size',
        'Size_UOM',
        'Class_Id',
        'Class_Name',
        'PBH_ID',
        'PBH',
        'Analytical_Hierarchy',
        'Temp_Min',
        'Temp_Max',
        'Benefits',
        'General_Description'
    ]]

    # print(f"Number of {status} items before merge: {len(filtered_items)}")

    merged_data = pd.merge(
        filtered_items,
        item_hierarchy,
        left_on="Analytical_Hierarchy",
        right_on="Hierarchy CD",
        how="left"
    )

    # print(f"Number of {status} items after merge: {len(merged_data)}")

    # Rename columns
    merged_data.rename(columns={
        'Analytical_Hierarchy': 'Analytical_Hierarchy_CD',
        'Hierarchy Detail': 'Analytical_Hierarchy'
    }, inplace=True)

    # Reorder columns
    merged_data = merged_data[[
        'Item_Num', 
        'Description_1',
        'Description_2',
        'GTIN',
        'Brand_Id',
        'Brand_Name',
        'GDSN_Brand',
        'Long_Product_Name',
        'Pack',
        'Size',
        'Size_UOM',
        'Class_Id',
        'Class_Name',
        'PBH_ID',
        'PBH',
        'Analytical_Hierarchy_CD',
        'Analytical_Hierarchy',
        'Temp_Min',
        'Temp_Max',
        'Benefits',
        'General_Description'
    ]]

    return merged_data


# Generate embeddings for each item
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60))
def generate_embedding(text):
    """
    Generate embeddings using Azure OpenAI with retry logic
    """
    response = client.embeddings.create(input=[text], model=text_embedding_model_name)
    return response.data[0].embedding


def save_documents_to_json(documents, filename):
    with open(filename, 'w') as f:
        json.dump(documents, f)
    print(f"Documents saved to {filename}")


def load_documents_from_json(filename):
    with open(filename, 'r') as f:
        documents = json.load(f)
    print(f"Documents loaded from {filename}")
    return documents


def clean_value(value):
    if pd.isna(value) or value is np.nan:
        return ""
    if isinstance(value, str):
        # Replace single quotes with double quotes and strip trailing whitespace
        return value.strip().replace("'", '"')
    return str(value)


def prepare_documents_for_upload(filtered_data):
    print(f"Preparing {len(filtered_data)} documents for upload...")
    documents = []

    fields = [
        "Item_Num", "Description_1", "Description_2", "GTIN", "Brand_Id", "Brand_Name", 
        "GDSN_Brand", "Long_Product_Name", "Pack", "Size", "Size_UOM", "Class_Id", 
        "Class_Name", "PBH_ID", "PBH", "Analytical_Hierarchy_CD", "Analytical_Hierarchy", 
        "Temp_Min", "Temp_Max", "Benefits", "General_Description"
    ]

    # Filter the fields to be used for embedding
    embedding_fields = [
        "Item_Num", "Brand_Id", "Brand_Name", "GDSN_Brand", "Long_Product_Name", 
        "Class_Id", "Class_Name", "PBH_ID", "PBH", "Analytical_Hierarchy_CD", 
        "Analytical_Hierarchy", "Temp_Min", "Temp_Max", "Benefits", "General_Description"
    ]

    for idx, row in filtered_data.iterrows():
        # if idx >= 100:
        #     break

        text_to_embed = " ".join(clean_value(row[field]) for field in embedding_fields)

        document = {"id": clean_value(row["Item_Num"])}
        document.update({field: clean_value(row[field]) for field in fields})
        document["embedding"] = generate_embedding(text_to_embed)
        
        documents.append(document)
        
        if idx % 100 == 0:
            print(f"Processed {idx} documents")
            
    print(f"Finished preparing {len(documents)} documents for upload.")
    
    # Save documents to JSON
    save_documents_to_json(documents, 'prepared_documents.json')
    
    return documents


def create_semantic_search_config():
    return SemanticSearch(
        configurations=[SemanticConfiguration(
            name="mySemanticConfig",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="Long_Product_Name"),
                content_fields=[
                    SemanticField(field_name="Brand_Name"),
                    SemanticField(field_name="GDSN_Brand"),
                    SemanticField(field_name="Long_Product_Name"),
                    SemanticField(field_name="Class_Name"),
                    SemanticField(field_name="PBH"),
                    SemanticField(field_name="Analytical_Hierarchy"),
                    SemanticField(field_name="Benefits"),
                    SemanticField(field_name="General_Description")
                ],
                keywords_fields=[
                    SemanticField(field_name="Long_Product_Name"),
                    SemanticField(field_name="Class_Name"),
                    SemanticField(field_name="PBH"),
                    SemanticField(field_name="Analytical_Hierarchy")
                ]
            )
        )]
    ) 


def create_vector_search_config():
    return VectorSearch(
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
                    deployment_name=text_embedding_deployment_name,
                    model_name=text_embedding_model_name,
                    api_key=azure_openai_key
                )
            )
        ]
    )

def upload_documents_to_search(documents, search_client):
    batch_size = 15
    total_batches = (len(documents) + batch_size - 1) // batch_size  # Calculate total number of batches
    successful_uploads = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            # Upload the batch
            response = search_client.upload_documents(documents=batch)
            successful_uploads += len(batch)
            print(f"Uploaded batch {i // batch_size + 1}/{total_batches} successfully. Batch size: {len(batch)}")
        except HttpResponseError as e:
            print(f"Error uploading batch {i // batch_size + 1}/{total_batches}: {e}")
            # Log the problematic batch for further inspection
            # print(f"Problematic batch: {batch}")
            continue

    print(f"Embedding index created and documents uploaded successfully. Total successful uploads: {successful_uploads}/{len(documents)}")


# Create an index in Azure Cognitive Search
def create_embedding_index(filtered_data, search_client):
    # Prepare the documents for upload
    try:
        # Try to load documents from JSON if they exist
        documents = load_documents_from_json('prepared_documents.json')
    except FileNotFoundError:
        # If JSON file does not exist, prepare documents and save them
        documents = prepare_documents_for_upload(filtered_data)

    # Define the semantic profile configuration
    semantic_search = create_semantic_search_config()

    # Configure the vector search configuration  
    vector_search = create_vector_search_config()

    # Define the embedding index schema
    index_schema = SearchIndex(
        name=index_name,
        fields=[
            # Define the key field
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            
            # Fields for search and filtering
            SimpleField(name="Item_Num", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchField(name="Description_1", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="Description_2", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="GTIN", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SimpleField(name="Brand_Id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchField(name="Brand_Name", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="GDSN_Brand", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="Long_Product_Name", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="Pack", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SimpleField(name="Size", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SimpleField(name="Size_UOM", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SimpleField(name="Class_Id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchField(name="Class_Name", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="PBH_ID", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchField(name="PBH", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="Analytical_Hierarchy_CD", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchField(name="Analytical_Hierarchy", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="Temp_Min", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SimpleField(name="Temp_Max", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchField(name="Benefits", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="General_Description", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            
            # Vector embedding field
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=3072, vector_search_profile_name="myHnswProfile")
        ],
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    # Create the index in Azure Cognitive Search
    search_index_client = SearchIndexClient(search_client._endpoint, search_client._credential)
    search_index_client.create_or_update_index(index=index_schema)
    print(f"Index '{index_name}' created successfully.")

    upload_documents_to_search(documents, search_client)


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
# 1011979       	2024-10-25 10:22:03.8120000	Approved	elafler4210	DISPLAY COUNTER UNIT COUGH DROP	NULL	00036602834224	3012          	RICOLA	NULL	DISPLAY COUNTER UNIT COUGH DROP	30	45   	GM	14            	MISCELLANEOUS	9999	MISCELLANEOUS USE	954           	42.00	80.00	NULL	NULL
# should be DISPOSABLES>>HEALTHCARE>>MISCELLANEOUS

# Define the function to choose the best hierarchy path
def choose_best_hierarchy_path(long_product_name, pbh, class_name, predictions):
    """
    Uses GPT-4 to reason and choose the best hierarchy path from the top predictions.
    
    Args:
        long_product_name (str): The long product name.
        pbh (str): The PBH (Product Business Hierarchy) value.
        class_name (str): The product class name.
        predictions (list[dict]): A list of dictionaries containing 'hierarchy_cd' and 'hierarchy_path'.
    
    Returns:
        str: The best matching hierarchy_cd chosen by GPT-4.
    """
    # Construct the GPT-4 prompt
    # TODO Correct and improve the few shot examples in the prompt
    prompt = f"""
    You are an expert in product categorization and hierarchy matching. Your task is to evaluate which of the given hierarchy paths best matches a product's details. Use the product's Long Product Name, PBH, and Class Name to guide your reasoning.

    Each prediction is associated with a Hierarchy CD and its full path. Evaluate the predictions based on the following criteria:
        1. Relevance to the Class Name.
        2. Alignment with the PBH (Product Business Hierarchy).
        3. Consistency with the Long Product Name.

    For each prediction, use your reasoning to analyze how well it matches the product details. Then, based on your analysis, select the Hierarchy CD that provides the most accurate categorization.

    **Few-Shot Examples:**

    **Example 1:**
    - Class Name: "DISPOSABLES"
    - PBH: "CARTONS/CONTAINERS/TRAYS"
    - Long Product Name: "BOX PIZZA 16" WHITE_KRAFT"
    - Search Results:
        1. Hierarchy CD: "2024", Path: "DISPOSABLES>>CONTAINERS/TO GO>>PAPER>>FOOD TRAY"
        2. Hierarchy CD: "298", Path: "DISPOSABLES>>BOX & ACCESSORIES>>PIZZA>>PIZZA BOX"
        3. Hierarchy CD: "298", Path: "DISPOSABLES>>BOX & ACCESSORIES>>PIZZA>>PIZZA BOX"

    **Best Hierarchy CD:** 298

    ---

    **Example 2:**
    - Class Name: "FROZEN FOOD PROCESS"
    - PBH: "APPETIZERS/HORS D OEUVRES"
    - Long Product Name: "EGG ROLL PIZZA PEPPERONI"
    - Search Results:
        1. Hierarchy CD: "1013", Path: "BAKERY, FROZEN>>PRETZELS>>THAW & SERVE"
        2. Hierarchy CD: "1013", Path: "BAKERY, FROZEN>>PRETZELS>>THAW & SERVE"
        3. Hierarchy CD: "2341", Path: "BAKERY, FROZEN>>PIZZA CRUSTS>>DOUGH BALL"

    **Best Hierarchy CD:** 1139

    **Example 3:**
    - Class Name: "DAIRY PROD & SUBS"
    - PBH: "MILKS/CREAMS/YOGURT"
    - Long Product Name: "EGGNOG ULTRA-HIGH-TEMPERATURE 32OZ PAPER CARTON"
    - Search Results:
        1. Hierarchy CD: "1420", Path: "CHEESE & DAIRY>>MILK & CREAMERS>>SOY & OTHER MILKS"
        2. Hierarchy CD: "2652", Path: "CHEESE & DAIRY>>MILK & CREAMERS>>FLUID MILK>>EGG NOG"
        3. Hierarchy CD: "2652", Path: "CHEESE & DAIRY>>MILK & CREAMERS>>FLUID MILK>>EGG NOG"

    **Best Hierarchy CD:** 2652


    **Output Requirements:**
    - Provide only the best Hierarchy CD as a single number. Do not include any additional text or reasoning in your response.

    **Input Details:**
"""

    # Add the predictions to the prompt
    for i, prediction in enumerate(predictions):
        prompt += f"{i+1}. Hierarchy CD: {prediction['hierarchy_cd']}, Path: {prediction['hierarchy_path']}\n"

    # Call GPT-4o
    response = client.chat.completions.create(
        model=openai_model_name,
        messages=[
            {"role": "system", "content": "You are an expert in product categorization."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5000,  # Limit tokens to focus on reasoning and decision
        temperature=0  # Set to 0 for deterministic output
    )

    # Extract the chosen Hierarchy CD from GPT-4's response
    chosen_hierarchy_cd = response.choices[0].message.content
    return chosen_hierarchy_cd

def run_test(data, max_retries=3, retry_backoff=5):
    predicted_hierarchies = []

    for idx, row in data.iterrows(): 
        retries = max_retries
        prediction = None

        while retries > 0:
            try:
                # TODO See about leveraging L2 Reranker and QR to improve the search results: https://techcommunity.microsoft.com/blog/azure-ai-services-blog/raising-the-bar-for-rag-excellence-query-rewriting-and-new-semantic-ranker/4302729 

                # TODO update the vector search to be on more fields: 
                # 'Item_Num', 
                # 'Description_1',
                # 'Description_2',
                # 'GTIN',
                # 'Brand_Id',
                # 'Brand_Name',
                # 'GDSN_Brand',
                # 'Long_Product_Name',
                # 'Pack',
                # 'Size',
                # 'Size_UOM',
                # 'Class_Id',
                # 'Class_Name',
                # 'PBH_ID',
                # 'PBH',
                # 'Analytical_Hierarchy_CD',
                # 'Analytical_Hierarchy',
                # 'Temp_Min',
                # 'Temp_Max',
                # 'Benefits',
                # 'General_Description'

                search_results = vector_search(row["Class_Name"] + " " + row["PBH"] + " " + row["Long_Product_Name"])
                
                # TODO make predictions for the Three fields, Class, PBH, and Analytical_Hierarchy
                prediction = choose_best_hierarchy_path(row["Long_Product_Name"], row["PBH"], row["Class_Name"], search_results)
                break
            except Exception as e:
                print(f"Error occurred for item {row['Item_Num']}: {e}")
                retries -= 1
                if retries > 0:
                    print(f"Retrying... {max_retries - retries + 1}/{max_retries}")
                    time.sleep(retry_backoff ** (max_retries - retries))  # Exponential backoff
                else:
                    print(f"Max retries reached for row {idx}. Skipping...")

        predicted_hierarchies.append(prediction)   
        
        if idx == 200:
            break
    # Add the predicted values as a new column to the DataFrame
    data = data.assign(Predicted_Hierarchy_CD=predicted_hierarchies)
    
    return data


# Save predictions
def save_predictions(items, filename):
    items.to_excel(filename, index=False, engine='openpyxl')


# Evaluate the predictions using the ground truth data
def evaluate_predictions(predictions, ground_truth_data):
    hierarchy_correct = 0
    total_predictions = len(predictions)

    class_name_correct = 0
    pbh_correct = 0

    for _, row in predictions.iterrows():
        item_num = row["Item_Num"]
        
        if row["Class_Name"] == ground_truth_data.loc[ground_truth_data["Item_Num"] == item_num, "Class_Name"].values[0]:
            class_name_correct += 1

        if row["PBH"] == ground_truth_data.loc[ground_truth_data["Item_Num"] == item_num, "PBH"].values[0]:
            pbh_correct += 1
        
        predicted_hierarchy = str(row["Predicted_Hierarchy_CD"])  # Convert to string
        ground_truth_hierarchy = str(ground_truth_data.loc[ground_truth_data["Item_Num"] == item_num, "Hierarchy CD"].values[0])  # Convert to string
        if predicted_hierarchy == ground_truth_hierarchy:
            hierarchy_correct += 1

    hierarchy_accuracy = hierarchy_correct / total_predictions
    class_name_accuracy = class_name_correct / total_predictions
    pbh_accuracy = pbh_correct / total_predictions
    overall_accuracy = (class_name_correct + pbh_correct + hierarchy_correct) / (3 * total_predictions)

    print(f"Class Name Accuracy: {class_name_accuracy:.2%}")
    print(f"PBH Accuracy: {pbh_accuracy:.2%}")
    print(f"Hierarchy Accuracy: {hierarchy_accuracy:.2%}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

def main():
    # Step 1: Load the data, get the approved and initiated items
    analytical_hierarchy, item_hierarchy, approved_october_items, initiated_october_items = load_data()

    # Step 2: Build an index of items with embeddings around the Approved items
    approved_october_items = pd.read_excel("approved_october_items.xlsx", engine="openpyxl")
    create_embedding_index(approved_october_items, search_client)

    # # Step 3: Query the model for each Initiated item, predict the Class, PBH, Analytical_Hierarchy, and save the results
    # # TODO test with up to ~2220 items and get accuracies for Class, PBH, and Analytical_Hierarchy
    # results = run_test(initiated_october_items.head(200))
    # save_predictions(results, "predicted_october_items.xlsx") # Save the predictions

    # # Step 4: Evaluate the predictions
    # predicted_items = pd.read_excel("predicted_october_items.xlsx", engine="openpyxl") # Load the saved predictions
    # evaluate_predictions(predicted_items, approved_october_items) # Evaluate the predictions

if __name__ == "__main__":
    # Apply this function to extract matching rows for each item
   
    main()
