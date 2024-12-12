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
from azure.core.pipeline.policies import RetryPolicy
from dotenv import load_dotenv
import time

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
    #class_pbh_detail = pd.read_excel(analytical_hierarchy_file, sheet_name=3, engine="openpyxl", skiprows=1)
    analytical_hierarchy = pd.read_excel(analytical_hierarchy_file, sheet_name=0, engine="openpyxl")
    item_hierarchy = pd.read_excel(item_hierarchy_file, engine="openpyxl")
    october_items = pd.read_excel(october_list_file, engine="openpyxl")
    
    october_items["Analytical_Hierarchy"] = october_items["Analytical_Hierarchy"].str.strip() # Remove leading/trailing whitespaces
    item_hierarchy["Hierarchy CD"] = item_hierarchy["Hierarchy CD"].astype(str).str.strip() # Remove leading/trailing whitespaces

    approved_october_items = filter_and_merge_items(october_items, 'Approved', item_hierarchy)
    initiated_october_items = filter_and_merge_items(october_items, 'Initiated', item_hierarchy)

    return analytical_hierarchy, item_hierarchy, approved_october_items, initiated_october_items


# Filter and merge items based on status and hierarchy
def filter_and_merge_items(items, status, item_hierarchy):
    filtered_items = items[items['Status'] == status]
    filtered_items = filtered_items[['Item_Num', 'Brand_Name', 'Long_Product_Name', 'Class_Name', 'PBH', 'Analytical_Hierarchy']]
    merged_data = pd.merge(
        filtered_items,
        item_hierarchy,
        left_on="Analytical_Hierarchy",
        right_on="Hierarchy CD",
        how="inner"
    )
    return merged_data


# Generate embeddings for each item
def generate_embedding(text):
    """
    Generate embeddings using Azure OpenAI
    """

    return client.embeddings.create(input = [text], model=text_embedding_model_name).data[0].embedding

# Create an index in Azure Cognitive Search
def create_embedding_index(filtered_data, search_client):
    # Prepare documents for upload
    documents = []
    for idx, row in filtered_data.iterrows():
        text_to_embed = f"{row['Class_Name']} {row['PBH']} {row['Hierarchy Detail']}"
        hierarchy_embedding = generate_embedding(text_to_embed)
        documents.append({
            "id": str(row["Item_Num"]),
            
            # TODO update the embedding index to be on more fields: 
            # Brand_Id
            # Description_1	
            # Description_2	
            # GTIN	
            # Brand_Id	
            # Brand_Name	
            # GDSN_Brand	
            # Long_Product_Name	<-
            # Pack	
            # Size	
            # Size_UOM	
            # Class_Id	
            # Class_Name	<-
            # PBH_ID	
            # PBH	<-
            # Analytical_Hierarchy_cd	<-
            # Analytical_Hierarchy	<-
            # Temp_Min	
            # Temp_Max	
            # Benefits	
            # General_Description
            
            
            "class_name": row["Class_Name"],
            "pbh": row["PBH"],
            "hierarchy_cd": row["Analytical_Hierarchy"],
            "hierarchy_path": row["Hierarchy Detail"],

            "embedding": hierarchy_embedding
        })

    # TODO add Semantic Configuration to the index
    
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
                    deployment_name=text_embedding_deployment_name,
                    model_name=text_embedding_model_name,
                    api_key=azure_openai_key
                )
            )
        ]
    )


    # Define the embedding index schema
    index_schema = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            
            # TODO update the embedding index to be on more fields: 
            # Brand_Id
            # Description_1	
            # Description_2	
            # GTIN	
            # Brand_Id	
            # Brand_Name	
            # GDSN_Brand	
            # Long_Product_Name	<-
            # Pack	
            # Size	
            # Size_UOM	
            # Class_Id	
            # Class_Name	<-
            # PBH_ID	
            # PBH	<-
            # Analytical_Hierarchy_cd	<-
            # Analytical_Hierarchy	<-
            # Temp_Min	
            # Temp_Max	
            # Benefits	
            # General_Description
            
            SearchField(name="class_name",  type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchField(name="pbh",  type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="hierarchy_cd", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True), 
            SearchField(name="hierarchy_path",  type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
            
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
        ],
        vector_search=vector_search
    )

    # Create the index in Azure Cognitive Search
    search_index_client = SearchIndexClient(search_client._endpoint, search_client._credential)
 
    search_index_client.create_or_update_index(index=index_schema)

    # Upload documents in batches
    batch_size = 15

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
                # TODO update the vector search to be on more fields: 
                # Brand_Id
                # Description_1	
                # Description_2	
                # GTIN	
                # Brand_Id	
                # Brand_Name	
                # GDSN_Brand	
                # Long_Product_Name	<-
                # Pack	
                # Size	
                # Size_UOM	
                # Class_Id	
                # Class_Name	<-
                # PBH_ID	
                # PBH	<-
                # Analytical_Hierarchy_cd	<-
                # Analytical_Hierarchy	<-
                # Temp_Min	
                # Temp_Max	
                # Benefits	
                # General_Description

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
    # analytical_hierarchy, item_hierarchy, approved_october_items, initiated_october_items = load_data()
    # TODO save the approved_october_items and initiated_october_items to files

    # Step 2: Build an index of items with embeddings around the Approved items
    # create_embedding_index(approved_october_items, search_client)

    # Step 3: Query the model for each Initiated item, predict the Class, PBH, Analytical_Hierarchy, and save the results
    # TODO test with up to ~2220 items
    # results = run_test(initiated_october_items.head(200))
    # save_predictions(results, "predicted_october_items.xlsx") # Save the predictions

    # Step 4: Evaluate the predictions
    # predicted_items = pd.read_excel("predicted_october_items.xlsx", engine="openpyxl") # Load the saved predictions
    # evaluate_predictions(predicted_items, approved_october_items) # Evaluate the predictions

if __name__ == "__main__":
    # Apply this function to extract matching rows for each item
   
    main()
