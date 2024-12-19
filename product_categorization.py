'''
TODO Turn off content filter for this. It triggers maybe once every 100 product items when classifying through gpt-4o: Content filter error: Could not parse response content as the request was rejected by the content filter
TODO Need to spit out each search results + prompt going to GPT-4o for verification purposes
TODO Update Readme as this might be a highly repeatable solution
TODO Deployable solution idea: Web UI, drop in a file in this format, columns, rows, each even row hasn't been looked at, each odd row is ground truth. Spits out the accuracy of having Azure AI Search + GPT-4o/4o-mini/o1 make the product categorization
TODO Make a One Click Deploy, yes it'll be painful, but do it and you enhance repeatability
TODO Test with Blanks on Class
'''

import openai
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
from pydantic import BaseModel

load_dotenv(override=True)

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

# Define the fields to be used for embedding and search
fields = [
    "Item_Num", "Description_1", "Description_2", "GTIN", "Brand_Id", "Brand_Name", 
    "GDSN_Brand", "Long_Product_Name", "Pack", "Size", "Size_UOM", "Class_Id", 
    "Class_Name", "PBH_ID", "PBH", "Analytical_Hierarchy_CD", "Analytical_Hierarchy", 
    "Temp_Min", "Temp_Max", "Benefits", "General_Description"
]

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

    october_items["Analytical_Hierarchy"] = october_items["Analytical_Hierarchy"].str.strip() # Remove leading/trailing whitespaces
    item_hierarchy["Hierarchy CD"] = item_hierarchy["Hierarchy CD"].astype(str).str.strip() # Remove leading/trailing whitespaces

    # TODO Only take items from the october_items that have hierarchies that are in the item_hierarchy reference file

    approved_october_items = filter_and_merge_items(october_items, 'Approved', item_hierarchy)
    initiated_october_items = filter_and_merge_items(october_items, 'Initiated', item_hierarchy)
    # print(f"Number of approved items: {len(approved_october_items)}")
    # print(f"Number of initiated items: {len(initiated_october_items)}")

    print("Data loaded successfully.")
    return analytical_hierarchy, item_hierarchy, approved_october_items, initiated_october_items

# Filter items based on status
def filter_items_by_status(items, status):
    return items[items['Status'] == status][[
        'Item_Num', 'Description_1', 'Description_2', 'GTIN', 'Brand_Id', 'Brand_Name', 
        'GDSN_Brand', 'Long_Product_Name', 'Pack', 'Size', 'Size_UOM', 'Class_Id', 
        'Class_Name', 'PBH_ID', 'PBH', 'Analytical_Hierarchy', 'Temp_Min', 'Temp_Max', 
        'Benefits', 'General_Description'
    ]]


# Merge items with hierarchy
def merge_items_with_hierarchy(filtered_items, item_hierarchy):
    merged_data = pd.merge(
        filtered_items,
        item_hierarchy,
        left_on="Analytical_Hierarchy",
        right_on="Hierarchy CD",
        how="left"
    )
    return merged_data


# Rename and reorder columns
def rename_and_reorder_columns(merged_data):
    merged_data.rename(columns={
        'Analytical_Hierarchy': 'Analytical_Hierarchy_CD',
        'Hierarchy Detail': 'Analytical_Hierarchy'
    }, inplace=True)
    merged_data = merged_data[fields]
    return merged_data


# Filter and merge items based on status and hierarchy
def filter_and_merge_items(items, status, item_hierarchy):
    filtered_items = filter_items_by_status(items, status)
    merged_data = merge_items_with_hierarchy(filtered_items, item_hierarchy)
    final_data = rename_and_reorder_columns(merged_data)
    return final_data


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
                name="myHnsw",
                # efConstruction=200,  # Larger value improves recall but increases indexing time
                # m=64                # Number of neighbors, higher values improve accuracy at the cost of memory
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

    """
    # TODO Enrich the item hierarchy data:
    To improve the scoring for similar Analytical Hierarchies, we can add more detailed context in column C. This context should contain specific product descriptions or characteristics that distinguish each hierarchy. Example based on the current data:
        Example for Column C (Detailed Context)
        
            Hierarchy CD    Hierarchy Detail                                    Detailed Context (Column C)
            315             BAKERY, FROZEN>>PIES/TARTS>>TARTS                   Pre-baked or ready-to-bake tarts; commonly used in dessert preparation.
            2341            BAKERY, FROZEN>>PIZZA CRUSTS>>DOUGH BALL            Unshaped pizza dough; requires manual rolling and shaping for pizza preparation.
            28              BAKERY, FROZEN>>PIZZA CRUSTS>>PAR-BAKED             Partially baked crusts; ready for toppings and quick oven finishing.
            845             BAKERY, FROZEN>>SANDWICH CARRIERS>>HOT DOG/BRAT BUN Buns tailored for hot dogs and bratwurst; includes standard and jumbo sizes.
            1307            BAKERY, FROZEN>>SANDWICH CARRIERS>>KAISER & DELI>>PAR-BAKED
                            Par-baked Kaiser rolls; used in deli sandwiches, ready for quick baking.
        
        Explanation
        The added context in column C:
            • Distinguishes similar hierarchies by describing their specific use cases or unique characteristics.
            • Provides GPT-4o and Azure AI Search with richer information to refine scoring and improve categorization.
        Ensures better alignment between ambiguous product descriptions and the correct hierarchy during categorization.
    
    """

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


def evaluate_predictions(initiated_items, predicted_items, approved_items):
    """
        Evaluate predictions against ground truth, comparing initial, predicted, and actual states.
        Save evaluation metrics and detailed comparison to an Excel file with multiple tabs.
        Tab 1: Initiated Items (raw values).
        Tab 2: Approved Items (ground truth).
        Tab 3: Predicted Items (with Analytical Hierarchy mapped).
        Tab 4: Detailed Comparison of initial, predicted, and actual values.
        Tab 5: Mismatch Summary.
        Tab 6: Accuracy Summary.
    """

    # Load item hierarchy data
    item_hierarchy_file = "Item Hierarchy 11.14.2024.xlsx"
    item_hierarchy = pd.read_excel(item_hierarchy_file, engine="openpyxl")
    item_hierarchy["Hierarchy CD"] = item_hierarchy["Hierarchy CD"].astype(str).str.strip()  # Remove leading/trailing whitespaces

    # Merge predicted items with item hierarchy
    predictions = merge_items_with_hierarchy(predicted_items, item_hierarchy)

    # Rename and reorder columns
    predictions = predictions.rename(columns={
        'Analytical_Hierarchy': 'Analytical_Hierarchy_CD',
        'Hierarchy Detail': 'Analytical_Hierarchy'
    }, inplace=True)
    # Merge predicted items with item hierarchy to get Analytical_Hierarchy_CD
    predictions = pd.merge(
        predicted_items,
        item_hierarchy[['Hierarchy CD', 'Hierarchy Detail']],
        left_on='Analytical_Hierarchy',
        right_on='Hierarchy Detail',
        how='left'
    ).drop(columns=['Hierarchy Detail']).rename(columns={'Hierarchy CD': 'Analytical_Hierarchy_CD'})

    # TODO add all of the other fields to the detailed comparison, not just Item_Num, Brand_Name, and Long_Product_Name
    # TODO do the same for the mismatch summary
    # Prepare the detailed comparison table
    # TODO once the evaluate_predictions function is more finalized, break into smaller functions for better readability
    # TODO for the accuracy summary, think about how to show root cause analysis for the mismatches
    detailed_comparison = initiated_items[[
        "Item_Num", "Brand_Name", "Long_Product_Name"
    ]].copy()

    # Add initial, predicted, and actual values side by side for comparison
    detailed_comparison["Class_Name_initial"] = initiated_items["Class_Name"].str.strip()
    detailed_comparison["Class_Name_predicted"] = predictions["Class_Name"].str.strip()
    detailed_comparison["Class_Name_actual"] = approved_items["Class_Name"].str.strip()

    detailed_comparison["PBH_initial"] = initiated_items["PBH"].str.strip()
    detailed_comparison["PBH_predicted"] = predictions["PBH"].str.strip()
    detailed_comparison["PBH_actual"] = approved_items["PBH"].str.strip()

    detailed_comparison["Analytical_Hierarchy_CD_initial"] = initiated_items["Analytical_Hierarchy_CD"].str.strip()
    detailed_comparison["Analytical_Hierarchy_CD_predicted"] = predictions["Analytical_Hierarchy_CD"].str.strip()
    detailed_comparison["Analytical_Hierarchy_CD_actual"] = approved_items["Analytical_Hierarchy_CD"].str.strip()

    detailed_comparison["Analytical_Hierarchy_initial"] = initiated_items["Analytical_Hierarchy"].str.strip()
    detailed_comparison["Analytical_Hierarchy_predicted"] = predictions["Analytical_Hierarchy"].str.strip()
    detailed_comparison["Analytical_Hierarchy_actual"] = approved_items["Analytical_Hierarchy"].str.strip()

    # Identify mismatches
    detailed_comparison["Class_Name_Match"] = detailed_comparison["Class_Name_predicted"] == detailed_comparison["Class_Name_actual"]
    detailed_comparison["PBH_Match"] = detailed_comparison["PBH_predicted"] == detailed_comparison["PBH_actual"]
    detailed_comparison["Analytical_Hierarchy_Match"] = detailed_comparison["Analytical_Hierarchy_predicted"] == detailed_comparison["Analytical_Hierarchy_actual"]

    # Identify Analytical_Hierarchy_CD not in item_hierarchy_file and add to each row
    detailed_comparison["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"] = detailed_comparison.apply(
        lambda row: row["Analytical_Hierarchy_CD_actual"] not in item_hierarchy["Hierarchy CD"].values, axis=1
    )

    # Create a mismatch summary
    mismatch_summary = detailed_comparison.loc[
        ~detailed_comparison[["Class_Name_Match", "PBH_Match", "Analytical_Hierarchy_Match"]].all(axis=1)
    ].reset_index(drop=True)


    # Add columns to indicate the type of mismatch, including missing hierarchy CD
    mismatch_summary["Mismatch_Type"] = mismatch_summary.apply(
        lambda row: ", ".join(
            filter(None, [
                "Class_Name" if not row["Class_Name_Match"] else "",
                "PBH" if not row["PBH_Match"] else "",
                "Analytical_Hierarchy" if not row["Analytical_Hierarchy_Match"] else "",
                "Missing Analytical_Hierarchy_CD" if row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"] else ""
            ])
        ),
        axis=1
    )

    # Calculate accuracies for each field
    class_accuracy = (detailed_comparison["Class_Name_predicted"] == detailed_comparison["Class_Name_actual"]).mean()
    pbh_accuracy = (detailed_comparison["PBH_predicted"] == detailed_comparison["PBH_actual"]).mean()
    analytical_accuracy = (detailed_comparison["Analytical_Hierarchy_predicted"] == detailed_comparison["Analytical_Hierarchy_actual"]).mean()

    # Calculate overall accuracy (all fields must match)
    # This is stricter than individual field accuracies as it requires all fields to match simultaneously
    overall_accuracy = (
        (detailed_comparison["Class_Name_predicted"] == detailed_comparison["Class_Name_actual"]) &
        (detailed_comparison["PBH_predicted"] == detailed_comparison["PBH_actual"]) &
        (detailed_comparison["Analytical_Hierarchy_predicted"] == detailed_comparison["Analytical_Hierarchy_actual"])
    ).mean()

    # Calculate the percentage of Analytical Hierarchy not in the hierarchy file
    analytical_hierarchy_not_in_file_percentage = detailed_comparison["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"].mean()

    # Create a DataFrame for the accuracy summary
    accuracy_summary = pd.DataFrame({
        "Metric": ["Class Accuracy", "PBH Accuracy", "Analytical Hierarchy Accuracy", "Overall Accuracy", "Analytical Hierarchy Not In File Percentage"],
        "Value": [class_accuracy, pbh_accuracy, analytical_accuracy, overall_accuracy, analytical_hierarchy_not_in_file_percentage]
    })
    
    # Replace NaN and Inf values with an empty string
    detailed_comparison = detailed_comparison.replace([np.nan, np.inf, -np.inf], '')
    mismatch_summary = mismatch_summary.replace([np.nan, np.inf, -np.inf], '')

    # Save all results to a single Excel file with multiple tabs
    with pd.ExcelWriter("evaluation_results.xlsx", engine="xlsxwriter") as writer:
        # Tab 1: Initiated Items
        initiated_items.to_excel(writer, sheet_name="Initiated Items", index=False)

        # Tab 2: Approved Items
        approved_items.to_excel(writer, sheet_name="Approved Items", index=False)

        # Tab 3: Predicted Items
        predictions.to_excel(writer, sheet_name="Predicted Items", index=False)

        # Tab 4: Detailed Comparison
        detailed_comparison.to_excel(writer, sheet_name="Detailed Comparison", index=False)

        # Tab 5: Mismatch Summary
        mismatch_summary.to_excel(writer, sheet_name="Mismatch Summary", index=False)

        # Tab 6: Accuracy Summary
        accuracy_summary.to_excel(writer, sheet_name="Accuracy Summary", index=False)

        # Apply conditional formatting to highlight mismatches
        workbook = writer.book
        detailed_comparison_worksheet = writer.sheets["Detailed Comparison"]
        mismatch_summary_worksheet = writer.sheets["Mismatch Summary"]

        # Define the format for mismatches
        mismatch_format = workbook.add_format({'bg_color': '#FFCCCC'})
        red_format = workbook.add_format({'font_color': 'red'})

        # Apply the format to mismatched cells in Detailed Comparison
        for idx, row in detailed_comparison.iterrows():
            if not row["Class_Name_Match"]:
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("Class_Name_predicted"), row["Class_Name_predicted"], mismatch_format)
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("Class_Name_actual"), row["Class_Name_actual"], mismatch_format)
            if not row["PBH_Match"]:
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("PBH_predicted"), row["PBH_predicted"], mismatch_format)
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("PBH_actual"), row["PBH_actual"], mismatch_format)
            if not row["Analytical_Hierarchy_Match"]:
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_predicted"), row["Analytical_Hierarchy_predicted"], mismatch_format)
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_actual"), row["Analytical_Hierarchy_actual"], mismatch_format)
            if row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"]:
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_CD_actual"), row["Analytical_Hierarchy_CD_actual"], red_format)
                detailed_comparison_worksheet.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_CD_Not_In_Hierarchy_File"), row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"], mismatch_format)

        # Apply the format to mismatched cells in Mismatch Summary
        for idx, row in mismatch_summary.iterrows():
            if not row["Class_Name_Match"]:
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("Class_Name_predicted"), row["Class_Name_predicted"], mismatch_format)
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("Class_Name_actual"), row["Class_Name_actual"], mismatch_format)
            if not row["PBH_Match"]:
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("PBH_predicted"), row["PBH_predicted"], mismatch_format)
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("PBH_actual"), row["PBH_actual"], mismatch_format)
            if not row["Analytical_Hierarchy_Match"]:
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_predicted"), row["Analytical_Hierarchy_predicted"], mismatch_format)
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_actual"), row["Analytical_Hierarchy_actual"], mismatch_format)
            if row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"]:
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_CD_actual"), row["Analytical_Hierarchy_CD_actual"], red_format)
                mismatch_summary_worksheet.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_CD_Not_In_Hierarchy_File"), row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"], mismatch_format)

    return {
        "Class Accuracy": class_accuracy,
        "PBH Accuracy": pbh_accuracy,
        "Analytical Hierarchy Accuracy": analytical_accuracy,
        "Overall Accuracy": overall_accuracy,
        "Detailed Comparison": detailed_comparison,
        "Mismatch Summary": mismatch_summary
    }

def construct_query_text(row):
    """
    Construct a query text based on the semantic configuration.
    """
    # Combine prioritized fields for the query
    query_parts = [
        str(row.get('Long_Product_Name', '')),  # Title field
        str(row.get('Class_Name', '')),         # Content field
        str(row.get('PBH', '')),                # Content field
        str(row.get('Analytical_Hierarchy', '')), # Keywords field
        str(row.get('Brand_Name', '')),         # Content field
        str(row.get('GDSN_Brand', '')),         # Content field
        str(row.get('Benefits', '')),           # Content field
        str(row.get('General_Description', '')) # Content field
    ]
    # Remove empty parts and join with a space, excluding 'nan' values
    query_text = " ".join(part for part in query_parts if part.strip() and part.lower() != 'nan')
    return query_text


def hybrid_search(row, search_client, top_k=5):
    """
    Perform a hybrid search for the given row.
    """
    # Construct query text
    query_text = construct_query_text(row)

    # Define vector query
    vector_query = VectorizableTextQuery(
        text=query_text,
        k_nearest_neighbors=50,
        fields="embedding"
    )
    
    # Perform hybrid search
    results = search_client.search(
        search_text=query_text,
        semantic_configuration_name="mySemanticConfig",
        vector_queries=[vector_query],
        select=["Class_Name", "PBH", "Analytical_Hierarchy"],
        top=top_k
    )

    # # Log results for debugging
    # for result in results:
    #     print(f"Class_Name: {result['Class_Name']}, PBH: {result['PBH']}, Analytical_Hierarchy: {result['Analytical_Hierarchy']}, Score: {result['@search.score']}")
    
    return results


class ProductClassification(BaseModel):
    class_name: str
    pbh: str
    analytical_hierarchy: str


def build_classification_prompt(row, search_results):
    """
    Build the GPT-4 prompt based on the product details and search results.
    """
    product_details = "Product Details:\n"

    # TODO empty the class_name, pbh, and analytical_hierarchy fields before adding the prompt
    for field in fields:
        product_details += f"    - {field.replace('_', ' ').title()}: {row.get(field, '')}\n"
        
    prompt = f"""
        You are a product classification expert. Your task is to evaluate the search results for the product described below and determine a classification for Class Name, PBH, and Analytical Hierarchy that best matches a product's details. Use the product's fields to guide your reasoning.

        Evaluate the predictions based on the following criteria:
        1. Relevance to the Class Name.
        2. Alignment with the PBH (Product Book Handler).
        3. Consistency with the Long Product Name.

        For each prediction, use your reasoning to analyze how well it matches the product details. Then, based on your analysis, select the Class Name, PBH, and Analytical Hierarchy that provides the most accurate categorization.

        {product_details}

        Search Results:
        """
    
    if not search_results:
        print("No search results found.")
    else:
        for i, result in enumerate(search_results):
            prompt += f"{i+1}. \n"
            for field in fields:
                prompt += f"    {field.replace('_', ' ').title()}: {result.get(field, '')}\n"
            prompt += "\n"

    prompt += """
        Based on the product details and search results, select the best match for:
        - Class Name
        - PBH
        - Analytical Hierarchy
        
        Provide your response as:
        - Class Name:
        - PBH:
        - Analytical Hierarchy:
    """
    
    return prompt


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60))
def choose_best_prediction(search_results, row):
    """
    Use GPT-4o to select the best prediction based on search results and all relevant fields in the row.
    """
    prompt = build_classification_prompt(row, search_results)

    try:
        response = client.beta.chat.completions.parse(
            model=openai_model_name, # TODO Change the model to GPT-4o mini
            messages=[
                {"role": "system", "content": "You are an expert in product categorization."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0,
            response_format=ProductClassification
        )

        prediction = response.choices[0].message.parsed
        return prediction

    except openai.ContentFilterFinishReasonError as e:
        print(f"Content filter error: {e}")
        print(f"Problematic prompt: {prompt}")
        return None


def predict_classifications(initiated_data, search_client, gpt_client):
    predictions = []
    total_items = len(initiated_data)

    for idx, (_, row) in enumerate(initiated_data.iterrows(), start=1):
        # Perform hybrid search
        search_results = hybrid_search(row, search_client)

        # Generate the best prediction
        prediction = choose_best_prediction(search_results, row)
        if prediction:
            predictions.append({
                "Item_Num": row["Item_Num"],
                "Class_Name": prediction.class_name,
                "PBH": prediction.pbh,
                "Analytical_Hierarchy": prediction.analytical_hierarchy
            })

        # Log progress
        if idx % 10 == 0 or idx == total_items:
            print(f"Processed {idx}/{total_items} items")

    return pd.DataFrame(predictions)


def main():
    # Step 1: Load the data, get the approved and initiated items
    analytical_hierarchy, item_hierarchy, approved_october_items, initiated_october_items = load_data()
    approved_october_items.to_excel("approved_october_items.xlsx", index=False, engine='openpyxl')
    initiated_october_items.to_excel("initiated_october_items.xlsx", index=False, engine='openpyxl')

    # Step 2: Build an index of items with embeddings around the Approved items
    approved_october_items = pd.read_excel("approved_october_items.xlsx", engine="openpyxl").head(100)
    create_embedding_index(approved_october_items, search_client)

    # Step 3: Predict the Class, PBH, Analytical_Hierarchy, and save the results
    initiated_october_items = pd.read_excel("initiated_october_items.xlsx", engine="openpyxl").head(100)
    predictions = predict_classifications(initiated_october_items, search_client, client)
    predictions.to_excel("predicted_results.xlsx", index=False, engine="openpyxl")

    # Step 4: Evaluate the predictions
    predictions = pd.read_excel("predicted_results.xlsx", engine="openpyxl").head(100)
    accuracy = evaluate_predictions(initiated_october_items, predictions, approved_october_items)
    print("Accuracy Summary:", accuracy)

if __name__ == "__main__":
    # Apply this function to extract matching rows for each item
   
    main()
