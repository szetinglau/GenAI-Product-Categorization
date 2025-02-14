import logging
import pandas as pd
import os
from data_loader import load_data
from embeddings import create_embeddings_and_documents
from search_utils import create_embedding_index
from evaluation import evaluate_predictions
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Load environment variables
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    openai_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
    text_embedding_deployment_name = os.getenv("AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME")
    text_embedding_model_name = os.getenv("AZURE_OPENAI_TEXT_EMBEDDING_MODEL_NAME")
    search_service_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
    search_api_key = os.getenv("SEARCH_API_KEY")
    index_name = os.getenv("SEARCH_INDEX_NAME")

    # Initialize clients
    aoai_client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_version=azure_api_version,
        api_key=azure_openai_key,
    )
    search_client = SearchClient(endpoint=search_service_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))

    # Define file paths
    class_pbh_file = "Class PBH.xlsx"
    item_hierarchy_file = "UPDATED HIERARCHY LIST.xlsx"
    product_sku_items_file = "Initiated and Approved Items V2.xlsx"
    
    # Get output directory
    output_dir = get_output_directory()

    # Load data
    class_pbh_df, item_hierarchy_df, approved_items_df, initiated_items_df = load_data(
        class_pbh_file, item_hierarchy_file, product_sku_items_file, output_dir, logger
    )
    
    # Create embedding index
    # create_embedding_index(approved_items_df, search_client, index_name, azure_openai_endpoint, text_embedding_deployment_name, text_embedding_model_name, azure_openai_key, aoai_client, output_dir, logger)

    # Predict classifications
    # predictions = create_embeddings_and_documents(initiated_items_df, aoai_client, text_embedding_model_name, output_dir)

    # Evaluate predictions with both reference DataFrames
    predictions_df = pd.read_excel(os.path.join(output_dir, "predicted_results.xlsx"), engine="openpyxl")
    accuracy = evaluate_predictions(
        initiated_items_df, predictions_df, approved_items_df, 
        item_hierarchy_df, class_pbh_df, logger, output_dir
    )
    logger.info("Accuracy Summary:")
    logger.info(accuracy)

def get_output_directory():
    # # Create a timestamped folder within the outputs directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join("outputs", timestamp)
    # os.makedirs(output_dir, exist_ok=True)

    # Use a fixed timestamp folder directory
    output_dir = os.path.join("outputs", "20250205_021108")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == "__main__":
    main()
