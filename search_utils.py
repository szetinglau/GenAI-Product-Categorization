from azure.search.documents import SearchClient
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
from azure.core.exceptions import HttpResponseError
import logging
import os
from embeddings import load_documents_from_json, create_embeddings_and_documents

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

def create_vector_search_config(azure_openai_endpoint, text_embedding_deployment_name, text_embedding_model_name, azure_openai_key):
    return VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
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

def create_embedding_index(product_data, search_client, index_name, azure_openai_endpoint, text_embedding_deployment_name, text_embedding_model_name, azure_openai_key, aoai_client, output_dir, logger):
    try:
        documents = load_documents_from_json('prepared_documents.json', output_dir)
    except FileNotFoundError:
        documents = create_embeddings_and_documents(product_data, aoai_client, text_embedding_model_name, output_dir)

    semantic_search = create_semantic_search_config()
    vector_search = create_vector_search_config(azure_openai_endpoint, text_embedding_deployment_name, text_embedding_model_name, azure_openai_key)

    index_schema = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
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
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=3072, vector_search_profile_name="myHnswProfile")
        ],
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    search_index_client = SearchIndexClient(search_client._endpoint, search_client._credential)
    search_index_client.create_or_update_index(index=index_schema)
    print(f"Index '{index_name}' created successfully.")

    upload_documents_to_search(documents, search_client, logger)

def upload_documents_to_search(documents, search_client, logger):
    batch_size = 15
    total_batches = (len(documents) + batch_size - 1) // batch_size
    successful_uploads = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_ids = ", ".join([str(doc["id"]) for doc in batch])
        logger.info(f"Uploading batch {i // batch_size + 1}/{total_batches} with document IDs: {batch_ids}")
        try:
            response = search_client.upload_documents(documents=batch)
            uploaded_ids = ", ".join([str(doc["id"]) for doc in batch])
            successful_uploads += len(batch)
            logger.info(f"Uploaded batch {i // batch_size + 1}/{total_batches} successfully. Batch size: {len(batch)}. Document IDs: {uploaded_ids}")
        except HttpResponseError as e:
            logger.error(f"Error uploading batch {i // batch_size + 1}/{total_batches}: {e}")
            logger.error(f"Problematic batch document IDs: {batch_ids}")
            continue

    logger.info(f"Embedding index created and documents uploaded successfully. Total successful uploads: {successful_uploads}/{len(documents)}")
