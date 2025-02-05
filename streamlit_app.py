import logging
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from product_categorization import (
    load_data, create_embedding_index, predict_classifications, evaluate_predictions
)

# ...existing code for setting up logging and folders...
os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "execution.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file), logging.StreamHandler()
])
logger = logging.getLogger()

st.title("Product Categorization v2")
st.sidebar.header("User Settings")
num_items = st.sidebar.number_input("Limit items for testing", min_value=1, value=100)

# Step 1: Load Data
st.header("Step 1: Load Data")
class_pbh_file = st.file_uploader("Upload Class PBH File", type=["xlsx"], key="1")
item_hierarchy_file = st.file_uploader("Upload Item Hierarchy File", type=["xlsx"], key="2")
product_sku_items_file = st.file_uploader("Upload Product SKU Items File", type=["xlsx"], key="3")

if st.button("Load Data"):
    with st.spinner("Uploading and processing files..."):
        if class_pbh_file and item_hierarchy_file and product_sku_items_file:
            # ...existing code to save uploaded files...
            class_pbh_file_path = os.path.join(output_dir, "class_pbh.xlsx")
            item_hierarchy_file_path = os.path.join(output_dir, "item_hierarchy.xlsx")
            product_sku_items_file_path = os.path.join(output_dir, "product_sku_items.xlsx")
            with open(class_pbh_file_path, "wb") as f: f.write(class_pbh_file.getbuffer())
            with open(item_hierarchy_file_path, "wb") as f: f.write(item_hierarchy_file.getbuffer())
            with open(product_sku_items_file_path, "wb") as f: f.write(product_sku_items_file.getbuffer())
            # Load data using product_categorization module
            analytical_df, hierarchy_df, approved_items, initiated_items = load_data(
                class_pbh_file_path, item_hierarchy_file_path, product_sku_items_file_path
            )
            st.session_state['approved'] = approved_items
            st.session_state['initiated'] = initiated_items
            approved_items.to_excel(os.path.join(output_dir, "approved_items.xlsx"), index=False, engine="openpyxl")
            initiated_items.to_excel(os.path.join(output_dir, "initiated_items.xlsx"), index=False, engine="openpyxl")
            st.success("Data loaded and saved.")
    st.divider()  # Visual divider between steps

# Step 2: Build Embedding Index
st.header("Step 2: Build Embedding Index")
if st.button("Build Index"):
    with st.spinner("Building embedding index..."):
        if 'approved' in st.session_state:
            approved_items = st.session_state['approved'].head(num_items)
            # ...existing code for creating embedding index...
            create_embedding_index(approved_items, search_client)
            st.success("Embedding index built.")
        else:
            st.error("Please load data first.")
    st.divider()  # Visual divider between steps

# Step 3: Predict Classifications
st.header("Step 3: Predict Classifications")
if st.button("Predict"):
    with st.spinner("Generating predictions..."):
        if 'initiated' in st.session_state:
            initiated_items = st.session_state['initiated'].head(num_items)
            # ...existing code for prediction...
            predictions = predict_classifications(initiated_items, search_client, client)
            predictions.to_excel(os.path.join(output_dir, "predicted_results.xlsx"), index=False, engine="openpyxl")
            st.session_state['predictions'] = predictions
            st.success("Predictions generated.")
        else:
            st.error("Please load data first.")
    st.divider()  # Visual divider between steps

# Step 4: Evaluate Predictions
st.header("Step 4: Evaluate Predictions")
if st.button("Evaluate"):
    with st.spinner("Evaluating predictions..."):
        if 'predictions' in st.session_state and 'initiated' in st.session_state and 'approved' in st.session_state:
            predictions = st.session_state['predictions'].head(num_items)
            accuracy = evaluate_predictions(st.session_state['initiated'], predictions, st.session_state['approved'])
            st.write("Accuracy Summary:")
            st.write(accuracy)
            st.success("Evaluation complete.")
        else:
            st.error("Ensure predictions, initiated, and approved data are available.")
    st.divider()  # Visual divider between steps
