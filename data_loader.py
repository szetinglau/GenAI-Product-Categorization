import pandas as pd
import os
import logging

# Define the fields to be used for embedding and search
fields = [
    "Item_Num", "Description_1", "Description_2", "GTIN", "Brand_Id", "Brand_Name", 
    "GDSN_Brand", "Long_Product_Name", "Pack", "Size", "Size_UOM", "Class_Id", 
    "Class_Name", "PBH_ID", "PBH", "Analytical_Hierarchy_CD", "Analytical_Hierarchy", 
    "Temp_Min", "Temp_Max", "Benefits", "General_Description"
]

def load_data(class_pbh_file, item_hierarchy_file, product_sku_items_file, output_dir, logger):
    # Load data into pandas DataFrames
    class_pbh_df = pd.read_excel(class_pbh_file, engine="openpyxl")
    item_hierarchy_df = pd.read_excel(item_hierarchy_file, engine="openpyxl")
    product_sku_items_df = pd.read_excel(product_sku_items_file, engine="openpyxl")

    # Log the size of each DataFrame
    logger.info(f"Class PBH DataFrame size: {class_pbh_df.shape}")
    logger.info(f"Item Hierarchy DataFrame size: {item_hierarchy_df.shape}")
    logger.info(f"Product SKU Items DataFrame size: {product_sku_items_df.shape}")

    # Log unique values in the Status column
    logger.info(f"Unique values in Status column: {product_sku_items_df['Status'].unique()}")

    # Log the length of the item hierarchy DataFrame before filtering
    logger.info(f"Item Hierarchy DataFrame length before filtering: {len(item_hierarchy_df)}")

    # Filter rows based on column C
    item_hierarchy_df = item_hierarchy_df[item_hierarchy_df.iloc[:, 2] == 'USE THESE VALUES']

    # Log the length of the item hierarchy DataFrame after filtering
    logger.info(f"Item Hierarchy DataFrame length after filtering: {len(item_hierarchy_df)}")

    logger.info("Processing approved items...")
    approved_items_df = filter_and_merge_items(product_sku_items_df, 'APPROVED', item_hierarchy_df, output_dir, logger)
    
    logger.info("Processing initiated items...")
    initiated_items_df = filter_and_merge_items(product_sku_items_df, 'INITIATED', item_hierarchy_df, output_dir, logger)
    
    # Clear specified classification fields in initiated_items_df
    for col in ["Class_Name", "PBH", "Analytical_Hierarchy", "Analytical_Hierarchy_CD"]:
        if col in initiated_items_df.columns:
            initiated_items_df[col] = ""
            
    # Check if there are any non-empty rows in the specified columns of initiated_items_df
    non_empty_rows = initiated_items_df[
        (initiated_items_df["Class_Name"] != "") |
        (initiated_items_df["PBH"] != "") |
        (initiated_items_df["Analytical_Hierarchy"] != "") |
        (initiated_items_df["Analytical_Hierarchy_CD"] != "")
    ]

    # Log a warning if there are any non-empty rows
    if not non_empty_rows.empty:
        logger.warning(f"Found {len(non_empty_rows)} rows in initiated items with non-empty classification fields.")
    else:
        logger.info("All specified classification fields in initiated items are empty.")
    
    logger.info(f"Number of approved items: {len(approved_items_df)}")
    logger.info(f"Number of initiated items: {len(initiated_items_df)}")

    logger.info("Data loaded successfully.")
    return class_pbh_df, item_hierarchy_df, approved_items_df, initiated_items_df

def filter_items_by_status(items, status):
    return items[items['Status'] == status][[
        'Item_Num', 'Description_1', 'Description_2', 'GTIN', 'Brand_Id', 'Brand_Name', 
        'GDSN_Brand', 'Long_Product_Name', 'Pack', 'Size', 'Size_UOM', 'Class_Id', 
        'Class_Name', 'PBH_ID', 'PBH', 'Hierarchy CD', 'Temp_Min', 'Temp_Max', 
        'Benefits', 'General_Description'
    ]]

def merge_items_with_hierarchy(filtered_items, item_hierarchy, output_dir, status, logger):
    merged_data = pd.merge(
        filtered_items,
        item_hierarchy,
        left_on="Hierarchy CD",
        right_on="Hierarchy CD",
        how="left"
    )
    
    # Only take items from the product sku items that have hierarchies that are in the item_hierarchy reference file
    missing = merged_data[merged_data['Hierarchy Detail'].isnull()]
    if not missing.empty:
        logger.warning("Warning: The following items have missing hierarchies in the reference file:")
        logger.warning(missing[['Item_Num', 'Hierarchy CD']])
        missing_csv_path = os.path.join(output_dir, f"{status.lower()}_items_missing_hierarchies.csv")
        missing[['Item_Num', 'Hierarchy CD']].to_csv(missing_csv_path, index=False)
    return merged_data

def rename_and_reorder_columns(merged_data):
    merged_data.rename(columns={
        'Hierarchy CD': 'Analytical_Hierarchy_CD',
        'Hierarchy Detail': 'Analytical_Hierarchy'
    }, inplace=True)
    merged_data = merged_data[fields]
    return merged_data

def filter_and_merge_items(items, status, item_hierarchy, output_dir, logger):
    filtered_items = filter_items_by_status(items, status)
    merged_data = merge_items_with_hierarchy(filtered_items, item_hierarchy, output_dir, status, logger)
    final_data = rename_and_reorder_columns(merged_data)
    return final_data
