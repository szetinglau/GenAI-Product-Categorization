import pandas as pd
import numpy as np
import difflib
import logging
from excel_utils import save_evaluation_excel

def evaluate_predictions(initiated_items, predicted_items, approved_items, item_hierarchy_df, class_pbh_df, logger, output_dir):
    predictions_df = merge_with_item_hierarchy(predicted_items, item_hierarchy_df, logger)
    detailed_comparison_df, invalid_counts = prepare_detailed_comparison(initiated_items, predictions_df, approved_items, 
                                                       item_hierarchy_df, class_pbh_df, logger)
    mismatch_summary_df = create_mismatch_summary(detailed_comparison_df, item_hierarchy_df)

    detailed_comparison_df = detailed_comparison_df.replace([np.nan, np.inf, -np.inf], '')
    mismatch_summary_df = mismatch_summary_df.replace([np.nan, np.inf, -np.inf], '')
    
    class_acc, pbh_acc, analytical_acc, accuracy_summary, category_performance_df, hl1, hl2, hl3 = compute_accuracies(
        detailed_comparison_df, initiated_items, invalid_counts
    )

    save_evaluation_excel(initiated_items, approved_items, predictions_df, detailed_comparison_df,
                          mismatch_summary_df, accuracy_summary, category_performance_df, hl1, hl2, hl3, output_dir)

    return {
        "Class Accuracy": class_acc,
        "PBH Accuracy": pbh_acc,
        "Analytical Hierarchy Accuracy": analytical_acc
    }

def merge_with_item_hierarchy(predicted_items, item_hierarchy, logger):
    merged = pd.merge(
        predicted_items,
        item_hierarchy[['Hierarchy CD', 'Hierarchy Detail']],
        left_on='Analytical_Hierarchy',
        right_on='Hierarchy Detail',
        how='left'
    ).rename(columns={'Hierarchy CD': 'Analytical_Hierarchy_CD'}).drop('Hierarchy Detail', axis=1)

    merged["Analytical_Hierarchy_CD"] = merged["Analytical_Hierarchy_CD"].apply(
        lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, float) and x.is_integer() else str(x).strip()
    )
    logger.info(f"Predictions shape after merge: {merged.shape}")
    logger.info(f"Null Analytical_Hierarchy_CD count: {merged['Analytical_Hierarchy_CD'].isnull().sum()}")
    return merged

def verify_classifications(predictions, item_hierarchy_df, class_pbh_df, logger):
    """
    Verify that predicted classifications exist in reference data.
    Returns both the verified predictions and counts of invalid entries.
    """
    # Create verification columns
    predictions["Class_In_Reference"] = predictions["Class_Name"].isin(class_pbh_df["Class_Name"])
    predictions["PBH_In_Reference"] = predictions["PBH"].isin(class_pbh_df["PBH"])
    predictions["Hierarchy_In_Reference"] = predictions["Analytical_Hierarchy"].isin(item_hierarchy_df["Hierarchy Detail"])

    # Get invalid counts
    invalid_counts = {
        "Class_Name": (~predictions["Class_In_Reference"]).sum(),
        "PBH": (~predictions["PBH_In_Reference"]).sum(),
        "Analytical_Hierarchy": (~predictions["Hierarchy_In_Reference"]).sum()
    }

    # Log verification results
    invalid_class = predictions[~predictions["Class_In_Reference"]]
    invalid_pbh = predictions[~predictions["PBH_In_Reference"]]
    invalid_hierarchy = predictions[~predictions["Hierarchy_In_Reference"]]

    if not invalid_class.empty:
        logger.warning(f"Found {len(invalid_class)} predictions with invalid Class_Name:")
        logger.warning(invalid_class[["Item_Num", "Class_Name"]].head())

    if not invalid_pbh.empty:
        logger.warning(f"Found {len(invalid_pbh)} predictions with invalid PBH:")
        logger.warning(invalid_pbh[["Item_Num", "PBH"]].head())

    if not invalid_hierarchy.empty:
        logger.warning(f"Found {len(invalid_hierarchy)} predictions with invalid Analytical_Hierarchy:")
        logger.warning(invalid_hierarchy[["Item_Num", "Analytical_Hierarchy"]].head())

    return predictions, invalid_counts

def prepare_detailed_comparison(initiated_items, predictions, approved_items, item_hierarchy_df, class_pbh_df, logger):
    """
    Create a detailed comparison DataFrame with reference data verification.
    """
    # Verify predictions against reference data
    verified_predictions, invalid_counts = verify_classifications(predictions.copy(), item_hierarchy_df, class_pbh_df, logger)
    
    detailed = initiated_items[["Item_Num", "Brand_Name", "Long_Product_Name"]].copy()

    # Compare predictions with approved items and add verification columns
    for field in ["Class_Name", "PBH", "Analytical_Hierarchy", "Analytical_Hierarchy_CD"]:
        detailed[f"{field}_predicted"] = verified_predictions[field].fillna("").astype(str).str.strip()
        detailed[f"{field}_actual"] = approved_items[field].fillna("").astype(str).str.strip()
        detailed[f"{field}_Match"] = detailed[f"{field}_predicted"] == detailed[f"{field}_actual"]

    # Add verification columns to detailed comparison
    detailed["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"] = False
    detailed["Class_In_Reference"] = verified_predictions["Class_In_Reference"]
    detailed["PBH_In_Reference"] = verified_predictions["PBH_In_Reference"]
    detailed["Hierarchy_In_Reference"] = verified_predictions["Hierarchy_In_Reference"]

    return detailed, invalid_counts

def create_mismatch_summary(detailed_comparison, item_hierarchy):
    detailed_comparison["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"] = detailed_comparison.apply(
        lambda row: row["Analytical_Hierarchy_CD_actual"] not in item_hierarchy["Hierarchy CD"].values, axis=1
    )

    mismatch = detailed_comparison.loc[
        ~detailed_comparison[[
            "Class_Name_Match", "PBH_Match", "Analytical_Hierarchy_Match", "Analytical_Hierarchy_CD_Match"
        ]].all(axis=1)
    ].reset_index(drop=True)

    mismatch["Mismatch_Type"] = mismatch.apply(
        lambda row: ", ".join(
            filter(None, [
                "Class_Name" if not row["Class_Name_Match"] else "",
                "PBH" if not row["PBH_Match"] else "",
                "Analytical_Hierarchy" if not row["Analytical_Hierarchy_Match"] else "",
                "Analytical_Hierarchy_CD" if not row["Analytical_Hierarchy_CD_Match"] else "",
                "Missing Analytical_Hierarchy_CD" if row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"] else ""
            ])
        ),
        axis=1
    )

    for field in ["Class_Name", "PBH", "Analytical_Hierarchy", "Analytical_Hierarchy_CD"]:
        mismatch[f"{field}_Diff"] = mismatch.apply(
            lambda row: diff_format_rich(row[f"{field}_predicted"], row[f"{field}_actual"]), axis=1
        )

    base_cols = ["Item_Num", "Brand_Name", "Long_Product_Name"]
    field_order = []
    for field in ["Class_Name", "PBH", "Analytical_Hierarchy", "Analytical_Hierarchy_CD"]:
        field_order.extend([f"{field}_predicted", f"{field}_Diff", f"{field}_actual"])
    extra_cols = [col for col in mismatch.columns if col not in base_cols + field_order]
    mismatch = mismatch[base_cols + field_order + extra_cols]

    return mismatch

def compute_accuracies(detailed_comparison_df, initiated_items, invalid_counts):
    # Calculate matches and totals
    total_items = len(detailed_comparison_df)
    class_matches = (detailed_comparison_df["Class_Name_predicted"] == detailed_comparison_df["Class_Name_actual"]).sum()
    pbh_matches = (detailed_comparison_df["PBH_predicted"] == detailed_comparison_df["PBH_actual"]).sum()
    analytical_matches = (detailed_comparison_df["Analytical_Hierarchy_predicted"] == detailed_comparison_df["Analytical_Hierarchy_actual"]).sum()
    
    # Calculate accuracies
    class_accuracy = class_matches / total_items
    pbh_accuracy = pbh_matches / total_items
    analytical_accuracy = analytical_matches / total_items

    # Calculate percentages of invalid entries relative to wrong predictions
    class_invalid_pct = (invalid_counts['Class_Name'] / (total_items - class_matches) * 100) if (total_items - class_matches) > 0 else 0
    pbh_invalid_pct = (invalid_counts['PBH'] / (total_items - pbh_matches) * 100) if (total_items - pbh_matches) > 0 else 0
    analytical_invalid_pct = (invalid_counts['Analytical_Hierarchy'] / (total_items - analytical_matches) * 100) if (total_items - analytical_matches) > 0 else 0

    # Create accuracy summary with counts and proportional invalid entries
    accuracy_summary = pd.DataFrame({
        "Metric": [
            "Class Accuracy", 
            "PBH Accuracy", 
            "Analytical Hierarchy Accuracy", 
            "Analytical Hierarchy Not In File Percentage"
        ],
        "Value": [
            f"{class_accuracy:.1%}",
            f"{pbh_accuracy:.1%}",
            f"{analytical_accuracy:.1%}",
            f"{detailed_comparison_df['Analytical_Hierarchy_CD_Not_In_Hierarchy_File'].mean():.1%}"
        ],
        "Count": [
            f"{total_items - class_matches}/{total_items} items wrong",
            f"{total_items - pbh_matches}/{total_items} items wrong",
            f"{total_items - analytical_matches}/{total_items} items wrong",
            f"{detailed_comparison_df['Analytical_Hierarchy_CD_Not_In_Hierarchy_File'].sum()}/{total_items} items"
        ],
        "Invalid Predictions": [
            f"{class_invalid_pct:.1f}% -- {invalid_counts['Class_Name']}/{total_items - class_matches} invalid entries",
            f"{pbh_invalid_pct:.1f}% -- {invalid_counts['PBH']}/{total_items - pbh_matches} invalid entries",
            f"{analytical_invalid_pct:.1f}% -- {invalid_counts['Analytical_Hierarchy']}/{total_items - analytical_matches} invalid entries",
            ""
        ]
    })
    
    category_counts = detailed_comparison_df.groupby("Analytical_Hierarchy_actual")["Item_Num"].count()
    correct_classifications = detailed_comparison_df[detailed_comparison_df["Analytical_Hierarchy_Match"] == True].groupby("Analytical_Hierarchy_actual")["Item_Num"].count()
    category_performance_df = pd.DataFrame({
        "Total Items": category_counts,
        "Correctly Classified": correct_classifications
    }).fillna(0)
    category_performance_df["Accuracy (%)"] = (category_performance_df["Correctly Classified"] / category_performance_df["Total Items"]) * 100
    category_performance_df = category_performance_df.reset_index().rename(columns={"Analytical_Hierarchy_actual": "Analytical_Hierarchy"})
    category_performance_df = category_performance_df.sort_values(by="Accuracy (%)", ascending=False)

    hl1 = compute_hierarchy_level_accuracy(detailed_comparison_df, 1)
    hl2 = compute_hierarchy_level_accuracy(detailed_comparison_df, 2)
    hl3 = compute_hierarchy_level_accuracy(detailed_comparison_df, 3)

    # Generate hierarchy analysis, passing both DataFrames
    analysis_rows = analyze_hierarchies(hl1, hl2, detailed_comparison_df, initiated_items)
    
    # Add analysis rows to accuracy_summary DataFrame
    for metric, value, count, invalid in analysis_rows:
        accuracy_summary.loc[len(accuracy_summary)] = [metric, value, count, invalid]
    
    return class_accuracy, pbh_accuracy, analytical_accuracy, accuracy_summary, category_performance_df, hl1, hl2, hl3

def compute_hierarchy_level_accuracy(detailed_comparison_df, level):
    column_name = f"Hierarchy_Level_{level}"
    detailed_comparison_df[column_name] = detailed_comparison_df["Analytical_Hierarchy_actual"].apply(
        lambda x: ">>".join(str(x).split(">>")[:level]) if pd.notnull(x) and len(str(x).split(">>")) >= level else "Unknown"
    )
    hl = detailed_comparison_df.groupby(column_name).agg(
        Total_Items=("Item_Num", "count"),
        Correctly_Classified=("Analytical_Hierarchy_Match", "sum")
    )
    hl = hl.reset_index()  # reset the index so the column becomes a proper column
    hl["Accuracy (%)"] = (hl["Correctly_Classified"] / hl["Total_Items"]) * 100
    hl = hl.sort_values(by="Accuracy (%)", ascending=False)
    hl = hl[[column_name, "Total_Items", "Correctly_Classified", "Accuracy (%)"]]
    return hl

def analyze_hierarchies(hl1, hl2, detailed_comparison_df, initiated_items):
    """
    Create analysis with root cause analysis for problematic categories.
    Returns rows with all columns to match accuracy_summary DataFrame structure.
    """
    # Find categories with low accuracy (below 80%)
    low_accuracy = hl2[hl2['Accuracy (%)'] < 80.0]
    priority_fixes = low_accuracy.sort_values('Total_Items', ascending=False).head(10)
    
    # Calculate impact metrics
    total_items_affected = priority_fixes['Total_Items'].sum()
    total_items_to_fix = (priority_fixes['Total_Items'] - priority_fixes['Correctly_Classified']).sum()
    total_dataset_size = hl2['Total_Items'].sum()
    current_correct_predictions = hl2['Correctly_Classified'].sum()
    current_accuracy = (current_correct_predictions / total_dataset_size) * 100
    potential_accuracy = ((current_correct_predictions + total_items_to_fix) / total_dataset_size) * 100
    
    analysis_rows = [
        ["Priority Categories Analysis", "Top 10 Largest Categories Below 80% Accuracy", "", ""],
        ["Total Items in Priority Categories", 
         f"{int(total_items_affected):,} ({(total_items_affected/total_dataset_size)*100:.1f}% of all items)",
         f"{total_items_affected}/{total_dataset_size} items",
         ""],
        ["Total Items Needing Fix", 
         f"{int(total_items_to_fix):,}",
         f"{total_items_to_fix}/{total_dataset_size} items",
         ""]
    ]
    
    # Analyze each priority category
    for priority_num, (_, row) in enumerate(priority_fixes.iterrows(), 1):  # Changed to use enumerate
        hierarchy = row['Hierarchy_Level_2']
        accuracy = row['Accuracy (%)']
        total_items = int(row['Total_Items'])
        correct = int(row['Correctly_Classified'])
        to_fix = total_items - correct
        
        # Get parent category accuracy
        level1 = hierarchy.split('>>')[0] if '>>' in hierarchy else hierarchy
        level1_stats = hl1[hl1['Hierarchy_Level_1'] == level1].iloc[0]
        level1_accuracy = level1_stats['Accuracy (%)']
        
        # Get misclassified items and their details from initiated_items
        misclassified_item_nums = detailed_comparison_df[
            (detailed_comparison_df["Analytical_Hierarchy_actual"] == hierarchy) & 
            ~detailed_comparison_df["Analytical_Hierarchy_Match"]
        ]["Item_Num"]
        
        misclassified_details = initiated_items[initiated_items["Item_Num"].isin(misclassified_item_nums)]
        
        # Important fields to check for root cause analysis
        key_fields = ["Brand_Name", "Long_Product_Name", "Description_1", "Description_2", "Benefits", "General_Description"]
        
        # Analyze missing or problematic fields
        field_analysis = {}
        for field in key_fields:
            empty_count = misclassified_details[field].isna().sum() + (misclassified_details[field] == "").sum()
            if empty_count > 0:
                field_analysis[field] = empty_count
        
        # Format root cause findings
        root_cause = ""
        if field_analysis:
            root_cause = "Missing fields in misclassified items:\n"
            for field, count in field_analysis.items():
                percentage = (count / to_fix) * 100
                root_cause += f"   • {field}: {count}/{to_fix} ({percentage:.1f}%)\n"
        else:
            root_cause = "No significant missing fields identified"
        
        metric = f"Priority {priority_num}: {hierarchy}"  # Using priority_num instead of idx + 1
        value = (f"{to_fix}/{total_items} items need fix - {accuracy:.1f}% accuracy (Parent: {level1_accuracy:.1f}%)\n"
                f"Root Cause: {root_cause}")
        count = f"{to_fix}/{total_items} items"
        invalid = ""  # Empty string instead of N/A
        analysis_rows.append([metric, value, count, invalid])

    # Add impact analysis with empty invalid predictions column
    analysis_rows.extend([
        ["", "", "", ""],
        ["Impact Analysis", "Potential Improvement from Fixing Priority Categories", "", ""],
        ["Current Overall Accuracy", 
         f"{current_accuracy:.1f}%",
         f"{current_correct_predictions}/{total_dataset_size} correct",
         ""],
        ["Potential Accuracy After Fixes", 
         f"{potential_accuracy:.1f}%",
         f"{current_correct_predictions + total_items_to_fix}/{total_dataset_size} potential correct",
         ""],
        ["Potential Accuracy Improvement", 
         f"+{(potential_accuracy - current_accuracy):.1f}%",
         f"+{total_items_to_fix} items",
         ""],
        ["ROI Analysis", 
         f"Fixing {total_items_to_fix:,} items could improve accuracy by {(potential_accuracy - current_accuracy):.1f}%",
         f"{total_items_to_fix}/{total_dataset_size} items to fix",
         ""]
    ])

    return analysis_rows

def diff_format_rich(predicted, actual):
    diff = difflib.ndiff(actual.split(), predicted.split())
    rich_text = ""
    for token in diff:
        if token.startswith('- '):
            rich_text += f"<span style='color:red'>{token[2:]}</span> "
        elif token.startswith('+ '):
            rich_text += f"<span style='color:blue'>{token[2:]}</span> "
        else:
            rich_text += token[2:] + " "
    return rich_text.strip()
