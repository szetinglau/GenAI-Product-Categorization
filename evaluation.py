import pandas as pd
import numpy as np
import difflib
import logging
from excel_utils import save_evaluation_excel

def evaluate_predictions(initiated_items, predicted_items, approved_items, item_hierarchy_df, logger, output_dir):
    predictions_df = merge_with_item_hierarchy(predicted_items, item_hierarchy_df, logger)
    detailed_comparison_df = prepare_detailed_comparison(initiated_items, predictions_df, approved_items)
    mismatch_summary_df = create_mismatch_summary(detailed_comparison_df, item_hierarchy_df)

    detailed_comparison_df = detailed_comparison_df.replace([np.nan, np.inf, -np.inf], '')
    mismatch_summary_df = mismatch_summary_df.replace([np.nan, np.inf, -np.inf], '')
    
    class_acc, pbh_acc, analytical_acc, accuracy_summary, category_performance_df, hl1, hl2, hl3 = compute_accuracies(detailed_comparison_df)

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

def prepare_detailed_comparison(initiated_items, predictions, approved_items):
    detailed = initiated_items[["Item_Num", "Brand_Name", "Long_Product_Name"]].copy()

    for field in ["Class_Name", "PBH", "Analytical_Hierarchy", "Analytical_Hierarchy_CD"]:
        detailed[f"{field}_predicted"] = predictions[field].fillna("").astype(str).str.strip()
        detailed[f"{field}_actual"] = approved_items[field].fillna("").astype(str).str.strip()
        detailed[f"{field}_Match"] = detailed[f"{field}_predicted"] == detailed[f"{field}_actual"]

    detailed["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"] = False

    return detailed

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

def compute_accuracies(detailed_comparison_df):
    class_accuracy = (detailed_comparison_df["Class_Name_predicted"] == detailed_comparison_df["Class_Name_actual"]).mean()
    pbh_accuracy = (detailed_comparison_df["PBH_predicted"] == detailed_comparison_df["PBH_actual"]).mean()
    analytical_accuracy = (detailed_comparison_df["Analytical_Hierarchy_predicted"] == detailed_comparison_df["Analytical_Hierarchy_actual"]).mean()

    accuracy_summary = pd.DataFrame({
        "Metric": ["Class Accuracy", "PBH Accuracy", "Analytical Hierarchy Accuracy", "Analytical Hierarchy Not In File Percentage"],
        "Value": [
            class_accuracy,
            pbh_accuracy,
            analytical_accuracy,
            detailed_comparison_df["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"].mean()
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
