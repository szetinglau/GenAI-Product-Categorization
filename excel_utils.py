import pandas as pd
import os
import openpyxl

def save_evaluation_excel(initiated_items, approved_items, predictions, detailed_comparison, mismatch_summary, accuracy_summary, category_performance_df, hl1, hl2, hl3, output_dir):
    output_path = os.path.join(output_dir, "evaluation_results.xlsx")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        initiated_items.to_excel(writer, sheet_name="Initiated Items", index=False)
        approved_items.to_excel(writer, sheet_name="Approved Items", index=False)
        predictions.to_excel(writer, sheet_name="Predicted Items", index=False)
        detailed_comparison.to_excel(writer, sheet_name="Detailed Comparison", index=False)
        mismatch_summary.to_excel(writer, sheet_name="Mismatch Summary", index=False)
        accuracy_summary.to_excel(writer, sheet_name="Accuracy Summary", index=False)
        category_performance_df.to_excel(writer, sheet_name="Individual Hierarchy Accuracy", index=False)
        hl1.to_excel(writer, sheet_name="Hierarchy Level 1", index=False)
        hl2.to_excel(writer, sheet_name="Hierarchy Level 2", index=False)
        hl3.to_excel(writer, sheet_name="Hierarchy Level 3", index=False)

        format_eval_results(detailed_comparison, mismatch_summary, writer)

    auto_format_excel(output_path)
    print(f"Evaluation results saved to: {output_path}")

def format_eval_results(detailed_comparison, mismatch_summary, writer):
    workbook = writer.book
    detailed_ws = writer.sheets["Detailed Comparison"]
    mismatch_ws = writer.sheets["Mismatch Summary"]

    mismatch_format = workbook.add_format({'bg_color': '#FFCCCC'})
    red_format = workbook.add_format({'font_color': 'red'})
    blue_format = workbook.add_format({'font_color': 'blue'})

    for idx, row in detailed_comparison.iterrows():
        if not row["Class_Name_Match"]:
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Class_Name_predicted"), row["Class_Name_predicted"], mismatch_format)
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Class_Name_actual"), row["Class_Name_actual"], mismatch_format)
        if not row["PBH_Match"]:
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("PBH_predicted"), row["PBH_predicted"], mismatch_format)
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("PBH_actual"), row["PBH_actual"], mismatch_format)
        if not row["Analytical_Hierarchy_Match"]:
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_predicted"), row["Analytical_Hierarchy_predicted"], mismatch_format)
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_actual"), row["Analytical_Hierarchy_actual"], mismatch_format)
        if not row["Analytical_Hierarchy_CD_Match"]:
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_CD_predicted"), row["Analytical_Hierarchy_CD_predicted"], mismatch_format)
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_CD_actual"), row["Analytical_Hierarchy_CD_actual"], mismatch_format)
        if row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"]:
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_CD_actual"), row["Analytical_Hierarchy_CD_actual"], red_format)
            detailed_ws.write(idx + 1, detailed_comparison.columns.get_loc("Analytical_Hierarchy_CD_Not_In_Hierarchy_File"), row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"], mismatch_format)

    for idx, row in mismatch_summary.iterrows():
        if not row["Class_Name_Match"]:
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Class_Name_predicted"), row["Class_Name_predicted"], mismatch_format)
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Class_Name_actual"), row["Class_Name_actual"], mismatch_format)
        if not row["PBH_Match"]:
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("PBH_predicted"), row["PBH_predicted"], mismatch_format)
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("PBH_actual"), row["PBH_actual"], mismatch_format)
        if not row["Analytical_Hierarchy_Match"]:
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_predicted"), row["Analytical_Hierarchy_predicted"], mismatch_format)
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_actual"), row["Analytical_Hierarchy_actual"], mismatch_format)
        if not row["Analytical_Hierarchy_CD_Match"]:
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_CD_predicted"), row["Analytical_Hierarchy_CD_predicted"], mismatch_format)
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_CD_actual"), row["Analytical_Hierarchy_CD_actual"], mismatch_format)
        if row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"]:
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_CD_actual"), row["Analytical_Hierarchy_CD_actual"], red_format)
            mismatch_ws.write(idx + 1, mismatch_summary.columns.get_loc("Analytical_Hierarchy_CD_Not_In_Hierarchy_File"), row["Analytical_Hierarchy_CD_Not_In_Hierarchy_File"], mismatch_format)

def auto_format_excel(file_path):
    wb = openpyxl.load_workbook(file_path)
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_length = 0
            col_letter = col[0].column_letter
            for cell in col:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            sheet.column_dimensions[col_letter].width = max_length + 2
        if sheet.max_row > 1 and sheet.max_column > 1:
            sheet.auto_filter.ref = sheet.dimensions
        sheet.freeze_panes = "A2"
    wb.save(file_path)
    print(f"Excel file '{file_path}' formatted successfully.")
