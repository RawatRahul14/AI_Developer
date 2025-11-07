# === Python Modules ===
import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path
from typing import Dict, Any


# === Function to summarize AWS Comprehend Medical attributes ===
def summarize_attributes(attr_value):
    """
    Safely summarizes AWS Comprehend Medical attributes.
    Works even if attr_value is list, np.ndarray, or stringified JSON.
    """
    if attr_value is None:
        return None
    if isinstance(attr_value, float) and np.isnan(attr_value):
        return None

    if isinstance(attr_value, np.ndarray):
        if len(attr_value) == 0:
            return None
        attr_value = attr_value[0]

    if isinstance(attr_value, str):
        try:
            attr_value = ast.literal_eval(attr_value)
        except Exception:
            return None

    if not isinstance(attr_value, list):
        return None

    parts = []
    for a in attr_value:
        if not isinstance(a, dict):
            continue
        t = a.get("Type")
        txt = a.get("Text")
        if t and txt:
            parts.append(f"{t}: {txt}")

    return " | ".join(parts) if parts else None

# === Function to process and save AWS Medical Comprehend output ===
def process_comprehend_results(
        comprehend_data: Dict[str, Any],
        save_data: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Converts raw Comprehend Medical output into Pandas DataFrames
    with summarized attributes, grouped by file name.
    Optionally saves each processed DataFrame as a CSV file.

    Args:
        - comprehend_data (Dict[str, Any]): Dictionary of Comprehend responses.
        - save_data (bool, optional): Whether to save output CSV files. Defaults to False.

    Returns:
        - Dict[str, pd.DataFrame]: Dictionary where keys are file names and values are summarized DataFrames.
    """
    if not comprehend_data:
        print("No Comprehend Medical data available.")
        return {}

    results: Dict[str, pd.DataFrame] = {}
    save_path = Path("data/processed_medical_data")

    ## === Create folder if saving is enabled ===
    if save_data:
        os.makedirs(
            save_path,
            exist_ok = True
        )

    ## === Loop through each file ===
    for file_name, response in comprehend_data.items():
        entities = response.get("Entities", [])
        rows = []

        for ent in entities:
            rows.append({
                "Text": ent.get("Text"),
                "Category": ent.get("Category"),
                "Type": ent.get("Type"),
                "Score": ent.get("Score"),
                "Attributes": summarize_attributes(ent.get("Attributes"))
            })

        df = pd.DataFrame(rows)
        results[file_name] = df

        ## === Save to CSV if enabled ===
        if save_data:
            csv_path = save_path / f"{file_name}_summary.csv"
            df.to_csv(
                csv_path,
                index = False
            )
            print(f"💾 Saved: {csv_path}")

        print(f"Processed {file_name}: {len(df)} entities")

    return results