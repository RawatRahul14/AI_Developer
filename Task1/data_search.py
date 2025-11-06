# === Python Modules ===
import numpy as np
from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from pathlib import Path

# === FastAPI App ===
app = FastAPI(
    title = "Medical Comprehend Search API",
    description = "Search extracted medical entities from Comprehend Medical outputs.",
    version = "1.0"
)

# === Load all processed CSVs on startup ===
DATA_PATH = Path("data/processed_medical_data")
merged_df = pd.DataFrame()

if DATA_PATH.exists():
    all_csvs = list(DATA_PATH.glob("*_summary.csv"))
    dfs = []
    for csv_file in all_csvs:
        df = pd.read_csv(csv_file)
        df["FileName"] = csv_file.stem.replace("_summary", "")
        dfs.append(df)
    if dfs:
        merged_df = pd.concat(dfs, ignore_index = True)
        print(f"Loaded {len(dfs)} processed files with {len(merged_df)} total rows.")
    else:
        print("No processed CSV files found.")
else:
    print("data/processed_medical directory not found.")

@app.get("/search")
def search_entities(
    query: str = Query(..., description = "Keyword to search in extracted medical entities."),
    limit: int = Query(10, description = "Maximum number of results to return.")
):
    """
    Searches across all Comprehend Medical summaries for the given keyword.
    Returns matching rows with their source file names.
    """
    if merged_df.empty:
        return {"message": "No processed data available."}

    ## === Case-insensitive search ===
    mask = (
        merged_df["Text"].astype(str).str.contains(query, case = False, na = False) |
        merged_df["Category"].astype(str).str.contains(query, case = False, na = False) |
        merged_df["Type"].astype(str).str.contains(query, case = False, na = False) |
        merged_df["Attributes"].astype(str).str.contains(query, case = False, na = False)
    )

    results = merged_df[mask].head(limit)

    if results.empty:
        return {"message": f"No matches found for '{query}'."}

    ## === Clean up invalid float values for JSON serialization ===
    results = results.replace([np.nan, np.inf, -np.inf], None)

    ## === Convert to list of dicts for JSON response ===
    return {
        "query": query,
        "total_results": len(results),
        "results": results.to_dict(orient="records")
    }