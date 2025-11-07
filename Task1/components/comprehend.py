# === Python Modules ===
import os
import json
import boto3
from typing import Dict, Any, List
from pathlib import Path
from typing import Dict, List

# === Function to check which extracted texts are not yet processed by Medical Comprehend ===
def check_existing_comprehend(
        textract_file_path: Path = Path("data/processed_images/processed_text.json"),
        comprehend_file_path: Path = Path("data/processed_medical/processed_entities.json")
) -> Dict[str, List[str]]:
    """
    Compares the keys (file names) between Textract and Comprehend Medical outputs to identify which files still need to be analyzed.

    Args:
        - textract_file_path (Path): Path to the JSON file containing Textract extracted text.
        - comprehend_file_path (Path): Path to the JSON file containing Comprehend Medical extracted entities.

    Returns:
        - Dict[str, List[str]]: {
                "to_analyze": [...],
                "already_analyzed": [...]
            }
    """
    try:
        ## === Load Textract processed data ===
        if textract_file_path.exists():
            with open(
                textract_file_path,
                "r",
                encoding = "utf-8"
            ) as f:
                textract_data = json.load(f)
            textract_keys = set(textract_data.keys())
        else:
            print("No Textract processed file found.")
            return {"to_analyze": [], "already_analyzed": []}

        ## === Load Comprehend processed data (if exists) ===
        if comprehend_file_path.exists():
            with open(
                comprehend_file_path,
                "r",
                encoding = "utf-8"
            ) as f:
                comprehend_data = json.load(f)
            comprehend_keys = set(comprehend_data.keys())
        else:
            comprehend_data = {}
            comprehend_keys = set()

        ## === Compare keys ===
        to_analyze = [key for key in textract_keys if key not in comprehend_keys]
        already_analyzed = [key for key in textract_keys if key in comprehend_keys]

        ## === Optional: Prepare a dict of texts to pass to Comprehend ===
        texts_to_analyze: Dict[str, str] = {
            key: textract_data[key] for key in to_analyze
        }

        ## === Return both lists and ready-to-process text dict ===
        return {
            "to_analyze": to_analyze,
            "already_analyzed": already_analyzed,
            "texts_to_analyze": texts_to_analyze
        }

    except Exception as e:
        raise e

# === Extract meaningful data using AWS Medical Comprehend ===
def extract_medical_entities(
        client: boto3.client,
        text_data: Dict[str, str],
        to_process: List[str] | None = None,
        save_data: bool = False
) -> Dict[str, Any]:
    """
    Uses AWS Comprehend Medical to extract entities such as conditions, medications, 
    and treatments from the already extracted text files.

    Args:
        - client (boto3.client): An initialized boto3 client for AWS Comprehend Medical.
        - text_data (Dict[str, str]): Dictionary where keys are file names and values are extracted text strings.
        - to_process (List[str] | None, optional): List of file names to analyze. If None, all files are processed.
        - save_data (bool, optional): Whether to save or update the output JSON file. Defaults to False.

    Returns:
        - entity_dict (Dict[str, Any]): Dictionary where keys are file names and values are Comprehend Medical API responses.
    """
    try:
        ## === If no text data available ===
        if not text_data:
            print("No text data available for processing.")
            return {}

        ## === If no specific files provided, process all ===
        if not to_process:
            print("No 'to_process' list provided — analyzing all files.")
            to_process = list(text_data.keys())

        ## === Filter text data for only the given files ===
        subset_data = {
            key: text_data[key]
            for key in to_process
            if key in text_data
        }

        ## === Initiate an empty dictionary for new results ===
        entity_dict: Dict[str, Any] = {}

        ## === Loop through the subset and run Comprehend Medical ===
        for file_name, txt in subset_data.items():
            try:
                response = client.detect_entities_v2(Text=txt)
                entity_dict[file_name] = response
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

        ## === Save or update the JSON file ===
        if save_data:
            data_path: Path = Path("data/processed_medical")
            os.makedirs(
                data_path,
                exist_ok = True
            )

            save_path: Path = data_path / "processed_entities.json"

            ## === Load existing data if exists ===
            if save_path.exists():
                with open(
                    save_path,
                    "r",
                    encoding = "utf-8"
                ) as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}

            ## === Merge new data with existing ===
            updated_data = {**existing_data, **entity_dict}

            ## === Save the merged data ===
            with open(
                save_path,
                "w",
                encoding = "utf-8"
            ) as f:
                """
                Saves or updates the Comprehend Medical extracted entities in JSON format.
                """
                json.dump(
                    updated_data,
                    f,
                    ensure_ascii = False,
                    indent = 4
                )

            print(f"File updated: {save_path}")

        return entity_dict

    except Exception as e:
        raise e