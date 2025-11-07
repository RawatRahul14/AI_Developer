# === Python Modules ===
import os
import boto3
import json
from pathlib import Path
from typing import Dict, Any, List

# === Functoin to create connection to aws ===
def get_conn(
        service: str,
        region_name: str = "ap-southeast-2"
) -> boto3.client:
    """
    Creates a connection to the aws.

    Args: 
        - service (str): Name of the service to use in the aws. In this case `textract` or `comprehend`.

    returns:
        - connection (boto3.Client): A low-level client representing the AWS service specified.
    """
    try:
        connection = boto3.client(
            service,
            region_name = region_name
        )

    except Exception as e:
        raise e

    return connection

# === Function to Open the already saved Extracted Data ===
def open_file(
        file_path: Path = Path("data/processed_images/processed_text.json")
) -> Dict[str, Any]:
    """
    Opens and loads the saved extracted text data from the given JSON file.

    Args:
        - file_path (Path, optional): Path to the saved JSON file containing the extracted text data. Defaults to 'data/processed_images/processed_text.json'.

    Returns:
        - data_dict (Dict[str, Any]): Dictionary where keys are filenames and values are the extracted text.
    """
    try:
        ## === Check if the file exists ===
        if not file_path.exists():
            raise FileNotFoundError(f"The specified file does not exist: {file_path}")

        ## === Opens and loads the file ===
        with open(
            file_path,
            "r",
            encoding = "utf-8"
        ) as f:
            data_dict: Dict[str, Any] = json.load(f)

        return data_dict

    except Exception as e:
        ValueError(f"Error Reading file: {e}")