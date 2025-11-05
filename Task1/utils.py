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

# === Function to loop through the images ===
def extract_image_data(
        client: boto3.client,
        image_folder_path: Path = Path("data/raw_images")
) -> Dict[str, Any]:
    """
    Extracts text data from all images in the specified folder using AWS Textract.

    Args:
        client (boto3.client): An initialized boto3 client for AWS Textract.
        image_folder_path (Path, optional): Path to the folder containing images. Defaults to 'data'.

    Returns:
        Dict[str, Any]:
            A dictionary where keys are image filenames and values are the Textract API JSON responses for each image.
    """
    data_dict: Dict[str, Any] = {}
    for img_name in os.listdir(image_folder_path):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = image_folder_path / img_name

            with open(image_path, "rb") as image:
                img = bytearray(image.read())

            try:
                response = client.detect_document_text(
                    Document = {"Bytes": img}
                )

                data_dict[img_name] = response

            except Exception as e:
                raise e

    return data_dict

# === Function to extract the relevant data from the extracted data ===
def get_relevant_data(
        data: Dict[str, Any],
        save_data: bool = False
) -> Dict[str, str]:
    """
    Extracts the relavant data needed from the response recieved from the Textract

    Args:
        - data (Dict[str, Any]): AWS Textract Output response

    returns:
        - texts (Dict[str, str]): Dictionary where the key is the file name and the values are the relevant data.
    """
    try:
        ## === Initiating an empty dictionary ===
        texts: Dict[str, str] = {}

        ## === looping through the data ===
        for filename, response in data.items():
            lines: List = []

            for block in response.get("Blocks", []):
                if block.get("BlockType") == "LINE":
                    lines.append(block.get("Text", ""))

            texts[filename] = " ".join(lines).strip()

        ## === Optional: Saving as JSON ===
        if save_data:

            ## === Path to the folder where to save the json data ===
            data_path: Path = Path("data/processed_images")
            os.makedirs(
                data_path,
                exist_ok = True
            )

            ## === Filename where to save ===
            save_path: Path = data_path / "processed_text.json"
            with open(
                save_path,
                "w",
                encoding = "utf-8"
            ) as f:
                """
                The file will store the extracted text dictionary as structured JSON data.
                """
                json.dump(
                    texts,
                    f,
                    indent = 4,
                    ensure_ascii = False
                )

        return texts

    except Exception as e:
        raise ValueError(f"Error Processing Textract Data: {e}")