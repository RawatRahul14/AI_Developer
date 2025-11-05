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
        image_folder_path: Path = Path("data/raw_images"),
        to_process: List[str] | None = None
) -> Dict[str, Any]:
    """
    Extracts text data from the specified images in the folder using AWS Textract.

    Args:
        - client (boto3.client): An initialized boto3 client for AWS Textract.
        - image_folder_path (Path, optional): Path to the folder containing images. Defaults to 'data/raw_images'.
        - to_process (List[str] | None, optional): List of image filenames to process. If None, all images will be processed.

    Returns:
        - data_dict (Dict[str, Any]): Dictionary where keys are image filenames and values are Textract API responses.
    """
    try:
        ## === If there are no new images to process ===
        if not to_process:
            print("No new images to process.")
            return {}

        ## === Initiating an empty dictionary ===
        data_dict: Dict[str, Any] = {}

        ## === Looping through the images ===
        for img_name in to_process:
            image_path: Path = image_folder_path / img_name

            with open(image_path, "rb") as image:
                img: bytearray = bytearray(image.read())

            try:
                response = client.detect_document_text(
                    Document = {"Bytes": img}
                )

                data_dict[img_name] = response
                print(f"Processed: {img_name}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

        return data_dict

    except Exception as e:
        raise e

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

            ## === Check if JSON already exists ===
            if save_path.exists():
                with open(
                    save_path,
                    "r",
                    encoding = "utf-8"
                ) as f:
                    existing_data: Dict[str, str] = json.load(f)
            else:
                existing_data: Dict[str, str] = {}

            ## === Merge old + new data ===
            existing_data.update(texts)

            ## === Save merged data ===
            with open(
                save_path,
                "w",
                encoding = "utf-8"
            ) as f:
                """
                The file will store the merged extracted text dictionary as structured JSON data.
                """
                json.dump(
                    existing_data,
                    f,
                    indent = 4,
                    ensure_ascii = False
                )

            print(f"Processed data saved/updated at: {save_path}")

    except Exception as e:
        raise ValueError(f"Error Processing Textract Data: {e}")

def check_existing_extractions(
        image_folder_path: Path = Path("data/raw_images"),
        processed_file_path: Path = Path("data/processed_images/processed_text.json")
) -> Dict[str, List[str]]:
    """
    Checks which image files are already processed and which are new.

    Args:
        image_folder_path (Path): Folder containing raw images.
        processed_file_path (Path): Path to the existing processed JSON file.

    Returns:
        Dict[str, List[str]]:
            {
                "to_process": [...],
                "already_processed": [...]
            }
    """
    all_images = [
        f for f in os.listdir(image_folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # === If processed file exists ===
    if processed_file_path.exists():
        with open(processed_file_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        processed_files = set(processed_data.keys())
    else:
        processed_data = {}
        processed_files = set()

    # === Compare lists ===
    to_process = [img for img in all_images if img not in processed_files]
    already_processed = [img for img in all_images if img in processed_files]

    return {
        "to_process": to_process,
        "already_processed": already_processed
    }