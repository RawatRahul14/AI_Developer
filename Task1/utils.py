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

# === Function to get the old and new images ===
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
            with open(save_path, "w", encoding="utf-8") as f:
                """
                Saves or updates the Comprehend Medical extracted entities in JSON format.
                """
                json.dump(
                    updated_data,
                    f,
                    ensure_ascii = False,
                    indent = 4
                )

            print(f"💾 File updated: {save_path}")

        return entity_dict

    except Exception as e:
        raise e

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