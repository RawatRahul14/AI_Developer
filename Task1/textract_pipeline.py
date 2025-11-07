# === Python Module ===
import os
from pathlib import Path

# === Utils ===
from Task1.utils import (
    get_conn,
    extract_image_data,
    get_relevant_data,
    check_existing_extractions,
    open_file
)

# === Main Data Extraction Body ===
class TextractPipeline:
    """
    Class-based pipeline to handle AWS Textract image-to-text extraction workflow.
    """

    def __init__(
            self
    ):
        """
        Initializes the TextractPipeline by setting up default paths and AWS client.
        """
        ## === AWS client placeholder (will be initialized during extraction) ===
        self.client = None

    # === Main function for data extraction ===
    def extract_data(self):
        """
        Controls the image-to-text extraction workflow using AWS Textract.

        This function ensures that:
            1. Previously processed images are skipped.
            2. Only new or unprocessed images are sent to AWS Textract.
            3. Extracted text data is cleaned, structured, and merged as JSON.

        Returns:
            - data_dict (Dict[str, str]): Dictionary containing filenames as keys and extracted text as values.
        """
        ## === Using a flag to make sure not to use AWS Textract on every run ===
        flag_dict = check_existing_extractions()

        ## === Checking if there are new images to process ===
        if not flag_dict.get("to_process", []):
            print("No new images to process.")
            data_dict = open_file()
            return data_dict

        try:
            ## === Step 1: Making the connection (only if not already initialized) ===
            if not self.client:
                self.client = get_conn(
                    service = "textract"
                )

            ## === Step 2: Extracting the data from the images ===
            data = extract_image_data(
                client = self.client,
                to_process = flag_dict.get("to_process")
            )

            ## === Step 3: Extracting relevant data and saving it as JSON ===
            _ = get_relevant_data(
                data = data,
                save_data = True
            )

            print("Extraction pipeline completed successfully.")
            data_dict = open_file()
            return data_dict

        except Exception as e:
            raise e