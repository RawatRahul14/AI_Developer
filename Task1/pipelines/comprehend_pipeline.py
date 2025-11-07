# === Python Modules ===
import os
from pathlib import Path
from typing import Dict

# === Components ===
from Task1.components.comprehend import (
    check_existing_comprehend,
    extract_medical_entities
)

# === Utils ===
from Task1.utils.common import (
    get_conn,
    open_file
)

# === Main Comprehend Body ===
class ComprehendPipeline:
    """
    Class-based pipeline to handle AWS Medical Comprehend workflow.
    """

    def __init__(
            self
    ):
        """
        Initializes the ComprehendPipeline by setting up default paths and AWS client.
        """
        ## === Default data directory path ===
        self.file_path: Path = Path("data")

        ## === AWS client placeholder (will be initialized during extraction) ===
        self.client = None

    def extract_info(self, data_dict):
        """
        Uses AWS Comprehend Medical to analyze and extract key medical entities such as conditions, medications, treatments, and test results from the previously extracted text data.
        """
        ## === Using a Flag to make sure AWS Comprehend Medical doesn't run on every run ===
        flag_dict: Dict[str, str] = check_existing_comprehend()

        if not flag_dict.get("to_analyze", []):
            print("No new comprehend run needed.")
            print(flag_dict.get("to_analyze"))
            comprehend_data = open_file(file_path = Path("data/processed_medical/processed_entities.json"))
            return comprehend_data

        try:
            ## === Step 1: Making the connection (only if not already initialized) ===
            self.client = get_conn(
                service = "comprehendmedical"
            )

            ## === Step 2: Extracting the medical data from the textract output ===
            _ = extract_medical_entities(
                client = self.client,
                text_data = data_dict,
                save_data = True,
                to_process = flag_dict.get("to_analyze")
            )

            print("Comprehend Pipeline completed successfully.")
            comprehend_data = open_file(file_path = Path("data/processed_medical/processed_entities.json"))
            return comprehend_data

        except Exception as e:
            raise e