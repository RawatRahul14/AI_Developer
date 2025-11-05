# === Python Module ===
import os
from pathlib import Path

# === Utils ===
from Task1.utils import (
    get_conn,
    extract_image_data,
    get_relevant_data
)

# === Main Data Extraction Body ===
def extract_data(
        file_path: Path = Path("data")
):
    """
    
    """
    ## === Making the connection ===
    client = get_conn(
        service = "textract"
    )

    ## === Extracting the data from the images ===
    data = extract_image_data(
        client = client
    )

    ## === Extracting relevant data ===
    data_dict = get_relevant_data(
        data = data,
        save_data = True
    )