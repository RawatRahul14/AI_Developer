# === Python Modules ===
import os
from typing import Dict
from pathlib import Path

# === Utils ===
from Task2.utils import (
    summarize,
    get_structured_summaries,
    check_existing_summaries
)

from Task1.utils import (
    open_file
)


# === Main Summarizer Pipeline ===
class SummarizerPipeline:
    def __init__(
            self,
            data_path: Path = Path("data/structured_json")
    ):
        """
        Initializes the SummarizerPipeline by setting up the default data directory path.

        Args:
            - data_path (Path, optional): Path to the folder containing processed medical CSV files. Defaults to 'data/processed_medical_data'.
        """
        ## === Default directory path for processed data ===
        self.data_path: Path = data_path

    def summarize_data(
            self
    ) -> Dict[str, Dict]:
        """
        Executes the summarization pipeline in two stages:
        1. Converts all processed medical CSV files into formatted clinical notes.
        2. Passes the notes through the LLM (LangChain + OpenAI) to generate structured outputs.

        Returns:
            - Dict[str, Dict]: Dictionary containing structured summaries for each file.
        """
        ## === Using a flag to make sure not to call openai on every run ===
        flag_dict = check_existing_summaries()

        ## === If there are no new summaries to process ===
        if not flag_dict.get("to_summarize", []):
            print("No new summaries to process.")

            ## === Initialize an empty dictionary for existing data ===
            data_dict: Dict[str, Dict] = {}

            ## === Loop through all existing structured files ===
            for file_name in os.listdir(self.data_path):
                file_path = self.data_path / file_name

                ## === Process only JSON files ===
                if file_path.suffix.lower() == ".json":
                    try:
                        data = open_file(file_path)
                        data_dict[file_name] = data
                    except Exception as e:
                        print(f"Error reading file {file_name}: {e}")
                        continue
            return data_dict

        try:
            ## === Converting each CSV file into formatted text for the LLM ===
            summaries_dict: Dict[str, str] = summarize(data_path = self.data_path)

            ## === Generating structured summaries for each formatted note ===
            _ = get_structured_summaries(
                summaries = summaries_dict
            )

            ## === Initialising an empty Dictionary ===
            data_dict: Dict[str, Dict] = {}

            ## === Loop through all existing structured files ===
            for file_name in os.listdir(self.data_path):
                file_path = self.data_path / file_name

                ## === Process only JSON files ===
                if file_path.suffix.lower() == ".json":
                    try:
                        data = open_file(file_path)
                        data_dict[file_name] = data
                    except Exception as e:
                        print(f"Error reading file {file_name}: {e}")
                        continue
            return data_dict

        except Exception as e:
            raise ValueError(f"Error summarizing files: {e}")