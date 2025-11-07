# === Python Modules ===
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# === Schema ===
from Task2.schema import MedicalNotes

# === Function to convert csv file to strings ===
def prepare_note_from_csv(
    csv_path: str
) -> str:
    """
    Converts a structured Medical Comprehend CSV file into a single plain-text clinical note that can be passed to an LLM for summarization.

    Args:
        - csv_path (str): Path to the CSV file (e.g., 'data/processed_medical/patient_1_summary.csv')

    Returns:
        - str: Formatted text combining all entities from the CSV.
    """
    # === Load CSV file ===
    df = pd.read_csv(csv_path)

    # === List to store formatted lines ===
    formatted_lines = []

    # === Iterate through each row in the CSV ===
    for _, row in df.iterrows():
        entity_text = str(row['Text']).strip()
        entity_category = str(row['Category']).strip()
        entity_type = str(row['Type']).strip()
        entity_attributes = str(row.get('Attributes', '')).strip()

        # === Create a readable string for each extracted entity ===
        line = f"{entity_category} ({entity_type}): {entity_text}"

        # === Add attributes if present (e.g., TEST_VALUE, DOSAGE, DURATION) ===
        if entity_attributes and entity_attributes.lower() != 'nan':
            line += f" | {entity_attributes}"

        formatted_lines.append(line)

    # === Combine all entity lines into a single note ===
    return "\n".join(formatted_lines)

# === Function to convert every csv into the string ===
def summarize(
        data_path: Path = Path("data/processed_medical_data")
) -> Dict[str, str]:
    """
    Loops through all processed Medical Comprehend CSV files, converts each into
    a formatted text note using prepare_note_from_csv(), and stores them in a dictionary.

    Args:
        - data_path (Path, optional): Path to the folder containing processed CSVs. Defaults to 'data/processed_medical_data'.

    Returns:
        - Dict[str, str]: Dictionary where keys are CSV filenames and values are formatted clinical notes (ready for LLM summarization).
    """
    try:
        ## === Initialize dictionary to store notes ===
        notes_dict: Dict[str, str] = {}

        ## === Ensure path exists ===
        if not os.path.exists(data_path):
            print(f"Provided path does not exist: {data_path}")
            return {}
        
        # === Loop through all files in the directory ===
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)

            # Process only CSV files
            if os.path.isfile(file_path) and file_name.lower().endswith(".csv"):
                print(f"Reading: {file_name}")
                note_text = prepare_note_from_csv(file_path)
                notes_dict[file_name] = note_text

        # === Return all generated notes ===
        print(f"Processed {len(notes_dict)} CSV files successfully.")
        return notes_dict

    except Exception as e:
        print(f"Error while summarizing data: {e}")
        return {}
    
# === Function to create summaries using LangChain ===
# === Function to create summaries using LangChain ===
def get_structured_summaries(
        summaries: Dict[str, str],
        to_summarize: List[str] | None = None,
        save_data: bool = True
) -> Dict[str, Dict]:
    """
    Takes formatted clinical notes (from prepare_note_from_csv) and generates
    structured medical summaries using LangChain + OpenAI with Pydantic validation.
    Processes only the files specified in 'to_summarize' and saves a new JSON
    file for each generated structured summary.

    Args:
        - summaries (Dict[str, str]): Dictionary where keys are file names and values are formatted clinical notes.
        - to_summarize (List[str] | None, optional): List of file names to process. If None, all files from summaries will be processed.
        - save_data (bool, optional): Whether to save generated structured summaries as JSON files. Defaults to True.

    Returns:
        - Dict[str, Dict]: Dictionary where keys are file names and values are structured JSON outputs validated by the MedicalNotes schema.
    """
    try:
        ## === Load environment variables ===
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        ## === Directory path for structured outputs ===
        save_dir = Path("data/structured_json")
        os.makedirs(
            save_dir,
            exist_ok = True
        )

        ## === Filter summaries for only the required files ===
        if to_summarize:
            filtered_summaries = {
                k: v for k, v in summaries.items() if k in to_summarize
            }
        else:
            print("No 'to_summarize' list provided - processing all available summaries.")
            filtered_summaries = summaries

        ## === If there’s nothing to process ===
        if not filtered_summaries:
            print("No new files to summarize.")
            return {}

        ## === Initialize model with structured output ===
        model = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0
        ).with_structured_output(MedicalNotes)

        ## === Initialize dictionary to store structured results ===
        structured_dict: Dict[str, Dict] = {}

        ## === Loop through each file and generate structured summary ===
        for file_name, summary_text in filtered_summaries.items():
            print(f"Generating structured summary for: {file_name}")

            try:
                response = model.invoke(
                    f"""
                    You are a careful medical summarizer. The clinical note below may contain:
                    - Duplicated information,
                    - OCR noise,
                    - Spelling/typographical variants (e.g., "Merpes" -> "Herpes").

                    Instructions:
                    - Think carefully before answering.
                    - Deduplicate repeated facts; state each fact once.
                    - Correct obvious, unambiguous medical spelling errors.
                    - Normalize units and medication names when clear (e.g., mg, %, °F/°C).
                    - If conflicting values appear, choose the most consistent/specific one; if uncertain, choose the most reasonable and concise phrasing.
                    - Output must strictly follow the structure: patient, diagnosis, treatment, follow_up.
                    - Do not add extra keys or commentary.

                    Clinical Note:
                    {summary_text}
                    """
                )

                ## === Convert Pydantic object to dictionary ===
                structured_output = response.dict()
                structured_dict[file_name] = structured_output

                ## === Save individual JSON file for each summary ===
                if save_data:
                    file_stem = Path(file_name).stem
                    save_path = save_dir / f"{file_stem}.json"
                    with open(
                        save_path,
                        "w",
                        encoding = "utf-8"
                    ) as f:
                        json.dump(
                            structured_output,
                            f,
                            indent = 4,
                            ensure_ascii = False
                        )
                    print(f"Saved structured summary: {save_path.name}")

                print(f"Completed: {file_name}")

            except Exception as inner_error:
                print(f"Error processing {file_name}: {inner_error}")
                continue

        ## === Return dictionary of structured outputs ===
        print("All structured summaries generated successfully.")
        return structured_dict

    except Exception as e:
        print(f"Error during structured summary generation: {e}")
        return {}

# === Function to check which formatted notes are not yet summarized by the LLM ===
def check_existing_summaries(
        note_data_path: Path = Path("data/processed_medical_data"),
        structured_data_path: Path = Path("data/structured_json")
) -> Dict[str, List[str]]:
    """
    Compares the available processed CSV files with existing structured summaries
    to identify which files still need to be summarized by the LLM.

    Args:
        - note_data_path (Path): Folder containing processed medical CSV files.
        - structured_data_path (Path): Folder containing already generated structured summaries.

    Returns:
        - Dict[str, List[str]]: {
                "to_summarize": [...],
                "already_summarized": [...]
            }
    """
    try:
        ## === Ensure folders exist ===
        if not note_data_path.exists():
            print(f"No processed medical data folder found at: {note_data_path}")
            return {"to_summarize": [], "already_summarized": []}

        if not structured_data_path.exists():
            os.makedirs(structured_data_path, exist_ok=True)

        ## === Get all available processed CSV files ===
        all_files = [
            f for f in os.listdir(note_data_path)
            if f.lower().endswith(".csv")
        ]

        ## === Get all structured summary files (if any) ===
        existing_summaries = [
            f.replace("_structured.json", "")
            for f in os.listdir(structured_data_path)
            if f.lower().endswith(".json")
        ]

        ## === Compare file names (without extensions) ===
        csv_stems = [Path(f).stem for f in all_files]
        summarized_stems = set(existing_summaries)

        to_summarize = [
            f for f in csv_stems if f not in summarized_stems
        ]
        already_summarized = [
            f for f in csv_stems if f in summarized_stems
        ]

        ## === Return comparison results ===
        return {
            "to_summarize": to_summarize,
            "already_summarized": already_summarized
        }

    except Exception as e:
        raise e