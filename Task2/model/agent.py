# === Python Modules ===
import os
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Dict, List
from pathlib import Path

# === Schema ===
from Task2.schema.schema import (
    MedicalNotes
)

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