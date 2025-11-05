# === Python Modules for Path Handling ===
import sys
from pathlib import Path

## === Adding the project root directory to sys.path. This ensures that modules inside the parent directory (like Task1) can be imported even when this script is executed directly ===
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

# === AWS Textract Pipeline ===
from Task1.textract_pipeline import TextractPipeline

# === Main Function Combineing both the pipelines ===
def main():
    try:
        ## === Textract Pipeline ===
        text_pipeline = TextractPipeline()
        data_dict = text_pipeline.extract_data()

    except Exception as e:
        raise e

if __name__ == "__main__":
    main()