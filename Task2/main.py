# === Python Modules for Path Handling ===
import sys
from pathlib import Path

## === Adding the project root directory to sys.path. This ensures that modules inside the parent directory (like Task1) can be imported even when this script is executed directly ===
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

# === Pipeline ===
from Task2.pipelines.summarizer_pipeline import (
    SummarizerPipeline
)

# === Main Pipeline body ===
def main():
    try:
        # === Summarizer Pipeline ===
        pipeline = SummarizerPipeline()
        data = pipeline.summarize_data()

        return data
    
    except Exception as e:
        ValueError(f"Error running the summarizer pipeline: {e}")