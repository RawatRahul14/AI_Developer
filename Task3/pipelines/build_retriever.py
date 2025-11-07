# === Python Modules ===

# === Components ===
from Task3.components.retriever.faiss_retriever import (
    FAISSRetriever
)

# === Task2 Main Function ===
from Task2.main import main

# === Main Pipeline Body for creating or loading the retriever ===
class CreateRetrieverPipeline:
    def __init__(
            self
    ):
        """
        Pipeline to either create or load a pre-existing retriever.
        """
        pass

    def _json_converter(
            self
    ):
        """
        Converts the JSON files into a str format for storing in vector Data base.
        """
        try:
            ## === Data Dict ===
            self.data_dict = main()

            ## === Data ===
            data = {}
            for k, v in self.data_dict.items():
                txt = "Name of the patient is " + v.get("patient", "Not given") + ". The Patient's diagnosed detail is " + v.get("diagnosis", "Not given") + " and the suggested treatment is " + v.get("treatment", "Not Given") + " and the followup is " + v.get("follow_up", "Not Given")
                data[k] = txt
            self.data = data

        except Exception as e:
            ValueError(f"Error Converting data from json to str: {e}")

    def main_fn(
            self,
            build: bool = False
    ):
        if build:
            try:
                retriever_pipeline = FAISSRetriever()
                retriever_pipeline.create_documents_from_json(
                    data_dict = self.data_dict,
                    data = self.data
                )

                return retriever_pipeline
                
            except Exception as e:
                ValueError(f"Error Creating Vector Database.")

        else:
            try:
                retriever_pipeline = FAISSRetriever()
                retriever_pipeline.load_index()

                return retriever_pipeline
            
            except Exception as e:
                ValueError(f"Error Loading the Database.")