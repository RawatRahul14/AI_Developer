# === Python Modules ===
from pydantic import BaseModel, Field

# === Structured Output Schema ===
class MedicalNotes(BaseModel):
    patient: str = Field(
        description = "The name or identifying information of the patient."
    )
    diagnosis: str = Field(
        description = "The diagnosed condition or medical issue."
    )
    treatment: str = Field(
        description = "Prescribed medications, dosages, or procedures."
    )
    follow_up: str = Field(
        description = "Follow-up instructions or timeline."
    )