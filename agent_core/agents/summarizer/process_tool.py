from langchain.tools import tool
from pydantic import BaseModel, Field
import sys
sys.path.append("C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/")

from pipelines.ingest.load_funcs import general_load
from agent_core.agents.llm import call_llm
from agent_core.agents.summarizer.prompts import SYSTEM, PROMPT

class DocumentInput(BaseModel):
    file_path: str = Field(description="Document path or URL")

@tool("process_and_summarize", args_schema=DocumentInput)
def process_and_summarize_tool(file_path: str) -> str:
    """Process a document and create a structured summary."""
    try:
        # Load document
        doc = general_load(file_path)
        
        # Create summary
        prompt_v1 = f"{SYSTEM}\n\nQuery:\n{PROMPT}\n\nContext:\n{doc.pages}\n\nAnswer:"
        result = call_llm(prompt_v1, model="gpt-5")
        with open("actual_output.json", "w", encoding="utf-8") as f:
            f.write(f"{doc.pages}")

        with open("summary_output.json", "w", encoding="utf-8") as f:
            f.write(result)
        
        return result
    except Exception as e:
        return f"Error processing document: {str(e)}"