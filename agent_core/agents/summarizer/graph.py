from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import sys
sys.path.append("C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/")

from pipelines.ingest.load_funcs import general_load
from pipelines.rag.service import build_prompt, retrieve
from agent_core.agents.llm import call_llm
from prompts import SYSTEM, PROMPT

# Import tools
from agent_core.agents.summarizer.process_tool import process_and_summarize_tool as pstool

# LangChain imports
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
#from langchain_openai import ChatOpenAI


def main():
    """Main execution - can use both traditional and agent approaches."""
    
    path_file = "C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/research_docs/datas/200313653v1.pdf"
    
    # Agent approach
    print("\n=== Agent Processing ===")
    #llm = ChatOpenAI(model="gpt-5", temperature=0.1)

    agent = create_agent(
    model="gpt-5",
    tools= [pstool],
    system_prompt="You are a document processing assistant. Use tools to process and summarize documents.",)
    result=agent.invoke(
    {"messages": [{"role": "user", "content": "Process and summarize the document at the following path: " + path_file}]})
    print("Agent Result:\n", result)


    
if __name__ == "__main__":
    main()