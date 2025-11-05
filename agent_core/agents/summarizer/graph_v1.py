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




#hits= retrieve(query=prompt, 
#               index_dir="C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/research_docs/sample_works/faiss_index")

path_file="C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/research_docs/datas/200313653v1.pdf"
path_file="https://www.youtube.com/watch?v=yrjFTlvSI1w&t=1s"
pdfs= general_load(path_file)
#pdfs.text=pdfs.text[:1000]

#prompt_v1= build_prompt(prompt, SYSTEM, hits)
prompt_v1 = f"{SYSTEM}\n\nQuery:\n{PROMPT}\n\nContext:\n{pdfs.pages}\n\nAnswer:"

result= call_llm(prompt_v1, model="gpt-5")
print(result)

with open("summary_output.json", "w", encoding="utf-8") as f:
   f.write(result)

with open("actual_output.json", "w", encoding="utf-8") as f:
   f.write(f"{pdfs.pages}")
