from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import sys
sys.path.append("C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/")


from pipelines.rag.service import build_prompt, retrieve
from agent_core.agents.llm import call_llm
from prompts import SYSTEM



prompt= "What is causal investing?"

hits= retrieve(query=prompt, 
               index_dir="C:/Users/Arda/Downloads/Upwork/Portfolio/book_teacher/research_docs/sample_works/faiss_index")

prompt_v1= build_prompt(prompt, SYSTEM, hits)

print(call_llm(prompt_v1, model="gpt-4o-mini", temperature=0.2, max_tokens=700))

