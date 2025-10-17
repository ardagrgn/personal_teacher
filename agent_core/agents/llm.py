import os
from dotenv import load_dotenv
load_dotenv() 
def call_llm(prompt: str, model: str | None = None, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    OpenAI-compatible Chat Completions caller.
    Requires OPENAI_API_KEY. Optional: OPENAI_BASE_URL, OPENAI_MODEL.
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            #base_url=os.environ.get("OPENAI_BASE_URL")
        )
        use_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


