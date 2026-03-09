import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from config import OLLAMA_BASE_URL, API_KEY, MODEL_NAME

# Initialize OpenAI client pointing to local Ollama
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=API_KEY
)

class Citation(BaseModel):
    file: str = Field(description="The path to the file containing the evidence.")
    line_start: int = Field(description="The starting line number of the evidence.")
    line_end: int = Field(description="The ending line number of the evidence.")

class AnswerResponse(BaseModel):
    answer: str = Field(description="A clear and accurate answer to the question based ONLY on the provided context.")
    evidence: List[Citation] = Field(description="A list of exact file paths and line ranges used to answer the question.")

def generate_answer(prompt_context: str, question: str, max_tokens: int = 2000) -> Dict[str, Any]:
    """Call the LLM with context and question, enforcing structured JSON output."""
    
    system_prompt = """You are an expert software engineer and technical documentation reader.
Your task is to answer questions about a codebase or technical documentation based STRICTLY on the provided context.
You MUST NOT use outside knowledge. If the answer is not in the context, say "I don't have enough information".

You must supply exact citations for your answer using the 'evidence' list, including the 'file' name, 'line_start', and 'line_end'.
Ensure your line references are 100% accurate relative to the provided line numbers in the context."""

    user_prompt = f"""CONTEXT:
{prompt_context}

---
QUESTION:
{question}
"""

    start_time = time.time()
    
    try:
        # Use OpenAI SDK's built-in structured output parsing for Pydantic models
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AnswerResponse,
            max_tokens=max_tokens,
            temperature=0.0 # Deterministic accuracy for evaluation
        )
        
        latency = time.time() - start_time
        result_obj = response.choices[0].message.parsed
        
        # Calculate tokens from usage metadata
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        return {
            "success": True,
            "answer": result_obj.answer if result_obj else "Failed to parse",
            "evidence": [dict(c) for c in result_obj.evidence] if result_obj else [],
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_calls": 1
        }
        
    except Exception as e:
        latency = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": latency,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 1
        }
