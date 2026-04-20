import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
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

    @model_validator(mode='after')
    def fix_line_order(self):
        """Swap line_start/line_end if the model outputs them inverted."""
        if self.line_start > self.line_end:
            self.line_start, self.line_end = self.line_end, self.line_start
        return self

class AnswerResponse(BaseModel):
    answer: str = Field(description="A clear and accurate answer to the question based ONLY on the provided context.")
    evidence: List[Citation] = Field(description="A list of exact file paths and line ranges used to answer the question.")

class AnswerOnlyResponse(BaseModel):
    answer: str = Field(description="A clear and accurate answer to the question based ONLY on the provided context.")

def _call_structured(messages: List[Dict[str, str]], response_format, max_tokens: int):
    return client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=0.0
    )

def generate_answer(prompt_context: str, question: str, max_tokens: int = 2000,
                    require_citations: bool = True) -> Dict[str, Any]:
    """Call the LLM with context and question, with optional citation extraction."""
    
    system_prompt = """You are an expert software engineer and technical documentation reader.
Your task is to answer questions about a codebase or technical documentation based STRICTLY on the provided context.
You MUST NOT use outside knowledge. If the answer is not in the context, say "I don't have enough information".
If you can answer part of the question but not all, answer the parts you CAN support with evidence and explicitly state which parts are not covered in the provided context. Do NOT speculate or fabricate details.

If citations are requested, supply exact citations using the 'evidence' list, including the 'file' name, 'line_start', and 'line_end'. Ensure line_start <= line_end.
Do not guess citations. If the answer is supported but exact line references are unclear, still answer and leave evidence empty."""

    user_prompt = f"""CONTEXT:
{prompt_context}

---
QUESTION:
{question}
"""

    start_time = time.time()
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = _call_structured(
            messages,
            AnswerResponse if require_citations else AnswerOnlyResponse,
            max_tokens=max_tokens
        )
        
        latency = time.time() - start_time
        result_obj = response.choices[0].message.parsed
        
        # Calculate tokens from usage metadata
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        return {
            "success": True,
            "answer": result_obj.answer if result_obj else "Failed to parse",
            "evidence": [dict(c) for c in result_obj.evidence] if require_citations and result_obj else [],
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_calls": 1
        }
        
    except Exception as e:
        if require_citations:
            try:
                response = _call_structured(
                    messages,
                    AnswerOnlyResponse,
                    max_tokens=max_tokens
                )
                latency = time.time() - start_time
                result_obj = response.choices[0].message.parsed
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                return {
                    "success": True,
                    "answer": result_obj.answer if result_obj else "Failed to parse",
                    "evidence": [],
                    "latency": latency,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model_calls": 1,
                    "warning": f"citation_extraction_failed:{e}"
                }
            except Exception as fallback_error:
                e = fallback_error
        latency = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": latency,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 1
        }
