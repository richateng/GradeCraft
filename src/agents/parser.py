import json
from typing import Dict, Any
from groq import AsyncGroq
from src.core.throttler import execute_throttled_inference

async def parse_document_answer_key(client: AsyncGroq, raw_scheme_text: str) -> Dict[str, Any]:
    """Normalizes unstructured textual marking metrics into a programmatic rubric structure."""
    prompt_payload = {
        "messages": [
            {
                "role": "system", 
                "content": "You are a data standardization engine. Extract questions, rubrics, and maximum scores into standard JSON structures. Your output must be a JSON object wrapping an array under a key 'questions', where each object has 'question' (string), 'rubric' (string), and 'max_score' (number) keys."
            },
            {"role": "user", "content": f"Parse this exam metadata into a clean JSON blueprint:\n{raw_scheme_text}"}
        ],
        "response_format": {"type": "json_object"}
    }
    extracted_json = await execute_throttled_inference(client, "rubric_evaluator", prompt_payload)
    return json.loads(extracted_json)
