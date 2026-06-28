import json
from typing import Dict, Any
from groq import AsyncGroq
from src.core.throttler import execute_throttled_inference

async def verify_math_calculations(client: AsyncGroq, student_transcription: str, extracted_rubrics: Dict[str, Any]) -> str:
    """Phase 2: Comprehensive Batch Verification Execution"""
    math_payload = {
        "messages": [{
            "role": "user",
            "content": f"Compare student calculations with the solution key. Use your built-in tool or code environment to evaluate mathematical equivalence. Context:\n{student_transcription}\nTarget Rules:\n{json.dumps(extracted_rubrics)}"
        }]
    }
    verification_log = await execute_throttled_inference(client, "math_verifier", math_payload)
    return verification_log

async def score_exam(client: AsyncGroq, student_transcription: str, verification_log: str, extracted_rubrics: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 3: Objective Analytical Scoring"""
    scoring_payload = {
        "messages": [
            {
                "role": "system", 
                "content": "You are an automated grader. Output a strict JSON structure containing an array of evaluations mapped to each question. Each object must have 'question', 'points_awarded', 'max_points', and a detailed 'deduction_rationale' string. Return ONLY the JSON object wrapping this array under a key 'evaluations'."
            },
            {"role": "user", "content": f"Assess performance:\nTranscription:\n{student_transcription}\nVerification:\n{verification_log}\nCriteria:\n{json.dumps(extracted_rubrics)}"}
        ],
        "response_format": {"type": "json_object"}
    }
    final_evaluation = await execute_throttled_inference(client, "rubric_evaluator", scoring_payload)
    return json.loads(final_evaluation)
