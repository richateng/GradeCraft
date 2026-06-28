from typing import List
from groq import AsyncGroq
from src.core.throttler import execute_throttled_inference

async def transcribe_student_pages(client: AsyncGroq, student_pages_b64: List[str]) -> str:
    """Orchestrates the vision transcription across multiple pages."""
    chunk_size = 5
    transcriptions = []
    
    for i in range(0, len(student_pages_b64), chunk_size):
        chunk = student_pages_b64[i:i + chunk_size]
        content = [{"type": "text", "text": "Transcribe the handwritten text, tables, matrix cells, and math equations into structured markdown + LaTeX. Preserve all question numbers and structure."}]
        
        for b64 in chunk:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
        vision_payload = {
            "messages": [{
                "role": "user",
                "content": content
            }]
        }
        
        chunk_transcription = await execute_throttled_inference(client, "vision_transcriber", vision_payload)
        transcriptions.append(chunk_transcription)
        
    return "\n\n---\n\n".join(transcriptions)
