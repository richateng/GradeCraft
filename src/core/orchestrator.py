import os
import io
import base64
from typing import List, Dict, Any
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from src.core.logger import get_logger
from src.config.settings import settings
from src.agents.parser import parse_document_answer_key
from src.agents.vision import transcribe_student_pages
from src.agents.evaluator import verify_math_calculations, score_exam
from groq import AsyncGroq

logger = get_logger(__name__)

def pdf_to_base64_images(pdf_bytes: bytes) -> List[str]:
    """Converts a PDF file (in bytes) to a list of Base64 encoded JPEG images."""
    try:
        b64_images = []
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Use 2x scaling for high-resolution images
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            b64_images.append(img_str)
        pdf_document.close()
        return b64_images
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise e

async def run_gradecraft_swarm_pipeline(solution_pdf_bytes: bytes, student_pdf_bytes: bytes) -> Dict[str, Any]:
    """Orchestrates multi-agent pipelines across vision and text tasks."""
    logger.info("Initializing Groq Async Client...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
    
    client = AsyncGroq(api_key=api_key)
    
    # 1. Extract text from solution PDF for the rubric parser
    logger.info("Extracting text from solution PDF...")
    reader = PdfReader(io.BytesIO(solution_pdf_bytes))
    solution_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    logger.info("Parsing solution key via llama-3.3-70b-versatile...")
    extracted_rubrics = await parse_document_answer_key(client, solution_text)
    
    # 2. Convert student PDF to images
    logger.info("Converting student PDF to Base64 images...")
    student_pages_b64 = pdf_to_base64_images(student_pdf_bytes)
    
    # 3. Vision Transcription
    logger.info("Transcribing student pages via llama-4-scout-17b-16e-instruct...")
    student_transcription = await transcribe_student_pages(client, student_pages_b64)
    
    # 4. Math Verification
    # To bypass compound-mini for non-math, we could parse the solution key to see if it requires math.
    # For now, we will route all verification through math_verifier as requested in the batch evaluation strategy.
    logger.info("Verifying mathematical equivalence via groq/compound-mini...")
    verification_log = await verify_math_calculations(client, student_transcription, extracted_rubrics)
    
    # 5. Final Scoring
    logger.info("Evaluating final scores via llama-3.3-70b-versatile...")
    final_evaluation = await score_exam(client, student_transcription, verification_log, extracted_rubrics)
    
    return final_evaluation
